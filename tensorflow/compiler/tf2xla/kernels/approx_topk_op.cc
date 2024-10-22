/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>

#include "absl/strings/match.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/hlo/builder/lib/approx_topk.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tpu/tpu_defs.h"
namespace tensorflow {
namespace {

xla::XlaComputation ComparatorBuilder(xla::XlaBuilder* builder,
                                      xla::PrimitiveType op_type,
                                      bool is_max_k) {
  auto p0 = xla::Parameter(builder, 0, xla::ShapeUtil::MakeScalarShape(op_type),
                           "v0");
  auto p1 = xla::Parameter(builder, 1, xla::ShapeUtil::MakeScalarShape(op_type),
                           "v1");
  xla::Parameter(builder, 2, xla::ShapeUtil::MakeScalarShape(xla::S32), "a2");
  xla::Parameter(builder, 3, xla::ShapeUtil::MakeScalarShape(xla::S32), "a3");
  if (is_max_k) {
    xla::Gt(p0, p1);
  } else {
    xla::Lt(p0, p1);
  }
  return builder->BuildAndNoteError();
}

class ApproxTopKOpBase : public XlaOpKernel {
 public:
  explicit ApproxTopKOpBase(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    // k is static instead of dynamic.
    // This is required for deriving the approximation algorithm.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduction_dimension", &reduction_dim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recall_target", &recall_target_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_max_k", &is_max_k_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduction_input_size_override",
                                     &reduction_input_size_override_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("aggregate_to_topk", &aggregate_to_topk_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::Shape op_shape = ctx->InputXlaShape(0).value();
    xla::PrimitiveType op_type = op_shape.element_type();

    int64_t reduction_dim = reduction_dim_;
    if (reduction_dim < 0) {
      // Reverse index.
      reduction_dim += op_shape.dimensions_size();
    }
    auto cmp_builder = ctx->builder()->CreateSubBuilder(
        absl::StrFormat("top_k_%s_comparator", is_max_k_ ? "gt" : "lt"));
    xla::XlaComputation comparator =
        ComparatorBuilder(cmp_builder.get(), op_type, is_max_k_);

    xla::XlaOp init_val = xla::ConstantLiteral(
        ctx->builder(), is_max_k_ ? xla::LiteralUtil::MinValue(op_type)
                                  : xla::LiteralUtil::MaxValue(op_type));
    xla::XlaOp init_arg = xla::ConstantR0(ctx->builder(), -1);
    xla::XlaOp iota = xla::Iota(
        ctx->builder(),
        xla::ShapeUtil::MakeShapeWithType<int32_t>(op_shape.dimensions()),
        reduction_dim);
    xla::XlaOp output_tuple = ApproxTopKFn(
        ctx->builder(), {ctx->Input(0), iota}, {init_val, init_arg}, k_,
        reduction_dim, comparator, recall_target_, aggregate_to_topk_,
        reduction_input_size_override_);
    ctx->SetOutput(0, xla::GetTupleElement(output_tuple, 0));
    ctx->SetOutput(1, xla::GetTupleElement(output_tuple, 1));
  }

 protected:
  virtual xla::XlaOp ApproxTopKFn(
      xla::XlaBuilder* builder, absl::Span<const xla::XlaOp> operands,
      absl::Span<const xla::XlaOp> init_values, int64_t top_k,
      int64_t reduction_dim, const xla::XlaComputation& comparator,
      float recall_target, bool aggregate_to_topk,
      int64_t reduction_input_size_override) const = 0;

 private:
  int64_t k_;
  int64_t reduction_dim_;
  float recall_target_;
  bool is_max_k_;
  int64_t reduction_input_size_override_;
  bool aggregate_to_topk_;

  ApproxTopKOpBase(const ApproxTopKOpBase&) = delete;
  void operator=(const ApproxTopKOpBase&) = delete;
};

class TpuApproxTopKOp : public ApproxTopKOpBase {
 public:
  explicit TpuApproxTopKOp(OpKernelConstruction* ctx) : ApproxTopKOpBase(ctx) {}

 protected:
  xla::XlaOp ApproxTopKFn(
      xla::XlaBuilder* builder, absl::Span<const xla::XlaOp> operands,
      absl::Span<const xla::XlaOp> init_values, int64_t top_k,
      int64_t reduction_dim, const xla::XlaComputation& comparator,
      float recall_target, bool aggregate_to_topk,
      int64_t reduction_input_size_override) const override {
    return xla::ApproxTopK(builder, operands, init_values, top_k, reduction_dim,
                           comparator, recall_target, aggregate_to_topk,
                           reduction_input_size_override);
  }
};

class FallbackApproxTopKOp : public ApproxTopKOpBase {
 public:
  explicit FallbackApproxTopKOp(OpKernelConstruction* ctx)
      : ApproxTopKOpBase(ctx) {}

 protected:
  xla::XlaOp ApproxTopKFn(
      xla::XlaBuilder* builder, absl::Span<const xla::XlaOp> operands,
      absl::Span<const xla::XlaOp> init_values, int64_t top_k,
      int64_t reduction_dim, const xla::XlaComputation& comparator,
      float recall_target, bool aggregate_to_topk,
      int64_t reduction_input_size_override) const override {
    return xla::ApproxTopKFallback(
        builder, operands, init_values, top_k, reduction_dim, comparator,
        recall_target, aggregate_to_topk, reduction_input_size_override);
  }
};

// Register for TPU
REGISTER_XLA_OP(Name("ApproxTopK")
                    .Device(absl::Span<const absl::string_view>{
                        DEVICE_TPU, DEVICE_TPU_XLA_JIT})
                    .TypeConstraint("T", {DT_FLOAT, DT_HALF, DT_BFLOAT16}),
                TpuApproxTopKOp);

// Register for all registered devices except for TPU since it is already
// registered.
REGISTER_XLA_OP(
    Name("ApproxTopK").TypeConstraint("T", {DT_FLOAT, DT_HALF, DT_BFLOAT16}),
    FallbackApproxTopKOp);

}  // namespace
}  // namespace tensorflow
