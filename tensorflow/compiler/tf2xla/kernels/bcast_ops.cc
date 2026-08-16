/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific Ops for broadcasting used in gradient
// code.

#include <cstdint>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/literal.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace {

// Given shapes of two tensors, computes the broadcast shape.
class BCastArgsOp : public XlaOpKernel {
 public:
  explicit BCastArgsOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 2,
        absl::UnimplementedError("Broadcast for n-ary operations (n > 2)"));
    absl::InlinedVector<BCast::Vec, 2> shapes;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const TensorShape in_shape = ctx->InputShape(i);
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsVector(in_shape),
          absl::InvalidArgumentError(absl::StrCat(
              "In[", i, "] must be a vector.", in_shape.DebugString())));
      std::vector<int64_t> shape;
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(
                              i, &shape, xla::ValueInferenceMode::kUpperBound));
      shapes.push_back(BCast::Vec(shape.begin(), shape.end()));
    }
    BCast bcast(shapes[0], shapes[1]);
    OP_REQUIRES(ctx, bcast.IsValid(),
                absl::InvalidArgumentError(absl::StrCat(
                    "Incompatible shapes: [", absl::StrJoin(shapes[0], ","),
                    "] vs. [", absl::StrJoin(shapes[1], ","), "]")));

    DataType val_type = ctx->expected_output_dtype(0);
    const int64_t len = bcast.output_shape().size();
    Tensor output(val_type, TensorShape({len}));
    for (int64_t i = 0; i < len; ++i) {
      if (val_type == DT_INT32) {
        output.flat<int32_t>()(i) =
            static_cast<int32_t>(bcast.output_shape()[i]);
      } else {
        output.flat<int64_t>()(i) =
            static_cast<int64_t>(bcast.output_shape()[i]);
      }
    }
    ctx->SetConstantOutput(0, output);
  }

 private:
  BCastArgsOp(const BCastArgsOp&) = delete;
  void operator=(const BCastArgsOp&) = delete;
};
REGISTER_XLA_OP(Name("BroadcastArgs")
                    .CompileTimeConstantInput("s0")
                    .CompileTimeConstantInput("s1"),
                BCastArgsOp);

// Given shapes of two tensors, computes the reduction indices for the
// gradient computation.
//
// TODO(zhifengc):
//   1. Adds support for n-ary (n >= 2).
class BCastGradArgsOp : public XlaOpKernel {
 public:
  explicit BCastGradArgsOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  }

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(
        ctx, ctx->num_inputs() == 2,
        absl::UnimplementedError("Broadcast for n-ary operations (n > 2)"));

    absl::InlinedVector<BCast::Vec, 4> shapes;
    absl::InlinedVector<std::vector<bool>, 2> dynamic_dims;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const TensorShape in_shape = ctx->InputShape(i);
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsVector(in_shape),
          absl::InvalidArgumentError(absl::StrCat(
              "In[", i, "] must be a vector.", in_shape.DebugString())));
      std::vector<int64_t> vec;
      // Technically we don't need to infer the upper-bound here. However the
      // forward path uses the upperbound as bounded shape so we need backward
      // path to use the same shape to decide the reduction indices.
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(
                              i, &vec, xla::ValueInferenceMode::kUpperBound));
      std::vector<bool> dynamic;
      OP_REQUIRES_OK(ctx, ctx->ResolveInputDynamismIntoPredVector(i, &dynamic));
      OP_REQUIRES(ctx, dynamic.size() == vec.size(),
                  absl::InternalError(absl::StrCat(
                      "Size mismatch between input shape vector size (",
                      vec.size(), ") and dynamism vector size (",
                      dynamic.size(), ") for input ", i)));

      shapes.push_back(BCast::Vec(vec.begin(), vec.end()));
      dynamic_dims.push_back(std::move(dynamic));
    }

    // A dimension resolved from its upper bound (rather than a true
    // compile-time constant) can numerically disagree with the other
    // operand's static dimension even though the two are guaranteed equal at
    // runtime. The forward broadcasting op already requires this, or it
    // would have failed itself. Reconcile such axes, aligned from the
    // trailing dimension per standard broadcasting rules, before the static
    // compatibility check below, so a dynamic upper bound doesn't cause a
    // false-positive shape-incompatibility failure.
    const int64_t rank0 = shapes[0].size();
    const int64_t rank1 = shapes[1].size();
    const int64_t common_rank = rank0 < rank1 ? rank0 : rank1;
    for (int64_t k = 0; k < common_rank; ++k) {
      const int64_t i0 = rank0 - 1 - k;
      const int64_t i1 = rank1 - 1 - k;
      int64_t& d0 = shapes[0][i0];
      int64_t& d1 = shapes[1][i1];
      if (d0 == d1 || d0 == 1 || d1 == 1) continue;
      if (dynamic_dims[0][i0]) {
        d0 = d1;
      } else if (dynamic_dims[1][i1]) {
        d1 = d0;
      }
    }

    BCast bcast(shapes[0], shapes[1]);
    OP_REQUIRES(ctx, bcast.IsValid(),
                absl::InvalidArgumentError(absl::StrCat(
                    "Incompatible shapes: [", absl::StrJoin(shapes[0], ","),
                    "] vs. [", absl::StrJoin(shapes[1], ","), "]")));
    Output(ctx, 0, bcast.grad_x_reduce_idx());
    Output(ctx, 1, bcast.grad_y_reduce_idx());
  }

 private:
  void Output(XlaOpKernelContext* ctx, int idx, const BCast::Vec& v) {
    const int64_t len = v.size();
    DataType val_type = ctx->expected_output_dtype(idx);
    Tensor constant(val_type, TensorShape({len}));
    for (int64_t i = 0; i < len; ++i) {
      if (val_type == DT_INT32) {
        constant.flat<int32_t>()(i) = static_cast<int32_t>(v[i]);
      } else {
        constant.flat<int64_t>()(i) = static_cast<int64_t>(v[i]);
      }
    }
    ctx->SetConstantOutput(idx, constant);
  }

  BCastGradArgsOp(const BCastGradArgsOp&) = delete;
  void operator=(const BCastGradArgsOp&) = delete;
};

REGISTER_XLA_OP(Name("BroadcastGradientArgs")
                    .CompileTimeConstantInput("s0")
                    .CompileTimeConstantInput("s1"),
                BCastGradArgsOp);

}  // namespace
}  // namespace tensorflow
