/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"

namespace tensorflow {
namespace {

class XlaSvdOp : public XlaOpKernel {
 public:
  explicit XlaSvdOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_iter", &max_iter_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    string precision_config_attr;
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("precision_config", &precision_config_attr));
    OP_REQUIRES(ctx,
                precision_config_.ParsePartialFromString(precision_config_attr),
                errors::InvalidArgument("Error parsing precision config."));
    if (precision_config_.operand_precision_size() == 0) {
      precision_config_.mutable_operand_precision()->Add(
          xla::PrecisionConfig::HIGHEST);
    }
  }
  void Compile(XlaOpKernelContext* ctx) override {
    auto result = xla::SVD(ctx->Input(0), max_iter_, epsilon_,
                           precision_config_.operand_precision(0));
    ctx->SetOutput(0, result.d);
    ctx->SetOutput(1, result.u);
    ctx->SetOutput(2, result.v);
  }

 private:
  int32 max_iter_;
  float epsilon_;
  xla::PrecisionConfig precision_config_;
};

class SvdOp : public XlaOpKernel {
 public:
  explicit SvdOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("compute_uv", &compute_uv_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("full_matrices", &full_matrices_));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape("input");
    int m = input_shape.dim_size(input_shape.dims() - 2);
    int n = input_shape.dim_size(input_shape.dims() - 1);
    // This is based on heuristics that approx log(n) sweep updates are needed.
    // Note: the heuristics provides no theoretical guarantee, max_iter=100 and
    // epsilon should be used to determine exit condition.
    int max_iter = 2 * tensorflow::Log2Ceiling(std::max(m, n));
    auto result = xla::SVD(ctx->Input(0), max_iter, 1e-6);
    ctx->SetOutput(0, result.d);
    if (compute_uv_) {
      int p = std::min(m, n);
      if (!full_matrices_) {
        if (p < m) {
          result.u = xla::SliceInMinorDims(result.u, {0, 0}, {m, p});
        }
        if (p < n) {
          result.v = xla::SliceInMinorDims(result.v, {0, 0}, {n, p});
        }
      }
      ctx->SetOutput(1, result.u);
      ctx->SetOutput(2, result.v);
    } else {
      auto shape =
          xla::ShapeUtil::MakeShape(ctx->input_xla_type(0), /*dimensions=*/{0});
      ctx->SetOutput(1, xla::Zeros(ctx->builder(), shape));
      ctx->SetOutput(2, xla::Zeros(ctx->builder(), shape));
    }
  }

 private:
  bool compute_uv_;
  bool full_matrices_;
};

REGISTER_XLA_OP(Name("XlaSvd").TypeConstraint("T", kFloatTypes), XlaSvdOp);
REGISTER_XLA_OP(Name("Svd").TypeConstraint("T", kFloatTypes), SvdOp);

}  // namespace
}  // namespace tensorflow
