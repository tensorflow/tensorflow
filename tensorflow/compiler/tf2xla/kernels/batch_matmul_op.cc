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

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace tensorflow {
namespace {

class BatchMatMulOp : public XlaOpKernel {
 public:
  explicit BatchMatMulOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_y", &adj_y_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("grad_x", &grad_x_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("grad_y", &grad_y_));

    if (ctx->HasAttr("Tout")) {
      DataType output_type;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_type));

      xla::PrimitiveType xla_type;
      OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(output_type, &xla_type));
      preferred_element_type_.emplace(xla_type);
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // TensorFlow's BatchMatMul requires the inner dimensions to match, but
    // xla::BatchDot broadcasts one of size 1, so check them here.
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape y_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, x_shape.dims() >= 2 && y_shape.dims() >= 2,
                absl::InvalidArgumentError(absl::StrCat(
                    "In[0] and In[1] ndims must be >= 2: ",
                    x_shape.DebugString(), " vs. ", y_shape.DebugString())));
    const int64_t x_inner = x_shape.dim_size(x_shape.dims() - (adj_x_ ? 2 : 1));
    const int64_t y_inner = y_shape.dim_size(y_shape.dims() - (adj_y_ ? 1 : 2));
    OP_REQUIRES(ctx, x_inner == y_inner,
                absl::InvalidArgumentError(absl::StrCat(
                    "Matrix size-incompatible: In[0]: ", x_shape.DebugString(),
                    ", In[1]: ", y_shape.DebugString())));

    xla::PrecisionConfig::Precision precision =
        tsl::tensor_float_32_execution_enabled()
            ? xla::PrecisionConfig::DEFAULT
            : xla::PrecisionConfig::HIGHEST;
    auto result =
        xla::BatchDot(MaybeConjugate(ctx->Input(0), adj_x_), adj_x_,
                      MaybeConjugate(ctx->Input(1), adj_y_), adj_y_, precision,
                      preferred_element_type_, grad_x_, grad_y_);
    ctx->SetOutput(0, result);
  }

 private:
  bool adj_x_;
  bool adj_y_;
  bool grad_x_;
  bool grad_y_;
  std::optional<xla::PrimitiveType> preferred_element_type_;
};

REGISTER_XLA_OP(Name("BatchMatMul"), BatchMatMulOp);
REGISTER_XLA_OP(Name("BatchMatMulV2"), BatchMatMulOp);
REGISTER_XLA_OP(Name("BatchMatMulV3"), BatchMatMulOp);

}  // namespace
}  // namespace tensorflow
