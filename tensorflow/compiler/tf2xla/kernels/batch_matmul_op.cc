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

#include <optional>

#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/matrix.h"
#include "xla/xla_data.pb.h"
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
