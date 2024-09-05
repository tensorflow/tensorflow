/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/stochastic_cast_op.h"

#include <string>

#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

class StochasticCastToInt : public XlaOpKernel {
  static constexpr int kInputIndex = 0;

 public:
  explicit StochasticCastToInt(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &from_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &to_type_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    shape = ctx->InputShape(kInputIndex);
    absl::StatusOr<xla::XlaOp> randoms_or = BuildUniformRandoms(
        ctx, from_type_, device_type_string_, shape, xla::Zero, xla::One);
    OP_REQUIRES_OK(ctx, randoms_or.status());
    xla::XlaOp input = ctx->Input(kInputIndex);
    if (from_type_ == DT_BFLOAT16) {
      input = xla::ConvertElementType(input, xla::F32);
    }
    xla::XlaOp result = xla::Select(
        xla::Lt(input, xla::ScalarLike(input, 0)),
        xla::Floor(xla::Add(input, randoms_or.value())),
        xla::Floor(xla::Sub(xla::Add(input, xla::ScalarLike(input, 1)),
                            randoms_or.value())));
    result = xla::Select(xla::Eq(input, xla::Floor(input)), input, result);
    xla::PrimitiveType to_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(to_type_, &to_type));
    result = xla::ConvertElementType(result, to_type);
    ctx->SetOutput(0, result);
  }

 private:
  DataType from_type_;
  DataType to_type_;
  std::string device_type_string_;
};

REGISTER_XLA_OP(Name("StochasticCastToInt")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("Tin",
                                    {DT_DOUBLE, DT_FLOAT, DT_HALF, DT_BFLOAT16})
                    .TypeConstraint("Tout", {DT_INT32, DT_INT16, DT_INT8}),
                StochasticCastToInt);
// TODO(b/232442915): add path to stochastically cast big floats to small
// floats.
}  // namespace
}  // namespace tensorflow
