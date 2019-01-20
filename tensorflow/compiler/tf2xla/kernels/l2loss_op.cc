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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/no_op.h"

namespace tensorflow {
namespace {

class L2LossOp : public XlaOpKernel {
 public:
  explicit L2LossOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<int64> dims(ctx->InputShape(0).dims());
    std::iota(dims.begin(), dims.end(), 0);

    DataType dtype = ctx->input_type(0);
    xla::XlaBuilder* const b = ctx->builder();

    //  output = sum(t ** 2) / 2
    const DataType accumulation_type = XlaHelpers::SumAccumulationType(dtype);
    auto t = XlaHelpers::ConvertElementType(ctx->Input(0), accumulation_type);
    auto square = xla::Mul(t, t);
    auto reduce = xla::Reduce(square, XlaHelpers::Zero(b, accumulation_type),
                              *ctx->GetOrCreateAdd(accumulation_type), dims);
    auto deconverted = XlaHelpers::ConvertElementType(reduce, dtype);
    auto two = XlaHelpers::IntegerLiteral(b, dtype, 2);
    ctx->SetOutput(0, xla::Div(deconverted, two));
  }
};

REGISTER_XLA_OP(Name("L2Loss"), L2LossOp);

}  // namespace
}  // namespace tensorflow
