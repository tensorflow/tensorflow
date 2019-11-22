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

#include <array>

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

constexpr std::array<DataType, 7> kEinsumTypes = {
    {DT_INT32, DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
     DT_COMPLEX128}};

// Kernel which compiles XlaEinsum, an einsum op accepting two inputs.
class XlaEinsumOp : public XlaOpKernel {
 public:
  explicit XlaEinsumOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("equation", &equation_));
  }

  ~XlaEinsumOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp lhs = ctx->Input(0);
    if (equation_.find(",") == equation_.npos) {
      ctx->SetOutput(0, xla::Einsum(lhs, equation_));
    } else {
      xla::XlaOp rhs = ctx->Input(1);
      ctx->SetOutput(0, xla::Einsum(lhs, rhs, equation_));
    }
  }

 private:
  string equation_;
  TF_DISALLOW_COPY_AND_ASSIGN(XlaEinsumOp);
};

// Kernel which compiles Einsum, an einsum op accepting a list of inputs.
class EinsumOp : public XlaOpKernel {
 public:
  explicit EinsumOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("equation", &equation_));
  }

  ~EinsumOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> input_handles;
    std::vector<TensorShape> input_shapes;
    OP_REQUIRES_OK(ctx,
                   ctx->InputList("inputs", &input_handles, &input_shapes));

    if (equation_.find(",") == equation_.npos) {
      OP_REQUIRES(
          ctx, input_handles.size() == 1,
          errors::InvalidArgument(
              "Einsum Op has ", input_handles.size(), " inputs ",
              " but expected 1 input since there is no ',' in equation: ",
              equation_));
      ctx->SetOutput(0, xla::Einsum(input_handles[0], equation_));
    } else {
      OP_REQUIRES(
          ctx, input_handles.size() == 2,
          errors::InvalidArgument(
              "Einsum Op has ", input_handles.size(), " inputs ",
              " but expected 2 inputs since there is a ',' in equation: ",
              equation_));
      ctx->SetOutput(
          0, xla::Einsum(input_handles[0], input_handles[1], equation_));
    }
  }

 private:
  string equation_;
  TF_DISALLOW_COPY_AND_ASSIGN(EinsumOp);
};

REGISTER_XLA_OP(Name("XlaEinsum").TypeConstraint("T", kEinsumTypes),
                XlaEinsumOp);
REGISTER_XLA_OP(Name("Einsum").TypeConstraint("T", kEinsumTypes), EinsumOp);

}  // namespace
}  // namespace tensorflow
