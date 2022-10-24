/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace tensorflow {
namespace {

REGISTER_XLA_OP(Name("XlaSort"), MlirXlaOpKernel);

class XlaKeyValueSortOp : public XlaOpKernel {
 public:
  explicit XlaKeyValueSortOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    xla::XlaOp result = xla::Sort(
        {context->Input("keys"), context->Input("values")},
        xla::CreateScalarLtComputation(
            {context->InputXlaType("keys"), context->InputXlaType("values")},
            context->builder()));
    context->SetOutput(0, xla::GetTupleElement(result, 0));
    context->SetOutput(1, xla::GetTupleElement(result, 1));
  }
};

REGISTER_XLA_OP(Name("XlaKeyValueSort"), XlaKeyValueSortOp);
REGISTER_XLA_OP(Name("XlaVariadicSort").CompileTimeConstantInput("dimension"),
                MlirXlaOpKernel);
}  // namespace
}  // namespace tensorflow
