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
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {

class CannotBeCompiledWithXlaOp : public XlaOpKernel {
 public:
  explicit CannotBeCompiledWithXlaOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}
  void Compile(XlaOpKernelContext* context) override {
    xla::Status status(
        errors::Unimplemented("CannotBeCompiledWithXla fails by design"));
    OP_REQUIRES_OK(context, status);
  }
  bool IsExpensive() override { return false; }
};

}  // namespace

REGISTER_XLA_OP(Name("CannotBeCompiledWithXla"), CannotBeCompiledWithXlaOp);

}  // namespace tensorflow
