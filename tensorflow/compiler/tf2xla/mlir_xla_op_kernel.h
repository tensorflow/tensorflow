/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_MLIR_XLA_OP_KERNEL_H_
#define TENSORFLOW_COMPILER_TF2XLA_MLIR_XLA_OP_KERNEL_H_

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// An XlaOpKernel that's implemented by lowering using MLIR TensorFlow to HLO
// legalization.
class MlirXlaOpKernel : public XlaOpKernel {
 public:
  explicit MlirXlaOpKernel(OpKernelConstruction* ctx);

 private:
  absl::Status ContextToXlaArgs(XlaOpKernelContext* ctx,
                                std::vector<XlaCompiler::Argument>& xla_args);
  void Compile(XlaOpKernelContext* ctx) override;
  absl::Status ConstructXlaOp(XlaOpKernelContext* ctx);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_MLIR_XLA_OP_KERNEL_H_
