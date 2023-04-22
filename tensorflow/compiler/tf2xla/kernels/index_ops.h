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

// Declarations of the ArgMax/ArgMin ops using a pure XLA implementation.

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_INDEX_OPS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_INDEX_OPS_H_

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class XlaArgMinMaxOp : public XlaOpKernel {
 public:
  explicit XlaArgMinMaxOp(OpKernelConstruction* ctx, bool is_min);
  void Compile(XlaOpKernelContext* ctx) override;

 private:
  const bool is_min_;  // Are we computing ArgMin (true) or ArgMax (false)?
  const bool is_gpu_;
};

class XlaArgMaxOp : public XlaArgMinMaxOp {
 public:
  explicit XlaArgMaxOp(OpKernelConstruction* ctx);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_INDEX_OPS_H_
