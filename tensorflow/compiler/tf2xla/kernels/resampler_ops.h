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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_RESAMPLER_OPS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_RESAMPLER_OPS_H_

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// XLA op kernel for both contrib and addon flavors of TenforFlow Resampler
class ResamplerOp : public XlaOpKernel {
 public:
  explicit ResamplerOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;
};

// XLA op kernel for both contrib and addon flavors of TenforFlow Resampler
// gradient.
class ResamplerGradOp : public XlaOpKernel {
 public:
  explicit ResamplerGradOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_RESAMPLER_OPS_H_
