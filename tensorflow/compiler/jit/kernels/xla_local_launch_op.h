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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_XLA_LOCAL_LAUNCH_OP_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_XLA_LOCAL_LAUNCH_OP_H_

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// XlaLocalLaunchOp is used to replace a region of the TensorFlow graph
// which will be compiled and executed using XLA.  The XlaLocalLaunchOp is
// responsible for handling interactions with the TensorFlow executor.
// Once all inputs are present, and their shapes are known, the op can
// use a 'XlaCompilationCache' to compile and execute code which is specific
// to the shapes of input Tensors.
// XlaLocalLaunchOp uses xla::LocalClient::Compile() and
// xla::LocalExecutable::Run(), and passes arguments into/out of XLA in device
// memory.
class XlaLocalLaunchOp : public OpKernel {
 public:
  explicit XlaLocalLaunchOp(OpKernelConstruction* ctx);
  ~XlaLocalLaunchOp() override;

  void Compute(OpKernelContext* ctx) override;

 private:
  // Builds a XlaCompilationCache class suitable for the current device.
  Status BuildCompilationCache(XlaCompilationCache** compiler);

  DeviceType device_type_;
  NameAttrList function_;
  int num_constant_args_;
  TF_DISALLOW_COPY_AND_ASSIGN(XlaLocalLaunchOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_XLA_LOCAL_LAUNCH_OP_H_
