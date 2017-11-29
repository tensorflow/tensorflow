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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_XLA_DEVICE_LAUNCH_OP_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_XLA_DEVICE_LAUNCH_OP_H_

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// Takes a snapshot of the values of resource variable arguments, which are
// the last `num_variables` arguments. We snapshot tensors that back
// resource variables since concurrent updates may modify the shape, and it is
// important that the shapes used for compilation match the true shapes of the
// buffers.
std::vector<OptionalTensor> SnapshotResourceVariables(OpKernelContext* ctx,
                                                      int num_variables);

// The XlaDeviceLaunchOp is used to replace a region of the TensorFlow graph
// which will be compiled and executed using XLA.  The XlaDeviceLaunchOp is
// responsible for handling interactions with the TensorFlow executor.
// Once all inputs are present, and their shapes are known, the op can
// use a 'TlaJit' to compile and execute code which is specific
// to the shapes of input Tensors.
class XlaDeviceLaunchOp : public OpKernel {
 public:
  explicit XlaDeviceLaunchOp(OpKernelConstruction* ctx);
  ~XlaDeviceLaunchOp() override;

  void Compute(OpKernelContext* ctx) override;

 private:
  NameAttrList function_;

  // Number of compile-time constant arguments.
  int num_constant_args_;

  // Number of resource variable arguments.
  int num_resource_args_;

  Tensor dummy_tensor_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaDeviceLaunchOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_XLA_DEVICE_LAUNCH_OP_H_
