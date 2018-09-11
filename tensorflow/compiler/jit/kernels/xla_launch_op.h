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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_XLA_LAUNCH_OP_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_XLA_LAUNCH_OP_H_

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {

// XlaLocalLaunchBase is almost the same as XlaLocalLaunchOp.
// The only difference is that it does not require arguments to follow
// the "constants, then regular args, then resources" order.
// It takes vectors of constant and resource arguments explicitly.
// It does not have corresponding OpDef because it is never present
// in the GraphDef.
// Currently, it is used by eager runtime. FunctionLibraryRuntime creates
// this kernel when asked to create a kernel for an XLA-compiled function.
class XlaLocalLaunchBase : public OpKernel {
 public:
  XlaLocalLaunchBase(OpKernelConstruction* ctx,
                     const std::vector<int>& constants,
                     const std::vector<int>& resources,
                     const NameAttrList& function);
  XlaLocalLaunchBase(const XlaLocalLaunchBase&) = delete;
  XlaLocalLaunchBase& operator=(const XlaLocalLaunchBase&) = delete;
  ~XlaLocalLaunchBase() override = default;

  void Compute(OpKernelContext* ctx) override;

 protected:
  // Builds a XlaCompilationCache class suitable for the current device.
  Status BuildCompilationCache(OpKernelContext* ctx,
                               XlaCompilationCache** cache);

  // Indexes of compile-time constant inputs
  std::vector<int> constants_;
  // Indexes of resource inputs
  std::vector<int> resources_;

  DeviceType device_type_;
  NameAttrList function_;
  se::Platform::Id platform_id_ = nullptr;
  bool use_multiple_streams_ = false;
  const XlaDevice::Metadata* xla_device_metadata_ = nullptr;
};

// XlaLocalLaunchOp is used to replace a region of the TensorFlow graph
// which will be compiled and executed using XLA.  The XlaLocalLaunchOp is
// responsible for handling interactions with the TensorFlow executor.
// Once all inputs are present, and their shapes are known, the op can
// use a 'XlaCompilationCache' to compile and execute code which is specific
// to the shapes of input Tensors.
// XlaLocalLaunchOp uses xla::LocalClient::Compile() and
// xla::LocalExecutable::Run(), and passes arguments into/out of XLA in device
// memory.
class XlaLocalLaunchOp : public XlaLocalLaunchBase {
 public:
  explicit XlaLocalLaunchOp(OpKernelConstruction* ctx);
  ~XlaLocalLaunchOp() override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaLocalLaunchOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_XLA_LAUNCH_OP_H_
