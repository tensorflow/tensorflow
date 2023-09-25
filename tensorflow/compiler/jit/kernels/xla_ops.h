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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_XLA_OPS_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_XLA_OPS_H_

#include <atomic>

#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "xla/stream_executor/tf_allocator_adapter.h"
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
//
// `has_ref_vars`: whether the input computation can have reference variables.
// TODO(cheshire): instead derive this information from the input graph.
class XlaLocalLaunchBase : public AsyncOpKernel {
 public:
  XlaLocalLaunchBase(OpKernelConstruction* ctx,
                     const std::vector<int>& constants,
                     const std::vector<int>& resources,
                     const NameAttrList& function, bool has_ref_vars);
  XlaLocalLaunchBase(const XlaLocalLaunchBase&) = delete;
  XlaLocalLaunchBase& operator=(const XlaLocalLaunchBase&) = delete;
  ~XlaLocalLaunchBase() override = default;

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 protected:
  // Indexes of compile-time constant inputs
  const std::vector<int> constants_;
  // Indexes of resource inputs
  const std::vector<int> resources_;

  const NameAttrList function_;
  const XlaPlatformInfo platform_info_;

  bool has_ref_vars_;
};

// XlaLocalLaunchOp is used to replace a region of the TensorFlow graph
// which will be compiled and executed using XLA.  The XlaLocalLaunchOp is
// responsible for handling interactions with the TensorFlow executor.
// Once all inputs are present, and their shapes are known, the op can
// use a 'DeviceCompiler' to compile and execute code which is specific
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

class XlaCompileOp : public OpKernel {
 public:
  explicit XlaCompileOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  // Indexes of compile-time constant inputs
  const std::vector<int> constants_;
  // Indexes of resource inputs
  const std::vector<int> resources_;

  const NameAttrList function_;

  XlaPlatformInfo platform_info_;

  const bool must_compile_;

  // Whether the graph has TF reference variables.
  const bool has_ref_vars_;

  // cannot_compile_cluster_ is set to true if XLA returns an Unimplemented
  // error when compiling the cluster this _XlaCompile is supposed to compile.
  // If `cannot_compile_cluster_` is true then we avoid compiling this cluster
  // on any future calls to _XlaCompile.
  bool cannot_compile_cluster_ TF_GUARDED_BY(cannot_compile_cluster_mu_) =
      false;

  mutex cannot_compile_cluster_mu_;
};

class XlaRunOp : public OpKernel {
 public:
  explicit XlaRunOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  const XlaPlatformInfo platform_info_;
};

class XlaMergeOp : public OpKernel {
 public:
  explicit XlaMergeOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_XLA_OPS_H_
