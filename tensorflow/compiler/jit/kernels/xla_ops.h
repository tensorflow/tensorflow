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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {

// Holds some information about the platform on which an
// XlaLaunch/_XlaCompile/_XlaRun op must run on.
class XlaPlatformInfo {
 public:
  XlaPlatformInfo() : device_type_("") {}
  XlaPlatformInfo(XlaPlatformInfo&&) = default;
  explicit XlaPlatformInfo(const DeviceType device_type,
                           se::Platform::Id platform_id,
                           const XlaDevice::Metadata* xla_device_metadata,
                           std::unique_ptr<XlaAllocator> xla_allocator,
                           xla::DeviceMemoryAllocator* device_allocator)
      : device_type_(device_type),
        platform_id_(platform_id),
        xla_device_metadata_(xla_device_metadata),
        xla_allocator_(std::move(xla_allocator)),
        device_allocator_(device_allocator) {
    CHECK((device_allocator_ != nullptr) ^ (xla_allocator_.get() != nullptr));
  }

  XlaPlatformInfo& operator=(XlaPlatformInfo&& other) = default;

  bool UseMultipleStreams() const {
    return xla_device_metadata_ && xla_device_metadata_->UseMultipleStreams();
  }

  xla::DeviceMemoryAllocator* allocator() const {
    return device_allocator_ ? device_allocator_ : xla_allocator_.get();
  }
  DeviceType device_type() const { return device_type_; }

  // This is equal to xla_device_metadata()->platform()->id() if
  // xla_device_metadata() is not nullptr.
  se::Platform::Id platform_id() const { return platform_id_; }

  // This may be null if the op this XlaPlatformInfo is for was not placed on an
  // XLA device.
  const XlaDevice::Metadata* xla_device_metadata() const {
    return xla_device_metadata_;
  }
  bool is_on_xla_device() const { return xla_device_metadata() != nullptr; }

 private:
  DeviceType device_type_;
  se::Platform::Id platform_id_;

  // xla_device_metadata_ lives in the tensorflow::DeviceBase in which the
  // XlaLaunch/_XlaCompile/_XlaRun op is placed and thus does not die before the
  // XlaLaunch/_XlaCompile/_XlaRun OpKernel.
  const XlaDevice::Metadata* xla_device_metadata_;

  // If the op associated with this XlaPlatformInfo is placed on an XLA device
  // then device_allocator_ is the xla::Backend's memory allocator and
  // xla_allocator_ is null.  If the op is placed on a regular CPU or GPU device
  // then device_allocator_ is null and xla_allocator_ points to an appropriate
  // XlaAllocator instance.
  std::unique_ptr<XlaAllocator> xla_allocator_;
  xla::DeviceMemoryAllocator* device_allocator_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaPlatformInfo);
};

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
  // Indexes of compile-time constant inputs
  const std::vector<int> constants_;
  // Indexes of resource inputs
  const std::vector<int> resources_;

  const NameAttrList function_;
  const XlaPlatformInfo platform_info_;
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

  // cannot_compile_cluster_ is set to true if XLA returns an Unimplemented
  // error when compiling the cluster this _XlaCompile is supposed to compile.
  // If `cannot_compile_cluster_` is true then we avoid compiling this cluster
  // on any future calls to _XlaCompile.
  bool cannot_compile_cluster_ GUARDED_BY(cannot_compile_cluster_mu_) = false;

  mutex cannot_compile_cluster_mu_;
};

class XlaRunOp : public OpKernel {
 public:
  explicit XlaRunOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  const XlaPlatformInfo platform_info_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_XLA_LAUNCH_OP_H_
