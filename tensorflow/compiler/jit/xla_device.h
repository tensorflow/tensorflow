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

// The XlaDevice executes a TensorFlow graph using the XLA linear algebra
// runtime.
//
// Operators assigned to an XlaDevice are compiled into XLA computations.
// Tensors on an XlaDevice are thin wrappers around XLA GlobalDataHandles; state
// is managed by XLA.
//
// XlaDevice is instantiated separately for each XLA backend (e.g., CPU or GPU),
// under different names (e.g., XLA_CPU or XLA_GPU).

#ifndef TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_
#define TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_

#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace tensorflow {

class XlaDevice : public LocalDevice {
 public:
  // Wrapper class to store metadata about the XlaDevice, where it can be
  // retrieved e.g., when lazily creating the XlaCompilationCache device.
  class Metadata {
   public:
    Metadata(int device_ordinal, se::Platform* platform,
             const DeviceType& device_type);

    // The index of the device on this host.
    int device_ordinal() const;

    se::Platform* platform() const;
    xla::LocalClient* client() const;
    const DeviceType& jit_device_type() const;

   private:
    const int device_ordinal_;
    const DeviceType device_type_;
    se::Platform* platform_;  // Not owned.

    TF_DISALLOW_COPY_AND_ASSIGN(Metadata);
  };

  // Sets `*metadata` to the XlaDevice Metadata in the XLA device used by `ctx`.
  static Status GetMetadata(OpKernelContext* ctx, const Metadata** metadata);

  // Factory function. 'platform_name' is the name of the XLA platform.
  // 'device_name' is the name of the Tensorflow device to create.
  // 'jit_device_name' is the name of the corresponding JIT device.
  // 'transfer_as_literal' is true if device<->host transfers must be done using
  // XLA's TransferLiteral{To,From}Device interface. If false, we can use
  // ThenMemcpy instead.
  static Status Create(const string& platform_name, const string& device_name,
                       int device_ordinal, const string& jit_device_name,
                       const SessionOptions& options, const string& name_prefix,
                       const XlaOpRegistry::DeviceRegistration& registration,
                       bool transfer_as_literal,
                       std::unique_ptr<XlaDevice>* device);

  XlaDevice(const SessionOptions& options, const DeviceAttributes& attrs,
            int device_ordinal, const DeviceType& jit_device_name,
            se::Platform* platform, bool transfer_as_literal);
  ~XlaDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;
  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;
  Status Sync() override { return Status::OK(); }

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map) override;

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  xla::LocalClient* client() const;
  xla::StatusOr<se::Stream*> GetStream();

  // If not already set, create and set GpuDeviceInfo.
  // Not thread-safe
  Status CreateAndSetGpuDeviceInfo();

 private:
  // The metadata of this XlaDevice.
  const Metadata xla_metadata_;
  // Which hardware device in the client's platform this XlaDevice controls.
  const int device_ordinal_;
  // The name of the device that is used to compile Ops for this XlaDevice.
  DeviceType jit_device_name_;
  // Memory allocator associated with this device.
  Allocator* xla_allocator_;                   // Not owned.
  se::Platform* platform_;                     // Not owned.
  // Stream associated with this device. Operations enqueued on this
  // stream are executed on the device. Operations include data
  // copying back and forth between CPU and the device, and
  // computations enqueued by XLA.
  xla::Backend::StreamPtr stream_;
  // Must we use XLA's transfer manager for correct host<->device transfers? if
  // false, we can use ThenMemcpy() instead.
  bool transfer_as_literal_;

  // If set, holds default device context (that we must Unref)
  // and its stream.
  std::unique_ptr<GpuDeviceInfo> gpu_device_info_;
};

// Builds OpKernel registrations on 'device' for the JIT operators
// registered on 'jit_device'. Returns ownership of a XlaDeviceOpRegistrations
// object that encapsulates the kernel registrations.
struct XlaDeviceOpRegistrations {
  std::vector<std::unique_ptr<kernel_factory::OpKernelRegistrar>>
      op_kernel_registrars;
};
XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(const char* device,
                                                   const char* jit_device);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_
