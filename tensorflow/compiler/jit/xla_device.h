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
  // Wrapper class to store metadata about the XlaDevice in the
  // resource manager, where it can be looked up e.g., when lazily
  // creating the XlaCompilationCache device.
  class Metadata : public ResourceBase {
   public:
    Metadata(int device_ordinal, perftools::gputools::Platform* platform,
             const DeviceType& device_type);
    ~Metadata() override;

    // The index of the device on this host.
    int device_ordinal() const;

    perftools::gputools::Platform* platform() const;
    xla::LocalClient* client() const;
    const DeviceType& jit_device_type() const;

    string DebugString() override;

   private:
    const int device_ordinal_;
    const DeviceType device_type_;
    perftools::gputools::Platform* platform_;  // Not owned.
  };

  // Sets `*metadata` to the XlaDevice Metadata in the resource manager of
  // `ctx`.
  static Status GetMetadata(OpKernelContext* ctx, Metadata** metadata);

  // Factory function. 'platform_name' is the name of the XLA platform.
  // 'device_name' is the name of the Tensorflow device to create.
  // 'jit_device_name' is the name of the corresponding JIT device.
  static Status Create(const string& platform_name, const string& device_name,
                       int device_ordinal, const string& jit_device_name,
                       const SessionOptions& options, const string& name_prefix,
                       std::unique_ptr<XlaDevice>* device);

  XlaDevice(const SessionOptions& options, const DeviceAttributes& attrs,
            int device_ordinal, const DeviceType& jit_device_name,
            ::perftools::gputools::Platform* platform,
            Allocator* xla_allocator);
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

 private:
  // Which hardware device in the client's platform this XlaDevice controls.
  const int device_ordinal_;
  // The name of the device that is used to compile Ops for this XlaDevice.
  const DeviceType& jit_device_name_;
  Allocator* xla_allocator_;                   // Not owned.
  ::perftools::gputools::Platform* platform_;  // Not owned.
};

// Builds dummy OpKernel registrations on 'device' for the JIT operators
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
