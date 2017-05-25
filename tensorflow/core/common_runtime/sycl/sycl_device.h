/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if !TENSORFLOW_USE_SYCL
#error This file must only be included when building TensorFlow with SYCL support
#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_DEVICE_H_

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/sycl/sycl_allocator.h"
#include "tensorflow/core/common_runtime/sycl/sycl_device_context.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class SYCLDevice : public LocalDevice {
 public:
  template <typename SYCLSelector>
  SYCLDevice(const SessionOptions &options, const string &name,
             Bytes memory_limit, const DeviceLocality &locality,
             const string &physical_device_desc, SYCLSelector sycl_selector,
             Allocator *cpu_allocator)
      : LocalDevice(
            options,
            Device::BuildDeviceAttributes(name, DEVICE_SYCL, memory_limit,
                                          locality, physical_device_desc),
            nullptr),
        cpu_allocator_(cpu_allocator),
        sycl_queue_(new Eigen::QueueInterface(sycl_selector)),
        sycl_device_(new Eigen::SyclDevice(sycl_queue_)),
        sycl_allocator_(new SYCLAllocator(sycl_queue_)),
        device_context_(new SYCLDeviceContext()) {
    set_eigen_sycl_device(sycl_device_);
    RegisterDevice();
  }

  ~SYCLDevice() override;

  void EnterLameDuckMode();

  void Compute(OpKernel *op_kernel, OpKernelContext *context) override;
  Allocator *GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto &tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor *tensor) override;

  Status FillContextMap(const Graph *graph,
                        DeviceContextMap *device_context_map) override;

  Status Sync() override;
  static string GetShortDeviceDescription(/*int device_id,
                                          const DeviceDescription& desc*/) {
    return strings::StrCat("device: 0, name SYCL, pci bus id: 0");
  }

 private:
  void RegisterDevice();

  Allocator *cpu_allocator_;           // owned
  Eigen::QueueInterface *sycl_queue_;  // owned
  Eigen::SyclDevice *sycl_device_;     // owned
  SYCLAllocator *sycl_allocator_;      // owned
  SYCLDeviceContext *device_context_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_DEVICE_H_
