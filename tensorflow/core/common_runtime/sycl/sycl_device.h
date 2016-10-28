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

#define EIGEN_USE_SYCL

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class SYCLDevice : public LocalDevice {
 public:
  SYCLDevice(const SessionOptions& options, const string& name,
             Bytes memory_limit, const DeviceLocality& locality,
             const string& physical_device_desc, Allocator* allocator);
  ~SYCLDevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Status Sync() override { return Status::OK(); }
  static string GetShortDeviceDescription(/*int device_id,
                                          const DeviceDescription& desc*/) {
    return strings::StrCat("device: 0, name SYCL, pci bus id: 0");
    // return strings::StrCat("device: ", device_id, ", name: ", desc.name(),
    //                       ", pci bus id: ", desc.pci_bus_id());
  }

 private:
  Allocator* allocator_;  // Not owned
  Eigen::SyclDevice device_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_DEVICE_H_
