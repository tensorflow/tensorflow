/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_DEVICE_WITH_CUSTOM_ALLOCATOR_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_DEVICE_WITH_CUSTOM_ALLOCATOR_H_

#include <utility>

#include "tensorflow/core/framework/device.h"
#include "tsl/framework/allocator.h"

namespace tensorflow {
namespace tfrt_stub {

class DeviceWithCustomAllocator : public tensorflow::Device {
 public:
  DeviceWithCustomAllocator(tensorflow::Device* device,
                            tensorflow::Allocator* allocator)
      : Device(device->env(), device->attributes()),
        device_(device),
        allocator_(allocator) {
    DCHECK(device_);
    DCHECK(allocator_);
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    return allocator_;
  }

  const DeviceBase* UnderlyingDevice() const override {
    return device_->UnderlyingDevice();
  }
  DeviceBase* UnderlyingDevice() override {
    return device_->UnderlyingDevice();
  }

  const CpuWorkerThreads* tensorflow_cpu_worker_threads() const override {
    return device_->tensorflow_cpu_worker_threads();
  }

  Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                int64_t step_id) override {
    return device_->GetScopedAllocator(attr, step_id);
  }

  ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
    return device_->GetScopedAllocatorMgr();
  }

  const Eigen::ThreadPoolDevice* eigen_cpu_device() override {
    return device_->eigen_cpu_device();
  }

  thread::ThreadPool* tensorflow_device_thread_pool() override {
    return device_->tensorflow_device_thread_pool();
  }

  bool has_eigen_cpu_device() const override {
    return device_->has_eigen_cpu_device();
  }

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override {
    return device_->MakeTensorFromProto(tensor_proto, alloc_attrs, tensor);
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override {
    device_->CopyTensorInSameDevice(input_tensor, output_tensor, device_context,
                                    std::move(done));
  }

  Status Sync() override { return device_->Sync(); }

  // Returns the resource manager associated w/ this device.
  ResourceMgr* resource_manager() override {
    return device_->resource_manager();
  }

 private:
  tensorflow::Device* device_ = nullptr;
  tensorflow::Allocator* allocator_ = nullptr;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_DEVICE_WITH_CUSTOM_ALLOCATOR_H_
