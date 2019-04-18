/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/platform/numa.h"

namespace tensorflow {

class GPUDevice : public BaseGPUDevice {
 public:
  GPUDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            TfGpuId tf_gpu_id, const string& physical_device_desc,
            Allocator* gpu_allocator, Allocator* cpu_allocator)
      : BaseGPUDevice(options, name, memory_limit, locality, tf_gpu_id,
                      physical_device_desc, gpu_allocator, cpu_allocator,
                      false /* sync every op */, 1 /* max_streams */) {
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
    }
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    CHECK(cpu_allocator_) << "bad place 1";
    if (attr.on_host()) {
      if (attr.gpu_compatible() || force_gpu_compatible_) {
        GPUProcessState* ps = GPUProcessState::singleton();
        return ps->GetGpuHostAllocator(0);
      } else {
        return cpu_allocator_;
      }
    } else {
      return gpu_allocator_;
    }
  }

 private:
  bool force_gpu_compatible_ = false;
};

class GPUDeviceFactory : public BaseGPUDeviceFactory {
 private:
  std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& locality, TfGpuId tf_gpu_id,
      const string& physical_device_desc, Allocator* gpu_allocator,
      Allocator* cpu_allocator) override {
    return absl::make_unique<GPUDevice>(options, name, memory_limit, locality,
                                        tf_gpu_id, physical_device_desc,
                                        gpu_allocator, cpu_allocator);
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210);

//------------------------------------------------------------------------------
// A CPUDevice that optimizes for interaction with GPUs in the
// process.
// -----------------------------------------------------------------------------
class GPUCompatibleCPUDevice : public ThreadPoolDevice {
 public:
  GPUCompatibleCPUDevice(const SessionOptions& options, const string& name,
                         Bytes memory_limit, const DeviceLocality& locality,
                         Allocator* allocator)
      : ThreadPoolDevice(options, name, memory_limit, locality, allocator),
        numa_node_(locality.numa_node()) {
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
    }
  }
  ~GPUCompatibleCPUDevice() override {}

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    GPUProcessState* ps = GPUProcessState::singleton();
    if (attr.gpu_compatible() || force_gpu_compatible_) {
      return ps->GetGpuHostAllocator(numa_node_);
    } else {
      // Call the parent's implementation.
      return ThreadPoolDevice::GetAllocator(attr);
    }
  }

 private:
  bool force_gpu_compatible_ = false;
  int numa_node_ = port::kNUMANoAffinity;
};

// The associated factory.
class GPUCompatibleCPUDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back("/physical_device:CPU:0");

    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    int num_numa_nodes = options.config.experimental().use_numa_affinity()
                             ? port::NUMANumNodes()
                             : 1;
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:CPU:", i);
      int numa_node = i % num_numa_nodes;
      DeviceLocality locality;
      locality.set_numa_node(numa_node);
      devices->push_back(absl::make_unique<GPUCompatibleCPUDevice>(
          options, name, Bytes(256 << 20), DeviceLocality(),
          ProcessState::singleton()->GetCPUAllocator(numa_node)));
    }

    return Status::OK();
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("CPU", GPUCompatibleCPUDeviceFactory, 70);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
