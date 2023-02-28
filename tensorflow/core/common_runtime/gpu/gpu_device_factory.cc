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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/tsl/framework/device_id.h"

namespace tensorflow {

class GPUDevice : public BaseGPUDevice {
 public:
  GPUDevice(const SessionOptions& options, const string& name,
            Bytes memory_limit, const DeviceLocality& locality,
            tsl::TfDeviceId tf_device_id, const string& physical_device_desc,
            Allocator* gpu_allocator, Allocator* cpu_allocator)
      : BaseGPUDevice(options, name, memory_limit, locality, tf_device_id,
                      physical_device_desc, gpu_allocator, cpu_allocator,
                      false /* sync every op */),
        gpu_options_(options.config.gpu_options()) {
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
    }
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_PER_STREAM_HOST_ALLOCATOR",
                                   /*default_val=*/false,
                                   &use_per_stream_host_allocator_));
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    CHECK(cpu_allocator_) << "bad place 1";
    if (attr.on_host()) {
      if (attr.gpu_compatible() || force_gpu_compatible_) {
        GPUProcessState* ps = GPUProcessState::singleton();
        if (use_per_stream_host_allocator_) {
          return ps->GetGpuHostAllocator(gpu_options_, 0, stream_id_);
        } else {
          return ps->GetGpuHostAllocator(gpu_options_, 0);
        }
      } else {
        return cpu_allocator_;
      }
    } else {
      return gpu_allocator_;
    }
  }

 private:
  GPUOptions gpu_options_;
  bool force_gpu_compatible_ = false;
  bool use_per_stream_host_allocator_ = false;
};

//------------------------------------------------------------------------------
// A StreamDevice that manages multi-stream in GPU.
// -----------------------------------------------------------------------------
class StreamDevice : public GPUDevice {
 public:
  StreamDevice(const SessionOptions& options, const string& name,
               Bytes memory_limit, const DeviceLocality& locality,
               tsl::TfDeviceId tf_device_id, const string& physical_device_desc,
               Allocator* gpu_allocator, Allocator* cpu_allocator)
      : GPUDevice(options, name, memory_limit, locality, tf_device_id,
                  physical_device_desc, gpu_allocator, cpu_allocator) {}

  void SetRealDevice(Device* device) override { real_device_ = device; }

  Device* GetRealDevice() override { return real_device_; }

  ResourceMgr* resource_manager() override {
    return real_device_->resource_manager();
  }

 private:
  Device* real_device_;
};

class GPUDeviceFactory : public BaseGPUDeviceFactory {
 private:
  std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& locality, tsl::TfDeviceId tf_device_id,
      const string& physical_device_desc, Allocator* gpu_allocator,
      Allocator* cpu_allocator) override {
    return absl::make_unique<GPUDevice>(options, name, memory_limit, locality,
                                        tf_device_id, physical_device_desc,
                                        gpu_allocator, cpu_allocator);
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210);

class StreamDeviceFactory : public BaseGPUDeviceFactory {
 public:
  StreamDeviceFactory() { is_multi_stream_ = true; }

 private:
  std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& locality, tsl::TfDeviceId tf_device_id,
      const string& physical_device_desc, Allocator* gpu_allocator,
      Allocator* cpu_allocator) override {
    return absl::make_unique<StreamDevice>(
        options, name, memory_limit, locality, tf_device_id,
        physical_device_desc, gpu_allocator, cpu_allocator);
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("STREAM_GPU", StreamDeviceFactory);

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
        numa_node_(locality.numa_node()),
        gpu_options_(options.config.gpu_options()) {
    if (options.config.has_gpu_options()) {
      force_gpu_compatible_ =
          options.config.gpu_options().force_gpu_compatible();
    }
  }
  ~GPUCompatibleCPUDevice() override {}

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    GPUProcessState* ps = GPUProcessState::singleton();
    if (attr.gpu_compatible() || force_gpu_compatible_) {
      return ps->GetGpuHostAllocator(gpu_options_, numa_node_, stream_id_);
    } else {
      // Call the parent's implementation.
      return ThreadPoolDevice::GetAllocator(attr);
    }
  }

 private:
  bool force_gpu_compatible_ = false;
  int numa_node_;
  GPUOptions gpu_options_;
};

//------------------------------------------------------------------------------
// A CPUDevice that optimizes for interaction with multi-stream GPUs in the
// process.
// -----------------------------------------------------------------------------
class StreamCompatibleCPUDevice : public GPUCompatibleCPUDevice {
 public:
  StreamCompatibleCPUDevice(const SessionOptions& options, const string& name,
                            Bytes memory_limit, const DeviceLocality& locality,
                            Allocator* allocator)
      : GPUCompatibleCPUDevice(options, name, memory_limit, locality,
                               allocator) {}
  ~StreamCompatibleCPUDevice() override {}

  void SetRealDevice(Device* device) override { real_device_ = device; }

  Device* GetRealDevice() override { return real_device_; }

  ResourceMgr* resource_manager() override {
    return real_device_->resource_manager();
  }

 private:
  Device* real_device_;
};

// The associated factory.
class GPUCompatibleCPUDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    devices->push_back("/physical_device:CPU:0");

    return OkStatus();
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
    bool use_per_stream_host_allocator;
    int64 max_stream_group_count(1);
    if (is_multi_stream_) {
      TF_CHECK_OK(ReadBoolFromEnvVar("TF_PER_STREAM_HOST_ALLOCATOR",
                                     /*default_val=*/false,
                                     &use_per_stream_host_allocator));
      if (!use_per_stream_host_allocator) {
        return OkStatus();
      }
      std::vector<int64> gpu_stream_group_count;
      TF_CHECK_OK(ReadInt64sFromEnvVar("TF_GPU_STREAM_GROUP_COUNT",
                                       /*default_val=*/1,
                                       &gpu_stream_group_count));
      max_stream_group_count = gpu_stream_group_count[0];
      for (auto cnt : gpu_stream_group_count) {
        if (cnt > max_stream_group_count) {
          max_stream_group_count = cnt;
        }
      }
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:CPU:", i);
      int numa_node = i % num_numa_nodes;
      DeviceLocality locality;
      locality.set_numa_node(numa_node);
      for (int j = 0; j < max_stream_group_count; ++j) {
        if (is_multi_stream_) {
          name = strings::StrCat(name_prefix, "/device:STREAM_CPU_", i, ":", j);
          devices->push_back(absl::make_unique<StreamCompatibleCPUDevice>(
              options, name, Bytes(256 << 20), DeviceLocality(),
              ProcessState::singleton()->GetCPUAllocator(numa_node)));
          devices->back()->SetStreamId(j);
        } else {
          devices->push_back(absl::make_unique<GPUCompatibleCPUDevice>(
              options, name, Bytes(256 << 20), DeviceLocality(),
              ProcessState::singleton()->GetCPUAllocator(numa_node)));
        }
      }
    }
    return OkStatus();
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("CPU", GPUCompatibleCPUDeviceFactory, 70);

class StreamCompatibleCPUDeviceFactory : public GPUCompatibleCPUDeviceFactory {
 public:
  StreamCompatibleCPUDeviceFactory() { is_multi_stream_ = true; }
};
REGISTER_LOCAL_DEVICE_FACTORY("STREAM_CPU", StreamCompatibleCPUDeviceFactory);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
