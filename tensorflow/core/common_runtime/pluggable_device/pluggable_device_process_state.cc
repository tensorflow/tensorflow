/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"

#include <cstring>
#include <unordered_map>
#include <vector>

#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_simple_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

/*static*/ PluggableDeviceProcessState* PluggableDeviceProcessState::singleton(
    const string& device_type, const string& platform_name) {
  using ProcessStateMap =
      std::unordered_map<string, PluggableDeviceProcessState*>;
  static ProcessStateMap* process_state_map = new ProcessStateMap;
  auto iter = process_state_map->find(platform_name);
  if (iter != process_state_map->end()) {
    return iter->second;
  }
  (*process_state_map)[platform_name] =
      new PluggableDeviceProcessState(device_type, platform_name);
  return (*process_state_map)[platform_name];
}

PluggableDeviceProcessState::PluggableDeviceProcessState(
    const string& device_type, const string& platform_name)
    : pluggable_device_enabled_(false),
      device_type_(device_type),
      platform_name_(platform_name) {
  process_state_ = ProcessState::singleton();
}

int PluggableDeviceProcessState::BusIdForPluggableDevice(
    TfDeviceId tf_device_id) {
  // Return the NUMA node associated with the PluggableDevice's StreamExecutor.
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  se::StreamExecutor* se = DeviceIdUtil::ExecutorForTfDeviceId(
                               DeviceType(device_type_), platform, tf_device_id)
                               .ValueOrDie();
  int numa_node = se->GetDeviceDescription().numa_node();
  // `bus_id` must be non-negative. If the `numa_node` is unknown, use 0.
  return numa_node >= 0 ? numa_node : 0;
}

Allocator* PluggableDeviceProcessState::GetPluggableDeviceAllocator(
    const GPUOptions& options, TfDeviceId tf_device_id, size_t total_bytes) {
  DCHECK(process_state_);
  const string& allocator_type = options.allocator_type();
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  mutex_lock lock(mu_);
  DeviceIdUtil::CheckValidTfDeviceId(DeviceType(device_type_), platform,
                                     tf_device_id);

  if (tf_device_id.value() >=
      static_cast<int64_t>(pluggable_device_allocators_.size())) {
    pluggable_device_allocators_.resize(tf_device_id.value() + 1);
  }

  AllocatorParts& allocator_parts =
      pluggable_device_allocators_[tf_device_id.value()];
  if (allocator_parts.allocator == nullptr) {
    if (!allocator_type.empty()) {
      LOG(ERROR) << "Invalid allocator type: " << allocator_type;
      return nullptr;
    }

    PlatformDeviceId platform_device_id;
    TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(
        DeviceType(device_type_), tf_device_id, &platform_device_id));

    int bus_id = BusIdForPluggableDevice(tf_device_id);
    DCHECK_GE(bus_id, 0);
    while (bus_id >= pluggable_device_visitors_.size()) {
      pluggable_device_visitors_.push_back({});
    }

    bool use_unified_memory = options.per_process_gpu_memory_fraction() > 1.0 ||
                              options.experimental().use_unified_memory();
    DeviceMemAllocator* sub_allocator = new DeviceMemAllocator(
        DeviceIdUtil::ExecutorForPlatformDeviceId(platform, platform_device_id)
            .ValueOrDie(),
        platform_device_id, use_unified_memory,
        pluggable_device_visitors_[bus_id], {});

    Allocator* device_allocator = nullptr;
    auto cplatform = dynamic_cast<se::CPlatform*>(platform);
    if (cplatform == nullptr) {
      LOG(FATAL) << "PluggableDevice's platform must be of type "  // Crash OK
                 << "stream_executor::CPlatform";
    }
    if (cplatform->UseBfcAllocator()) {
      device_allocator = new PluggableDeviceBFCAllocator(
          sub_allocator, total_bytes, options,
          strings::StrCat("PluggableDevice_", tf_device_id.value(), "_bfc"));
    } else {
      device_allocator = new PluggableDeviceSimpleAllocator(sub_allocator);
    }

    allocator_parts = {std::unique_ptr<Allocator>(device_allocator),
                       device_allocator, sub_allocator};
  }
  return allocator_parts.allocator.get();
}

Allocator* PluggableDeviceProcessState::GetPluggableDeviceHostAllocator(
    int numa_node) {
  DCHECK(process_state_);
  if (!HasPluggableDevice()) {
    return process_state_->GetCPUAllocator(numa_node);
  }
  if (numa_node == port::kNUMANoAffinity) {
    numa_node = 0;
  }
  {
    // Here we optimize the most common use case where
    // pluggable_device_host_allocators_ have already been populated and since
    // we're only reading these vectors, we can get by with a shared lock. In
    // the slower case, we take a unique lock and populate these vectors.
    tf_shared_lock lock(mu_);
    if (static_cast<int>(pluggable_device_host_allocators_.size()) >
        numa_node) {
      return pluggable_device_host_allocators_[0].allocator.get();
    }
  }

  mutex_lock lock(mu_);
  // Find the first valid StreamExecutor to request PluggableDevice host memory
  // through, since any will work.
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  se::StreamExecutor* se = nullptr;
  for (int i = 0; i < static_cast<int>(pluggable_device_allocators_.size());
       ++i) {
    if (pluggable_device_allocators_[i].allocator != nullptr) {
      se = DeviceIdUtil::ExecutorForTfDeviceId(DeviceType(device_type_),
                                               platform, TfDeviceId(i))
               .ValueOrDie();
      break;
    }
  }

  DCHECK_NE(nullptr, se);

  while (static_cast<int>(pluggable_device_host_allocators_.size()) <=
         numa_node) {
    while (pluggable_device_host_alloc_visitors_.size() <= numa_node) {
      pluggable_device_host_alloc_visitors_.push_back({});
    }
    while (pluggable_device_host_free_visitors_.size() <= numa_node) {
      pluggable_device_host_free_visitors_.push_back({});
    }
    SubAllocator* sub_allocator = new DeviceHostAllocator(
        se, numa_node, pluggable_device_host_alloc_visitors_[numa_node],
        pluggable_device_host_free_visitors_[numa_node]);
    int64_t pluggable_device_host_mem_limit_in_mb = -1;
    Status status = ReadInt64FromEnvVar("TF_GPU_HOST_MEM_LIMIT_IN_MB",
                                        1LL << 16 /*64GB max by default*/,
                                        &pluggable_device_host_mem_limit_in_mb);
    if (!status.ok()) {
      LOG(ERROR) << "GetPluggableDeviceHostAllocator: "
                 << status.error_message();
    }
    int64_t pluggable_device_host_mem_limit =
        pluggable_device_host_mem_limit_in_mb << 20;

    Allocator* allocator = new BFCAllocator(
        sub_allocator, pluggable_device_host_mem_limit, true /*allow_growth*/,
        "pluggable_device_host_bfc" /*name*/);

    if (LogMemory::IsEnabled() && !allocator->TracksAllocationSizes()) {
      // Wrap the allocator to track allocation ids for better logging
      // at the cost of performance.
      allocator = new TrackingAllocator(allocator, true);
    }
    pluggable_device_host_allocators_.push_back(
        {std::unique_ptr<Allocator>(allocator), nullptr /*bfc_allocator*/,
         sub_allocator});
  }
  return pluggable_device_host_allocators_[0].allocator.get();
}

}  // namespace tensorflow
