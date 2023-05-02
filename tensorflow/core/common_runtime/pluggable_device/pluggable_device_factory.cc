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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"

#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/stream_executor/device_id_utils.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/tsl/framework/device_id_utils.h"

namespace tensorflow {
namespace {

int64_t MinSystemMemory(int64_t available_memory) {
  // We use the following heuristic for now:
  //
  // If the available_memory is < 2GiB, we allocate 225MiB to system memory,
  // Otherwise, allocate max(300MiB, kMinSystemMemoryFraction *
  // available_memory) to system memory.
  //
  // In the future we could be more sophisticated by using a table of devices.
  int64_t min_system_memory;
  constexpr float kMinSystemMemoryFraction = 0.06;
  if (available_memory < (1LL << 31)) {
    // 225MiB
    min_system_memory = 255 * 1024 * 1024;
  } else {
    // max(300 MiB, kMinSystemMemoryFraction * available_memory)
    min_system_memory = std::max(
        int64_t{314572800},
        static_cast<int64_t>(available_memory * kMinSystemMemoryFraction));
  }
#if defined(__GNUC__) && defined(__OPTIMIZE__)
// Do nothing
#elif !defined(__GNUC__) && defined(NDEBUG)
// Do nothing
#else
  // Double the amount of available PluggableDevice memory in non-opt builds
  // (debug builds in windows); because in non-opt builds more system memory is
  // necessary.
  min_system_memory *= 2;
#endif
  VLOG(5) << "available_memory = " << available_memory;
  VLOG(5) << "min_system_memory = " << min_system_memory;
  return min_system_memory;
}

// Get the memory limit for the virtual device being created on PluggableDevice
// with 'platform_device_id', when that virtual device is the only
// virtual device being created on that PluggableDevice.
Status SingleVirtualDeviceMemoryLimit(const string& platform_name,
                                      const GPUOptions& device_options,
                                      PlatformDeviceId platform_device_id,
                                      int64_t* memory_limit) {
  int64_t total_memory = 0;
  int64_t available_memory = 0;
  se::Platform* platform = PluggableDeviceMachineManager(platform_name);
  se::StreamExecutor* se = se::DeviceIdUtil::ExecutorForPlatformDeviceId(
                               platform, platform_device_id)
                               .value();
  if (!se->DeviceMemoryUsage(&available_memory, &total_memory)) {
    return errors::Unknown(
        "Failed to query available memory for PluggableDevice ",
        platform_device_id.value());
  }

  int64_t allocated_memory = 0;
  const double per_process_device_memory_fraction =
      device_options.per_process_gpu_memory_fraction();
  if (per_process_device_memory_fraction > 1.0 ||
      device_options.experimental().use_unified_memory()) {
    return errors::Internal("Unified memory is not supported yet.");
  }

  if (per_process_device_memory_fraction == 0) {
    allocated_memory = available_memory;
    const int64_t min_system_memory = MinSystemMemory(available_memory);
    if (min_system_memory < allocated_memory) {
      allocated_memory -= min_system_memory;
    }
  } else {
    allocated_memory = total_memory * per_process_device_memory_fraction;
  }
  *memory_limit = allocated_memory;
  return OkStatus();
}
}  // namespace

PluggableDeviceFactory::PluggableDeviceFactory(const string& device_type,
                                               const string& platform_name)
    : device_type_(device_type), platform_name_(platform_name) {}

Status PluggableDeviceFactory::ListPhysicalDevices(
    std::vector<string>* devices) {
  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);

  int device_count = platform->VisibleDeviceCount();
  for (int i = 0; i < device_count; ++i) {
    const string device_name =
        strings::StrCat("/physical_device:", device_type_, ":", i);
    devices->push_back(device_name);
  }

  return OkStatus();
}

Status PluggableDeviceFactory::GetDeviceDetails(
    int device_index, std::unordered_map<string, string>* details) {
  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  if (platform == nullptr) {
    return OkStatus();
  }

  int device_count = platform->VisibleDeviceCount();
  if (device_index < 0 || device_index >= device_count) {
    return errors::Internal("Invalid device index: ", device_index);
  }

  auto desc_status = platform->DescriptionForDevice(device_index);
  if (!desc_status.ok()) {
    return desc_status.status();
  }

  auto desc = std::move(desc_status).value();
  (*details)["device_name"] = desc->name();
  return OkStatus();
}

Status PluggableDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  if (platform == nullptr) {
    return OkStatus();
  }
  const int visible_device_count = platform->VisibleDeviceCount();
  if (visible_device_count <= 0) {
    return OkStatus();
  }
  const absl::flat_hash_map<std::string, int64_t> device_count_map(
      options.config.device_count().begin(),
      options.config.device_count().end());
  const auto& device_options = options.config.gpu_options();
  TF_ASSIGN_OR_RETURN(
      const size_t num_tf_devices,
      tsl::GetNumberTfDevicesAndConfigurePlatformDeviceId(
          device_count_map, device_type_, device_options.visible_device_list(),
          visible_device_count));

  const auto& virtual_devices = device_options.experimental().virtual_devices();
  if (!virtual_devices.empty())
    VLOG(2) << "Pluggable device does not support virtual device setting yet";
  std::vector<int64_t> memory_limit_bytes;
  for (int i = 0; i < num_tf_devices; ++i) {
    const TfDeviceId tf_device_id(i);
    PlatformDeviceId platform_device_id;
    TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
        DeviceType(device_type_), tf_device_id, &platform_device_id));
    int64_t single_virtual_device_memory_limit = 0;
    TF_RETURN_IF_ERROR(SingleVirtualDeviceMemoryLimit(
        platform_name_, device_options, platform_device_id,
        &single_virtual_device_memory_limit));
    memory_limit_bytes.push_back(single_virtual_device_memory_limit);
  }

  std::vector<DeviceLocality> device_localities;
  TF_RETURN_IF_ERROR(GetDeviceLocalities(num_tf_devices, &device_localities));

  // Build the PluggableDevices.
  for (int di = 0; di < num_tf_devices; ++di) {
    TfDeviceId tf_device_id(di);
    int64_t bytes = memory_limit_bytes[di];
    TF_RETURN_IF_ERROR(CreatePluggableDevice(options, name_prefix, tf_device_id,
                                             bytes, device_localities[di],
                                             devices));
  }
  return OkStatus();
}

static string GetShortDeviceDescription(PlatformDeviceId platform_device_id,
                                        const se::DeviceDescription& desc) {
  return strings::StrCat("device: ", platform_device_id.value(),
                         ", name: ", desc.name(),
                         ", pci bus id: ", desc.pci_bus_id());
}

Status PluggableDeviceFactory::CreatePluggableDevice(
    const SessionOptions& options, const string& name_prefix,
    TfDeviceId tf_device_id, int64_t memory_limit,
    const DeviceLocality& dev_locality,
    std::vector<std::unique_ptr<Device>>* devices) {
  DCHECK_GE(tf_device_id.value(), 0);
  const string device_name = strings::StrCat(
      name_prefix, "/device:", device_type_, ":", tf_device_id.value());

  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  tsl::CheckValidTfDeviceId(DeviceType(device_type_),
                            platform->VisibleDeviceCount(), tf_device_id);
  PlatformDeviceId platform_device_id;
  TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
      DeviceType(device_type_), tf_device_id, &platform_device_id));
  int numa_node = dev_locality.numa_node();

  auto desc_status = platform->DescriptionForDevice(platform_device_id.value());
  if (!desc_status.ok()) {
    return desc_status.status();
  }
  auto desc = std::move(desc_status).value();
  PluggableDeviceProcessState* process_state =
      PluggableDeviceProcessState::singleton(device_type_, platform_name_);
  Allocator* device_allocator = process_state->GetPluggableDeviceAllocator(
      options.config.gpu_options(), tf_device_id, memory_limit);
  if (device_allocator == nullptr) {
    return errors::Internal(
        "Failed to get memory allocator for TF PluggableDevice ",
        tf_device_id.value(), " with", memory_limit, " bytes of memory. ");
  }
  const std::optional<AllocatorStats> stats = device_allocator->GetStats();
  if (!stats) {
    return errors::Internal("No allocator statistics");
  }
  // 'memory_limit' is the required memory size, but if the allocator with
  // given 'tf_device_id' was created before, we'll use it instead of creating
  // a new one (as TF Device is a shared resource), in which case the actual
  // memory limit represented by 'stats.bytes_limit' used by that allocator
  // may be different (which should be an error).
  int64_t bytes_limit = stats->bytes_limit ? *stats->bytes_limit : 0;
  auto pluggable_device = std::make_unique<PluggableDevice>(
      options, device_name, device_type_, platform_name_,
      static_cast<Bytes>(bytes_limit), dev_locality, tf_device_id,
      GetShortDeviceDescription(platform_device_id, *desc), device_allocator,
      ProcessState::singleton()->GetCPUAllocator(numa_node),
      false /*sync every op*/);
  LOG(INFO) << "Created TensorFlow device (" << device_name << " with "
            << (bytes_limit >> 20)
            << " MB memory) -> physical PluggableDevice ("
            << GetShortDeviceDescription(platform_device_id, *desc) << ")";
  TF_RETURN_IF_ERROR(pluggable_device->Init(options));
  devices->push_back(std::move(pluggable_device));
  return OkStatus();
}

Status PluggableDeviceFactory::GetDeviceLocalities(
    int num_tf_devices, std::vector<DeviceLocality>* device_localities) {
  for (int i = 0; i < num_tf_devices; ++i) {
    TfDeviceId tf_device_id(i);
    PlatformDeviceId platform_device_id;
    TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
        DeviceType(device_type_), tf_device_id, &platform_device_id));
    // Get PluggableDevice bus_id from its reported NUMA affinity. Because
    // devices are virtualized in some environment, we can't just use the device
    // id. NUMA locales are indexed from 0, buses are indexed from 1.
    se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
    auto desc_status =
        platform->DescriptionForDevice(platform_device_id.value());
    if (!desc_status.ok()) {
      return desc_status.status();
    }
    auto desc = std::move(desc_status).value();
    int numa_node = desc->numa_node();
    if (numa_node < 0) {
      // For some reason the StreamExecutor couldn't get the NUMA
      // affinity of the device. If this is not a multi-socket mobo with
      // devices local to different buses, it doesn't matter. If it is,
      // we may run into trouble later with data transfer operations.
      // The trouble may manifest as slower than expected performance,
      // or outright failures.
      LOG(INFO) << "Could not identify NUMA node of platform " << device_type_
                << " ID " << platform_device_id
                << ", defaulting to 0. Your kernel may not have been built "
                << "with NUMA support.";
      numa_node = 0;
    }
    DeviceLocality dev_locality;
    dev_locality.set_numa_node(numa_node);
    dev_locality.set_bus_id(numa_node + 1);
    device_localities->push_back(dev_locality);
    VLOG(1) << "PluggableDevice PlatformDeviceId " << platform_device_id
            << " TfDeviceId " << tf_device_id << " on bus "
            << dev_locality.bus_id() << " numa: " << numa_node
            << "DeviceLocality: " << dev_locality.DebugString();
  }
  return OkStatus();
}

}  // namespace tensorflow
