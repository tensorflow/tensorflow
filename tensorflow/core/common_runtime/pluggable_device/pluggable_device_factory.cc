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

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/device_id_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/types.h"

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
absl::Status SingleVirtualDeviceMemoryLimit(const std::string& platform_name,
                                            const GPUOptions& device_options,
                                            PlatformDeviceId platform_device_id,
                                            int64_t* memory_limit) {
  int64_t total_memory = 0;
  int64_t available_memory = 0;
  se::Platform* platform = PluggableDeviceMachineManager(platform_name);
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * se,
                      platform->ExecutorForDevice(platform_device_id.value()));
  if (!se->DeviceMemoryUsage(&available_memory, &total_memory)) {
    return absl::UnknownError(
        absl::StrCat("Failed to query available memory for PluggableDevice ",
                     platform_device_id.value()));
  }

  int64_t allocated_memory = 0;
  const double per_process_device_memory_fraction =
      device_options.per_process_gpu_memory_fraction();
  if (per_process_device_memory_fraction > 1.0 ||
      device_options.experimental().use_unified_memory()) {
    return absl::InternalError("Unified memory is not supported yet.");
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
  return absl::OkStatus();
}

struct TfDeviceSpec {
  PlatformDeviceId platform_device_id;
  int64_t memory_limit_bytes;
  std::optional<int> priority;
  TfDeviceId tf_device_id;

  TfDeviceSpec(PlatformDeviceId platform_device_id, int64_t memory_limit_bytes,
               std::optional<int> priority, TfDeviceId tf_device_id)
      : platform_device_id(platform_device_id),
        memory_limit_bytes(memory_limit_bytes),
        priority(priority),
        tf_device_id(tf_device_id) {}
};

// Parses the virtual devices from the device options, returns the memory
// limit, priority, and TF device IDs for each virtual device. Mappings between
// TF/virtual device IDs and  platform device IDs are registered using
// `DeviceIdManager::InsertTfPlatformDeviceIdPair`.
absl::StatusOr<std::vector<TfDeviceSpec>> ExtractVirtualDevices(
    const std::string& device_type, const std::string& platform_name,
    const GPUOptions& device_options, int visible_device_count) {
  const auto& virtual_devices = device_options.experimental().virtual_devices();
  if (std::any_of(virtual_devices.begin(), virtual_devices.end(),
                  [](const auto& virtual_device) {
                    return !virtual_device.device_ordinal().empty();
                  })) {
    return absl::UnimplementedError(
        "Device ordinal is not yet supported for pluggable virtual devices.");
  }
  if (std::any_of(virtual_devices.begin(), virtual_devices.end(),
                  [](const auto& virtual_device) {
                    return !virtual_device.priority().empty() &&
                           virtual_device.priority_size() !=
                               virtual_device.memory_limit_mb_size();
                  })) {
    return absl::InvalidArgumentError(
        "`priority` is set but its size does not match `memory_limit_mb`.");
  }

  std::vector<PlatformDeviceId> visible_device_order;
  TF_RETURN_IF_ERROR(
      tsl::ParseVisibleDeviceList(device_options.visible_device_list(),
                                  visible_device_count, &visible_device_order));
  int num_devices_to_use = visible_device_order.size();
  if (num_devices_to_use < virtual_devices.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Not enough devices to create virtual devices."
        " num_devices_to_use: ",
        num_devices_to_use, " #virtual_devices: ", virtual_devices.size()));
  }

  std::vector<TfDeviceSpec> tf_device_specs;
  int device_idx = 0;
  constexpr int64_t kMegaByte = 1ll << 20;
  for (int i = 0; i < virtual_devices.size(); ++i) {
    const PlatformDeviceId platform_device_id = visible_device_order[i];
    const auto& virtual_device = virtual_devices.Get(i);
    if (virtual_device.memory_limit_mb().empty()) {
      // Empty memory_limit_mb() implies a single virtual device using all
      // available memory.
      int64_t single_virtual_device_memory_limit = 0;
      TF_RETURN_IF_ERROR(SingleVirtualDeviceMemoryLimit(
          platform_name, device_options, platform_device_id,
          &single_virtual_device_memory_limit));
      TfDeviceId tf_device_id(device_idx++);
      tf_device_specs.emplace_back(platform_device_id,
                                   single_virtual_device_memory_limit,
                                   /*priority=*/std::nullopt, tf_device_id);
    } else {
      for (int j = 0; j < virtual_device.memory_limit_mb().size(); j++) {
        // Cast float mb value to double first for increased precision.
        int64_t memory_limit_bytes = static_cast<int64_t>(
            static_cast<double>(virtual_device.memory_limit_mb(j)) * kMegaByte);
        std::optional<int> priority =
            virtual_device.priority().empty()
                ? std::nullopt
                : std::make_optional(virtual_device.priority(j));
        TfDeviceId tf_device_id(device_idx++);
        tf_device_specs.emplace_back(platform_device_id, memory_limit_bytes,
                                     priority, tf_device_id);
      }
    }
  }

  for (const auto& tf_device_spec : tf_device_specs) {
    TF_RETURN_IF_ERROR(DeviceIdManager::InsertTfPlatformDeviceIdPair(
        DeviceType(device_type), tf_device_spec.tf_device_id,
        tf_device_spec.platform_device_id));
  }

  return tf_device_specs;
}

}  // namespace

PluggableDeviceFactory::PluggableDeviceFactory(const std::string& device_type,
                                               const std::string& platform_name)
    : device_type_(device_type), platform_name_(platform_name) {}

absl::Status PluggableDeviceFactory::ListPhysicalDevices(
    std::vector<std::string>* devices) {
  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);

  int device_count = platform->VisibleDeviceCount();
  for (int i = 0; i < device_count; ++i) {
    const std::string device_name =
        absl::StrCat("/physical_device:", device_type_, ":", i);
    devices->push_back(device_name);
  }

  return absl::OkStatus();
}

absl::Status PluggableDeviceFactory::GetDeviceDetails(
    int device_index, std::unordered_map<std::string, std::string>* details) {
  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  if (platform == nullptr) {
    return absl::OkStatus();
  }

  int device_count = platform->VisibleDeviceCount();
  if (device_index < 0 || device_index >= device_count) {
    return absl::InternalError(
        absl::StrCat("Invalid device index: ", device_index));
  }

  auto desc_status = platform->DescriptionForDevice(device_index);
  if (!desc_status.ok()) {
    return desc_status.status();
  }

  auto desc = std::move(desc_status).value();
  (*details)["device_name"] = desc->name();
  return absl::OkStatus();
}

absl::Status PluggableDeviceFactory::CreateDevices(
    const SessionOptions& options, const std::string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  TF_RETURN_IF_ERROR(ValidatePluggableDeviceMachineManager(platform_name_));
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  if (platform == nullptr) {
    return absl::OkStatus();
  }
  const int visible_device_count = platform->VisibleDeviceCount();
  if (visible_device_count <= 0) {
    return absl::OkStatus();
  }
  const auto& device_options = options.config.pluggable_device_options();
  const auto& virtual_devices = device_options.experimental().virtual_devices();
  std::vector<TfDeviceSpec> tf_device_specs;
  if (!virtual_devices.empty()) {
    TF_ASSIGN_OR_RETURN(
        tf_device_specs,
        ExtractVirtualDevices(device_type_, platform_name_, device_options,
                              visible_device_count));
  } else {
    const absl::flat_hash_map<std::string, int64_t> device_count_map(
        options.config.device_count().begin(),
        options.config.device_count().end());
    TF_ASSIGN_OR_RETURN(
        const size_t num_tf_devices,
        tsl::GetNumberTfDevicesAndConfigurePlatformDeviceId(
            device_count_map, device_type_,
            device_options.visible_device_list(), visible_device_count));

    for (int i = 0; i < num_tf_devices; ++i) {
      const TfDeviceId tf_device_id(i);
      PlatformDeviceId platform_device_id;
      TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
          DeviceType(device_type_), tf_device_id, &platform_device_id));
      int64_t single_virtual_device_memory_limit = 0;
      TF_RETURN_IF_ERROR(SingleVirtualDeviceMemoryLimit(
          platform_name_, device_options, platform_device_id,
          &single_virtual_device_memory_limit));
      tf_device_specs.emplace_back(platform_device_id,
                                   single_virtual_device_memory_limit,
                                   /*priority=*/std::nullopt, tf_device_id);
    }
  }

  const size_t total_tf_devices = tf_device_specs.size();
  std::vector<DeviceLocality> device_localities;
  TF_RETURN_IF_ERROR(GetDeviceLocalities(total_tf_devices, &device_localities));

  // Build the PluggableDevices.
  for (int di = 0; di < total_tf_devices; ++di) {
    TfDeviceId tf_device_id = tf_device_specs[di].tf_device_id;
    int64_t bytes = tf_device_specs[di].memory_limit_bytes;
    std::optional<int> priority = tf_device_specs[di].priority;
    TF_RETURN_IF_ERROR(CreatePluggableDevice(options, name_prefix, tf_device_id,
                                             bytes, priority,
                                             device_localities[di], devices));
  }
  return absl::OkStatus();
}

static std::string GetShortDeviceDescription(
    PlatformDeviceId platform_device_id, const se::DeviceDescription& desc) {
  return strings::StrCat("device: ", platform_device_id.value(),
                         ", name: ", desc.name(),
                         ", pci bus id: ", desc.pci_bus_id());
}

absl::Status PluggableDeviceFactory::CreatePluggableDevice(
    const SessionOptions& options, const std::string& name_prefix,
    TfDeviceId tf_device_id, int64_t memory_limit, std::optional<int> priority,
    const DeviceLocality& dev_locality,
    std::vector<std::unique_ptr<Device>>* devices) {
  DCHECK_GE(tf_device_id.value(), 0);
  const std::string device_name = strings::StrCat(
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
      options.config.pluggable_device_options(), tf_device_id, memory_limit);
  if (device_allocator == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Failed to get memory allocator for TF PluggableDevice ",
        tf_device_id.value(), " with", memory_limit, " bytes of memory. "));
  }
  const std::optional<AllocatorStats> stats = device_allocator->GetStats();
  if (!stats) {
    return absl::InternalError("No allocator statistics");
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
  std::string priority_str =
      priority.has_value() ? std::to_string(*priority) : "DEFAULT";
  LOG(INFO) << "Created TensorFlow device (" << device_name << " with "
            << (bytes_limit >> 20)
            << " MB memory and priority: " << priority_str
            << ") -> physical PluggableDevice ("
            << GetShortDeviceDescription(platform_device_id, *desc) << ")";
  TF_RETURN_IF_ERROR(pluggable_device->Init(options, priority));
  devices->push_back(std::move(pluggable_device));
  return absl::OkStatus();
}

absl::Status PluggableDeviceFactory::GetDeviceLocalities(
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
    // Make sure executor for this platform device is created before getting
    // device description.
    TF_RETURN_IF_ERROR(
        platform->ExecutorForDevice(platform_device_id.value()).status());
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
  return absl::OkStatus();
}

}  // namespace tensorflow
