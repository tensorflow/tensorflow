/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_device_description.h"

// clang-format off
#include <level_zero/ze_api.h>
// clang-format on

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/read_numa_node.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/numbers.h"

namespace stream_executor::sycl {

// TODO(intel-tf): Use common error utility code across level zero api uses.
#define RETURN_IF_ZE_ERROR(expr, msg)                            \
  do {                                                           \
    ze_result_t result = (expr);                                 \
    if (result != ZE_RESULT_SUCCESS) {                           \
      return absl::InternalError(                                \
          absl::StrCat(msg, ", got Level Zero error ", result)); \
    }                                                            \
  } while (0)

namespace {

// Current level zero release drivers report incorrect memory properties on some
// platforms. Therefore, we use values from the official product specifications.
// TODO(intel-tf): Implement a proper memory bandwidth estimation based on
// level zero memory properties.
int64_t EstimateMemoryBandwidth(const OneAPIComputeCapability& oneapi_cc) {
  if (oneapi_cc.IsBMG()) {
    // https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html
    return 456'000'000'000;
  } else if (oneapi_cc.IsPVC()) {
    // https://www.intel.com/content/www/us/en/products/sku/232876/intel-data-center-gpu-max-1100/specifications.html
    return 1'228'800'000'000;
  } else if (oneapi_cc.IsDG2()) {
    // https://www.intel.com/content/www/us/en/products/sku/227959/intel-arc-a380-graphics/specifications.html
    return 186'000'000'000;
  } else {
    return 456'000'000'000;  // Default to BMG bandwidth.
  }
}

// TODO(intel-tf): Use direct Level Zero API when available.
int DetermineFpusPerCore(const ze_device_properties_t& props) {
  return props.numEUsPerSubslice * props.physicalEUSimdWidth;
}

// Subgroup size is equivalent to warp size in CUDA terminology. Howerver, on
// Intel GPUs, a device may support multiple subgroup sizes. Compiler can
// generate code for any of the available subgroup sizes. Here we pick the
// largest one as the representative warp size, though kernel performance may
// vary with different subgroup sizes.
int64_t DetermineThreadsPerWarp(
    const ze_device_compute_properties_t& compute_props,
    const ze_device_properties_t& device_props) {
  uint32_t subgroup_size = 0;
  for (uint32_t i = 0; i < compute_props.numSubGroupSizes; ++i) {
    subgroup_size = std::max(subgroup_size, compute_props.subGroupSizes[i]);
  }
  if (subgroup_size != 0) {
    return subgroup_size;
  }
  return 32;  // Default subgroup size to conform to CUDA warp size.
}

// Parses version strings of the form "major.minor.patch+build#". For example,
// "1.2.3+456" will be parsed as {1, 2, 3}. If parsing fails, returns {0, 0, 0}.
SemanticVersion ParseOrDefaultDriverVersion(absl::string_view version_str) {
  if (version_str.empty()) {
    return SemanticVersion{0, 0, 0};
  }
  // Split by '+' to remove build# if present.
  std::array<absl::string_view, 1> version_str_parts =
      absl::StrSplit(version_str, '+');

  absl::StatusOr<SemanticVersion> version =
      SemanticVersion::ParseFromString(version_str_parts[0]);
  if (version.ok()) {
    return version.value();
  }
  return SemanticVersion{0, 0, 0};
}

SemanticVersion GetOrDefaultLevelZeroDriverVersion(ze_driver_handle_t driver) {
  using pfn_driver_version_t =
      ze_result_t (*)(ze_driver_handle_t, char*, size_t*);
  pfn_driver_version_t driver_version_func = nullptr;
  zeDriverGetExtensionFunctionAddress(
      driver, "zeIntelGetDriverVersionString",
      reinterpret_cast<void**>(&driver_version_func));
  if (driver_version_func) {
    size_t driver_version_string_size = 0;
    // Note: level zero api for driver version string uses array of char that is
    // not null-terminated. The first call is to get the required buffer size
    // for the driver version string, and the second call is to get the actual
    // driver version string.
    driver_version_func(driver, nullptr, &driver_version_string_size);
    std::vector<char> driver_version_string(driver_version_string_size);
    driver_version_func(driver, driver_version_string.data(),
                        &driver_version_string_size);
    return ParseOrDefaultDriverVersion(absl::string_view(
        driver_version_string.data(), driver_version_string_size));
  }
  return SemanticVersion{0, 0, 0};
}

std::string GetPciBusId(ze_device_handle_t lz_device) {
  ze_pci_ext_properties_t pci_props{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
  if (zeDevicePciGetPropertiesExt(lz_device, &pci_props) == ZE_RESULT_SUCCESS) {
    return absl::StrFormat("%04x:%02x:%02x.%x", pci_props.address.domain,
                           pci_props.address.bus, pci_props.address.device,
                           pci_props.address.function);
  }
  return "";
}

SemanticVersion CompileTimeToolkitVersion() { return SemanticVersion{0, 0, 0}; }

}  // namespace

absl::StatusOr<std::unique_ptr<DeviceDescription>>
CreateOneApiDeviceDescription(int device_ordinal) {
  TF_ASSIGN_OR_RETURN(::sycl::device sycl_device,
                      SyclDevicePool::GetDevice(device_ordinal));
  ze_device_handle_t lz_device =
      ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(sycl_device);
  ze_driver_handle_t lz_driver =
      ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(
          sycl_device.get_platform());

  ze_device_ip_version_ext_t ip_version_ext{
      ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT, nullptr};
  ze_device_properties_t device_props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
                                      &ip_version_ext};
  RETURN_IF_ZE_ERROR(zeDeviceGetProperties(lz_device, &device_props),
                     "zeDeviceGetProperties");

  ze_device_compute_properties_t compute_props{
      ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
  RETURN_IF_ZE_ERROR(zeDeviceGetComputeProperties(lz_device, &compute_props),
                     "zeDeviceGetComputeProperties");

  uint32_t memory_prop_count = 0;
  RETURN_IF_ZE_ERROR(
      zeDeviceGetMemoryProperties(lz_device, &memory_prop_count, nullptr),
      "zeDeviceGetMemoryProperties(count)");
  std::vector<ze_device_memory_properties_t> memory_props(memory_prop_count);
  for (auto& prop : memory_props) {
    prop.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    prop.pNext = nullptr;
  }
  if (!memory_props.empty()) {
    RETURN_IF_ZE_ERROR(zeDeviceGetMemoryProperties(
                           lz_device, &memory_prop_count, memory_props.data()),
                       "zeDeviceGetMemoryProperties");
  }

  uint32_t cache_prop_count = 0;
  RETURN_IF_ZE_ERROR(
      zeDeviceGetCacheProperties(lz_device, &cache_prop_count, nullptr),
      "zeDeviceGetCacheProperties(count)");
  std::vector<ze_device_cache_properties_t> cache_props(cache_prop_count);
  for (auto& prop : cache_props) {
    prop.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    prop.pNext = nullptr;
  }
  if (!cache_props.empty()) {
    RETURN_IF_ZE_ERROR(zeDeviceGetCacheProperties(lz_device, &cache_prop_count,
                                                  cache_props.data()),
                       "zeDeviceGetCacheProperties");
  }

  ze_driver_properties_t driver_props{ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES};
  RETURN_IF_ZE_ERROR(zeDriverGetProperties(lz_driver, &driver_props),
                     "zeDriverGetProperties");

  DeviceDescription desc;
  desc.set_name(device_props.name);
  desc.set_device_vendor("Intel Corporation");

  desc.set_oneapi_compute_capability(ip_version_ext.ipVersion);

  SemanticVersion driver_version =
      GetOrDefaultLevelZeroDriverVersion(lz_driver);
  desc.set_driver_version(driver_version);
  // The sycl executor uses sycl runtime which is built on top of Level Zero.
  desc.set_runtime_version(SemanticVersion{__LIBSYCL_MAJOR_VERSION,
                                           __LIBSYCL_MINOR_VERSION,
                                           __LIBSYCL_PATCH_VERSION});
  desc.set_compile_time_toolkit_version(CompileTimeToolkitVersion());

  std::string pci_bus_id = GetPciBusId(lz_device);
  desc.set_pci_bus_id(pci_bus_id);
  std::optional<int> numa_node = gpu::ReadNumaNode(pci_bus_id, device_ordinal);
  desc.set_numa_node(numa_node.has_value() ? std::max(0, *numa_node)
                                           : tsl::port::kNUMANoAffinity);

  ThreadDim thread_dim_limit;
  thread_dim_limit.x = compute_props.maxGroupSizeX;
  thread_dim_limit.y = compute_props.maxGroupSizeY;
  thread_dim_limit.z = compute_props.maxGroupSizeZ;
  desc.set_thread_dim_limit(thread_dim_limit);

  BlockDim block_dim_limit;
  block_dim_limit.x = compute_props.maxGroupCountX;
  block_dim_limit.y = compute_props.maxGroupCountY;
  block_dim_limit.z = compute_props.maxGroupCountZ;
  desc.set_block_dim_limit(block_dim_limit);

  desc.set_threads_per_warp(
      DetermineThreadsPerWarp(compute_props, device_props));
  desc.set_threads_per_core_limit(compute_props.maxTotalGroupSize);
  desc.set_threads_per_block_limit(compute_props.maxTotalGroupSize);

  desc.set_clock_rate_ghz(static_cast<float>(device_props.coreClockRate) /
                          1000.0f);
  int core_count = static_cast<int>(device_props.numSubslicesPerSlice) *
                   static_cast<int>(device_props.numSlices);
  desc.set_core_count(core_count);
  desc.set_fpus_per_core(DetermineFpusPerCore(device_props));

  uint64_t total_memory = 0;
  // Memory properties include DDR and HBM memories. Sum them up to get total
  // device memory. Ideally both types can be present on a device. However, in
  // reality, only one of these types is present on a device.
  for (const auto& mem : memory_props) {
    total_memory += mem.totalSize;
  }
  desc.set_device_memory_size(static_cast<int64_t>(total_memory));
  desc.set_memory_bandwidth(
      EstimateMemoryBandwidth(desc.oneapi_compute_capability()));
  int64_t l2_cache_size = 0;
  // Presumably, L2 cache is the last level cache on Intel GPUs. When there are
  // multiple cache properties, find the largest cache size to represent L2
  // cache size.
  for (const auto& cache : cache_props) {
    l2_cache_size =
        std::max<int64_t>(l2_cache_size, static_cast<int64_t>(cache.cacheSize));
  }
  desc.set_l2_cache_size(l2_cache_size);
  desc.set_shared_memory_per_block(compute_props.maxSharedLocalMemory);
  desc.set_shared_memory_per_block_optin(compute_props.maxSharedLocalMemory);
  desc.set_shared_memory_per_core(compute_props.maxSharedLocalMemory);

  desc.set_ecc_enabled((device_props.flags & ZE_DEVICE_PROPERTY_FLAG_ECC) != 0);

  desc.set_model_str(absl::StrFormat(
      "%s with %s RAM, %d XeCores, %.2fGHz clock, %s L2$", device_props.name,
      tsl::strings::HumanReadableNumBytes(desc.device_memory_size()),
      desc.core_count(), desc.clock_rate_ghz(),
      tsl::strings::HumanReadableNumBytes(desc.l2_cache_size())));

  return std::make_unique<DeviceDescription>(std::move(desc));
}

}  // namespace stream_executor::sycl
