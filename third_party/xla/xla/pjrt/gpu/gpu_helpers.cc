/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/gpu_helpers.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "xla/client/client_library.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/integrations/device_host_allocator.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"

namespace xla {

// Builds an xla::LocalClient for the GPU platform.
absl::StatusOr<LocalClient*> GetGpuXlaClient(
    const std::optional<std::string>& platform_name,
    const std::optional<std::set<int>>& allowed_devices) {
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      PlatformUtil::GetPlatform(platform_name ? *platform_name : "gpu"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("No visible GPU devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  options.set_allowed_devices(allowed_devices);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

void EnablePeerAccess(absl::Span<se::StreamExecutor* const> executors) {
  for (int i = 0; i < executors.size(); ++i) {
    for (int j = 0; j < executors.size(); ++j) {
      if (i == j) {
        continue;
      }
      se::StreamExecutor* from = executors[i];
      se::StreamExecutor* to = executors[j];
      if (from->CanEnablePeerAccessTo(to)) {
        absl::Status status = from->EnablePeerAccessTo(to);
        if (!status.ok()) {
          LOG(WARNING) << "Unable to enable peer access between GPUs " << i
                       << " and " << j << "; status: " << status;
        } else {
          VLOG(2) << "Enabled peer access from GPU " << i << " to GPU " << j;
        }
      }
    }
  }
}

// Builds a BFCAllocator for all local GPUs.
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> CreateBFCAllocator(
    se::StreamExecutor* executor, double memory_fraction, bool preallocate,
    std::optional<int64_t> gpu_system_memory_size) {
  bool enable_unified_memory;
  absl::Status status = tsl::ReadBoolFromEnvVar("TF_FORCE_UNIFIED_MEMORY",
                                                false, &enable_unified_memory);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to read TF_FORCE_UNIFIED_MEMORY: "
               << status.message();
  }

  int device_ordinal = executor->device_ordinal();
  auto sub_allocator = std::make_unique<se::DeviceMemAllocator>(
      executor, tsl::PlatformDeviceId(device_ordinal),
      /*memory_type=*/
      enable_unified_memory ? stream_executor::MemoryType::kUnified
                            : stream_executor::MemoryType::kDevice,
      /*alloc_visitors=*/std::vector<tsl::SubAllocator::Visitor>(),
      /*free_visitors=*/std::vector<tsl::SubAllocator::Visitor>());

  int64_t free_memory;
  int64_t total_memory;
  if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
    return Unavailable("Failed to query available memory from device %i",
                       device_ordinal);
  }
  // To allow full GPU memory to be visible to the BFC allocator if using
  // unified memory.
  // When unified memory is enabled, allow GPU memory oversubscription by
  // setting memory_fraction > 1.
  size_t allocator_memory = enable_unified_memory
                                ? total_memory * fmax(1.0, memory_fraction)
                                : total_memory * memory_fraction;
  // If gpu_system_memory_size is set, use it instead of default value.
  if (gpu_system_memory_size.has_value()) {
    allocator_memory = gpu_system_memory_size.value();
  }

  if (preallocate) {
    LOG(INFO) << "XLA backend allocating " << allocator_memory
              << " bytes on device " << device_ordinal << " for BFCAllocator.";
  } else {
    LOG(INFO) << "XLA backend will use up to " << allocator_memory
              << " bytes on device " << device_ordinal << " for BFCAllocator.";
  }

  tsl::BFCAllocator::Options opts;
  opts.allow_growth = !preallocate;
  return std::make_unique<tsl::BFCAllocator>(
      std::move(sub_allocator), allocator_memory,
      absl::StrCat("GPU_", device_ordinal, "_bfc"), opts);
}

// Builds a BFCAllocator for all local GPUs that uses collective memory.
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> CreateCollectiveBFCAllocator(
    se::StreamExecutor* executor, double memory_fraction,
    size_t collective_memory_size) {
  int device_ordinal = executor->device_ordinal();
  auto sub_allocator = std::make_unique<se::DeviceMemAllocator>(
      executor, tsl::PlatformDeviceId(device_ordinal),
      /*memory_type=*/stream_executor::MemoryType::kCollective,
      /*alloc_visitors=*/std::vector<tsl::SubAllocator::Visitor>(),
      /*free_visitors=*/std::vector<tsl::SubAllocator::Visitor>());

  int64_t free_memory;
  int64_t total_memory;
  if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
    return Unavailable("Failed to query available memory from device %i",
                       device_ordinal);
  }
  bool preallocate = collective_memory_size != 0;
  size_t allocator_memory =
      preallocate ? collective_memory_size : total_memory * memory_fraction;

  if (preallocate) {
    LOG(INFO) << "XLA backend allocating " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CollectiveBFCAllocator.";
  } else {
    LOG(INFO) << "XLA backend will use up to " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CollectiveBFCAllocator.";
  }

  tsl::BFCAllocator::Options opts;
  opts.allow_growth = !preallocate;
  return std::make_unique<tsl::BFCAllocator>(
      std::move(sub_allocator), allocator_memory,
      absl::StrCat("GPU_collectivememory_", device_ordinal, "_bfc"), opts);
}

// Returns a GPU pinned host memory allocator to use when staging host->GPU
// transfers. We use a fixed 64GB pool of pinned memory.
std::unique_ptr<tsl::BFCAllocator> GetGpuHostAllocator(
    se::StreamExecutor* executor) {
  std::unique_ptr<tsl::SubAllocator> sub_allocator(
      new se::DeviceHostAllocator(executor, /*numa_node=*/0,
                                  /*alloc_visitors=*/{},
                                  /*free_visitors=*/{}));

  int64_t xla_pjrt_gpu_host_memory_limit_gb;
  absl::Status status =
      tsl::ReadInt64FromEnvVar("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB", 64,
                               &xla_pjrt_gpu_host_memory_limit_gb);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to read XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB: "
               << status.message();
  }

  const int64_t kGpuHostMemoryLimitBytes =
      xla_pjrt_gpu_host_memory_limit_gb * (1LL << 30);

  tsl::BFCAllocator::Options opts;
  opts.allow_growth = true;
  return std::make_unique<tsl::BFCAllocator>(std::move(sub_allocator),
                                             kGpuHostMemoryLimitBytes,
                                             /*name=*/"xla_gpu_host_bfc", opts);
}

}  // namespace xla
