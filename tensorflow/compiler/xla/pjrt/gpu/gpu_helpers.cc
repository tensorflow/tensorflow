/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/gpu/gpu_helpers.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_host_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/device_mem_allocator.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/framework/device_id.h"
#include "tensorflow/tsl/util/env_var.h"

namespace xla {

// Builds an xla::LocalClient for the GPU platform.
StatusOr<LocalClient*> GetGpuXlaClient(
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
        Status status = from->EnablePeerAccessTo(to);
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
StatusOr<std::unique_ptr<tsl::BFCAllocator>> CreateBFCAllocator(
    se::StreamExecutor* executor, double memory_fraction, bool preallocate) {
  bool enable_unified_memory;
  Status status = tsl::ReadBoolFromEnvVar("TF_FORCE_UNIFIED_MEMORY", false,
                                          &enable_unified_memory);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to read TF_FORCE_UNIFIED_MEMORY: "
               << status.error_message();
  }

  int device_ordinal = executor->device_ordinal();
  auto sub_allocator = std::make_unique<se::DeviceMemAllocator>(
      executor, tsl::PlatformDeviceId(device_ordinal),
      /*use_unified_memory=*/enable_unified_memory,
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

// Returns a GPU pinned host memory allocator to use when staging host->GPU
// transfers. We use a fixed 64MB pool of pinned memory.
std::unique_ptr<tsl::BFCAllocator> GetGpuHostAllocator(
    se::StreamExecutor* executor) {
  std::unique_ptr<tsl::SubAllocator> sub_allocator(
      new se::DeviceHostAllocator(executor, /*numa_node=*/0,
                                  /*alloc_visitors=*/{},
                                  /*free_visitors=*/{}));
  // TODO(phawkins): allow the user to tune this.
  const int64_t kGpuHostMemoryLimitBytes = 64 * (1LL << 30);

  tsl::BFCAllocator::Options opts;
  opts.allow_growth = true;
  return std::make_unique<tsl::BFCAllocator>(std::move(sub_allocator),
                                             kGpuHostMemoryLimitBytes,
                                             /*name=*/"xla_gpu_host_bfc", opts);
}

}  // namespace xla
