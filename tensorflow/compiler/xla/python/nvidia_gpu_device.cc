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

#include "tensorflow/compiler/xla/python/nvidia_gpu_device.h"

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace xla {

static const char kGpuPlatformName[] = "gpu";

GpuDevice::GpuDevice(int id,
                     std::unique_ptr<LocalDeviceState> local_device_state)
    : Device(id, std::move(local_device_state), kGpuPlatformName) {}

static StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateBFCAllocator(
    se::Platform* platform,
    absl::Span<const std::shared_ptr<Device>> local_devices,
    LocalClient* client, double memory_fraction, bool preallocate) {
  CHECK_GT(client->backend().device_count(), 0);
  std::vector<se::MultiDeviceAdapter::AllocatorWithStream> allocators;
  for (se::StreamExecutor* executor : client->backend().stream_executors()) {
    int device_ordinal = executor->device_ordinal();
    auto sub_allocator = absl::make_unique<tensorflow::GPUMemAllocator>(
        executor, tensorflow::PlatformGpuId(device_ordinal),
        /*use_unified_memory=*/false,
        /*alloc_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>(),
        /*free_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>());

    int64 free_memory;
    int64 total_memory;
    if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
      return Unavailable("Failed to query available memory from device %i",
                         device_ordinal);
    }
    size_t allocator_memory = free_memory * memory_fraction;
    if (preallocate) {
      LOG(INFO) << "XLA backend allocating " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for BFCAllocator.";
    } else {
      LOG(INFO) << "XLA backend will use up to " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for BFCAllocator.";
    }
    auto gpu_bfc_allocator = absl::make_unique<tensorflow::BFCAllocator>(
        sub_allocator.release(), allocator_memory,
        /*allow_growth=*/!preallocate,
        absl::StrCat("GPU_", device_ordinal, "_bfc"));
    allocators.emplace_back(std::move(gpu_bfc_allocator),
                            local_devices.at(device_ordinal)
                                ->local_device_state()
                                ->compute_stream());
  }
  return absl::make_unique<se::MultiDeviceAdapter>(platform,
                                                   std::move(allocators));
}

StatusOr<std::shared_ptr<PyLocalClient>> GetNvidiaGpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("CUDA"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("No visible NVidia GPU devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::shared_ptr<Device>> devices;
  for (int i = 0; i < client->device_count(); ++i) {
    se::StreamExecutor* executor =
        client->backend().stream_executor(i).ValueOrDie();
    auto device_state = absl::make_unique<LocalDeviceState>(
        executor, client, /*synchronous_deallocation=*/false, asynchronous,
        /*allow_event_reuse=*/true);
    std::shared_ptr<Device> device =
        std::make_shared<GpuDevice>(i, std::move(device_state));
    devices.push_back(std::move(device));
  }

  std::unique_ptr<se::DeviceMemoryAllocator> allocator;
  std::unique_ptr<tensorflow::Allocator> host_memory_allocator;
  if (allocator_config.kind != GpuAllocatorConfig::Kind::kPlatform) {
    TF_ASSIGN_OR_RETURN(allocator,
                        CreateBFCAllocator(platform, devices, client,
                                           allocator_config.memory_fraction,
                                           allocator_config.preallocate));
  }

  tensorflow::SubAllocator* sub_allocator = new tensorflow::GpuHostAllocator(
      client->backend().stream_executor(0).ValueOrDie(), /*numa_node=*/0,
      /*alloc_visitors=*/{},
      /*free_visitors=*/{});
  // TODO(phawkins): allow the user to tune this.
  const int64 kGpuHostMemoryLimitBytes = 64 * (1LL << 30);
  host_memory_allocator = absl::make_unique<tensorflow::BFCAllocator>(
      sub_allocator, kGpuHostMemoryLimitBytes, /*allow_growth=*/true,
      /*name=*/"xla_gpu_host_bfc");

  return std::make_shared<PyLocalClient>("gpu", client, std::move(devices),
                                         /*host_id=*/0, std::move(allocator),
                                         std::move(host_memory_allocator));
}

}  // namespace xla
