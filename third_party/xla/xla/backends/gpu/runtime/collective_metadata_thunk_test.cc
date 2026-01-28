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

#include "xla/backends/gpu/runtime/collective_metadata_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/collective_kernel_metadata.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

using ::testing::Eq;

se::StreamExecutor* GetGpuExecutor(int64_t device_ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(device_ordinal).value();
}

TEST(CollectiveMetadataThunkTest,
     CreatesAndPopulatesMetadataForMultipleDevices) {
  constexpr int num_devices = 2;
  constexpr int num_parameters = 2;
  se::StreamExecutor* executors[num_devices] = {GetGpuExecutor(0),
                                                GetGpuExecutor(1)};
  std::unique_ptr<se::Stream> streams[num_devices] = {nullptr, nullptr};

  size_t metadata_size =
      sizeof(CollectiveKernelMetadata::rank) +
      sizeof(CollectiveKernelMetadata::param_to_peers) +
      sizeof(CollectiveKernelMetadata::multicast_buffer_ptr) +
      num_parameters * num_devices * sizeof(void*);
  se::DeviceAddressBase destinations[num_devices];
  for (int device_number = 0; device_number < num_devices; ++device_number) {
    TF_ASSERT_OK_AND_ASSIGN(streams[device_number],
                            executors[device_number]->CreateStream());
    destinations[device_number] =
        executors[device_number]->AllocateArray<uint64_t>(metadata_size);
  }

  xla::gpu::GpuCliqueKey clique_key({GlobalDeviceId(0), GlobalDeviceId(1)}, 2);
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "device_threads",
                                      num_devices);
  std::vector<tsl::Future<CollectiveKernelMetadata>> futures;

  for (int device_number = 0; device_number < num_devices; ++device_number) {
    futures.push_back(tsl::MakeFutureOn<CollectiveKernelMetadata>(
        *thread_pool.AsExecutor(),
        [device_number, &streams, &clique_key, &destinations] {
          RankId rank(device_number);
          std::vector<se::DeviceAddressBase> parameters{
              se::DeviceAddressBase(reinterpret_cast<void*>(device_number), 1),
              se::DeviceAddressBase(reinterpret_cast<void*>(device_number + 1),
                                    2)};
          absl::StatusOr<CollectiveKernelMetadata> metadata =
              CollectiveMetadataThunk::ConstructAndReturnCollectiveMetadata(
                  clique_key, rank, streams[device_number].get(),
                  std::move(parameters),
                  /*multimem=*/nullptr, destinations[device_number]);
          CHECK_OK(metadata);
          return metadata.value();
        }));
  }

  for (int device_number = 0; device_number < num_devices; ++device_number) {
    int other_device = (device_number + 1) % num_devices;
    TF_ASSERT_OK_AND_ASSIGN(CollectiveKernelMetadata metadata,
                            futures[device_number].Await());
    EXPECT_THAT(metadata.rank, Eq(device_number));
    EXPECT_THAT(metadata.param_to_peers_host.size(),
                Eq(num_devices * num_parameters));

    for (int parameter = 0; parameter < num_parameters; ++parameter) {
      EXPECT_THAT(
          metadata.param_to_peers_host[parameter * num_devices + device_number],
          Eq(reinterpret_cast<void*>(device_number + parameter)));
      EXPECT_THAT(
          metadata.param_to_peers_host[parameter * num_devices + other_device],
          Eq(reinterpret_cast<void*>(other_device + parameter)));
    }
  }
}
}  // namespace xla::gpu
