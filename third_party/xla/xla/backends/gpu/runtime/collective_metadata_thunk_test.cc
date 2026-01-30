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
#include <cstring>
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
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::Eq;

se::StreamExecutor* GetGpuExecutor(int64_t device_ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(device_ordinal).value();
}

constexpr int kNumDevices = 2;
constexpr int kNumParameters = 2;

struct RankAndParamToPeers {
  uint64_t rank;
  std::vector<void*> param_to_peers;

  RankAndParamToPeers(uint64_t rank, std::vector<void*> param_to_peers)
      : rank(rank), param_to_peers(std::move(param_to_peers)) {}
  RankAndParamToPeers() = default;
};

absl::StatusOr<std::vector<tsl::Future<RankAndParamToPeers>>>
CreateMetadataConstructionFutures() {
  size_t metadata_size =
      sizeof(CollectiveKernelMetadata::rank) +
      sizeof(CollectiveKernelMetadata::param_to_peers) +
      sizeof(CollectiveKernelMetadata::multicast_buffer_ptr) +
      kNumParameters * kNumDevices * sizeof(void*);
  se::StreamExecutor* executors[kNumDevices] = {GetGpuExecutor(0),
                                                GetGpuExecutor(1)};
  std::unique_ptr<se::Stream> streams[kNumDevices] = {nullptr, nullptr};
  se::DeviceAddressBase destinations[kNumDevices];
  for (int device_number = 0; device_number < kNumDevices; ++device_number) {
    TF_ASSIGN_OR_RETURN(streams[device_number],
                        executors[device_number]->CreateStream());
    destinations[device_number] =
        executors[device_number]->AllocateArray<uint64_t>(metadata_size);
  }

  xla::gpu::GpuCliqueKey clique_key({GlobalDeviceId(0), GlobalDeviceId(1)}, 2);
  // Simulate multiple devices by running the metadata construction in parallel
  // threads. Each thread acts as a different rank within the clique.
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "device_threads",
                                      kNumDevices);
  std::vector<tsl::Future<RankAndParamToPeers>> futures;

  for (int device_number = 0; device_number < kNumDevices; ++device_number) {
    futures.push_back(tsl::MakeFutureOn<RankAndParamToPeers>(
        *thread_pool.AsExecutor(),
        [device_number, &streams, &clique_key,
         &destinations]() -> absl::StatusOr<RankAndParamToPeers> {
          RankId rank(device_number);
          std::vector<se::DeviceAddressBase> parameters{
              se::DeviceAddressBase(reinterpret_cast<void*>(device_number), 1),
              se::DeviceAddressBase(reinterpret_cast<void*>(device_number + 1),
                                    2)};

          TF_ASSIGN_OR_RETURN(
              std::vector<void*> param_to_peers,
              CollectiveMetadataThunk::CollectParamToPeers(
                  clique_key, rank, streams[device_number].get(), parameters));
          TF_ASSIGN_OR_RETURN(
              CollectiveKernelMetadata metadata,
              CollectiveMetadataThunk::CreateCollectiveMetadata(
                  clique_key, rank, streams[device_number].get(),
                  /*multimem=*/nullptr));

          TF_RETURN_IF_ERROR(
              CollectiveMetadataThunk::CopyCollectiveMetadataToDevice(
                  streams[device_number].get(), metadata, param_to_peers,
                  destinations[device_number]));
          RankAndParamToPeers result;
          result.rank = metadata.rank;
          result.param_to_peers = std::move(param_to_peers);
          return result;
        }));
  }
  return futures;
}

TEST(CollectiveMetadataThunkTest,
     CreatesAndPopulatesMetadataForMultipleDevices) {
  TF_ASSERT_OK_AND_ASSIGN(std::vector<tsl::Future<RankAndParamToPeers>> futures,
                          CreateMetadataConstructionFutures());
  for (int device_number = 0; device_number < kNumDevices; ++device_number) {
    int other_device = (device_number + 1) % kNumDevices;
    TF_ASSERT_OK_AND_ASSIGN(RankAndParamToPeers rank_and_param_to_peers,
                            futures[device_number].Await());
    EXPECT_THAT(rank_and_param_to_peers.rank, Eq(device_number))
        << "rank is not equal to device_number: " << device_number;
    EXPECT_THAT(rank_and_param_to_peers.param_to_peers.size(),
                Eq(kNumParameters * kNumDevices))
        << "param_to_peers size is incorrect for device_number: "
        << device_number;

    for (int parameter = 0; parameter < kNumParameters; ++parameter) {
      EXPECT_THAT(rank_and_param_to_peers
                      .param_to_peers[parameter * kNumDevices + device_number],
                  Eq(reinterpret_cast<void*>(device_number + parameter)))
          << "pointer to parameter " << parameter
          << " is incorrect for device_number: " << device_number;
      EXPECT_THAT(rank_and_param_to_peers
                      .param_to_peers[parameter * kNumDevices + other_device],
                  Eq(reinterpret_cast<void*>(other_device + parameter)))
          << "pointer to parameter " << parameter
          << " is incorrect for peer of device_number: " << device_number;
    }
  }
}
}  // namespace
}  // namespace xla::gpu
