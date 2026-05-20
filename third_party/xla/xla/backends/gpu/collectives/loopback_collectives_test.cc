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

#include "xla/backends/gpu/collectives/loopback_collectives.h"

#include <cstddef>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;

static constexpr GlobalDeviceId kD0(0);
static constexpr GlobalDeviceId kD1(1);

static absl::StatusOr<std::vector<se::StreamExecutor*>> CreateExecutors(
    se::Platform* platform, size_t n) {
  std::vector<se::StreamExecutor*> executors(n);
  for (size_t d = 0; d < n; ++d) {
    ASSIGN_OR_RETURN(executors[d], platform->ExecutorForDevice(d));
  }
  return executors;
}

// Creates loopback communicators through the clique acquisition API.
static absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
CreateCommunicators(LoopbackCollectives* collectives,
                    absl::Span<se::StreamExecutor* const> executors,
                    std::vector<GlobalDeviceId> device_ids) {
  std::vector<GpuCollectives::Device> devices;
  devices.reserve(executors.size());
  for (se::StreamExecutor* executor : executors) {
    devices.emplace_back(executor);
  }

  std::vector<GpuCollectives::DeviceRank> device_ranks;
  device_ranks.reserve(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    device_ranks.emplace_back(&devices[i], RankId(i));
  }

  ASSIGN_OR_RETURN(CliqueId clique_id, collectives->CreateUniqueCliqueId());
  CliqueIds clique_ids;
  clique_ids.Add(clique_id);

  GpuCliqueKey clique_key(device_ids, executors.size());
  GpuCollectives::Config config;

  ASSIGN_OR_RETURN(auto comms,
                   collectives->CreateCommunicators(clique_key, clique_ids,
                                                    device_ranks, config));

  std::vector<std::unique_ptr<GpuCommunicator>> gpu_comms;
  gpu_comms.reserve(comms.size());
  for (auto& comm : comms) {
    gpu_comms.emplace_back(dynamic_cast<GpuCommunicator*>(comm.release()));
  }
  return gpu_comms;
}

// Creates loopback communicators by splitting existing ones.
static absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
SplitCommunicators(
    LoopbackCollectives* collectives,
    absl::Span<se::StreamExecutor* const> executors,
    absl::Span<const std::unique_ptr<GpuCommunicator>> existing_comms,
    std::vector<GlobalDeviceId> device_ids) {
  std::vector<GpuCollectives::Device> devices;
  devices.reserve(executors.size());
  for (se::StreamExecutor* executor : executors) {
    devices.emplace_back(executor);
  }

  std::vector<GpuCollectives::DeviceRank> device_ranks;
  device_ranks.reserve(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    device_ranks.emplace_back(&devices[i], RankId(i));
  }

  std::vector<RankId> keys(existing_comms.size());
  std::vector<const Communicator*> existing_comms_ptrs(existing_comms.size());
  for (size_t i = 0; i < existing_comms.size(); ++i) {
    keys[i] = RankId(i);
    existing_comms_ptrs[i] = existing_comms[i].get();
  }

  GpuCollectives::Config config;

  ASSIGN_OR_RETURN(auto comms, collectives->SplitCommunicators(
                                   existing_comms_ptrs, /*color=*/0, keys,
                                   config, device_ranks));

  std::vector<std::unique_ptr<GpuCommunicator>> gpu_comms;
  gpu_comms.reserve(comms.size());
  for (auto& comm : comms) {
    gpu_comms.emplace_back(dynamic_cast<GpuCommunicator*>(comm.release()));
  }
  return gpu_comms;
}

// Verifies that LoopbackCollectives fits into the clique acquisition API by
// creating communicators through CreateCommunicators.
TEST(LoopbackCollectivesTest, CreateCommunicators) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 1) {
    GTEST_SKIP() << "Test requires at least 1 GPU";
  }

  ASSERT_OK_AND_ASSIGN(auto executors, CreateExecutors(platform, 1));

  LoopbackCollectives collectives;
  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(&collectives, executors, {kD0}));

  ASSERT_EQ(comms.size(), 1);
  ASSERT_OK_AND_ASSIGN(size_t num_ranks, comms[0]->NumRanks());
  EXPECT_EQ(num_ranks, 1);
}

// Verifies that LoopbackCollectives can create multi-rank communicators.
TEST(LoopbackCollectivesTest, CreateMultiRankCommunicators) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(auto executors, CreateExecutors(platform, 2));

  LoopbackCollectives collectives;
  ASSERT_OK_AND_ASSIGN(
      auto comms, CreateCommunicators(&collectives, executors, {kD0, kD1}));

  ASSERT_EQ(comms.size(), 2);
  ASSERT_OK_AND_ASSIGN(size_t num_ranks0, comms[0]->NumRanks());
  ASSERT_OK_AND_ASSIGN(size_t num_ranks1, comms[1]->NumRanks());
  EXPECT_EQ(num_ranks0, 2);
  EXPECT_EQ(num_ranks1, 2);
}

// Verifies that LoopbackCollectives supports SplitCommunicators.
TEST(LoopbackCollectivesTest, SplitCommunicators) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(auto executors, CreateExecutors(platform, 2));

  LoopbackCollectives collectives;
  ASSERT_OK_AND_ASSIGN(
      auto comms, CreateCommunicators(&collectives, executors, {kD0, kD1}));

  ASSERT_OK_AND_ASSIGN(
      auto split_comms,
      SplitCommunicators(&collectives, executors, comms, {kD0, kD1}));
  ASSERT_EQ(split_comms.size(), 2);
  ASSERT_OK_AND_ASSIGN(size_t num_ranks, split_comms[0]->NumRanks());
  EXPECT_EQ(num_ranks, 2);
}

// Verifies that AllReduce copies send to recv on a single device.
TEST(LoopbackCollectivesTest, AllReduce) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 1) {
    GTEST_SKIP() << "Test requires at least 1 GPU";
  }

  ASSERT_OK_AND_ASSIGN(auto executors, CreateExecutors(platform, 1));

  LoopbackCollectives collectives;
  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(&collectives, executors, {kD0}));

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executors[0]->CreateStream());

  constexpr size_t kCount = 4;
  se::DeviceAddress<float> send = executors[0]->AllocateArray<float>(kCount);
  se::DeviceAddress<float> recv = executors[0]->AllocateArray<float>(kCount);

  // Initialize send buffer with test data.
  float host_send[] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_THAT(stream->Memcpy(&send, host_send, sizeof(host_send)), IsOk());

  // Run AllReduce.
  GpuCollectives::Executor executor(stream.get());
  auto future = comms[0]->AllReduce(send, recv, F32, kCount, ReductionKind::SUM,
                                    executor);
  ASSERT_THAT(future.Await(), IsOk());

  // Verify output = input.
  float host_recv[kCount];
  ASSERT_THAT(stream->Memcpy(host_recv, recv, sizeof(host_recv)), IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(host_recv[i], host_send[i]);
  }

  executors[0]->Deallocate(&send);
  executors[0]->Deallocate(&recv);
}

// Verifies that AllGather replicates input into every rank's slot.
TEST(LoopbackCollectivesTest, AllGather) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 1) {
    GTEST_SKIP() << "Test requires at least 1 GPU";
  }

  ASSERT_OK_AND_ASSIGN(auto executors, CreateExecutors(platform, 1));

  LoopbackCollectives collectives;
  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(&collectives, executors, {kD0}));

  // Simulate 2-rank communicator to verify replication.
  // Create communicator directly with 2 ranks, rank 0.
  ASSERT_OK_AND_ASSIGN(auto comms2,
                       CreateCommunicators(&collectives, executors, {kD0}));

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executors[0]->CreateStream());

  constexpr size_t kCount = 2;
  constexpr size_t kNumRanks = 1;
  se::DeviceAddress<float> send = executors[0]->AllocateArray<float>(kCount);
  se::DeviceAddress<float> recv =
      executors[0]->AllocateArray<float>(kCount * kNumRanks);

  float host_send[] = {5.0f, 6.0f};
  ASSERT_THAT(stream->Memcpy(&send, host_send, sizeof(host_send)), IsOk());

  GpuCollectives::Executor executor(stream.get());
  auto future = comms2[0]->AllGather(send, recv, F32, kCount, executor);
  ASSERT_THAT(future.Await(), IsOk());

  float host_recv[kCount * kNumRanks];
  ASSERT_THAT(stream->Memcpy(host_recv, recv, sizeof(host_recv)), IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());

  // With 1 rank, output should just be the input replicated once.
  EXPECT_EQ(host_recv[0], 5.0f);
  EXPECT_EQ(host_recv[1], 6.0f);

  executors[0]->Deallocate(&send);
  executors[0]->Deallocate(&recv);
}

// Verifies that CollectivePermute copies send to recv (loopback).
TEST(LoopbackCollectivesTest, CollectivePermute) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 1) {
    GTEST_SKIP() << "Test requires at least 1 GPU";
  }

  ASSERT_OK_AND_ASSIGN(auto executors, CreateExecutors(platform, 1));

  LoopbackCollectives collectives;
  ASSERT_OK_AND_ASSIGN(auto comms,
                       CreateCommunicators(&collectives, executors, {kD0}));

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executors[0]->CreateStream());

  constexpr size_t kCount = 4;
  se::DeviceAddress<float> send = executors[0]->AllocateArray<float>(kCount);
  se::DeviceAddress<float> recv = executors[0]->AllocateArray<float>(kCount);

  float host_send[] = {10.0f, 20.0f, 30.0f, 40.0f};
  ASSERT_THAT(stream->Memcpy(&send, host_send, sizeof(host_send)), IsOk());

  GpuCollectives::Executor executor(stream.get());
  auto future = comms[0]->CollectivePermute(send, recv, F32, kCount, RankId(0),
                                            {RankId(0)}, executor);
  ASSERT_THAT(future.Await(), IsOk());

  float host_recv[kCount];
  ASSERT_THAT(stream->Memcpy(host_recv, recv, sizeof(host_recv)), IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());

  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(host_recv[i], host_send[i]);
  }

  executors[0]->Deallocate(&send);
  executors[0]->Deallocate(&recv);
}

}  // namespace
}  // namespace xla::gpu
