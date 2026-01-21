/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tests/collective_ops_ffi_kernels.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {

// Creates a pair of communicators for the given executors.
static absl::StatusOr<std::vector<std::unique_ptr<GpuCommunicator>>>
CreateCommunicators(se::StreamExecutor* executor0,
                    se::StreamExecutor* executor1) {
  GpuCollectives::Device device0(executor0);
  GpuCollectives::Device device1(executor1);

  GpuCollectives* collectives = GpuCollectives::Default("CUDA");

  TF_ASSIGN_OR_RETURN(CliqueId clique_id, collectives->CreateUniqueCliqueId());
  CliqueIds clique_ids(clique_id);

  GpuCliqueKey clique_key({GlobalDeviceId(0), GlobalDeviceId(1)},
                          /*num_local_participants=*/2);

  Collectives::DeviceRank rank0(&device0, RankId(0));
  Collectives::DeviceRank rank1(&device1, RankId(1));

  TF_ASSIGN_OR_RETURN(auto comms, collectives->CreateCommunicators(
                                      clique_key, clique_ids, {rank0, rank1},
                                      GpuCollectives::Config{}));
  CHECK_EQ(comms.size(), 2);

  std::vector<std::unique_ptr<GpuCommunicator>> gpu_comms;
  gpu_comms.emplace_back(dynamic_cast<GpuCommunicator*>(comms[0].release()));
  gpu_comms.emplace_back(dynamic_cast<GpuCommunicator*>(comms[1].release()));
  return gpu_comms;
}

TEST(CollectiveOpsFfiKernelsTest, CollectiveKernelLaunch) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("CUDA"));

  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
  }

  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor0,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor1,
                       platform->ExecutorForDevice(1));

  // We need peer access between devices to test LSA collective kernel.
  ASSERT_OK(executor0->EnablePeerAccessTo(executor1));
  ASSERT_OK(executor1->EnablePeerAccessTo(executor0));

  ASSERT_OK_AND_ASSIGN(auto comms, CreateCommunicators(executor0, executor1));

  if (!comms[0]->SupportsDeviceComm() || !comms[1]->SupportsDeviceComm()) {
    GTEST_SKIP() << "GPU communicators do not support device communicators";
  }

  size_t num_elements = 16;
  size_t num_bytes = 16 * sizeof(int32_t);

  // Create memory allocators that allocate physical memory in the collective
  // memory space, which makes them compatible with symmetric memory
  // requirements.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<se::MemoryAllocator> allocator0,
      executor0->CreateMemoryAllocator(se::MemorySpace::kCollective));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<se::MemoryAllocator> allocator1,
      executor1->CreateMemoryAllocator(se::MemorySpace::kCollective));

  // Allocate device memory on each participating rank.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::MemoryAllocation> alloc0,
                       allocator0->Allocate(num_bytes));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::MemoryAllocation> alloc1,
                       allocator1->Allocate(num_bytes));

  se::DeviceAddressBase addr0 = alloc0->address();
  se::DeviceAddressBase addr1 = alloc1->address();

  // Because creating symmetric memory is a collective operation, we must call
  // it from a thead pool to avoid deadlocks.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl", 2);
  tsl::Executor& exec = *pool.AsExecutor();

  // Register allocated buffers as symmetric memory.
  auto fsymm0 = MakeFutureOn(
      exec, [&] { return comms[0]->CreateSymmetricMemory(addr0); });
  auto fsymm1 = MakeFutureOn(
      exec, [&] { return comms[1]->CreateSymmetricMemory(addr1); });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SymmetricMemory> symm0,
                       std::move(fsymm0).Await());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<SymmetricMemory> symm1,
                       std::move(fsymm1).Await());

  // In this test we use kernels that use LSA for communication.
  GpuDeviceCommunicator::Requirements reqs;
  reqs.lsa_barrier_count = 1;

  // Create device communicators to be passed to kernels.
  auto fdev_comm0 =
      MakeFutureOn(exec, [&] { return comms[0]->CreateDeviceComm(reqs); });
  auto fdev_comm1 =
      MakeFutureOn(exec, [&] { return comms[1]->CreateDeviceComm(reqs); });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuDeviceCommunicator> dev_comm0,
                       std::move(fdev_comm0).Await());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuDeviceCommunicator> dev_comm1,
                       std::move(fdev_comm1).Await());

  // Initialize memory with some data before running collective kernel.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream0,
                       executor0->CreateStream());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream1,
                       executor1->CreateStream());
  ASSERT_OK(stream0->Memset32(&addr0, 1, num_bytes));
  ASSERT_OK(stream1->Memset32(&addr1, 2, num_bytes));

  // Load collective kernels on both executors.
  ASSERT_OK_AND_ASSIGN(auto kernel0,
                       se::gpu::GpuKernelRegistry::GetGlobalRegistry()
                           .LoadKernel<CollectiveInPlaceAllReduce>(executor0));
  ASSERT_OK_AND_ASSIGN(auto kernel1,
                       se::gpu::GpuKernelRegistry::GetGlobalRegistry()
                           .LoadKernel<CollectiveInPlaceAllReduce>(executor1));

  se::BlockDim block_dims(1);
  se::ThreadDim thread_dims(8);

  size_t offset = 0;

  ASSERT_OK(kernel0.Launch(thread_dims, block_dims, stream0.get(),
                           dev_comm0.get(), symm0.get(), offset, num_elements));
  ASSERT_OK(kernel1.Launch(thread_dims, block_dims, stream1.get(),
                           dev_comm1.get(), symm1.get(), offset, num_elements));

  // Copy data back to host and check it was all-reduced
  std::vector<int32_t> data0(num_elements, 0);
  std::vector<int32_t> data1(num_elements, 0);

  ASSERT_OK(stream0->Memcpy(data0.data(), addr0, num_bytes));
  ASSERT_OK(stream1->Memcpy(data1.data(), addr1, num_bytes));

  std::vector<int32_t> expected(num_elements, 3);
  EXPECT_EQ(data0, expected);
  EXPECT_EQ(data1, expected);
}

}  // namespace xla::gpu
