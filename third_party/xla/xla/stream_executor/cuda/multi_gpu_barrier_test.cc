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

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/platform/status_macros.h"

// Include NCCL after XLA headers.
#include "third_party/nccl/nccl.h"

namespace stream_executor::gpu {
namespace {

class MultiGpuBarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(platform_,
                         stream_executor::PlatformManager::PlatformWithId(
                             stream_executor::cuda::kCudaPlatformId));
    int visible_device_count = platform_->VisibleDeviceCount();

    if (visible_device_count < 2) {
      GTEST_SKIP() << "Test requires at least 2 GPUs, found "
                   << visible_device_count;
    }

    // Limit to MultiGpuBarrierKernel::kMaxPeers (32)
    num_devices_ =
        std::min<int>(visible_device_count, MultiGpuBarrierKernel::kMaxPeers);

    executors_.reserve(num_devices_);
    streams_.reserve(num_devices_);

    for (int i = 0; i < num_devices_; ++i) {
      ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                           platform_->ExecutorForDevice(i));
      executors_.push_back(executor);

      ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
      streams_.push_back(std::move(stream));
    }

    // Enable P2P access.
    for (int i = 0; i < num_devices_; ++i) {
      for (int j = 0; j < num_devices_; ++j) {
        if (i == j) {
          continue;
        }
        // If P2P is not supported, we must SKIP the test.
        // Proceeding without P2P will cause an illegal address error in the
        // kernel.
        if (!executors_[i]->CanEnablePeerAccessTo(executors_[j])) {
          GTEST_SKIP()
              << "Test requires direct peer memory access between devices.";
        }

        ASSERT_OK(executors_[i]->EnablePeerAccessTo(executors_[j]));
      }
    }
  }

  Platform* platform_;
  int num_devices_ = 0;
  std::vector<StreamExecutor*> executors_;
  std::vector<std::unique_ptr<Stream>> streams_;
};

absl::StatusOr<std::vector<std::unique_ptr<MemoryAllocation>>> AllocateBuffers(
    const std::vector<std::unique_ptr<MemoryAllocator>>& allocators,
    int64_t num_devices, int64_t buffer_size) {
  std::vector<std::unique_ptr<MemoryAllocation>> buffers;
  buffers.reserve(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    ASSIGN_OR_RETURN(std::unique_ptr<MemoryAllocation> buffer,
                     allocators[i]->Allocate(buffer_size));
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::SymmetricMemory>>>
CreateSymmetricMemory(
    tsl::Executor& exec, const std::vector<ncclComm_t>& comms,
    const std::vector<std::unique_ptr<MemoryAllocation>>& buffers) {
  int64_t num_devices = comms.size();
  std::vector<tsl::Future<std::unique_ptr<xla::gpu::NcclSymmetricMemory>>>
      symmetric_memory_futures(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    symmetric_memory_futures[i] = tsl::MakeFutureOn(exec, [&, i]() {
      return xla::gpu::NcclSymmetricMemory::Create(comms[i],
                                                   buffers[i]->address());
    });
  }

  std::vector<std::unique_ptr<xla::SymmetricMemory>> symmetric_memory;
  for (int i = 0; i < num_devices; ++i) {
    ASSIGN_OR_RETURN(auto mem, std::move(symmetric_memory_futures[i]).Await());
    symmetric_memory.push_back(std::move(mem));
  }
  return symmetric_memory;
}

template <typename T>
absl::StatusOr<std::vector<T>> CopyToHost(Stream* stream,
                                          DeviceAddressBase device_address,
                                          int64_t num_elements) {
  std::vector<T> host_buffer(num_elements);
  RETURN_IF_ERROR(stream->Memcpy(host_buffer.data(), device_address,
                                 num_elements * sizeof(T)));
  RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return host_buffer;
}

TEST_F(MultiGpuBarrierTest, BarrierSynchronization) {
  // 1. Allocate Signal Buffers on each device
  std::vector<DeviceAddress<uint32_t>> signal_buffers;

  for (int i = 0; i < num_devices_; ++i) {
    auto alloc = executors_[i]->AllocateArray<uint32_t>(num_devices_);
    ASSERT_OK(streams_[i]->MemZero(&alloc, num_devices_ * sizeof(uint32_t)));
    signal_buffers.push_back(alloc);
  }

  // 2. Allocate Counters on each device
  std::vector<DeviceAddress<uint32_t>> counters;
  std::vector<void*> counter_ptrs;
  for (int i = 0; i < num_devices_; ++i) {
    auto c = executors_[i]->AllocateArray<uint32_t>(1);
    // It is ok to initialize the counter to 0,
    // as the kernel pre-increments signal_value.
    ASSERT_OK(streams_[i]->MemZero(&c, sizeof(uint32_t)));
    counters.push_back(c);
    counter_ptrs.push_back(c.opaque());
  }

  // 3. Prepare Kernel Arguments
  std::array<void*, MultiGpuBarrierKernel::kMaxPeers> kernel_arg_ptrs;
  for (int i = 0; i < num_devices_; ++i) {
    kernel_arg_ptrs[i] = signal_buffers[i].opaque();
  }

  // 4. Launch Kernel REPEATEDLY 8 times to verify auto-increment
  for (int step = 0; step < 8; ++step) {
    for (int i = 0; i < num_devices_; ++i) {
      ASSERT_OK_AND_ASSIGN(
          auto kernel, (GpuKernelRegistry::GetGlobalRegistry()
                            .LoadKernel<MultiGpuBarrierKernel>(executors_[i])));

      ASSERT_OK(kernel.Launch(ThreadDim(num_devices_, 1, 1), BlockDim(1, 1, 1),
                              streams_[i].get(), static_cast<int64_t>(i),
                              static_cast<int64_t>(num_devices_),
                              kernel_arg_ptrs, counters[i]));
    }
  }

  for (int i = 0; i < num_devices_; ++i) {
    ASSERT_OK(streams_[i]->BlockHostUntilDone());
  }

  // 5. Verify Counters
  // After 8 runs, counters should be 8.
  for (int i = 0; i < num_devices_; ++i) {
    uint32_t val;
    ASSERT_OK(streams_[i]->Memcpy(&val, counters[i], sizeof(uint32_t)));
    ASSERT_OK(streams_[i]->BlockHostUntilDone());
    EXPECT_EQ(val, 8) << "Counter on device " << i << " failed to increment";
  }

  // 6. Verify Signal Buffers
  // Each device's signal buffer is an array of size 'num_devices'.
  // By the end of step 8, every device should have written the value '8'
  // into its designated slot on every peer's buffer.
  for (int i = 0; i < num_devices_; ++i) {
    // Copy the device's signal buffer back to the host
    ASSERT_OK_AND_ASSIGN(std::vector<uint32_t> host_buffer,
                         CopyToHost<uint32_t>(streams_[i].get(),
                                              signal_buffers[i], num_devices_));

    // Verify every slot contains the final step value (8)
    for (int j = 0; j < num_devices_; ++j) {
      EXPECT_EQ(host_buffer[j], 8)
          << "Signal buffer on Device " << i << " at slot " << j
          << " (which belongs to Peer " << j << ")"
          << " has incorrect value.";
    }
  }
}

TEST_F(MultiGpuBarrierTest, BarrierSynchronizationWithNccl) {
#if NCCL_VERSION_CODE < 22800
  GTEST_SKIP() << "Test requires NCCL 2.28.0 or later.";
#endif  // NCCL_VERSION_CODE < 22800

  std::vector<std::unique_ptr<MemoryAllocator>> collective_allocators;
  for (int i = 0; i < num_devices_; ++i) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<MemoryAllocator> allocator,
        executors_[i]->CreateMemoryAllocator(MemorySpace::kCollective));
    collective_allocators.push_back(std::move(allocator));
  }

  // 1. Allocate Signal Buffers on each device
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<MemoryAllocation>> signal_buffers,
      AllocateBuffers(collective_allocators, num_devices_,
                      sizeof(uint32_t) * num_devices_));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<MemoryAllocation>> ptr_storage,
      AllocateBuffers(collective_allocators, num_devices_,
                      num_devices_ * sizeof(void*)));

  std::vector<ncclComm_t> comms(num_devices_);
  ncclResult_t result =
      ncclCommInitAll(comms.data(), num_devices_, /*devlist=*/nullptr);
  ASSERT_EQ(result, ncclSuccess);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl", num_devices_);
  tsl::Executor& exec = *pool.AsExecutor();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<xla::SymmetricMemory>>
                           signal_buffer_symmetric_memory,
                       CreateSymmetricMemory(exec, comms, signal_buffers));

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<xla::SymmetricMemory>>
                           ptr_storage_symmetric_memory,
                       CreateSymmetricMemory(exec, comms, ptr_storage));

  // 2. Allocate Counters on each device
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<MemoryAllocation>> counters,
      AllocateBuffers(collective_allocators, num_devices_, sizeof(uint32_t)));

  // 4. Launch Kernel REPEATEDLY 8 times to verify auto-increment
  for (int step = 0; step < 8; ++step) {
    for (int i = 0; i < num_devices_; ++i) {
      ASSERT_OK_AND_ASSIGN(
          auto kernel,
          (GpuKernelRegistry::GetGlobalRegistry()
               .LoadKernel<MultiGpuBarrierWithNcclKernel>(executors_[i])));

      ASSERT_OK(kernel.Launch(
          ThreadDim(num_devices_, 1, 1), BlockDim(1, 1, 1), streams_[i].get(),
          static_cast<int64_t>(i), static_cast<int64_t>(num_devices_),
          signal_buffer_symmetric_memory[i].get(),
          DeviceAddress<uint32_t>(counters[i]->address()),
          counters[i]->address(), ptr_storage_symmetric_memory[i].get()));
    }
  }

  for (int i = 0; i < num_devices_; ++i) {
    ASSERT_OK(streams_[i]->BlockHostUntilDone());
  }

  // 5. Verify Counters
  // After 8 runs, counters should be 8.
  for (int i = 0; i < num_devices_; ++i) {
    uint32_t val;
    ASSERT_OK(
        streams_[i]->Memcpy(&val, counters[i]->address(), sizeof(uint32_t)));
    ASSERT_OK(streams_[i]->BlockHostUntilDone());
    EXPECT_EQ(val, 8) << "Counter on device " << i << " failed to increment";
  }

  // 6. Verify Signal Buffers
  // Each device's signal buffer is an array of size 'num_devices'.
  // By the end of step 8, every device should have written the value '8'
  // into its designated slot on every peer's buffer.
  for (int i = 0; i < num_devices_; ++i) {
    // Copy the device's signal buffer back to the host
    ASSERT_OK_AND_ASSIGN(
        std::vector<uint32_t> host_buffer,
        CopyToHost<uint32_t>(streams_[i].get(), signal_buffers[i]->address(),
                             num_devices_));

    // Verify every slot contains the final step value (8)
    for (int j = 0; j < num_devices_; ++j) {
      EXPECT_EQ(host_buffer[j], 8)
          << "Signal buffer on Device " << i << " at slot " << j
          << " (which belongs to Peer " << j << ")"
          << " has incorrect value.";
    }
  }

  // 7. Verify Pointer Storage
  // Each device's pointer storage is an array of size 'num_devices'.
  // After one execution, counter pointer should have been written to pointer
  // storage symmetric memory.
  for (int i = 0; i < num_devices_; ++i) {
    // Copy the device's signal buffer back to the host
    ASSERT_OK_AND_ASSIGN(
        std::vector<void*> host_buffer,
        CopyToHost<void*>(streams_[i].get(), ptr_storage[i]->address(),
                          num_devices_));

    for (int j = 0; j < num_devices_; ++j) {
      EXPECT_EQ(host_buffer[j], counters[j]->address().opaque())
          << "Pointer storage on Device " << i << " at slot " << j
          << " (which belongs to Peer " << j << ")"
          << " has incorrect value.";
    }
  }
}

}  // namespace
}  // namespace stream_executor::gpu
