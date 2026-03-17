/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_device_address_vmm_allocator.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/vmm_device_address_allocator.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::Ne;

class DeviceAddressVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("CUDA");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "CUDA platform not available";
    }
    platform_ = platform_or.value();

    auto executor_or = platform_->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "CUDA executor not available";
    }
    executor_ = executor_or.value();

    auto stream_or = executor_->CreateStream();
    if (!stream_or.ok()) {
      GTEST_SKIP() << "Failed to create stream";
    }
    stream_ = std::move(stream_or.value());

    // Probe for cuStreamWriteValue64 support (requires CC >= 7.0).
    auto probe =
        gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get());
    if (absl::IsUnimplemented(probe.status())) {
      GTEST_SKIP() << "Device does not support cuStreamWriteValue64 "
                      "(requires compute capability >= 7.0): "
                   << probe.status();
    }
  }

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
};

TEST_F(DeviceAddressVmmAllocatorTest, AllocateAndDeallocate) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kP2P)));

  EXPECT_FALSE(scoped_address.is_null());
  EXPECT_EQ(scoped_address->size(), 1024);
  EXPECT_NE(
      allocator->GetRawAllocation(executor_->device_ordinal(), *scoped_address),
      nullptr);
  EXPECT_NE(
      allocator->GetReservation(executor_->device_ordinal(), *scoped_address),
      nullptr);

  // The ScopedDeviceAddress will automatically deallocate when it goes out of
  // scope.
}

TEST_F(DeviceAddressVmmAllocatorTest, AllocateZeroSize) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate zero-size memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 0,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kP2P)));

  // Zero-size allocation should return a null address.
  EXPECT_TRUE(scoped_address.is_null());
}

TEST_F(DeviceAddressVmmAllocatorTest, AllocateMultiple) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate multiple memory regions.
  TF_ASSERT_OK_AND_ASSIGN(
      auto addr1, allocator->Allocate(executor_->device_ordinal(), 1024,
                                      /*retry_on_failure=*/true,
                                      static_cast<int64_t>(MemorySpace::kP2P)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr2, allocator->Allocate(executor_->device_ordinal(), 2048,
                                      /*retry_on_failure=*/true,
                                      static_cast<int64_t>(MemorySpace::kP2P)));

  // Both allocations should be valid and distinct.
  EXPECT_FALSE(addr1.is_null());
  EXPECT_FALSE(addr2.is_null());
  EXPECT_NE(addr1->opaque(), addr2->opaque());
  EXPECT_EQ(addr1.cref().size(), 1024);
  EXPECT_EQ(addr2.cref().size(), 2048);
}

TEST_F(DeviceAddressVmmAllocatorTest, MemoryReadWrite) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kP2P)));

  ASSERT_NE(scoped_address->opaque(), nullptr);

  // Create a stream for memory operations.
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());

  // Write data to the allocated memory.
  constexpr uint64_t kTestValue = 0xDEADBEEFCAFEBABE;
  DeviceAddressBase addr = scoped_address.cref();
  EXPECT_THAT(stream->Memcpy(&addr, &kTestValue, sizeof(kTestValue)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  // Read data back.
  uint64_t read_value = 0;
  EXPECT_THAT(stream->Memcpy(&read_value, addr, sizeof(read_value)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  EXPECT_EQ(read_value, kTestValue);
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStream) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Get the stream - should return the same stream that was provided at
  // construction.
  TF_ASSERT_OK_AND_ASSIGN(Stream * stream,
                          allocator->GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream_.get());

  // Getting the stream again should return the same pointer.
  TF_ASSERT_OK_AND_ASSIGN(Stream * stream2,
                          allocator->GetStream(executor_->device_ordinal()));
  EXPECT_EQ(stream, stream2);
}

TEST_F(DeviceAddressVmmAllocatorTest, GetStreamExecutor) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      StreamExecutor * se,
      allocator->GetStreamExecutor(executor_->device_ordinal()));
  EXPECT_EQ(se, executor_);
}

TEST_F(DeviceAddressVmmAllocatorTest, AllowsAsynchronousDeallocation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Virtual address allocator supports asynchronous deallocation via
  // GPU timeline-based processing.
  EXPECT_TRUE(allocator->AllowsAsynchronousDeallocation());
}

TEST_F(DeviceAddressVmmAllocatorTest, ExplicitDeallocate) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Allocate memory.
  TF_ASSERT_OK_AND_ASSIGN(
      auto scoped_address,
      allocator->Allocate(executor_->device_ordinal(), 1024,
                          /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kP2P)));

  ASSERT_NE(scoped_address->opaque(), nullptr);
  DeviceAddressBase addr = scoped_address.cref();

  // Explicitly deallocate.
  EXPECT_THAT(allocator->Deallocate(executor_->device_ordinal(), addr),
              absl_testing::IsOk());

  // Release ownership to prevent double-free.
  scoped_address.Release();
}

TEST_F(DeviceAddressVmmAllocatorTest, DeallocateNull) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  // Deallocating null address should succeed.
  DeviceAddressBase null_addr;
  EXPECT_THAT(allocator->Deallocate(executor_->device_ordinal(), null_addr),
              absl_testing::IsOk());
}

// --- Timeline / sequence-number design tests ---
//
// These tests exercise the cuStreamWriteValue64-based deferred deallocation
// mechanism. Each pending Deallocate() call records an increasing seqno and
// enqueues a GPU timeline write; the CPU checks the pinned counter to decide
// when memory is safe to free.

// Verifies that TryReusePendingDeallocation returns the same virtual address
// when a new allocation of the same rounded size is requested immediately
// after a Deallocate. The reuse is safe because stream ordering guarantees
// all prior GPU work finishes before any new work submitted after Allocate.
TEST_F(DeviceAddressVmmAllocatorTest,
       PendingDeallocationReusesSameVirtualAddress) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  constexpr uint64_t kSize = 1024;

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr1, allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                                      static_cast<int64_t>(MemorySpace::kP2P)));
  void* const va = addr1->opaque();

  // Deallocate: timeline write is enqueued but VA is not freed yet.
  DeviceAddressBase raw = addr1.cref();
  addr1.Release();
  ASSERT_THAT(allocator->Deallocate(ordinal, raw), IsOk());

  // Allocate the same size — TryReusePendingDeallocation should match the
  // pending entry and return the identical virtual address.
  TF_ASSERT_OK_AND_ASSIGN(
      auto addr2, allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                                      static_cast<int64_t>(MemorySpace::kP2P)));
  EXPECT_EQ(addr2->opaque(), va);

  // Sync to drain all pending GPU timeline writes before the allocator
  // is destroyed.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

// Verifies that deallocating memory while the GPU is still writing to it is
// safe. The timeline write for the deallocation is enqueued on the stream
// AFTER the memcpy, so the physical memory is not freed until the GPU finishes.
TEST_F(DeviceAddressVmmAllocatorTest,
       DeferredDeallocationSafeWhileGpuWritesData) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();

  TF_ASSERT_OK_AND_ASSIGN(
      auto addr,
      allocator->Allocate(ordinal, sizeof(uint64_t), /*retry_on_failure=*/true,
                          static_cast<int64_t>(MemorySpace::kP2P)));

  // Enqueue a memcpy to the allocated buffer on stream_.
  constexpr uint64_t kPattern = 0xCAFEBABEDEADBEEFULL;
  DeviceAddressBase dev_addr = addr.cref();
  ASSERT_THAT(stream_->Memcpy(&dev_addr, &kPattern, sizeof(kPattern)), IsOk());

  // Deallocate while the memcpy is still queued. The seqno timeline write is
  // appended to the stream AFTER the memcpy, so the VA cannot be reused until
  // the GPU advances past it.
  ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());

  // Sync: both the memcpy and the timeline write execute in order.
  // No crash here means the physical memory was not freed prematurely.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

// Allocates and deallocates N buffers, recording a distinct seqno for each.
// After a single stream sync all seqnos have been written by the GPU, so
// re-allocating the same size should succeed by reusing the pending entries.
TEST_F(DeviceAddressVmmAllocatorTest,
       MultipleSeqnosAllCompleteAfterStreamSync) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();
  constexpr int kCount = 8;
  constexpr uint64_t kSize = 1024;

  // Allocate kCount buffers and immediately queue their deallocation.
  // Each Deallocate increments next_seqno and enqueues a timeline write.
  for (int i = 0; i < kCount; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kP2P)));
    ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  }

  // Sync the stream: all kCount timeline writes (seqnos 1..kCount) complete.
  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());

  // Each new Allocate call finds a matching pending entry via
  // TryReusePendingDeallocation (or via ProcessCompletedPendingDeallocations
  // once the pending queue is exhausted).
  for (int i = 0; i < kCount; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, kSize, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kP2P)));
    EXPECT_FALSE(addr.is_null());
  }

  ASSERT_THAT(stream_->BlockHostUntilDone(), IsOk());
}

// Verifies that the destructor correctly spin-waits on the pinned timeline
// counter until all pending GPU timeline writes complete, then frees the
// physical memory without crashing.
TEST_F(DeviceAddressVmmAllocatorTest,
       DestructorWithPendingDeallocationsDoesNotCrash) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int ordinal = executor_->device_ordinal();

  // Queue several deallocations without syncing the stream first.
  for (int i = 0; i < 4; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(ordinal, 1024, /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kP2P)));
    ASSERT_THAT(allocator->Deallocate(ordinal, addr.Release()), IsOk());
  }

  // Destroy without an explicit stream sync. The destructor must spin on the
  // pinned_timeline until the GPU writes all pending seqnos, then free
  // each virtual address safely.
  allocator.reset();  // Must not crash or leak.
}

TEST_F(DeviceAddressVmmAllocatorTest, UnknownDeviceOrdinalReturnsError) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(executor_, stream_.get()));

  const int unknown_ordinal = 9999;
  EXPECT_FALSE(allocator
                   ->Allocate(unknown_ordinal, 1024, /*retry_on_failure=*/true,
                              static_cast<int64_t>(MemorySpace::kP2P))
                   .ok());
  // Null Deallocate always succeeds (early return before ordinal lookup).
  EXPECT_THAT(allocator->Deallocate(unknown_ordinal, DeviceAddressBase{}),
              absl_testing::IsOk());
  // Non-null address on unknown ordinal returns error.
  DeviceAddressBase fake_addr(reinterpret_cast<void*>(0x1000), 64);
  EXPECT_FALSE(allocator->Deallocate(unknown_ordinal, fake_addr).ok());
  EXPECT_FALSE(allocator->GetStream(unknown_ordinal).ok());
  EXPECT_FALSE(allocator->GetStreamExecutor(unknown_ordinal).ok());
}

// Multi-device fixture: skips the test if fewer than two CUDA devices are
// available.
class MultiDeviceVmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("CUDA");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "CUDA platform not available";
    }
    platform_ = platform_or.value();

    if (platform_->VisibleDeviceCount() < 2) {
      GTEST_SKIP() << "Fewer than two CUDA devices available";
    }

    for (int i = 0; i < 2; ++i) {
      auto executor_or = platform_->ExecutorForDevice(i);
      if (!executor_or.ok()) {
        GTEST_SKIP() << "CUDA executor not available for device " << i;
      }
      executors_.push_back(executor_or.value());

      auto stream_or = executors_.back()->CreateStream();
      if (!stream_or.ok()) {
        GTEST_SKIP() << "Failed to create stream for device " << i;
      }
      streams_.push_back(std::move(stream_or.value()));
    }

    // Probe for cuStreamWriteValue64 support using device 0 (requires CC
    // >= 7.0).
    auto probe = gpu::CudaDeviceAddressVmmAllocator::Create(executors_[0],
                                                            streams_[0].get());
    if (absl::IsUnimplemented(probe.status())) {
      GTEST_SKIP() << "Device does not support cuStreamWriteValue64 "
                      "(requires compute capability >= 7.0): "
                   << probe.status();
    }
  }

  Platform* platform_ = nullptr;
  std::vector<StreamExecutor*> executors_;
  std::vector<std::unique_ptr<Stream>> streams_;
};

TEST_F(MultiDeviceVmmAllocatorTest, AllocateOnBothDevices) {
  std::vector<DeviceAddressVmmAllocator::DeviceConfig> configs;
  for (int i = 0; i < 2; ++i) {
    configs.push_back({executors_[i], streams_[i].get()});
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(platform_, configs));

  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto addr,
        allocator->Allocate(executors_[i]->device_ordinal(), 1024,
                            /*retry_on_failure=*/true,
                            static_cast<int64_t>(MemorySpace::kP2P)));
    EXPECT_FALSE(addr.is_null());
    EXPECT_EQ(addr->size(), 1024);
    EXPECT_NE(
        allocator->GetRawAllocation(executors_[i]->device_ordinal(), *addr),
        nullptr);
    EXPECT_NE(allocator->GetReservation(executors_[i]->device_ordinal(), *addr),
              nullptr);
    TF_ASSERT_OK_AND_ASSIGN(
        StreamExecutor * se,
        allocator->GetStreamExecutor(executors_[i]->device_ordinal()));
    EXPECT_EQ(se, executors_[i]);
    TF_ASSERT_OK_AND_ASSIGN(
        Stream * stream, allocator->GetStream(executors_[i]->device_ordinal()));
    EXPECT_EQ(stream, streams_[i].get());
  }
}

TEST_F(MultiDeviceVmmAllocatorTest, AllocationOnOneDeviceDoesNotAffectOther) {
  std::vector<DeviceAddressVmmAllocator::DeviceConfig> configs;
  for (int i = 0; i < 2; ++i) {
    configs.push_back({executors_[i], streams_[i].get()});
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(platform_, configs));

  // Allocate on device 0.
  TF_ASSERT_OK_AND_ASSIGN(
      auto addr0, allocator->Allocate(executors_[0]->device_ordinal(), 4096,
                                      /*retry_on_failure=*/true,
                                      static_cast<int64_t>(MemorySpace::kP2P)));
  EXPECT_FALSE(addr0.is_null());

  // GetRawAllocation for addr0's pointer on device 1 should return nullptr
  // (different per-device map).
  EXPECT_EQ(
      allocator->GetRawAllocation(executors_[1]->device_ordinal(), *addr0),
      nullptr);
}

}  // namespace
}  // namespace stream_executor
