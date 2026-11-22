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

#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/check.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/device_id.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace se = stream_executor;

namespace {

using ::testing::_;
using ::testing::MatchesRegex;

static se::StreamExecutor* GpuExecutor() {
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}
}  // namespace

namespace stream_executor {

TEST(GpuCudaMallocAsyncAllocator, TwoAllocatorsShareDefaultPool) {
  se::StreamExecutor* executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream1, executor->CreateStream());
  auto allocator1 = GpuCudaMallocAsyncAllocator(
      /*platform_device_id*/ tsl::PlatformDeviceId(executor->device_ordinal()),
      /*pool_size*/ 2048,
      /*new_pool_size*/ true,
      /*release_threshold*/ true);
  allocator1.SetStreamAndPreallocateMemory(
      stream1->platform_specific_handle().stream);
  ASSERT_OK_AND_ASSIGN(auto stream2, executor->CreateStream());
  auto allocator2 = GpuCudaMallocAsyncAllocator(
      /*platform_device_id*/ tsl::PlatformDeviceId(executor->device_ordinal()),
      /*pool_size*/ 2048,
      /*new_pool_size*/ true,
      /*release_threshold*/ true);
  allocator2.SetStreamAndPreallocateMemory(
      stream2->platform_specific_handle().stream);
  void* addr1 = allocator1.AllocateRaw(128, 127);
  void* addr2 = allocator2.AllocateRaw(128, 129);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr1) & 127), 0);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr2) & 127), 0);
  allocator1.DeallocateRaw(addr1);
  allocator2.DeallocateRaw(addr2);
  EXPECT_TRUE(stream1->ok());
  EXPECT_TRUE(stream2->ok());
}

TEST(GpuCudaMallocAsyncAllocator, AddressAlignedDefaultPool) {
  se::StreamExecutor* executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  auto allocator = GpuCudaMallocAsyncAllocator(
      /*platform_device_id*/ tsl::PlatformDeviceId(executor->device_ordinal()),
      /*pool_size*/ 2048,
      /*new_pool_size*/ true,
      /*release_threshold*/ true);
  allocator.SetStreamAndPreallocateMemory(
      stream->platform_specific_handle().stream);
  void* addr1 = allocator.AllocateRaw(128, 127);
  void* addr2 = allocator.AllocateRaw(128, 129);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr1) & 127), 0);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr2) & 127), 0);
  allocator.DeallocateRaw(addr1);
  allocator.DeallocateRaw(addr2);
  EXPECT_TRUE(stream->ok());
}

TEST(GpuCudaMallocAsyncAllocator, AddressAlignedNewPool) {
  se::StreamExecutor* executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  auto allocator = GpuCudaMallocAsyncAllocator(
      /*platform_device_id*/ tsl::PlatformDeviceId(executor->device_ordinal()),
      /*create_new_pool*/ true,
      /*new_pool_size*/ 2048,
      /*reserve_memory*/ true,
      /*release_threshold*/ 0,
      /*sync_mode*/ false,
      /*compute_stats*/ false);
  allocator.SetStreamAndPreallocateMemory(
      stream->platform_specific_handle().stream);
  void* addr1 = allocator.AllocateRaw(128, 127);
  void* addr2 = allocator.AllocateRaw(128, 129);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr1) & 127), 0);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr2) & 127), 0);
  allocator.DeallocateRaw(addr1);
  allocator.DeallocateRaw(addr2);
  EXPECT_TRUE(stream->ok());
}

TEST(GpuCudaMallocAsyncAllocator, SyncAddressAlignedNewPool) {
  se::StreamExecutor* executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  auto allocator = GpuCudaMallocAsyncAllocator(
      /*platform_device_id*/ tsl::PlatformDeviceId(executor->device_ordinal()),
      /*create_new_pool*/ true,
      /*new_pool_size*/ 2048,
      /*reserve_memory*/ true,
      /*release_threshold*/ 0,
      /*sync_mode*/ true,
      /*compute_stats*/ true);
  allocator.SetStreamAndPreallocateMemory(
      stream->platform_specific_handle().stream);
  void* addr1 = allocator.AllocateRaw(128, 127);
  void* addr2 = allocator.AllocateRaw(128, 129);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr1) & 127), 0);
  CHECK_EQ((reinterpret_cast<uintptr_t>(addr2) & 127), 0);
  allocator.DeallocateRaw(addr1);
  allocator.DeallocateRaw(addr2);
  EXPECT_TRUE(stream->ok());
}

TEST(GpuCudaMallocAsyncAllocator, LogsPoolStatisticsWhenAllocationFails) {
  se::StreamExecutor* executor = GpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  auto allocator = GpuCudaMallocAsyncAllocator(
      tsl::PlatformDeviceId(executor->device_ordinal()),
      /*release_threshold=*/2048, /*reserve_memory=*/false,
      /*compute_stats=*/true);
  allocator.SetStreamAndPreallocateMemory(
      stream->platform_specific_handle().stream);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  for (absl::string_view attribute :
       {"CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT",
        "CU_MEMPOOL_ATTR_USED_MEM_CURRENT", "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH",
        "CU_MEMPOOL_ATTR_USED_MEM_HIGH"}) {
    EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                         MatchesRegex(absl::StrCat(attribute, ": [0-9]+"))));
  }
  log.StartCapturingLogs();

  // No GPU can allocate 1 EiB, so the allocation fails and the allocator logs
  // the statistics of its memory pool.
  EXPECT_EQ(allocator.AllocateRaw(128, uint64_t{1} << 60), nullptr);
}

}  // namespace stream_executor
