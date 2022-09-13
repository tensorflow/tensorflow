/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifdef GOOGLE_CUDA

#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"

#include <cstdint>

#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/test.h"

namespace stream_executor {
namespace cuda {
namespace {

using RedzoneCheckStatus = RedzoneAllocator::RedzoneCheckStatus;

static void EXPECT_REDZONE_OK(port::StatusOr<RedzoneCheckStatus> status) {
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(status.value().ok());
}

static void EXPECT_REDZONE_VIOLATION(
    port::StatusOr<RedzoneCheckStatus> status) {
  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(status.value().ok());
}

TEST(RedzoneAllocatorTest, WriteToRedzone) {
  constexpr int64_t kRedzoneSize = 1 << 23;  // 8MiB redzone on each side
  // Redzone pattern should not be equal to zero; otherwise modify_redzone will
  // break.
  constexpr uint8_t kRedzonePattern = 0x7e;

  // Allocate 32MiB + 1 byte (to make things misaligned)
  constexpr int64_t kAllocSize = (1 << 25) + 1;

  Platform* platform = MultiPlatformManager::PlatformWithName("cuda").value();
  StreamExecutor* stream_exec = platform->ExecutorForDevice(0).value();
  GpuAsmOpts opts;
  StreamExecutorMemoryAllocator se_allocator(platform, {stream_exec});

  Stream stream(stream_exec);
  stream.Init();
  RedzoneAllocator allocator(&stream, &se_allocator, opts,
                             /*memory_limit=*/(1LL << 32),
                             /*redzone_size=*/kRedzoneSize,
                             /*redzone_pattern=*/kRedzonePattern);
  TF_ASSERT_OK_AND_ASSIGN(DeviceMemory<uint8_t> buf,
                          allocator.AllocateBytes(/*byte_size=*/kAllocSize));
  EXPECT_REDZONE_OK(allocator.CheckRedzones());

  char* buf_addr = reinterpret_cast<char*>(buf.opaque());
  DeviceMemoryBase lhs_redzone(buf_addr - kRedzoneSize, kRedzoneSize);
  DeviceMemoryBase rhs_redzone(buf_addr + kAllocSize, kRedzoneSize);

  // Check that the redzones are in fact filled with kRedzonePattern.
  auto check_redzone = [&](DeviceMemoryBase redzone, absl::string_view name) {
    std::vector<uint8_t> host_buf(kRedzoneSize);
    TF_ASSERT_OK(stream.ThenMemcpy(host_buf.data(), redzone, kRedzoneSize)
                     .BlockHostUntilDone());
    const int64_t kMaxMismatches = 16;
    int64_t mismatches = 0;
    for (int64_t i = 0; i < host_buf.size(); ++i) {
      if (mismatches == kMaxMismatches) {
        ADD_FAILURE() << "Hit max number of mismatches; skipping others.";
        break;
      }
      if (host_buf[i] != kRedzonePattern) {
        ++mismatches;
        EXPECT_EQ(host_buf[i], kRedzonePattern)
            << "at index " << i << " of " << name << " redzone";
      }
    }
  };
  check_redzone(lhs_redzone, "lhs");
  check_redzone(rhs_redzone, "rhs");

  // Modifies a redzone, checks that RedzonesAreUnmodified returns false, then
  // reverts it back to its original value and checks that RedzonesAreUnmodified
  // returns true.
  auto modify_redzone = [&](DeviceMemoryBase redzone, int64_t offset,
                            absl::string_view name) {
    SCOPED_TRACE(absl::StrCat(name, ", offset=", offset));
    DeviceMemoryBase redzone_at_offset(
        reinterpret_cast<char*>(redzone.opaque()) + offset, 1);
    char old_redzone_value = 0;
    { EXPECT_REDZONE_OK(allocator.CheckRedzones()); }
    stream.ThenMemcpy(&old_redzone_value, redzone_at_offset, 1)
        .ThenMemZero(&redzone_at_offset, 1);
    EXPECT_REDZONE_VIOLATION(allocator.CheckRedzones());

    // Checking reinitializes the redzone.
    EXPECT_REDZONE_OK(allocator.CheckRedzones());
  };

  modify_redzone(lhs_redzone, /*offset=*/0, "lhs");
  modify_redzone(lhs_redzone, /*offset=*/kRedzoneSize - 1, "lhs");
  modify_redzone(rhs_redzone, /*offset=*/0, "rhs");
  modify_redzone(rhs_redzone, /*offset=*/kRedzoneSize - 1, "rhs");
}

// Older CUDA compute capabilities (<= 2.0) have a limitation that grid
// dimension X cannot be larger than 65535.
//
// Make sure we can launch kernels on sizes larger than that, given that the
// maximum number of threads per block is 1024.
TEST(RedzoneAllocatorTest, VeryLargeRedzone) {
  // Make sure the redzone size would require grid dimension > 65535.
  constexpr int64_t kRedzoneSize = 65535 * 1024 + 1;
  Platform* platform = MultiPlatformManager::PlatformWithName("cuda").value();
  StreamExecutor* stream_exec = platform->ExecutorForDevice(0).value();
  GpuAsmOpts opts;
  StreamExecutorMemoryAllocator se_allocator(platform, {stream_exec});
  Stream stream(stream_exec);
  stream.Init();
  RedzoneAllocator allocator(&stream, &se_allocator, opts,
                             /*memory_limit=*/(1LL << 32),
                             /*redzone_size=*/kRedzoneSize,
                             /*redzone_pattern=*/-1);
  (void)allocator.AllocateBytes(/*byte_size=*/1);
  EXPECT_REDZONE_OK(allocator.CheckRedzones());
}

}  // namespace
}  // namespace cuda
}  // namespace stream_executor

#endif  // GOOGLE_CUDA
