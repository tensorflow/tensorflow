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

#include "xla/backends/gpu/runtime/make_batch_pointers.h"

#include <array>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAreArray;
using tsl::testing::IsOk;

static absl::StatusOr<stream_executor::StreamExecutor*> GpuExecutor() {
  TF_ASSIGN_OR_RETURN(stream_executor::Platform * platform,
                      PlatformUtil::GetDefaultPlatform());
  return platform->ExecutorForDevice(0);
}

TEST(MakeBatchPointersTest, Basic) {
  TF_ASSERT_OK_AND_ASSIGN(stream_executor::StreamExecutor * executor,
                          GpuExecutor());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<stream_executor::Stream> stream,
                          executor->CreateStream());

  // We don't care what `base` points to, we only need a pointer to a buffer
  // that we can use as a base.
  stream_executor::DeviceMemory<char> base = executor->AllocateScalar<char>();
  stream_executor::DeviceMemory<void*> ptrs_out =
      executor->AllocateArray<void*>(8);

  constexpr int kStride = 13;
  constexpr int kN = 8;

  EXPECT_THAT(MakeBatchPointers(stream.get(), base, kStride, kN, ptrs_out),
              IsOk());

  std::array<void*, kN> result = {};

  EXPECT_THAT(
      executor->SynchronousMemcpy(result.data(), ptrs_out, kN * sizeof(void*)),
      IsOk());

  std::array<void*, kN> expected = {
      base.base() + 0 * kStride, base.base() + 1 * kStride,
      base.base() + 2 * kStride, base.base() + 3 * kStride,
      base.base() + 4 * kStride, base.base() + 5 * kStride,
      base.base() + 6 * kStride, base.base() + 7 * kStride,
  };

  EXPECT_THAT(result, ElementsAreArray(expected));
}

}  // namespace
}  // namespace xla::gpu
