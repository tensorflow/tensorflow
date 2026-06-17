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

#include "xla/backends/gpu/runtime/print_buffer_contents.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/ascii.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla::gpu {
namespace {
using ::testing::_;
using ::testing::HasSubstr;

TEST(PrintBufferContentsTest, PrintBufferContents) {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  stream_executor::Platform* platform =
      stream_executor::PlatformManager::PlatformWithName(name).value();
  stream_executor::StreamExecutor* executor =
      platform->ExecutorForDevice(0).value();

  auto stream = executor->CreateStream().value();

  stream_executor::DeviceAddress<int> arg1 =
      executor->AllocateArray<int32_t>(10, 0);

  TF_ASSERT_OK(stream->Memset32(&arg1, 0x12345678, 10 * sizeof(int32_t)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<stream_executor::KernelArg> kernel_args;
  kernel_args.push_back(arg1);
  stream_executor::TensorMap tensor_map;
  for (int i = 0; i < 128; ++i) {
    tensor_map.storage[i] = static_cast<std::byte>(i);
  }
  kernel_args.push_back(tensor_map);
  kernel_args.push_back(0x123456789);

  absl::ScopedMockLog log{absl::MockLogDefault::kIgnoreUnexpected};
  EXPECT_CALL(log, Log(_, _, HasSubstr("BUF(0) = 78 56 34 12 78 56 34 12")));
  EXPECT_CALL(
      log,
      Log(_, _,
          HasSubstr("TENSOR_MAP(1) = 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d "
                    "0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f")));
  EXPECT_CALL(log, Log(_, _, HasSubstr("INT(2) = 0x123456789")));

  log.StartCapturingLogs();
  PrintBufferContents(stream.get(), kernel_args);
  log.StopCapturingLogs();
}

}  // namespace
}  // namespace xla::gpu
