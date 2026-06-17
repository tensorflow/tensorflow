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

#include "xla/pjrt/gpu/se_gpu_pjrt_runtime_abi_version.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/pjrt/stream_executor_pjrt_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/abi/mock_runtime_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::Return;
using ::tsl::proto_testing::EqualsProto;

TEST(StreamExecutorGpuPjRtRuntimeAbiVersionTest, platform_id) {
  constexpr PjRtPlatformId kPlatformId = 42;
  StreamExecutorGpuPjRtRuntimeAbiVersion runtime_abi_version(kPlatformId, {});
  EXPECT_EQ(runtime_abi_version.platform_id(), kPlatformId);
}

TEST(StreamExecutorGpuPjRtRuntimeAbiVersionTest, ToProto) {
  auto mock_runtime_abi_version =
      std::make_unique<stream_executor::MockRuntimeAbiVersion>();
  EXPECT_CALL(*mock_runtime_abi_version, ToProto())
      .WillOnce(Return(stream_executor::RuntimeAbiVersionProto()));

  constexpr PjRtPlatformId kPlatformId = 42;
  StreamExecutorGpuPjRtRuntimeAbiVersion runtime_abi_version(
      kPlatformId, std::move(mock_runtime_abi_version));

  PjRtRuntimeAbiVersionProto expected_proto;
  expected_proto.set_platform(kPlatformId);
  expected_proto.set_version(
      stream_executor::RuntimeAbiVersionProto().SerializeAsString());

  EXPECT_THAT(runtime_abi_version.ToProto(),
              IsOkAndHolds(EqualsProto(expected_proto)));
}

TEST(StreamExecutorGpuPjRtRuntimeAbiVersionTest, IsCompatibleWith) {
  auto mock_runtime_abi_version =
      std::make_unique<stream_executor::MockRuntimeAbiVersion>();
  EXPECT_CALL(*mock_runtime_abi_version, IsCompatibleWith(_))
      .WillOnce(Return(absl::OkStatus()));

  constexpr PjRtPlatformId kPlatformId = 42;
  StreamExecutorGpuPjRtRuntimeAbiVersion runtime_abi_version(
      kPlatformId, std::move(mock_runtime_abi_version));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion executable_abi_version,
      stream_executor::ExecutableAbiVersion::FromProto(
          stream_executor::ExecutableAbiVersionProto()));

  EXPECT_OK(runtime_abi_version.IsCompatibleWith(
      StreamExecutorPjRtExecutableAbiVersion(kPlatformId,
                                             executable_abi_version)));

  constexpr PjRtPlatformId kOtherPlatformId = 43;
  EXPECT_THAT(runtime_abi_version.IsCompatibleWith(
                  StreamExecutorPjRtExecutableAbiVersion(
                      kOtherPlatformId, executable_abi_version)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
