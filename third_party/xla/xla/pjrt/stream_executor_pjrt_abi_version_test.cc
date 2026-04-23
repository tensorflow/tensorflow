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

#include "xla/pjrt/stream_executor_pjrt_abi_version.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using tsl::proto_testing::EqualsProto;

TEST(StreamExecutorPjRtExecutableAbiVersionTest, GetPlatformId) {
  constexpr PjRtPlatformId kPlatformId = 123;
  StreamExecutorPjRtExecutableAbiVersion abi_version(
      kPlatformId, stream_executor::ExecutableAbiVersion());
  EXPECT_EQ(abi_version.platform_id(), kPlatformId);
}

TEST(StreamExecutorPjRtExecutableAbiVersionTest, ToProto) {
  constexpr PjRtPlatformId kPlatformId = 123;

  stream_executor::ExecutableAbiVersionProto executable_abi_version_proto;
  executable_abi_version_proto.set_platform_name("test_platform");
  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion executable_abi_version,
      stream_executor::ExecutableAbiVersion::FromProto(
          executable_abi_version_proto));

  StreamExecutorPjRtExecutableAbiVersion abi_version(kPlatformId,
                                                     executable_abi_version);
  EXPECT_EQ(abi_version.platform_id(), kPlatformId);
  ASSERT_OK_AND_ASSIGN(PjRtExecutableAbiVersionProto proto,
                       abi_version.ToProto());
  EXPECT_EQ(proto.platform(), kPlatformId);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StreamExecutorPjRtExecutableAbiVersion> from_proto,
      StreamExecutorPjRtExecutableAbiVersion::FromProto(proto));
  EXPECT_EQ(from_proto->platform_id(), kPlatformId);
  EXPECT_THAT(from_proto->executable_abi_version().proto(),
              EqualsProto(executable_abi_version_proto));
}

}  // namespace
}  // namespace xla
