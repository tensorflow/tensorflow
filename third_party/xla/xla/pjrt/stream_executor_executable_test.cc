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

#include "xla/pjrt/stream_executor_executable.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiled_module.h"
#include "xla/service/mock_compiled_module.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::testing::Return;
using ::tsl::proto_testing::EqualsProto;

TEST(StreamExecutorExecutableTest, GetAbiVersion) {
  stream_executor::ExecutableAbiVersionProto executable_abi_version_proto;
  executable_abi_version_proto.mutable_cuda_platform_version()
      ->set_cuda_toolkit_version("1.2.3");
  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion executable_abi_version,
      stream_executor::ExecutableAbiVersion::FromProto(
          executable_abi_version_proto));

  constexpr PjRtPlatformId kPlatformId = 42;

  auto module = std::make_unique<MockCompiledModule>();
  EXPECT_CALL(*module, GetExecutableAbiVersion())
      .WillOnce(Return(executable_abi_version));
  std::vector<std::unique_ptr<CompiledModule>> modules{};
  modules.push_back(std::move(module));
  StreamExecutorExecutable executable(kPlatformId, CompileOptions(),
                                      std::move(modules), 0, 0, "name",
                                      "fingerprint", "memory_kind");

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtExecutableAbiVersion> abi_version,
                       executable.GetAbiVersion());
  EXPECT_EQ(abi_version->platform_id(), kPlatformId);

  ASSERT_OK_AND_ASSIGN(PjRtExecutableAbiVersionProto proto,
                       abi_version->ToProto());
  stream_executor::ExecutableAbiVersionProto
      reconstructed_executable_abi_version_proto;
  EXPECT_TRUE(reconstructed_executable_abi_version_proto.ParseFromString(
      proto.version()));
  EXPECT_THAT(reconstructed_executable_abi_version_proto,
              EqualsProto(executable_abi_version_proto));
}

}  // namespace
}  // namespace xla
