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

#include "xla/pjrt/se/stream_executor_executable.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/service/compiled_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/mock_compiled_module.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

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
  auto hlo_module = std::make_shared<HloModule>("name", HloModuleConfig());
  EXPECT_CALL(*module, shared_optimized_module())
      .WillRepeatedly(Return(hlo_module));
  EXPECT_CALL(*module, GetExecutableAbiVersion())
      .WillOnce(Return(executable_abi_version));
  StreamExecutorExecutable executable(kPlatformId, CompileOptions(),
                                      std::move(module), 0, 0, "name",
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

std::unique_ptr<MockCompiledModule> MockModule() {
  auto module = std::make_unique<MockCompiledModule>();
  EXPECT_CALL(*module, shared_optimized_module())
      .WillRepeatedly(
          Return(std::make_shared<HloModule>("name", HloModuleConfig())));
  return module;
}

CompileOptions CompileOptionsWithNonRuntimeAndRuntimeFields() {
  CompileOptions options;
  DebugOptions& debug_options =
      *options.executable_build_options.mutable_debug_options();
  debug_options.set_xla_gpu_enable_fast_min_max(true);
  debug_options.set_xla_gpu_nccl_termination_timeout_seconds(30);
  return options;
}

TEST(StreamExecutorExecutableTest, SerializeExecutableStripsDebugOptions) {
  std::unique_ptr<MockCompiledModule> module = MockModule();
  EXPECT_CALL(*module, SerializeAsString())
      .WillOnce(Return(std::string("executable")));
  StreamExecutorExecutable executable(
      /*platform_id=*/42, CompileOptionsWithNonRuntimeAndRuntimeFields(),
      std::move(module), 1, 1, "name", "fingerprint", "memory_kind");

  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       executable.SerializeExecutable());
  ASSERT_OK_AND_ASSIGN(ExecutableAndOptionsProto proto,
                       SerializedGpuExecutableFromString(serialized));

  const DebugOptions& debug_options =
      proto.compile_options().executable_build_options().debug_options();
  EXPECT_FALSE(debug_options.has_xla_gpu_enable_fast_min_max());
  EXPECT_EQ(debug_options.xla_gpu_nccl_termination_timeout_seconds(), 30);
  EXPECT_EQ(proto.serialized_executable(), "executable");
}

TEST(StreamExecutorExecutableTest,
     SerializeEarlyExitExecutableStripsDebugOptions) {
  CompileOptions options = CompileOptionsWithNonRuntimeAndRuntimeFields();
  options.executable_build_options.mutable_debug_options()
      ->set_xla_early_exit_with_layouts(true);
  StreamExecutorExecutable executable(
      /*platform_id=*/42, options, MockModule(), 1, 1, "name", "fingerprint",
      "memory_kind");

  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       executable.SerializeExecutable());
  ASSERT_OK_AND_ASSIGN(ExecutableAndOptionsProto proto,
                       SerializedGpuExecutableFromString(serialized));
  ASSERT_OK_AND_ASSIGN(CompileOptions deserialized,
                       CompileOptions::FromProto(proto.compile_options()));

  EXPECT_TRUE(IsEarlyExitCompilation(deserialized));
  const DebugOptions& debug_options =
      deserialized.executable_build_options.debug_options();
  EXPECT_FALSE(debug_options.has_xla_gpu_enable_fast_min_max());
  EXPECT_EQ(debug_options.xla_gpu_nccl_termination_timeout_seconds(), 30);
}

}  // namespace
}  // namespace xla
