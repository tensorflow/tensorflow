/* Copyright 2022 The OpenXLA Authors.

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
#include "xla/pjrt/pjrt_executable.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/executable_metadata.pb.h"
#include "xla/pjrt/proto/execute_options.pb.h"
#include "xla/service/computation_placer.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(CompileOptionsTest, Serialization) {
  CompileOptions src;
  const std::string kCompilerVariant = "linked_compiler";
  src.compile_portable_executable = true;
  src.parameter_is_tupled_arguments = true;
  src.profile_version = 1;
  src.argument_layouts = {ShapeUtil::MakeShape(S32, {1})};
  src.matrix_unit_operand_precision = PrecisionConfig::HIGHEST;
  src.allow_in_place_mlir_modification = true;
  ExecutableBuildOptions build_option;
  build_option.set_device_assignment(DeviceAssignment(1, 1));
  src.executable_build_options = build_option;
  src.compiler_variant = kCompilerVariant;

  TF_ASSERT_OK_AND_ASSIGN(CompileOptionsProto proto, src.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(CompileOptions output,
                          CompileOptions::FromProto(proto));
  TF_ASSERT_OK_AND_ASSIGN(CompileOptionsProto output_proto, src.ToProto());

  EXPECT_EQ(proto.SerializeAsString(), output_proto.SerializeAsString());
}

TEST(CompileOptionsTest, DeserializeSerializedMultiSliceConfig) {
  CompileOptionsProto proto;
  std::string serialized_config = "multi_size_config";
  *proto.mutable_serialized_multi_slice_config() = serialized_config;

  TF_ASSERT_OK_AND_ASSIGN(CompileOptions option,
                          CompileOptions::FromProto(proto));

  EXPECT_EQ(option.multi_slice_config, nullptr);
  EXPECT_EQ(option.serialized_multi_slice_config, serialized_config);
}

TEST(CompileOptionsTest, Defaults) {
  CompileOptions src;
  EXPECT_EQ(src.compile_portable_executable, false);
  EXPECT_EQ(src.parameter_is_tupled_arguments, false);
  EXPECT_EQ(src.allow_in_place_mlir_modification, false);
  EXPECT_EQ(src.matrix_unit_operand_precision, PrecisionConfig::DEFAULT);
  EXPECT_EQ(src.compiler_variant, std::nullopt);
}

TEST(ExecuteOptionsTest, Serialization) {
  ExecuteOptions src;
  src.launch_id = 1234;
  src.strict_shape_checking = true;
  src.execution_mode = ExecuteOptions::ExecutionMode::kAsynchronous;
  src.non_donatable_input_indices = {2, 3};
  src.call_location = "foo:1";

  TF_ASSERT_OK_AND_ASSIGN(ExecuteOptionsProto proto, src.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(ExecuteOptions output,
                          ExecuteOptions::FromProto(proto));
  TF_ASSERT_OK_AND_ASSIGN(ExecuteOptionsProto output_proto, src.ToProto());

  EXPECT_EQ(proto.SerializeAsString(), output_proto.SerializeAsString());
}

TEST(ExecuteOptionsTest, SendRecvNotSupported) {
  ExecuteOptions options;
  std::vector<std::vector<SendCallback>> send_callbacks(1);
  options.send_callbacks = send_callbacks;
  std::vector<std::vector<RecvCallback>> recv_callbacks(1);
  options.recv_callbacks = recv_callbacks;

  EXPECT_THAT(
      options.ToProto(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          "ExecuteOptions with send/recv calbacks is not serializable"));
}

TEST(ExecuteOptionsTest, ApplyOptionsCanParseStringsAndEnums) {
  CompileOptions src;
  src.env_option_overrides = {
      {"xla_gpu_use_runtime_fusion", std::string("True")},
      {"xla_gpu_graph_min_graph_size", std::string("2")},
      {"xla_gpu_auto_spmd_partitioning_memory_budget_ratio", 0.9},
      {"xla_gpu_pgle_profile_file_or_directory_path", std::string("abc")},
      // Repeated fields.
      {"xla_gpu_disable_async_collectives", std::string("2,REDUCESCATTER")},
      {"xla_disable_hlo_passes",
       std::string("rematerialization,something else")},
      // Repeated fields provided twice. The last one wins.
      {"xla_enable_hlo_passes_only", std::string("one, two, three")},
      {"xla_enable_hlo_passes_only", std::string(",,second, , third,")},
      {"xla_gpu_enable_command_buffer", std::string("CUSTOM_CALL,COLLECTIVES")},
      {"xla_gpu_enable_command_buffer",
       static_cast<int64_t>(DebugOptions::CUSTOM_CALL)}};
  TF_EXPECT_OK(src.ApplyAllOptionOverrides());
  auto& debug_options = src.executable_build_options.debug_options();
  EXPECT_EQ(debug_options.xla_gpu_use_runtime_fusion(), true);
  EXPECT_EQ(debug_options.xla_gpu_graph_min_graph_size(), 2);
  EXPECT_FLOAT_EQ(
      debug_options.xla_gpu_auto_spmd_partitioning_memory_budget_ratio(), 0.9);
  EXPECT_EQ(debug_options.xla_gpu_pgle_profile_file_or_directory_path(), "abc");
  EXPECT_THAT(debug_options.xla_gpu_disable_async_collectives(),
              testing::ElementsAre(xla::DebugOptions::ALLGATHER,
                                   xla::DebugOptions::REDUCESCATTER));
  EXPECT_THAT(debug_options.xla_disable_hlo_passes(),
              testing::ElementsAre("rematerialization", "something else"));
  EXPECT_THAT(debug_options.xla_enable_hlo_passes_only(),
              testing::ElementsAre("", "", "second", " ", " third", ""));
  EXPECT_THAT(debug_options.xla_gpu_enable_command_buffer(),
              testing::ElementsAre(DebugOptions::CUSTOM_CALL));

  // Test that repeated fields are cleared when empty string is provided.
  src.env_option_overrides = {
      {"xla_gpu_enable_command_buffer", std::string("")}};
  TF_EXPECT_OK(src.ApplyAllOptionOverrides());
  EXPECT_TRUE(debug_options.xla_gpu_enable_command_buffer().empty());
}

TEST(CompiledMemoryStatsTest, Serialization) {
  CompiledMemoryStats stats;
  stats.generated_code_size_in_bytes = 2;
  stats.argument_size_in_bytes = 3;
  stats.output_size_in_bytes = 5;
  stats.alias_size_in_bytes = 7;
  stats.temp_size_in_bytes = 11;
  stats.host_generated_code_size_in_bytes = 13;
  stats.host_argument_size_in_bytes = 17;
  stats.host_output_size_in_bytes = 19;
  stats.host_alias_size_in_bytes = 23;
  stats.host_temp_size_in_bytes = 29;
  stats.peak_memory_in_bytes = 31;
  stats.total_size_in_bytes = 37;

  CompiledMemoryStatsProto serialized = stats.ToProto();
  CompiledMemoryStats deserialized = CompiledMemoryStats::FromProto(serialized);
  EXPECT_EQ(serialized.SerializeAsString(),
            deserialized.ToProto().SerializeAsString());
}

}  // namespace
}  // namespace xla
