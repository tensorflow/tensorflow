/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/pjrt/compile_options.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace xla {
namespace {

using ::tsl::testing::StatusIs;

TEST(CompileOptionsTest, Serialization) {
  CompileOptions src;
  src.compile_portable_executable = true;
  src.parameter_is_tupled_arguments = true;
  src.profile_version = 1;
  src.argument_layouts = {ShapeUtil::MakeShape(S32, {1})};
  ExecutableBuildOptions build_option;
  build_option.set_device_assignment(DeviceAssignment(1, 1));
  src.executable_build_options = build_option;

  TF_ASSERT_OK_AND_ASSIGN(CompileOptionsProto proto, src.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(CompileOptions output,
                          CompileOptions::FromProto(proto));
  TF_ASSERT_OK_AND_ASSIGN(CompileOptionsProto output_proto, src.ToProto());

  EXPECT_EQ(proto.SerializeAsString(), output_proto.SerializeAsString());
}

TEST(CompileOptionsTest, MultiSliceConfigNotSupported) {
  CompileOptionsProto proto;
  *proto.mutable_serialized_multi_slice_config() = "multi_size_config";

  auto option = CompileOptions::FromProto(proto);

  EXPECT_THAT(
      option.status(),
      StatusIs(
          absl::StatusCode::kUnimplemented,
          "multi_slice_config not supported in CompileOptions::FromProto."));
}

TEST(ExecuteOptionsTest, Serialization) {
  ExecuteOptions src;
  src.arguments_are_tupled = true;
  src.untuple_result = false;
  src.launch_id = 1234;
  src.strict_shape_checking = true;
  src.execution_mode = ExecuteOptions::ExecutionMode::kAsynchronous;
  src.non_donatable_input_indices = {2, 3};

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
      StatusIs(absl::StatusCode::kUnimplemented,
               "ExecuteOptions with send/recv calbacks is not serializable"));
}

}  // namespace
}  // namespace xla
