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

#include "xla/pjrt/extensions/abi_version/abi_version_extension.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_abi_version.h"

namespace pjrt {
namespace {

TEST(CommonRuntimeAbiVersionFromProtoTest, FailsOnInvalidStructSize) {
  PJRT_RuntimeAbiVersion_FromProto_Args args{};
  args.struct_size = 1;
  args.serialized_proto = "";
  args.serialized_proto_size = 0;

  const auto from_proto = [](const xla::PjRtRuntimeAbiVersionProto& proto)
      -> absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>> {
    return absl::UnimplementedError("Not implemented");
  };

  PJRT_Error* error = CommonRuntimeAbiVersionFromProto(from_proto, &args);
  ASSERT_NE(error, nullptr);
  absl::Cleanup cleanup_error = [&] {
    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    pjrt::PJRT_Error_Destroy(&destroy_args);
  };

  PJRT_Error_GetCode_Args get_code_args{};
  get_code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  get_code_args.error = error;
  EXPECT_EQ(pjrt::PJRT_Error_GetCode(&get_code_args), nullptr);
  EXPECT_EQ(get_code_args.code, PJRT_Error_Code_INVALID_ARGUMENT);
}

TEST(CommonRuntimeAbiVersionFromProtoTest, FailsOnInvalidProto) {
  PJRT_RuntimeAbiVersion_FromProto_Args args{};
  args.struct_size = PJRT_RuntimeAbiVersion_FromProto_Args_STRUCT_SIZE;
  args.serialized_proto = "\0";
  args.serialized_proto_size = 1;

  const auto from_proto = [](const xla::PjRtRuntimeAbiVersionProto& proto)
      -> absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>> {
    return absl::UnimplementedError("Not implemented");
  };

  PJRT_Error* error = CommonRuntimeAbiVersionFromProto(from_proto, &args);
  ASSERT_NE(error, nullptr);
  absl::Cleanup cleanup_error = [&] {
    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    pjrt::PJRT_Error_Destroy(&destroy_args);
  };

  PJRT_Error_GetCode_Args get_code_args{};
  get_code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  get_code_args.error = error;
  EXPECT_EQ(pjrt::PJRT_Error_GetCode(&get_code_args), nullptr);
  EXPECT_EQ(get_code_args.code, PJRT_Error_Code_INVALID_ARGUMENT);
}

TEST(CommonRuntimeAbiVersionFromProtoTest, CallsFromProtoFactoryFunction) {
  PJRT_RuntimeAbiVersion_FromProto_Args args{};
  args.struct_size = PJRT_RuntimeAbiVersion_FromProto_Args_STRUCT_SIZE;

  xla::PjRtRuntimeAbiVersionProto proto;
  std::string serialized = proto.SerializeAsString();
  args.serialized_proto = serialized.c_str();
  args.serialized_proto_size = serialized.size();

  const auto from_proto = [](const xla::PjRtRuntimeAbiVersionProto& proto)
      -> absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>> {
    return absl::UnimplementedError("Not implemented");
  };

  PJRT_Error* error = CommonRuntimeAbiVersionFromProto(from_proto, &args);
  ASSERT_NE(error, nullptr);
  absl::Cleanup cleanup_error = [&] {
    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    pjrt::PJRT_Error_Destroy(&destroy_args);
  };

  PJRT_Error_GetCode_Args get_code_args{};
  get_code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  get_code_args.error = error;
  EXPECT_EQ(pjrt::PJRT_Error_GetCode(&get_code_args), nullptr);
  EXPECT_EQ(get_code_args.code, PJRT_Error_Code_UNIMPLEMENTED);
}

TEST(CommonExecutableAbiVersionFromProtoTest, FailsOnInvalidStructSize) {
  PJRT_ExecutableAbiVersion_FromProto_Args args{};
  args.struct_size = 1;
  args.serialized_proto = "";
  args.serialized_proto_size = 0;

  const auto from_proto = [](const xla::PjRtExecutableAbiVersionProto& proto)
      -> absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>> {
    return absl::UnimplementedError("Not implemented");
  };

  PJRT_Error* error = CommonExecutableAbiVersionFromProto(from_proto, &args);
  ASSERT_NE(error, nullptr);
  absl::Cleanup cleanup_error = [&] {
    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    pjrt::PJRT_Error_Destroy(&destroy_args);
  };

  PJRT_Error_GetCode_Args get_code_args{};
  get_code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  get_code_args.error = error;
  EXPECT_EQ(pjrt::PJRT_Error_GetCode(&get_code_args), nullptr);
  EXPECT_EQ(get_code_args.code, PJRT_Error_Code_INVALID_ARGUMENT);
}

TEST(CommonExecutableAbiVersionFromProtoTest, FailsOnInvalidProto) {
  PJRT_ExecutableAbiVersion_FromProto_Args args{};
  args.struct_size = PJRT_ExecutableAbiVersion_FromProto_Args_STRUCT_SIZE;
  args.serialized_proto = "\0";
  args.serialized_proto_size = 1;

  const auto from_proto = [](const xla::PjRtExecutableAbiVersionProto& proto)
      -> absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>> {
    return absl::UnimplementedError("Not implemented");
  };

  PJRT_Error* error = CommonExecutableAbiVersionFromProto(from_proto, &args);
  ASSERT_NE(error, nullptr);
  absl::Cleanup cleanup_error = [&] {
    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    pjrt::PJRT_Error_Destroy(&destroy_args);
  };

  PJRT_Error_GetCode_Args get_code_args{};
  get_code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  get_code_args.error = error;
  EXPECT_EQ(pjrt::PJRT_Error_GetCode(&get_code_args), nullptr);
  EXPECT_EQ(get_code_args.code, PJRT_Error_Code_INVALID_ARGUMENT);
}

TEST(CommonExecutableAbiVersionFromProtoTest, CallsFromProtoFactoryFunction) {
  PJRT_ExecutableAbiVersion_FromProto_Args args{};
  args.struct_size = PJRT_ExecutableAbiVersion_FromProto_Args_STRUCT_SIZE;

  xla::PjRtExecutableAbiVersionProto proto;
  std::string serialized = proto.SerializeAsString();
  args.serialized_proto = serialized.c_str();
  args.serialized_proto_size = serialized.size();

  const auto from_proto = [](const xla::PjRtExecutableAbiVersionProto& proto)
      -> absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>> {
    return absl::UnimplementedError("Not implemented");
  };

  PJRT_Error* error = CommonExecutableAbiVersionFromProto(from_proto, &args);
  ASSERT_NE(error, nullptr);
  absl::Cleanup cleanup_error = [&] {
    PJRT_Error_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.error = error;
    pjrt::PJRT_Error_Destroy(&destroy_args);
  };

  PJRT_Error_GetCode_Args get_code_args{};
  get_code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  get_code_args.error = error;
  EXPECT_EQ(pjrt::PJRT_Error_GetCode(&get_code_args), nullptr);
  EXPECT_EQ(get_code_args.code, PJRT_Error_Code_UNIMPLEMENTED);
}

}  // namespace
}  // namespace pjrt
