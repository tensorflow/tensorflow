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

#include "xla/pjrt/c/pjrt_c_api_abi_version_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"

namespace pjrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

PJRT_Error* FakeRuntimeAbiVersionFromProto(
    PJRT_RuntimeAbiVersion_FromProto_Args* args) {
  args->abi_version = reinterpret_cast<PJRT_RuntimeAbiVersion*>(0x12345);
  return nullptr;
}

PJRT_Error* FakeExecutableAbiVersionFromProto(
    PJRT_ExecutableAbiVersion_FromProto_Args* args) {
  args->abi_version = reinterpret_cast<PJRT_ExecutableAbiVersion*>(0x6789A);
  return nullptr;
}

PJRT_Error* FakeRuntimeAbiVersionDestroy(
    PJRT_RuntimeAbiVersion_Destroy_Args* args) {
  return nullptr;
}

PJRT_Error* FakeExecutableAbiVersionDestroy(
    PJRT_ExecutableAbiVersion_Destroy_Args* args) {
  return nullptr;
}

TEST(PjRtCApiAbiVersionHelpersTest, ExtensionNull) {
  PJRT_Api c_api;
  c_api.struct_size = sizeof(PJRT_Api);
  c_api.extension_start = nullptr;

  xla::PjRtRuntimeAbiVersionProto runtime_proto;
  auto runtime_status = CApiRuntimeAbiVersionFromProto(runtime_proto, &c_api);
  // Fallback to nullptr check message.
  EXPECT_THAT(runtime_status,
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("not implemented in this PJRT plugin")));

  xla::PjRtExecutableAbiVersionProto executable_proto;
  auto executable_status =
      CApiExecutableAbiVersionFromProto(executable_proto, &c_api);
  // Fallback to nullptr check message.
  EXPECT_THAT(executable_status,
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("not implemented in this PJRT plugin")));
}

TEST(PjRtCApiAbiVersionHelpersTest, ExtensionTooSmall) {
  PJRT_AbiVersion_Extension extension;
  extension.base.struct_size = PJRT_STRUCT_SIZE(
      PJRT_AbiVersion_Extension, executable_abi_version_platform_id);
  extension.base.type = PJRT_Extension_Type::PJRT_Extension_Type_AbiVersion;
  extension.base.next = nullptr;

  PJRT_Api c_api;
  c_api.struct_size = sizeof(PJRT_Api);
  c_api.extension_start = &extension.base;

  xla::PjRtRuntimeAbiVersionProto runtime_proto;
  auto runtime_status = CApiRuntimeAbiVersionFromProto(runtime_proto, &c_api);
  EXPECT_THAT(
      runtime_status,
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("does not implement runtime_abi_version_from_proto")));

  xla::PjRtExecutableAbiVersionProto executable_proto;
  auto executable_status =
      CApiExecutableAbiVersionFromProto(executable_proto, &c_api);
  EXPECT_THAT(
      executable_status,
      StatusIs(
          absl::StatusCode::kUnimplemented,
          HasSubstr("does not implement executable_abi_version_from_proto")));
}

TEST(PjRtCApiAbiVersionHelpersTest, FunctionPointerNull) {
  PJRT_AbiVersion_Extension extension;
  extension.base.struct_size = PJRT_STRUCT_SIZE(
      PJRT_AbiVersion_Extension, executable_abi_version_from_proto);
  extension.base.type = PJRT_Extension_Type::PJRT_Extension_Type_AbiVersion;
  extension.base.next = nullptr;
  extension.runtime_abi_version_from_proto = nullptr;
  extension.executable_abi_version_from_proto = nullptr;

  PJRT_Api c_api;
  c_api.struct_size = sizeof(PJRT_Api);
  c_api.extension_start = &extension.base;

  xla::PjRtRuntimeAbiVersionProto runtime_proto;
  auto runtime_status = CApiRuntimeAbiVersionFromProto(runtime_proto, &c_api);
  EXPECT_THAT(
      runtime_status,
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("does not implement runtime_abi_version_from_proto")));

  xla::PjRtExecutableAbiVersionProto executable_proto;
  auto executable_status =
      CApiExecutableAbiVersionFromProto(executable_proto, &c_api);
  EXPECT_THAT(
      executable_status,
      StatusIs(
          absl::StatusCode::kUnimplemented,
          HasSubstr("does not implement executable_abi_version_from_proto")));
}

TEST(PjRtCApiAbiVersionHelpersTest, Success) {
  PJRT_AbiVersion_Extension extension;
  extension.base.struct_size = PJRT_STRUCT_SIZE(
      PJRT_AbiVersion_Extension, executable_abi_version_from_proto);
  extension.base.type = PJRT_Extension_Type::PJRT_Extension_Type_AbiVersion;
  extension.base.next = nullptr;
  extension.runtime_abi_version_destroy = FakeRuntimeAbiVersionDestroy;
  extension.executable_abi_version_destroy = FakeExecutableAbiVersionDestroy;
  extension.runtime_abi_version_from_proto = FakeRuntimeAbiVersionFromProto;
  extension.executable_abi_version_from_proto =
      FakeExecutableAbiVersionFromProto;

  PJRT_Api c_api;
  c_api.struct_size = sizeof(PJRT_Api);
  c_api.extension_start = &extension.base;

  xla::PjRtRuntimeAbiVersionProto runtime_proto;
  auto runtime_status = CApiRuntimeAbiVersionFromProto(runtime_proto, &c_api);
  EXPECT_TRUE(runtime_status.ok()) << runtime_status.status();

  xla::PjRtExecutableAbiVersionProto executable_proto;
  auto executable_status =
      CApiExecutableAbiVersionFromProto(executable_proto, &c_api);
  EXPECT_TRUE(executable_status.ok()) << executable_status.status();
}

}  // namespace
}  // namespace pjrt
