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
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"

struct PJRT_RuntimeAbiVersion {
  std::unique_ptr<xla::PjRtRuntimeAbiVersion> version;
};

struct PJRT_ExecutableAbiVersion {
  std::unique_ptr<xla::PjRtExecutableAbiVersion> version;
};

struct PJRT_SerializedProto {
  std::string str;
};

namespace pjrt {

namespace {

void SerializedProtoDeleter(PJRT_SerializedProto* ptr) { delete ptr; }

PJRT_Error* ClientRuntimeAbiVersion(PJRT_Client_RuntimeAbiVersion_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_RuntimeAbiVersion_Args",
      PJRT_Client_RuntimeAbiVersion_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(auto version,
                        args->client->client->RuntimeAbiVersion());
  args->abi_version = new PJRT_RuntimeAbiVersion{std::move(version)};
  return nullptr;
}

PJRT_Error* ExecutableGetAbiVersion(PJRT_Executable_GetAbiVersion_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Executable_GetAbiVersion_Args",
      PJRT_Executable_GetAbiVersion_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(auto version,
                        args->executable->executable->GetAbiVersion());
  args->abi_version = new PJRT_ExecutableAbiVersion{std::move(version)};
  return nullptr;
}

PJRT_Error* RuntimeAbiVersionDestroy(
    PJRT_RuntimeAbiVersion_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RuntimeAbiVersion_Destroy_Args",
      PJRT_RuntimeAbiVersion_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->abi_version;
  return nullptr;
}

PJRT_Error* RuntimeAbiVersionIsCompatibleWithRuntime(
    PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args",
      PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_RETURN_IF_ERROR(args->abi_version->version->IsCompatibleWith(
      *args->other_abi_version->version));
  return nullptr;
}

PJRT_Error* RuntimeAbiVersionIsCompatibleWithExecutable(
    PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args",
      PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_RETURN_IF_ERROR(args->abi_version->version->IsCompatibleWith(
      *args->executable_abi_version->version));
  return nullptr;
}

PJRT_Error* RuntimeAbiVersionToProto(
    PJRT_RuntimeAbiVersion_ToProto_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RuntimeAbiVersion_ToProto_Args",
      PJRT_RuntimeAbiVersion_ToProto_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(auto proto, args->abi_version->version->ToProto());
  auto holder = new PJRT_SerializedProto;
  holder->str = proto.SerializeAsString();
  args->serialized_proto = holder->str.c_str();
  args->serialized_proto_size = holder->str.size();
  args->serialized_proto_holder = holder;
  args->serialized_proto_deleter = SerializedProtoDeleter;
  return nullptr;
}

PJRT_Error* RuntimeAbiVersionPlatformId(
    PJRT_RuntimeAbiVersion_PlatformId_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RuntimeAbiVersion_PlatformId_Args",
      PJRT_RuntimeAbiVersion_PlatformId_Args_STRUCT_SIZE, args->struct_size));
  args->platform_id = args->abi_version->version->platform_id();
  return nullptr;
}

PJRT_Error* ExecutableAbiVersionDestroy(
    PJRT_ExecutableAbiVersion_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecutableAbiVersion_Destroy_Args",
      PJRT_ExecutableAbiVersion_Destroy_Args_STRUCT_SIZE, args->struct_size));
  delete args->abi_version;
  return nullptr;
}

PJRT_Error* ExecutableAbiVersionToProto(
    PJRT_ExecutableAbiVersion_ToProto_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecutableAbiVersion_ToProto_Args",
      PJRT_ExecutableAbiVersion_ToProto_Args_STRUCT_SIZE, args->struct_size));
  PJRT_ASSIGN_OR_RETURN(auto proto, args->abi_version->version->ToProto());
  auto holder = new PJRT_SerializedProto;
  holder->str = proto.SerializeAsString();
  args->serialized_proto = holder->str.c_str();
  args->serialized_proto_size = holder->str.size();
  args->serialized_proto_holder = holder;
  args->serialized_proto_deleter = SerializedProtoDeleter;
  return nullptr;
}

PJRT_Error* ExecutableAbiVersionPlatformId(
    PJRT_ExecutableAbiVersion_PlatformId_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecutableAbiVersion_PlatformId_Args",
      PJRT_ExecutableAbiVersion_PlatformId_Args_STRUCT_SIZE,
      args->struct_size));
  args->platform_id = args->abi_version->version->platform_id();
  return nullptr;
}

}  // namespace

PJRT_Error* CommonRuntimeAbiVersionFromProto(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>>(
            const xla::PjRtRuntimeAbiVersionProto&)>
        from_proto,
    PJRT_RuntimeAbiVersion_FromProto_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RuntimeAbiVersion_FromProto_Args",
      PJRT_RuntimeAbiVersion_FromProto_Args_STRUCT_SIZE, args->struct_size));
  xla::PjRtRuntimeAbiVersionProto proto;
  if (!proto.ParseFromString(absl::string_view(args->serialized_proto,
                                               args->serialized_proto_size))) {
    return new PJRT_Error{absl::InvalidArgumentError(
        "Failed to parse PjRtRuntimeAbiVersionProto")};
  }

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtRuntimeAbiVersion> version,
                        from_proto(proto));
  args->abi_version = new PJRT_RuntimeAbiVersion{std::move(version)};
  return nullptr;
}

PJRT_Error* CommonExecutableAbiVersionFromProto(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>(
            const xla::PjRtExecutableAbiVersionProto&)>
        from_proto,
    PJRT_ExecutableAbiVersion_FromProto_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecutableAbiVersion_FromProto_Args",
      PJRT_ExecutableAbiVersion_FromProto_Args_STRUCT_SIZE, args->struct_size));
  xla::PjRtExecutableAbiVersionProto proto;
  if (!proto.ParseFromString(absl::string_view(args->serialized_proto,
                                               args->serialized_proto_size))) {
    return new PJRT_Error{absl::InvalidArgumentError(
        "Failed to parse PjRtExecutableAbiVersionProto")};
  }

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtExecutableAbiVersion> version,
                        from_proto(proto));
  args->abi_version = new PJRT_ExecutableAbiVersion{std::move(version)};
  return nullptr;
}

PJRT_AbiVersion_Extension CreateAbiVersionExtension(
    PJRT_RuntimeAbiVersion_FromProto* runtime_abi_version_from_proto,
    PJRT_ExecutableAbiVersion_FromProto* executable_abi_version_from_proto,
    PJRT_Extension_Base* next) {
  return PJRT_AbiVersion_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_AbiVersion_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_AbiVersion,
          /*next=*/next,
      },
      /*client_runtime_abi_version=*/ClientRuntimeAbiVersion,
      /*executable_get_abi_version=*/ExecutableGetAbiVersion,
      /*runtime_abi_version_destroy=*/RuntimeAbiVersionDestroy,
      /*runtime_abi_version_is_compatible_with_runtime=*/
      RuntimeAbiVersionIsCompatibleWithRuntime,
      /*runtime_abi_version_is_compatible_with_executable=*/
      RuntimeAbiVersionIsCompatibleWithExecutable,
      /*runtime_abi_version_to_proto=*/RuntimeAbiVersionToProto,
      /*runtime_abi_version_platform_id=*/RuntimeAbiVersionPlatformId,
      /*executable_abi_version_destroy=*/ExecutableAbiVersionDestroy,
      /*executable_abi_version_to_proto=*/ExecutableAbiVersionToProto,
      /*executable_abi_version_platform_id=*/ExecutableAbiVersionPlatformId,
      /*runtime_abi_version_from_proto=*/runtime_abi_version_from_proto,
      /*executable_abi_version_from_proto=*/executable_abi_version_from_proto,
  };
}

}  // namespace pjrt
