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

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_abi_version.h"

namespace pjrt {

absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>>
CApiRuntimeAbiVersionFromProto(const xla::PjRtRuntimeAbiVersionProto& proto,
                               const PJRT_Api* c_api) {
  PJRT_AbiVersion_Extension* extension =
      pjrt::FindExtension<PJRT_AbiVersion_Extension>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_AbiVersion);
  if (extension == nullptr) {
    return absl::UnimplementedError(
        "AbiVersion extension not implemented in this PJRT plugin.");
  }
  std::string serialized_proto = proto.SerializeAsString();
  PJRT_RuntimeAbiVersion_FromProto_Args args;
  args.struct_size = PJRT_RuntimeAbiVersion_FromProto_Args_STRUCT_SIZE;
  args.serialized_proto = serialized_proto.c_str();
  args.serialized_proto_size = serialized_proto.size();
  RETURN_STATUS_IF_PJRT_ERROR(extension->runtime_abi_version_from_proto(&args),
                              c_api);
  return std::make_unique<xla::PjRtCApiRuntimeAbiVersion>(args.abi_version,
                                                          c_api, extension);
}

absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
CApiExecutableAbiVersionFromProto(
    const xla::PjRtExecutableAbiVersionProto& proto, const PJRT_Api* c_api) {
  PJRT_AbiVersion_Extension* extension =
      pjrt::FindExtension<PJRT_AbiVersion_Extension>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_AbiVersion);
  if (extension == nullptr) {
    return absl::UnimplementedError(
        "AbiVersion extension not implemented in this PJRT plugin.");
  }
  std::string serialized_proto = proto.SerializeAsString();
  PJRT_ExecutableAbiVersion_FromProto_Args args;
  args.struct_size = PJRT_ExecutableAbiVersion_FromProto_Args_STRUCT_SIZE;
  args.serialized_proto = serialized_proto.c_str();
  args.serialized_proto_size = serialized_proto.size();
  RETURN_STATUS_IF_PJRT_ERROR(
      extension->executable_abi_version_from_proto(&args), c_api);
  return std::make_unique<xla::PjRtCApiExecutableAbiVersion>(args.abi_version,
                                                             c_api, extension);
}

}  // namespace pjrt
