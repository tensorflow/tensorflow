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

#include "xla/service/pjrt_gpu_utils.h"

#include <string>
#include <variant>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

absl::StatusOr<stream_executor::GpuTargetConfigProto> GetTargetConfigProto(
    PjRtClient* const client) {
  ASSIGN_OR_RETURN(const PjRtTopologyDescription* topology,
                   client->GetTopologyDescription());
  auto it = topology->Attributes().find("target_config");
  if (it == topology->Attributes().end()) {
    return absl::InvalidArgumentError(
        "Topology description does not contain target config");
  }
  if (!std::holds_alternative<std::string>(it->second)) {
    return absl::InvalidArgumentError(
        "Target config is not a string in topology description");
  }
  stream_executor::GpuTargetConfigProto target_config_proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(
          std::get<std::string>(it->second), &target_config_proto)) {
    return absl::InvalidArgumentError(
        "Failed to parse target config from topology description");
  }
  return target_config_proto;
}
}  // namespace

GpuTargetConfig GetGpuTargetConfig(PjRtClient* const absl_nonnull client) {
  absl::StatusOr<stream_executor::GpuTargetConfigProto> target_config_proto =
      GetTargetConfigProto(client);
  CHECK_OK(target_config_proto);
  absl::StatusOr<GpuTargetConfig> target_config =
      GpuTargetConfig::FromProto(*target_config_proto);
  CHECK_OK(target_config);
  return *target_config;
}

}  // namespace xla::gpu
