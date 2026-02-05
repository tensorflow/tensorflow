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

#include "xla/service/gpu/tests/hlo_pjrt_gpu_test_base.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tests/aot_utils.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {
std::unique_ptr<PjRtClient> GetPjRtClientForTest() {
  CHECK(ShouldUsePjRt())
      << "PjRt is required for tests extending HloPjRtGpuTestBase.";
  absl::StatusOr<std::unique_ptr<PjRtClient>> client =
      GetGlobalPjRtClientTestFactory().Get()();
  CHECK_OK(client.status())
      << "Failed to create PjRt client. " << client.status();
  return *std::move(client);
}

HloRunnerAgnosticTestBaseOptions BuildOptions(HloPjRtTestBaseOptions options) {
  HloRunnerAgnosticTestBaseOptions new_options;
  new_options.verifier_layout_sensitive = options.verifier_layout_sensitive;
  new_options.allow_mixed_precision_in_hlo_verifier =
      options.allow_mixed_precision_in_hlo_verifier;
  new_options.instruction_can_change_layout_func =
      std::move(options.instruction_can_change_layout_func);
  new_options.swallow_execution_errors =
      HasPjRtAotAwareSwallowExecutionErrors();
  return new_options;
}

absl::StatusOr<GpuTargetConfig> GetGpuTargetConfig(PjRtClient* const client) {
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
  return gpu::GpuTargetConfig::FromProto(target_config_proto);
}

stream_executor::DeviceDescription GetDeviceDescription(
    PjRtClient* const client) {
  absl::StatusOr<GpuTargetConfig> target_config = GetGpuTargetConfig(client);
  CHECK_OK(target_config.status());
  return std::move(target_config)->device_description;
}
}  // namespace

HloPjRtGpuTestBase::HloPjRtGpuTestBase(HloPjRtTestBaseOptions options)
    : HloPjRtGpuTestBase(GetPjRtClientForTest().release(), std::move(options)) {
}

HloPjRtGpuTestBase::HloPjRtGpuTestBase(PjRtClient* client,
                                       HloPjRtTestBaseOptions options)
    : HloPjRtGpuTestBase(
          GetGlobalPjRtClientTestFactory().GetDeviceShapeRepresentationFn(
              client),
          GetGlobalPjRtClientTestFactory().GetDeviceShapeSizeFn(client),
          GetDeviceDescription(client), absl::WrapUnique(client),
          std::move(options)) {}

HloPjRtGpuTestBase::HloPjRtGpuTestBase(
    DeviceShapeRepresentationFn device_shape_representation_fn,
    DeviceShapeSizeFn device_shape_size_fn,
    stream_executor::DeviceDescription device_description,
    std::unique_ptr<PjRtClient> client, HloPjRtTestBaseOptions options)
    : HloRunnerAgnosticTestBase(MakeHloRunnerPjRtAotAware(std::move(client)),
                                std::move(device_shape_representation_fn),
                                std::move(device_shape_size_fn),
                                BuildOptions(std::move(options))),
      device_description_(std::move(device_description)) {}

}  // namespace xla::gpu
