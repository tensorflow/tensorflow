/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tests/hlo_pjrt_test_base.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tests/aot_utils.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/pjrt_client_registry.h"

namespace xla {
namespace {
std::unique_ptr<PjRtClient> GetPjRtClientForTest() {
  CHECK(ShouldUsePjRt())
      << "PjRt is required for tests extending HloPjRtTestBase.";
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
}  // namespace

HloPjRtTestBase::HloPjRtTestBase(HloPjRtTestBaseOptions options)
    : HloPjRtTestBase(GetPjRtClientForTest().release(), std::move(options)) {}

HloPjRtTestBase::HloPjRtTestBase(PjRtClient* client,
                                 HloPjRtTestBaseOptions options)
    : HloPjRtTestBase(
          GetGlobalPjRtClientTestFactory().GetDeviceShapeRepresentationFn(
              client),
          GetGlobalPjRtClientTestFactory().GetDeviceShapeSizeFn(client),
          absl::WrapUnique(client), std::move(options)) {}

HloPjRtTestBase::HloPjRtTestBase(
    DeviceShapeRepresentationFn device_shape_representation_fn,
    DeviceShapeSizeFn device_shape_size_fn, std::unique_ptr<PjRtClient> client,
    HloPjRtTestBaseOptions options)
    : HloRunnerAgnosticTestBase(MakeHloRunnerPjRtAotAware(std::move(client)),
                                std::move(device_shape_representation_fn),
                                std::move(device_shape_size_fn),
                                BuildOptions(std::move(options))) {}

}  // namespace xla
