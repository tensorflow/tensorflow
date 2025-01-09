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
#include "absl/status/statusor.h"
#include "xla/pjrt/interpreter/interpreter_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace {

std::unique_ptr<HloRunnerInterface> GetHloRunnerForTest() {
  CHECK(ShouldUsePjRt())
      << "PjRt is required for tests extending HloPjRtTestBase.";

  PjRtClientTestFactoryRegistry& pjrt_registry =
      GetGlobalPjRtClientTestFactory();
  absl::StatusOr<std::unique_ptr<PjRtClient>> client = pjrt_registry.Get()();
  CHECK_OK(client.status())
      << "Failed to create PjRt client for test. " << client.status();
  PjRtClientTestFactoryRegistry::DeviceShapeRepresentationFn
      device_shape_representation_fn =
          pjrt_registry.GetDeviceShapeRepresentationFn(client->get());
  PjRtClientTestFactoryRegistry::DeviceShapeSizeFn device_shape_size_fn =
      pjrt_registry.GetDeviceShapeSizeFn(client->get());

  return std::make_unique<HloRunnerPjRt>(
      *std::move(client), device_shape_representation_fn, device_shape_size_fn);
}

std::unique_ptr<HloRunnerInterface> GetHloRunnerForReference() {
  return std::make_unique<HloRunnerPjRt>(
      std::make_unique<InterpreterClient>(),
      InterpreterClient::DeviceShapeRepresentation,
      InterpreterClient::ShapeSizeBytes,
      /*use_parameter_layout_on_device=*/true);
}

}  // namespace

HloPjRtTestBase::HloPjRtTestBase(HloPjRtTestBaseOptions options)
    : HloRunnerAgnosticTestBase(GetHloRunnerForTest(),
                                GetHloRunnerForReference(),
                                options.verifier_layout_sensitive,
                                options.allow_mixed_precision_in_hlo_verifier,
                                options.instruction_can_change_layout_func) {}

}  // namespace xla
