/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TESTS_HLO_RUNNER_PJRT_TEST_UTILS_H_
#define XLA_TESTS_HLO_RUNNER_PJRT_TEST_UTILS_H_

#include <memory>

#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"

namespace xla {

// Constructs a HloRunnerPjRt depending on the value of
// --xla_pjrt_split_phase_mode and --xla_pjrt_split_phase_dir. If
// --xla_pjrt_split_phase_mode is not set / set to "disabled", this function
// returns a standard HloRunnerPjRt.
std::unique_ptr<HloRunnerPjRt> MakeHloRunnerPjRtSplitPhaseAware(
    std::unique_ptr<PjRtClient> client,
    HloRunnerInterface::DeviceShapeRepresentationFn
        device_shape_representation_fn,
    HloRunnerInterface::DeviceShapeSizeFn device_shape_size_fn);
}  // namespace xla

#endif  // XLA_TESTS_HLO_RUNNER_PJRT_TEST_UTILS_H_
