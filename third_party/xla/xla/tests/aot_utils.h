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

#ifndef XLA_TESTS_AOT_UTILS_H_
#define XLA_TESTS_AOT_UTILS_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "xla/hlo/evaluator/hlo_evaluator_interface.h"
#include "xla/pjrt/interpreter/interpreter_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_runner_pjrt.h"

namespace xla {

// Constructs a HloRunnerPjRt depending on the value of env vars
// XLA_TEST_HLO_RUNNER_AOT_MODE and XLA_TEST_HLO_RUNNER_AOT_DIR
// If XLA_TEST_HLO_RUNNER_AOT_MODE is not set / set to "disabled", this
// function returns a standard HloRunnerPjRt.
std::unique_ptr<HloRunnerPjRt> MakeHloRunnerPjRtAotAware(
    std::unique_ptr<PjRtClient> client);

// Constructs an InterpreterClient depending on the value of env vars
// XLA_TEST_HLO_RUNNER_AOT_MODE and XLA_TEST_HLO_RUNNER_AOT_DIR
// If XLA_TEST_HLO_RUNNER_AOT_MODE is not set / set to "disabled", this
// function returns a standard InterpreterClient.
std::unique_ptr<InterpreterClient> MakeInterpreterClientAotAware(
    absl::AnyInvocable<std::unique_ptr<HloEvaluatorInterface>() const>
        hlo_evaluator_factory);

bool HasPjRtAotAwareSwallowExecutionErrors();
}  // namespace xla

#endif  // XLA_TESTS_AOT_UTILS_H_
