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

#ifndef XLA_TESTS_HLO_INTERPRETER_REFERENCE_MIXIN_H_
#define XLA_TESTS_HLO_INTERPRETER_REFERENCE_MIXIN_H_

#include <memory>

#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/service/hlo_runner.h"
#include "xla/tests/aot_utils.h"
#include "xla/tests/hlo_runner_agnostic_reference_mixin.h"

namespace xla {

// A wrapper mixin around HloRunnerAgnosticReferenceMixin which provides a
// default reference backend via HloRunnerPjRt using the PjRt InterpreterClient.
//
// The mixin requires that that the test class is a subclass of
// HloRunnerAgnosticTestBase.
//
// === DO NOT ADD FUNCTIONS TO THIS MIXIN. ===
// Functions interacting with the reference backend should be added to
// HloRunnerAgnosticReferenceMixin. Functions interacting with the test backend
// should be added to T.
template <typename T>
class HloInterpreterReferenceMixin : public HloRunnerAgnosticReferenceMixin<T> {
 protected:
  template <typename... BaseArgs>
  explicit HloInterpreterReferenceMixin(BaseArgs&&... base_args)
      : HloRunnerAgnosticReferenceMixin<T>(
            std::make_unique<HloRunner>(MakeAotAwareInterpreterClient(
                []() { return std::make_unique<HloEvaluator>(); })),
            std::forward<BaseArgs>(base_args)...) {}
  ~HloInterpreterReferenceMixin() override = default;
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_INTERPRETER_REFERENCE_MIXIN_H_
