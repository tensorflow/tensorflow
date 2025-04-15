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

#ifndef XLA_TESTS_HLO_PJRT_INTERPRETER_REFERENCE_MIXIN_H_
#define XLA_TESTS_HLO_PJRT_INTERPRETER_REFERENCE_MIXIN_H_

#include <memory>

#include "xla/pjrt/interpreter/interpreter_client.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/tests/hlo_runner_agnostic_reference_mixin.h"

namespace xla {

// A wrapper mixin around HloRunnerAgnosticReferenceMixin which provides a
// default reference backend via HloRunnerPjRt using the PjRt InterpreterClient.
//
// The mixin requires that that the test class is a subclass of
// HloRunnerAgnosticTestBase.
template <typename T>
class HloPjRtInterpreterReferenceMixin
    : public HloRunnerAgnosticReferenceMixin<T> {
 protected:
  template <typename... BaseArgs>
  explicit HloPjRtInterpreterReferenceMixin(BaseArgs&&... base_args)
      : HloRunnerAgnosticReferenceMixin<T>(
            std::make_unique<HloRunnerPjRt>(
                std::make_unique<InterpreterClient>(),
                InterpreterClient::DeviceShapeRepresentation,
                InterpreterClient::ShapeSizeBytes),
            std::forward<BaseArgs>(base_args)...) {}
  ~HloPjRtInterpreterReferenceMixin() override = default;
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_PJRT_INTERPRETER_REFERENCE_MIXIN_H_
