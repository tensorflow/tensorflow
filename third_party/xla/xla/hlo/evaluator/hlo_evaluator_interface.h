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

#ifndef XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERFACE_H_
#define XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERFACE_H_

#include <functional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/service/dynamic_dimension_inference.h"

namespace xla {

// An interface class for HloEvaluator. This captures a minimal set of methods
// that are needed by e.g. the InterpreterClient. This interface allows us to
// create decorator implementations.
class HloEvaluatorInterface {
 public:
  // Handles evaluation of a custom-call op.
  // Operand literals are provided in |operands| and implementations must
  // populate |output| before returning.
  using CustomCallHandler = std::function<absl::StatusOr<Literal>(
      const HloInstruction* custom_call, absl::Span<const Literal*> operands)>;

  virtual ~HloEvaluatorInterface() = default;

  virtual absl::StatusOr<Literal> Evaluate(
      const HloComputation& computation,
      absl::Span<const Literal* const> args) = 0;

  // Should be called before Evaluate() is called again to reset the internal
  // visit state map.
  virtual void ResetVisitStates() = 0;

  virtual void set_dynamic_dimension_inference(
      DynamicDimensionInference* dynamic_dimension_inference) = 0;

  // Enable the fast path for certain operations like dot or convolution.
  virtual void set_use_fast_path(bool value) = 0;

  // Sets a handler that is called during evaluation for custom-call ops.
  // If no handler is defined the default error behavior will occur. The handler
  // will be provided evaluated literals for all operands and is expected to
  // return an output literal of the appropriate shape.
  virtual void set_custom_call_handler(CustomCallHandler handler) = 0;
};
}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERFACE_H_
