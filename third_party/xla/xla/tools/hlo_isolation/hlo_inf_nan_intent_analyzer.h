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

#ifndef XLA_TOOLS_HLO_ISOLATION_HLO_INF_NAN_INTENT_ANALYZER_H_
#define XLA_TOOLS_HLO_ISOLATION_HLO_INF_NAN_INTENT_ANALYZER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"

namespace xla {
namespace hlo_isolation {

// Returns true if the literal contains at least one infinite or NaN cell.
bool LiteralContainsInfOrNan(const LiteralSlice& literal);

// Returns true if any constant instruction in the module contains an Inf or
// NaN.
bool ModuleContainsConstantInfOrNan(const HloModule& module);

// Options controlling HLO Inf/NaN intent analysis.
struct InfNanIntentOptions {
  // If true, strictly rejects masked reductions when the computation contains
  // unconstrained operations with domain singularities (e.g. sqrt, log, div)
  // on parameters, as unconstrained inputs (such as legacy random sampling)
  // may trigger genuine NaN/Inf errors.
  //
  // When false (the default, intended for sane input generation with dataflow
  // constraint propagation enabled), parameter domains are guaranteed safe,
  // so operations like sqrt or log will not disqualify intentional masked
  // reductions.
  bool reject_unconstrained_ops = false;
};

// Analyzes an HLO module to determine whether any output Inf or NaN values
// are intentional (e.g., originating from attention masks, padding, or
// numerical guardrails) rather than genuine numerical errors.
//
// Returns true if the module contains constant Inf/NaN literals that flow
// forward to the root output instruction, or if intentional numerical
// guardrail / masked reduction idioms are detected.
bool IsIntentionalInfNan(const HloModule& module,
                         const InfNanIntentOptions& options = {});

}  // namespace hlo_isolation
}  // namespace xla

#endif  // XLA_TOOLS_HLO_ISOLATION_HLO_INF_NAN_INTENT_ANALYZER_H_
