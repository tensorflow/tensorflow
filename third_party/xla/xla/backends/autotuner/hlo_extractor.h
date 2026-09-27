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

#ifndef XLA_BACKENDS_AUTOTUNER_HLO_EXTRACTOR_H_
#define XLA_BACKENDS_AUTOTUNER_HLO_EXTRACTOR_H_

#include <functional>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

using EquivalentInstructions = std::vector<HloInstruction*>;
// TODO(b/519057668): Use absl::FunctionRef for filter_fn.
using InstructionFilterFn = std::function<bool(const xla::HloInstruction&)>;

// Returns a list of equivalent instructions. Each inner vector is a set of
// instructions that are equivalent as they have the same HLO fingerprint.
std::vector<EquivalentInstructions> ExtractEquivalentInstructions(
    const HloModule& module, const InstructionFilterFn& filter_fn);

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_HLO_EXTRACTOR_H_
