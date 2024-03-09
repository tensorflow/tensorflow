/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_TOOLS_HLO_EXTRACTOR_H_
#define XLA_TOOLS_HLO_EXTRACTOR_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Define ExtractSelector, which is a lambda that, given an HLO
// instruction, returns true if selected, otherwise return false.
using ExtractSelector = std::function<bool(const HloInstruction*)>;

// Define ReplaceTypeSelector, which is a lambda that, given an HLO
// instruction, returns ReplaceType, which indicated which type of op should be
// used to replace.
//
// kReplaceParam: hlo instruction will be replaced with parameter. Note that it
// can only replace the instructions at the entry computation with parameters.
// If `cross_computation` is enabled and users attempt to replace an instruction
// in non-entry computation with a parameter, this library would report FATAL.
//
// kReplaceConst: hlo instruction will be replaced with randomly-generated
// constant of the same shape. Note that it could be very slow if hlo
// instruction has a large shape. It can be used in both entry and non-entry
// computation.
//
// kReplaceZeroBroadcast: hlo instruction will be replaced with a broadcasted
// zero constant of the same shape. It can be used in both entry and non-entry
// computation.
//
// kReplaceRandomBroadcast: hlo instruction will be replaced with a broadcasted
// random constant of the same shape. It can be used in both entry and non-entry
// computation.
enum class ReplaceType {
  kReplaceParam,
  kReplaceConst,
  kReplaceZeroBroadcast,
  kReplaceRandomBroadcast
};
using ReplaceTypeSelector = std::function<ReplaceType(const HloInstruction*)>;

// Creates a new HLO module rooted with an entry computation rooted at the given
// instruction.
//
// By default (height == -1), the new computation includes all transitive
// operands of `root`.  If you specify a different height, the new computation
// will include all instructions <= `height` hops away from `root`.
// Instructions at the boundary are replaced by parameters.
//
// The `extractor_selector` will return true/false for each hlo instruction. If
// false is returned, the corresponding instruction and its predecessors will
// not be included in the extracted hlo module
//
// The `replace_type_selector` specify, if an HLO instruction is determined to
// be excluded, which type of node should be the replacement. Please check the
// comments for ReplaceTypeSelector for details.
//
// If the `cross_computation` is enabled, we would be capable of visiting the
// instructions at the non-entry computations and exclude/replace some
// instructions there.
// There are two restrictions if this option is enabled:
//   1. `height` must be -1, as we do not support calculating boundary across
//   computations.
//   2. We do not support replace an instruction at non-entry computation with
//   parameter.
// Please check test cases `HloExtractorTest.ExtractFromMultipleComputation` for
// more details.
std::unique_ptr<HloModule> ExtractModule(
    const HloInstruction* instruction, int64_t height = -1,
    ExtractSelector extract_selector = nullptr,
    ReplaceTypeSelector replace_type_selector = nullptr,
    bool cross_computation = false);

}  // namespace xla

#endif  // XLA_TOOLS_HLO_EXTRACTOR_H_
