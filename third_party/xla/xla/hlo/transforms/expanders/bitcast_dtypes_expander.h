/* Copyright 2021 The OpenXLA Authors.

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

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_BITCAST_DTYPES_EXPANDER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_BITCAST_DTYPES_EXPANDER_H_

namespace xla {

// A pass which expands bitcast-convert between differently sized dtypes to a
// reduction.
class BitcastDtypesExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "bitcast_dtypes_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

 private:
  absl::flat_hash_map<std::string, HloComputation*> computation_cache_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_BITCAST_DTYPES_EXPANDER_H_
