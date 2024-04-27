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

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/op_expander_pass.h"
#include "xla/statusor.h"

#ifndef XLA_SERVICE_BITCAST_DTYPES_EXPANDER_H_
#define XLA_SERVICE_BITCAST_DTYPES_EXPANDER_H_

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

#endif  // XLA_SERVICE_BITCAST_DTYPES_EXPANDER_H_
