/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_ROOT_INSTRUCTION_SINKER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_ROOT_INSTRUCTION_SINKER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Given a scheduled HLO module, this pass sinks the ROOT of the instruction to
// the bottom of the non-fusion computations. To avoid dependency violations of
// moving the ROOT instruction, it creates a new ROOT instruction that looks
// like the following:
//   - For tuple ROOT type:
//        new_root = tuple(gte(old_root), gte(old_root), ...)
//   - For non-tuple ROOT type:
//        new_root = bitcast(old_root)
class RootInstructionSinker : public HloModulePass {
 public:
  ~RootInstructionSinker() override = default;
  absl::string_view name() const override { return "root-instruction-sinker"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_ROOT_INSTRUCTION_SINKER_H_
