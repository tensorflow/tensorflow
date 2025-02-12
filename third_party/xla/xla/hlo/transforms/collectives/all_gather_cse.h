/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_CSE_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_CSE_H_

#include <cstdint>
#include <tuple>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

//  This pass performs common subexpression elimination (CSE) on all-gathers
//  of parameters. It serves as a setup pass for more advanced collective
//  transformation strategies by ensuring there is only one all-gather per
//  parameter. This enables subsequent passes to perform operations like
//  reinserting all-gathers or all-gather code motion. Example:
//
//  Before the pass:
//  while_loop {
//      all-gather.1 = all-gather(param_0)
//      some_computation.1 = compute(all-gather.1)
//      all-gather.2 = all-gather(param_0)
//      some_computation.2 = compute(all-gather.2)
//  }
//
//  After the pass:
//  while_loop {
//      all-gather.0 = all-gather(param_0)
//      some_computation.1 = compute(all-gather.0)
//      some_computation.2 = compute(all-gather.0)
//  }
class AllGatherCSE : public HloModulePass {
 public:
  absl::string_view name() const override { return "all-gather-cse"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::tuple<HloInstruction*, int64_t, PrimitiveType> FindRawParameter(
      HloInstruction* instruction);
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_CSE_H_
