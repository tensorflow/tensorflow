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

#ifndef XLA_SERVICE_CPU_SMALL_WHILE_LOOP_HOISTING_PASS_H_
#define XLA_SERVICE_CPU_SMALL_WHILE_LOOP_HOISTING_PASS_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_cost_analysis.h"

namespace xla::cpu {

struct InstructionRun {
  std::vector<HloInstruction*> instructions;
  int64_t total_bytes_accessed = 0;
  bool contains_while_loop = false;
};

// Hoists small runs of HLO instructions (including while loops) into a separate
// function. This pass enables the thunk emitter to emit small instruction runs
// as a single kernel instead of a series of thunks.
class SmallWhileLoopHoistingPass final : public HloModulePass {
 public:
  explicit SmallWhileLoopHoistingPass(int64_t small_buffer_access_size);

  absl::string_view name() const final { return "small-while-loop-hoisting"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) final;

 private:
  std::vector<InstructionRun> IdentifyCandidateRuns(
      HloComputation* comp,
      absl::flat_hash_map<const HloInstruction*, bool>& unavailable_cache);

  bool IsBeneficialRun(const InstructionRun& run) const;

  absl::StatusOr<bool> OutlineRun(HloComputation* comp,
                                  const InstructionRun& run, HloModule* module);

 private:
  int64_t small_buffer_access_size_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_SMALL_WHILE_LOOP_HOISTING_PASS_H_
