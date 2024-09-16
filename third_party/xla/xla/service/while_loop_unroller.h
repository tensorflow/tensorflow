/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_WHILE_LOOP_UNROLLER_H_
#define XLA_SERVICE_WHILE_LOOP_UNROLLER_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Config for unroll thresholds.
struct UnrollConfig {
  int64_t trip_count_threshold = 64;
  int64_t instruction_count_threshold = 800;
  int64_t expand_factor_threshold = 10000;
};

// Config for unrollable while loops.
struct WhileLoopConfig {
  const HloInstruction* while_instr;
  // The initial value of the induction variable of the while loop.
  int64_t init;
  // The number of iterations the loop executes.
  int64_t trip_count;
  // The index of the induction variable in the input tuple of the while loop.
  int64_t induction_var_idx;
};

// Config for unrollable while loops.
struct UnrollResult {
  // Whether it's unrolled.
  bool unrolled = false;
  // If the while-loop has been unrolled and replaced with a new unrolled
  // while-loop with a trip count of 1, this is the new while-loop.
  // Otherwise this is nullptr.
  HloInstruction* new_while_op = nullptr;
};

// Check if `instr` is a dynamic index instruction, i.e., dynamic-slice or
// dynamic-update-slice with the given input that operates on the entire
// shape of the instruction. To satisfy this:
// 1. All start indices must be constant zero except only a single dimension.
// 2. The start index of that dimension should be equal to the enclosing loop
//    induction variable.
// 3. And, the size of that dimension must match the loop trip count.
// If so, it returns the dynamic index.
std::optional<int64_t> MatchShapeCoveringDynamicIndexInstruction(
    const HloInstruction* instr, const HloInstruction* input, HloOpcode opcode,
    const WhileLoopConfig& config);

// Check if `instr` is a dynamic-slice with the given input and a single dynamic
// start index that is effectively static, i.e., it is an expression that only
// involves the iteration variable of the surrounding loop and some constants,
// if we unroll the surrounding loop. If so, it returns the dynamic index.
std::optional<int64_t> MatchEffectivelyStaticDynamicSliceInsideLoop(
    const HloInstruction* instr, const HloInstruction* input,
    const WhileLoopConfig& config);

// This pass unrolls while loops with the given unrolling factor. The value of
// unroll_factor = -1 will fully unroll the loop.
//
// TODO(b/288130138): Currently, we `only` support full unrolling.
//
// The trip count for loops is calculated based on
// `MatchTrivialLoopTripCount` function in
// tensorflow/compiler/xla/service/while_loop_analysis.h`
//
// TODO(b/301472793): Add utility functions to unroll specific loops.
class WhileLoopUnroller : public HloModulePass {
 public:
  ~WhileLoopUnroller() override = default;

  // Default unroll_factor of -1 indicates full unrolling
  explicit WhileLoopUnroller(int64_t unroll_factor = -1,
                             bool wrap_in_trivial_loop = false,
                             UnrollConfig config = UnrollConfig())
      : unroll_factor_(unroll_factor),
        wrap_in_trivial_loop_(wrap_in_trivial_loop),
        unroll_config_(config) {}

  absl::string_view name() const override { return "while_loop_unroller"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Runs a sequence of passes that are necessary to prepare loops for
  // unrolling. Failure to run these passes will prevent unroller from unrolling
  // loops that would have been otherwise unrollable.
  static absl::StatusOr<bool> PrepareModuleForUnrolling(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Function that decides whether a loop is unrollable or not and returns the
  // loop config.
  static std::optional<WhileLoopConfig> IsLoopUnrollable(
      HloInstruction* while_op);

  // Returns the list of unrollable loops in the given module
  static std::vector<std::pair<HloInstruction*, WhileLoopConfig>>
  GetUnrollableLoops(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      const UnrollConfig& unroll_config = UnrollConfig());

  // Unrolls the given while loop with the default behaviour set to full unroll.
  // If wrap_in_trivial_loop is set, the unrolled body of the loop will be
  // wrapped in a loop with trip count of one. Forcing unroll will not perform
  // soft checking of the conditions. If prepare is set, it will run the
  // necessary passes to prepare the module for unrolling. Returns the unrolled
  // flag and the new unrolled while instruction.
  static absl::StatusOr<UnrollResult> UnrollAndReturnReplacement(
      HloInstruction* while_op, int64_t unroll_factor = -1,
      bool wrap_in_trivial_loop = false, bool force_unroll = false,
      bool prepare = true, const UnrollConfig& unroll_config = UnrollConfig());

 private:
  int64_t unroll_factor_;
  // Whether to wrap the unrolled computation in a loop with trip count of one.
  bool wrap_in_trivial_loop_;
  UnrollConfig unroll_config_;
};

}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_UNROLLER_H_
