/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_UTIL_H_

#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
class WhileUtil {
 public:
  // Holds a return value from MakeInstructionsLiveIn.
  struct MakeInstructionsLiveInResult {
    // The new while operation that has the requested values live in.
    HloInstruction* new_while_instr;

    // The i'th element of `while_body_live_in_values` is an instruction in the
    // while body that holds the i'th *newly added* live in value at runtime.
    std::vector<HloInstruction*> while_body_live_in_values;

    // `while_body_instruction_map` maps instructions in the original while body
    // to the corresponding instructions in the body for the newly created while
    // operation.
    CallInliner::InlinedInstructionMap while_body_instruction_map;
  };

  // Replaces `while_instr` with a new while instruction that is equivalent to
  // `while_instr` except that it has all of the HLO instructions in
  // `instructions` as live-in, loop invariant values.  These new live in values
  // are represented as new elements appended to the parameter of the while
  // loop, which must be of tuple shape.  GetTupleElement instructions computing
  // each new live in value is returned in the `while_body_live_in_values`
  // vector.
  //
  // Deletes `while_instr` after replacing it.
  //
  // Preconditions:
  //
  //  `while_instr` must have a tuple shaped state.
  //
  //   Every instruction in `instructions` must be contained in the computation
  //   that contains `while_instr`.
  static StatusOr<MakeInstructionsLiveInResult> MakeInstructionsLiveIn(
      HloInstruction* while_instr,
      tensorflow::gtl::ArraySlice<HloInstruction*> instructions);

  using LoopStateTy = std::vector<HloInstruction*>;
  using LoopBodyGeneratorTy = std::function<StatusOr<LoopStateTy>(
      HloInstruction* /*induction_var*/,
      const LoopStateTy& /*current_values*/)>;

  // Creates a while loop in `computation` that runs for `trip_count`
  // iterations.  The structure of the while loop is as follows, in pseudocode:
  //
  //  loop_state while_loop() {
  //    indvar = 0;
  //    loop_state = init_values
  //    while (indvar < trip_count) {
  //      loop_state = loop_body_generator(loop_state)
  //      indvar++;
  //    }
  //    return loop_state;
  //  }
  static StatusOr<LoopStateTy> MakeCountedLoop(
      HloComputation* computation, int32 trip_count,
      const LoopStateTy& init_values,
      const LoopBodyGeneratorTy& loop_body_generator);

  // Returns the GetTupleElement instructions in `while_body` that access
  // elements in the parameter tuple that don't change across iterations.
  // Assumes `while_body` is the body computation of the while loop in question.
  static std::vector<HloInstruction*> GetInvariantGTEsForWhileBody(
      const HloComputation& while_body);
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_UTIL_H_
