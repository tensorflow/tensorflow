/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_HLO_UTILS_HLO_QUERY_H_
#define XLA_HLO_UTILS_HLO_QUERY_H_

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"

namespace xla {

// Helper interface for making queries about the HLO IR.
namespace hlo_query {

// Returns whether the given opcode is a collective communications operation
// that is represented as HloCollectiveInstruction.
//
// Do not rely on this to detect any async computation. In particular wrapped
// async op `kCall` is not considered an async collective, even if it is
// wrapping `kAsyncStart` or `kAsyncDone` ops.
bool IsCollectiveCommunicationOp(HloOpcode op);

// Returns whether the given instruction represents the start operation for a
// collective communication, may include send & recv operations.
// Do not rely on this to detect any async computation. See caveats in
// `IsCollectiveCommunicationOp`.
bool IsAsyncCollectiveStartOp(const HloInstruction* instruction,
                              bool include_send_recv = false);
// Returns whether the given instruction represents the done operation for a
// collective communication, may include send & recv operations.
// Do not rely on this to detect any async computation. See caveats in
// `IsCollectiveCommunicationOp`.
bool IsAsyncCollectiveDoneOp(const HloInstruction* instruction,
                             bool include_send_recv = false);

// Returns whether the instruction provided is a constant rank-0 float32, and
// if so, places the constant value into out.
// Precondition: out != nullptr
bool IsConstantR0F32(HloInstruction* instruction, float* out);

// Returns whether all of an instruction's operands are of the types constants
// and parameters.
bool AllOperandsAreParametersOrConstants(const HloInstruction& instruction);

// Returns whether all of an instruction's operands are of the type constant
// or parameter and the instruction is their only user.
bool AllOperandsAreParametersOrConstantsWithSingleUser(
    const HloInstruction& instruction);

// Returns whether all of an instruction's operands are parameters.
bool AllOperandsAreParameters(const HloInstruction& instruction);

// Returns whether all of an instruction's operands are constants.
bool AllOperandsAreConstants(const HloInstruction& instruction);

// Returns whether the instruction is a scalar constant.
bool IsScalarConstant(const HloInstruction* instruction);

// Returns whether the `instr` is either a constant, a scalar, or a
// broadcasted constant/scalar.
bool IsBroadcastedConstantOrScalar(const HloInstruction& instr);

// Returns whether the `instr` is a broadcast and its input is a
// scalar constant.
bool IsBroadcastOfScalarConstant(const HloInstruction& instr);

// Returns whether the `instr` is a broadcast and its input is a parameter.
bool IsBroadcastOfParameter(const HloInstruction& instr);

// Returns true for a parameter or a parameter followed by a chain of no-op
// instructions (bitcast, get-tuple-element).
bool IsEffectiveParameter(const HloInstruction&);

// Returns first HLO of the computation with the opcode, otherwise nullptr.
HloInstruction* GetFirstInstructionWithOpcode(const HloComputation& computation,
                                              HloOpcode opcode);

// Applies `fn` to a collection of instruction with `opcode` for a given
// `computation`.
template <typename Fn>
void ForEachInstructionWithOpcode(const HloComputation& computation,
                                  HloOpcode opcode, Fn&& fn) {
  for (HloInstruction* instr : computation.instructions()) {
    if (instr->opcode() == opcode) {
      fn(instr);
    }
  }
}

// Applies `fn` to a collection of instruction with `opcode` for a given
// `module`.
template <typename Fn>
void ForEachInstructionWithOpcode(const HloModule& module, HloOpcode opcode,
                                  Fn&& fn) {
  for (HloComputation* computation : module.computations()) {
    ForEachInstructionWithOpcode(*computation, opcode, fn);
  }
}

// Applies `fn` to a collection of instruction satisfying `pred` for a given
// `computation`.
template <typename Fn>
void ForEachInstructionWithPred(const HloComputation& computation,
                                HloPredicate pred, Fn&& fn) {
  for (HloInstruction* instr : computation.instructions()) {
    if (pred(instr)) {
      fn(instr);
    }
  }
}

// Applies `fn` to a collection of instruction satisfying `pred` for a given
// `module`.
template <typename Fn>
void ForEachInstructionWithPred(const HloModule& module, HloPredicate pred,
                                Fn&& fn) {
  for (HloComputation* computation : module.computations()) {
    ForEachInstructionWithPred(*computation, pred, fn);
  }
}

// Determines whether the given computation contains an instruction with one of
// the given opcodes.  Checks both comp's instructions and the instructions of
// any computations nested within it.
bool ContainsInstrWithOpcode(const HloComputation* comp,
                             const absl::flat_hash_set<HloOpcode>& opcodes);

// Returns an operand of an instruction with the given opcode. If there are
// multiple matching operands, then the first matching operand is returned. If
// there are no matching operands then nullptr is returned.
HloInstruction* GetMatchingOperand(const HloPredicate& matcher,
                                   HloInstruction* instruction);

// Returns whether a binary instruction has a matching operand. Sets
// matching_operand to the matching operand and the other operand to
// other_operand. Note: in the case where both operands match, the first operand
// of the instruction is returned.
bool MatchBinaryInstructionOperand(const HloPredicate& matcher,
                                   HloInstruction* instruction,
                                   HloInstruction** matching_operand,
                                   HloInstruction** other_operand);

// Returns whether a binary instruction has a operand with a given opcode.
// This is a special case of MatchingBinaryInstructionOperand.
bool MatchBinaryInstructionOperandOpcode(HloOpcode opcode,
                                         HloInstruction* instruction,
                                         HloInstruction** matching_operand,
                                         HloInstruction** other_operand);

// Returns whether the module contains the given collective communication
// instructions with constrained layout.
bool ContainsLayoutConstrainedCollective(const HloModule& module, HloOpcode op);

// Returns whether the module contains all-reduce instructions with constrained
// layout.
inline bool ContainsLayoutConstrainedAllReduce(const HloModule& module) {
  return ContainsLayoutConstrainedCollective(module, HloOpcode::kAllReduce);
}

// Returns the next available channel id that can be used in the given module
// (for HloChannelInstructions).
int64_t NextChannelId(const HloModule& module);

// Returns whether the module contains host send/recv with X64 data type.
// This function is called after X64Rewriter, so X64 host transfers are already
// rewritten into tuple shaped transfers.
bool HasX64TransformedHostTransfer(const HloModule& module);

// Returns the unique GTE instruction with the given operand and index. Returns
// nullptr if no such instruction exists or is not unique.
HloInstruction* GetUniqueGteInstruction(const HloInstruction* operand,
                                        int64_t index);

// Returns the number of GTE instructions with the given index.
int64_t CountGteInstructionsWithIndex(const HloComputation* computation,
                                      int64_t index);

// Gets the computation from the given module with the given name.
HloComputation* FindComputation(HloModule* module, absl::string_view name);

// Gets the instruction from the given computation with the given instruction
// name. Returns nullptr if no such instruction can be found.
HloInstruction* FindInstruction(const HloComputation* computation,
                                absl::string_view name);

// Gets any instruction from the given computation with the given opcode.
// Returns nullptr if no such instruction can be found.
HloInstruction* FindInstruction(const HloComputation* computation,
                                HloOpcode opcode);

}  // namespace hlo_query
}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_QUERY_H_
