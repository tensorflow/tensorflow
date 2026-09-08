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

#ifndef XLA_HLO_IR_HLO_INSTRUCTION_UTILS_H_
#define XLA_HLO_IR_HLO_INSTRUCTION_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla {
namespace hlo_instruction_utils {
// Returns true if the given HLO is a slice operation which has a unit stride in
// all dimensions.
bool IsUnstridedSlice(const HloInstruction* hlo);

// Checks that all instruction operands have the same bitwidth as its output.
bool KeepsBitwidth(const HloInstruction&);

// Adds or updates the attributes for an instruction. If the attribute is
// already present, then it is overwritten. Otherwise, this is added as another
// attribute.
void AddOrUpdateVectorOfPairsAsAttribute(
    HloInstruction* instr, std::string attr_name,
    std::vector<std::pair<int64_t, int64_t>> intervals);

// Returns the nesting depth in computations from the top-level computation of
// `hlo`. i.e. 0 = in the top-level computation, ...
int32_t NestingDepth(const HloInstruction* hlo);

// Checks if CustomCall TopK instruction is stable. Defaults to true.
bool IsTopKStable(const HloCustomCallInstruction* inst);

namespace async {

// Represents a step in the dataflow path.
template <typename HloInstructionT>
struct AsyncTraceStepT {
  HloInstructionT* instruction = nullptr;
  int64_t operand_index = 0;
};
using AsyncTraceStep = AsyncTraceStepT<HloInstruction>;

namespace async_detail {

template <typename HloInstructionT>
inline HloInstructionT* GetHloOperand(HloInstructionT* instr, int64_t i) {
  if constexpr (std::is_const_v<HloInstructionT>) {
    return instr->operand(i);
  } else {
    return instr->mutable_operand(i);
  }
}

template <typename HloInstructionT>
HloInstructionT* TraceAsyncDataflowImpl(
    HloInstructionT* instr,
    absl::FunctionRef<bool(const HloInstruction*)> is_target, ShapeIndex index,
    absl::flat_hash_set<const HloInstruction*>& visited,
    std::vector<AsyncTraceStepT<HloInstructionT>>* backward_steps) {
  if (instr == nullptr || !visited.insert(instr).second) {
    return nullptr;
  }

  if (is_target(instr)) {
    return instr;
  }

  auto record_step = [&](int64_t operand_idx) {
    if (backward_steps != nullptr) {
      backward_steps->push_back({instr, operand_idx});
    }
  };

  size_t steps_size = (backward_steps != nullptr) ? backward_steps->size() : 0;

  auto revert_step = [&]() {
    if (backward_steps != nullptr) {
      backward_steps->resize(steps_size);
    }
    visited.erase(instr);
  };

  switch (instr->opcode()) {
    case HloOpcode::kGetTupleElement: {
      record_step(0);
      index.push_back(instr->tuple_index());
      HloInstructionT* res = TraceAsyncDataflowImpl(
          GetHloOperand(instr, 0), is_target, index, visited, backward_steps);
      if (res != nullptr) {
        return res;
      }
      revert_step();
      return nullptr;
    }
    case HloOpcode::kTuple:
    case HloOpcode::kOptimizationBarrier: {
      if (instr->opcode() == HloOpcode::kOptimizationBarrier &&
          instr->operand_count() == 1) {
        record_step(0);
        HloInstructionT* res = TraceAsyncDataflowImpl(
            GetHloOperand(instr, 0), is_target, index, visited, backward_steps);
        if (res != nullptr) {
          return res;
        }
        revert_step();
        return nullptr;
      }
      if (index.empty()) {
        for (int64_t idx = 0; idx < instr->operand_count(); ++idx) {
          record_step(idx);
          HloInstructionT* res =
              TraceAsyncDataflowImpl(GetHloOperand(instr, idx), is_target,
                                     index, visited, backward_steps);
          if (res != nullptr) {
            return res;
          }
          if (backward_steps != nullptr) {
            backward_steps->resize(steps_size);
          }
        }
        revert_step();
        return nullptr;
      }
      int64_t idx = index.back();
      index.pop_back();
      if (idx >= 0 && idx < instr->operand_count()) {
        record_step(idx);
        HloInstructionT* res =
            TraceAsyncDataflowImpl(GetHloOperand(instr, idx), is_target, index,
                                   visited, backward_steps);
        if (res != nullptr) {
          return res;
        }
      }
      revert_step();
      return nullptr;
    }
    case HloOpcode::kCopy:
    case HloOpcode::kCustomCall: {
      if (instr->opcode() == HloOpcode::kCustomCall &&
          !instr->IsAllowedAsyncIntermediaryCustomCall()) {
        revert_step();
        return nullptr;
      }
      if (instr->operand_count() > 0) {
        record_step(0);
        HloInstructionT* res = TraceAsyncDataflowImpl(
            GetHloOperand(instr, 0), is_target, index, visited, backward_steps);
        if (res != nullptr) {
          return res;
        }
      }
      revert_step();
      return nullptr;
    }
    case HloOpcode::kParameter: {
      auto* comp = instr->parent();
      if (comp != nullptr) {
        auto callers = comp->caller_instructions(HloOpcode::kWhile);
        if (callers.size() == 1) {
          auto* while_op = callers.front();
          if (while_op->while_body() == comp) {
            record_step(0);
            HloInstructionT* res = TraceAsyncDataflowImpl(
                const_cast<HloInstructionT*>(GetHloOperand(while_op, 0)),
                is_target, index, visited, backward_steps);
            if (res != nullptr) {
              return res;
            }
            if (backward_steps != nullptr) {
              backward_steps->resize(steps_size);
            }
            if (comp->root_instruction() != nullptr) {
              HloInstructionT* root_res = TraceAsyncDataflowImpl(
                  const_cast<HloInstructionT*>(comp->root_instruction()),
                  is_target, index, visited, backward_steps);
              if (root_res != nullptr) {
                return root_res;
              }
            }
          }
        }
      }
      revert_step();
      return nullptr;
    }
    case HloOpcode::kWhile: {
      if (instr->while_body() != nullptr &&
          instr->while_body()->root_instruction() != nullptr) {
        HloInstructionT* root_res =
            TraceAsyncDataflowImpl(const_cast<HloInstructionT*>(
                                       instr->while_body()->root_instruction()),
                                   is_target, index, visited, backward_steps);
        if (root_res != nullptr) {
          return root_res;
        }
        if (backward_steps != nullptr) {
          backward_steps->resize(steps_size);
        }
      }
      if (instr->operand_count() > 0) {
        record_step(0);
        HloInstructionT* res = TraceAsyncDataflowImpl(
            const_cast<HloInstructionT*>(GetHloOperand(instr, 0)), is_target,
            index, visited, backward_steps);
        if (res != nullptr) {
          return res;
        }
      }
      revert_step();
      return nullptr;
    }
    default: {
      revert_step();
      return nullptr;
    }
  }
}

}  // namespace async_detail

// Traces backward from `instr` through allowed async intermediary
// instructions (tuples, get-tuple-elements, optimization barriers, copies,
// allowed async intermediary custom-calls) until `is_target` returns true.
//
// Returns the matched instruction, or nullptr if no path is found.
// If `backward_steps` is provided, records the sequence of (instruction,
// operand_index) steps taken from `instr` to the target in backward order.
template <typename HloInstructionT>
HloInstructionT* TraceAsyncDataflow(
    HloInstructionT* instr,
    absl::FunctionRef<bool(const HloInstruction*)> is_target,
    std::vector<AsyncTraceStepT<HloInstructionT>>* backward_steps = nullptr) {
  ShapeIndex index;
  absl::flat_hash_set<const HloInstruction*> visited;
  return async_detail::TraceAsyncDataflowImpl(instr, is_target, index, visited,
                                              backward_steps);
}

template <typename HloInstructionT>
HloInstructionT* TraceAsyncDataflow(
    HloInstructionT* instr, const HloInstruction* target,
    std::vector<AsyncTraceStepT<HloInstructionT>>* backward_steps = nullptr) {
  return TraceAsyncDataflow(
      instr,
      [target](const HloInstruction* candidate) { return candidate == target; },
      backward_steps);
}

// Traces backward from an instruction to find the immediate async producer
// (AsyncStart, AsyncUpdate, or legacy collective start) satisfying
// IsAsyncProducer(). In contrast to FindAsyncStart, which finds the root
// start, FindAsyncProducer returns the immediate producer, which can be an
// intermediate AsyncUpdate.
HloInstruction* FindAsyncProducer(HloInstruction* instr);
const HloInstruction* FindAsyncProducer(const HloInstruction* instr);

// Traces backward from an instruction to find the root async start
// (AsyncStart or legacy collective start) satisfying IsAsyncStart(),
// traversing any intermediate AsyncUpdate instructions.
HloInstruction* FindAsyncStart(HloInstruction* instr);
const HloInstruction* FindAsyncStart(const HloInstruction* instr);

// Traces forward from an instruction through allowed async intermediaries to
// find the immediate async consumer (AsyncUpdate, AsyncDone, or legacy
// collective done) satisfying IsAsyncConsumer(). In contrast to
// FindAsyncDone, which finds the terminal done op, FindAsyncConsumer returns
// the immediate consumer, which can be an intermediate AsyncUpdate.
HloInstruction* FindAsyncConsumer(HloInstruction* instr);
const HloInstruction* FindAsyncConsumer(const HloInstruction* instr);

// Traces forward from an instruction through allowed async intermediaries
// and intermediate AsyncUpdate instructions to find the terminal async
// done op (AsyncDone or legacy collective done) satisfying IsAsyncDone().
HloInstruction* FindAsyncDone(HloInstruction* instr);
const HloInstruction* FindAsyncDone(const HloInstruction* instr);

// Traces backward from `from` through allowed async intermediary instructions
// (tuples, get-tuple-elements, optimization barriers, copies, allowed async
// intermediary custom-calls) until matching `is_target`.
// Returns the matched instruction and the sequence of forward trace steps from
// that instruction to `from`, or std::nullopt if no path is found.
std::optional<std::pair<HloInstruction*, std::vector<AsyncTraceStep>>>
TraceDataflowPath(HloInstruction* from,
                  absl::FunctionRef<bool(const HloInstruction*)> is_target);

// Overload of TraceDataflowPath that traces from `from` to a specific `target`.
// Returns the forward sequence of trace steps to reach `from` from `target`,
// or std::nullopt if no path is found.
std::optional<std::vector<AsyncTraceStep>> TraceDataflowPath(
    HloInstruction* from, HloInstruction* target);

// Propagates a replacement value forward along the traced dataflow path,
// updating operand links and shapes of the intermediary nodes.
absl::StatusOr<HloInstruction*> PropagateDataflow(
    absl::Span<const AsyncTraceStep> forward_path, HloInstruction* source);

// Utilities for async instructions.

// Determines if the operands and output of the async
// instruction is fully bound at the given shape
// index, which is empty by default.
// Returns an error if the index is invalid, or index does not start with 0 or
// 1.
absl::StatusOr<bool> AreOperandsAndOutputFullyBound(
    const HloInstruction* async_op, const ShapeIndex& index = {});

// Returns true if the async-op is the first fully bound instruction in the
// async chain.
absl::StatusOr<bool> IsFirstFullyBound(const HloInstruction* async_inst);

// Returns all data operands accumulated in the async chain up to and including
// `async_op`.
// For `async-update` and `async-done`, the first operand (the chaining operand)
// is skipped.
//
// Example:
//   as = async-start(p0)
//   au1 = async-update(as, p1)
//   au2 = async-update(au1, p2)
//   ad = async-done(au2)
//
//   For `as`, it returns {p0}
//   For `au1`, it returns {p0, p1}
//   For `au2`, it returns {p0, p1, p2}
//   For `ad`, it returns {p0, p1, p2}
std::vector<const HloInstruction*> GetAsyncBoundOperands(
    const HloAsyncInstruction* async_op);
}  // namespace async

}  // namespace hlo_instruction_utils
}  // namespace xla

#endif  // XLA_HLO_IR_HLO_INSTRUCTION_UTILS_H_
