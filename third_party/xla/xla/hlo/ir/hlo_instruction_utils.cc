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

#include "xla/hlo/ir/hlo_instruction_utils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "re2/re2.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace hlo_instruction_utils {
bool IsUnstridedSlice(const HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kSlice) {
    return false;
  }
  return absl::c_all_of(hlo->slice_strides(),
                        [](int64_t stride) { return stride == 1; });
}

bool KeepsBitwidth(const HloInstruction& hlo) {
  CHECK(hlo.shape().IsArray());
  if (absl::c_any_of(hlo.operands(), [&](const HloInstruction* operand) {
        return primitive_util::BitWidth(operand->shape().element_type()) !=
               primitive_util::BitWidth(hlo.shape().element_type());
      })) {
    return false;
  }
  return true;
}

using Interval = std::pair<int64_t, int64_t>;
void AddOrUpdateVectorOfPairsAsAttribute(HloInstruction* instr,
                                         std::string attr_name,
                                         std::vector<Interval> intervals) {
  std::string intervals_str =
      "{" +
      absl::StrJoin(intervals, ",",
                    [](std::string* out, Interval item) {
                      absl::StrAppend(out, "{", item.first, ",", item.second,
                                      "}");
                    }) +
      "}";
  FrontendAttributes attributes;
  attributes = instr->frontend_attributes();
  (*attributes.mutable_map())[attr_name] = intervals_str;
  instr->set_frontend_attributes(attributes);
}

int32_t NestingDepth(const HloInstruction* hlo) {
  int level = 0;
  const HloComputation* c = hlo->parent();
  while (c != nullptr) {
    auto callers = c->caller_instructions();
    if (callers.empty()) {
      break;
    }
    // TODO(b/260601110): it's not clear what we should do if there are
    // multiple callers. For now, we just pick the first one.
    c = callers.front()->parent();
    ++level;
  }
  return level;
}

bool IsTopKStable(const HloCustomCallInstruction* inst) {
  static const LazyRE2 kUnstableRegex = {R"((?i)is_stable\s*=\s*false)"};
  if (RE2::PartialMatch(inst->raw_backend_config_string(), *kUnstableRegex)) {
    return false;
  }
  return true;
}

namespace async {
namespace {

const HloInstruction* GetAsyncChainStart(const HloInstruction* async_op) {
  const HloInstruction* current = async_op;
  while (current != nullptr && current->opcode() != HloOpcode::kAsyncStart) {
    if (current->opcode() != HloOpcode::kAsyncUpdate &&
        current->opcode() != HloOpcode::kAsyncDone) {
      return nullptr;
    }
    if (current->operand_count() == 0 || current->operand(0) == nullptr) {
      return nullptr;
    }
    current = current->operand(0);
  }
  return current;
}

absl::StatusOr<bool> AreOperandsAndOutputFullyBoundImpl(
    const HloInstruction* async_op, const Shape& expected_shape,
    const Shape& async_tuple_shape, const ShapeIndex& index) {
  if (index.empty()) {
    ASSIGN_OR_RETURN(bool operands_bound,
                     AreOperandsAndOutputFullyBoundImpl(
                         async_op, expected_shape, async_tuple_shape, {0}));
    ASSIGN_OR_RETURN(bool output_bound,
                     AreOperandsAndOutputFullyBoundImpl(
                         async_op, expected_shape, async_tuple_shape, {1}));
    return operands_bound && output_bound;
  }

  if (index.front() > 1 || !ShapeUtil::IndexIsValid(expected_shape, index)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid index: ", index.ToString(),
        ", index must start with 0 (for operands) or 1 (for output)."));
  }

  // Check operands
  if (index.front() == 0) {
    if (!ShapeUtil::IndexIsValid(async_tuple_shape, index)) {
      return false;
    }
    const Shape& expected_subshape =
        ShapeUtil::GetSubshape(expected_shape, index);
    const Shape& async_tuple_subshape =
        ShapeUtil::GetSubshape(async_tuple_shape, index);
    return ShapeUtil::Compatible(expected_subshape, async_tuple_subshape);
  }

  // Check output
  if (index.front() == 1) {
    const Shape& output_shape = (async_op->opcode() == HloOpcode::kAsyncDone)
                                    ? async_op->shape()
                                    : async_op->shape().tuple_shapes(1);
    ShapeIndex sub_index(index.begin() + 1, index.end());
    if (!ShapeUtil::IndexIsValid(output_shape, sub_index)) {
      return false;
    }
    const Shape& expected_subshape =
        ShapeUtil::GetSubshape(expected_shape, index);
    const Shape& async_tuple_subshape =
        ShapeUtil::GetSubshape(output_shape, sub_index);
    return ShapeUtil::Compatible(expected_subshape, async_tuple_subshape);
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> AreOperandsAndOutputFullyBound(
    const HloInstruction* async_op, const ShapeIndex& index) {
  if (!async_op->IsAsynchronous()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Instruction is not asynchronous: ", async_op->name()));
  }

  const HloInstruction* async_start = GetAsyncChainStart(async_op);
  if (async_start == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Async instruction ", async_op->name(),
        " is not part of a valid async chain starting with AsyncStart."));
  }

  if (async_start->called_computations().empty() ||
      async_start->called_computations().front() == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Async instruction ", async_op->name(),
                     " has no valid wrapped computation."));
  }
  HloComputation* async_wrapped_computation =
      async_start->called_computations().front();

  if (async_op->opcode() == HloOpcode::kAsyncDone) {
    if (async_op->operand_count() == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("AsyncDone instruction ", async_op->name(),
                       " does not have exactly one operand."));
    }
    if (async_op->operand(0) == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("AsyncDone instruction ", async_op->name(),
                       " does not have a valid operand."));
    }
  }
  const Shape& async_tuple_shape = (async_op->opcode() == HloOpcode::kAsyncDone)
                                       ? async_op->operand(0)->shape()
                                       : async_op->shape();
  if (!async_tuple_shape.IsTuple() ||
      async_tuple_shape.tuple_shapes().size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected async tuple shape to be a tuple with at least 2 shapes, "
        "got: ",
        async_tuple_shape.ToString()));
  }

  const ProgramShape called_computation_shape =
      async_wrapped_computation->ComputeProgramShape();
  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape(called_computation_shape.parameters()),
       called_computation_shape.result()});

  return AreOperandsAndOutputFullyBoundImpl(async_op, expected_shape,
                                            async_tuple_shape, index);
}

std::vector<const HloInstruction*> GetAsyncBoundOperands(
    const HloAsyncInstruction* async_op) {
  std::vector<const HloInstruction*> bound_operands;
  for (const HloInstruction* instr :
       Cast<HloAsyncStartInstruction>(async_op->async_chain_start())
           ->GetAsyncChain()) {
    int start_idx = (instr->opcode() == HloOpcode::kAsyncStart) ? 0 : 1;

    for (int i = start_idx; i < instr->operand_count(); ++i) {
      bound_operands.push_back(instr->operand(i));
    }
    if (instr == async_op) {
      break;
    }
  }

  return bound_operands;
}

absl::StatusOr<bool> IsFirstFullyBound(const HloInstruction* async_inst) {
  ASSIGN_OR_RETURN(bool fully_bound,
                   AreOperandsAndOutputFullyBound(async_inst));
  if (!fully_bound) {
    return false;
  }
  if (async_inst->opcode() == HloOpcode::kAsyncStart) {
    return true;
  }

  ASSIGN_OR_RETURN(bool prev_fully_bound,
                   AreOperandsAndOutputFullyBound(async_inst->operand(0)));
  return !prev_fully_bound;
}
}  // namespace async

}  // namespace hlo_instruction_utils
}  // namespace xla
