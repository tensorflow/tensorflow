/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_layout_assignment.h"

#include <numeric>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {
namespace cpu {

// We want to change the layout of constant arrays to be column major when all
// of their users are dot operations that can be made faster with the flipped
// layout.  To avoid going quadratic over the # of instructions, we cache this
// property in should_make_rhs_col_major -- it maps a constant to true if all of
// the users of said constant are dot operations that can be sped up.  This
// cache is populated lazily as we encounter dot operations traversing the
// instruction stream.

namespace {
using std::nullopt;
using std::optional;

using ShouldMakeOperandColMajorCache =
    absl::flat_hash_map<const HloInstruction*, bool>;
}  // namespace

static bool ShouldMakeAllUsersColMajor(const HloInstruction* instruction) {
  for (auto* user : instruction->users()) {
    optional<int64_t> operand_idx =
        ProfitableToMakeDotOperandColumnMajor(*user);
    if (!operand_idx || user->operand(*operand_idx) != instruction ||
        absl::c_count(user->operands(), instruction) != 1) {
      return false;
    }
  }
  return true;
}

static optional<int64_t> ShouldMakeOperandColumnMajor(
    ShouldMakeOperandColMajorCache* cache, const HloInstruction& instruction) {
  optional<int64_t> operand_idx =
      ProfitableToMakeDotOperandColumnMajor(instruction);
  if (!operand_idx) {
    return nullopt;
  }

  const HloInstruction* operand = instruction.operand(*operand_idx);
  if (operand->opcode() != HloOpcode::kConstant) {
    return nullopt;
  }

  auto it = cache->find(operand);
  if (it == cache->end()) {
    auto insert_result =
        cache->insert({operand, ShouldMakeAllUsersColMajor(operand)});
    CHECK(insert_result.second);
    it = insert_result.first;
  }

  return it->second ? operand_idx : nullopt;
}

static Shape RowMajorShape(const Shape& old_shape) {
  Shape new_shape(old_shape);
  std::vector<int64_t> dimension_order(new_shape.dimensions_size());
  std::iota(dimension_order.rbegin(), dimension_order.rend(), 0);
  *new_shape.mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
  return new_shape;
}

static Shape ColMajorShape(const Shape& old_shape) {
  Shape new_shape(old_shape);
  std::vector<int64_t> dimension_order(new_shape.dimensions_size());
  std::iota(dimension_order.begin(), dimension_order.end(), 0);
  *new_shape.mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
  return new_shape;
}

static bool OperandsAndResultMustHaveRowMajorLayout(
    const HloInstruction& instr,
    const TargetMachineFeatures& target_machine_features) {
  if (instr.opcode() == HloOpcode::kConvolution) {
    return PotentiallyImplementedAsEigenConvolution(instr,
                                                    target_machine_features);
  } else if (instr.opcode() == HloOpcode::kDot) {
    return DotOperandsAndResultMustHaveRowMajorLayout(instr,
                                                      target_machine_features);
  }
  return false;
}

Status CpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  ShouldMakeOperandColMajorCache cache;

  const HloComputation* computation = constraints->computation();
  for (auto* instruction : computation->instructions()) {
    if (OperandsAndResultMustHaveRowMajorLayout(*instruction,
                                                target_machine_features_)) {
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          RowMajorShape(instruction->shape()), instruction));
      for (int i = 0; i < instruction->operand_count(); i++) {
        TF_RETURN_IF_ERROR(SetOperandLayout(
            RowMajorShape(instruction->operand(i)->shape()), instruction, i));
      }
    } else if (optional<int64_t> op_idx =
                   ShouldMakeOperandColumnMajor(&cache, *instruction)) {
      const HloInstruction* op = instruction->operand(*op_idx);
      TF_RETURN_IF_ERROR(
          SetOperandLayout(ColMajorShape(op->shape()), instruction, *op_idx));
    } else {
      for (int64_t operand_no = 0; operand_no < instruction->operand_count();
           ++operand_no) {
        // Skip operands which already have a constraint.
        if (constraints->OperandLayout(instruction, operand_no) != nullptr) {
          continue;
        }
        // Skip over forwarded operands.
        if (AnyOperandBufferForwarded(instruction, operand_no)) {
          continue;
        }
        // Skip operands with non-array shapes.
        if (!instruction->operand(operand_no)->shape().IsArray()) {
          continue;
        }
        Shape operand_shape(
            RowMajorShape(instruction->operand(operand_no)->shape()));
        TF_RETURN_IF_ERROR(
            SetOperandLayout(operand_shape, instruction, operand_no));
      }
      // Skip over the root instruction for the top-level computation.
      if (computation->parent()->entry_computation() == computation &&
          computation->root_instruction() == instruction) {
        continue;
      }
      // Skip instructions which don't produce array shapes (tuples, opaque,
      // etc.).
      if (!instruction->shape().IsArray()) {
        continue;
      }
    }
  }
  return OkStatus();
}
}  // namespace cpu
}  // namespace xla
