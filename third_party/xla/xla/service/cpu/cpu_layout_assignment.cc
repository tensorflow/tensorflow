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

#include "xla/service/cpu/cpu_layout_assignment.h"

#include <cstdint>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/layout_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"

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
  if (operand->opcode() != HloOpcode::kConstant ||
      operand->shape().dimensions().size() <= 1) {
    return nullopt;
  }

  auto [it, inserted] = cache->try_emplace(operand, false);
  if (inserted) {
    it->second = ShouldMakeAllUsersColMajor(operand);
  }

  return it->second ? operand_idx : nullopt;
}

static bool OperandsAndResultMustHaveRowMajorLayout(
    const HloInstruction& instr,
    const TargetMachineFeatures& target_machine_features) {
  if (instr.opcode() == HloOpcode::kConvolution) {
    return PotentiallyImplementedAsEigenConvolution(instr,
                                                    target_machine_features);
  }
  if (instr.opcode() == HloOpcode::kDot) {
    return DotOperandsAndResultMustHaveRowMajorLayout(
        instr, target_machine_features, /*allow_runtime_calls=*/true);
  }
  if (instr.opcode() == HloOpcode::kCustomCall) {
    return instr.custom_call_target() == "TopK";
  }
  return false;
}

absl::Status CpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  ShouldMakeOperandColMajorCache cache;

  auto set_collective_dim_major_layout = [&](const HloInstruction* inst,
                                             int64_t dim) -> absl::Status {
    Shape shape = inst->shape();
    ShapeUtil::ForEachMutableLeafShape(
        &shape, [dim](Shape* subshape, const ShapeIndex& /*index*/) {
          if (!subshape->IsArray()) {
            return;
          }
          Layout layout =
              (!subshape->has_layout() ||
               subshape->layout().minor_to_major().empty())
                  ? LayoutUtil::GetDefaultLayoutForShape(*subshape)
                  : LayoutUtil::MakeLayout(subshape->layout().minor_to_major());
          *subshape->mutable_layout() =
              layout.minor_to_major().empty()
                  ? layout
                  : LayoutUtil::MoveDimToMajor(layout, dim);
        });
    ABSL_RETURN_IF_ERROR(SetInstructionLayout(shape, inst));
    if (shape.IsTuple()) {
      TF_RET_CHECK(shape.tuple_shapes().size() == inst->operand_count());
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        ABSL_RETURN_IF_ERROR(
            SetArrayOperandLayout(shape.tuple_shapes(i).layout(), inst, i));
      }
    } else {
      TF_RET_CHECK(inst->operand_count() == 1);
      ABSL_RETURN_IF_ERROR(SetArrayOperandLayout(shape.layout(), inst, 0));
    }
    return absl::OkStatus();
  };

  const HloComputation* computation = constraints->computation();
  for (auto* instruction : computation->instructions()) {
    if (OperandsAndResultMustHaveRowMajorLayout(*instruction,
                                                target_machine_features_)) {
      Shape row_major_shape = instruction->shape();
      LayoutUtil::ClearLayout(&row_major_shape);
      LayoutUtil::SetToDefaultLayout(&row_major_shape);
      ABSL_RETURN_IF_ERROR(SetInstructionLayout(row_major_shape, instruction));
    } else if (optional<int64_t> op_idx =
                   ShouldMakeOperandColumnMajor(&cache, *instruction)) {
      const HloInstruction* op = instruction->operand(*op_idx);
      ABSL_RETURN_IF_ERROR(SetArrayOperandLayout(
          LayoutUtil::MakeAscendingLayout(op->shape().dimensions().size()),
          instruction, *op_idx));
    } else if (instruction->opcode() == HloOpcode::kReduceScatter) {
      // XLA:CPU can only support reduce-scatter where the scatter dimension
      // is the most major dimension in the layout.
      auto ars = Cast<HloReduceScatterInstruction>(instruction);
      ABSL_RETURN_IF_ERROR(
          set_collective_dim_major_layout(ars, ars->scatter_dimension()));
      continue;
    } else if (instruction->opcode() == HloOpcode::kAllGather) {
      // XLA:CPU can only support all-gathers where the gather dimension is the
      // most major dimension in the layout.
      auto ag = Cast<HloAllGatherInstruction>(instruction);
      ABSL_RETURN_IF_ERROR(
          set_collective_dim_major_layout(ag, ag->all_gather_dimension()));
      continue;
    }
    for (int64_t operand_no = 0; operand_no < instruction->operand_count();
         ++operand_no) {
      const Shape& operand_shape = instruction->operand(operand_no)->shape();
      if (!operand_shape.IsArray() || operand_shape.dimensions().size() <= 1) {
        continue;
      }
      if (constraints->OperandLayout(instruction, operand_no) != nullptr) {
        continue;
      }
      if (AnyOperandBufferForwarded(instruction, operand_no)) {
        continue;
      }
      ABSL_RETURN_IF_ERROR(SetArrayOperandLayout(
          LayoutUtil::GetDefaultLayoutForShape(operand_shape), instruction,
          operand_no));
    }
  }
  return absl::OkStatus();
}
}  // namespace cpu
}  // namespace xla
