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

#include "xla/hlo/transforms/memory_space_propagation.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

constexpr HloCallBoundaryOptions kFusionBoundaryOptions(
    /*include_calls_in=*/false, /*include_control_flow_in=*/false,
    /*include_fusions_in=*/true);

}  // namespace

bool MemorySpacePropagation::RunOnComputation(HloComputation* computation) {
  CHECK(dataflow_analysis_ != nullptr);
  bool modified = false;
  // Propagate the parameter subshapes.
  for (int parameter_idx = 0; parameter_idx < computation->num_parameters();
       ++parameter_idx) {
    ShapeUtil::ForEachLeafShape(
        computation->parameter_instruction(parameter_idx)->shape(),
        [&](const Shape& sub_shape, const ShapeIndex& index) {
          absl::flat_hash_set<const HloValue*> visited;
          modified |= Propagate(
              index, computation->parameter_instruction(parameter_idx),
              sub_shape, visited);
        });
  }
  // Propagate output subshapes.
  ShapeUtil::ForEachLeafShape(
      computation->root_instruction()->shape(),
      [&](const Shape& sub_shape, const ShapeIndex& index) {
        absl::flat_hash_set<const HloValue*> visited;
        modified |= Propagate(index, computation->root_instruction(), sub_shape,
                              visited);
      });
  return modified;
}

absl::StatusOr<bool> MemorySpacePropagation::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  // Configure bitcasts to define values. Otherwise, if there is only a bitcast
  // between a fusion input and output and these two values are in different
  // memory spaces, we can get inconsistent memory spaces between the parameter
  // and fusion operand or root and fusion output.
  ABSL_ASSIGN_OR_RETURN(auto dataflow_analysis,
                   HloDataflowAnalysis::Run(*module, /*ssa_form=*/false,
                                            /*bitcast_defines_value=*/true));
  dataflow_analysis_ = std::move(dataflow_analysis);

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      HloDataflowPropagation::ForEachCallBoundary(
          instruction,
          [&](const HloCallBoundary& boundary) {
            // Propagate the operand subshapes.
            for (int64_t operand_idx = 0;
                 operand_idx < boundary.num_parameters(); ++operand_idx) {
              ShapeUtil::ForEachLeafShape(
                  boundary.caller_operand(operand_idx)->shape(),
                  [&](const Shape& sub_shape, const ShapeIndex& index) {
                    absl::flat_hash_set<const HloValue*> visited;
                    modified |=
                        Propagate(index, boundary.callee_parameter(operand_idx),
                                  sub_shape, visited);
                  });
            }

            // Propagate output subshapes.
            ShapeUtil::ForEachLeafShape(
                instruction->shape(),
                [&](const Shape& sub_shape, const ShapeIndex& index) {
                  absl::flat_hash_set<const HloValue*> visited;
                  modified |= Propagate(index, boundary.callee_root(),
                                        sub_shape, visited);
                });
          },
          kFusionBoundaryOptions);
    }
  }
  return modified;
}

bool MemorySpacePropagation::Propagate(
    ShapeIndexView index, const HloInstruction* callee_instruction,
    const Shape& src_shape,
    absl::flat_hash_set<const HloValue*>& visited) const {
  bool modified = false;
  const HloValue& value = dataflow_analysis_->GetUniqueValueAt(
      callee_instruction, ShapeIndex(index));

  if (visited.contains(&value)) {
    return false;
  }
  visited.insert(&value);

  for (const HloPosition& position : value.positions()) {
    HloInstruction* instruction = position.instruction;
    Shape* shape = ShapeUtil::GetMutableSubshape(instruction->mutable_shape(),
                                                 position.index);
    std::optional<SplitConfig> dest_split_config =
        LayoutUtil::GetSplitConfig(*shape);
    std::optional<SplitConfig> src_split_config =
        LayoutUtil::GetSplitConfig(src_shape);

    if (shape->layout().memory_space() != src_shape.layout().memory_space() ||
        dest_split_config != src_split_config) {
      shape->mutable_layout()->set_memory_space(
          src_shape.layout().memory_space());
      shape->mutable_layout()->clear_split_configs();
      if (src_split_config.has_value()) {
        shape->mutable_layout()->add_split_configs(*src_split_config);
      }
      modified = true;
    }

    if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      modified |= Propagate(position.index, instruction->operand(0), src_shape,
                            visited);
    }

    // For fusion outputs, propagate the memory space to the fusion root.
    HloDataflowPropagation::ForEachCallBoundary(
        instruction,
        [&](const HloCallBoundary& boundary) {
          modified |= Propagate(position.index, boundary.callee_root(),
                                src_shape, visited);
        },
        kFusionBoundaryOptions);

    // For nested fusion roots and parameters, pop one level up and propagate
    // the memory space to the output or operand of the calling fusion
    // instruction.
    HloDataflowPropagation::ForEachCallerBoundary(
        instruction->parent(),
        [&](const HloCallBoundary& boundary) {
          if (!boundary.callsite->parent()->IsFusionComputation()) {
            return;
          }
          if (instruction == boundary.callee_root()) {
            modified |= Propagate(position.index, boundary.callsite, src_shape,
                                  visited);
          }
          if (instruction->opcode() == HloOpcode::kParameter &&
              instruction->parameter_number() < boundary.num_parameters()) {
            modified |= Propagate(
                position.index,
                boundary.caller_operand(instruction->parameter_number()),
                src_shape, visited);
          }
        },
        kFusionBoundaryOptions);
  }

  for (const HloUse& use : value.GetUses()) {
    // For fusion uses, propagate the memory space to the fusion parameter.
    HloDataflowPropagation::ForEachCalledParameter(
        use.instruction, use.operand_number,
        [&](HloInstruction* param) {
          modified |= Propagate(use.operand_index, param, src_shape, visited);
        },
        kFusionBoundaryOptions);
  }
  return modified;
}

}  // namespace xla
