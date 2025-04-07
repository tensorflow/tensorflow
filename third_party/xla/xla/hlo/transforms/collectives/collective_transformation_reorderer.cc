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

#include "xla/hlo/transforms/collectives/collective_transformation_reorderer.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/shape.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
struct CollectiveTransformation {
  HloInstruction* hlo;
  int64_t transformed_collective_dimension;
};

// Find a list of reshapes following the all-gather that could be moved to
// before the all-gather.
std::optional<std::vector<CollectiveTransformation>>
GetAllGatherTransformations(HloInstruction* all_gather) {
  std::vector<HloInstruction*> transformation_hlos;
  {
    // First find the list of reshapes.
    HloInstruction* transformation_hlo = all_gather;
    bool found_unsupported_transformation = false;
    while (transformation_hlo->user_count() == 1 &&
           !found_unsupported_transformation) {
      transformation_hlo = transformation_hlo->users()[0];
      switch (transformation_hlo->opcode()) {
        case HloOpcode::kReshape: {
          transformation_hlos.push_back(transformation_hlo);
          break;
        }
        default:
          found_unsupported_transformation = true;
      }
    }
  }
  if (transformation_hlos.empty()) {
    return std::nullopt;
  }
  // Find the all-gather dimension if the all-gather is to be applied to the
  // reshaped input.
  auto get_reshaped_all_gather_dimension =
      [](const Shape& all_gather_shape, int64_t all_gather_dimension,
         HloInstruction* transformation_hlo) -> std::optional<int64_t> {
    // Stride refers to the maximal region of continuous memory before
    // all-gather that remains continuous after all-gather. This function
    // finds how much such regions exist before all-gather.
    int64_t all_gather_num_strides = absl::c_accumulate(
        all_gather_shape.dimensions().subspan(0, all_gather_dimension), 1,
        [](int64_t product, int64_t dimension_size) {
          return product * dimension_size;
        });
    // If the reshape is eligible for this transformation, it does not change
    // the number of strides.
    int64_t reshaped_all_gather_dimension = 0;
    int64_t reshaped_num_strides = 1;
    while (reshaped_all_gather_dimension <
               transformation_hlo->shape().dimensions().size() &&
           reshaped_num_strides < all_gather_num_strides) {
      reshaped_num_strides *=
          transformation_hlo->shape().dimensions(reshaped_all_gather_dimension);
      ++reshaped_all_gather_dimension;
    }
    if (reshaped_num_strides != all_gather_num_strides) {
      return std::nullopt;
    }
    // Additionally, we make sure the reshape does not change the size of the
    // all-gather dimension.
    // TODO(jlwei@): support merging dimensions following the all-gather
    // dimension into the all-gather dimension.
    if (transformation_hlo->shape().dimensions(reshaped_all_gather_dimension) !=
        all_gather_shape.dimensions(all_gather_dimension)) {
      return std::nullopt;
    }
    return reshaped_all_gather_dimension;
  };

  std::vector<CollectiveTransformation> transformations;
  HloAllGatherInstruction* all_gather_instruction =
      DynCast<HloAllGatherInstruction>(all_gather);
  Shape all_gather_shape = all_gather_instruction->shape();
  int64_t all_gather_dimension = all_gather_instruction->all_gather_dimension();
  CHECK(all_gather_instruction != nullptr);
  // Then find the reshapes that are eligible for this transformation.
  for (HloInstruction* transformation_hlo : transformation_hlos) {
    bool found_unsupported_transformation = false;
    switch (transformation_hlo->opcode()) {
      case HloOpcode::kReshape: {
        std::optional<int64_t> reshaped_all_gather_dimension =
            get_reshaped_all_gather_dimension(
                all_gather_shape, all_gather_dimension, transformation_hlo);
        if (reshaped_all_gather_dimension.has_value()) {
          transformations.push_back(
              {transformation_hlo, *reshaped_all_gather_dimension});
          all_gather_shape = transformation_hlo->shape();
          all_gather_dimension = *reshaped_all_gather_dimension;
        } else {
          found_unsupported_transformation = true;
        }
        break;
      }
      default:
        return std::nullopt;
    }
    if (found_unsupported_transformation) {
      break;
    }
  }
  if (transformations.empty()) {
    return std::nullopt;
  }
  return transformations;
}

// Find a list of reshapes feeding the all-reduce that could be moved to after
// the all-reduce.
std::vector<HloInstruction*> GetAllReduceTransformations(
    HloInstruction* all_reduce) {
  HloAllReduceInstruction* all_reduce_instruction =
      DynCast<HloAllReduceInstruction>(all_reduce);
  CHECK_NE(all_reduce_instruction, nullptr);
  if (all_reduce_instruction->constrain_layout()) {
    return {};
  }
  std::vector<HloInstruction*> transformation_hlos;
  HloInstruction* transformation_hlo = all_reduce->mutable_operand(0);
  while (transformation_hlo->opcode() == HloOpcode::kReshape &&
         transformation_hlo->user_count() == 1) {
    transformation_hlos.push_back(transformation_hlo);
    transformation_hlo = transformation_hlo->mutable_operand(0);
  }
  return transformation_hlos;
}
}  // namespace

absl::StatusOr<bool>
CollectiveTransformationReorder::ReorderAllGatherTransformations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // First, find all all-gathers and reshapes that are eligible for this
  // transformation.
  HloInstructionMap<std::vector<CollectiveTransformation>>
      all_gather_to_transformations;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kAllGather) {
        if (instruction->operand_count() != 1) {
          continue;
        }
        std::optional<std::vector<CollectiveTransformation>>
            all_gather_transformations =
                GetAllGatherTransformations(instruction);
        if (all_gather_transformations.has_value()) {
          all_gather_to_transformations[instruction] =
              *std::move(all_gather_transformations);
        }
      }
    }
  }
  if (all_gather_to_transformations.empty()) {
    return false;
  }
  auto reshape_all_gather_operand =
      [](HloInstruction* all_gather_operand,
         int64_t original_all_gather_dimension,
         const CollectiveTransformation& transformation) {
        Shape reshaped_all_gather_operand_shape = transformation.hlo->shape();
        int64_t operand_all_gather_dimension_size =
            all_gather_operand->shape().dimensions(
                original_all_gather_dimension);
        reshaped_all_gather_operand_shape.set_dimensions(
            transformation.transformed_collective_dimension,
            operand_all_gather_dimension_size);
        HloComputation* computation = all_gather_operand->parent();
        return computation->AddInstruction(HloInstruction::CreateReshape(
            reshaped_all_gather_operand_shape, all_gather_operand));
      };
  for (auto& [instruction, transformations] : all_gather_to_transformations) {
    HloAllGatherInstruction* all_gather =
        DynCast<HloAllGatherInstruction>(instruction);
    int64_t all_gather_dimension = all_gather->all_gather_dimension();
    int64_t original_all_gather_dimension_size =
        all_gather->shape().dimensions(all_gather_dimension);
    HloInstruction* all_gather_operand = instruction->mutable_operand(0);
    // For each eligible reshape on the all-gather result, we reshape the
    // all-gather operand instead.
    for (const CollectiveTransformation& transformation : transformations) {
      all_gather_operand = reshape_all_gather_operand(
          all_gather_operand, all_gather_dimension, transformation);
      all_gather_dimension = transformation.transformed_collective_dimension;
    }
    Shape new_all_gather_shape = all_gather_operand->shape();
    new_all_gather_shape.set_dimensions(all_gather_dimension,
                                        original_all_gather_dimension_size);
    HloComputation* computation = all_gather_operand->parent();
    HloInstruction* new_all_gather =
        all_gather->AddInstruction(HloInstruction::CreateAllGather(
            new_all_gather_shape, {all_gather_operand}, all_gather_dimension,
            all_gather->device_list(), all_gather->constrain_layout(),
            all_gather->channel_id(), all_gather->use_global_device_ids()));
    TF_RETURN_IF_ERROR(
        transformations.back().hlo->ReplaceAllUsesWith(new_all_gather));
    if (computation->root_instruction() == transformations.back().hlo) {
      computation->set_root_instruction(new_all_gather);
    }
  }
  return true;
}

absl::StatusOr<bool>
CollectiveTransformationReorder::ReorderAllReduceTransformations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // First, find all reshapes and all-reduces that are eligible for this
  // transformation.
  HloInstructionMap<std::vector<HloInstruction*>> all_reduce_to_transformations;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kAllReduce) {
        if (instruction->user_count() != 1 ||
            computation->root_instruction() == instruction) {
          continue;
        }
        std::vector<HloInstruction*> reshapes =
            GetAllReduceTransformations(instruction);
        if (reshapes.empty()) {
          continue;
        }
        all_reduce_to_transformations[instruction] = std::move(reshapes);
      }
    }
  }
  if (all_reduce_to_transformations.empty()) {
    return false;
  }
  for (auto& [inst, reshapes] : all_reduce_to_transformations) {
    HloComputation* computation = inst->parent();
    HloAllReduceInstruction* all_reduce =
        DynCast<HloAllReduceInstruction>(inst);
    CHECK(!reshapes.empty());
    HloInstruction* cur_operand = reshapes.back()->mutable_operand(0);
    HloInstruction* new_all_reduce =
        all_reduce->AddInstruction(HloInstruction::CreateAllReduce(
            cur_operand->shape(), {cur_operand}, all_reduce->to_apply(),
            all_reduce->device_list(), all_reduce->constrain_layout(),
            all_reduce->channel_id(), all_reduce->use_global_device_ids()));

    // For each eligible reshape on the old all-reduce's operand, we reshape the
    // new all-reduce result instead.
    cur_operand = new_all_reduce;
    for (int64_t i = reshapes.size() - 1; i >= 0; --i) {
      cur_operand = computation->AddInstruction(
          HloInstruction::CreateReshape(reshapes[i]->shape(), cur_operand));
    }
    TF_RETURN_IF_ERROR(
        computation->ReplaceInstruction(all_reduce, cur_operand));
  }
  return true;
}

absl::StatusOr<bool> CollectiveTransformationReorder::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(bool ag_changed, ReorderAllGatherTransformations(
                                           module, execution_threads));
  TF_ASSIGN_OR_RETURN(bool ar_changed, ReorderAllReduceTransformations(
                                           module, execution_threads));
  if (ag_changed || ar_changed) {
    // Remove the original all-gathers/all-reduces and reshapes.
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
  }
  return ag_changed || ar_changed;
}

}  // namespace xla
