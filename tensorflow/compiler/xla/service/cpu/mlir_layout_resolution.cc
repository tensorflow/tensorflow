/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/mlir_layout_resolution.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {
namespace {

// Checks if the shape 1) has a non-default layout 2) without tiles.
bool LayoutRequiresNormalization(const Shape& shape) {
  if (shape.IsTuple()) {
    return absl::c_any_of(shape.tuple_shapes(), LayoutRequiresNormalization);
  }
  if (!shape.has_layout()) return false;

  // This code can't handle tiles.
  if (!shape.layout().tiles().empty()) return false;

  // For default shapes, there's nothing to do.
  return shape.layout() != LayoutUtil::GetDefaultLayoutForShape(shape);
}

bool EntryComputationLayoutCanBeNormalized(const HloModule& module) {
  return LayoutRequiresNormalization(
             module.entry_computation_layout().result_layout().shape()) ||
         absl::c_any_of(
             module.entry_computation_layout().parameter_layouts(),
             [](const auto& shape_layout) {
               return LayoutRequiresNormalization(shape_layout.shape());
             });
}

// For a tensor with a non-default layout, inserts a reshape and a transpose to
// convert from the non-default layout to the default.
HloInstruction* NormalizeTensor(HloInstruction* tensor, const Shape& shape,
                                bool is_input) {
  // Reshape the parameter into the shape that's actually passed.
  int64_t rank = shape.rank();
  std::vector<int64_t> permutation(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    permutation[dim] = rank - 1ll - shape.layout().minor_to_major(dim);
  }
  auto inverse_permutation = InversePermutation(permutation);

  auto physical_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      shape.element_type(),
      Permute(shape.dimensions(),
              is_input ? permutation : inverse_permutation));

  auto* computation = tensor->parent();
  if (is_input) {
    auto* reshape = computation->AddInstruction(
        HloInstruction::CreateReshape(physical_shape, tensor));
    return computation->AddInstruction(
        HloInstruction::CreateTranspose(shape, reshape, inverse_permutation));
  }
  auto* transpose = computation->AddInstruction(HloInstruction::CreateTranspose(
      physical_shape, tensor, inverse_permutation));
  return computation->AddInstruction(
      HloInstruction::CreateReshape(shape, transpose));
}

// If the value is a tuple, normalizes each element. If it is a tensor,
// normalizes it using NormalizeTensor.
HloInstruction* NormalizeValue(HloInstruction* value, const Shape& shape,
                               bool is_input) {
  if (!LayoutRequiresNormalization(shape)) return value;

  if (!shape.IsTuple()) {
    return NormalizeTensor(value, shape, is_input);
  }

  HloComputation* computation = value->parent();
  std::vector<HloInstruction*> elements;
  for (int64_t i = 0; i < shape.tuple_shapes_size(); ++i) {
    auto* element = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(value, i));
    elements.push_back(
        NormalizeValue(element, shape.tuple_shapes(i), is_input));
  }

  return computation->AddInstruction(HloInstruction::CreateTuple(elements));
}

// Inserts reshapes and transposes for entry computation parameters and results
// that have non-default layouts.
Status NormalizeEntryComputationLayout(xla::HloModule* module) {
  if (!EntryComputationLayoutCanBeNormalized(*module)) {
    return ::tsl::OkStatus();
  }
  auto* computation = module->entry_computation();
  const auto& computation_layout = module->entry_computation_layout();
  for (int i = 0; i < computation->num_parameters(); ++i) {
    auto* param = computation->parameter_instruction(i);
    const auto& shape = computation_layout.parameter_layout(i).shape();
    std::vector<HloInstruction*> users = param->users();
    auto* normalized = NormalizeValue(param, shape, /*is_input=*/true);
    TF_RETURN_IF_ERROR(param->ReplaceUsesWith(users, normalized));
  }

  const auto& result_shape = computation_layout.result_layout().shape();
  auto* normalized_result = NormalizeValue(computation->root_instruction(),
                                           result_shape, /*is_input=*/false);
  computation->set_root_instruction(normalized_result);

  return ::tsl::OkStatus();
}

Status NormalizeCustomCallLayouts(xla::HloModule* module) {
  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->instructions()) {
      auto* call = DynCast<HloCustomCallInstruction>(instruction);
      if (!call) {
        continue;
      }

      std::vector<HloInstruction*> users = instruction->users();

      // Normalize the output, if necessary. Note that the output of the custom
      // call is an "input" from the MLIR perspective.
      auto* normalized = NormalizeValue(call, call->shape(), /*is_input=*/true);
      if (normalized != call) {
        TF_RETURN_IF_ERROR(call->ReplaceUsesWith(users, normalized));
      }

      // Normalize any operands that require it.
      if (call->layout_constrained()) {
        const auto& operand_shapes = call->operand_shapes_with_layout();
        for (int64_t i = 0, e = call->operand_count(); i != e; ++i) {
          auto* normalized_operand = NormalizeValue(
              call->mutable_operand(i), operand_shapes[i], /*is_input=*/false);
          TF_RETURN_IF_ERROR(call->ReplaceOperandWith(i, normalized_operand));
        }
      }
    }
  }
  return ::tsl::OkStatus();
}

}  // namespace

StatusOr<bool> MlirLayoutResolution::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(NormalizeEntryComputationLayout(module));
  TF_RETURN_IF_ERROR(NormalizeCustomCallLayouts(module));
  return true;
}

}  // namespace xla
