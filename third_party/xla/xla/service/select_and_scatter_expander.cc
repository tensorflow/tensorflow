/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/select_and_scatter_expander.h"

#include <numeric>
#include <vector>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"

namespace xla {

absl::StatusOr<HloInstruction*> SelectAndScatterExpander::ExpandInstruction(
    HloInstruction* instruction) {
  // Prepare the original values
  auto* computation = instruction->parent();
  auto* sas = Cast<HloSelectAndScatterInstruction>(instruction);
  auto* operand = sas->mutable_operand(0);
  auto operand_shape = operand->shape();
  auto* source = sas->mutable_operand(1);
  auto* select = sas->select();
  auto* init_value = sas->mutable_operand(2);

  // Useful shapes
  const auto iota_shape = ShapeUtil::ChangeElementType(operand_shape, S32);
  const auto scalar_operand =
      ShapeUtil::MakeScalarShape(operand->shape().element_type());
  const auto scalar_iota =
      ShapeUtil::MakeScalarShape(iota_shape.element_type());
  const auto source_shape = source->shape();
  const Shape iota_shape_reduced =
      ShapeUtil::ChangeElementType(source_shape, S32);

  // Construct one iota for each dimension. This will reduced in the reduction
  // to determine the indices to be scattered to.
  std::vector<HloInstruction*> iotas;
  iotas.reserve(operand_shape.rank());
  for (int i = 0; i < operand_shape.rank(); ++i) {
    iotas.push_back(
        computation->AddInstruction(HloInstruction::CreateIota(iota_shape, i)));
  }

  // Construct the WindowReduction region
  HloComputation* new_comp = [&]() -> HloComputation* {
    HloComputation::Builder builder(
        absl::StrCat(select->name(), ".reduce_window"));
    auto rhs_begin = static_cast<int64_t>(iotas.size() + 1);
    auto first_iota_index = 1;
    auto* neg_one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(-1)));
    auto* first_lhs_iota =
        builder.AddInstruction(HloInstruction::CreateParameter(
            first_iota_index, scalar_iota, "iota_lhs"));
    auto* first_rhs_iota =
        builder.AddInstruction(HloInstruction::CreateParameter(
            first_iota_index + rhs_begin, scalar_iota, "iota_lhs"));
    auto* lhs_first_in_window =
        builder.AddInstruction(HloInstruction::CreateCompare(
            sas->select()->root_instruction()->shape(), first_lhs_iota, neg_one,
            Comparison::Direction::kNe, {}));
    // Current implementations of ReduceWindow do not need the following line in
    // their implementations, but it is actually required in the documented
    // behavior of the implementation which allows the seed value to occur on
    // both lhs and rhs sides when padding occurs.
    auto* rhs_first_in_window =
        builder.AddInstruction(HloInstruction::CreateCompare(
            sas->select()->root_instruction()->shape(), first_rhs_iota, neg_one,
            Comparison::Direction::kNe, {}));
    auto rhs_not_first_in_window = builder.AddInstruction(
        HloInstruction::CreateUnary(sas->select()->root_instruction()->shape(),
                                    HloOpcode::kNot, rhs_first_in_window));

    auto* operand_lhs = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_operand, "operand_lhs"));
    auto* operand_rhs = builder.AddInstruction(HloInstruction::CreateParameter(
        rhs_begin, scalar_operand, "operand_rhs"));
    auto* call = builder.AddInstruction(
        HloInstruction::CreateCall(sas->select()->root_instruction()->shape(),
                                   {operand_lhs, operand_rhs}, sas->select()));

    auto* pred = builder.AddInstruction(HloInstruction::CreateBinary(
        call->shape(), HloOpcode::kAnd, call, lhs_first_in_window));
    pred = builder.AddInstruction(HloInstruction::CreateBinary(
        call->shape(), HloOpcode::kOr, pred, rhs_not_first_in_window));

    std::vector<HloInstruction*> result_tuple;
    result_tuple.push_back(builder.AddInstruction(HloInstruction::CreateTernary(
        scalar_operand, HloOpcode::kSelect, pred, operand_lhs, operand_rhs)));
    for (auto i = first_iota_index; i < rhs_begin; ++i) {
      // Special case the first iota because the same parameter instruction
      // cannot occur multiple times.
      xla::HloInstruction *iota_lhs, *iota_rhs;
      if (i == first_iota_index) {
        iota_lhs = first_lhs_iota;
        iota_rhs = first_rhs_iota;
      } else {
        iota_lhs = builder.AddInstruction(
            HloInstruction::CreateParameter(i, scalar_iota, "iota_lhs"));
        iota_rhs = builder.AddInstruction(HloInstruction::CreateParameter(
            i + rhs_begin, scalar_iota, "iota_rhs"));
      }
      result_tuple.push_back(
          builder.AddInstruction(HloInstruction::CreateTernary(
              scalar_iota, HloOpcode::kSelect, pred, iota_lhs, iota_rhs)));
    }
    builder.AddInstruction(HloInstruction::CreateTuple(result_tuple));
    auto* result = select->parent()->AddEmbeddedComputation(builder.Build());

    // This computation cannot have a call op, so finally inline the select
    // computation.
    if (!CallInliner::Inline(call).ok()) {
      return nullptr;
    }
    return result;
  }();

  if (!new_comp) {
    return nullptr;
  }

  // ReduceWindow arguments
  auto num_reduce_values = iotas.size() + 1;
  std::vector<HloInstruction*> ops;
  ops.reserve(num_reduce_values);
  ops.push_back(operand);
  ops.insert(ops.end(), iotas.begin(), iotas.end());

  auto* neg_one = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(-1)));
  std::vector<HloInstruction*> reduce_init_values;
  reduce_init_values.reserve(num_reduce_values);
  reduce_init_values.push_back(init_value);
  for (auto i = 0; i < iotas.size(); ++i) {
    reduce_init_values.push_back(neg_one);
  }

  std::vector<xla::Shape> shapes;
  shapes.reserve(num_reduce_values);
  shapes.push_back(source->shape());
  for (auto i = 0; i < iotas.size(); ++i) {
    shapes.push_back(iota_shape_reduced);
  }

  auto* reduce_window =
      computation->AddInstruction(HloInstruction::CreateReduceWindow(
          ShapeUtil::MakeTupleShape(shapes), ops, reduce_init_values,
          sas->window(), new_comp));

  // Handle the results of the reduction
  std::vector<HloInstruction*> iota_indices;
  std::vector<int64_t> broadcasted_iota_dims;
  broadcasted_iota_dims.reserve(iota_shape_reduced.rank() + 1);
  broadcasted_iota_dims.insert(broadcasted_iota_dims.end(),
                               iota_shape_reduced.dimensions().begin(),
                               iota_shape_reduced.dimensions().end());
  broadcasted_iota_dims.push_back(1);
  auto broadcasted_iota_shape = ShapeUtil::MakeShape(
      iota_shape_reduced.element_type(), broadcasted_iota_dims);

  for (int i = 1; i < num_reduce_values; ++i) {
    auto* element = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(reduce_window, i));
    iota_indices.push_back(computation->AddInstruction(
        HloInstruction::CreateReshape(broadcasted_iota_shape, element)));
  }

  // Prepare scatter inputs
  std::vector<int64_t> scatter_dims(operand->shape().rank());
  std::iota(scatter_dims.begin(), scatter_dims.end(), 0);
  auto* broadcasted_init_value = computation->AddInstruction(
      HloInstruction::CreateBroadcast(instruction->shape(), init_value, {}));

  std::vector<int64_t> concatenated_iotas_dims;
  concatenated_iotas_dims.reserve(iota_indices.front()->shape().rank());
  concatenated_iotas_dims.insert(concatenated_iotas_dims.end(),
                                 broadcasted_iota_dims.begin(),
                                 broadcasted_iota_dims.end());
  concatenated_iotas_dims.back() = static_cast<int64_t>(iota_indices.size());
  auto* indices = computation->AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(iota_shape.element_type(), concatenated_iotas_dims),
      iota_indices, iota_shape.rank()));

  // Scatter
  ScatterDimensionNumbers dim_nums =
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/scatter_dims,
          /*scatter_dims_to_operand_dims=*/scatter_dims,
          /*index_vector_dim=*/source->shape().rank());
  return computation->AddInstruction(HloInstruction::CreateScatter(
      /*shape=*/sas->shape(), /*operand=*/broadcasted_init_value,
      /*scatter_indices=*/indices, /*updates=*/source,
      /*update_computation=*/sas->scatter(), /*scatter_dim_numbers=*/dim_nums,
      /*indices_are_sorted=*/false, /*unique_indices=*/false));
}

bool SelectAndScatterExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kSelectAndScatter;
}

}  // namespace xla
