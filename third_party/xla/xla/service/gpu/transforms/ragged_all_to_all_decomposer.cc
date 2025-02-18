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

#include "xla/service/gpu/transforms/ragged_all_to_all_decomposer.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

// Returns a multi-index offset for the ith row. The tensors are always ragged
// by the outmost dimension, `offsets` contains indexes of the outmost dimension
// and outher dimensions are 0.
absl::InlinedVector<HloInstruction*, 4> GetOffsetMultiIndex(
    HloComputation* computation, HloInstruction* offsets, int64_t index,
    int64_t rank) {
  absl::InlinedVector<HloInstruction*, 4> result(
      rank, computation->AddInstruction(
                HloInstruction::CreateConstant(LiteralUtil::Zero(S32))));

  HloInstruction* index_value =
      computation->AddInstruction(HloInstruction::CreateSlice(
          /*shape=*/ShapeUtil::MakeShape(S32, {1}),
          /*operand=*/offsets,
          /*start_indices=*/{index},
          /*limit_indices=*/{index + 1},
          /*strides=*/{1}));
  result[0] = computation->AddInstruction(
      HloInstruction::CreateReshape(/*shape=*/
                                    ShapeUtil::MakeScalarShape(S32),
                                    index_value));
  return result;
}

// Pads the outermost dimension of the input tensor to double the size.
HloInstruction* PadOutermostDimension(HloComputation* computation,
                                      HloInstruction* input) {
  Shape padded_shape = input->shape();
  PaddingConfig padding_config = MakeNoPaddingConfig(padded_shape.rank());
  padding_config.mutable_dimensions(0)->set_edge_padding_high(
      padded_shape.dimensions(0));

  padded_shape.set_dimensions(0, 2 * padded_shape.dimensions(0));

  HloInstruction* padding_value =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(input->shape().element_type())));

  return computation->AddInstruction(HloInstruction::CreatePad(
      padded_shape, input, padding_value, padding_config));
}

// Takes a ragged tensor and a vector of chunk sizes. Returns a ragged tensor
// where padding is filled with zeros.
HloInstruction* FillPaddingWithZeros(HloComputation* computation,
                                     HloInstruction* input,
                                     HloInstruction* sizes) {
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));

  // Create reduction computation.
  auto embedded_builder = HloComputation::Builder("add");
  auto lhs = embedded_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "lhs"));
  auto rhs = embedded_builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(S32, {}), "rhs"));
  embedded_builder.AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));

  HloComputation* add_computation =
      computation->parent()->AddEmbeddedComputation(embedded_builder.Build());

  // Find total sizes of the significant data in the ragged tensor.
  HloInstruction* total_size =
      computation->AddInstruction(HloInstruction::CreateReduce(
          ShapeUtil::MakeScalarShape(S32), sizes, zero, {0}, add_computation));

  Shape iota_shape = ShapeUtil::MakeShape(S32, input->shape().dimensions());

  HloInstruction* iota =
      computation->AddInstruction(HloInstruction::CreateIota(iota_shape, 0));

  HloInstruction* total_size_broadcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(iota_shape, total_size, {}));

  Shape mask_shape = ShapeUtil::MakeShape(PRED, iota_shape.dimensions());

  // Get bitmask for the significant data in the ragged tensor.
  HloInstruction* iota_mask =
      computation->AddInstruction(HloInstruction::CreateCompare(
          mask_shape, iota, total_size_broadcast, Comparison::Direction::kLt));

  HloInstruction* padding_value =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(input->shape().element_type())));

  HloInstruction* zero_broadcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(input->shape(), padding_value, {}));

  // Fill padding with zeros.
  return computation->AddInstruction(HloInstruction::CreateTernary(
      input->shape(), HloOpcode::kSelect, iota_mask, input, zero_broadcast));
}

// Returns dense representation of the ragged input tensor.
//
// The dense representation is a tuple of slices of the input tensor, where each
// element of the tuple is an ragged row padded with zeros to the same size as
// the ragged input.
std::vector<HloInstruction*> RaggedToDense(HloComputation* computation,
                                           HloInstruction* ragged_input,
                                           HloInstruction* offsets,
                                           HloInstruction* sizes) {
  int64_t num_rows = offsets->shape().dimensions(0);

  std::vector<HloInstruction*> sliced_operands;

  for (int64_t i = 0; i < num_rows; ++i) {
    auto offset_multi_index = GetOffsetMultiIndex(computation, offsets, i,
                                                  ragged_input->shape().rank());

    HloInstruction* padded_input =
        PadOutermostDimension(computation, ragged_input);

    HloInstruction* row_slice =
        computation->AddInstruction(HloInstruction::CreateDynamicSlice(
            ragged_input->shape(), padded_input, offset_multi_index,
            ragged_input->shape().dimensions()));

    sliced_operands.push_back(row_slice);
  }

  return sliced_operands;
}

// Returns ragged representation of the dense output tensor.
HloInstruction* DenseToRagged(HloComputation* computation,
                              HloInstruction* dense_inputs,
                              HloInstruction* ragged_output,
                              HloInstruction* offsets, HloInstruction* sizes) {
  int64_t num_rows = offsets->shape().dimensions(0);
  int64_t rank = ragged_output->shape().rank();

  Shape original_shape = ragged_output->shape();

  HloInstruction* padded_ragged_output =
      PadOutermostDimension(computation, ragged_output);

  for (int64_t i = 0; i < num_rows; ++i) {
    auto offset_multi_index = GetOffsetMultiIndex(
        computation, offsets, i, padded_ragged_output->shape().rank());

    HloInstruction* update = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(dense_inputs, i));

    padded_ragged_output =
        computation->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            padded_ragged_output->shape(), padded_ragged_output, update,
            offset_multi_index));
  }

  ragged_output = computation->AddInstruction(HloInstruction::CreateSlice(
      original_shape, padded_ragged_output, std::vector<int64_t>(rank, 0),
      original_shape.dimensions(), std::vector<int64_t>(rank, 1)));

  ragged_output = FillPaddingWithZeros(computation, ragged_output, sizes);

  return ragged_output;
}

// Rewrites a ragged all-to-all to a sequence dynamic-slicer, an all-to-all,
// and a sequence dynamic-update-slices.
absl::Status DecomposeRaggedAllToAll(HloInstruction* hlo,
                                     HloComputation* computation,
                                     HloModule* module) {
  HloRaggedAllToAllInstruction* all_to_all =
      Cast<HloRaggedAllToAllInstruction>(hlo);
  HloInstruction* input_operand = all_to_all->mutable_operand(0);
  HloInstruction* output_operand = all_to_all->mutable_operand(1);

  HloInstruction* input_offsets = all_to_all->mutable_operand(2);
  HloInstruction* send_sizes = all_to_all->mutable_operand(3);
  HloInstruction* output_offsets = all_to_all->mutable_operand(4);
  HloInstruction* recv_sizes = all_to_all->mutable_operand(5);

  auto dense_input =
      RaggedToDense(computation, input_operand, input_offsets, send_sizes);

  std::vector<Shape> dense_input_shapes;
  dense_input_shapes.reserve(dense_input.size());
  for (auto* dense_input : dense_input) {
    dense_input_shapes.push_back(dense_input->shape());
  }

  auto dense_output =
      computation->AddInstruction(HloInstruction::CreateAllToAll(
          ShapeUtil::MakeTupleShape(dense_input_shapes), dense_input,
          all_to_all->device_list(),
          /*constrain_layout=*/false,
          /*channel_id=*/all_to_all->channel_id()));

  auto* ragged_output = DenseToRagged(computation, dense_output, output_operand,
                                      output_offsets, recv_sizes);

  TF_RETURN_IF_ERROR(all_to_all->ReplaceAllUsesWith(ragged_output));
  TF_RETURN_IF_ERROR(
      computation->RemoveInstructionAndUnusedOperands(all_to_all));

  return absl::OkStatus();
}

absl::StatusOr<bool> RaggedAllToAllDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (auto computation : module->computations(execution_threads)) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (HloPredicateIsNotOp<HloOpcode::kRaggedAllToAll>(hlo)) {
        continue;
      }
      changed = true;
      TF_RETURN_IF_ERROR(DecomposeRaggedAllToAll(hlo, computation, module));
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
