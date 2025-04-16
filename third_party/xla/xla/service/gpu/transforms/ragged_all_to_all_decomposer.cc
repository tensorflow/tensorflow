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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// Runs all-to-all to exchange output offsets for each participating device.
HloInstruction* RunAllToAllOnOutputOffsets(HloComputation* computation,
                                           HloInstruction* ragged_all_to_all,
                                           HloInstruction* output_offsets,
                                           int64_t num_updates_per_replica,
                                           int64_t num_participating_devices) {
  int64_t num_total_updates = output_offsets->shape().dimensions(0);

  // `output_offsets` is a tensor of shape [num_total_updates]. Reshape it to
  // [num_participating_devices, num_updates_per_replica]. This is needed to
  // run all-to-all with split dimension 0 and exchage offsets for each
  // participating device.
  output_offsets = computation->AddInstruction(HloInstruction::CreateReshape(
      /*shape=*/ShapeUtil::MakeShape(
          output_offsets->shape().element_type(),
          {num_participating_devices, num_updates_per_replica}),
      output_offsets));

  // Run all-to-all.
  output_offsets = computation->AddInstruction(HloInstruction::CreateAllToAll(
      output_offsets->shape(), {output_offsets},
      ragged_all_to_all->device_list(),
      /*constrain_layout=*/false,
      /*channel_id=*/ragged_all_to_all->channel_id(), /*split_dimension=*/0));

  // Reshape it back to [num_total_updates].
  return computation->AddInstruction(HloInstruction::CreateReshape(
      /*shape=*/ShapeUtil::MakeShape(output_offsets->shape().element_type(),
                                     {num_total_updates}),
      output_offsets));
}

// Returns a scalar value of the ith element of the given HLO instruction.
HloInstruction* GetScalarValue(HloInstruction* hlo, int64_t index) {
  HloComputation* computation = hlo->parent();
  HloInstruction* index_value =
      computation->AddInstruction(HloInstruction::CreateSlice(
          /*shape=*/ShapeUtil::MakeShape(hlo->shape().element_type(), {1}),
          /*operand=*/hlo,
          /*start_indices=*/{index},
          /*limit_indices=*/{index + 1},
          /*strides=*/{1}));
  return computation->AddInstruction(HloInstruction::CreateReshape(
      /*shape=*/ShapeUtil::MakeScalarShape(hlo->shape().element_type()),
      index_value));
}

// Returns a multi-index offset for the ith row. The tensors are always ragged
// by the outmost dimension, `offsets` contains indexes of the outmost dimension
// and outher dimensions are 0.
absl::InlinedVector<HloInstruction*, 4> GetOffsetMultiIndex(
    HloComputation* computation, HloInstruction* offsets, int64_t index,
    int64_t rank) {
  absl::InlinedVector<HloInstruction*, 4> result(
      rank, computation->AddInstruction(
                HloInstruction::CreateConstant(LiteralUtil::Zero(S64))));
  result[0] = GetScalarValue(offsets, index);
  return result;
}

// Adds a size-1 major dimension to the given HLO instruction.
HloInstruction* AddSize1MajorDimension(HloInstruction* hlo,
                                       HloComputation* computation) {
  absl::InlinedVector<int64_t, 4> reshape_dimensions;
  reshape_dimensions.reserve(hlo->shape().dimensions().size() + 1);
  reshape_dimensions.push_back(1);
  absl::c_copy(hlo->shape().dimensions(),
               std::back_inserter(reshape_dimensions));

  Shape reshape_shape =
      ShapeUtil::MakeShape(hlo->shape().element_type(), reshape_dimensions);
  return computation->AddInstruction(
      HloInstruction::CreateReshape(reshape_shape, hlo));
}

// Returns a slice of the given HLO instruction for the given index in the major
// dimension. If input shape is [N, M, K], then the output shape is [M, K]
// (without 1 in the first dimension).
HloInstruction* GetRowSlice(HloInstruction* hlo, int64_t row_index) {
  HloComputation* computation = hlo->parent();
  Shape row_shape = hlo->shape();
  row_shape.set_dimensions(0, 1);

  std::vector<int64_t> slice_start_indices(row_shape.dimensions_size(), 0);
  slice_start_indices[0] = row_index;
  std::vector<int64_t> slice_limit_indices{row_shape.dimensions().begin(),
                                           row_shape.dimensions().end()};
  slice_limit_indices[0] = row_index + 1;
  std::vector<int64_t> slice_strides(row_shape.dimensions_size(), 1);

  HloInstruction* row_slice =
      computation->AddInstruction(HloInstruction::CreateSlice(
          /*shape=*/row_shape,
          /*operand=*/hlo,
          /*start_indices=*/slice_start_indices,
          /*limit_indices=*/slice_limit_indices,
          /*strides=*/slice_strides));

  absl::InlinedVector<int64_t, 4> reshape_dimensions;
  std::copy(row_slice->shape().dimensions().begin() + 1,
            row_slice->shape().dimensions().end(),
            std::back_inserter(reshape_dimensions));

  Shape reshape_shape = ShapeUtil::MakeShape(row_slice->shape().element_type(),
                                             reshape_dimensions);
  return computation->AddInstruction(
      HloInstruction::CreateReshape(reshape_shape, row_slice));
}

// Returns a bitmask of shape `padded_ragged_output_shape` with 1s in the
// positions with major dimension index between [offset, offset + size).
HloInstruction* CreateUpdateMask(HloInstruction* offset_value,
                                 HloInstruction* size_value, int64_t idx,
                                 const Shape& padded_ragged_output_shape) {
  HloComputation* computation = offset_value->parent();
  Shape iota_shape =
      ShapeUtil::MakeShape(S64, padded_ragged_output_shape.dimensions());
  Shape compare_shape =
      ShapeUtil::MakeShape(PRED, padded_ragged_output_shape.dimensions());

  HloInstruction* iota =
      computation->AddInstruction(HloInstruction::CreateIota(iota_shape, 0));

  HloInstruction* broadcast_offset_value = computation->AddInstruction(
      HloInstruction::CreateBroadcast(iota_shape, offset_value, {}));
  HloInstruction* broadcast_size_value = computation->AddInstruction(
      HloInstruction::CreateBroadcast(iota_shape, size_value, {}));

  HloInstruction* upper_bound =
      computation->AddInstruction(HloInstruction::CreateBinary(
          iota_shape, HloOpcode::kAdd, broadcast_offset_value,
          broadcast_size_value));

  HloInstruction* greater_than_lower_bound = computation->AddInstruction(
      HloInstruction::CreateCompare(compare_shape, iota, broadcast_offset_value,
                                    Comparison::Direction::kGe));
  HloInstruction* less_than_upper_bound =
      computation->AddInstruction(HloInstruction::CreateCompare(
          compare_shape, iota, upper_bound, Comparison::Direction::kLt));

  return computation->AddInstruction(HloInstruction::CreateBinary(
      compare_shape, HloOpcode::kAnd, greater_than_lower_bound,
      less_than_upper_bound));
}

// Pads the outermost dimension of the hlo result by the given padding size.
HloInstruction* PadOutermostDimension(HloComputation* computation,
                                      HloInstruction* hlo,
                                      int64_t padding_size) {
  Shape padded_shape = hlo->shape();

  PaddingConfig padding_config =
      MakeNoPaddingConfig(padded_shape.dimensions_size());
  padding_config.mutable_dimensions(0)->set_edge_padding_high(padding_size);

  padded_shape.set_dimensions(0, padded_shape.dimensions(0) + padding_size);

  HloInstruction* padding_value =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(hlo->shape().element_type())));

  return computation->AddInstruction(HloInstruction::CreatePad(
      padded_shape, hlo, padding_value, padding_config));
}

// Returns dense representation of the ragged input tensor.
//
// The dense representation is a tuple of slices of the input tensor, where each
// element of the tuple is an ragged row padded with zeros to the same size as
// the ragged input.
std::vector<HloInstruction*> RaggedToDense(HloComputation* computation,
                                           HloInstruction* ragged_input,
                                           HloInstruction* offsets,
                                           int64_t num_updates_per_replica,
                                           int64_t max_update_size) {
  int64_t num_rows = offsets->shape().dimensions(0);

  std::vector<HloInstruction*> result;

  for (int64_t i = 0; i < num_rows / num_updates_per_replica; ++i) {
    std::vector<HloInstruction*> sliced_operands;
    for (int64_t j = 0; j < num_updates_per_replica; ++j) {
      auto offset_multi_index = GetOffsetMultiIndex(
          computation, offsets, i * num_updates_per_replica + j,
          ragged_input->shape().dimensions_size());

      HloInstruction* padded_input =
          PadOutermostDimension(computation, ragged_input, max_update_size);

      HloInstruction* row_slice =
          computation->AddInstruction(HloInstruction::CreateDynamicSlice(
              ragged_input->shape(), padded_input, offset_multi_index,
              ragged_input->shape().dimensions()));

      row_slice = AddSize1MajorDimension(row_slice, computation);

      sliced_operands.push_back(row_slice);
    }

    auto concat_shape = sliced_operands[0]->shape();
    concat_shape.set_dimensions(0, num_updates_per_replica);
    result.push_back(
        computation->AddInstruction(HloInstruction::CreateConcatenate(
            concat_shape, sliced_operands, /*dimension=*/0)));
  }

  return result;
}

// Returns ragged representation of the dense output tensor.
HloInstruction* DenseToRagged(HloComputation* computation,
                              HloInstruction* dense_inputs,
                              HloInstruction* ragged_output,
                              HloInstruction* offsets, HloInstruction* sizes,
                              int64_t num_updates_per_replica,
                              int64_t max_update_size) {
  int64_t num_rows = offsets->shape().dimensions(0);
  int64_t rank = ragged_output->shape().dimensions_size();

  Shape original_shape = ragged_output->shape();

  // Pad the outermost dimension of the ragged output by dense inputs update
  // size. This is needed to be able to insert updates with dynamic-update-slice
  // to the ragged output.
  HloInstruction* padded_ragged_output =
      PadOutermostDimension(computation, ragged_output,
                            /*padding_size=*/max_update_size);

  for (int64_t i = 0; i < num_rows / num_updates_per_replica; ++i) {
    for (int64_t j = 0; j < num_updates_per_replica; ++j) {
      int idx = i * num_updates_per_replica + j;
      auto offset_multi_index =
          GetOffsetMultiIndex(computation, offsets, idx,
                              padded_ragged_output->shape().dimensions_size());

      // `dense_inputs` is a tuple of updates for each replica. The number of
      // elements in the tuple is equal to the number of replicas.
      HloInstruction* updates = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(dense_inputs, i));

      // `updates` from each replica contains `num_updates_per_replica` rows.
      HloInstruction* update = GetRowSlice(updates, j);

      HloInstruction* padding_value =
          computation->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(padded_ragged_output->shape().element_type())));

      HloInstruction* zero_broadcast =
          computation->AddInstruction(HloInstruction::CreateBroadcast(
              padded_ragged_output->shape(), padding_value, {}));

      // Pad the update with zeros so significant value start from the `offset`
      // position. We can't use HLO pad instruction because it doesn't support
      // dynamic offsets. Instead we create an array of zeros and use
      // dynamic-update-slice to insert the update at the correct position.
      HloInstruction* padded_update =
          computation->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              padded_ragged_output->shape(), zero_broadcast, update,
              offset_multi_index));

      HloInstruction* offset_value = GetScalarValue(offsets, idx);
      HloInstruction* size_value = GetScalarValue(sizes, idx);

      // We can't use `dynamic-update-slice` because it doesn't support
      // dynamic slice sizes, only dynamic offsets.
      // Instead we generate a bit mask with 1s in the positions where the
      // update should be inserted and 0s elsewhere. Then we use HLO `select`
      // instruction to insert the update only in the positions where the mask
      // is 1.
      HloInstruction* update_mask = CreateUpdateMask(
          offset_value, size_value, idx, padded_ragged_output->shape());

      padded_ragged_output =
          computation->AddInstruction(HloInstruction::CreateTernary(
              padded_ragged_output->shape(), HloOpcode::kSelect, update_mask,
              padded_update, padded_ragged_output));
    }
  }

  // Slice the padded ragged output back to the original shape.
  ragged_output = computation->AddInstruction(HloInstruction::CreateSlice(
      original_shape, padded_ragged_output, std::vector<int64_t>(rank, 0),
      original_shape.dimensions(), std::vector<int64_t>(rank, 1)));

  return ragged_output;
}

// Rewrites a ragged all-to-all to a sequence dynamic-slicer, an all-to-all,
// and a sequence dynamic-update-slices.
absl::StatusOr<bool> DecomposeRaggedAllToAll(HloInstruction* hlo,
                                             HloComputation* computation,
                                             HloModule* module) {
  HloRaggedAllToAllInstruction* all_to_all =
      Cast<HloRaggedAllToAllInstruction>(hlo);

  TF_ASSIGN_OR_RETURN(auto replica_group_count_and_size,
                      GetReplicaGroupCountAndSize(all_to_all));
  if (!replica_group_count_and_size.has_value()) {
    return false;
  }

  int64_t num_participating_devices = replica_group_count_and_size->second;

  HloInstruction* input_operand = all_to_all->mutable_operand(0);
  HloInstruction* output_operand = all_to_all->mutable_operand(1);

  HloInstruction* input_offsets = all_to_all->mutable_operand(2);
  HloInstruction* output_offsets = all_to_all->mutable_operand(4);
  HloInstruction* recv_sizes = all_to_all->mutable_operand(5);

  int64_t num_total_updates = input_offsets->shape().dimensions(0);
  int64_t num_updates_per_replica =
      num_total_updates / num_participating_devices;
  int64_t max_update_size = input_operand->shape().dimensions(0);

  // Runs all-to-all to exchange output offsets for each participating device.
  // RaggedAllToAll API requires that output offsets are calculated from the
  // perspective of the target buffer to be used to push updates with memcpy. To
  // make it work with this pass, we need to exchange output offsets to get them
  // from the perspective of the local buffer.
  output_offsets = RunAllToAllOnOutputOffsets(
      computation, all_to_all, output_offsets, num_updates_per_replica,
      num_participating_devices);

  auto dense_input = RaggedToDense(computation, input_operand, input_offsets,
                                   num_updates_per_replica, max_update_size);

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

  auto* ragged_output =
      DenseToRagged(computation, dense_output, output_operand, output_offsets,
                    recv_sizes, num_updates_per_replica, max_update_size);

  TF_RETURN_IF_ERROR(all_to_all->ReplaceAllUsesWith(ragged_output));
  TF_RETURN_IF_ERROR(
      computation->RemoveInstructionAndUnusedOperands(all_to_all));

  return true;
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

      if (hlo->operand(2)->shape().element_type() != S64) {
        return absl::InvalidArgumentError(
            "RaggedAllToAllDecomposer only supports S64 offsets. Was "
            "`ragged-all-to-all-canonicalizer` pass executed?");
      }

      TF_ASSIGN_OR_RETURN(bool result,
                          DecomposeRaggedAllToAll(hlo, computation, module));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
