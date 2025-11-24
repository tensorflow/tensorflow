/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the apecific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/transforms/simplifiers/all_gather_pad_ds_simplifier.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Validates initial properties of the HloPadInstruction.
bool ValidateInitialPadProperties(const HloPadInstruction& pad,
                                  const Shape& ds_shape,
                                  const Shape& ag_shape) {
  // 1. Check if the padding value is a zero constant.
  if (pad.padding_value() == nullptr ||
      pad.padding_value()->opcode() != HloOpcode::kConstant ||
      !pad.padding_value()->literal().IsZero({})) {
    VLOG(2) << "Pad value is not a zero constant: " << pad.ToString()
            << " value: " << pad.padding_value()->ToString();
    return false;
  }

  // 2. Check if the padding config dimension size matches the ag/ds shapes.
  if (pad.padding_config().dimensions_size() != ag_shape.dimensions().size() ||
      pad.padding_config().dimensions_size() != ds_shape.dimensions().size()) {
    VLOG(2) << "pad dimension size mismatches the ag/ds dimension size: "
            << pad.ToString();
    return false;
  }
  return true;
}

// Processes the padding configuration for the split dimension.
std::optional<OffsetSpec> ProcessSplitDimensionPadding(
    const HloPadInstruction& pad,
    const PaddingConfig::PaddingConfigDimension& padding_dim, int64_t dim) {
  const bool has_low_padding = padding_dim.edge_padding_low() != 0;
  const bool has_high_padding = padding_dim.edge_padding_high() != 0;

  if (!has_low_padding && !has_high_padding) {
    VLOG(2) << "Do not support no padding on the split dim: " << dim
            << " pad:" << pad.ToString();
    return std::nullopt;
  }
  if (has_low_padding && has_high_padding) {
    VLOG(2) << "Do not support pad has both low and high edge padding on "
               "the split dim: "
            << dim << " pad:" << pad.ToString();
    return std::nullopt;
  }

  OffsetSpec valid_pad_spec;
  valid_pad_spec.split_dim = dim;
  if (has_low_padding) {
    valid_pad_spec.start_offset = 0;
    valid_pad_spec.end_offset = padding_dim.edge_padding_low();
  } else {
    CHECK(has_high_padding);
    valid_pad_spec.start_offset = pad.operand(0)->shape().dimensions(dim);
    valid_pad_spec.end_offset = pad.operand(0)->shape().dimensions(dim) +
                                padding_dim.edge_padding_high();
    CHECK_EQ(valid_pad_spec.end_offset, pad.shape().dimensions(dim));
  }
  return valid_pad_spec;
}

// Processes the padding configuration for non-split dimensions.
bool ProcessNonSplitDimensionPadding(
    const HloPadInstruction& pad,
    const PaddingConfig::PaddingConfigDimension& padding_dim, int64_t dim) {
  const bool has_low_padding = padding_dim.edge_padding_low() != 0;
  const bool has_high_padding = padding_dim.edge_padding_high() != 0;
  if (has_low_padding || has_high_padding) {
    VLOG(2) << "Pad has edge padding on non-split dim " << dim << ": "
            << pad.ToString();
    return false;
  }
  return true;
}

std::optional<OffsetSpec> ExtractValidPadSpec(const HloPadInstruction& pad,
                                              const Shape& ds_shape,
                                              const Shape& ag_shape,
                                              int64_t split_dim) {
  if (!ValidateInitialPadProperties(pad, ds_shape, ag_shape)) {
    return std::nullopt;
  }
  CHECK_LT(split_dim, pad.padding_config().dimensions_size());

  std::optional<OffsetSpec> valid_pad_spec;

  for (int dim = 0; dim < pad.padding_config().dimensions().size(); ++dim) {
    const PaddingConfig::PaddingConfigDimension& padding_dim =
        pad.padding_config().dimensions(dim);
    VLOG(2) << "padding_dim: " << padding_dim.DebugString();

    if (padding_dim.interior_padding() != 0) {
      VLOG(2) << "Interior padding is not allowed: " << pad.ToString();
      return std::nullopt;
    }

    if (dim == split_dim) {
      std::optional<OffsetSpec> spec =
          ProcessSplitDimensionPadding(pad, padding_dim, dim);
      if (!spec.has_value()) {
        return std::nullopt;
      }
      valid_pad_spec = spec;
    } else {
      if (!ProcessNonSplitDimensionPadding(pad, padding_dim, dim)) {
        return std::nullopt;
      }
    }
  }
  return valid_pad_spec;
}

// Extracts the offset range of the original AllGather data within the padded
// tensor.
//
// Given the OffsetSpec of the padded tensor, this function calculates the start
// and end offsets that the AllGather tensor occupies within the larger tensor
// produced by the Pad instruction. It assumes the padding is only on one side
// (either low or high) along the split dimension.
std::optional<OffsetSpec> ExtractAllGatherOffsetSpec(
    const OffsetSpec& valid_pad_spec, const Shape& pad_shape,
    const Shape& ag_shape) {
  OffsetSpec ag_offset_spec;
  // [valid_pad_spec.start_offset, valid_pad_spec.end_offset] belongs to [0,
  // pad_shape.dimensions(valid_pad_spec.split_dim)]
  int64_t ag_low_edge = valid_pad_spec.start_offset - 0;
  int64_t ag_high_edge = pad_shape.dimensions(valid_pad_spec.split_dim) -
                         valid_pad_spec.end_offset;
  if (ag_low_edge == 0) {
    if (ag_high_edge != ag_shape.dimensions(valid_pad_spec.split_dim)) {
      VLOG(2) << "invalid shape between ag: " << ag_shape.ToString()
              << "and  pad: " << pad_shape.ToString();
      return std::nullopt;
    }
    ag_offset_spec.start_offset = valid_pad_spec.end_offset;
    ag_offset_spec.end_offset = pad_shape.dimensions(valid_pad_spec.split_dim);
    ag_offset_spec.split_dim = valid_pad_spec.split_dim;
  } else if (ag_high_edge == 0) {
    if (ag_low_edge != ag_shape.dimensions(valid_pad_spec.split_dim)) {
      VLOG(2) << "invalid shape between ag: " << ag_shape.ToString()
              << "and  pad: " << pad_shape.ToString();
      return std::nullopt;
    }
    ag_offset_spec.start_offset = 0;
    ag_offset_spec.end_offset = valid_pad_spec.start_offset;
    ag_offset_spec.split_dim = valid_pad_spec.split_dim;
  }
  return ag_offset_spec;
}

std::optional<OffsetToIdMap::const_iterator> GetPartitionIdForOffset(
    const OffsetToIdMap& offset_to_partition_map, int64_t offset) {
  if (offset_to_partition_map.empty()) {
    return std::nullopt;  // Handle empty map case
  }

  // 1. Find the upper_bound
  auto it = offset_to_partition_map.upper_bound(offset);

  // 2. Check if it points to the beginning
  if (it == offset_to_partition_map.begin()) {
    return std::nullopt;  // Provided offset is smaller than the first known
                          // offset in map.
  }
  // 3. Check if `it` points to the end, meaning the provided offset is larger
  // than the last known offset in the map.
  if (it == offset_to_partition_map.end()) {
    return std::prev(it);
  }
  // 4. Get the previous element and return the value.
  return std::prev(it);
}

// Adjusts the keys of the AllGather offset-to-ID map based on the AllGather's
// position within the padded tensor.
//
// The AllGather output is a slice within the result of the Pad instruction.
// This function shifts the offsets in the `ag_map` by the
// `ag_offset_spec.start_offset` to reflect their absolute positions
// in the padded tensor.
std::optional<OffsetToIdMap> GenerateOffsettedAgMap(
    const OffsetToIdMap& ag_map, const OffsetSpec& ag_offset_spec) {
  OffsetToIdMap ag_offsetted_map;
  for (const auto& [offset, partition_id] : ag_map) {
    ag_offsetted_map[offset + ag_offset_spec.start_offset] = partition_id;
  }
  for (const auto& [ag_offsetted_offset, partition_id] : ag_offsetted_map) {
    if (ag_offsetted_offset >= ag_offset_spec.end_offset ||
        ag_offsetted_offset < ag_offset_spec.start_offset) {
      VLOG(2) << "ag_offsetted_offset: " << ag_offsetted_offset
              << " is not in the range of ag_offset_spec: "
              << ag_offset_spec.ToString();
      return std::nullopt;
    }
  }
  return ag_offsetted_map;
}

// Identifies the target partition/offset in the dynamic-slice for
// all-gather.
// Given the offset-to-partition-id maps for both dynamic-slice (ds_map) and
// the offsetted all-gather (ag_offsetted_map), this function determines which
// partition within the dynamic-slice result the all-gather data belongs to.
//
// It checks that all shards of the all-gather map to the *same* partition
// in the dynamic-slice.
//
// Returns a pair containing the start offset and partition ID of the target
// partition in the dynamic-slice map. If the all-gather shards map to
// different dynamic-slice partitions, or if the map is empty, it returns
// std::nullopt.
//
// Example:
//   ds_map: {0:0, 24:1, 48:2, 72:3} (Offsets in the large padded tensor)
//   ag_offsetted_map: {88:0, 90:1, 92:2, 94:3} (Offsets of AG shards in the
//   large padded tensor) Since all offsets in ag_offsetted_map (88-95) fall
//   within the range [72, 96) of partition 3 in ds_map, this function would
//   return {72, 3}.
std::optional<std::pair<int64_t, int64_t>> IdentifyTargetPartition(
    const OffsetToIdMap& ds_map, const OffsetToIdMap& ag_offsetted_map) {
  if (ag_offsetted_map.empty() || ds_map.empty()) {
    VLOG(2) << "The all-gather or dynamic-slice offset map is empty.";
    return std::nullopt;
  }

  int64_t target_partition_id = -1;
  int64_t target_partition_offset = -1;

  for (const auto& [ag_offset, ag_partition_id] : ag_offsetted_map) {
    std::optional<OffsetToIdMap::const_iterator> ds_partition_it =
        GetPartitionIdForOffset(ds_map, ag_offset);

    if (!ds_partition_it.has_value()) {
      VLOG(2) << "All-gather offset " << ag_offset
              << " is out of the range of the dynamic-slice map.";
      return std::nullopt;
    }

    const int64_t current_ds_offset = ds_partition_it.value()->first;
    const int64_t current_ds_partition_id = ds_partition_it.value()->second;

    if (target_partition_id == -1) {
      // First element, initialize the target partition.
      target_partition_id = current_ds_partition_id;
      target_partition_offset = current_ds_offset;
    } else if (target_partition_id != current_ds_partition_id) {
      // Consistency check: all all-gather shards must map to the same
      // dynamic-slice partition.
      VLOG(2) << "All-gather offset " << ag_offset << " maps to partition "
              << current_ds_partition_id << " (offset " << current_ds_offset
              << "), but previous shards mapped to partition "
              << target_partition_id << " (offset " << target_partition_offset
              << ").";
      return std::nullopt;
    }
  }
  return std::make_pair(target_partition_offset, target_partition_id);
}

// Generates an instruction that broadcasts a zero literal to a specified shape.
//
// The output instruction will have the same shape as the given `ds_instr`,
// except for the `split_dim`, where the size is set to the given `size`.
// The element type is taken from `ds_instr`.
//
// Example:
//   ds_instr.shape() = f32[10, 20]
//   split_dim = 1
//   size = 5
//   Returns a broadcast of a zero scalar to a shape of f32[10, 5].
HloInstruction* CreateBroadcastConstant(HloComputation* computation,
                                        const HloInstruction& ds_instr,
                                        int64_t split_dim, int64_t size) {
  Shape new_shape = ds_instr.shape();
  if (size <= 0) {
    VLOG(2) << "Size of the split dimension must be positive, but got " << size;
    return nullptr;
  }
  new_shape.set_dimensions(split_dim, size);

  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(new_shape.element_type())));
  // Broadcast the zero scalar to the target shape.
  HloInstruction* broadcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape, zero, {}));
  return broadcast;
}

// Checks if the total size of operands along the split dimension matches the
// target shape and creates a concatenate instruction.
//
// The function verifies that the sum of the sizes of the `operands` along
// the `split_dim` equals the size of the `split_dim` in the target `shape`.
// If they match, it adds a new concatenate instruction to the computation
// and returns it. Otherwise, it returns nullptr.
HloInstruction* CheckSizeAndConcatenate(HloComputation* computation,
                                        std::vector<HloInstruction*> operands,
                                        const Shape& shape, int64_t split_dim) {
  int64_t total_split_dim_size = 0;
  for (HloInstruction* operand : operands) {
    VLOG(10) << "CheckSizeAndConcatenate operand: " << operand->ToString();
    total_split_dim_size += operand->shape().dimensions(split_dim);
  }

  if (total_split_dim_size != shape.dimensions(split_dim)) {
    VLOG(2) << "Concatenate size mismatch along dimension " << split_dim
            << ". Expected total size: " << shape.dimensions(split_dim)
            << ", but got: " << total_split_dim_size;
    return nullptr;
  }

  return computation->AddInstruction(
      HloInstruction::CreateConcatenate(shape, operands, split_dim));
}

// Creates a predicate instruction based on the current partition ID and a
// partition mask.
//
// The result is a PRED scalar that is true if the value in `select_list`
// at the index of the current partition ID is 1, and false otherwise.
//
// Example HLO:
//   %const_list = s64[8]{0} constant({0, 0, 0, 1, 0, 0, 0, 1})
//   %partition_id = u32[] partition-id()
//   %slice = s64[1]{0} dynamic-slice(%const_list, %partition_id),
//            slice_sizes={1} %reshape = s64[] reshape(%slice)
//   %const_one = s64[] constant(1)
//  %pred = pred[] compare(%reshape, %const_one), direction=EQ
HloInstruction* AddPredInstrBasedOnPartitionIdAndList(
    HloComputation* computation, std::vector<int64_t> select_list) {
  HloInstruction* const_list =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int64_t>(select_list)));

  // Get the partition ID.
  HloInstruction* partition_id =
      computation->AddInstruction(HloInstruction::CreatePartitionId());

  // Dynamic slice the constant list based on the partition ID.
  Shape slice_shape = ShapeUtil::MakeShape(S64, {1});
  HloInstruction* sliced_value =
      computation->AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape, const_list, {partition_id}, /*slice_sizes=*/{1}));

  // Reshape the sliced value to a scalar.
  HloInstruction* scalar_sliced_value =
      computation->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(S64, {}), sliced_value));

  // Create a constant 1 for comparison.
  HloInstruction* const_one = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));

  // Compare the sliced value with 1.
  HloInstruction* pred =
      computation->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), scalar_sliced_value, const_one,
          ComparisonDirection::kEq));

  return pred;
}

// Creates a new CollectivePermute instruction by adding new source-target pairs
// to an existing CollectivePermute instruction.
HloInstruction* AddCollectivePermuteWithNewSourceTargetPair(
    HloCollectivePermuteInstruction* cp,
    const std::vector<std::pair<int64_t, int64_t>>& new_source_target_pairs) {
  std::vector<std::pair<int64_t, int64_t>> combined_pairs =
      new_source_target_pairs;
  combined_pairs.insert(combined_pairs.end(), cp->source_target_pairs().begin(),
                        cp->source_target_pairs().end());

  // TODO(wfelix): pass the channel id from the caller to avoid the expensive
  // channel id query and enable parallelism.
  return cp->parent()->AddInstruction(HloInstruction::CreateCollectivePermute(
      cp->shape(), cp->mutable_operand(0), combined_pairs,
      hlo_query::NextChannelId(*cp->GetModule())));
}

// Processes a single replica group to generate the necessary HLO instructions.
//
// This function identifies the target partition within the dynamic-slice window
// for the AllGather data of the current replica group. It then builds a
// vector of HLO instructions, including CollectivePermutes to fetch remote data
// shards, the local AllGather operand, and zero padding where necessary, to
// construct the slice for this replica group.
//
// Returns a vector of HLO instructions if successful, or std::nullopt if any
// validation fails or the configuration is unsupported.
// The target_partition_id is an output parameter.
std::optional<std::vector<HloInstruction*>> ProcessReplicaGroup(
    const OffsetToIdMap& ds_map, const OffsetToIdMap& ag_map,
    const OffsetSpec& ag_offset_spec, HloComputation* computation,
    HloDynamicSliceInstruction& ds, HloAllGatherInstruction& ag,
    const int64_t split_dim, int64_t& target_partition_id) {
  const int64_t ds_result_size = ds.shape().dimensions(split_dim);
  const int64_t ag_shard_size = ag.operand(0)->shape().dimensions(split_dim);

  std::optional<OffsetToIdMap> ag_offsetted_map =
      GenerateOffsettedAgMap(ag_map, ag_offset_spec);
  if (!ag_offsetted_map.has_value()) {
    VLOG(2) << "No valid ag offsetted map found for ag offset spec: "
            << ag_offset_spec.ToString();
    return std::nullopt;
  }

  std::optional<std::pair<int64_t, int64_t>> target_partition =
      IdentifyTargetPartition(ds_map, ag_offsetted_map.value());
  if (!target_partition.has_value()) {
    VLOG(2) << "No valid target partition found for ag offset spec: "
            << ag_offset_spec.ToString();
    return std::nullopt;
  }

  target_partition_id = target_partition.value().second;
  const int64_t target_partition_offset = target_partition.value().first;

  std::vector<HloInstruction*> new_instrs_per_rg;
  int64_t start_offset = target_partition_offset;
  int64_t end_offset = target_partition_offset + ds_result_size;

  if (ag_offsetted_map.value().rbegin()->first + ag_shard_size > end_offset) {
    VLOG(2) << "ag_offsetted_map.rbegin()->first + ag_shard_size: "
            << ag_offsetted_map.value().rbegin()->first + ag_shard_size
            << " is out of range of end_offset: " << end_offset;
    return std::nullopt;
  }

  // Add zero padding at the beginning if necessary.
  OffsetToIdMap::iterator iter = ag_offsetted_map.value().begin();
  if (iter->first < start_offset) {
    return std::nullopt;
  }
  if (iter->first > start_offset) {
    new_instrs_per_rg.push_back(CreateBroadcastConstant(
        computation, ds, split_dim, iter->first - start_offset));
  }

  // Add instructions for each all-gather shard: if the shard is not on the
  // target partition, insert a collective permute; otherwise, use the all-
  // gather operand directly.
  while (iter != ag_offsetted_map.value().end()) {
    if (iter->first >= end_offset) {
      VLOG(2) << "ag iter offset: " << iter->first
              << " is out of range of end_offset: " << end_offset;
      return std::nullopt;
    }

    if (iter->second != target_partition_id) {
      // Insert collective permute from iter->second to target_partition_id.
      new_instrs_per_rg.push_back(
          computation->AddInstruction(HloInstruction::CreateCollectivePermute(
              ag.operand(0)->shape(), ag.mutable_operand(0),
              {{iter->second, target_partition_id}},
              hlo_query::NextChannelId(*ag.GetModule()))));
    } else {
      new_instrs_per_rg.push_back(ag.mutable_operand(0));
    }
    iter++;
  }

  // Add zero padding at the end if necessary.
  int64_t ag_end_offset =
      std::prev(ag_offsetted_map.value().end())->first + ag_shard_size;
  if (ag_end_offset < end_offset) {
    new_instrs_per_rg.push_back(CreateBroadcastConstant(
        computation, ds, split_dim, end_offset - ag_end_offset));
  }
  for (const HloInstruction* instr : new_instrs_per_rg) {
    VLOG(10) << "new_instrs_per_rg added: " << instr->ToString();
  }
  return new_instrs_per_rg;
}

// Condenses the instruction lists generated for each replica group into a
// single list of instructions.
//
// This function takes a list of instruction lists (one for each replica group)
// and merges them. It ensures that the instruction lists are compatible (same
// size and matching opcodes). CollectivePermute instructions are combined by
// merging their source-target pairs.
//
// Returns a single vector of HloInstructions if successful, or std::nullopt if
// the lists cannot be condensed (e.g., size or opcode mismatches).
std::optional<std::vector<HloInstruction*>> CondenseInstructionLists(
    const std::vector<std::vector<HloInstruction*>>& new_instrs_list) {
  if (new_instrs_list.empty()) {
    VLOG(2) << "No instructions generated for any replica group.";
    return std::nullopt;
  }
  std::vector<HloInstruction*> final_instrs;
  final_instrs.reserve(new_instrs_list[0].size());
  for (HloInstruction* instr : new_instrs_list[0]) {
    final_instrs.push_back(instr);
  }

  for (size_t rg_idx = 1; rg_idx < new_instrs_list.size(); ++rg_idx) {
    const std::vector<HloInstruction*>& appending_instrs =
        new_instrs_list[rg_idx];
    if (final_instrs.size() != appending_instrs.size()) {
      VLOG(2) << "final_instrs.size(): " << final_instrs.size()
              << " != appending_instrs.size(): " << appending_instrs.size();
      return std::nullopt;
    }
    for (size_t i = 0; i < final_instrs.size(); ++i) {
      if (final_instrs[i]->opcode() != appending_instrs[i]->opcode()) {
        VLOG(2) << "Opcode mismatch: final_instrs[i]: between "
                << final_instrs[i]->ToString() << " and appending_instrs[i]: "
                << appending_instrs[i]->ToString();
        return std::nullopt;
      }
      if (final_instrs[i]->opcode() == HloOpcode::kCollectivePermute) {
        HloCollectivePermuteInstruction* cp =
            Cast<HloCollectivePermuteInstruction>(appending_instrs[i]);
        final_instrs[i] = AddCollectivePermuteWithNewSourceTargetPair(
            Cast<HloCollectivePermuteInstruction>(final_instrs[i]),
            cp->source_target_pairs());
      }
    }
  }
  return final_instrs;
}

// Creates the final Concatenate and Select instructions.
//
// This function takes the condensed list of instructions, concatenates them,
// and then uses a predicate based on the partition ID to select either the
// concatenated result or a zero-filled tensor.
//
// Returns the final Select instruction if successful, or std::nullopt
// otherwise.
std::optional<HloInstruction*> CreateFinalConcatAndSelect(
    HloComputation* computation,
    const std::vector<HloInstruction*>& final_instrs,
    HloDynamicSliceInstruction& ds, const int64_t split_dim,
    const std::vector<int64_t>& selection_list, const int64_t ds_result_size) {
  HloInstruction* concat =
      CheckSizeAndConcatenate(computation, final_instrs, ds.shape(), split_dim);
  if (concat == nullptr) {
    VLOG(2) << "No valid concat instruction generated.";
    return std::nullopt;
  }

  HloInstruction* pred =
      AddPredInstrBasedOnPartitionIdAndList(computation, selection_list);
  if (pred == nullptr) {
    VLOG(2) << "No valid pred instruction generated.";
    return std::nullopt;
  }

  HloInstruction* zero_filled =
      CreateBroadcastConstant(computation, ds, split_dim, ds_result_size);
  if (zero_filled == nullptr) {
    VLOG(2) << "No valid zero_filled instruction generated.";
    return std::nullopt;
  }

  return computation->AddInstruction(HloInstruction::CreateTernary(
      ds.shape(), HloOpcode::kSelect, pred, concat, zero_filled));
}

// Replaces a DynamicSlice(Pad(AllGather)) pattern with a more efficient
// sequence of HLOs, primarily using CollectivePermute and Concatenate.
//
// Purpose:
// This function aims to optimize the case where a sharded AllGather result is
// padded and then sliced by DynamicSlice. Instead of materializing the large
// padded tensor, it constructs the result for the current partition's slice
// by directly permuting the necessary data from other partitions.
//
// Workflow:
// 1. Initial Validation: Checks for size compatibility and valid partition
// specs.
// 2. Per-Replica Group Processing: For each replica group:
//    a. Identify Target Partition: Determines which partition's DynamicSlice
//       window the AllGather data falls into.
//    b. Build Instruction List: Generates HLOs to form the slice. This
//    involves:
//       - CollectivePermute: To fetch data shards from remote partitions.
//       - AllGather Operand: To use the local shard data.
//       - Zero Padding: To fill gaps at the start or end of the slice if the
//         AllGather data doesn't perfectly align.
// 3. Condense Instruction Lists: Merges the instruction lists from each replica
//    group. For CollectivePermute instructions, source-target pairs are
//    combined.
// 4. Final HLO Generation: Constructs the final set of instructions:
//    a. Concatenate: Assembles the (potentially permuted) data shards and
//       padding into the shape of the DynamicSlice output.
//    b. Predicate: Creates a predicate based on the current partition ID to
//       determine if this partition is the target for the AllGather data.
//    c. Select: Uses the predicate to either output the concatenated data (if
//       it's the target partition) or a zero-filled tensor of the same shape.
std::optional<HloInstruction*> InsertCollectivePermuteInstrSet(
    const PartitionOffsetSpec& ds_offset_partition_spec,
    const PartitionOffsetSpec& ag_offset_partition_spec,
    const OffsetSpec& ag_offset_spec, HloDynamicSliceInstruction& ds,
    HloAllGatherInstruction& ag, const int64_t split_dim) {
  // Section 1: Initial Validation and Setup
  const int64_t ds_result_size = ds.shape().dimensions(split_dim);
  const int64_t ag_result_size = ag.shape().dimensions(split_dim);
  HloComputation* computation = ds.parent();

  // Validate the ag shard can fit into a single ds partition.
  if (ds_result_size < ag_result_size) {
    VLOG(2) << "ds_result_size(" << ds_result_size << ") < ag_result_size("
            << ag_result_size << ").";
    return std::nullopt;
  }
  if (ds_offset_partition_spec.per_replica_group_offsets.size() !=
          ag_offset_partition_spec.per_replica_group_offsets.size() ||
      ds_offset_partition_spec.per_replica_group_offsets.empty()) {
    VLOG(2)
        << "ds_offset_partition_spec or ag_offset_partition_spec is not valid.";
    return std::nullopt;
  }

  std::vector<int64_t> selection_list;
  selection_list.resize(ag.replica_groups().size() *
                        ag.replica_groups()[0].replica_ids_size());
  std::vector<std::vector<HloInstruction*>> new_instrs_list;

  // Section 2: Per-Replica Group Processing
  for (int64_t rg_idx = 0;
       rg_idx < ds_offset_partition_spec.per_replica_group_offsets.size();
       ++rg_idx) {
    const OffsetToIdMap& ds_map =
        ds_offset_partition_spec.per_replica_group_offsets[rg_idx];
    const OffsetToIdMap& ag_map =
        ag_offset_partition_spec.per_replica_group_offsets[rg_idx];

    int64_t target_partition_id = -1;
    std::optional<std::vector<HloInstruction*>> new_instrs_per_rg =
        ProcessReplicaGroup(ds_map, ag_map, ag_offset_spec, computation, ds, ag,
                            split_dim, target_partition_id);

    if (!new_instrs_per_rg.has_value()) {
      VLOG(2) << "Failed to process replica group " << rg_idx;
      return std::nullopt;
    }
    if (target_partition_id >= selection_list.size()) {
      VLOG(2) << "target_partition_id: " << target_partition_id
              << " is out of range of selection_list: "
              << selection_list.size();
      return std::nullopt;
    }
    // Select the concatenate result on target partition only.
    selection_list[target_partition_id] = 1;
    new_instrs_list.push_back(std::move(new_instrs_per_rg.value()));
  }

  // Section 3: Condense Instruction Lists from Replica Groups.
  // only one instruction set will be used, all other instruction sets will
  // be discarded in following passes.
  std::optional<std::vector<HloInstruction*>> final_instrs_optional =
      CondenseInstructionLists(new_instrs_list);
  if (!final_instrs_optional.has_value()) {
    VLOG(2) << "Failed to condense instruction lists.";
    return std::nullopt;
  }
  std::vector<HloInstruction*> final_instrs = final_instrs_optional.value();

  // Section 4: Create Final Concatenation and Selection
  return CreateFinalConcatAndSelect(computation, final_instrs, ds, split_dim,
                                    selection_list, ds_result_size);
}

// Checks if the partition IDs within each replica group in the
// `PartitionOffsetSpec` are strictly ascending
// with respect to their corresponding offsets.
bool IsAscendingPartitionId(const PartitionOffsetSpec& offset_spec) {
  if (offset_spec.per_replica_group_offsets.empty()) {
    return false;
  }
  for (const OffsetToIdMap& rg_map : offset_spec.per_replica_group_offsets) {
    OffsetToIdMap::const_iterator iter = rg_map.begin();
    int64_t prev_id = iter->second;

    while (++iter != rg_map.end()) {
      if (iter->second <= prev_id) {
        VLOG(2) << "iter->second: " << iter->second
                << " <= prev_id: " << prev_id;
        return false;
      }
      prev_id = iter->second;
    }
  }
  return true;
}

struct PatternMatchResult {
  HloDynamicSliceInstruction* ds;
  HloPadInstruction* pad;
  HloAllGatherInstruction* ag;
};

// Matches the pattern: dynamic-slice(pad(all-gather)).
std::optional<PatternMatchResult> MatchDynamicSlicePadAllGather(
    HloInstruction* dynamic_slice_hlo) {
  HloDynamicSliceInstruction* dynamic_slice =
      Cast<HloDynamicSliceInstruction>(dynamic_slice_hlo);

  // TODO(wfelix): remove the module access and pass config through the callers.
  const HloModuleConfig& config = dynamic_slice->GetModule()->config();
  // Match the module level scope.
  if (config.num_partitions() <= 1 || config.replica_count() > 1) {
    VLOG(2) << "Do not match module with num_partitions: "
            << config.num_partitions()
            << " and replica_count: " << config.replica_count();
    return std::nullopt;
  }

  // Match the pattern: dynamic-slice(pad(all-gather)).
  HloInstruction* pad_hlo = nullptr;
  HloInstruction* ag_hlo = nullptr;
  if (!MatchDsPadAllGather(dynamic_slice, &pad_hlo, &ag_hlo)) {
    VLOG(2) << "Do not find pattern for dynamic-slice(pad(ag)) "
            << dynamic_slice->ToString();
    return std::nullopt;
  }
  CHECK_EQ(pad_hlo->opcode(), HloOpcode::kPad);
  CHECK_EQ(ag_hlo->opcode(), HloOpcode::kAllGather);

  // Match all-gather for kFlattenedID collective mode.
  absl::StatusOr<CollectiveOpGroupMode> mode = GetCollectiveOpGroupMode(ag_hlo);

  if (!mode.ok() ||
      mode.value() !=
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID) {
    VLOG(2) << "AG does not use global device ids or channel id "
            << ag_hlo->ToString();
    return std::nullopt;
  }

  return PatternMatchResult{dynamic_slice, Cast<HloPadInstruction>(pad_hlo),
                            Cast<HloAllGatherInstruction>(ag_hlo)};
}

std::optional<SplitDimSpec> ExtractAndValidateSplitDim(
    const HloDynamicSliceInstruction& dynamic_slice,
    const HloAllGatherInstruction& all_gather) {
  // Extract and validate the split dimension all partitions from the dynamic
  // slice.
  std::optional<SplitDimSpec> split_dim_spec =
      ExtractSplitDimSpec(dynamic_slice, /*allow_multiple_split_dims*/ false);

  // Match split dimension check:
  //  1. the split dim is unique.
  //  2. the split dim is the same as the all gather dim.
  if (!split_dim_spec.has_value() || split_dim_spec->split_dims.size() != 1 ||
      split_dim_spec->split_dim != all_gather.all_gather_dimension()) {
    VLOG(2) << "In dynamic-slice(pad(ag)), ds split dim mismatch "
            << dynamic_slice.ToString()
            << " or all-gather: " << all_gather.ToString();
    return std::nullopt;
  }
  VLOG(10) << "Identified split dim for dynamic-slice "
           << split_dim_spec->split_dim
           << " with split_dim_size: " << split_dim_spec->split_dim_size;
  return split_dim_spec;
}

struct PadOffsetSpecs {
  OffsetSpec pad_spec;
  OffsetSpec ag_spec;
};

std::optional<PadOffsetSpecs> ValidatePadAndExtractOffsetSpecs(
    const HloPadInstruction& pad, const Shape& ds_shape, const Shape& ag_shape,
    int64_t split_dim) {
  // Match the pad pattern:
  //  1. pad only one edge padding
  //  2. pad value == 0
  //  3. pad size is large: ag result size is smaller than dynamic slice result
  //  size.
  std::optional<OffsetSpec> valid_pad_split_offset_spec =
      ExtractValidPadSpec(pad, ds_shape, ag_shape, split_dim);
  if (!valid_pad_split_offset_spec.has_value()) {
    VLOG(2) << "No valid pad spec found from pad: " << pad.ToString();
    return std::nullopt;
  }

  VLOG(10) << "Got valid_pad_split_offset_spec: "
           << valid_pad_split_offset_spec->ToString();

  // Extracts the all-gather offset spec on dynamic-slice result
  // based on pad split offset spec.
  std::optional<OffsetSpec> valid_ag_split_offset_spec =
      ExtractAllGatherOffsetSpec(*valid_pad_split_offset_spec, pad.shape(),
                                 ag_shape);
  if (!valid_ag_split_offset_spec.has_value()) {
    VLOG(2) << "No valid ag offset spec found from pad: " << pad.ToString()
            << ", ag: " << ag_shape.ToString() << ", pad_spec"
            << valid_pad_split_offset_spec->ToString();
    return std::nullopt;
  }

  VLOG(10) << "Got valid_ag_offset_spec: "
           << valid_ag_split_offset_spec->ToString();
  return PadOffsetSpecs{valid_pad_split_offset_spec.value(),
                        valid_ag_split_offset_spec.value()};
}

struct OffsetToIdMaps {
  PartitionOffsetSpec ds_spec;
  PartitionOffsetSpec ag_spec;
};

std::optional<OffsetToIdMaps> ExtractAndValidateOffsetToIdMaps(
    const HloDynamicSliceInstruction& dynamic_slice,
    const HloAllGatherInstruction& all_gather, const HloModuleConfig& config,
    int64_t split_dim, int64_t split_dim_size) {
  // dynamic-slice result should be larger than the all-gather result.
  if (dynamic_slice.shape().dimensions(split_dim) <
      all_gather.shape().dimensions(split_dim)) {
    VLOG(2) << "dynamic-slice result size is smaller than all-gather result "
               "size.";
    return std::nullopt;
  }

  MapIdToTableOffset map_partition_id = [&](const HloInstruction* hlo,
                                            int64_t id) {
    return HloPredicateIsOp<HloOpcode::kPartitionId>(hlo) ? id : -1;
  };

  // Extracts the dynamic slice offset to partition id map spec on the split
  // dimension.
  std::optional<PartitionOffsetSpec> ds_offset_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          &all_gather, dynamic_slice.operand(split_dim + 1), map_partition_id,
          split_dim_size);

  if (!ds_offset_spec.has_value()) {
    VLOG(2) << "No valid ds offset spec on dim" << split_dim
            << "found from dynamic_slice: " << dynamic_slice.ToString()
            << " and ag: " << all_gather.ToString();
    return std::nullopt;
  }

  // Validates the dynamic slice offset list is based on ascending order.
  if (!IsAscendingPartitionId(ds_offset_spec.value())) {
    VLOG(2) << "Partition id is not ascending on dim" << split_dim
            << "found from dynamic_slice: " << dynamic_slice.ToString()
            << " ds offset: "
            << dynamic_slice.operand(split_dim + 1)->ToString();
    return std::nullopt;
  }

  // Extracts the all-gather offset to partition id map spec on the split
  // dimension.
  std::optional<PartitionOffsetSpec> ag_offset_partition_spec =
      ExtractPartitionOffsetSpec(&all_gather, config.num_partitions());
  if (!ag_offset_partition_spec.has_value()) {
    VLOG(2) << "No valid ag offset spec " << all_gather.ToString()
            << " with num_partitions " << config.num_partitions();
    return std::nullopt;
  }
  return OffsetToIdMaps{ds_offset_spec.value(),
                        ag_offset_partition_spec.value()};
}

absl::Status AllGatherPadDsSimplifierVisitor::HandleDynamicSlice(
    HloInstruction* dynamic_slice_hlo) {
  // Match and Validate Pattern
  std::optional<PatternMatchResult> pattern_match =
      MatchDynamicSlicePadAllGather(dynamic_slice_hlo);

  if (!pattern_match.has_value()) {
    return absl::OkStatus();
  }
  HloDynamicSliceInstruction* dynamic_slice = pattern_match.value().ds;
  HloPadInstruction* pad = pattern_match.value().pad;
  HloAllGatherInstruction* all_gather = pattern_match.value().ag;

  // Extract and Validate Split Dimension
  std::optional<SplitDimSpec> split_dim_spec =
      ExtractAndValidateSplitDim(*dynamic_slice, *all_gather);
  if (!split_dim_spec.has_value()) {
    return absl::OkStatus();
  }
  const int64_t split_dim = split_dim_spec->split_dim;
  const int64_t split_dim_size = split_dim_spec->split_dim_size;

  // Validate Pad and Extract Offset Specs
  std::optional<PadOffsetSpecs> offset_specs = ValidatePadAndExtractOffsetSpecs(
      *pad, dynamic_slice->shape(), all_gather->shape(), split_dim);
  if (!offset_specs.has_value()) {
    return absl::OkStatus();
  }
  const OffsetSpec& ag_spec = offset_specs->ag_spec;

  // Extract and Validate OffsetToId Maps
  const HloModuleConfig& config = dynamic_slice->GetModule()->config();
  std::optional<OffsetToIdMaps> offset_maps = ExtractAndValidateOffsetToIdMaps(
      *dynamic_slice, *all_gather, config, split_dim, split_dim_size);
  if (!offset_maps.has_value()) {
    return absl::OkStatus();
  }
  const PartitionOffsetSpec& ds_offset_spec = offset_maps.value().ds_spec;
  const PartitionOffsetSpec& ag_offset_spec = offset_maps.value().ag_spec;

  // HLO Generation
  std::optional<HloInstruction*> selected =
      InsertCollectivePermuteInstrSet(ds_offset_spec, ag_offset_spec, ag_spec,
                                      *dynamic_slice, *all_gather, split_dim);

  if (!selected.has_value() || *selected == nullptr) {
    VLOG(2) << "No valid selected found from dynamic_slice: "
            << dynamic_slice->ToString()
            << " and ag: " << all_gather->ToString();
    return absl::OkStatus();
  }

  // Replacement
  return ReplaceInstruction(dynamic_slice, *selected);
}

absl::StatusOr<bool> AllGatherPadDsSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    AllGatherPadDsSimplifierVisitor visitor;
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace xla
