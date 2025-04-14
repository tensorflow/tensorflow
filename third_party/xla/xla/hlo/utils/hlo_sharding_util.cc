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

#include "xla/hlo/utils/hlo_sharding_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/utils/hlo_container_util.h"
#include "xla/literal_util.h"
#include "xla/map_util.h"
#include "xla/protobuf_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace hlo_sharding_util {

// Apply the formatting steps to get a shardable shape.
HloInstruction* FormatShape(HloInstruction* data,
                            absl::Span<const FormattingStep> formatting_steps,
                            HloComputation* computation) {
  for (const FormattingStep& step : formatting_steps) {
    switch (step.formatting_opcode) {
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy: {
        data = computation->AddInstruction(HloInstruction::CreateUnary(
            step.output_shape, step.formatting_opcode, data));
        break;
      }
      case HloOpcode::kReshape: {
        data = computation->AddInstruction(
            HloInstruction::CreateReshape(step.output_shape, data));
        break;
      }
      case HloOpcode::kPad: {
        PaddingConfig padding_config;
        for (int64_t i = 0; i < step.output_shape.dimensions().size(); ++i) {
          auto padding_config_dim = padding_config.add_dimensions();
          padding_config_dim->set_edge_padding_low(0);
          padding_config_dim->set_interior_padding(0);
          padding_config_dim->set_edge_padding_high(
              step.output_shape.dimensions(i) - data->shape().dimensions(i));
        }
        HloInstruction* padding =
            step.padding_value
                ? step.padding_value
                : computation->AddInstruction(HloInstruction::CreateConstant(
                      LiteralUtil::Zero(step.output_shape.element_type())));
        data = computation->AddInstruction(HloInstruction::CreatePad(
            step.output_shape, data, padding, padding_config));
        break;
      }
      case HloOpcode::kTranspose: {
        CHECK(step.xpose_permutation.has_value());
        data = computation->AddInstruction(HloInstruction::CreateTranspose(
            step.output_shape, data, *step.xpose_permutation));
        break;
      }
      default:
        LOG(FATAL) << "Unsupported formatting step";
    }
  }
  return data;
}

HloInstruction* ReverseFormatShape(
    HloInstruction* data, absl::Span<const FormattingStep> formatting_steps,
    HloComputation* computation) {
  for (int64_t i = formatting_steps.size() - 1; i >= 0; --i) {
    const FormattingStep& step = formatting_steps[i];
    const Shape& previous_shape =
        step.reverse_input_shape ? *step.reverse_input_shape : step.input_shape;
    switch (step.formatting_opcode) {
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy: {
        data = computation->AddInstruction(HloInstruction::CreateUnary(
            previous_shape, step.formatting_opcode, data));
        break;
      }
      case HloOpcode::kReshape: {
        data = computation->AddInstruction(
            HloInstruction::CreateReshape(previous_shape, data));
        break;
      }
      case HloOpcode::kPad: {
        std::vector<int64_t> start_indices(previous_shape.dimensions().size(),
                                           0);
        std::vector<int64_t> strides(previous_shape.dimensions().size(), 1);
        data = computation->AddInstruction(
            HloInstruction::CreateSlice(previous_shape, data, start_indices,
                                        previous_shape.dimensions(), strides));
        break;
      }
      case HloOpcode::kTranspose: {
        CHECK(step.xpose_permutation.has_value());
        std::vector<int64_t> reverse_permutation;
        reverse_permutation.reserve(step.xpose_permutation->size());
        for (int64_t i = 0; i < step.xpose_permutation->size(); ++i) {
          reverse_permutation.push_back(
              absl::c_find(*step.xpose_permutation, i) -
              step.xpose_permutation->begin());
        }
        data = computation->AddInstruction(HloInstruction::CreateTranspose(
            previous_shape, data, reverse_permutation));
        break;
      }
      default:
        LOG(FATAL) << "Unsupported formatting step";
    }
  }
  return data;
}

void GatherScatterDims::append(const GatherScatterDims& other) {
  operand_dims.insert(operand_dims.end(), other.operand_dims.begin(),
                      other.operand_dims.end());
  indices_dims.insert(indices_dims.end(), other.indices_dims.begin(),
                      other.indices_dims.end());
  output_dims.insert(output_dims.end(), other.output_dims.begin(),
                     other.output_dims.end());
}

void GatherScatterDims::FillOutputDimsWithIndicesDims(
    int64_t index_vector_dim, absl::Span<const int64_t> offset_or_window_dims) {
  int64_t max_indices_dim =
      *std::max_element(indices_dims.begin(), indices_dims.end());
  absl::flat_hash_map<int64_t, int64_t> indices_dim_to_output_dim;
  for (int64_t indices_dim = 0, output_dim = 0; indices_dim <= max_indices_dim;
       indices_dim++, output_dim++) {
    if (indices_dim == index_vector_dim) {
      indices_dim++;
    }
    while (absl::c_linear_search(offset_or_window_dims, output_dim)) {
      output_dim++;
    }
    indices_dim_to_output_dim[indices_dim] = output_dim;
  }

  CHECK(output_dims.empty());
  output_dims.reserve(indices_dims.size());
  for (int64_t indices_dim : indices_dims) {
    CHECK_NE(indices_dim, index_vector_dim);
    output_dims.push_back(indices_dim_to_output_dim[indices_dim]);
  }
}

bool IsSubTilingOrEqualSharding(const Shape& potential_sharded_shape,
                                const HloSharding& potential_subsharding,
                                const HloSharding& sharding) {
  // Some early exit cases.
  // If any manual sharding return false.
  if (potential_subsharding.IsManual() || sharding.IsManual()) {
    return false;
  }
  // If the tile we are comparing with is maximal, then we are guaranteed to be
  // equal or contained in it.
  if (sharding.IsTileMaximal()) {
    return true;
  }
  // If the subsharding tile is maximal and the sharding we are comparing with
  // is not then it can't be contained.
  if (potential_subsharding.IsTileMaximal()) {
    return false;
  }
  const int32_t tiled_data_rank = potential_subsharding.TiledDataRank();
  // Different tiled ranks can't be compared (something is wrong, are the
  // shardings for different shapes?)
  if (tiled_data_rank != sharding.TiledDataRank() ||
      tiled_data_rank != potential_sharded_shape.dimensions().size()) {
    return false;
  }

  DimensionVector potential_base_tile(tiled_data_rank);
  DimensionVector base_tile(tiled_data_rank);
  bool shortcut = true;
  int64_t diff_dim_counter = 0;
  DimensionVector reshape_dims(
      potential_subsharding.tile_assignment().dimensions().begin(),
      potential_subsharding.tile_assignment().dimensions().end());
  for (int64_t i = 0; i < tiled_data_rank; ++i) {
    const auto shape_i = potential_sharded_shape.dimensions(i);
    const auto p_tile_dim_i = potential_subsharding.tile_assignment().dim(i);
    const auto s_tile_dim_i = sharding.tile_assignment().dim(i);
    if (p_tile_dim_i < s_tile_dim_i) {
      return false;
    }
    potential_base_tile[i] = CeilOfRatio(shape_i, p_tile_dim_i);
    base_tile[i] = CeilOfRatio(shape_i, s_tile_dim_i);

    if (s_tile_dim_i != 1 &&
        (p_tile_dim_i % s_tile_dim_i != 0 ||
         base_tile[i] % potential_base_tile[i] != 0 ||
         shape_i <= (p_tile_dim_i - 1) * potential_base_tile[i] ||
         shape_i <= (s_tile_dim_i - 1) * base_tile[i])) {
      // The comment below explains this condition.
      shortcut = false;
    }
    if (shortcut && p_tile_dim_i != s_tile_dim_i) {
      reshape_dims[i + diff_dim_counter] = s_tile_dim_i;
      reshape_dims.insert(reshape_dims.begin() + i + diff_dim_counter + 1,
                          p_tile_dim_i / s_tile_dim_i);
      diff_dim_counter++;
    }
  }

  if (shortcut) {
    // In the shortcut, we ensure that (1) p_tile_dim_i is divisible by
    // s_tile_dim_i, (2) base_tile[i] is divisible by potential_base_tile[i],
    // and (3) all devices have raw data of the tensor (a counterexample is that
    // a device may only have paddings). We can use this shortcut to quickly
    // make the decision.
    //
    // s_tile_dim_i == 1 means that it is replicated along dimension i with
    // `sharding`, which is compatible with the shortcut.
    //
    // We cannot extend the shortcut if the condition fails. An example is
    // listed below. Given potential_sharded_shape = [1, 1, 1, ..., 1], the raw
    // data of the tensor is only on the first tile. Thus, we only need to focus
    // on the first tile in the two input shardings.
    if (!sharding.HasPartialReplication()) {
      return potential_subsharding == sharding;
    }

    std::vector<int> perm(reshape_dims.size());
    absl::c_iota(perm, 0);
    for (int64_t i = 0; i < tiled_data_rank; ++i) {
      if (potential_subsharding.tile_assignment().dim(i) !=
          sharding.tile_assignment().dim(i)) {
        auto element = perm[i + 1];
        perm.erase(perm.begin() + i + 1);
        perm.push_back(element);
      }
    }

    auto reshaped_ta = potential_subsharding.tile_assignment()
                           .Reshape(reshape_dims)
                           .Transpose(perm)
                           .Reshape(sharding.tile_assignment().dimensions());
    return HloSharding::PartialTile(reshaped_ta).tile_assignment() ==
           sharding.tile_assignment();
  }

  // Use one contiguous storage to reduce allocation overhead.
  auto storage = std::make_unique<int32_t[]>(
      sharding.tile_assignment().num_elements() * tiled_data_rank);
  int32_t* storage_cursor = storage.get();
  // Need a map here, because the MPMD partitioner sharding annotations can have
  // non contiguous partition numbers.
  absl::flat_hash_map<int32_t, int32_t*> sharding_offsets;
  sharding_offsets.reserve(sharding.tile_assignment().num_elements());
  auto get_sharding_offsets = [&](int64_t device) -> absl::Span<int32_t> {
    auto it = sharding_offsets.find(device);
    if (it == sharding_offsets.end()) {
      bool emplaced;
      std::tie(it, emplaced) = sharding_offsets.emplace(device, storage_cursor);
      DCHECK(emplaced);
      storage_cursor += tiled_data_rank;
    }
    return absl::MakeSpan(it->second, tiled_data_rank);
  };
  // Collect the start offsets for each tile for the sharding we are evaluating
  // against.
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t device) {
        auto indices_per_device = get_sharding_offsets(device);
        for (int64_t i = 0; i < tiled_data_rank; ++i) {
          indices_per_device[i] = base_tile[i] * indices[i];
        }
      });
  // Compare the start offsets and the end offset of the tiles for each device.
  auto& potential_ta = potential_subsharding.tile_assignment().array();
  absl::Status ok_if_no_violation = potential_ta.EachStatus(
      [&](absl::Span<const int64_t> indices, int64_t device) -> absl::Status {
        auto sharding_offset = get_sharding_offsets(device);
        for (int j = 0; j < tiled_data_rank; ++j) {
          const int32_t subsharding_offset_j =
              potential_base_tile[j] * indices[j];
          // The subsharding contains data outside of the tile we are comparing
          // against.
          if (subsharding_offset_j < sharding_offset[j]) {
            return Internal("");
          }
          // Skip last tile. It can never go beyond the limit as the shape is
          // the same for both shardings and sometimes there's padding making
          // one of the two limits bigger than the other, but it shouldn't be
          // counted.
          if (subsharding_offset_j + potential_base_tile[j] <=
                  potential_sharded_shape.dimensions(j) &&
              subsharding_offset_j + potential_base_tile[j] >
                  sharding_offset[j] + base_tile[j]) {
            return Internal("");
          }
        }
        return absl::OkStatus();
      });
  return ok_if_no_violation.ok();
}

static bool IsLeafShardingMoreSpecific(const HloSharding& lhs,
                                       const HloSharding& rhs) {
  DCHECK(!lhs.IsTuple());
  DCHECK(!rhs.IsTuple());
  // Manual sharding is more specific than tile maximal sharding.
  if (lhs.IsManualLeaf() && rhs.IsTileMaximalLeaf()) {
    return true;
  }
  if (lhs.IsManualLeaf() || rhs.IsManualLeaf()) {
    return false;
  }
  if (!rhs.IsTileMaximalLeaf()) {
    return lhs.NumTilesLeaf() > rhs.NumTilesLeaf();
  }
  // If we are not replicated then only tiled (not tile maximal) shardings
  // can improve us.
  // If we are replicated then any non-replicated sharding can improve us.
  return !(rhs.IsReplicatedLeaf() ? lhs.IsReplicatedLeaf()
                                  : lhs.IsTileMaximalLeaf());
}

bool IsShardingMoreSpecific(const HloSharding& lhs, const HloSharding& rhs) {
  CHECK_EQ(lhs.IsTuple(), rhs.IsTuple()) << lhs << " <> " << rhs;
  if (lhs.IsTuple()) {
    // For tuples we consider lhs to have a better sharding if none of the
    // elements are worse and at least one element is better then in rhs
    // sharding.
    const auto& lhs_shardings = lhs.tuple_elements();
    const auto& rhs_shardings = rhs.tuple_elements();
    CHECK_EQ(lhs_shardings.size(), rhs_shardings.size());
    bool is_better = false;
    for (int64_t i = 0; i < lhs_shardings.size(); ++i) {
      if (IsShardingMoreSpecific(rhs_shardings[i], lhs_shardings[i])) {
        return false;
      }
      if (IsShardingMoreSpecific(lhs_shardings[i], rhs_shardings[i])) {
        is_better = true;
      }
    }
    return is_better;
  }
  return IsLeafShardingMoreSpecific(lhs, rhs);
}

bool MergeSharding(const HloSharding& to_merge, HloSharding* dst,
                   bool may_combine_partial_sharding) {
  if (to_merge.IsTuple()) {
    CHECK(dst->IsTuple());
    bool changed = false;
    for (int64_t i = 0; i < to_merge.tuple_elements().size(); ++i) {
      changed |=
          MergeSharding(to_merge.tuple_elements()[i], &dst->tuple_elements()[i],
                        may_combine_partial_sharding);
    }
    return changed;
  }
  if (!may_combine_partial_sharding || !to_merge.HasPartialReplication() ||
      !dst->HasPartialReplication() ||
      to_merge.tile_assignment().num_elements() !=
          dst->tile_assignment().num_elements()) {
    goto check_if_more_specific;
  }

  if (MergeShardingIfCompatible(
          to_merge,
          /*minimum_tiles=*/std::max(to_merge.NumTiles(), dst->NumTiles()) + 1,
          dst)) {
    return true;
  }
check_if_more_specific:
  return IsLeafShardingMoreSpecific(*dst, to_merge);
}

bool MergeShardingIfCompatible(const HloSharding& to_merge, HloSharding* dst) {
  return MergeShardingIfCompatible(to_merge,
                                   /*minimum_tiles=*/dst->NumTiles() + 1, dst);
}

bool MergeShardingIfCompatible(const HloSharding& to_merge,
                               int64_t minimum_tiles, HloSharding* dst) {
  CHECK(!to_merge.IsTuple() && !to_merge.IsManual() && !dst->IsTuple() &&
        !dst->IsManual());
  if (to_merge.IsTileMaximal()) {
    return false;
  }
  if (dst->IsTileMaximal()) {
    *dst = to_merge;
    return true;
  }
  if (!dst->HasPartialReplication()) {
    return false;
  }
  if (dst->TiledDataRank() != to_merge.TiledDataRank()) {
    return false;
  }

  const int64_t to_merge_man_dim = to_merge.SubgroupManualDim();
  const int64_t dst_man_dim = dst->SubgroupManualDim();
  if ((to_merge_man_dim >= 0) != (dst_man_dim >= 0)) {
    return false;
  }

  // Combine the tile dimension sizes from dst and to_merge.
  DimensionVector perm_merge(dst->tile_assignment().num_dimensions(), -1);
  DimensionVector perm_dst(dst->tile_assignment().num_dimensions(), -1);
  int64_t perm_merge_counter = 0;
  int64_t perm_dst_counter = 0;
  DimensionVector merge_old_tile_dim, dst_old_tile_dim;
  DimensionVector merge_new_tile_dim, dst_new_tile_dim;
  DimensionVector merge_new_tile_index, dst_new_tile_index;
  DimensionVector merged_tile_dims;
  merged_tile_dims.reserve(dst->tile_assignment().num_dimensions());
  int64_t num_merge_groups = 1;
  int64_t num_dst_groups = 1;
  for (int64_t i = 0; i < to_merge.TiledDataRank(); ++i) {
    int64_t merge_dim = to_merge.tile_assignment().dim(i);
    int64_t dst_dim = dst->tile_assignment().dim(i);
    num_merge_groups *= merge_dim;
    num_dst_groups *= dst_dim;
    if (dst_dim == merge_dim) {
      merge_old_tile_dim.push_back(merge_dim);
      perm_merge[i] = perm_merge_counter++;
      dst_old_tile_dim.push_back(dst_dim);
      perm_dst[i] = perm_dst_counter++;
      merged_tile_dims.push_back(dst_dim);
    } else if (dst_dim == 1) {
      merge_old_tile_dim.push_back(merge_dim);
      perm_merge[i] = perm_merge_counter++;
      dst_new_tile_dim.push_back(merge_dim);
      dst_new_tile_index.push_back(i);
      merged_tile_dims.push_back(merge_dim);
    } else if (merge_dim == 1) {
      merge_new_tile_dim.push_back(dst_dim);
      merge_new_tile_index.push_back(i);
      dst_old_tile_dim.push_back(dst_dim);
      perm_dst[i] = perm_dst_counter++;
      merged_tile_dims.push_back(dst_dim);
    } else {
      return false;
    }
  }

  const int64_t num_devices = to_merge.tile_assignment().num_elements();
  const int64_t new_num_tiles = Product(merged_tile_dims);
  if (num_devices % new_num_tiles != 0 || new_num_tiles < minimum_tiles) {
    return false;
  }
  int64_t replication;

  if (to_merge_man_dim >= 0) {
    int64_t man_group_size = to_merge.tile_assignment().dim(to_merge_man_dim);
    if (man_group_size != dst->tile_assignment().dim(dst_man_dim)) {
      return false;
    }
    merge_old_tile_dim.push_back(man_group_size);
    dst_old_tile_dim.push_back(man_group_size);
    perm_merge[to_merge.TiledDataRank()] = perm_merge_counter++;
    perm_dst[to_merge.TiledDataRank()] = perm_dst_counter++;

    merged_tile_dims.push_back(man_group_size);
    num_merge_groups *= man_group_size;
    num_dst_groups *= man_group_size;
    if (num_devices % (new_num_tiles * man_group_size) != 0) {
      return false;
    }
    replication = num_devices / (new_num_tiles * man_group_size);
  } else {
    replication = num_devices / new_num_tiles;
  }

  if (replication > 1) {
    merged_tile_dims.push_back(replication);
  }

  std::optional<TileAssignment> compatible_tile_assignment;
  // We use two methods to find compatible_tile_assignment. The comparisons are
  // liste below.
  // 1. In terms of compilation speed, the first method is usually faster than
  // the second one, especially when the number of devices is large.
  // 2. The first method is friendly to the iota tile assignment. If to_merge or
  // dst has iota tile assignment, the resultant sharding also has iota tile
  // assignment. The second method always generates v1 sharding.
  // 3. The first method can handle the common cases. However, it fails on
  // corner cases, such as the arbitrary device order. Conversely, the second
  // method can handle all cases. Above all, we initially try the first method,
  // and proceed with the second one if the first one fails.

  {
    // In the first method, we use reshape and transpose to generate the
    // compatible tile assignments for the input sharding. Reshape: decompose
    // the input sharding along the replicated dimension. Transpose: assign the
    // decomposed dimensions to the new tiled dimensions.
    auto get_compatible_tile_assignment =
        [&](const HloSharding& sharding, const DimensionVector& old_tile_dims,
            DimensionVector& new_tile_dims, DimensionVector& new_tile_indices,
            DimensionVector& perm,
            const int64_t perm_counter) -> std::vector<TileAssignment> {
      if (!sharding.HasPartialReplication() ||
          sharding.tile_assignment().dim(sharding.SubgroupReplicationDim()) ==
              replication) {
        return {sharding.tile_assignment()};
      }
      if (replication == 1) {
        perm.pop_back();
      } else {
        new_tile_dims.push_back(replication);
        new_tile_indices.push_back(dst->tile_assignment().num_dimensions() - 1);
      }

      std::vector<TileAssignment> result;
      DimensionVector iota(new_tile_dims.size());
      absl::c_iota(iota, 0);
      do {
        std::vector<int> local_perm(perm.begin(), perm.end());
        int64_t local_perm_counter = perm_counter;
        DimensionVector reshape_dims(old_tile_dims.begin(),
                                     old_tile_dims.end());
        reshape_dims.reserve(old_tile_dims.size() + new_tile_dims.size());
        for (auto i : iota) {
          reshape_dims.push_back(new_tile_dims[i]);
          local_perm[new_tile_indices[i]] = local_perm_counter++;
        }
        result.push_back(sharding.tile_assignment()
                             .Reshape(reshape_dims)
                             .Transpose(local_perm));
      } while (std::next_permutation(iota.begin(), iota.end()));
      return result;
    };

    auto merge_compatible_tile_assignment = get_compatible_tile_assignment(
        to_merge, merge_old_tile_dim, merge_new_tile_dim, merge_new_tile_index,
        perm_merge, perm_merge_counter);
    auto dst_compatible_tile_assignment = get_compatible_tile_assignment(
        *dst, dst_old_tile_dim, dst_new_tile_dim, dst_new_tile_index, perm_dst,
        perm_dst_counter);

    // Find the intersection of merge_compatible_tile_assignment and
    // dst_compatible_tile_assignment, such that the resultant tile assignment
    // is compatible to both to_merge and dst.
    for (const auto& ta1 : dst_compatible_tile_assignment) {
      for (const auto& ta2 : merge_compatible_tile_assignment) {
        if (ta1 == ta2) {
          // Try to get the tile assignment in the iota format
          compatible_tile_assignment = ta1.iota() ? ta1 : ta2;
        }
      }
    }
  }

  // If the first method fails, try the second method, which handles the element
  // in the new tile assignment one by one.
  if (!compatible_tile_assignment.has_value()) {
    Array<int64_t> new_tile_array(merged_tile_dims);
    // Maps from replication group ID to sorted members.
    std::vector<absl::btree_set<int64_t>> merge_group_members(num_merge_groups);
    std::vector<absl::btree_set<int64_t>> dst_group_members(num_dst_groups);
    const int64_t merge_group_size = num_devices / num_merge_groups;
    const int64_t dst_group_size = num_devices / num_dst_groups;
    const auto* merge_begin = to_merge.tile_assignment().array().begin();
    const auto* dst_begin = dst->tile_assignment().array().begin();
    for (int64_t i = 0; i < num_merge_groups; ++i) {
      merge_group_members[i] =
          absl::btree_set<int64_t>{merge_begin + i * merge_group_size,
                                   merge_begin + (i + 1) * merge_group_size};
    }
    for (int64_t i = 0; i < num_dst_groups; ++i) {
      dst_group_members[i] = absl::btree_set<int64_t>{
          dst_begin + i * dst_group_size, dst_begin + (i + 1) * dst_group_size};
    }

    auto get_group_index = [&](absl::Span<const int64_t> tile_indices,
                               const HloSharding& sharding,
                               int64_t manual_dim) {
      int64_t group_id = 0;
      for (int64_t i = 0; i < to_merge.TiledDataRank(); ++i) {
        group_id *= sharding.tile_assignment().dim(i);
        group_id += tile_indices[i];
      }
      if (manual_dim >= 0) {
        group_id *= sharding.tile_assignment().dim(manual_dim);
        group_id += tile_indices[manual_dim];
      }
      return group_id;
    };
    // Try to find the intersection of to_merge and dst replication groups, in
    // order to determine the merged tile assignment.
    absl::Status compatible =
        new_tile_array.EachStatus([&](absl::Span<const int64_t> indices,
                                      int64_t* device) -> absl::Status {
          DimensionVector to_merge_index(
              to_merge.tile_assignment().num_dimensions());
          DimensionVector dst_index(dst->tile_assignment().num_dimensions());
          for (int64_t i = 0; i < to_merge.TiledDataRank(); ++i) {
            if (to_merge.tile_assignment().dim(i) == 1) {
              to_merge_index[i] = 0;
            } else {
              to_merge_index[i] = indices[i];
            }
            if (dst->tile_assignment().dim(i) == 1) {
              dst_index[i] = 0;
            } else {
              dst_index[i] = indices[i];
            }
          }
          if (to_merge_man_dim >= 0) {
            to_merge_index[to_merge_man_dim] =
                indices[to_merge.TiledDataRank()];
            dst_index[dst_man_dim] = indices[to_merge.TiledDataRank()];
          }
          if (to_merge.HasPartialReplication()) {
            to_merge_index[to_merge.SubgroupReplicationDim()] = indices.back();
          }
          dst_index[dst->SubgroupReplicationDim()] = indices.back();

          int64_t to_merge_group_id =
              get_group_index(to_merge_index, to_merge, to_merge_man_dim);
          int64_t dst_group_id = get_group_index(dst_index, *dst, dst_man_dim);
          auto& gm1 = merge_group_members[to_merge_group_id];
          auto& gm2 = dst_group_members[dst_group_id];

          // Find the smallest element in the intersection of gm1 and gm2.
          auto it1 = gm1.begin();
          auto it2 = gm2.begin();
          while (it1 != gm1.end() && it2 != gm2.end()) {
            if (*it1 == *it2) {
              *device = *it1;
              gm1.erase(it1);
              gm2.erase(it2);
              return absl::OkStatus();
            } else if (*it1 < *it2) {
              it1++;
            } else {
              it2++;
            }
          }
          return InvalidArgument("Not compatible");
        });
    if (!compatible.ok()) {
      return false;
    }
    compatible_tile_assignment =
        TileAssignment(std::make_shared<const Array<int64_t>>(new_tile_array));
  }

  std::vector<OpMetadata> merged_metadata(std::move(dst->metadata()));
  merged_metadata.reserve(merged_metadata.size() + to_merge.metadata().size());
  const absl::flat_hash_set<OpMetadata,
                            protobuf_util::ProtobufHashBySerializationFunctor,
                            protobuf_util::HaveSameSerializationFunctor>
      metadata_set(merged_metadata.begin(), merged_metadata.end());
  absl::c_copy_if(to_merge.metadata(), std::back_inserter(merged_metadata),
                  [&metadata_set](const OpMetadata& data) {
                    return !ContainsKey(metadata_set, data);
                  });
  std::vector<OpSharding::Type> subgroup_types;
  if (to_merge_man_dim >= 0) {
    subgroup_types.push_back(OpSharding::MANUAL);
  }
  if (replication > 1) {
    subgroup_types.push_back(OpSharding::REPLICATED);
  }
  *dst = HloSharding::Subgroup(compatible_tile_assignment.value(),
                               subgroup_types, merged_metadata);
  return true;
}

std::optional<int64_t> SelectDominantDevice(
    const std::map<int64_t, int64_t>& device_map, int64_t* top_count) {
  int64_t device = 0;
  int64_t count = 0;
  for (auto& it : device_map) {
    if (it.second > count) {
      count = it.second;
      device = it.first;
    }
  }
  if (top_count != nullptr) {
    *top_count = count;
  }
  return count > 0 ? std::optional<int64_t>(device) : std::optional<int64_t>();
}

HloSharding FindCommonSharding(absl::Span<const HloSharding> shardings,
                               std::optional<HloSharding> default_sharding) {
  CHECK(!shardings.empty());
  bool all_compatible = true;
  HloSharding common_sharding = shardings[0];
  for (int i = 1; i != shardings.size(); ++i) {
    if (common_sharding != shardings[i] &&
        !MergeShardingIfCompatible(shardings[i], common_sharding.NumTiles(),
                                   &common_sharding)) {
      all_compatible = false;
      break;
    }
  }
  if (all_compatible) {
    return common_sharding;
  }
  // TODO(tongfei): instead of return the first sharding in case not all
  // shardings are compatible, we should find a sharding that's compatible with
  // the most number of shardings instead.
  return default_sharding.has_value() ? default_sharding.value() : shardings[0];
}

void AssignComputationDevice(HloComputation* computation, int64_t device) {
  VLOG(4) << "Assigning device " << device << " to " << computation->name()
          << " computation";
  for (HloInstruction* instruction : computation->instructions()) {
    if (!instruction->has_sharding()) {
      VLOG(4) << "Assigning device " << device << " to " << instruction->name();
      instruction->set_device_sharding(device);
    }
  }
}

std::optional<int64_t> GetMostOccurringDevice(
    absl::Span<HloInstruction* const> instructions) {
  std::map<int64_t, int64_t> device_map;
  for (HloInstruction* instruction : instructions) {
    if (instruction->has_sharding()) {
      for (auto& it : instruction->sharding().UsedDevices(nullptr)) {
        // The UsedDevices() API returns a map<device, occurrence_count>.
        device_map[it.first] += it.second;
      }
    }
  }
  return SelectDominantDevice(device_map, nullptr);
}

std::optional<int64_t> GetDominantDevice(
    absl::Span<HloComputation* const> computations, double dominant_factor) {
  int64_t instruction_count = 0;
  std::map<int64_t, int64_t> device_map;
  for (HloComputation* computation : computations) {
    for (HloInstruction* instruction : computation->instructions()) {
      int64_t count = 1;
      if (instruction->has_sharding()) {
        for (auto& it : instruction->sharding().UsedDevices(&count)) {
          // The UsedDevices() API returns a map<device, occurrence_count>.
          device_map[it.first] += it.second;
        }
      }
      instruction_count += count;
    }
  }
  int64_t count;
  std::optional<int64_t> device = SelectDominantDevice(device_map, &count);
  std::optional<int64_t> dominant_device;
  if (device) {
    double factor =
        static_cast<double>(count) / static_cast<double>(instruction_count);
    if (factor >= dominant_factor) {
      dominant_device = device;
    }
  }
  return dominant_device;
}

HloSharding MoveAndMergeShardingTiles(const HloSharding& sharding,
                                      int64_t source_dim, int64_t target_dim) {
  CHECK(sharding.IsTiled());

  CHECK_NE(source_dim, target_dim);
  CHECK_GE(source_dim, 0);
  CHECK_GE(target_dim, 0);
  CHECK_LT(source_dim, sharding.TiledDataRank());
  CHECK_LT(target_dim, sharding.TiledDataRank());

  // There are 3 steps to move and merge the sharding tiles. Given the sharding
  // with tile assignment [a, b, c, d, e], source_dim = 1, target_dim = 3, the
  // steps are:
  // 1. Reshape the tile assignment to [a, b, c, d, 1, e] by inserting a 1 after
  // the target_dim.
  // 2. Transpose the tile assignment to [a, 1, c, d, b, e] by swapping the
  // source_dim and inserted dim of size 1.
  // 3. Reshape the tile assignment to [a, 1, c, db, e] by merging the
  // target_dim and the swapped source_dim.

  // Step 1. Adding a dummy dim of size 1 after the target_dim.
  std::vector<int64_t> ta_dims_1(
      sharding.tile_assignment().dimensions().begin(),
      sharding.tile_assignment().dimensions().end());
  ta_dims_1.insert(ta_dims_1.begin() + target_dim + 1, 1);
  TileAssignment new_tile_assignment =
      sharding.tile_assignment().Reshape(ta_dims_1);

  // Step 2. Transpose the tile assignment to swap the source_dim and the
  // inserted dim of size 1.
  std::vector<int> permutation(new_tile_assignment.num_dimensions());
  absl::c_iota(permutation, 0);
  std::swap(permutation[target_dim + 1],
            permutation[source_dim + (source_dim < target_dim ? 0 : 1)]);
  new_tile_assignment = new_tile_assignment.Transpose(permutation);

  // Step 3. Reshape the tile assignment to merge the target_dim and the swapped
  // source_dim.
  std::vector<int64_t> ta_dims_2(new_tile_assignment.dimensions().begin(),
                                 new_tile_assignment.dimensions().end());
  ta_dims_2[target_dim] *= ta_dims_2[target_dim + 1];
  ta_dims_2.erase(ta_dims_2.begin() + target_dim + 1);
  new_tile_assignment = new_tile_assignment.Reshape(ta_dims_2);

  if (sharding.ReplicateOnLastTileDim()) {
    return HloSharding::PartialTile(new_tile_assignment, sharding.metadata());
  }
  return HloSharding::Subgroup(new_tile_assignment, sharding.subgroup_types(),
                               sharding.metadata());
}

HloSharding TransposeSharding(const HloSharding& sharding,
                              absl::Span<const int64_t> dimensions) {
  if (sharding.IsTileMaximal() || sharding.IsManual()) {
    return sharding;
  }
  std::vector<int> perm_dimensions(dimensions.begin(), dimensions.end());
  // Add subgroup dims if missing.
  if (sharding.TiledDataRank() == dimensions.size()) {
    for (int64_t i = sharding.TiledDataRank();
         i < sharding.tile_assignment().num_dimensions(); ++i) {
      perm_dimensions.push_back(i);
    }
  } else {
    CHECK_EQ(sharding.tile_assignment().num_dimensions(), dimensions.size());
  }
  auto tile_assignment = sharding.tile_assignment().Transpose(perm_dimensions);
  if (!sharding.ReplicateOnLastTileDim()) {
    std::vector<OpSharding::Type> subgroup_types;
    for (int64_t i = sharding.TiledDataRank(); i < perm_dimensions.size();
         ++i) {
      int64_t src_i = perm_dimensions[i] - sharding.TiledDataRank();
      subgroup_types.push_back(sharding.subgroup_types()[src_i]);
    }
    return HloSharding::Subgroup(tile_assignment, subgroup_types,
                                 sharding.metadata());
  } else {
    return HloSharding::PartialTile(tile_assignment, sharding.metadata());
  }
}

std::optional<HloSharding> ReshapeSharding(const Shape& source_shape,
                                           const Shape& target_shape,
                                           const HloSharding& source_sharding) {
  if (source_sharding.IsTileMaximal() || source_sharding.IsManual()) {
    return source_sharding;
  }

  // In case of a tiled sharding, the reshaped sharding will be valid if the
  // reshape is composed from the following operations:
  // * Adding or removing dimensions with size 1.
  // * Merging consecutive dimensions where only the most major is sharded.
  // * Splitting a dimension to consecutive dimensions.
  // * Any reshaping of unsharded dimensions.
  //
  // Merge and split can happen consecutively on the same dimension, e.g.,
  // f32[1024,256] to f32[128,2048] can be considered that 1024 gets split into
  // 128 and 8, but 8 then gets merged with 256. We use stacks to make
  // supporting such cases easy.
  //
  // If transpose is needed between source and target shapes, we use the GCD of
  // (target_shape_dim, sharding_dim) if source_shape_dim % sharding_dim == 0.
  // For example, given the source_shape f32[6,4], target_shape f32[4,6] and
  // sharding {devices=[6,1]<=[6]}, the output sharding is {devices=[2,1,3]<=[6]
  // last_tile_dim_replicate}.
  DimensionVector target_tile_assignment_dimensions;
  DimensionVector source_dims_stack(source_shape.dimensions().rbegin(),
                                    source_shape.dimensions().rend());
  DimensionVector target_dims_stack(target_shape.dimensions().rbegin(),
                                    target_shape.dimensions().rend());
  DimensionVector sharding_tile_dims_stack(
      source_sharding.tile_assignment().dimensions().begin(),
      source_sharding.tile_assignment().dimensions().begin() +
          source_shape.dimensions().size());
  std::reverse(sharding_tile_dims_stack.begin(),
               sharding_tile_dims_stack.end());
  int64_t source_dims_index = -1;
  std::vector<int64_t> dims_to_replicate;

  auto source_dims_push = [&](int64_t shape_size, int64_t partitions) {
    source_dims_stack.push_back(shape_size);
    sharding_tile_dims_stack.push_back(partitions);
    source_dims_index--;
  };
  auto source_dims_pop = [&]() {
    source_dims_stack.pop_back();
    sharding_tile_dims_stack.pop_back();
    source_dims_index++;
  };

  bool inplace_add_sharding_dim = false;
  auto append_target_sharding_dim = [&](int64_t size) {
    if (inplace_add_sharding_dim) {
      target_tile_assignment_dimensions.back() *= size;
    } else {
      target_tile_assignment_dimensions.push_back(size);
    }
    inplace_add_sharding_dim = false;
  };

  while (!source_dims_stack.empty() && !target_dims_stack.empty() &&
         Product(sharding_tile_dims_stack) != 1) {
    int64_t source_dims_product = 1;
    while (!sharding_tile_dims_stack.empty() &&
           sharding_tile_dims_stack.back() == 1) {
      source_dims_product *= source_dims_stack.back();
      source_dims_pop();
    }
    while (!target_dims_stack.empty() && target_dims_stack.back() > 1 &&
           source_dims_product % target_dims_stack.back() == 0) {
      source_dims_product /= target_dims_stack.back();
      target_dims_stack.pop_back();
      append_target_sharding_dim(1);
    }
    if (source_dims_product != 1) {
      source_dims_push(source_dims_product, 1);
    }

    if (source_dims_stack.empty() || target_dims_stack.empty()) {
      break;
    }
    int64_t s_size = source_dims_stack.back();
    int64_t s_partitions = sharding_tile_dims_stack.back();
    source_dims_pop();

    int64_t t_size = target_dims_stack.back();
    target_dims_stack.pop_back();

    if (s_size == t_size) {
      // Same dimension size.
      append_target_sharding_dim(s_partitions);
    } else if (t_size == 1) {
      // Trivial dimension added.
      append_target_sharding_dim(1);
      source_dims_push(s_size, s_partitions);
    } else if (s_size == 1) {
      // Trivial dimension removed.
      target_dims_stack.push_back(t_size);
      if (s_partitions > 1) {
        dims_to_replicate.push_back(source_dims_index);
      }
    } else if (s_partitions == 1) {
      if (!source_dims_stack.empty() && sharding_tile_dims_stack.back() == 1) {
        source_dims_stack.back() *= s_size;
      } else {
        break;
      }
    } else if (s_size % s_partitions != 0) {
      // TODO(zixuanjiang): Although we can propagate thd gcd(s_size,
      // s_partitions), we return std::nullopt since the current partitioner
      // reply on that to create halo exchange. Revisit it later.
      return std::nullopt;
    } else {
      int64_t gcd = std::gcd(s_partitions, t_size);
      if (gcd == 1) {
        break;
      }

      source_dims_push(s_size / gcd, s_partitions / gcd);
      target_dims_stack.push_back(t_size / gcd);
      append_target_sharding_dim(gcd);
      inplace_add_sharding_dim = true;
    }
  }

  if (Product(target_tile_assignment_dimensions) == 1) {
    return std::nullopt;
  }
  while (target_tile_assignment_dimensions.size() <
         target_shape.dimensions().size()) {
    target_tile_assignment_dimensions.push_back(1);
  }

  // If there is a source dimension satisfying (1) size is 1, (2) partition > 1,
  // and (3) there is no corresponding target dimension, we replicate the source
  // sharding along this dimension since the source sharding cannot be
  // propagated along this dimension.
  const HloSharding sharding = !dims_to_replicate.empty()
                                   ? PartiallyReplicateTiledShardingOnDims(
                                         source_sharding, dims_to_replicate)
                                   : source_sharding;

  for (int64_t i = sharding.TiledDataRank();
       i < sharding.tile_assignment().num_dimensions(); ++i) {
    target_tile_assignment_dimensions.push_back(
        i == sharding.SubgroupReplicationDim()
            ? 1
            : sharding.tile_assignment().dim(i));
  }

  auto subgroup_types = sharding.subgroup_types();
  auto partially_replicated = std::div(
      sharding.TotalNumTiles(), Product(target_tile_assignment_dimensions));
  CHECK_EQ(partially_replicated.rem, 0);
  if (partially_replicated.quot > 1) {
    if (sharding.ReplicateOnLastTileDim()) {
      target_tile_assignment_dimensions.back() = partially_replicated.quot;
      subgroup_types.push_back(OpSharding::REPLICATED);
    } else if (absl::c_linear_search(subgroup_types, OpSharding::REPLICATED)) {
      target_tile_assignment_dimensions[sharding.SubgroupReplicationDim() -
                                        sharding.TiledDataRank() +
                                        target_shape.dimensions().size()] =
          partially_replicated.quot;
    } else {
      target_tile_assignment_dimensions.push_back(partially_replicated.quot);
      subgroup_types.push_back(OpSharding::REPLICATED);
    }
  }

  auto new_tile_assignment =
      sharding.tile_assignment().Reshape(target_tile_assignment_dimensions);
  return HloSharding::Subgroup(new_tile_assignment, subgroup_types,
                               sharding.metadata());
}

HloSharding PropagateShardingThroughReshape(const Shape& source_shape,
                                            const Shape& target_shape,
                                            const HloSharding& sharding) {
  if (sharding.IsTileMaximal() || sharding.IsManual()) {
    return sharding;
  }
  if (sharding.IsManualSubgroup()) {
    auto group =
        GroupShardingOnDims(sharding, {sharding.SubgroupManualDim()}, true);
    HloSharding inner_reshaped = PropagateShardingThroughReshape(
        source_shape, target_shape, group.sharding);
    group.sharding = std::move(inner_reshaped);
    group.data_rank = target_shape.dimensions().size();
    group.group_dims[0] +=
        target_shape.dimensions().size() - source_shape.dimensions().size();
    return UngroupSharding(group);
  }
  // Find intervals of consecutive dimensions that could use ReshapeSharding().
  // then merge the results. We start with the longest interval (whole shape),
  // and if it fails, we find a sub-interval of it or a disjoint interval.
  HloSharding result = HloSharding::Replicate();
  int64_t start_dim = 0;
  while (start_dim < source_shape.dimensions().size()) {
    bool found_compatible = false;
    // For each start_dim, try to use all dims after it. If that fails, reduce
    // the range.
    for (int64_t end_dim = source_shape.dimensions().size();
         end_dim > start_dim; --end_dim) {
      DimensionVector grouped_tiling_dims(source_shape.dimensions().size(), 1);
      for (int64_t i = start_dim; i < end_dim; ++i) {
        grouped_tiling_dims[i] = sharding.tile_assignment().dim(i);
      }
      HloSharding grouped_sharding =
          HloSharding::Tile(TileAssignment(grouped_tiling_dims));
      if (auto reshaped =
              ReshapeSharding(source_shape, target_shape, grouped_sharding)) {
        std::vector<int> perm;
        perm.reserve(sharding.tile_assignment().num_dimensions());
        for (int64_t i = start_dim; i < end_dim; i++) {
          perm.push_back(i);
        }
        for (int64_t i = 0; i < start_dim; i++) {
          perm.push_back(i);
        }
        for (int64_t i = end_dim;
             i < sharding.tile_assignment().num_dimensions(); i++) {
          perm.push_back(i);
        }

        DimensionVector reshape_dims(
            reshaped->tile_assignment().dimensions().begin(),
            reshaped->tile_assignment().dimensions().end());
        CHECK_EQ(
            sharding.tile_assignment().num_elements() % Product(reshape_dims),
            0);
        int64_t num_replicated_dims =
            sharding.tile_assignment().num_elements() / Product(reshape_dims);
        const int64_t diff =
            reshape_dims.size() - target_shape.dimensions().size();
        CHECK(diff == 0 || diff == 1);
        if (diff == 0) {
          reshape_dims.push_back(num_replicated_dims);
        } else {
          reshape_dims.back() *= num_replicated_dims;
        }
        HloSharding ungrouped_sharding = HloSharding::PartialTile(
            sharding.tile_assignment().Transpose(perm).Reshape(reshape_dims));
        if (MergeShardingIfCompatible(ungrouped_sharding, &result)) {
          // If the current interval works, we can skip all dimensions within
          // or before it in future intervals, since they have been considered
          // already. Set start_dim to end_dim to start with the next disjoint
          // interval.
          start_dim = end_dim;
          found_compatible = true;
          break;
        }
      }
    }
    if (!found_compatible) {
      // All sub-intervals with the current start_dim failed. Try the next
      // start_dim.
      start_dim += 1;
    }
  }
  result.metadata() = sharding.metadata();
  return result;
}

HloSharding ReverseSharding(const HloSharding& sharding,
                            absl::Span<const int64_t> dimensions) {
  if (sharding.IsTileMaximal() || dimensions.empty()) {
    return sharding;
  }

  Array<int64_t> new_tile_assignment(sharding.tile_assignment().dimensions());
  new_tile_assignment.Each(
      [&](absl::Span<const int64_t> indices, int64_t* device) {
        std::vector<int64_t> original_indices(indices.begin(), indices.end());
        for (int64_t d : dimensions) {
          original_indices[d] =
              new_tile_assignment.dim(d) - 1 - original_indices[d];
        }
        *device = sharding.tile_assignment()(original_indices);
      });
  return sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment,
                                        sharding.metadata())
             : HloSharding::Subgroup(new_tile_assignment,
                                     sharding.subgroup_types(),
                                     sharding.metadata());
}

HloSharding ReshapeToTileDimension(const HloSharding& sharding, int64_t dim,
                                   absl::Span<const int64_t> dims) {
  CHECK(!sharding.IsTuple() && !sharding.IsTileMaximal());
  CHECK_NE(absl::c_find(dims, dim), dims.end()) << "dim is not in dims";
  // We optimize the tile assignment on the single dimension dim in a way to
  // minimize communication among devices caused by the reshard:
  // +---+---+               +---+---+              +-+-+-+-+
  // |   |   |               |   0   |              | | | | |
  // | 0 | 1 |               +-------+              | | | | |
  // |   |   |  reshape on   |   1   |  reshape on  | | | | |
  // +---+---+   dim 0  =>   +-------+   dim 1  =>  |0|2|1|3|
  // |   |   |               |   2   |              | | | | |
  // | 2 | 3 |               +-------+              | | | | |
  // |   |   |               |   3   |              | | | | |
  // +---+---+               +---+---+              +-+-+-+-+

  auto old_dims = sharding.tile_assignment().dimensions();
  DimensionVector new_dims(old_dims.begin(), old_dims.end());
  std::vector<int> not_in_dims, dims_except_the_dim;
  for (int64_t i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (i == dim) {
      continue;
    } else if (absl::c_find(dims, i) != dims.end()) {
      dims_except_the_dim.push_back(i);
      new_dims[dim] *= old_dims[i];
      new_dims[i] = 1;
    } else {
      not_in_dims.push_back(i);
    }
  }
  // perm = not_in_dims + {dim} + dims_except_the_dim
  std::vector<int> perm;
  perm.reserve(sharding.tile_assignment().num_dimensions());
  perm.insert(perm.end(), not_in_dims.begin(), not_in_dims.end());
  perm.push_back(dim);
  perm.insert(perm.end(), dims_except_the_dim.begin(),
              dims_except_the_dim.end());

  auto new_tile_assignment =
      sharding.tile_assignment().Transpose(perm).Reshape(new_dims);
  return HloSharding::Tile(new_tile_assignment, sharding.metadata());
}

bool ContainsTileSharding(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->has_sharding() &&
          !instruction->sharding().IsTileMaximal()) {
        return true;
      }
    }
  }
  return false;
}

HloSharding PropagateShardingAlongDimsAndReplicateOthers(
    const HloSharding& source_sharding, absl::Span<const int64_t> source_dims,
    absl::Span<const int64_t> target_dims, int64_t target_shape_rank) {
  CHECK_EQ(source_dims.size(), target_dims.size());
  if (source_sharding.IsTileMaximal() || source_sharding.IsManual()) {
    return source_sharding;
  }

  HloSharding replicate_other_dims =
      PartiallyReplicateTiledShardingOnAllDimsExcept(source_sharding,
                                                     source_dims);
  if (replicate_other_dims.IsTileMaximal()) {
    return replicate_other_dims;
  }

  std::vector<int64_t> argsort_source_dims = ArgSort(source_dims);
  std::vector<int64_t> argsort_target_dims = ArgSort(target_dims);
  if (argsort_source_dims != argsort_target_dims) {
    std::vector<int64_t> perm;
    perm.reserve(replicate_other_dims.tile_assignment().num_dimensions());
    for (int64_t i = 0; i < source_dims.size(); ++i) {
      perm.push_back(source_dims[argsort_target_dims[i]]);
    }
    for (int64_t non_source_dim = 0, i = source_dims.size();
         i < replicate_other_dims.tile_assignment().num_dimensions(); ++i) {
      while (absl::c_linear_search(source_dims, non_source_dim)) {
        non_source_dim++;
      }
      perm.push_back(non_source_dim++);
    }
    replicate_other_dims = TransposeSharding(replicate_other_dims, perm);
  }

  std::vector<int64_t> target_tile_dims(target_shape_rank, 1);
  for (int i = 0; i < source_dims.size(); ++i) {
    target_tile_dims[target_dims[i]] =
        source_sharding.tile_assignment().dim(source_dims[i]);
  }
  for (int64_t i = replicate_other_dims.TiledDataRank();
       i < replicate_other_dims.tile_assignment().num_dimensions(); ++i) {
    target_tile_dims.push_back(replicate_other_dims.tile_assignment().dim(i));
  }

  auto target_tile_assignment =
      replicate_other_dims.tile_assignment().Reshape(target_tile_dims);
  return replicate_other_dims.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(target_tile_assignment,
                                        replicate_other_dims.metadata())
             : HloSharding::Subgroup(target_tile_assignment,
                                     replicate_other_dims.subgroup_types(),
                                     replicate_other_dims.metadata());
}

HloSharding GatherOutputShardingFromIndex(const HloSharding& index_sharding,
                                          const HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kGather);
  if (index_sharding.IsTileMaximal() || index_sharding.IsManual()) {
    return index_sharding;
  }

  const GatherDimensionNumbers& dnums = hlo->gather_dimension_numbers();
  const GatherScatterDims indices_output_dims =
      GetGatherConnectedDimsAcrossIndicesAndOutput(
          hlo->operand(1)->shape().dimensions().size(),
          dnums.index_vector_dim(), hlo->shape().dimensions().size(),
          dnums.offset_dims());
  return PropagateShardingAlongDimsAndReplicateOthers(
      index_sharding, indices_output_dims.indices_dims,
      indices_output_dims.output_dims, hlo->shape().dimensions().size());
}

HloSharding GatherIndexShardingFromOutput(const HloSharding& output_sharding,
                                          const HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kGather);
  if (output_sharding.IsTileMaximal() || output_sharding.IsManual()) {
    return output_sharding;
  }

  const GatherDimensionNumbers& dnums = hlo->gather_dimension_numbers();
  const GatherScatterDims indices_output_dims =
      GetGatherConnectedDimsAcrossIndicesAndOutput(
          hlo->operand(1)->shape().dimensions().size(),
          dnums.index_vector_dim(), hlo->shape().dimensions().size(),
          dnums.offset_dims());
  return PropagateShardingAlongDimsAndReplicateOthers(
      output_sharding, indices_output_dims.output_dims,
      indices_output_dims.indices_dims,
      hlo->operand(1)->shape().dimensions().size());
}

HloSharding GatherEffectiveOutputSharding(const HloInstruction& hlo) {
  if (hlo.sharding().IsTileMaximal() || hlo.sharding().IsManual()) {
    return hlo.sharding();
  }

  const GatherDimensionNumbers& dnums = hlo.gather_dimension_numbers();
  DimensionVector tile_assignment_dims(hlo.shape().dimensions().size());
  int64_t num_elements = 1;
  for (int64_t i = 0; i < hlo.shape().dimensions().size(); ++i) {
    if (!absl::c_binary_search(dnums.offset_dims(), i)) {
      tile_assignment_dims[i] = hlo.sharding().tile_assignment().dim(i);
      num_elements *= hlo.sharding().tile_assignment().dim(i);
    } else {
      tile_assignment_dims[i] = 1;
    }
  }
  if (num_elements == hlo.sharding().tile_assignment().num_elements()) {
    // Output sharding is only on non offset dimensions. We use output sharding
    // to shard this gather op directly.
    return hlo.sharding();
  }

  if (num_elements == 1) {
    // Output sharding is only on offset dimensions. We do not shard this gather
    // op. Return a tile maximal sharding with the first device in output
    // sharding tile assignment.
    return HloSharding::AssignDevice(hlo.sharding().tile_assignment().first(),
                                     hlo.sharding().metadata());
  }

  // Output sharding is on both offset and non offset dimensions. We shard the
  // gather op only on non offset dimensions.
  // For example:
  // - the gather op has sharding [2,2]{0,1,2,3},
  // - first dimension is non offset dimension,
  // - second dimension is offset dimension,
  // Then the result sharding will be [2,1]{0,2}.
  DimensionVector slice_starts(hlo.shape().dimensions().size(), 0LL),
      slice_limits(hlo.shape().dimensions().size());
  for (int64_t i = 0; i < hlo.shape().dimensions().size(); ++i) {
    if (!absl::c_binary_search(dnums.offset_dims(), i)) {
      slice_limits[i] = hlo.sharding().tile_assignment().dim(i);
    } else {
      slice_limits[i] = 1;
    }
  }
  Array<int64_t> tile_assignment =
      hlo.sharding().tile_assignment().array().Slice(slice_starts,
                                                     slice_limits);
  return HloSharding::Tile(tile_assignment, hlo.sharding().metadata());
}

HloSharding ScatterIndexShardingFromUpdate(
    const HloSharding& update_sharding, const HloScatterInstruction* scatter) {
  if (update_sharding.IsTileMaximal() || update_sharding.IsManual()) {
    return update_sharding;
  }

  const ScatterDimensionNumbers& dnums = scatter->scatter_dimension_numbers();
  const GatherScatterDims indices_update_dims =
      GetGatherConnectedDimsAcrossIndicesAndOutput(
          scatter->scatter_indices()->shape().dimensions().size(),
          dnums.index_vector_dim(),
          scatter->scatter_updates()[0]->shape().dimensions().size(),
          dnums.update_window_dims());
  return PropagateShardingAlongDimsAndReplicateOthers(
      update_sharding, indices_update_dims.output_dims,
      indices_update_dims.indices_dims,
      scatter->scatter_indices()->shape().dimensions().size());
}

HloSharding ScatterUpdateShardingFromIndex(
    const HloSharding& index_sharding, const HloScatterInstruction* scatter) {
  if (index_sharding.IsTileMaximal() || index_sharding.IsManual()) {
    return index_sharding;
  }

  const ScatterDimensionNumbers& dnums = scatter->scatter_dimension_numbers();
  const GatherScatterDims indices_update_dims =
      GetGatherConnectedDimsAcrossIndicesAndOutput(
          scatter->scatter_indices()->shape().dimensions().size(),
          dnums.index_vector_dim(),
          scatter->scatter_updates()[0]->shape().dimensions().size(),
          dnums.update_window_dims());
  return PropagateShardingAlongDimsAndReplicateOthers(
      index_sharding, indices_update_dims.indices_dims,
      indices_update_dims.output_dims,
      scatter->scatter_updates()[0]->shape().dimensions().size());
}

HloSharding ScatterEffectiveIndexSharding(
    const HloSharding& index_sharding, const HloScatterInstruction& scatter) {
  if (index_sharding.IsTileMaximal() || index_sharding.IsManual()) {
    return index_sharding;
  }

  // Only shard on first "number of scatter_window_dims" dimensions.
  const ScatterDimensionNumbers& dnums = scatter.scatter_dimension_numbers();
  int64_t num_elements = 1;
  int64_t index_dim = 0;
  for (int64_t i = 0; i < scatter.shape().dimensions().size(); ++i) {
    if (absl::c_binary_search(dnums.inserted_window_dims(), i)) {
      num_elements *= index_sharding.tile_assignment().dim(index_dim);
      index_dim++;
    }
  }
  if (num_elements == index_sharding.tile_assignment().num_elements()) {
    // Index sharding is only on scatter_window_dims. We use this index sharding
    // directly.
    return index_sharding;
  }

  // Index sharding is only on update_window_dims. We do not shard this scatter
  // op. Return a tile maximal sharding with the first device in index sharding
  // tile assignment.
  if (num_elements == 1) {
    return HloSharding::AssignDevice(index_sharding.tile_assignment().first(),
                                     index_sharding.metadata());
  }

  const int64_t index_rank =
      scatter.scatter_indices()->shape().dimensions().size();
  DimensionVector slice_starts(index_rank, 0LL), slice_limits(index_rank);
  for (int64_t i = 0; i < index_rank; ++i) {
    if (i < index_dim) {
      slice_limits[i] = index_sharding.tile_assignment().dim(i);
    } else {
      slice_limits[i] = 1;
    }
  }
  Array<int64_t> tile_assignment =
      index_sharding.tile_assignment().array().Slice(slice_starts,
                                                     slice_limits);
  return HloSharding::Tile(tile_assignment, index_sharding.metadata());
}

HloSharding ScatterEffectiveDataSharding(const HloSharding& data_sharding,
                                         const HloScatterInstruction& scatter) {
  if (data_sharding.IsTileMaximal() || data_sharding.IsManual()) {
    return data_sharding;
  }

  const ScatterDimensionNumbers& dnums = scatter.scatter_dimension_numbers();
  const int64_t data_rank =
      scatter.scatter_updates()[0]->shape().dimensions().size();
  DimensionVector tile_assignment_dims(data_rank, 1LL);
  int64_t num_elements = 1;
  for (int64_t i = 0; i < scatter.shape().dimensions().size(); ++i) {
    if (absl::c_binary_search(dnums.inserted_window_dims(), i)) {
      CHECK_LT(i, data_rank);
      tile_assignment_dims[i] = data_sharding.tile_assignment().dim(i);
      num_elements *= data_sharding.tile_assignment().dim(i);
    }
  }
  if (num_elements == data_sharding.tile_assignment().num_elements()) {
    // Data sharding is only on scatter_window_dims. We use this data sharding
    // directly.
    return data_sharding;
  }

  if (num_elements == 1) {
    // Data sharding is only on update_window_dims. We do not shard this
    // scatter op. Return a tile maximal sharding with the first device in
    // data sharding tile assignment.
    return HloSharding::AssignDevice(data_sharding.tile_assignment().first(),
                                     data_sharding.metadata());
  }

  // Data sharding is on both update_window_dims and scatter_window_dims. We
  // shard the scatter op only on scatter_window_dims. For example:
  // - the scatter data has sharding [2,2]{0,1,2,3},
  // - first dimension is scatter_window_dims,
  // - second dimension is update_window_dims,
  // Then the result sharding will be [2,1]{0,2}.
  DimensionVector slice_starts(data_rank, 0LL);
  Array<int64_t> tile_assignment =
      data_sharding.tile_assignment().array().Slice(slice_starts,
                                                    tile_assignment_dims);
  return HloSharding::Tile(tile_assignment, data_sharding.metadata());
}

namespace {

GatherScatterDims GetGatherScatterOperandPassthroughDims(
    const Shape& operand_shape,
    absl::Span<const int64_t> collapsed_or_inserted_dims,
    absl::Span<const int64_t> operand_batching_dims,
    absl::Span<const int64_t> offset_or_window_dims,
    absl::Span<const int64_t> slice_size) {
  GatherScatterDims result;
  CHECK(absl::c_is_sorted(offset_or_window_dims));

  int64_t collapsed_or_batching = 0;
  for (int64_t i = 0; i < operand_shape.dimensions().size(); ++i) {
    if (IsCollapsedOrBatchingDim(collapsed_or_inserted_dims,
                                 operand_batching_dims, i)) {
      collapsed_or_batching++;
      continue;
    }
    if (slice_size[i] != operand_shape.dimensions(i)) {
      continue;
    }
    result.operand_dims.push_back(i);
    result.output_dims.push_back(
        offset_or_window_dims[i - collapsed_or_batching]);
  }

  return result;
}

// If partitioning in the operand only happens in dimensions in passthrough
// dimensions (offset dimensions in the gather output (or scatter update) that
// have the same size as the operand), returns the corresponding output (or
// update) sharding by passing through the input sharding.
std::optional<HloSharding> PassthroughOperandToGatherOutputOrScatterUpdate(
    const Shape& operand_shape, const HloSharding& operand_sharding,
    const int64_t output_or_update_rank,
    absl::Span<const int64_t> collapsed_or_inserted_dims,
    absl::Span<const int64_t> operand_batching_dims,
    absl::Span<const int64_t> offset_or_window_dims,
    absl::Span<const int64_t> slice_size) {
  if (operand_sharding.IsTileMaximal() || operand_sharding.IsManual()) {
    return std::nullopt;
  }

  GatherScatterDims operand_passthrough_dims =
      GetGatherScatterOperandPassthroughDims(
          operand_shape, collapsed_or_inserted_dims, operand_batching_dims,
          offset_or_window_dims, slice_size);
  HloSharding result = PropagateShardingAlongDimsAndReplicateOthers(
      operand_sharding, operand_passthrough_dims.operand_dims,
      operand_passthrough_dims.output_dims, output_or_update_rank);
  if (result.IsTileMaximal()) {
    return std::nullopt;
  }
  return result;
}

// Inverse of PassthroughOperandToGatherOutputOrScatterUpdate.
std::optional<HloSharding> PassthroughGatherOutputOrScatterUpdateToOperand(
    const Shape& operand_shape, const HloSharding& output_or_update_sharding,
    absl::Span<const int64_t> collapsed_or_inserted_dims,
    absl::Span<const int64_t> operand_batching_dims,
    absl::Span<const int64_t> offset_or_window_dims,
    absl::Span<const int64_t> slice_size) {
  if (output_or_update_sharding.IsTileMaximal() ||
      output_or_update_sharding.IsManual()) {
    return output_or_update_sharding;
  }

  GatherScatterDims operand_passthrough_dims =
      GetGatherScatterOperandPassthroughDims(
          operand_shape, collapsed_or_inserted_dims, operand_batching_dims,
          offset_or_window_dims, slice_size);
  HloSharding result = PropagateShardingAlongDimsAndReplicateOthers(
      output_or_update_sharding, operand_passthrough_dims.output_dims,
      operand_passthrough_dims.operand_dims, operand_shape.dimensions().size());
  if (result.IsTileMaximal()) {
    return std::nullopt;
  }
  return result;
}

std::optional<HloSharding> GatherOperandShardingFromOutputParallelDimensions(
    const HloSharding& output_sharding, const HloInstruction& gather,
    const CallGraph& call_graph) {
  if (output_sharding.IsTileMaximal() || output_sharding.IsManual()) {
    return output_sharding;
  }

  GatherScatterDims parallel_dims;

  const GatherDimensionNumbers& dnums = gather.gather_dimension_numbers();
  if (!dnums.operand_batching_dims().empty()) {
    parallel_dims.operand_dims.assign(dnums.operand_batching_dims().begin(),
                                      dnums.operand_batching_dims().end());
    parallel_dims.indices_dims.assign(
        dnums.start_indices_batching_dims().begin(),
        dnums.start_indices_batching_dims().end());
    parallel_dims.FillOutputDimsWithIndicesDims(dnums.index_vector_dim(),
                                                dnums.offset_dims());
  }
  if (std::optional<GatherScatterDims> implicit_parallel_dims =
          GetGatherParallelBatchDims(gather, call_graph)) {
    parallel_dims.append(*implicit_parallel_dims);
  }

  if (parallel_dims.operand_dims.empty()) {
    return std::nullopt;
  }

  return PropagateShardingAlongDimsAndReplicateOthers(
      output_sharding, parallel_dims.output_dims, parallel_dims.operand_dims,
      gather.operand(0)->shape().dimensions().size());
}

}  // namespace

std::optional<HloSharding>
GatherOutputShardingFromOperandOperandPassthroughDimensions(
    const HloSharding& operand_sharding, const HloInstruction& hlo) {
  return GatherOutputShardingFromOperandOperandPassthroughDimensions(
      hlo.operand(0)->shape(), operand_sharding, hlo, hlo.gather_slice_sizes());
}

std::optional<HloSharding>
GatherOutputShardingFromOperandOperandPassthroughDimensions(
    const Shape& operand_shape, const HloSharding& operand_sharding,
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes) {
  const auto& dnums = hlo.gather_dimension_numbers();
  return PassthroughOperandToGatherOutputOrScatterUpdate(
      operand_shape, operand_sharding, hlo.shape().dimensions().size(),
      dnums.collapsed_slice_dims(), dnums.operand_batching_dims(),
      dnums.offset_dims(), slice_sizes);
}

std::optional<HloSharding> GatherOperandShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo,
    const CallGraph& call_graph) {
  const auto& dnums = hlo.gather_dimension_numbers();
  // Prioritize parallel sharding first as this is how it is in
  // spmd_partitioner.
  std::optional<HloSharding> parallel_sharding =
      GatherOperandShardingFromOutputParallelDimensions(output_sharding, hlo,
                                                        call_graph);
  std::optional<HloSharding> passthrough_sharding =
      PassthroughGatherOutputOrScatterUpdateToOperand(
          hlo.operand(0)->shape(), output_sharding,
          dnums.collapsed_slice_dims(), dnums.operand_batching_dims(),
          dnums.offset_dims(), hlo.gather_slice_sizes());
  // Try to merge the two shardings or return the one that is present if only
  // one of the two is.
  if (!passthrough_sharding) {
    return parallel_sharding;
  }
  if (!parallel_sharding) {
    return passthrough_sharding;
  }
  if (MergeSharding(*parallel_sharding, &*passthrough_sharding,
                    /*may_combine_partial_sharding=*/true)) {
    return passthrough_sharding;
  }
  if (MergeSharding(*passthrough_sharding, &*parallel_sharding,
                    /*may_combine_partial_sharding=*/true)) {
    return parallel_sharding;
  }
  return parallel_sharding;
}

std::vector<int64_t> GetScatterSliceSize(const Shape& operand_shape,
                                         const Shape& update_shape,
                                         const ScatterDimensionNumbers& dnums) {
  std::vector<int64_t> slice_size(operand_shape.dimensions().size(), 1);
  int64_t num_update_window_dims = 0;
  for (int64_t i = 0; i < operand_shape.dimensions().size(); ++i) {
    if (IsCollapsedOrBatchingDim(dnums.inserted_window_dims(),
                                 dnums.input_batching_dims(), i)) {
      continue;
    }
    slice_size[i] = update_shape.dimensions(
        dnums.update_window_dims(num_update_window_dims++));
  }
  CHECK_EQ(num_update_window_dims, dnums.update_window_dims_size());
  return slice_size;
}

std::optional<HloSharding> ScatterOutputShardingFromUpdate(
    const HloSharding& update_sharding, const HloScatterInstruction& scatter) {
  const auto& dnums = scatter.scatter_dimension_numbers();
  std::vector<int64_t> slice_size =
      GetScatterSliceSize(scatter.scatter_operands()[0]->shape(),
                          scatter.scatter_updates()[0]->shape(), dnums);
  return PassthroughGatherOutputOrScatterUpdateToOperand(
      scatter.scatter_operands()[0]->shape(), update_sharding,
      dnums.inserted_window_dims(), dnums.input_batching_dims(),
      dnums.update_window_dims(), slice_size);
}

std::optional<HloSharding> ScatterUpdateShardingFromOutput(
    const HloSharding& per_output_sharding,
    const HloScatterInstruction& scatter, const CallGraph& call_graph) {
  // Prioritize parallel sharding first as this is how it is in
  // spmd_partitioner.
  std::optional<HloSharding> parallel_sharding =
      ScatterUpdateShardingFromOutputParallelDimensions(per_output_sharding,
                                                        scatter, call_graph);
  std::optional<HloSharding> passthrough_sharding =
      ScatterUpdateShardingFromOutputOperandPassthroughDimensions(
          per_output_sharding, scatter);
  // Try to merge the two shardings or return the one that is present if only
  // one of the two is.
  if (!passthrough_sharding) {
    return parallel_sharding;
  }
  if (!parallel_sharding) {
    return passthrough_sharding;
  }
  if (MergeSharding(*parallel_sharding, &*passthrough_sharding,
                    /*may_combine_partial_sharding=*/true)) {
    return passthrough_sharding;
  }
  if (MergeSharding(*passthrough_sharding, &*parallel_sharding,
                    /*may_combine_partial_sharding=*/true)) {
    return parallel_sharding;
  }
  return parallel_sharding;
}

std::optional<HloSharding>
ScatterUpdateShardingFromOutputOperandPassthroughDimensions(
    const HloSharding& output_sharding, const HloInstruction& hlo) {
  const HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(&hlo);
  CHECK(scatter);
  const Shape& operand_shape = scatter->scatter_operands()[0]->shape();
  const Shape& update_shape = scatter->scatter_updates()[0]->shape();
  const Shape& output_shape = operand_shape;
  return ScatterUpdateShardingFromOutputOperandPassthroughDimensions(
      output_shape, output_sharding, *scatter,
      GetScatterSliceSize(operand_shape, update_shape,
                          scatter->scatter_dimension_numbers()));
}

std::optional<HloSharding>
ScatterUpdateShardingFromOutputOperandPassthroughDimensions(
    const Shape& output_shape, const HloSharding& output_sharding,
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes) {
  const HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(&hlo);
  CHECK(scatter);
  const auto& dnums = scatter->scatter_dimension_numbers();
  return PassthroughOperandToGatherOutputOrScatterUpdate(
      output_shape, output_sharding,
      scatter->scatter_updates()[0]->shape().dimensions().size(),
      dnums.inserted_window_dims(), dnums.input_batching_dims(),
      dnums.update_window_dims(), slice_sizes);
}

std::optional<HloSharding> ScatterUpdateShardingFromOutputParallelDimensions(
    const HloSharding& output_sharding, const HloScatterInstruction& scatter,
    const CallGraph& call_graph) {
  if (output_sharding.IsTileMaximal() || output_sharding.IsManual()) {
    return output_sharding;
  }

  GatherScatterDims parallel_dims;

  const ScatterDimensionNumbers& dnums = scatter.scatter_dimension_numbers();
  if (!dnums.input_batching_dims().empty()) {
    parallel_dims.operand_dims.assign(dnums.input_batching_dims().begin(),
                                      dnums.input_batching_dims().end());
    parallel_dims.indices_dims.assign(
        dnums.scatter_indices_batching_dims().begin(),
        dnums.scatter_indices_batching_dims().end());
    parallel_dims.FillOutputDimsWithIndicesDims(dnums.index_vector_dim(),
                                                dnums.update_window_dims());
  }
  if (std::optional<GatherScatterDims> implicit_parallel_dims =
          GetScatterParallelBatchDims(scatter, call_graph)) {
    parallel_dims.append(*implicit_parallel_dims);
  }

  if (parallel_dims.operand_dims.empty()) {
    return std::nullopt;
  }

  return PropagateShardingAlongDimsAndReplicateOthers(
      output_sharding, parallel_dims.operand_dims, parallel_dims.output_dims,
      scatter.scatter_updates()[0]->shape().dimensions().size());
}

absl::StatusOr<std::pair<std::unique_ptr<HloInstruction>, HloOpcode>>
IdentityValueAndHloOpcodeForScatterReduceComputation(
    const HloScatterInstruction& scatter) {
  auto computation = scatter.to_apply();
  // We only handle computations with 2 parameters and only 1 calculation.
  if (computation->instruction_count() != 3) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        "Expected scatter reduce computation with 2 parameters and only 1 "
        "calculation");
  }

  auto root_instruction = computation->root_instruction();
  if (root_instruction->opcode() == HloOpcode::kAdd ||
      root_instruction->opcode() == HloOpcode::kOr) {
    return std::make_pair(HloInstruction::CreateConstant(LiteralUtil::Zero(
                              scatter.shape().element_type())),
                          root_instruction->opcode());
  } else if (root_instruction->opcode() == HloOpcode::kMultiply ||
             root_instruction->opcode() == HloOpcode::kAnd) {
    return std::make_pair(HloInstruction::CreateConstant(
                              LiteralUtil::One(scatter.shape().element_type())),
                          root_instruction->opcode());
  } else if (root_instruction->opcode() == HloOpcode::kMaximum) {
    return std::make_pair(HloInstruction::CreateConstant(LiteralUtil::MinValue(
                              scatter.shape().element_type())),
                          root_instruction->opcode());
  } else if (root_instruction->opcode() == HloOpcode::kMinimum) {
    return std::make_pair(HloInstruction::CreateConstant(LiteralUtil::MaxValue(
                              scatter.shape().element_type())),
                          root_instruction->opcode());
  }

  return absl::Status(absl::StatusCode::kInvalidArgument,
                      "Expected scatter reduce computation which is "
                      "add/or/multiply/add/min/max");
}

namespace {

void DevicesForShardingInternal(
    const HloSharding& sharding,
    const absl::flat_hash_set<int64_t>& available_devices,
    absl::flat_hash_set<int64_t>* used) {
  if (sharding.IsTuple()) {
    for (const auto& subsharding : sharding.tuple_elements()) {
      DevicesForShardingInternal(subsharding, available_devices, used);
    }
    return;
  }

  if (sharding.IsReplicated()) {
    for (int64_t device : available_devices) {
      if (!HloSharding::IsReservedDevice(device)) {
        used->insert(device);
      }
    }
    return;
  }

  DCHECK(std::all_of(
      sharding.tile_assignment().array().begin(),
      sharding.tile_assignment().array().end(),
      [&](int64_t device) { return available_devices.contains(device); }));
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> /*indices*/, int64_t device) {
        used->insert(device);
      });
}

}  // namespace

std::vector<int64_t> DevicesForSharding(
    const HloSharding& sharding, absl::Span<const int64_t> available_devices) {
  absl::flat_hash_set<int64_t> available_set;
  for (int64_t device : available_devices) {
    available_set.insert(device);
  }
  absl::flat_hash_set<int64_t> used_set;
  DevicesForShardingInternal(sharding, available_set, &used_set);
  std::vector<int64_t> devices;
  for (int64_t device : available_devices) {
    if (used_set.contains(device)) {
      devices.push_back(device);
    }
  }
  return devices;
}

HloSharding PartiallyReplicateTiledShardingOnDims(
    const HloSharding& sharding, absl::Span<const int64_t> dims_to_replicate) {
  if (sharding.IsTileMaximal() || sharding.IsManual()) {
    return sharding;
  }
  int64_t group_count = 1;
  DimensionVector valid_dims_to_replicate;
  for (int64_t dim : dims_to_replicate) {
    if (dim >= sharding.TiledDataRank()) {
      continue;
    }
    valid_dims_to_replicate.push_back(dim);
    group_count *= sharding.tile_assignment().dim(dim);
  }
  if (group_count == 1) {
    return sharding;
  }
  if (group_count == sharding.NumTiles() && sharding.subgroup_types().empty()) {
    return HloSharding::Replicate(sharding.metadata());
  }
  DimensionVector dim_permutation(sharding.TiledDataRank());
  absl::c_iota(dim_permutation, 0);
  absl::c_stable_sort(dim_permutation, [&](const int64_t a, const int64_t b) {
    return absl::c_linear_search(valid_dims_to_replicate, a) <
           absl::c_linear_search(valid_dims_to_replicate, b);
  });
  auto new_tile =
      TransposeSharding(sharding, dim_permutation).tile_assignment();
  DimensionVector new_tile_shape(
      sharding.tile_assignment().dimensions().begin(),
      sharding.tile_assignment().dimensions().end());
  for (int64_t dim : valid_dims_to_replicate) {
    new_tile_shape[dim] = 1;
  }
  if (sharding.ReplicateOnLastTileDim()) {
    new_tile_shape.back() *= group_count;
    new_tile = new_tile.Reshape(new_tile_shape);
    return HloSharding::PartialTile(new_tile, sharding.metadata());
  } else {
    new_tile_shape.insert(new_tile_shape.begin() + sharding.TiledDataRank(),
                          group_count);
    new_tile = new_tile.Reshape(new_tile_shape);
    std::vector<OpSharding::Type> subgroup_types;
    subgroup_types.push_back(OpSharding::REPLICATED);
    for (OpSharding::Type type : sharding.subgroup_types()) {
      subgroup_types.push_back(type);
    }
    return HloSharding::Subgroup(new_tile, subgroup_types, sharding.metadata());
  }
}

HloSharding PartiallyReplicateTiledShardingOnAllDimsExcept(
    const HloSharding& sharding, absl::Span<const int64_t> dims_to_keep) {
  if (sharding.IsTileMaximal() || sharding.IsManual()) {
    return sharding;
  }
  DimensionVector dims_to_replicate(sharding.TiledDataRank());
  absl::c_iota(dims_to_replicate, 0);

  dims_to_replicate.erase(
      std::remove_if(
          dims_to_replicate.begin(), dims_to_replicate.end(),
          [&](int64_t i) { return absl::c_linear_search(dims_to_keep, i); }),
      dims_to_replicate.end());
  return PartiallyReplicateTiledShardingOnDims(sharding, dims_to_replicate);
}

HloSharding ReplicateAllDataDims(const HloSharding& sharding,
                                 int64_t data_rank) {
  if (sharding.IsManual()) {
    return sharding;
  }
  if (sharding.subgroup_types().empty()) {
    return HloSharding::Replicate(sharding.metadata());
  }
  HloSharding result =
      PartiallyReplicateTiledShardingOnAllDimsExcept(sharding, {});
  if (data_rank >= 0 && data_rank != result.TiledDataRank() &&
      !result.IsTileMaximal()) {
    DimensionVector new_tile_shape(data_rank, 1);
    for (int64_t i = result.TiledDataRank();
         i < result.tile_assignment().num_dimensions(); ++i) {
      new_tile_shape.push_back(result.tile_assignment().dim(i));
    }
    auto tile = result.tile_assignment().Reshape(new_tile_shape);
    result = HloSharding::Subgroup(tile, result.subgroup_types());
  }
  return result;
}

HloSharding RemoveShapeDimensions(const HloSharding& sharding,
                                  absl::Span<const int64_t> dims_to_remove) {
  if (sharding.IsTileMaximal() || dims_to_remove.empty()) {
    return sharding;
  }
  DimensionVector new_tile_shape;
  new_tile_shape.reserve(sharding.tile_assignment().num_dimensions() -
                         dims_to_remove.size());
  for (int64_t i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (absl::c_linear_search(dims_to_remove, i)) {
      CHECK_EQ(sharding.tile_assignment().dim(i), 1);
    } else {
      new_tile_shape.push_back(sharding.tile_assignment().dim(i));
    }
  }
  auto new_tile = sharding.tile_assignment().Reshape(new_tile_shape);
  return sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile, sharding.metadata())
             : HloSharding::Subgroup(new_tile, sharding.subgroup_types(),
                                     sharding.metadata());
}

std::optional<HloSharding> TransposeShardingWithCollapsedDims(
    const HloSharding& source, absl::Span<int64_t const> src_to_tgt,
    absl::Span<int64_t const> tgt_to_src) {
  if (source.IsTileMaximal() || source.IsManual()) {
    return source;
  }
  if (src_to_tgt.size() < source.tile_assignment().num_dimensions()) {
    // Add missing subgroup dims.
    DimensionVector new_src_to_tgt(src_to_tgt.begin(), src_to_tgt.end());
    DimensionVector new_tgt_to_src(tgt_to_src.begin(), tgt_to_src.end());
    for (int64_t i = 0;
         i < source.tile_assignment().num_dimensions() - src_to_tgt.size();
         ++i) {
      new_src_to_tgt.push_back(tgt_to_src.size() + i);
      new_tgt_to_src.push_back(src_to_tgt.size() + i);
    }
    return TransposeShardingWithCollapsedDims(source, new_src_to_tgt,
                                              new_tgt_to_src);
  }
  DimensionVector tgt_dims_skipping_new(tgt_to_src.size(), -1);
  int64_t skipped_tgt_dims = 0;
  int64_t src_non_subgroup_dims =
      src_to_tgt.size() - source.subgroup_types().size();
  int64_t tgt_non_subgroup_dims =
      tgt_to_src.size() - source.subgroup_types().size();
  for (int64_t i = 0; i < tgt_to_src.size(); ++i) {
    if (tgt_to_src[i] < 0) {
      CHECK_LT(i, tgt_non_subgroup_dims)
          << "Sharding transpose should not remove subgroup dims.";
      skipped_tgt_dims++;
    } else {
      tgt_dims_skipping_new[i] = i - skipped_tgt_dims;
    }
  }
  int64_t skipped_src_dims = absl::c_count(src_to_tgt, -1);
  DimensionVector perm(src_to_tgt.size());
  for (int64_t i = 0; i < src_non_subgroup_dims; ++i) {
    if (src_to_tgt[i] < 0) {
      if (source.tile_assignment().dim(i) > 1) {
        return std::nullopt;
      }
      perm[src_non_subgroup_dims - skipped_src_dims] = i;
      skipped_src_dims--;
    } else {
      perm[tgt_dims_skipping_new[src_to_tgt[i]]] = i;
    }
  }
  skipped_src_dims = absl::c_count(src_to_tgt, -1);
  for (int64_t i = src_non_subgroup_dims; i < src_to_tgt.size(); ++i) {
    CHECK_GE(src_to_tgt[i], tgt_non_subgroup_dims)
        << "Sharding transpose should not move subgroup dims before data dims.";
    perm[src_to_tgt[i] - skipped_tgt_dims + skipped_src_dims] = i;
  }
  auto tgt_sharding = TransposeSharding(source, perm);
  DimensionVector tgt_tiles(tgt_to_src.size(), 1);
  for (int64_t i = 0; i < tgt_tiles.size(); ++i) {
    if (tgt_to_src[i] >= 0) {
      int64_t dim = tgt_dims_skipping_new[i];
      if (i >= tgt_non_subgroup_dims) {
        dim += skipped_src_dims;
      }
      tgt_tiles[i] = tgt_sharding.tile_assignment().dim(dim);
    }
  }
  auto reshape_tiles = tgt_sharding.tile_assignment().Reshape(tgt_tiles);
  return source.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(reshape_tiles, source.metadata())
             : HloSharding::Subgroup(reshape_tiles, source.subgroup_types(),
                                     source.metadata());
}

std::optional<int64_t> GetDimensionForIota(const HloInstruction* maybe_iota,
                                           const CallGraph& call_graph) {
  if (auto* iota = DynCast<HloIotaInstruction>(maybe_iota)) {
    return iota->iota_dimension();
  }

  if (maybe_iota->shape().element_type() != S32) {
    return std::nullopt;
  }
  if (maybe_iota->IsConstant()) {
    std::vector<bool> is_iota_dim(maybe_iota->shape().dimensions().size(),
                                  true);
    maybe_iota->literal().EachCell<int32_t>(
        [&](absl::Span<const int64_t> indices, int32_t val) {
          for (int64_t i = 0; i < indices.size(); ++i) {
            if (val != indices[i]) {
              is_iota_dim[i] = false;
            }
          }
        });
    for (int64_t i = 0; i < is_iota_dim.size(); ++i) {
      if (is_iota_dim[i] && maybe_iota->shape().dimensions(i) > 1) {
        return i;
      }
    }
    return std::nullopt;
  }

  if (maybe_iota->opcode() == HloOpcode::kBroadcast) {
    auto operand_dim = GetDimensionForIota(maybe_iota->operand(0), call_graph);
    if (operand_dim) {
      return maybe_iota->dimensions(*operand_dim);
    }
    return std::nullopt;
  }

  // Returns the iota dimension if maybe_iota is of the following pattern:
  //
  //                                                     Parameter
  //        Op       Iota                                   |
  //         |        |                               +-----+-----+
  //         +--------+                               |           |
  //             |                                   GTE         GTE(to_match)
  //           Tuple                                  |           |
  //             |     while_body/call_computation    .           |
  //           While ------------------------------>  .           |
  //                                                  .           |
  //                                                  |           |
  //                                                  +-----+-----+
  //                                                        |
  //                                                      Tuple
  //
  if (maybe_iota->opcode() == HloOpcode::kGetTupleElement &&
      maybe_iota->operand(0)->opcode() == HloOpcode::kParameter) {
    // If it traces back to the argument from a non-entry computation,
    // check if the argument in the caller's computation could be a iota.
    const HloComputation* called_computation = maybe_iota->parent();
    if (!called_computation->IsEntryComputation()) {
      const HloInstruction* gte = maybe_iota;
      const int64_t gte_index = gte->tuple_index();
      std::vector<HloInstruction*> callers =
          call_graph.GetComputationCallers(called_computation);
      // FlattenCallGraph pass should have ensured that this call site is
      // associated with an unique computation.
      CHECK_EQ(callers.size(), 1);
      HloInstruction* caller =
          call_graph.GetComputationCallers(called_computation)[0];
      // Support tracing only caller that's either a conditional or while
      // (other types of non-entry computations are not partitioned).
      if (caller->opcode() == HloOpcode::kWhile &&
          caller->operand(0)->opcode() == HloOpcode::kTuple) {
        // Check tuple parameter of the while body is invariant at tuple index
        // position across 0th and remaining iterations.
        HloInstruction* while_root = called_computation->root_instruction();
        if (while_root->opcode() == HloOpcode::kTuple &&
            while_root->operand(gte_index) == gte) {
          return GetDimensionForIota(caller->operand(0)->operand(gte_index),
                                     call_graph);
        }
      }
      if (caller->opcode() == HloOpcode::kConditional) {
        int64_t cond_comp_idx =
            absl::c_find(caller->branch_computations(), called_computation) -
            caller->branch_computations().begin();
        CHECK(cond_comp_idx < caller->branch_computations().size());
        const HloInstruction* branch_comp_arg =
            caller->operand(cond_comp_idx + 1);
        CHECK(branch_comp_arg->shape().IsTuple());
        return GetDimensionForIota(branch_comp_arg->operand(gte_index),
                                   call_graph);
      }
    }
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<GatherScatterDims> GetGatherScatterBatchParallelDims(
    const HloInstruction* operand, const HloInstruction* indices,
    absl::Span<const int64_t> slice_sizes, int64_t index_vector_dim,
    absl::Span<const int64_t> index_map,
    absl::Span<const int64_t> indices_batching_dims,
    absl::Span<const int64_t> offset_or_window_dims,
    const CallGraph& call_graph) {
  // Try to identify if there's a dimension in the indices that is monotonically
  // increasing with a Iota across a certain dimension. This would mean that the
  // access in the relative dimension indexed by this index in the operand is
  // parallelizable and that we can shard the operand (and the index/output)
  // across such dimension.
  // For example the pattern:
  //   %iota.1 = iota()
  //   %indices = concatenate(..., %iota.1, ...)
  //   ... = gather(..., %indices)
  // is common for tf.reverse_sequence and would match this case.
  std::vector<int64_t> index_parallel_in_dim(index_map.size(), -1);

  // looks through any copies to find the concatenate.
  auto findConcatenate = [&](const HloInstruction* indices) {
    const HloInstruction* orig_indices = indices;
    while (indices->opcode() == HloOpcode::kCopy) {
      indices = indices->operand(0);
    }
    if (indices->opcode() == HloOpcode::kConcatenate) {
      return indices;
    }
    return orig_indices;
  };
  indices = findConcatenate(indices);

  // Handle cases where we concatenate pieces of the indices one at a time.
  if (indices->opcode() == HloOpcode::kConcatenate &&
      indices->concatenate_dimension() == index_vector_dim) {
    int concatenated_dims = 0;
    for (const HloInstruction* op : indices->operands()) {
      const int64_t num_indices_from_element =
          op->shape().dimensions().size() > index_vector_dim
              ? op->shape().dimensions(index_vector_dim)
              : 1;
      if (std::optional<int64_t> maybe_iota_dim =
              GetDimensionForIota(op, call_graph)) {
        if (*maybe_iota_dim != index_vector_dim) {
          for (int j = 0; j < num_indices_from_element; ++j) {
            index_parallel_in_dim[concatenated_dims + j] = *maybe_iota_dim;
          }
        }
      }
      concatenated_dims += num_indices_from_element;
    }
  } else if (std::optional<int64_t> maybe_iota_dim =
                 GetDimensionForIota(indices, call_graph)) {
    if (*maybe_iota_dim != index_vector_dim) {
      // This is a case of a single iota with index_dim being out of bounds.
      const int64_t num_indices_from_element =
          indices->shape().dimensions().size() > index_vector_dim
              ? indices->shape().dimensions(index_vector_dim)
              : 1;
      index_parallel_in_dim.assign(num_indices_from_element, *maybe_iota_dim);
    }
  }

  GatherScatterDims result;
  // Map the parallelizable dimension from the iota to the dimensions of the
  // output and the operand. These dimensions are interconnected, but between
  // operands and index they could have different spots in the shape because the
  // position of the index dimension in the operand is determined by index_map.
  for (int64_t i = 0; i < index_parallel_in_dim.size(); ++i) {
    int64_t indices_parallel_dim = index_parallel_in_dim[i];
    int64_t operand_parallel_dim = index_map[i];
    if (indices_parallel_dim == -1) {
      continue;
    }
    if (absl::c_linear_search(result.indices_dims, indices_parallel_dim)) {
      return std::nullopt;
    }
    // Considered parallel if (1) the slice size is 1 over the operand, (2) it
    // is not explicit batching dimension, and (3) the dimension size is the
    // same in the operand and the indices.
    if (slice_sizes[operand_parallel_dim] == 1 &&
        !absl::c_linear_search(indices_batching_dims, indices_parallel_dim)) {
      if (operand->shape().dimensions(operand_parallel_dim) !=
          indices->shape().dimensions(indices_parallel_dim)) {
        return std::nullopt;
      }
      result.indices_dims.push_back(indices_parallel_dim);
      result.operand_dims.push_back(operand_parallel_dim);
    }
  }

  if (result.indices_dims.empty()) {
    return std::nullopt;
  }

  result.FillOutputDimsWithIndicesDims(index_vector_dim, offset_or_window_dims);
  return result;
}

std::optional<GatherScatterDims> GetGatherParallelBatchDims(
    const HloInstruction& hlo, const CallGraph& call_graph) {
  CHECK(DynCast<HloGatherInstruction>(&hlo));
  const HloInstruction* operand = hlo.operand(0);
  const HloInstruction* indices = hlo.operand(1);
  absl::Span<const int64_t> slice_sizes = hlo.gather_slice_sizes();
  const auto& dnums = hlo.gather_dimension_numbers();
  return GetGatherScatterBatchParallelDims(
      operand, indices, slice_sizes, dnums.index_vector_dim(),
      dnums.start_index_map(), dnums.start_indices_batching_dims(),
      dnums.offset_dims(), call_graph);
}

std::optional<GatherScatterDims> GetScatterParallelBatchDims(
    const HloInstruction& hlo, const CallGraph& call_graph) {
  const HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(&hlo);
  CHECK(scatter);
  const HloInstruction* operand = scatter->scatter_operands()[0];
  const HloInstruction* indices = scatter->scatter_indices();
  const auto& dnums = hlo.scatter_dimension_numbers();
  std::vector<int64_t> slice_sizes =
      GetScatterSliceSize(scatter->scatter_operands()[0]->shape(),
                          scatter->scatter_updates()[0]->shape(), dnums);
  return GetGatherScatterBatchParallelDims(
      operand, indices, slice_sizes, dnums.index_vector_dim(),
      dnums.scatter_dims_to_operand_dims(),
      dnums.scatter_indices_batching_dims(), dnums.update_window_dims(),
      call_graph);
}

GatherScatterDims GetGatherOperandPassthroughDims(
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes) {
  const auto& dnums = hlo.gather_dimension_numbers();
  return GetGatherScatterOperandPassthroughDims(
      hlo.operand(0)->shape(), dnums.collapsed_slice_dims(),
      dnums.operand_batching_dims(), dnums.offset_dims(), slice_sizes);
}

GatherScatterDims GetScatterOperandPassthroughDims(
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes) {
  const auto& dnums = hlo.scatter_dimension_numbers();
  return GetGatherScatterOperandPassthroughDims(
      hlo.operand(0)->shape(), dnums.inserted_window_dims(),
      dnums.input_batching_dims(), dnums.update_window_dims(), slice_sizes);
}

GatherScatterDims GetGatherConnectedDimsAcrossIndicesAndOutput(
    int64_t indices_rank, int64_t index_vector_dim, int64_t output_rank,
    absl::Span<const int64_t> offset_dims,
    absl::Span<const int64_t> excluded_indices_dims) {
  GatherScatterDims result;
  for (int64_t output_dim = 0, indices_dim = 0; output_dim < output_rank;
       ++output_dim) {
    if (absl::c_linear_search(offset_dims, output_dim)) {
      continue;
    }
    if (indices_dim == index_vector_dim) {
      indices_dim++;
    }
    CHECK_LT(indices_dim, indices_rank);
    if (!absl::c_linear_search(excluded_indices_dims, indices_dim)) {
      result.indices_dims.push_back(indices_dim);
      result.output_dims.push_back(output_dim);
    }
    ++indices_dim;
  }
  return result;
}

GatherScatterDims GetGatherScatterIndexPassThroughDims(
    const HloInstruction& hlo, const CallGraph& call_graph) {
  if (const auto* gather = DynCast<HloGatherInstruction>(&hlo)) {
    const GatherDimensionNumbers& dnums = gather->gather_dimension_numbers();
    DimensionVector excluded_indices_dims{
        dnums.start_indices_batching_dims().begin(),
        dnums.start_indices_batching_dims().end()};
    if (std::optional<GatherScatterDims> implicit_batch_dims =
            GetGatherParallelBatchDims(hlo, call_graph)) {
      absl::c_copy(implicit_batch_dims->indices_dims,
                   std::back_inserter(excluded_indices_dims));
    }
    return GetGatherConnectedDimsAcrossIndicesAndOutput(
        gather->operand(1)->shape().dimensions().size(),
        dnums.index_vector_dim(), hlo.shape().dimensions().size(),
        dnums.offset_dims(), excluded_indices_dims);
  }

  if (const auto* scatter = DynCast<HloScatterInstruction>(&hlo)) {
    const ScatterDimensionNumbers& dnums = scatter->scatter_dimension_numbers();
    DimensionVector excluded_indices_dims{
        dnums.scatter_indices_batching_dims().begin(),
        dnums.scatter_indices_batching_dims().end()};
    if (std::optional<GatherScatterDims> implicit_batch_dims =
            GetScatterParallelBatchDims(hlo, call_graph)) {
      absl::c_copy(implicit_batch_dims->indices_dims,
                   std::back_inserter(excluded_indices_dims));
    }
    return GetGatherConnectedDimsAcrossIndicesAndOutput(
        scatter->scatter_indices()->shape().dimensions().size(),
        dnums.index_vector_dim(),
        scatter->scatter_updates()[0]->shape().dimensions().size(),
        dnums.update_window_dims(), excluded_indices_dims);
  }

  LOG(FATAL) << "Expected gather or scatter, got " << hlo.ToString();
}

HloSharding InferGatherScatterParallelShardingFromOperandSharding(
    const HloSharding& operand_sharding, const Shape& shape,
    absl::Span<const int64_t> output_aligned_operand_parallel_dims,
    absl::Span<const int64_t> output_parallel_dims) {
  return PropagateShardingAlongDimsAndReplicateOthers(
      operand_sharding, output_aligned_operand_parallel_dims,
      output_parallel_dims, shape.dimensions().size());
}

std::string GroupedSharding::ToString() const {
  auto result =
      absl::StrCat("group dims: ", absl::StrJoin(group_dims, ","), "\n");
  absl::StrAppend(
      &result, "group dim sizes: ", absl::StrJoin(group_dim_sizes, ","), "\n");
  absl::StrAppend(&result, "data rank: ", data_rank, "\n");
  absl::StrAppend(&result, "subgroup manual: ", subgroup_manual, "\n");
  absl::StrAppend(&result, "inner sharding: ", sharding.ToString(), "\n");
  absl::StrAppend(&result, "device groups:", "\n");
  for (auto& device_group : device_groups) {
    absl::StrAppend(&result, "\t", absl::StrJoin(device_group, ","), "\n");
  }
  return result;
}

GroupedSharding GroupShardingOnAllDimsExcept(
    const HloSharding& sharding, absl::Span<const int64_t> non_group_dims,
    bool subgroup_manual) {
  std::vector<int64_t> group_dims(sharding.tile_assignment().num_dimensions());
  absl::c_iota(group_dims, 0);

  group_dims.erase(
      std::remove_if(
          group_dims.begin(), group_dims.end(),
          [&](int64_t i) { return absl::c_linear_search(non_group_dims, i); }),
      group_dims.end());
  return GroupShardingOnDims(sharding, group_dims, subgroup_manual);
}

GroupedSharding GroupShardingOnDims(const HloSharding& sharding,
                                    absl::Span<const int64_t> group_dims,
                                    bool subgroup_manual) {
  std::vector<int64_t> group_dim_shards(group_dims.size(), 1);
  return GroupShardingOnDims(sharding, group_dims, group_dim_shards,
                             subgroup_manual);
}

GroupedSharding GroupShardingOnDims(const HloSharding& sharding,
                                    absl::Span<const int64_t> group_dims,
                                    absl::Span<const int64_t> group_dim_shards,
                                    bool subgroup_manual) {
  CHECK(!sharding.IsTileMaximal());

  // The first item of the pair is the group_dim_size. The second item is the
  // group_dim_shard.
  std::vector<std::pair<int64_t, int64_t>> decomposed_tiling_dims(
      sharding.tile_assignment().num_dimensions());
  for (int64_t i = 0; i < decomposed_tiling_dims.size(); ++i) {
    // Set default values for group_dim_size and group_dim_shard.
    decomposed_tiling_dims[i] =
        std::make_pair(1, sharding.tile_assignment().dim(i));
  }

  DimensionVector group_dim_sizes(group_dims.size());
  for (int64_t i = 0; i < group_dims.size(); ++i) {
    CHECK_EQ(
        sharding.tile_assignment().dim(group_dims[i]) % group_dim_shards[i], 0);
    group_dim_sizes[i] =
        sharding.tile_assignment().dim(group_dims[i]) / group_dim_shards[i];

    decomposed_tiling_dims[group_dims[i]].first = group_dim_sizes[i];
    decomposed_tiling_dims[group_dims[i]].second = group_dim_shards[i];
  }

  DimensionVector grouped_tiling_dims(decomposed_tiling_dims.size());
  for (int64_t i = 0; i < decomposed_tiling_dims.size(); ++i) {
    grouped_tiling_dims[i] = decomposed_tiling_dims[i].second;
  }

  DimensionVector sorted_group_dims(group_dims.size());
  std::partial_sort_copy(group_dims.begin(), group_dims.end(),
                         sorted_group_dims.begin(), sorted_group_dims.end());

  absl::flat_hash_map<int64_t, int64_t> group_dim_to_index(group_dims.size());
  DimensionVector reshape_dimensions(grouped_tiling_dims.begin(),
                                     grouped_tiling_dims.end());
  reshape_dimensions.reserve(decomposed_tiling_dims.size() + group_dims.size());
  for (int64_t i = 0; i < sorted_group_dims.size(); ++i) {
    int64_t index = sorted_group_dims[i] + i;
    group_dim_to_index[sorted_group_dims[i]] = index;
    reshape_dimensions.insert(
        reshape_dimensions.begin() + index,
        decomposed_tiling_dims[sorted_group_dims[i]].first);
  }

  std::vector<int> perm(reshape_dimensions.size());
  absl::c_iota(perm, 0);
  for (int64_t i = 0; i < group_dims.size(); ++i) {
    const int64_t index = group_dim_to_index[group_dims[i]];
    perm.erase(std::remove(perm.begin(), perm.end(), index), perm.end());
    perm.insert(perm.begin() + i, index);
  }

  auto grouped_array = sharding.tile_assignment()
                           .Reshape(reshape_dimensions)
                           .Transpose(perm)
                           .array();

  const int64_t num_device_groups = Product(group_dim_sizes);
  const int64_t num_devices = sharding.tile_assignment().num_elements();
  CHECK_EQ(num_devices % num_device_groups, 0);
  const int64_t device_group_size = num_devices / num_device_groups;
  std::vector<std::vector<int64_t>> device_groups(
      num_device_groups, std::vector<int64_t>(device_group_size));
  for (int64_t i = 0; i < num_device_groups; ++i) {
    device_groups[i].assign(
        grouped_array.begin() + i * device_group_size,
        grouped_array.begin() + (i + 1) * device_group_size);
  }

  auto grouped = GroupedSharding(
      std::move(device_groups),
      DimensionVector(group_dims.begin(), group_dims.end()),
      std::move(group_dim_sizes), sharding.tile_assignment().num_dimensions(),
      HloSharding::Replicate(), subgroup_manual);
  if (sharding.ReplicateOnLastTileDim()) {
    grouped.data_rank--;
  }
  if (sharding.IsManualSubgroup()) {
    grouped.data_rank -= sharding.subgroup_types().size();
  }
  if (Product(grouped_tiling_dims) == 1 ||
      (sharding.ReplicateOnLastTileDim() &&
       Product(grouped_tiling_dims) == grouped_tiling_dims.back())) {
    return grouped;
  }
  if (sharding.IsManualSubgroup()) {
    int64_t tile_dimensions = sharding.tile_assignment().num_dimensions();
    int64_t subgroup_size = sharding.subgroup_types().size();
    int64_t rank = tile_dimensions - subgroup_size;
    int num_dims_erase = 0;
    for (int i = 0; i < subgroup_size; i++) {
      if (sharding.subgroup_types()[i] == OpSharding::MANUAL) {
        grouped_tiling_dims.erase(grouped_tiling_dims.begin() + i + rank -
                                  num_dims_erase);
        num_dims_erase++;
      }
    }
  }
  if (sharding.ReplicateOnLastTileDim() && grouped_tiling_dims.back() == 1) {
    grouped_tiling_dims.pop_back();
  }
  TileAssignment grouped_tiling(grouped_tiling_dims);
  grouped.sharding =
      sharding.ReplicateOnLastTileDim() &&
              grouped_tiling_dims.size() ==
                  sharding.tile_assignment().num_dimensions()
          ? HloSharding::PartialTile(grouped_tiling, sharding.metadata())
          : HloSharding::Tile(grouped_tiling, sharding.metadata());
  return grouped;
}

namespace {

std::vector<int64_t> PrimeFactorization(int64_t num) {
  std::vector<int64_t> prime_factors;
  while (num % 2 == 0) {
    prime_factors.push_back(2);
    num /= 2;
  }
  for (int64_t i = 3; i <= sqrt(num); i += 2) {
    while (num % i == 0) {
      prime_factors.push_back(i);
      num /= i;
    }
  }
  return prime_factors;
}

}  // namespace

GroupedSharding GroupShardingOnReplicatedDim(
    const HloSharding& sharding, int64_t num_groups, int64_t num_tiles,
    int64_t data_rank, absl::Span<const int64_t> replicable_dims) {
  // 1. Try group sharding on partially replicated dim.
  if (sharding.ReplicateOnLastTileDim() &&
      sharding.tile_assignment().dimensions().back() % num_groups == 0) {
    absl::InlinedVector<int64_t, 1> group_dim_shards = {
        sharding.tile_assignment().dimensions().back() / num_groups};
    return GroupShardingOnDims(
        sharding, {sharding.tile_assignment().num_dimensions() - 1},
        group_dim_shards);
  }

  // 2. Try borrow dimensions from replicable_dims in order, and group sharding.
  if (sharding.IsTiled()) {
    const int64_t reps_on_last_tile_dim =
        sharding.ReplicateOnLastTileDim()
            ? sharding.tile_assignment().dimensions().back()
            : 1;

    const int64_t max_replicable_dimensions = absl::c_accumulate(
        replicable_dims, reps_on_last_tile_dim,
        [&](int64_t product, int64_t dim) {
          return product * sharding.tile_assignment().dim(dim);
        });

    if (max_replicable_dimensions % num_groups == 0 &&
        num_groups % reps_on_last_tile_dim == 0) {
      auto tile_assignment = [&]() -> std::optional<TileAssignment> {
        int dimensions_to_borrow = num_groups / reps_on_last_tile_dim;
        DimensionVector tile_dims(
            sharding.tile_assignment().dimensions().begin(),
            sharding.tile_assignment().dimensions().end());
        if (!sharding.ReplicateOnLastTileDim()) {
          tile_dims.push_back(1);
        }
        for (auto replicable_dim : replicable_dims) {
          for (auto factor : PrimeFactorization(
                   sharding.tile_assignment().dim(replicable_dim))) {
            if (dimensions_to_borrow % factor == 0) {
              tile_dims[replicable_dim] /= factor;
              tile_dims.back() *= factor;
              dimensions_to_borrow /= factor;
              if (dimensions_to_borrow == 1) {
                return TileAssignment(tile_dims);
              }
            }
          }
        }
        return std::nullopt;
      }();
      if (tile_assignment.has_value()) {
        HloSharding partial_sharding = HloSharding::PartialTile(
            tile_assignment.value(), sharding.metadata());
        if (!partial_sharding.IsReplicated()) {
          return GroupShardingOnDims(
              partial_sharding,
              {partial_sharding.tile_assignment().num_dimensions() - 1});
        }
      }
    }
  }

  // 3. Otherwise return a grouped replicated sharding.
  return GetGroupedReplicatedSharding(num_groups, num_tiles, data_rank);
}

GroupedSharding GetGroupedReplicatedSharding(const int64_t num_groups,
                                             const int64_t num_tiles,
                                             const int64_t data_rank) {
  CHECK_EQ(num_tiles % num_groups, 0);
  const int64_t group_size = num_tiles / num_groups;
  std::vector<std::vector<int64_t>> device_groups(
      num_groups, std::vector<int64_t>(group_size));
  int64_t device_id = 0;
  for (auto& device_group : device_groups) {
    absl::c_iota(device_group, device_id);
    device_id = device_group.back() + 1;
  }
  return GroupedSharding(std::move(device_groups), {data_rank}, {num_groups},
                         data_rank, HloSharding::Replicate(),
                         /*subgroup_manual=*/false);
}

GroupedSharding GetManualSubgroupSharding(const HloSharding& sharding) {
  CHECK(sharding.IsManualSubgroup());
  int64_t tile_dimensions = sharding.tile_assignment().num_dimensions();
  int64_t subgroup_size = sharding.subgroup_types().size();
  int64_t rank = tile_dimensions - subgroup_size;
  DimensionVector group_dims;
  bool last_tile_dim_replicate = false;

  for (int64_t i = 0; i < subgroup_size; i++) {
    if (sharding.subgroup_types()[i] == OpSharding::MANUAL) {
      group_dims.push_back(rank + i);
    } else if (sharding.subgroup_types()[i] == OpSharding::REPLICATED) {
      last_tile_dim_replicate = true;
    }
  }

  GroupedSharding group_sharding =
      GroupShardingOnDims(sharding, group_dims, /*subgroup_manual=*/true);

  if (last_tile_dim_replicate ||
      group_sharding.sharding.tile_assignment().num_dimensions() > rank) {
    group_sharding.sharding = HloSharding::PartialTile(
        group_sharding.sharding.tile_assignment(), sharding.metadata());
  }
  return group_sharding;
}

std::optional<GroupedSharding>
PartialReplicatedGroupShardingWithAssignedDeviceGroups(
    const HloSharding& sharding, int64_t num_shards,
    const std::vector<std::vector<int64_t>>& device_groups) {
  if (!sharding.ReplicateOnLastTileDim() ||
      sharding.tile_assignment().dimensions().back() % device_groups.size() !=
          0) {
    VLOG(5) << "Failed because not partial replicated or not divisible";
    return std::nullopt;
  }
  std::vector<DimensionVector> device_to_index(
      Product(sharding.tile_assignment().dimensions()),
      DimensionVector(sharding.tile_assignment().num_dimensions()));
  sharding.tile_assignment().Each(
      [&device_to_index](absl::Span<const int64_t> indices, int64_t device) {
        device_to_index[device].assign(indices.begin(), indices.end());
      });
  DimensionVector grouped_tiling_dims(
      sharding.tile_assignment().dimensions().begin(),
      sharding.tile_assignment().dimensions().end());
  grouped_tiling_dims.back() /= device_groups.size();
  std::optional<HloSharding> final_sharding;
  const int64_t shard_size_on_replicated_dim =
      sharding.tile_assignment().dimensions().back() / num_shards;
  for (int64_t group_idx = 0; group_idx < device_groups.size(); ++group_idx) {
    HloSharding group_sharding = HloSharding::Replicate();
    Array<int64_t> grouped_tiling(grouped_tiling_dims);
    Array<int64_t> stacked_pos(
        absl::MakeConstSpan(grouped_tiling_dims.data(),
                            grouped_tiling_dims.size() - 1),
        0);
    for (int64_t device_idx = 0; device_idx < device_groups[group_idx].size();
         ++device_idx) {
      VLOG(5) << "Device idx: " << device_idx;
      const int64_t device = device_groups[group_idx][device_idx];
      const auto& indices = device_to_index[device];
      absl::Span<const int64_t> stacked_pos_idx =
          absl::MakeConstSpan(indices.data(), indices.size() - 1);
      int64_t& position = stacked_pos(stacked_pos_idx);
      if (position == num_shards) {
        VLOG(5) << "Fail because stacked position overflow " << position
                << " device_groups " << device_groups.size() << " ["
                << absl::StrJoin(indices, ",") << "]";
        VLOG(5) << "Device: " << device << " "
                << device_groups[group_idx][device_idx];
        VLOG(5) << "Indices: " << absl::StrJoin(indices, ",");
        VLOG(5) << "Grouped tiling: " << grouped_tiling.ToString();
        return std::nullopt;
      }
      auto stacked_indices = indices;
      stacked_indices.back() = position++;
      grouped_tiling(stacked_indices) = device_idx;
    }
    group_sharding =
        HloSharding::PartialTile(grouped_tiling, sharding.metadata());
    if (!final_sharding) {
      final_sharding = group_sharding;
      continue;
    }
    if (*final_sharding != group_sharding) {
      VLOG(5) << "Fail because final sharding different from group sharding: "
              << final_sharding->ToString() << " vs "
              << group_sharding.ToString();
      return std::nullopt;
    }
  }
  return GroupedSharding(device_groups,
                         {sharding.tile_assignment().num_dimensions() - 1},
                         {shard_size_on_replicated_dim},
                         sharding.tile_assignment().num_dimensions() - 1,
                         *final_sharding, /*subgroup_manual=*/false);
}

HloSharding UngroupSharding(const GroupedSharding& grouped_sharding) {
  DimensionVector tiling_dims;
  bool partial_sharding = false;
  std::vector<OpSharding::Type> subgroup_types;
  auto grouped_tiling = grouped_sharding.sharding.tile_assignment();
  if (grouped_sharding.sharding.IsTileMaximal()) {
    tiling_dims = DimensionVector(grouped_sharding.data_rank, 1);
    if (grouped_sharding.device_groups[0].size() != 1 ||
        absl::c_linear_search(grouped_sharding.group_dims,
                              tiling_dims.size())) {
      // This is partial sharding.
      tiling_dims.push_back(grouped_sharding.device_groups[0].size());
      partial_sharding = true;
    }
    grouped_tiling = TileAssignment(tiling_dims);
  }

  // Handles subgroup manual first.
  if (grouped_sharding.subgroup_manual) {
    partial_sharding = grouped_sharding.sharding.ReplicateOnLastTileDim() ||
                       grouped_sharding.sharding.IsReplicated();
    int64_t subgroup_dim_size = grouped_sharding.group_dims.size();
    if (partial_sharding) {
      subgroup_dim_size++;
    }
    subgroup_types = std::vector<OpSharding::Type>(subgroup_dim_size,
                                                   OpSharding::REPLICATED);
    if (!grouped_sharding.sharding.IsTileMaximal()) {
      tiling_dims.assign(
          grouped_sharding.sharding.tile_assignment().dimensions().begin(),
          grouped_sharding.sharding.tile_assignment().dimensions().end());
    }
    for (int i = 0; i < grouped_sharding.group_dims.size(); i++) {
      subgroup_types[grouped_sharding.group_dims[i] -
                     grouped_sharding.data_rank] = OpSharding::MANUAL;
      tiling_dims.insert(tiling_dims.begin() + grouped_sharding.group_dims[i],
                         1);
    }
  } else if (!grouped_sharding.sharding.IsTileMaximal()) {
    // Handles tile replicated.
    partial_sharding = grouped_sharding.sharding.ReplicateOnLastTileDim();
    tiling_dims.assign(
        grouped_sharding.sharding.tile_assignment().dimensions().begin(),
        grouped_sharding.sharding.tile_assignment().dimensions().end());
    if (absl::c_linear_search(grouped_sharding.group_dims,
                              tiling_dims.size())) {
      tiling_dims.push_back(1);
      partial_sharding = true;
    }
  }

  DimensionVector group_dim_sizes_and_tiling_dims(
      grouped_sharding.group_dim_sizes.begin(),
      grouped_sharding.group_dim_sizes.end());
  group_dim_sizes_and_tiling_dims.insert(group_dim_sizes_and_tiling_dims.end(),
                                         tiling_dims.begin(),
                                         tiling_dims.end());
  Array<int64_t> tiling(group_dim_sizes_and_tiling_dims);

  DimensionVector sorted_group_dims(grouped_sharding.group_dims.size());
  std::partial_sort_copy(grouped_sharding.group_dims.begin(),
                         grouped_sharding.group_dims.end(),
                         sorted_group_dims.begin(), sorted_group_dims.end());
  absl::flat_hash_map<int64_t, int64_t> group_dim_to_index(
      grouped_sharding.group_dims.size());
  for (int64_t i = 0; i < sorted_group_dims.size(); ++i) {
    group_dim_to_index[sorted_group_dims[i]] = sorted_group_dims[i] + i;
  }

  std::vector<int> perm(tiling_dims.size() + grouped_sharding.group_dims.size(),
                        -1);
  for (int64_t i = 0; i < grouped_sharding.group_dims.size(); i++) {
    perm[group_dim_to_index[grouped_sharding.group_dims[i]]] = i;
  }
  int64_t j = grouped_sharding.group_dims.size();
  for (int64_t i = 0; i < perm.size(); i++) {
    if (perm[i] == -1) {
      perm[i] = j++;
    }
  }

  std::vector<int64_t> flattened_device_groups;
  flattened_device_groups.reserve(grouped_sharding.device_groups.size() *
                                  grouped_sharding.device_groups[0].size());
  bool same_length =
      grouped_tiling.num_elements() == grouped_sharding.device_groups[0].size();
  for (auto const& v : grouped_sharding.device_groups) {
    if (same_length) {
      // Reorder the device_groups based on the grouped_tiling.array()
      for (int64_t i = 0; i < v.size(); ++i) {
        flattened_device_groups.push_back(
            v[*(grouped_tiling.array().begin() + i)]);
      }
    } else {
      flattened_device_groups.insert(flattened_device_groups.end(), v.begin(),
                                     v.end());
    }
  }
  tiling.SetValues(flattened_device_groups);
  TileAssignment tile_assignment(
      std::make_shared<const Array<int64_t>>(std::move(tiling)));

  for (int64_t i = 0; i < grouped_sharding.group_dims.size(); ++i) {
    int64_t dim = grouped_sharding.group_dims[i];
    tiling_dims[dim] *= grouped_sharding.group_dim_sizes[i];
  }
  tile_assignment = tile_assignment.Transpose(perm).Reshape(tiling_dims);

  if (grouped_sharding.subgroup_manual) {
    return HloSharding::Subgroup(tile_assignment, subgroup_types,
                                 grouped_sharding.sharding.metadata());
  }
  return partial_sharding ? HloSharding::PartialTile(tile_assignment)
                          : HloSharding::Tile(tile_assignment);
}

bool DeviceGroupsAreMatch(GroupedSharding& lhs, GroupedSharding& rhs,
                          bool ignore_group_order) {
  if (lhs.device_groups.size() != rhs.device_groups.size()) {
    return false;
  }

  bool matching_groups = true;
  std::vector<int64_t> device_to_ref_group(lhs.device_groups.size() *
                                           lhs.device_groups[0].size());
  for (int64_t g = 0; g < lhs.device_groups.size(); ++g) {
    for (int64_t device : lhs.device_groups[g]) {
      device_to_ref_group[device] = g;
    }
  }
  auto unique_ref_dev_group =
      [&](absl::Span<const int64_t> devices) -> int64_t {
    int64_t ref_g = -1;
    for (int64_t device : devices) {
      if (ref_g == -1) {
        ref_g = device_to_ref_group[device];
      } else if (ref_g != device_to_ref_group[device]) {
        return -1;
      }
    }
    return ref_g;
  };
  for (int64_t g = 0; g < rhs.device_groups.size(); ++g) {
    int64_t ref_g = unique_ref_dev_group(rhs.device_groups[g]);
    if (ref_g < 0 || (!ignore_group_order && g != ref_g)) {
      matching_groups = false;
      break;
    }
  }

  return matching_groups;
}

HloSharding SplitShardingDimension(const HloSharding& sharding,
                                   int64_t dimension, int64_t new_dim_size) {
  CHECK_GT(sharding.TiledDataRank(), dimension);
  CHECK_EQ(sharding.tile_assignment().dim(dimension) % new_dim_size, 0)
      << "dim size " << new_dim_size;
  DimensionVector dimensions(sharding.tile_assignment().dimensions().begin(),
                             sharding.tile_assignment().dimensions().end());
  int64_t current_dimension = dimensions[dimension];
  dimensions.insert(dimensions.begin() + dimension + 1,
                    current_dimension / new_dim_size);
  dimensions[dimension] = new_dim_size;
  auto new_tile_assignment = sharding.tile_assignment().Reshape(dimensions);
  return sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment)
             : HloSharding::Subgroup(new_tile_assignment,
                                     sharding.subgroup_types());
}

HloSharding MergeShardingDimension(const HloSharding& sharding,
                                   int64_t dimension) {
  CHECK_GT(sharding.TiledDataRank(), dimension);
  DimensionVector dimensions(sharding.tile_assignment().dimensions().begin(),
                             sharding.tile_assignment().dimensions().end());
  dimensions[dimension] *= dimensions[dimension + 1];
  dimensions.erase(dimensions.begin() + dimension + 1);
  auto new_tile_assignment = sharding.tile_assignment().Reshape(dimensions);
  return sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment)
             : HloSharding::Subgroup(new_tile_assignment,
                                     sharding.subgroup_types());
}

std::shared_ptr<const HloSharding> CreateTupleSharding(
    const Shape& shape, absl::Span<const HloInstruction* const> elements) {
  bool any_sharding = false;
  for (const HloInstruction* element : elements) {
    any_sharding |= element->has_sharding();
  }
  if (!any_sharding) {
    return nullptr;
  }

  std::vector<HloSharding> sub_shardings;
  sub_shardings.reserve(elements.size());
  for (const HloInstruction* element : elements) {
    if (element->has_sharding()) {
      sub_shardings.push_back(element->sharding());
    } else {
      sub_shardings.push_back(HloSharding::Replicate());
    }
  }
  return std::make_shared<const HloSharding>(
      HloSharding::Tuple(shape, sub_shardings));
}

std::optional<int64_t> GetFirstTargetDimToMoveShardingTiles(
    const Shape& shape, const HloSharding& sharding, int64_t source_dim,
    std::function<bool(int64_t)> can_be_target_dim) {
  if (shape.dimensions().size() < 2 || shape.dimensions(source_dim) == 1) {
    return std::nullopt;
  }
  if (!sharding.IsTiled() || sharding.tile_assignment().dim(source_dim) == 1) {
    return std::nullopt;
  }

  for (int64_t dim = 0; dim < shape.dimensions().size(); ++dim) {
    if (dim == source_dim) {
      continue;
    }
    if (!can_be_target_dim(dim)) {
      continue;
    }
    const int64_t merged_tile_dims =
        sharding.tile_assignment().dim(source_dim) *
        sharding.tile_assignment().dim(dim);
    if (shape.dimensions(dim) % merged_tile_dims == 0) {
      return dim;
    }
  }

  return std::nullopt;
}

std::optional<HloSharding> GetOutputSharding(
    const HloInstruction* instruction) {
  if (!instruction->has_sharding()) {
    return std::nullopt;
  }
  if (instruction->opcode() == HloOpcode::kOutfeed) {
    // Sometime when the same sharding is applied to both the tuple sharding
    // encoding is a single element sharding.
    if (!instruction->sharding().IsTuple()) {
      return instruction->sharding();
    }
    // Token sharding is always the last one.
    return instruction->sharding().tuple_elements().back();
  }
  return instruction->sharding();
}

Shape UntileShape(const HloSharding& sharding, const Shape& shape) {
  if (!sharding.IsTuple()) {
    return UntileLeafShape(sharding, shape);
  }
  Shape result_shape = shape;
  ShapeUtil::ForEachMutableSubshape(
      &result_shape,
      [&shape, &sharding](Shape* subshape, const ShapeIndex& index) {
        if (!ShapeUtil::IsLeafIndex(shape, index)) {
          return;
        }
        const HloSharding& subshape_sharding =
            sharding.GetSubSharding(shape, index);
        *subshape = UntileLeafShape(subshape_sharding, *subshape);
      });

  return result_shape;
}

Shape UntileLeafShape(const HloSharding& sharding, const Shape& shape) {
  if (sharding.IsTileMaximal() || sharding.IsManual() || sharding.IsUnknown()) {
    return shape;
  }
  if (!shape.IsArray()) {
    return shape;
  }
  Shape result_shape = shape;
  // sharding.TiledDataRank() == i < shape.dimensions_size() is not always true?
  for (int64_t i = 0;
       i < sharding.TiledDataRank() && i < shape.dimensions().size(); ++i) {
    result_shape.set_dimensions(
        i, shape.dimensions(i) * sharding.tile_assignment().dim(i));
  }
  return result_shape;
}

Shape TileShape(const HloSharding& sharding, const Shape& shape) {
  if (!sharding.IsTuple()) {
    return TileLeafShape(sharding, shape);
  }
  Shape result_shape = shape;
  ShapeUtil::ForEachMutableSubshape(
      &result_shape,
      [&shape, &sharding](Shape* subshape, const ShapeIndex& index) {
        if (!ShapeUtil::IsLeafIndex(shape, index)) {
          return;
        }
        const HloSharding& subshape_sharding =
            sharding.GetSubSharding(shape, index);
        *subshape = TileLeafShape(subshape_sharding, *subshape);
      });

  return result_shape;
}

Shape TileLeafShape(const HloSharding& sharding, const Shape& shape) {
  if (sharding.IsTileMaximal() || sharding.IsManual() || sharding.IsUnknown()) {
    return shape;
  }
  if (!shape.IsArray()) {
    return shape;
  }
  Shape result_shape = shape;
  for (int64_t i = 0;
       i < sharding.TiledDataRank() && i < shape.dimensions().size(); ++i) {
    CHECK_EQ(shape.dimensions(i) % sharding.tile_assignment().dim(i), 0);
    result_shape.set_dimensions(
        i, shape.dimensions(i) / sharding.tile_assignment().dim(i));
  }
  return result_shape;
}

absl::Status CanonicalizeLayoutAfterShardingPropagation(
    HloModule* module, const std::vector<bool>& update_output_layout,
    const std::vector<bool>& update_parameters_layout) {
  if (!module->layout_canonicalization_callback()) {
    LOG(INFO) << "There is no registered layout_canonicalization_callback.";
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(auto shapes_with_layout,
                      module->layout_canonicalization_callback()(*module));

  if (module->entry_computation_layout().result_layout().LayoutIsSet() &&
      absl::c_any_of(update_output_layout, [](bool v) { return v; })) {
    if (absl::c_all_of(update_output_layout, [](bool v) { return v; })) {
      TF_RETURN_IF_ERROR(module->mutable_entry_computation_layout()
                             ->mutable_result_layout()
                             ->CopyLayoutFromShape(shapes_with_layout.second));
    } else {
      Shape result_shape = module->mutable_entry_computation_layout()
                               ->mutable_result_layout()
                               ->shape();
      CHECK_EQ(result_shape.tuple_shapes_size(),
               shapes_with_layout.second.tuple_shapes_size());
      for (int64_t i = 0; i < result_shape.tuple_shapes_size(); ++i) {
        if (update_output_layout[i]) {
          *result_shape.mutable_tuple_shapes(i) =
              shapes_with_layout.second.tuple_shapes(i);
        }
      }
      TF_RETURN_IF_ERROR(module->mutable_entry_computation_layout()
                             ->mutable_result_layout()
                             ->CopyLayoutFromShape(result_shape));
    }
  }

  if (absl::c_any_of(update_parameters_layout, [](bool v) { return v; })) {
    for (int64_t i = 0; i < module->entry_computation()->num_parameters();
         ++i) {
      bool update_parameter_layout = update_parameters_layout.size() == 1
                                         ? update_parameters_layout[0]
                                         : update_parameters_layout[i];
      bool parameter_layout_is_set =
          module->entry_computation_layout().parameter_layout(i).LayoutIsSet();
      if (update_parameter_layout && parameter_layout_is_set) {
        TF_RETURN_IF_ERROR(
            module->mutable_entry_computation_layout()
                ->mutable_parameter_layout(i)
                ->CopyLayoutFromShape(shapes_with_layout.first[i]));
      }
    }
  }

  return absl::OkStatus();
}

bool IsSpatiallyPartitioned(const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    return absl::c_any_of(sharding.tuple_elements(),
                          [](const HloSharding& sub_sharding) {
                            return IsSpatiallyPartitioned(sub_sharding);
                          });
  } else {
    return !sharding.IsTileMaximal() || sharding.IsReplicated();
  }
}

// Returns
// - 1, iff `lhs` is strictly better than `rhs`.
// - 2, iff `rhs` is strictly better than `lhs`.
// - 0 or 3, otherwise.
//
// Notes:
// - We think manual shardings are strictly better than tile maximal shardings.
// - For tuples we consider lhs to have a better sharding if none of the
//   elements are worse and at least one element is better then in rhs
//   sharding.
int MaskTupleShardingStrictlyBetter(const HloSharding& lhs,
                                    const HloSharding& rhs) {
  DCHECK(lhs.IsTuple());
  DCHECK(rhs.IsTuple());
  const auto& lhs_shardings = lhs.tuple_elements();
  const auto& rhs_shardings = rhs.tuple_elements();
  CHECK_EQ(lhs_shardings.size(), rhs_shardings.size());
  int mask = 0;
  for (int64_t i = 0; i < lhs_shardings.size(); ++i) {
    const auto& lhs_shard = lhs_shardings[i];
    const auto& rhs_shard = rhs_shardings[i];
    CHECK_EQ(lhs_shard.IsTuple(), rhs_shard.IsTuple());
    if (lhs_shard.IsTuple()) {
      mask |= MaskTupleShardingStrictlyBetter(lhs_shard, rhs_shard);
    } else {
      if (lhs_shard.IsManualLeaf() && rhs_shard.IsTileMaximalLeaf()) {
        mask |= 1;
      }
      if (rhs_shard.IsManualLeaf() && lhs_shard.IsTileMaximalLeaf()) {
        mask |= 2;
      }
    }
    if (mask == 3) break;
  }
  return mask;
}

bool IsShardingStrictlyBetter(const HloSharding& lhs, const HloSharding& rhs) {
  CHECK_EQ(lhs.IsTuple(), rhs.IsTuple()) << lhs << " <> " << rhs;
  if (lhs.IsTuple()) {
    return MaskTupleShardingStrictlyBetter(lhs, rhs) == 1;
  }
  return lhs.IsManualLeaf() && rhs.IsTileMaximalLeaf();
}

std::optional<HloSharding> ReturnImprovedShardingImpl(
    HloSharding from, const HloSharding* to_improved,
    const Shape& to_improved_shape, bool may_combine_partial_sharding,
    bool allow_aggressive_resharding) {
  // Always allow improve the sharding if it's straightly better.
  if (to_improved != nullptr && IsShardingStrictlyBetter(from, *to_improved)) {
    return from;
  }
  // We don't want to propagate tile maximal shardings.
  if (!IsSpatiallyPartitioned(from)) {
    return std::nullopt;
  }
  // Any sharding is better than no sharding.
  if (to_improved == nullptr) {
    return from;
  }
  // We don't want to propagate manual shardings.
  if (from.IsManual()) {
    return std::nullopt;
  }
  int64_t sharding_tiles = from.NumTiles();
  if (MergeSharding(*to_improved, &from, may_combine_partial_sharding)) {
    // Override existing tiled sharding only when the new sharding is compatible
    // with the existing one. This avoids unexpected resharding when `sharding`
    // just has more tiles than existing sharding but they are not mergeable.
    if (!allow_aggressive_resharding && to_improved_shape.IsArray() &&
        !to_improved->IsTileMaximal() && from.NumTiles() == sharding_tiles) {
      if (!IsSubTilingOrEqualSharding(to_improved_shape, from, *to_improved)) {
        VLOG(10) << "Not merging because of different device distribution";
        VLOG(10) << "Instr sharding: " << to_improved->ToString();
        VLOG(10) << "New sharding " << from.ToString();
        return std::nullopt;
      }
    }
    return from;
  }
  return std::nullopt;
}

HloSharding InferDotOperandSharding(
    const HloSharding* dot_sharding, const HloSharding* other_operand_sharding,
    int64_t operand_index,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dnums,
    bool consider_other_operand, bool may_combine_partial_sharding) {
  CHECK(operand_index == 0 || operand_index == 1);
  CHECK(dnums.conv_spatial_dims.empty());

  std::vector<int64_t> output_dims_to_replicate;
  std::vector<int64_t> other_operand_dims_to_replicate;
  for (const auto& dim : operand_index == 0 ? dnums.rhs_non_contracting_dims
                                            : dnums.lhs_non_contracting_dims) {
    output_dims_to_replicate.push_back(dim.output);
    other_operand_dims_to_replicate.push_back(operand_index == 0 ? dim.rhs
                                                                 : dim.lhs);
  }
  // If this dot is interpreted from a conv, then contracting dims may have
  // corresponding spatial dimensions in the output, and this operand's
  // non-contracting dims may have corresponding spatial dims in the other
  // operand.
  for (const auto& dim : dnums.contracting_dims) {
    if (dim.output >= 0) {
      output_dims_to_replicate.push_back(dim.output);
    }
  }
  for (const auto& dim : operand_index == 0 ? dnums.lhs_non_contracting_dims
                                            : dnums.rhs_non_contracting_dims) {
    int64_t other_dim = operand_index == 0 ? dim.rhs : dim.lhs;
    if (other_dim >= 0) {
      other_operand_dims_to_replicate.push_back(other_dim);
    }
  }

  int64_t operand_shape_rank =
      operand_index == 0 ? dnums.lhs_shape_rank : dnums.rhs_shape_rank;
  int64_t other_shape_rank =
      operand_index == 0 ? dnums.rhs_shape_rank : dnums.lhs_shape_rank;

  HloSharding sharding = HloSharding::Replicate();

  if (dot_sharding != nullptr) {
    HloSharding output_other_dims_replicated =
        PartiallyReplicateTiledShardingOnDims(*dot_sharding,
                                              output_dims_to_replicate);

    std::vector<int64_t> output_to_operand_dims(dnums.output_shape_rank, -1);
    std::vector<int64_t> operand_to_output_dims(operand_shape_rank, -1);
    for (const auto& dim : dnums.batch_dims) {
      output_to_operand_dims[dim.output] =
          operand_index == 0 ? dim.lhs : dim.rhs;
      operand_to_output_dims[operand_index == 0 ? dim.lhs : dim.rhs] =
          dim.output;
    }
    for (const auto& dim : operand_index == 0
                               ? dnums.lhs_non_contracting_dims
                               : dnums.rhs_non_contracting_dims) {
      output_to_operand_dims[dim.output] =
          operand_index == 0 ? dim.lhs : dim.rhs;
      operand_to_output_dims[operand_index == 0 ? dim.lhs : dim.rhs] =
          dim.output;
    }
    sharding = std::move(*TransposeShardingWithCollapsedDims(
        output_other_dims_replicated, output_to_operand_dims,
        operand_to_output_dims));
  }

  if (consider_other_operand && other_operand_sharding != nullptr &&
      IsSpatiallyPartitioned(*other_operand_sharding)) {
    auto other_operand_dims_replicated = PartiallyReplicateTiledShardingOnDims(
        *other_operand_sharding, other_operand_dims_to_replicate);

    std::vector<int64_t> other_to_operand_dims(other_shape_rank, -1);
    std::vector<int64_t> operand_to_other_dims(operand_shape_rank, -1);
    for (const auto& dim : dnums.batch_dims) {
      other_to_operand_dims[operand_index == 0 ? dim.rhs : dim.lhs] =
          operand_index == 0 ? dim.lhs : dim.rhs;
      operand_to_other_dims[operand_index == 0 ? dim.lhs : dim.rhs] =
          operand_index == 0 ? dim.rhs : dim.lhs;
    }
    for (const auto& dim : dnums.contracting_dims) {
      other_to_operand_dims[operand_index == 0 ? dim.rhs : dim.lhs] =
          operand_index == 0 ? dim.lhs : dim.rhs;
      operand_to_other_dims[operand_index == 0 ? dim.lhs : dim.rhs] =
          operand_index == 0 ? dim.rhs : dim.lhs;
    }
    HloSharding sharding_from_other = *TransposeShardingWithCollapsedDims(
        other_operand_dims_replicated, other_to_operand_dims,
        operand_to_other_dims);
    if (MergeSharding(sharding, &sharding_from_other,
                      may_combine_partial_sharding)) {
      sharding = std::move(sharding_from_other);
    }
  }

  return sharding;
}

HloSharding InferDotOperandSharding(
    const HloInstruction* dot, int64_t operand_index,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dnums,
    bool consider_other_operand, bool may_combine_partial_sharding) {
  CHECK(dot->opcode() == HloOpcode::kDot ||
        dot->opcode() == HloOpcode::kConvolution);

  const HloInstruction* other_operand = dot->operand(1 - operand_index);
  return InferDotOperandSharding(
      dot->has_sharding() ? &dot->sharding() : nullptr,
      other_operand->has_sharding() ? &other_operand->sharding() : nullptr,
      operand_index, dnums, consider_other_operand,
      may_combine_partial_sharding);
}

}  // namespace hlo_sharding_util
}  // namespace xla
