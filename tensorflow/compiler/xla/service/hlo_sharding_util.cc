/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace hlo_sharding_util {

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
  // Different tiled ranks can't be compared (something is wrong, are the
  // shardings for different shapes?)
  if (potential_subsharding.TiledDataRank() != sharding.TiledDataRank()) {
    return false;
  }
  // Helper to construct the base tile bounds based on a shape and a sharding.
  auto get_base_tile_for_sharding = [](const Shape& shape,
                                       const HloSharding& sharding) {
    absl::InlinedVector<int32_t, 5> base_tile;
    base_tile.resize(shape.dimensions_size());
    for (int64_t i = 0; i < shape.dimensions_size(); ++i) {
      base_tile[i] =
          CeilOfRatio(shape.dimensions(i), sharding.tile_assignment().dim(i));
    }
    return base_tile;
  };
  auto potential_base_tile = get_base_tile_for_sharding(potential_sharded_shape,
                                                        potential_subsharding);
  auto base_tile =
      get_base_tile_for_sharding(potential_sharded_shape, sharding);
  // If the potential_base_tile is bigger than the base_tile on any dimension
  // then it can't be contained regardless.
  for (int64_t i = 0; i < potential_base_tile.size(); ++i) {
    if (potential_base_tile[i] > base_tile[i]) {
      return false;
    }
  }
  const int32_t num_devices =
      potential_subsharding.tile_assignment().num_elements();
  // Need a map here, because the MPMD partitioner sharding annotations can have
  // non contiguous partition numbers.
  absl::flat_hash_map<int32_t, std::vector<int32_t>> subsharding_offsets;
  absl::flat_hash_map<int32_t, std::vector<int32_t>> sharding_offsets;
  const int32_t indices_count = potential_subsharding.TiledDataRank();
  // Collect the start offsets for each tile for the subsharding we are
  // evaluating.
  potential_subsharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t device) {
        auto& indices_per_device = subsharding_offsets[device];
        for (int64_t i = 0; i < indices_count; ++i) {
          indices_per_device.push_back(potential_base_tile[i] * indices[i]);
        }
      });
  // Collect the start offsets for each tile for the sharding we are evaluating
  // against.
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t device) {
        auto& indices_per_device = sharding_offsets[device];
        for (int64_t i = 0; i < indices_count; ++i) {
          indices_per_device.push_back(base_tile[i] * indices[i]);
        }
      });
  // Compare the start offsets and the end offset of the tiles for each device.
  for (int i = 0; i < num_devices; ++i) {
    const int32_t device_id = potential_subsharding.tile_assignment().data()[i];
    auto& subsharding_offset = subsharding_offsets[device_id];
    auto& sharding_offset = sharding_offsets[device_id];
    for (int j = 0; j < indices_count; ++j) {
      // The subsharding contains data outside of the tile we are comparing
      // against.
      if (subsharding_offset[j] < sharding_offset[j]) {
        return false;
      }
      // Skip last tile. It can never go beyond the limit as the shape is the
      // same for both shardings and sometimes there's padding making one of the
      // two limits bigger than the other, but it shouldn't be counted.
      const bool is_last_tile =
          subsharding_offset[j] + potential_base_tile[j] >=
          potential_sharded_shape.dimensions(j);
      if (!is_last_tile && subsharding_offset[j] + potential_base_tile[j] >
                               sharding_offset[j] + base_tile[j]) {
        return false;
      }
    }
  }
  return true;
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
  // Manual sharding is more specific than tile maximal sharding.
  if (lhs.IsManual() && rhs.IsTileMaximal()) {
    return true;
  }
  if (lhs.IsManual() || rhs.IsManual()) {
    return false;
  }
  if (!rhs.IsTileMaximal()) {
    return lhs.NumTiles() > rhs.NumTiles();
  } else if (!rhs.IsReplicated()) {
    // If we are not replicated then only tiled (not tile maximal) shardings
    // can improve us.
    return !lhs.IsTileMaximal();
  } else {
    // If we are replicated then any non-replicated sharding can improve us.
    return !lhs.IsReplicated();
  }
}

bool MergeSharding(const HloSharding& old, HloSharding* to_merge,
                   bool may_combine_partial_sharding) {
  if (old.IsTuple()) {
    CHECK(to_merge->IsTuple());
    bool changed = false;
    for (int64_t i = 0; i < old.tuple_elements().size(); ++i) {
      changed |=
          MergeSharding(old.tuple_elements()[i], &to_merge->tuple_elements()[i],
                        may_combine_partial_sharding);
    }
    return changed;
  }
  if (!may_combine_partial_sharding || !old.HasPartialReplication() ||
      !to_merge->HasPartialReplication() ||
      old.tile_assignment().num_elements() !=
          to_merge->tile_assignment().num_elements()) {
    return IsShardingMoreSpecific(*to_merge, old);
  }

  if (MergeShardingIfCompatible(
          old,
          /*minimum_tiles=*/std::max(old.NumTiles(), to_merge->NumTiles()) + 1,
          to_merge)) {
    return true;
  }
  return IsShardingMoreSpecific(*to_merge, old);
}

bool MergeShardingIfCompatible(const HloSharding& to_merge,
                               int64_t minimum_tiles, HloSharding* dst) {
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
  // Combine the tile dimension sizes from dst and to_merge.
  int64_t num_devices = to_merge.tile_assignment().num_elements();
  std::vector<int64_t> merged_tile_dims;
  merged_tile_dims.reserve(dst->tile_assignment().num_dimensions());
  for (int64_t i = 0; i < to_merge.TiledDataRank(); ++i) {
    int64_t dst_dim = dst->tile_assignment().dim(i);
    int64_t merge_dim = to_merge.tile_assignment().dim(i);
    if (dst_dim == 1) {
      merged_tile_dims.push_back(merge_dim);
    } else if (merge_dim == 1) {
      merged_tile_dims.push_back(dst_dim);
    } else if (dst_dim == merge_dim) {
      merged_tile_dims.push_back(dst_dim);
    } else {
      return false;
    }
  }
  const int64_t num_tiles = Product(merged_tile_dims);
  if (num_devices % num_tiles != 0 || num_tiles < minimum_tiles) {
    return false;
  }
  int64_t to_merge_man_dim = to_merge.SubgroupManualDim();
  int64_t dst_man_dim = dst->SubgroupManualDim();
  if (to_merge_man_dim >= 0) {
    if (dst_man_dim < 0) {
      return false;
    }
    int64_t man_group_size = to_merge.tile_assignment().dim(to_merge_man_dim);
    if (man_group_size != dst->tile_assignment().dim(dst_man_dim)) {
      return false;
    }
    merged_tile_dims.push_back(man_group_size);
  }
  int64_t replication = num_devices / Product(merged_tile_dims);
  merged_tile_dims.push_back(replication);
  Array<int64_t> merged_tile(merged_tile_dims);
  // Maps from replication group ID to sorted members.
  absl::flat_hash_map<int64_t, std::set<int64_t>> merge_group_members;
  absl::flat_hash_map<int64_t, std::set<int64_t>> dst_group_members;
  auto get_group_index = [&](absl::Span<const int64_t> tile_indices,
                             const HloSharding& sharding, int64_t manual_dim) {
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
  to_merge.tile_assignment().Each([&](absl::Span<const int64_t> indices,
                                      int64_t device) {
    merge_group_members[get_group_index(indices, to_merge, to_merge_man_dim)]
        .insert(device);
  });
  dst->tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t device) {
        dst_group_members[get_group_index(indices, *dst, dst_man_dim)].insert(
            device);
      });
  // Try to find the intersection of to_merge and dst replication groups, in
  // order to determine the merged tile assignment.
  Status compatible = merged_tile.EachStatus(
      [&](absl::Span<const int64_t> indices, int64_t* device) {
        std::vector<int64_t> to_merge_index(
            to_merge.tile_assignment().num_dimensions());
        std::vector<int64_t> dst_index(dst->tile_assignment().num_dimensions());
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
          to_merge_index[to_merge_man_dim] = indices[to_merge.TiledDataRank()];
          dst_index[dst_man_dim] = indices[to_merge.TiledDataRank()];
        }
        if (to_merge.HasPartialReplication()) {
          to_merge_index[to_merge.SubgroupReplicationDim()] = indices.back();
        }
        dst_index[dst->SubgroupReplicationDim()] = indices.back();
        int64_t to_merge_group_id =
            get_group_index(to_merge_index, to_merge, to_merge_man_dim);
        int64_t dst_group_id = get_group_index(dst_index, *dst, dst_man_dim);
        if (merge_group_members[to_merge_group_id].empty() ||
            dst_group_members[dst_group_id].empty()) {
          return InvalidArgument("Not compatible");
        }

        int64_t smallest_to_merge =
            *merge_group_members[to_merge_group_id].begin();
        int64_t smallest_dst = *dst_group_members[dst_group_id].begin();
        if (smallest_to_merge < smallest_dst) {
          if (merge_group_members[to_merge_group_id].count(smallest_dst) == 0) {
            return InvalidArgument("Not compatible");
          }
          *device = smallest_dst;
        } else {
          if (dst_group_members[dst_group_id].count(smallest_to_merge) == 0) {
            return InvalidArgument("Not compatible");
          }
          *device = smallest_to_merge;
        }
        merge_group_members[to_merge_group_id].erase(*device);
        dst_group_members[dst_group_id].erase(*device);
        return OkStatus();
      });
  if (!compatible.ok()) {
    return false;
  }
  std::vector<OpMetadata> merged_metadata(std::move(dst->metadata()));
  merged_metadata.reserve(merged_metadata.size() + to_merge.metadata().size());
  const absl::flat_hash_set<OpMetadata, protobuf_util::ProtobufHashWrapper,
                            protobuf_util::ProtobufEqualsWrapper>
      metadata_set(merged_metadata.begin(), merged_metadata.end());
  absl::c_copy_if(to_merge.metadata(), std::back_inserter(merged_metadata),
                  [&metadata_set](const OpMetadata& data) {
                    return !ContainsKey(metadata_set, data);
                  });
  std::vector<OpSharding::Type> subgroup_types;
  if (to_merge_man_dim >= 0) {
    subgroup_types.push_back(OpSharding::MANUAL);
  }
  subgroup_types.push_back(OpSharding::REPLICATED);
  *dst = HloSharding::Subgroup(merged_tile, subgroup_types, merged_metadata);
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

HloSharding TransposeSharding(const HloSharding& sharding,
                              absl::Span<const int64_t> dimensions) {
  if (sharding.IsTileMaximal() || sharding.IsManual()) {
    return sharding;
  }
  DimensionVector perm_dimensions(dimensions.begin(), dimensions.end());
  // Add subgroup dims if missing.
  if (sharding.TiledDataRank() == dimensions.size()) {
    for (int64_t i = sharding.TiledDataRank();
         i < sharding.tile_assignment().num_dimensions(); ++i) {
      perm_dimensions.push_back(i);
    }
  } else {
    CHECK_EQ(sharding.tile_assignment().num_dimensions(), dimensions.size());
  }
  Array<int64_t> tile_assignment = sharding.tile_assignment();
  tile_assignment.TransposeDimensions(perm_dimensions);
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
                                           const HloSharding& sharding) {
  if (sharding.IsTileMaximal() || sharding.IsManual()) {
    return sharding;
  }

  // In case of a tiled sharding the reshaped sharding will be a valid if the
  // reshape is composed from the following operations:
  // * Adding or removing dimensions with size 1.
  // * Merging consecutive dimensions where only the most major is sharded.
  // * Splitting a dimension to consecutive dimensions.
  // * Any reshaping of unsharded dimensions.
  // Note that merge and split can happen consecutively on the same dimension,
  // e.g., f32[1024,256,1024] to f32[128,2048,1024] can be considered that 1024
  // gets split into 128 and 8, but 8 then gets merged with 256. We use stacks
  // to make supporting such cases easy.
  const Shape tile_shape = sharding.TileShape(source_shape);
  std::vector<int64_t> target_tile_assignment_dimensions;
  std::vector<int64_t> source_dims_stack(source_shape.rank());
  std::vector<int64_t> target_dims_stack(target_shape.rank());
  std::vector<int64_t> sharding_tile_dims_stack(source_shape.rank());
  int64_t added_to_partially_replicated = 1;
  for (int64_t i = 0; i < source_shape.rank(); ++i) {
    source_dims_stack[i] = source_shape.dimensions(source_shape.rank() - 1 - i);
    sharding_tile_dims_stack[i] =
        sharding.tile_assignment().dim(source_shape.rank() - 1 - i);
  }
  for (int64_t i = 0; i < target_shape.rank(); ++i) {
    target_dims_stack[i] = target_shape.dimensions(target_shape.rank() - 1 - i);
  }
  while (!source_dims_stack.empty() || !target_dims_stack.empty()) {
    if (target_dims_stack.empty()) {
      if (Product(sharding_tile_dims_stack) != 1) {
        return std::nullopt;
      }
      break;
    }
    int64_t s_size = 1;
    int64_t t_size = 1;
    int64_t s_partitions = 1;
    if (!source_dims_stack.empty()) {
      s_size = source_dims_stack.back();
      source_dims_stack.pop_back();
      s_partitions = sharding_tile_dims_stack.back();
      sharding_tile_dims_stack.pop_back();
    }
    t_size = target_dims_stack.back();
    target_dims_stack.pop_back();
    if (s_partitions * Product(sharding_tile_dims_stack) == 1) {
      // No more partitions left.
      target_tile_assignment_dimensions.push_back(1);
      continue;
    }
    if (s_size == t_size) {
      // Same dimension.
      target_tile_assignment_dimensions.push_back(s_partitions);
    } else if (t_size == 1) {
      // Trivial dimension added.
      target_tile_assignment_dimensions.push_back(1);
      source_dims_stack.push_back(s_size);
      sharding_tile_dims_stack.push_back(s_partitions);
    } else if (s_size == 1) {
      // Trivial dimension removed.
      if (s_partitions != 1) {
        added_to_partially_replicated *= s_partitions;
      }
      target_dims_stack.push_back(t_size);
    } else if (s_size > t_size) {
      // Dimension split.
      if (s_size % t_size != 0 || s_size % s_partitions != 0) {
        return std::nullopt;
      }
      if (t_size % s_partitions == 0) {
        target_tile_assignment_dimensions.push_back(s_partitions);
        // We have part of the s_size unprocessed, so put it back to stack.
        source_dims_stack.push_back(s_size / t_size);
        sharding_tile_dims_stack.push_back(1);
      } else if (s_partitions % t_size == 0) {
        target_tile_assignment_dimensions.push_back(t_size);
        // We have part of the s_size unprocessed, so put it back to stack.
        source_dims_stack.push_back(s_size / t_size);
        sharding_tile_dims_stack.push_back(s_partitions / t_size);
      } else {
        return std::nullopt;
      }
    } else {
      // Dimension merge. Also merge the source dimension with the next, and
      // process it next time.
      if (s_size % s_partitions != 0) {
        return std::nullopt;
      }
      CHECK(!source_dims_stack.empty());
      if (sharding_tile_dims_stack.back() != 1 && s_size != s_partitions) {
        // If the next dimension to combine is sharded, we require that the
        // current dimension's shard size to be 1. Otherwise, the new shard
        // would be non-contiguous.
        return std::nullopt;
      }
      source_dims_stack.back() *= s_size;
      sharding_tile_dims_stack.back() *= s_partitions;
      target_dims_stack.push_back(t_size);
    }
  }
  if (Product(target_tile_assignment_dimensions) == 1) {
    return std::nullopt;
  }
  Array<int64_t> new_tile_assignment = sharding.tile_assignment();
  for (int64_t i = sharding.TiledDataRank();
       i < sharding.tile_assignment().num_dimensions(); ++i) {
    target_tile_assignment_dimensions.push_back(
        sharding.tile_assignment().dim(i));
  }

  auto subgroup_types = sharding.subgroup_types();
  // If we added dimensions to the partially replicated dimension then add the
  // additional dimension on the partially replicated tiling.
  if (added_to_partially_replicated > 1) {
    if (sharding.ReplicateOnLastTileDim()) {
      target_tile_assignment_dimensions.back() *= added_to_partially_replicated;
    } else {
      target_tile_assignment_dimensions.push_back(
          added_to_partially_replicated);
    }
  }
  // If subgroup_types doesn't have already partially replicated as a sharding
  // type then add it.
  if ((sharding.ReplicateOnLastTileDim() ||
       added_to_partially_replicated > 1) &&
      (subgroup_types.empty() ||
       subgroup_types.back() != OpSharding::REPLICATED)) {
    subgroup_types.push_back(OpSharding::REPLICATED);
  }
  new_tile_assignment.Reshape(target_tile_assignment_dimensions);
  return HloSharding::Subgroup(new_tile_assignment, subgroup_types,
                               sharding.metadata());
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

  std::vector<int64_t> tile_dims(sharding.tile_assignment().num_dimensions(),
                                 1);
  // Handle ignore dimensions.
  std::vector<int64_t> ignore_sizes;
  int64_t ignore_size = 1;
  for (int64_t i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (absl::c_find(dims, i) == dims.end()) {
      int64_t size = sharding.tile_assignment().dim(i);
      ignore_sizes.push_back(size);
      tile_dims[i] = size;
      ignore_size *= size;
    }
  }

  using Buckets = std::vector<std::vector<int64_t>>;
  Array<Buckets> buckets(ignore_sizes,
                         Buckets(sharding.tile_assignment().dim(dim)));
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> index, int64_t device) {
        std::vector<int64_t> ignore_index;
        for (int64_t i = 0; i < index.size(); ++i) {
          if (absl::c_find(dims, i) == dims.end()) {
            ignore_index.push_back(index[i]);
          }
        }
        buckets(ignore_index)[index[dim]].push_back(device);
      });
  std::vector<int64_t> devices;
  buckets.Each([&](absl::Span<const int64_t> index, const Buckets& buckets) {
    for (auto& bucket : buckets) {
      devices.insert(devices.end(), bucket.begin(), bucket.end());
    }
  });
  tile_dims[dim] = devices.size() / ignore_size;
  Array<int64_t> tile_assignment(tile_dims);
  tile_assignment.SetValues(devices);
  return HloSharding::Tile(tile_assignment, sharding.metadata());
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

HloSharding GatherOutputShardingFromIndexIndexPassthroughDimensions(
    const HloSharding& index_sharding, const HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kGather);
  if (index_sharding.IsTileMaximal() || index_sharding.IsManual()) {
    return index_sharding;
  }

  const GatherDimensionNumbers& dnums = hlo->gather_dimension_numbers();
  const absl::InlinedVector<int64_t, 1> index_passthrough_dims =
      GetGatherScatterIndexPassthroughIndexDims(hlo->operand(1)->shape().rank(),
                                                dnums.index_vector_dim());
  const absl::InlinedVector<int64_t, 1> output_passthrough_dims =
      GetGatherScatterIndexPassthroughOutputOrUpdateDims(hlo->shape().rank(),
                                                         dnums.offset_dims());
  CHECK_EQ(index_passthrough_dims.size(), output_passthrough_dims.size());
  std::vector<int64_t> output_tile(hlo->shape().rank(), 1);
  for (auto i = 0; i != index_passthrough_dims.size(); ++i) {
    output_tile[output_passthrough_dims[i]] =
        index_sharding.tile_assignment().dim(index_passthrough_dims[i]);
  }

  HloSharding relevant_index_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(index_sharding,
                                                     index_passthrough_dims);
  if (relevant_index_sharding.IsTileMaximal()) {
    return relevant_index_sharding;
  }
  for (int64_t i = relevant_index_sharding.TiledDataRank();
       i != relevant_index_sharding.tile_assignment().num_dimensions(); ++i) {
    output_tile.push_back(relevant_index_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> tile_assignment = relevant_index_sharding.tile_assignment();
  tile_assignment.Reshape(output_tile);
  return relevant_index_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment,
                                        index_sharding.metadata())
             : HloSharding::Subgroup(tile_assignment,
                                     relevant_index_sharding.subgroup_types(),
                                     index_sharding.metadata());
}

HloSharding GatherIndexShardingFromOutputIndexPassthroughDimensions(
    const HloSharding& output_sharding, const HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kGather);
  if (output_sharding.IsTileMaximal() || output_sharding.IsManual()) {
    return output_sharding;
  }

  const GatherDimensionNumbers& dnums = hlo->gather_dimension_numbers();
  const absl::InlinedVector<int64_t, 1> index_passthrough_dims =
      GetGatherScatterIndexPassthroughIndexDims(hlo->operand(1)->shape().rank(),
                                                dnums.index_vector_dim());
  const absl::InlinedVector<int64_t, 1> output_passthrough_dims =
      GetGatherScatterIndexPassthroughOutputOrUpdateDims(hlo->shape().rank(),
                                                         dnums.offset_dims());
  CHECK_EQ(index_passthrough_dims.size(), output_passthrough_dims.size());
  std::vector<int64_t> index_tile(hlo->operand(1)->shape().rank(), 1);
  for (auto i = 0; i != index_passthrough_dims.size(); ++i) {
    index_tile[index_passthrough_dims[i]] =
        output_sharding.tile_assignment().dim(output_passthrough_dims[i]);
  }

  HloSharding relevant_output_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(output_sharding,
                                                     output_passthrough_dims);
  if (relevant_output_sharding.IsTileMaximal()) {
    return relevant_output_sharding;
  }
  for (int64_t i = relevant_output_sharding.TiledDataRank();
       i != relevant_output_sharding.tile_assignment().num_dimensions(); ++i) {
    index_tile.push_back(relevant_output_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> tile_assignment = relevant_output_sharding.tile_assignment();
  tile_assignment.Reshape(index_tile);
  return relevant_output_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment,
                                        output_sharding.metadata())
             : HloSharding::Subgroup(tile_assignment,
                                     relevant_output_sharding.subgroup_types(),
                                     output_sharding.metadata());
}

HloSharding GatherEffectiveOutputSharding(const HloInstruction& hlo) {
  if (hlo.sharding().IsTileMaximal() || hlo.sharding().IsManual()) {
    return hlo.sharding();
  }

  const GatherDimensionNumbers& dnums = hlo.gather_dimension_numbers();
  std::vector<int64_t> tile_assignment_dims(hlo.shape().rank());
  int64_t num_elements = 1;
  for (int64_t i = 0; i < hlo.shape().rank(); ++i) {
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
    return HloSharding::AssignDevice(*hlo.sharding().tile_assignment().begin(),
                                     hlo.sharding().metadata());
  }

  // Output sharding is on both offset and non offset dimensions. We shard the
  // gather op only on non offset dimensions.
  // For example:
  // - the gather op has sharding [2,2]{0,1,2,3},
  // - first dimension is non offset dimension,
  // - second dimension is offset dimension,
  // Then the result sharding will be [2,1]{0,2}.
  std::vector<int64_t> slice_starts(hlo.shape().rank(), 0LL),
      slice_limits(hlo.shape().rank());
  for (int64_t i = 0; i < hlo.shape().rank(); ++i) {
    if (!absl::c_binary_search(dnums.offset_dims(), i)) {
      slice_limits[i] = hlo.sharding().tile_assignment().dim(i);
    } else {
      slice_limits[i] = 1;
    }
  }
  Array<int64_t> tile_assignment =
      hlo.sharding().tile_assignment().Slice(slice_starts, slice_limits);
  return HloSharding::Tile(tile_assignment, hlo.sharding().metadata());
}

HloSharding ScatterIndexShardingFromUpdateIndexPassthroughDimensions(
    const HloSharding& update_sharding, const HloScatterInstruction* scatter) {
  if (update_sharding.IsTileMaximal() || update_sharding.IsManual()) {
    return update_sharding;
  }

  const ScatterDimensionNumbers& dnums = scatter->scatter_dimension_numbers();
  const absl::InlinedVector<int64_t, 1> index_passthrough_dims =
      GetGatherScatterIndexPassthroughIndexDims(
          scatter->scatter_indices()->shape().rank(), dnums.index_vector_dim());
  const absl::InlinedVector<int64_t, 1> update_passthrough_dims =
      GetGatherScatterIndexPassthroughOutputOrUpdateDims(
          scatter->scatter_updates()[0]->shape().rank(),
          dnums.update_window_dims());
  CHECK_EQ(index_passthrough_dims.size(), update_passthrough_dims.size());
  std::vector<int64_t> index_tile(scatter->scatter_indices()->shape().rank(),
                                  1);
  for (auto i = 0; i != index_passthrough_dims.size(); ++i) {
    index_tile[index_passthrough_dims[i]] =
        update_sharding.tile_assignment().dim(update_passthrough_dims[i]);
  }

  HloSharding relevant_update_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(update_sharding,
                                                     update_passthrough_dims);
  if (relevant_update_sharding.IsTileMaximal()) {
    return relevant_update_sharding;
  }
  for (int64_t i = relevant_update_sharding.TiledDataRank();
       i != relevant_update_sharding.tile_assignment().num_dimensions(); ++i) {
    index_tile.push_back(relevant_update_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> tile_assignment = relevant_update_sharding.tile_assignment();
  tile_assignment.Reshape(index_tile);
  return relevant_update_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment,
                                        update_sharding.metadata())
             : HloSharding::Subgroup(tile_assignment,
                                     relevant_update_sharding.subgroup_types(),
                                     update_sharding.metadata());
}

HloSharding ScatterUpdateShardingFromIndexIndexPassthroughDimensions(
    const HloSharding& index_sharding, const HloScatterInstruction* scatter) {
  if (index_sharding.IsTileMaximal() || index_sharding.IsManual()) {
    return index_sharding;
  }

  const ScatterDimensionNumbers& dnums = scatter->scatter_dimension_numbers();
  const absl::InlinedVector<int64_t, 1> index_passthrough_dims =
      GetGatherScatterIndexPassthroughIndexDims(
          scatter->scatter_indices()->shape().rank(), dnums.index_vector_dim());
  const absl::InlinedVector<int64_t, 1> update_passthrough_dims =
      GetGatherScatterIndexPassthroughOutputOrUpdateDims(
          scatter->scatter_updates()[0]->shape().rank(),
          dnums.update_window_dims());
  CHECK_EQ(index_passthrough_dims.size(), update_passthrough_dims.size());
  std::vector<int64_t> update_tile(
      scatter->scatter_updates()[0]->shape().rank(), 1);
  for (auto i = 0; i != index_passthrough_dims.size(); ++i) {
    update_tile[update_passthrough_dims[i]] =
        index_sharding.tile_assignment().dim(index_passthrough_dims[i]);
  }

  HloSharding relevant_index_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(index_sharding,
                                                     index_passthrough_dims);
  if (relevant_index_sharding.IsTileMaximal()) {
    return relevant_index_sharding;
  }
  for (int64_t i = relevant_index_sharding.TiledDataRank();
       i != relevant_index_sharding.tile_assignment().num_dimensions(); ++i) {
    update_tile.push_back(relevant_index_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> tile_assignment = relevant_index_sharding.tile_assignment();
  tile_assignment.Reshape(update_tile);
  return relevant_index_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment,
                                        index_sharding.metadata())
             : HloSharding::Subgroup(tile_assignment,
                                     relevant_index_sharding.subgroup_types(),
                                     index_sharding.metadata());
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
  for (int64_t i = 0; i < scatter.shape().rank(); ++i) {
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
    return HloSharding::AssignDevice(*index_sharding.tile_assignment().begin(),
                                     index_sharding.metadata());
  }

  const int64_t index_rank = scatter.scatter_indices()->shape().rank();
  std::vector<int64_t> slice_starts(index_rank, 0LL), slice_limits(index_rank);
  for (int64_t i = 0; i < index_rank; ++i) {
    if (i < index_dim) {
      slice_limits[i] = index_sharding.tile_assignment().dim(i);
    } else {
      slice_limits[i] = 1;
    }
  }
  Array<int64_t> tile_assignment =
      index_sharding.tile_assignment().Slice(slice_starts, slice_limits);
  return HloSharding::Tile(tile_assignment, index_sharding.metadata());
}

HloSharding ScatterEffectiveDataSharding(const HloSharding& data_sharding,
                                         const HloScatterInstruction& scatter) {
  if (data_sharding.IsTileMaximal() || data_sharding.IsManual()) {
    return data_sharding;
  }

  const ScatterDimensionNumbers& dnums = scatter.scatter_dimension_numbers();
  const int64_t data_rank = scatter.scatter_updates()[0]->shape().rank();
  std::vector<int64_t> tile_assignment_dims(data_rank, 1LL);
  int64_t num_elements = 1;
  for (int64_t i = 0; i < scatter.shape().rank(); ++i) {
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
    return HloSharding::AssignDevice(*data_sharding.tile_assignment().begin(),
                                     data_sharding.metadata());
  }

  // Data sharding is on both update_window_dims and scatter_window_dims. We
  // shard the scatter op only on scatter_window_dims. For example:
  // - the scatter data has sharding [2,2]{0,1,2,3},
  // - first dimension is scatter_window_dims,
  // - second dimension is update_window_dims,
  // Then the result sharding will be [2,1]{0,2}.
  std::vector<int64_t> slice_starts(data_rank, 0LL);
  Array<int64_t> tile_assignment =
      data_sharding.tile_assignment().Slice(slice_starts, tile_assignment_dims);
  return HloSharding::Tile(tile_assignment, data_sharding.metadata());
}

namespace {

// Returns the operand pass-through dimensions for gather/scatter operand(s).
absl::InlinedVector<int64_t, 1> GetGatherScatterOperandPassthroughOperandDims(
    const Shape& operand_shape,
    absl::Span<const int64_t> collapsed_or_inserted_dims,
    absl::Span<const int64_t> index_map,
    absl::Span<const int64_t> offset_or_window_dims,
    absl::Span<const int64_t> slice_size) {
  absl::InlinedVector<int64_t, 1> passthrough_dims;
  int64_t collapsed = 0;
  for (int64_t i = 0; i != operand_shape.rank(); ++i) {
    if (absl::c_linear_search(collapsed_or_inserted_dims, i)) {
      collapsed++;
      continue;
    }
    if (slice_size[i] != operand_shape.dimensions(i)) {
      continue;
    }
    int64_t offset_dim = offset_or_window_dims[i - collapsed];
    if (i - collapsed > 0 &&
        offset_dim < offset_or_window_dims[i - collapsed - 1]) {
      // Output offsets are transposed, we do not support this case.
      continue;
    }
    passthrough_dims.push_back(i);
  }
  return passthrough_dims;
}

// If partitioning in the operand only happens in dimensions in passthrough
// dimensions (offset dimensions in the gather output (or scatter update) that
// have the same size as the operand), returns the corresponding output (or
// update) sharding by passing through the input sharding.
std::optional<HloSharding> PassthroughOperandToGatherOutputOrScatterUpdate(
    const Shape& operand_shape, const HloSharding& operand_sharding,
    const int64_t output_or_update_rank,
    absl::Span<const int64_t> collapsed_or_inserted_dims,
    absl::Span<const int64_t> index_map,
    absl::Span<const int64_t> offset_or_window_dims,
    absl::Span<const int64_t> slice_size, const int64_t index_vector_dim) {
  if (operand_sharding.IsTileMaximal() || operand_sharding.IsManual()) {
    return operand_sharding;
  }
  auto operand_passthrough_dims = GetGatherScatterOperandPassthroughOperandDims(
      operand_shape, collapsed_or_inserted_dims, index_map,
      offset_or_window_dims, slice_size);
  std::vector<int64_t> passthrough_tile(output_or_update_rank, 1);
  int64_t collapsed = 0;
  for (int64_t i = 0; i < operand_shape.rank(); ++i) {
    if (absl::c_linear_search(collapsed_or_inserted_dims, i)) {
      collapsed++;
    }
    if (!absl::c_linear_search(operand_passthrough_dims, i)) {
      continue;
    }
    int64_t offset_dim = offset_or_window_dims[i - collapsed];
    passthrough_tile[offset_dim] = operand_sharding.tile_assignment().dim(i);
  }
  HloSharding replicate_non_passthrough_dims =
      PartiallyReplicateTiledShardingOnAllDimsExcept(operand_sharding,
                                                     operand_passthrough_dims);
  if (replicate_non_passthrough_dims.IsTileMaximal()) {
    return std::nullopt;
  }
  for (int64_t i = replicate_non_passthrough_dims.TiledDataRank();
       i < replicate_non_passthrough_dims.tile_assignment().num_dimensions();
       ++i) {
    passthrough_tile.push_back(
        replicate_non_passthrough_dims.tile_assignment().dim(i));
  }
  auto tile_assignment = replicate_non_passthrough_dims.tile_assignment();
  tile_assignment.Reshape(passthrough_tile);
  return replicate_non_passthrough_dims.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(
                   tile_assignment, replicate_non_passthrough_dims.metadata())
             : HloSharding::Subgroup(
                   tile_assignment,
                   replicate_non_passthrough_dims.subgroup_types(),
                   replicate_non_passthrough_dims.metadata());
}

// Inverse of PassthroughOperandToGatherOutputOrScatterUpdate.
std::optional<HloSharding> PassthroughGatherOutputOrScatterUpdateToOperand(
    const Shape& operand_shape, const HloSharding& output_or_update_sharding,
    absl::Span<const int64_t> collapsed_or_inserted_dims,
    absl::Span<const int64_t> index_map,
    absl::Span<const int64_t> offset_or_window_dims,
    absl::Span<const int64_t> slice_size) {
  if (output_or_update_sharding.IsTileMaximal() ||
      output_or_update_sharding.IsManual()) {
    return output_or_update_sharding;
  }
  auto operand_passthrough_dims = GetGatherScatterOperandPassthroughOperandDims(
      operand_shape, collapsed_or_inserted_dims, index_map,
      offset_or_window_dims, slice_size);
  std::vector<int64_t> passthrough_tile(operand_shape.rank(), 1);
  int64_t collapsed = 0;
  // Relevant dims have shardings passed to the operand.
  std::vector<int64_t> relevant_output_or_update_dims;
  for (int64_t i = 0; i < operand_shape.rank(); ++i) {
    if (absl::c_linear_search(collapsed_or_inserted_dims, i)) {
      collapsed++;
    }
    if (!absl::c_linear_search(operand_passthrough_dims, i)) {
      continue;
    }
    int64_t offset_dim = offset_or_window_dims[i - collapsed];
    passthrough_tile[i] =
        output_or_update_sharding.tile_assignment().dim(offset_dim);
    relevant_output_or_update_dims.push_back(offset_dim);
  }

  HloSharding relevant_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(
          output_or_update_sharding, relevant_output_or_update_dims);
  if (relevant_sharding.IsTileMaximal()) {
    return std::nullopt;
  }
  for (int64_t i = relevant_sharding.TiledDataRank();
       i < relevant_sharding.tile_assignment().num_dimensions(); ++i) {
    passthrough_tile.push_back(relevant_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> tile_assignment = relevant_sharding.tile_assignment();
  tile_assignment.Reshape(passthrough_tile);
  return relevant_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment,
                                        output_or_update_sharding.metadata())
             : HloSharding::Subgroup(tile_assignment,
                                     relevant_sharding.subgroup_types(),
                                     output_or_update_sharding.metadata());
}

std::optional<HloSharding> GatherOperandShardingFromOutputParallelDimensions(
    const HloSharding& output_sharding, const HloInstruction& gather,
    const CallGraph& call_graph) {
  if (output_sharding.IsTileMaximal() || output_sharding.IsManual()) {
    return output_sharding;
  }
  auto parallel_dims = GetGatherParallelBatchDims(gather, call_graph);
  if (parallel_dims) {
    auto output_parallel_dims =
        GetGatherParallelOutputDims(gather, *parallel_dims);
    auto output_aligned_operand_parallel_dims =
        IndexAlignedOperandParallelDims(*parallel_dims);
    const Shape gather_shape = gather.shape();
    CHECK_EQ(output_parallel_dims.size(),
             output_aligned_operand_parallel_dims.size());
    std::vector<int64_t> operand_tile_assignment(
        gather.operand(0)->shape().rank(), 1);
    std::vector<int64_t> relevant_output_dims;
    for (int i = 0, parallel_idx = 0; i < gather_shape.rank(); ++i) {
      if (parallel_idx >= output_parallel_dims.size() ||
          output_parallel_dims[parallel_idx] != i) {
        continue;
      }
      const int64_t operand_dim =
          output_aligned_operand_parallel_dims[parallel_idx++];
      operand_tile_assignment[operand_dim] =
          output_sharding.tile_assignment().dim(i);
      relevant_output_dims.push_back(i);
    }
    HloSharding relevant_output_sharding =
        PartiallyReplicateTiledShardingOnAllDimsExcept(output_sharding,
                                                       relevant_output_dims);
    if (relevant_output_sharding.IsTileMaximal()) {
      return std::move(relevant_output_sharding);
    }

    for (int64_t i = relevant_output_sharding.TiledDataRank();
         i < relevant_output_sharding.tile_assignment().num_dimensions(); ++i) {
      operand_tile_assignment.push_back(
          relevant_output_sharding.tile_assignment().dim(i));
    }
    Array<int64_t> tile_assignment = relevant_output_sharding.tile_assignment();
    tile_assignment.Reshape(operand_tile_assignment);
    return relevant_output_sharding.ReplicateOnLastTileDim()
               ? HloSharding::PartialTile(tile_assignment,
                                          output_sharding.metadata())
               : HloSharding::Subgroup(
                     tile_assignment, relevant_output_sharding.subgroup_types(),
                     output_sharding.metadata());
  }
  return std::nullopt;
}

// Reorders `to_align` based on the order of how `target_permuted` is reordered
// from `target`, expecting the container size to be small.
absl::InlinedVector<int64_t, 1> AlignSmallContainers(
    absl::Span<const int64_t> to_align, absl::Span<const int64_t> target,
    absl::Span<const int64_t> target_permuted) {
  CHECK(absl::c_is_permutation(target_permuted, target));
  CHECK_EQ(to_align.size(), target.size());
  absl::InlinedVector<int64_t, 1> to_align_permuted(to_align.size());
  for (auto i = 0; i < target.size(); ++i) {
    // This is small so just look linearly.
    for (auto j = 0; j < target_permuted.size(); ++j) {
      if (target_permuted[j] == target[i]) {
        to_align_permuted[j] = to_align[i];
        break;
      }
    }
  }
  return to_align_permuted;
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
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  return PassthroughOperandToGatherOutputOrScatterUpdate(
      operand_shape, operand_sharding, hlo.shape().rank(), collapsed_slice_dims,
      start_index_map, offset_dims, slice_sizes, dnums.index_vector_dim());
}

std::optional<HloSharding> GatherOperandShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo,
    const CallGraph& call_graph) {
  const auto& dnums = hlo.gather_dimension_numbers();
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  // Prioritize parallel sharding first as this is how it is in
  // spmd_partitioner.
  std::optional<HloSharding> parallel_sharding =
      GatherOperandShardingFromOutputParallelDimensions(output_sharding, hlo,
                                                        call_graph);
  std::optional<HloSharding> passthrough_sharding =
      PassthroughGatherOutputOrScatterUpdateToOperand(
          hlo.operand(0)->shape(), output_sharding, collapsed_slice_dims,
          start_index_map, offset_dims, hlo.gather_slice_sizes());
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
  std::vector<int64_t> slice_size(operand_shape.rank(), 1);
  int64_t num_update_window_dims = 0;
  for (int64_t i = 0; i < operand_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.inserted_window_dims(), i)) {
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
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  std::vector<int64_t> slice_size =
      GetScatterSliceSize(scatter.scatter_operands()[0]->shape(),
                          scatter.scatter_updates()[0]->shape(), dnums);
  return PassthroughGatherOutputOrScatterUpdateToOperand(
      scatter.scatter_operands()[0]->shape(), update_sharding,
      inserted_window_dims, scatter_dims_to_operand_dims, update_window_dims,
      slice_size);
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
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  return PassthroughOperandToGatherOutputOrScatterUpdate(
      output_shape, output_sharding,
      scatter->scatter_updates()[0]->shape().rank(), inserted_window_dims,
      scatter_dims_to_operand_dims, update_window_dims, slice_sizes,
      dnums.index_vector_dim());
}

std::optional<HloSharding> ScatterUpdateShardingFromOutputParallelDimensions(
    const HloSharding& output_sharding, const HloScatterInstruction& scatter,
    const CallGraph& call_graph) {
  if (output_sharding.IsTileMaximal() || output_sharding.IsManual()) {
    return output_sharding;
  }
  auto parallel_dims = GetScatterParallelBatchDims(scatter, call_graph);
  if (parallel_dims) {
    auto update_parallel_dims =
        GetScatterParallelUpdateDims(scatter, *parallel_dims);
    auto index_aligned_operand_parallel_dims =
        IndexAlignedOperandParallelDims(*parallel_dims);
    auto operand_parallel_dims_sorted = index_aligned_operand_parallel_dims;
    absl::c_sort(operand_parallel_dims_sorted);
    auto operand_aligned_update_parallel_dims = AlignSmallContainers(
        update_parallel_dims, index_aligned_operand_parallel_dims,
        operand_parallel_dims_sorted);
    const Shape scatter_shape = scatter.shape();
    CHECK_EQ(update_parallel_dims.size(),
             index_aligned_operand_parallel_dims.size());
    std::vector<int64_t> update_tile_assignment(
        scatter.scatter_updates()[0]->shape().rank(), 1);
    std::vector<int64_t> relevant_output_dims;
    for (int i = 0, parallel_idx = 0; i < scatter_shape.rank(); ++i) {
      if (parallel_idx >= operand_parallel_dims_sorted.size() ||
          operand_parallel_dims_sorted[parallel_idx] != i) {
        continue;
      }
      const int64_t update_dim =
          operand_aligned_update_parallel_dims[parallel_idx++];
      update_tile_assignment[update_dim] =
          output_sharding.tile_assignment().dim(i);
      relevant_output_dims.push_back(i);
    }
    HloSharding relevant_output_sharding =
        PartiallyReplicateTiledShardingOnAllDimsExcept(output_sharding,
                                                       relevant_output_dims);
    if (relevant_output_sharding.IsTileMaximal()) {
      return std::move(relevant_output_sharding);
    }

    for (int64_t i = relevant_output_sharding.TiledDataRank();
         i < relevant_output_sharding.tile_assignment().num_dimensions(); ++i) {
      update_tile_assignment.push_back(
          relevant_output_sharding.tile_assignment().dim(i));
    }
    Array<int64_t> tile_assignment = relevant_output_sharding.tile_assignment();
    tile_assignment.Reshape(update_tile_assignment);
    return relevant_output_sharding.ReplicateOnLastTileDim()
               ? HloSharding::PartialTile(tile_assignment,
                                          output_sharding.metadata())
               : HloSharding::Subgroup(
                     tile_assignment, relevant_output_sharding.subgroup_types(),
                     output_sharding.metadata());
  }
  return std::nullopt;
}

HloSharding GatherOutputOrScatterUpdateShardingFromIndicesParallelDimensions(
    const HloSharding& indices_sharding,
    const int64_t output_or_update_shape_rank,
    absl::Span<const int64_t> indices_parallel_dims,
    absl::Span<const int64_t> output_or_update_parallel_dims) {
  if (indices_sharding.IsTileMaximal() || indices_sharding.IsManual()) {
    return indices_sharding;
  }
  CHECK_EQ(output_or_update_parallel_dims.size(), indices_parallel_dims.size());
  absl::InlinedVector<int64_t, 4> output_or_update_tiling(
      output_or_update_shape_rank, 1);
  absl::InlinedVector<int64_t, 4> relevant_indices_dims;
  // Pass through indices' sharding on index parallel dimensions.
  for (int i = 0; i != output_or_update_parallel_dims.size(); ++i) {
    const int output_or_update_idx = output_or_update_parallel_dims[i];
    CHECK_LT(output_or_update_idx, output_or_update_shape_rank);
    const int indices_idx = indices_parallel_dims[i];
    output_or_update_tiling[output_or_update_idx] =
        indices_sharding.tile_assignment().dim(indices_idx);
    relevant_indices_dims.push_back(indices_idx);
  }

  HloSharding relevant_indices_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(indices_sharding,
                                                     relevant_indices_dims);
  if (relevant_indices_sharding.IsTileMaximal()) {
    return relevant_indices_sharding;
  }

  // Append subgroup dimensions.
  for (int64_t i = relevant_indices_sharding.TiledDataRank();
       i != relevant_indices_sharding.tile_assignment().num_dimensions(); ++i) {
    output_or_update_tiling.push_back(
        relevant_indices_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> output_tile_assignment =
      relevant_indices_sharding.tile_assignment();
  output_tile_assignment.Reshape(output_or_update_tiling);
  return relevant_indices_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(output_tile_assignment,
                                        indices_sharding.metadata())
             : HloSharding::Subgroup(output_tile_assignment,
                                     relevant_indices_sharding.subgroup_types(),
                                     indices_sharding.metadata());
}

StatusOr<std::pair<std::unique_ptr<HloInstruction>, HloOpcode>>
IdentityValueAndHloOpcodeForScatterReduceComputation(
    const HloScatterInstruction& scatter) {
  auto computation = scatter.to_apply();
  // We only handle computations with 2 parameters and only 1 calculation.
  if (computation->instruction_count() != 3) {
    return Status(
        tsl::error::Code::INVALID_ARGUMENT,
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

  return Status(tsl::error::Code::INVALID_ARGUMENT,
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
      sharding.tile_assignment().begin(), sharding.tile_assignment().end(),
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
  std::vector<int64_t> valid_dims_to_replicate;
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
  std::vector<int64_t> dim_permutation(sharding.TiledDataRank());
  absl::c_iota(dim_permutation, 0);
  absl::c_stable_sort(dim_permutation, [&](const int64_t a, const int64_t b) {
    return absl::c_linear_search(valid_dims_to_replicate, a) <
           absl::c_linear_search(valid_dims_to_replicate, b);
  });
  auto new_tile =
      TransposeSharding(sharding, dim_permutation).tile_assignment();
  std::vector<int64_t> new_tile_shape(
      sharding.tile_assignment().dimensions().begin(),
      sharding.tile_assignment().dimensions().end());
  for (int64_t dim : valid_dims_to_replicate) {
    new_tile_shape[dim] = 1;
  }
  if (sharding.ReplicateOnLastTileDim()) {
    new_tile_shape.back() *= group_count;
    new_tile.Reshape(new_tile_shape);
    return HloSharding::PartialTile(new_tile, sharding.metadata());
  } else {
    new_tile_shape.insert(new_tile_shape.begin() + sharding.TiledDataRank(),
                          group_count);
    new_tile.Reshape(new_tile_shape);
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
  std::vector<int64_t> dims_to_replicate(sharding.TiledDataRank());
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
    std::vector<int64_t> new_tile_shape(data_rank, 1);
    for (int64_t i = result.TiledDataRank();
         i < result.tile_assignment().num_dimensions(); ++i) {
      new_tile_shape.push_back(result.tile_assignment().dim(i));
    }
    auto tile = result.tile_assignment();
    tile.Reshape(new_tile_shape);
    result = HloSharding::Subgroup(tile, result.subgroup_types());
  }
  return result;
}

HloSharding RemoveShapeDimensions(const HloSharding& sharding,
                                  absl::Span<const int64_t> dims_to_remove) {
  if (sharding.IsTileMaximal() || dims_to_remove.empty()) {
    return sharding;
  }
  std::vector<int64_t> new_tile_shape;
  new_tile_shape.reserve(sharding.tile_assignment().num_dimensions() -
                         dims_to_remove.size());
  for (int64_t i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (absl::c_linear_search(dims_to_remove, i)) {
      CHECK_EQ(sharding.tile_assignment().dim(i), 1);
    } else {
      new_tile_shape.push_back(sharding.tile_assignment().dim(i));
    }
  }
  auto new_tile = sharding.tile_assignment();
  new_tile.Reshape(new_tile_shape);
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
    std::vector<int64_t> new_src_to_tgt(src_to_tgt.begin(), src_to_tgt.end());
    std::vector<int64_t> new_tgt_to_src(tgt_to_src.begin(), tgt_to_src.end());
    for (int64_t i = 0;
         i < source.tile_assignment().num_dimensions() - src_to_tgt.size();
         ++i) {
      new_src_to_tgt.push_back(tgt_to_src.size() + i);
      new_tgt_to_src.push_back(src_to_tgt.size() + i);
    }
    return TransposeShardingWithCollapsedDims(source, new_src_to_tgt,
                                              new_tgt_to_src);
  }
  std::vector<int64_t> tgt_dims_skipping_new(tgt_to_src.size(), -1);
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
  std::vector<int64_t> perm(src_to_tgt.size());
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
  auto tgt_sharding = hlo_sharding_util::TransposeSharding(source, perm);
  auto reshape_tiles = tgt_sharding.tile_assignment();
  std::vector<int64_t> tgt_tiles(tgt_to_src.size(), 1);
  for (int64_t i = 0; i < tgt_tiles.size(); ++i) {
    if (tgt_to_src[i] >= 0) {
      int64_t dim = tgt_dims_skipping_new[i];
      if (i >= tgt_non_subgroup_dims) {
        dim += skipped_src_dims;
      }
      tgt_tiles[i] = reshape_tiles.dim(dim);
    }
  }
  reshape_tiles.Reshape(tgt_tiles);
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
    std::vector<bool> is_iota_dim(maybe_iota->shape().rank(), true);
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
    const HloInstruction* gte = maybe_iota;
    const int64_t gte_index = gte->tuple_index();
    if (!called_computation->IsEntryComputation()) {
      // Support tracing only caller that's either a conditional or while
      // (other types of non-entry computations are not partitioned).
      std::vector<HloInstruction*> callers =
          call_graph.GetComputationCallers(called_computation);
      HloInstruction* caller =
          call_graph.GetComputationCallers(called_computation)[0];
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
        return GetDimensionForIota(caller->operand(0)->operand(gte_index),
                                   call_graph);
      }
    }
    return std::nullopt;
  }

  return std::nullopt;
}

static std::optional<GatherScatterParallelDims>
GetGatherScatterBatchParallelDims(const HloInstruction* indices,
                                  absl::Span<const int64_t> slice_sizes,
                                  int64_t index_vector_dim,
                                  absl::Span<const int64_t> index_map,
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
  absl::InlinedVector<const HloIotaInstruction*, 4> iotas;
  const int num_indices = index_map.size();
  std::vector<int64_t> index_parallel_in_dim(num_indices, -1);
  // Handle cases where we concatenate pieces of the indices one at a time.
  if (indices->opcode() == HloOpcode::kConcatenate &&
      indices->concatenate_dimension() == index_vector_dim) {
    int concatenated_dims = 0;
    for (int i = 0; i < indices->operand_count(); ++i) {
      const HloInstruction* op = indices->operand(i);
      const int64_t num_indices_from_element =
          op->shape().dimensions_size() > index_vector_dim
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
          indices->shape().dimensions_size() > index_vector_dim
              ? indices->shape().dimensions(index_vector_dim)
              : 1;
      index_parallel_in_dim.assign(num_indices_from_element, *maybe_iota_dim);
    }
  }
  absl::InlinedVector<int64_t, 1> indices_parallel_dims;
  absl::InlinedVector<int64_t, 1> operand_parallel_dims;
  // Map the parallelizable dimension from the iota to the dimensions of the
  // output and the operand. These dimensions are interconnected, but between
  // operands and index they could have different spots in the shape because the
  // position of the index dimension in the operand is determined by index_map.
  for (int i = 0; i < index_parallel_in_dim.size(); ++i) {
    int index_parallel_dim = index_parallel_in_dim[i];
    if (index_parallel_dim == -1) {
      continue;
    }
    if (absl::c_linear_search(indices_parallel_dims, index_parallel_dim)) {
      return std::nullopt;
    }
    // Considered parallel only if the slice is of size 1 over the operand.
    if (slice_sizes[index_map[i]] == 1) {
      indices_parallel_dims.push_back(index_parallel_dim);
      operand_parallel_dims.push_back(index_map[i]);
    } else {
      index_parallel_in_dim[i] = -1;
    }
  }
  absl::c_sort(indices_parallel_dims);
  if (!indices_parallel_dims.empty()) {
    return GatherScatterParallelDims{
        indices_parallel_dims, operand_parallel_dims, index_parallel_in_dim};
  }
  return std::nullopt;
}

std::optional<GatherScatterParallelDims> GetGatherParallelBatchDims(
    const HloInstruction& hlo, const CallGraph& call_graph) {
  CHECK(DynCast<HloGatherInstruction>(&hlo));
  const HloInstruction* indices = hlo.operand(1);
  absl::Span<const int64_t> slice_sizes = hlo.gather_slice_sizes();
  const auto& dnums = hlo.gather_dimension_numbers();
  int64_t index_vector_dim = dnums.index_vector_dim();
  const auto& index_map = dnums.start_index_map();
  return GetGatherScatterBatchParallelDims(
      indices, slice_sizes, index_vector_dim, index_map, call_graph);
}

std::optional<GatherScatterParallelDims> GetScatterParallelBatchDims(
    const HloInstruction& hlo, const CallGraph& call_graph) {
  const HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(&hlo);
  CHECK(scatter);
  const HloInstruction* indices = scatter->scatter_indices();
  const auto& dnums = hlo.scatter_dimension_numbers();
  std::vector<int64_t> slice_sizes =
      GetScatterSliceSize(scatter->scatter_operands()[0]->shape(),
                          scatter->scatter_updates()[0]->shape(), dnums);
  int64_t index_vector_dim = dnums.index_vector_dim();
  const auto& index_map = dnums.scatter_dims_to_operand_dims();
  return GetGatherScatterBatchParallelDims(
      indices, slice_sizes, index_vector_dim, index_map, call_graph);
}

static absl::InlinedVector<int64_t, 1>
GetGatherOutputOrScatterUpdateParallelDims(
    const Shape& shape, const GatherScatterParallelDims& parallel_dim,
    int64_t index_vector_dim, absl::Span<const int64_t> offset_or_window_dims) {
  absl::InlinedVector<int64_t, 1> output_parallel_dims;
  auto indices_parallel_dims = parallel_dim.indices_parallel_dims;
  for (int i = 0, idx_dim = 0; i < shape.dimensions_size(); ++i) {
    if (absl::c_linear_search(offset_or_window_dims, i)) {
      continue;
    }
    const int index_dim = idx_dim < index_vector_dim ? idx_dim : idx_dim + 1;
    if (absl::c_binary_search(indices_parallel_dims, index_dim)) {
      output_parallel_dims.push_back(i);
    }
    ++idx_dim;
  }
  return output_parallel_dims;
}

absl::InlinedVector<int64_t, 1> GetGatherParallelOutputDims(
    const HloInstruction& hlo, const GatherScatterParallelDims& parallel_dim) {
  CHECK(DynCast<HloGatherInstruction>(&hlo));
  const Shape& output_shape = hlo.shape();
  const auto& dnums = hlo.gather_dimension_numbers();
  int64_t index_vector_dim = dnums.index_vector_dim();
  const auto& offset_dims = dnums.offset_dims();
  return GetGatherOutputOrScatterUpdateParallelDims(
      output_shape, parallel_dim, index_vector_dim, offset_dims);
}

absl::InlinedVector<int64_t, 1> GetScatterParallelUpdateDims(
    const HloInstruction& hlo, const GatherScatterParallelDims& parallel_dim) {
  const HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(&hlo);
  CHECK(scatter);
  const Shape update_shape = scatter->scatter_updates()[0]->shape();
  const auto& dnums = hlo.scatter_dimension_numbers();
  int64_t index_vector_dim = dnums.index_vector_dim();
  const auto& window_dims = dnums.update_window_dims();
  return GetGatherOutputOrScatterUpdateParallelDims(
      update_shape, parallel_dim, index_vector_dim, window_dims);
}

absl::InlinedVector<int64_t, 1> GetGatherOperandPassthroughOperandDims(
    const Shape& operand_shape, const HloSharding& operand_sharding,
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes) {
  const auto& dnums = hlo.gather_dimension_numbers();
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  return GetGatherScatterOperandPassthroughOperandDims(
      operand_shape, collapsed_slice_dims, start_index_map, offset_dims,
      slice_sizes);
}

absl::InlinedVector<int64_t, 1> GetScatterOperandPassthroughOperandDims(
    const Shape& operand_shape, const HloSharding& operand_sharding,
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes) {
  const auto& dnums = hlo.scatter_dimension_numbers();
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  return GetGatherScatterOperandPassthroughOperandDims(
      operand_shape, inserted_window_dims, scatter_dims_to_operand_dims,
      update_window_dims, slice_sizes);
}

absl::InlinedVector<int64_t, 1> GetGatherScatterIndexPassthroughIndexDims(
    const int64_t indices_rank, const int64_t index_vector_dim) {
  absl::InlinedVector<int64_t, 1> passthrough_dims;
  for (int64_t i = 0; i != indices_rank; ++i) {
    if (i == index_vector_dim) {
      continue;
    }
    passthrough_dims.push_back(i);
  }
  return passthrough_dims;
}

absl::InlinedVector<int64_t, 1>
GetGatherScatterIndexPassthroughOutputOrUpdateDims(
    const int64_t output_or_update_rank,
    absl::Span<const int64_t> offset_or_window_dims) {
  absl::InlinedVector<int64_t, 1> passthrough_dims;
  for (int64_t i = 0; i != output_or_update_rank; ++i) {
    if (!absl::c_linear_search(offset_or_window_dims, i)) {
      passthrough_dims.push_back(i);
    }
  }
  return passthrough_dims;
}

absl::InlinedVector<int64_t, 1> IndexAlignedOperandParallelDims(
    const GatherScatterParallelDims& parallel_dims) {
  CHECK_EQ(parallel_dims.indices_parallel_dims.size(),
           parallel_dims.operand_parallel_dims.size());
  std::vector<int64_t> index_parallel_in_dim =
      parallel_dims.index_parallel_in_dim;
  // Remove all -1s in `index_parallel_in_dim`.
  index_parallel_in_dim.erase(std::remove(index_parallel_in_dim.begin(),
                                          index_parallel_in_dim.end(), -1),
                              index_parallel_in_dim.end());
  // Populate the operand parallel dimensions based on the order of the index
  // batch dims (which is the same order as the output).
  return AlignSmallContainers(parallel_dims.operand_parallel_dims,
                              index_parallel_in_dim,
                              parallel_dims.indices_parallel_dims);
}

std::string GroupedSharding::ToString() const {
  auto result = absl::StrCat("dims: ", absl::StrJoin(group_dims, ","),
                             "\ndevice_groups:\n");
  absl::StrAppend(&result,
                  "group dim sizes: ", absl::StrJoin(group_dim_sizes, ","));
  absl::StrAppend(&result, "data rank: ", data_rank);
  absl::StrAppend(&result, "subgroup manual: ", subgroup_manual);
  for (auto& device_group : device_groups) {
    absl::StrAppend(&result, "\t", absl::StrJoin(device_group, ","), "\n");
  }
  absl::StrAppend(&result, "inner sharding: ", sharding.ToString());
  return result;
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
  std::vector<int64_t> grouped_tiling_dims =
      sharding.tile_assignment().dimensions();
  std::vector<int64_t> group_dim_sizes(group_dims.size());
  for (int64_t i = 0; i < group_dims.size(); ++i) {
    CHECK_EQ(grouped_tiling_dims[group_dims[i]] % group_dim_shards[i], 0);
    group_dim_sizes[i] =
        grouped_tiling_dims[group_dims[i]] / group_dim_shards[i];
    grouped_tiling_dims[group_dims[i]] = group_dim_shards[i];
  }

  std::vector<std::vector<int64_t>> device_groups(Product(group_dim_sizes));
  sharding.tile_assignment().Each([&](absl::Span<const int64_t> indices,
                                      int64_t device) {
    int64_t group_id = 0;
    for (int64_t i = 0; i < group_dims.size(); ++i) {
      group_id *=
          sharding.tile_assignment().dim(group_dims[i]) / group_dim_shards[i];
      group_id += indices[group_dims[i]] / group_dim_shards[i];
    }
    device_groups[group_id].push_back(device);
  });
  auto grouped = GroupedSharding(
      std::move(device_groups),
      std::vector<int64_t>(group_dims.begin(), group_dims.end()),
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
  Array<int64_t> grouped_tiling(grouped_tiling_dims);
  grouped_tiling.FillIota(0);
  grouped.sharding =
      sharding.ReplicateOnLastTileDim() &&
              grouped_tiling_dims.size() ==
                  sharding.tile_assignment().num_dimensions()
          ? HloSharding::PartialTile(grouped_tiling, sharding.metadata())
          : HloSharding::Tile(grouped_tiling, sharding.metadata());
  return grouped;
}

// See if we can group sharding on partially replicated dimensions, otherwise
// replicate it.
GroupedSharding GroupShardingOnReplicatedDim(const HloSharding& sharding,
                                             const int64_t num_groups,
                                             const int64_t num_tiles,
                                             const int64_t data_rank) {
  if (sharding.ReplicateOnLastTileDim() &&
      sharding.tile_assignment().dimensions().back() % num_groups == 0) {
    absl::InlinedVector<int64_t, 1> group_dim_shards = {
        sharding.tile_assignment().dimensions().back() / num_groups};
    return GroupShardingOnDims(
        sharding, {sharding.tile_assignment().num_dimensions() - 1},
        group_dim_shards);
  }
  // Otherwise return a grouped replicated sharding.
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
  return GroupedSharding(std::move(device_groups), {data_rank}, {group_size},
                         data_rank, HloSharding::Replicate(),
                         /*subgroup_manual=*/false);
}

GroupedSharding GetManualSubgroupSharding(const HloSharding& sharding) {
  CHECK(sharding.IsManualSubgroup());
  int64_t tile_dimensions = sharding.tile_assignment().num_dimensions();
  int64_t subgroup_size = sharding.subgroup_types().size();
  int64_t rank = tile_dimensions - subgroup_size;
  std::vector<int64_t> group_dims;
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
  std::vector<std::vector<int64_t>> device_to_index(
      Product(sharding.tile_assignment().dimensions()),
      std::vector<int64_t>(sharding.tile_assignment().num_dimensions()));
  sharding.tile_assignment().Each(
      [&device_to_index](absl::Span<const int64_t> indices, int64_t device) {
        device_to_index[device].assign(indices.begin(), indices.end());
      });
  std::vector<int64_t> grouped_tiling_dims =
      sharding.tile_assignment().dimensions();
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
  std::vector<int64_t> tiling_dims;
  bool partial_sharding = false;
  std::vector<OpSharding::Type> subgroup_types;
  Array<int64_t> grouped_tiling = grouped_sharding.sharding.tile_assignment();
  if (grouped_sharding.sharding.IsTileMaximal()) {
    tiling_dims = std::vector<int64_t>(grouped_sharding.data_rank, 1);
    if (grouped_sharding.device_groups[0].size() != 1 ||
        absl::c_linear_search(grouped_sharding.group_dims,
                              tiling_dims.size())) {
      // This is partial sharding.
      tiling_dims.push_back(grouped_sharding.device_groups[0].size());
      partial_sharding = true;
    }
    grouped_tiling = Array<int64_t>(tiling_dims);
    grouped_tiling.FillIota(0);
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
      tiling_dims = grouped_sharding.sharding.tile_assignment().dimensions();
    }
    for (int i = 0; i < grouped_sharding.group_dims.size(); i++) {
      subgroup_types[grouped_sharding.group_dims[i] -
                     grouped_sharding.data_rank] = OpSharding::MANUAL;
      tiling_dims.insert(tiling_dims.begin() + grouped_sharding.group_dims[i],
                         1);
    }
    grouped_tiling.Reshape(tiling_dims);
  } else if (!grouped_sharding.sharding.IsTileMaximal()) {
    // Handles tile replicated.
    partial_sharding = grouped_sharding.sharding.ReplicateOnLastTileDim();
    tiling_dims = grouped_sharding.sharding.tile_assignment().dimensions();
    if (absl::c_linear_search(grouped_sharding.group_dims,
                              tiling_dims.size())) {
      tiling_dims.push_back(1);
      grouped_tiling.Reshape(tiling_dims);
      partial_sharding = true;
    }
  }

  // Update group dim sizes.
  for (int64_t i = 0; i < grouped_sharding.group_dims.size(); ++i) {
    int64_t dim = grouped_sharding.group_dims[i];
    tiling_dims[dim] *= grouped_sharding.group_dim_sizes[i];
  }
  Array<int64_t> tiling(tiling_dims);
  grouped_tiling.Each([&](absl::Span<const int64_t> indices, int64_t device) {
    std::vector<int64_t> ungrouped_inds(indices.begin(), indices.end());
    for (int64_t g = 0; g < grouped_sharding.device_groups.size(); ++g) {
      int64_t remaining_group_index = g;
      for (int64_t i = grouped_sharding.group_dims.size() - 1; i >= 0; --i) {
        int64_t dim = grouped_sharding.group_dims[i];
        int64_t groups_in_this_dim = grouped_sharding.group_dim_sizes[i];
        ungrouped_inds[dim] = (remaining_group_index % groups_in_this_dim) *
                                  grouped_tiling.dim(dim) +
                              indices[dim];
        remaining_group_index /= groups_in_this_dim;
      }
      tiling(ungrouped_inds) = grouped_sharding.device_groups[g][device];
    }
  });

  if (grouped_sharding.subgroup_manual) {
    return HloSharding::Subgroup(tiling, subgroup_types,
                                 grouped_sharding.sharding.metadata());
  }
  return partial_sharding ? HloSharding::PartialTile(tiling)
                          : HloSharding::Tile(tiling);
}

bool DeviceGroupsAreMatch(GroupedSharding& lhs, GroupedSharding& rhs,
                          bool ignore_group_order) {
  if (lhs.device_groups.size() != rhs.device_groups.size()) {
    return false;
  }

  bool matching_groups = true;
  absl::flat_hash_map<int64_t, int64_t> device_to_ref_group;
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
  auto new_tile_assignment = sharding.tile_assignment();
  std::vector<int64_t> dimensions = new_tile_assignment.dimensions();
  int64_t current_dimension = dimensions[dimension];
  dimensions.insert(dimensions.begin() + dimension + 1,
                    current_dimension / new_dim_size);
  dimensions[dimension] = new_dim_size;
  new_tile_assignment.Reshape(dimensions);
  auto new_sharding = sharding.ReplicateOnLastTileDim()
                          ? HloSharding::PartialTile(new_tile_assignment)
                          : HloSharding::Subgroup(new_tile_assignment,
                                                  sharding.subgroup_types());
  std::vector<int64_t> permutation(new_sharding.tile_assignment().dimensions());
  absl::c_iota(permutation, 0);
  std::swap(permutation[dimension], permutation[dimension + 1]);
  return TransposeSharding(new_sharding, permutation);
}

HloSharding MergeShardingDimension(const HloSharding& sharding,
                                   int64_t dimension) {
  CHECK_GT(sharding.TiledDataRank(), dimension);
  std::vector<int64_t> permutation(sharding.tile_assignment().dimensions());
  absl::c_iota(permutation, 0);
  std::swap(permutation[dimension], permutation[dimension + 1]);
  auto transposed_sharding = TransposeSharding(sharding, permutation);
  auto new_tile_assignment = transposed_sharding.tile_assignment();
  std::vector<int64_t> dimensions = new_tile_assignment.dimensions();
  dimensions[dimension] *= dimensions[dimension + 1];
  dimensions.erase(dimensions.begin() + dimension + 1);
  new_tile_assignment.Reshape(dimensions);
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

bool IsSortOperandShardingMovable(const HloInstruction* sort_operand,
                                  int64_t sort_dim) {
  // Some early exit cases.
  if (sort_operand == nullptr || sort_operand->shape().rank() < 2 ||
      !sort_operand->has_sharding()) {
    return false;
  }
  const auto& sharding = sort_operand->sharding();
  if (!sharding.IsTiled() || sharding.IsTileMaximal() ||
      sharding.tile_assignment().dim(sort_dim) == 1) {
    return false;
  }
  // Test whether there exist a free dimension to move the sharding into
  auto tile_assignment_dims = sharding.tile_assignment().dimensions();
  const int rank = sort_operand->shape().rank();
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (dim == sort_dim || tile_assignment_dims[dim] != 1) {
      continue;
    }
    return true;
  }
  return false;
}
}  // namespace hlo_sharding_util
}  // namespace xla
