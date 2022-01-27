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
#include <map>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace hlo_sharding_util {

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
  if (num_devices % Product(merged_tile_dims) != 0 ||
      Product(merged_tile_dims) < minimum_tiles) {
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
  bool compatible = true;
  merged_tile.Each([&](absl::Span<const int64_t> indices, int64_t* device) {
    if (!compatible) {
      return;
    }
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
      compatible = false;
      return;
    }

    int64_t smallest_to_merge = *merge_group_members[to_merge_group_id].begin();
    int64_t smallest_dst = *dst_group_members[dst_group_id].begin();
    if (smallest_to_merge < smallest_dst) {
      if (merge_group_members[to_merge_group_id].count(smallest_dst) == 0) {
        compatible = false;
        return;
      }
      *device = smallest_dst;
    } else {
      if (dst_group_members[dst_group_id].count(smallest_to_merge) == 0) {
        compatible = false;
        return;
      }
      *device = smallest_to_merge;
    }
    merge_group_members[to_merge_group_id].erase(*device);
    dst_group_members[dst_group_id].erase(*device);
  });
  if (!compatible) {
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

absl::optional<int64_t> SelectDominantDevice(
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
  return count > 0 ? absl::optional<int64_t>(device)
                   : absl::optional<int64_t>();
}

Status AssignComputationDevice(HloComputation* computation, int64_t device) {
  VLOG(4) << "Assigning device " << device << " to " << computation->name()
          << " computation";
  for (HloInstruction* instruction : computation->instructions()) {
    if (!instruction->has_sharding()) {
      VLOG(4) << "Assigning device " << device << " to " << instruction->name();
      instruction->set_device_sharding(device);
    }
  }
  return Status::OK();
}

absl::optional<int64_t> GetMostOccurringDevice(
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

StatusOr<absl::optional<int64_t>> GetDominantDevice(
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
  absl::optional<int64_t> device = SelectDominantDevice(device_map, &count);
  absl::optional<int64_t> dominant_device;
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
                              const std::vector<int64_t>& dimensions) {
  if (sharding.IsTileMaximal()) {
    return sharding;
  }
  auto perm_dimensions = dimensions;
  // Add subgroup dims if missing.
  if (sharding.TiledDataRank() == dimensions.size()) {
    for (int64_t i = sharding.TiledDataRank();
         i < sharding.tile_assignment().num_dimensions(); ++i) {
      perm_dimensions.push_back(i);
    }
  } else {
    CHECK_EQ(sharding.tile_assignment().num_dimensions(), dimensions.size());
  }
  const int64_t rank = perm_dimensions.size();
  std::vector<int64_t> tile_assignment_dim(rank);
  for (int64_t i = 0; i < rank; ++i) {
    tile_assignment_dim[i] = sharding.tile_assignment().dim(perm_dimensions[i]);
  }
  Array<int64_t> tile_assignment(tile_assignment_dim);
  tile_assignment.Each([&](absl::Span<const int64_t> indices, int64_t* value) {
    std::vector<int64_t> src_indices(indices.size(), -1);
    for (int64_t i = 0; i < indices.size(); ++i) {
      src_indices[perm_dimensions[i]] = indices[i];
    }
    *value = sharding.tile_assignment()(src_indices);
  });
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

absl::optional<HloSharding> ReshapeSharding(const Shape& source_shape,
                                            const Shape& target_shape,
                                            const HloSharding& sharding) {
  if (sharding.IsTileMaximal()) {
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
        return absl::nullopt;
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
        return absl::nullopt;
      }
      target_dims_stack.push_back(t_size);
    } else if (s_size > t_size) {
      // Dimension split.
      if (s_size % t_size != 0 || s_size % s_partitions != 0) {
        return absl::nullopt;
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
        return absl::nullopt;
      }
    } else {
      // Dimension merge. Also merge the source dimension with the next, and
      // process it next time.
      if (s_size % s_partitions != 0) {
        return absl::nullopt;
      }
      CHECK(!source_dims_stack.empty());
      if (sharding_tile_dims_stack.back() != 1 && s_size != s_partitions) {
        // If the next dimension to combine is sharded, we require that the
        // current dimension's shard size to be 1. Otherwise, the new shard
        // would be non-contiguous.
        return absl::nullopt;
      }
      source_dims_stack.back() *= s_size;
      sharding_tile_dims_stack.back() *= s_partitions;
      target_dims_stack.push_back(t_size);
    }
  }
  Array<int64_t> new_tile_assignment = sharding.tile_assignment();
  for (int64_t i = sharding.TiledDataRank();
       i < sharding.tile_assignment().num_dimensions(); ++i) {
    target_tile_assignment_dimensions.push_back(
        sharding.tile_assignment().dim(i));
  }
  new_tile_assignment.Reshape(target_tile_assignment_dimensions);
  return sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment,
                                        sharding.metadata())
             : HloSharding::Subgroup(new_tile_assignment,
                                     sharding.subgroup_types(),
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

HloSharding GatherOutputSharding(const HloSharding& index_sharding,
                                 const HloInstruction* hlo) {
  if (index_sharding.IsTileMaximal()) {
    return index_sharding;
  }

  const GatherDimensionNumbers& dnums = hlo->gather_dimension_numbers();
  std::vector<int64_t> output_tile_assignment_dims;
  const int64_t rank = hlo->shape().rank(),
                num_dimensions =
                    index_sharding.tile_assignment().num_dimensions();
  output_tile_assignment_dims.reserve(rank + num_dimensions);
  for (int64_t i = 0, index_dim = 0; i < rank; ++i) {
    if (absl::c_binary_search(dnums.offset_dims(), i)) {
      output_tile_assignment_dims.push_back(1);
    } else {
      const int64_t new_tile_dimension =
          index_dim >= dnums.index_vector_dim() ? index_dim + 1 : index_dim;
      output_tile_assignment_dims.push_back(
          index_sharding.tile_assignment().dim(new_tile_dimension));
      ++index_dim;
    }
  }

  for (int64_t i = index_sharding.TiledDataRank(); i < num_dimensions; ++i) {
    output_tile_assignment_dims.push_back(
        index_sharding.tile_assignment().dim(i));
  }

  Array<int64_t> new_tile_assignment = index_sharding.tile_assignment();
  if (new_tile_assignment.num_elements() !=
      Product(output_tile_assignment_dims)) {
    return HloSharding::Replicate(index_sharding.metadata());
  }
  new_tile_assignment.Reshape(output_tile_assignment_dims);
  return index_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment,
                                        index_sharding.metadata())
             : HloSharding::Subgroup(new_tile_assignment,
                                     index_sharding.subgroup_types(),
                                     index_sharding.metadata());
}

HloSharding GatherIndexSharding(const HloSharding& output_sharding,
                                const HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kGather);
  if (output_sharding.IsTileMaximal()) {
    return output_sharding;
  }

  const GatherDimensionNumbers& dnums = hlo->gather_dimension_numbers();
  std::vector<int64_t> index_tile_assignment_dims;
  // Relevant output dims have shardings passed to the index.
  std::vector<int64_t> relevant_output_dims;
  for (int64_t i = 0; i < hlo->shape().rank(); ++i) {
    if (!absl::c_binary_search(dnums.offset_dims(), i)) {
      index_tile_assignment_dims.push_back(
          output_sharding.tile_assignment().dim(i));
      relevant_output_dims.push_back(i);
    }
  }
  int64_t index_rank = hlo->operand(1)->shape().rank();

  // Vector indices sharding is not supported yet.
  if (index_rank > index_tile_assignment_dims.size()) {
    index_tile_assignment_dims.insert(
        index_tile_assignment_dims.begin() + dnums.index_vector_dim(), 1);
  }

  if (Product(index_tile_assignment_dims) == 1) {
    return HloSharding::Replicate(output_sharding.metadata());
  }
  HloSharding relevant_output_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(output_sharding,
                                                     relevant_output_dims);
  if (relevant_output_sharding.IsTileMaximal()) {
    return relevant_output_sharding;
  }
  for (int64_t i = relevant_output_sharding.TiledDataRank();
       i < relevant_output_sharding.tile_assignment().num_dimensions(); ++i) {
    index_tile_assignment_dims.push_back(
        relevant_output_sharding.tile_assignment().dim(i));
  }

  Array<int64_t> new_tile_assignment =
      relevant_output_sharding.tile_assignment();
  new_tile_assignment.Reshape(index_tile_assignment_dims);
  return relevant_output_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment,
                                        output_sharding.metadata())
             : HloSharding::Subgroup(new_tile_assignment,
                                     relevant_output_sharding.subgroup_types(),
                                     output_sharding.metadata());
}

HloSharding GatherEffectiveOutputSharding(const HloInstruction& hlo) {
  if (hlo.sharding().IsTileMaximal()) {
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

HloSharding ScatterIndexSharding(const HloSharding& data_sharding,
                                 const HloInstruction* hlo) {
  if (data_sharding.IsTileMaximal()) {
    return data_sharding;
  }

  const ScatterDimensionNumbers& dnums = hlo->scatter_dimension_numbers();
  std::vector<int64_t> index_tile_assignment_dims;
  std::vector<int64_t> relevant_data_dims;
  for (int64_t i = 0; i < hlo->operand(2)->shape().rank(); ++i) {
    if (!absl::c_binary_search(dnums.update_window_dims(), i)) {
      index_tile_assignment_dims.push_back(
          data_sharding.tile_assignment().dim(i));
      relevant_data_dims.push_back(i);
    }
  }
  if (index_tile_assignment_dims.size() < hlo->operand(1)->shape().rank()) {
    index_tile_assignment_dims.push_back(1);
  }
  HloSharding relevant_data_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(data_sharding,
                                                     relevant_data_dims);
  if (relevant_data_sharding.IsTileMaximal()) {
    return relevant_data_sharding;
  }
  for (int64_t i = relevant_data_sharding.TiledDataRank();
       i < relevant_data_sharding.tile_assignment().num_dimensions(); ++i) {
    index_tile_assignment_dims.push_back(
        relevant_data_sharding.tile_assignment().dim(i));
  }
  auto new_tile_assignment = relevant_data_sharding.tile_assignment();
  new_tile_assignment.Reshape(index_tile_assignment_dims);
  return relevant_data_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment,
                                        data_sharding.metadata())
             : HloSharding::Subgroup(new_tile_assignment,
                                     relevant_data_sharding.subgroup_types(),
                                     data_sharding.metadata());
}

HloSharding ScatterDataSharding(const HloSharding& index_sharding,
                                const HloInstruction* hlo) {
  if (index_sharding.IsTileMaximal()) {
    return index_sharding;
  }

  const ScatterDimensionNumbers& dnums = hlo->scatter_dimension_numbers();
  std::vector<int64_t> data_tile_assignment_dims;
  std::vector<int64_t> relevant_index_dims;
  const int64_t rank = hlo->operand(2)->shape().rank();
  data_tile_assignment_dims.reserve(rank);
  for (int64_t i = 0, index_dim = 0; i < rank; ++i) {
    if (absl::c_binary_search(dnums.update_window_dims(), i)) {
      data_tile_assignment_dims.push_back(1);
    } else {
      data_tile_assignment_dims.push_back(
          index_sharding.tile_assignment().dim(index_dim));
      relevant_index_dims.push_back(index_dim);
      index_dim++;
    }
  }
  auto relevant_index_sharding = PartiallyReplicateTiledShardingOnAllDimsExcept(
      index_sharding, relevant_index_dims);
  if (relevant_index_sharding.IsTileMaximal()) {
    return relevant_index_sharding;
  }
  for (int64_t i = relevant_index_sharding.TiledDataRank();
       i < relevant_index_sharding.tile_assignment().num_dimensions(); ++i) {
    data_tile_assignment_dims.push_back(
        relevant_index_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> new_tile_assignment =
      relevant_index_sharding.tile_assignment();
  new_tile_assignment.Reshape(data_tile_assignment_dims);
  return relevant_index_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment,
                                        index_sharding.metadata())
             : HloSharding::Subgroup(new_tile_assignment,
                                     relevant_index_sharding.subgroup_types(),
                                     index_sharding.metadata());
}

HloSharding ScatterEffectiveIndexSharding(const HloSharding& index_sharding,
                                          const HloInstruction& hlo) {
  if (index_sharding.IsTileMaximal()) {
    return index_sharding;
  }

  // Only shard on first "number of scatter_window_dims" dimensions.
  const ScatterDimensionNumbers& dnums = hlo.scatter_dimension_numbers();
  int64_t num_elements = 1;
  int64_t index_dim = 0;
  for (int64_t i = 0; i < hlo.shape().rank(); ++i) {
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

  const int64_t index_rank = hlo.operand(1)->shape().rank();
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
                                         const HloInstruction& hlo) {
  if (data_sharding.IsTileMaximal()) {
    return data_sharding;
  }

  const ScatterDimensionNumbers& dnums = hlo.scatter_dimension_numbers();
  const int64_t data_rank = hlo.operand(2)->shape().rank();
  std::vector<int64_t> tile_assignment_dims(data_rank, 1LL);
  int64_t num_elements = 1;
  for (int64_t i = 0; i < hlo.shape().rank(); ++i) {
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

// If partitioning in the operand only happens in dimensions in passthrough
// dimensions (offset dimensions in the gather output (or scatter update) that
// have the same size as the operand), returns the corresponding output (or
// update) sharding by passing through the input sharding.
absl::optional<HloSharding> PassthroughOperandToGatherOutputOrScatterUpdate(
    const Shape& operand_shape, const HloSharding& operand_sharding,
    const Shape& update_or_gather_shape,
    absl::Span<const int64_t> collapsed_or_inserted_dims,
    absl::Span<const int64_t> index_map,
    absl::Span<const int64_t> offset_or_window_dims,
    absl::Span<const int64_t> slice_size) {
  if (operand_sharding.IsTileMaximal()) {
    return operand_sharding;
  }
  std::vector<int64_t> passthrough_tile(update_or_gather_shape.rank(), 1);
  int64_t collapsed = 0;
  for (int64_t i = 0; i < operand_shape.rank(); ++i) {
    int64_t dim_partitions = operand_sharding.tile_assignment().dim(i);
    if (absl::c_linear_search(collapsed_or_inserted_dims, i) ||
        absl::c_linear_search(index_map, i)) {
      if (dim_partitions > 1) {
        return absl::nullopt;
      }
      collapsed++;
      continue;
    }
    if (slice_size[i] != operand_shape.dimensions(i) && dim_partitions > 1) {
      return absl::nullopt;
    }
    int64_t offset_dim = offset_or_window_dims[i - collapsed];
    if (i - collapsed > 0 &&
        offset_dim < offset_or_window_dims[i - collapsed - 1]) {
      // Output offsets are transposed, we do not support this case.
      return absl::nullopt;
    }
    passthrough_tile[offset_dim] = dim_partitions;
  }
  for (int64_t i = operand_sharding.TiledDataRank();
       i < operand_sharding.tile_assignment().num_dimensions(); ++i) {
    passthrough_tile.push_back(operand_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> tile_assignment = operand_sharding.tile_assignment();
  tile_assignment.Reshape(passthrough_tile);
  return operand_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment,
                                        operand_sharding.metadata())
             : HloSharding::Subgroup(tile_assignment,
                                     operand_sharding.subgroup_types(),
                                     operand_sharding.metadata());
}

// Inverse of PassthroughOperandToGatherOutputOrScatterUpdate.
absl::optional<HloSharding> PassthroughGatherOutputOrScatterUpdateToOperand(
    const Shape& operand_shape, const HloSharding& update_or_gather_sharding,
    absl::Span<const int64_t> collapsed_or_inserted_dims,
    absl::Span<const int64_t> index_map,
    absl::Span<const int64_t> offset_or_window_dims,
    absl::Span<const int64_t> slice_size) {
  if (update_or_gather_sharding.IsTileMaximal()) {
    return update_or_gather_sharding;
  }
  std::vector<int64_t> passthrough_tile(operand_shape.rank(), 1);
  int64_t collapsed = 0;
  // Relevant dims have shardings passed to the operand.
  std::vector<int64_t> relevant_update_or_gather_dims;
  for (int64_t i = 0; i < operand_shape.rank(); ++i) {
    if (absl::c_linear_search(collapsed_or_inserted_dims, i) ||
        absl::c_linear_search(index_map, i)) {
      collapsed++;
      continue;
    }
    int64_t offset_dim = offset_or_window_dims[i - collapsed];
    int64_t dim_partitions =
        update_or_gather_sharding.tile_assignment().dim(offset_dim);
    if (slice_size[i] != operand_shape.dimensions(i) && dim_partitions > 1) {
      return absl::nullopt;
    }
    if (i - collapsed > 0 &&
        offset_dim < offset_or_window_dims[i - collapsed - 1]) {
      // Output offsets are transposed, we do not support this case.
      return absl::nullopt;
    }
    relevant_update_or_gather_dims.push_back(offset_dim);
    passthrough_tile[i] = dim_partitions;
  }

  HloSharding relevant_sharding =
      PartiallyReplicateTiledShardingOnAllDimsExcept(
          update_or_gather_sharding, relevant_update_or_gather_dims);
  if (relevant_sharding.IsTileMaximal()) {
    return relevant_sharding;
  }
  for (int64_t i = relevant_sharding.TiledDataRank();
       i < relevant_sharding.tile_assignment().num_dimensions(); ++i) {
    passthrough_tile.push_back(relevant_sharding.tile_assignment().dim(i));
  }
  Array<int64_t> tile_assignment = relevant_sharding.tile_assignment();
  tile_assignment.Reshape(passthrough_tile);
  return relevant_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(tile_assignment,
                                        update_or_gather_sharding.metadata())
             : HloSharding::Subgroup(tile_assignment,
                                     relevant_sharding.subgroup_types(),
                                     update_or_gather_sharding.metadata());
}

// Collect data operand sharding for a gather with parallel dimensions from
// the output.
absl::optional<HloSharding> GatherParallelDataOperandSharding(
    const HloSharding& output_sharding, const HloInstruction& gather,
    const GatherParallelDims& parallel_dims) {
  if (output_sharding.IsTileMaximal()) {
    return output_sharding;
  }
  auto output_parallel_dims = GatherParallelOutputDims(gather, parallel_dims);
  auto output_aligned_operand_parallel_dims =
      GatherOutputAlignedOperandParallelDims(gather, parallel_dims);
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
             : HloSharding::Subgroup(tile_assignment,
                                     relevant_output_sharding.subgroup_types(),
                                     output_sharding.metadata());
}

}  // namespace

absl::optional<HloSharding> GatherOutputShardingFromDataOperand(
    const HloSharding& data_operand_sharding, const HloInstruction& hlo,
    absl::Span<const int64_t> slice_sizes, const Shape& output_shape,
    const Shape& operand_shape) {
  const auto& dnums = hlo.gather_dimension_numbers();
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());
  return PassthroughOperandToGatherOutputOrScatterUpdate(
      operand_shape, data_operand_sharding, output_shape, collapsed_slice_dims,
      start_index_map, offset_dims, slice_sizes);
}

absl::optional<HloSharding> GatherDataOperandShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo) {
  const auto& dnums = hlo.gather_dimension_numbers();
  std::vector<int64_t> collapsed_slice_dims(
      dnums.collapsed_slice_dims().begin(), dnums.collapsed_slice_dims().end());
  std::vector<int64_t> start_index_map(dnums.start_index_map().begin(),
                                       dnums.start_index_map().end());
  std::vector<int64_t> offset_dims(dnums.offset_dims().begin(),
                                   dnums.offset_dims().end());

  absl::optional<HloSharding> parallel_sharding;
  auto parallel_dims = GetGatherBatchParallelDims(hlo);
  if (parallel_dims) {
    // Prioritize parallel sharding first as this is how it is in
    // spmd_partitioner.
    parallel_sharding =
        GatherParallelDataOperandSharding(output_sharding, hlo, *parallel_dims);
  }
  absl::optional<HloSharding> passthrough_sharding =
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

absl::optional<HloSharding> ScatterOutputShardingFromUpdate(
    const HloSharding& update_sharding, const HloInstruction& hlo) {
  const auto& dnums = hlo.scatter_dimension_numbers();
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  std::vector<int64_t> slice_size(hlo.shape().rank(), 1);
  int64_t num_update_window_dims = 0;
  for (int64_t i = 0; i < hlo.shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.inserted_window_dims(), i)) {
      continue;
    }
    slice_size[i] = hlo.operand(2)->shape().dimensions(
        dnums.update_window_dims(num_update_window_dims++));
  }
  return PassthroughGatherOutputOrScatterUpdateToOperand(
      hlo.shape(), update_sharding, inserted_window_dims,
      scatter_dims_to_operand_dims, update_window_dims, slice_size);
}

absl::optional<HloSharding> ScatterUpdateShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo) {
  const auto& dnums = hlo.scatter_dimension_numbers();
  std::vector<int64_t> inserted_window_dims(
      dnums.inserted_window_dims().begin(), dnums.inserted_window_dims().end());
  std::vector<int64_t> scatter_dims_to_operand_dims(
      dnums.scatter_dims_to_operand_dims().begin(),
      dnums.scatter_dims_to_operand_dims().end());
  std::vector<int64_t> update_window_dims(dnums.update_window_dims().begin(),
                                          dnums.update_window_dims().end());
  std::vector<int64_t> slice_size(hlo.shape().rank(), 1);
  int64_t num_update_window_dims = 0;
  for (int64_t i = 0; i < hlo.shape().rank(); ++i) {
    if (absl::c_linear_search(dnums.inserted_window_dims(), i)) {
      continue;
    }
    slice_size[i] = hlo.operand(2)->shape().dimensions(
        dnums.update_window_dims(num_update_window_dims++));
  }
  return PassthroughOperandToGatherOutputOrScatterUpdate(
      hlo.shape(), output_sharding, hlo.operand(2)->shape(),
      inserted_window_dims, scatter_dims_to_operand_dims, update_window_dims,
      slice_size);
}

StatusOr<std::pair<std::unique_ptr<HloInstruction>, HloOpcode>>
IdentityValueAndHloOpcodeForScatterReduceComputation(
    const HloScatterInstruction& scatter) {
  auto computation = scatter.to_apply();
  // We only handle computations with 2 parameters and only 1 calculation.
  if (computation->instruction_count() != 3) {
    return Status(
        tensorflow::error::Code::INVALID_ARGUMENT,
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

  return Status(tensorflow::error::Code::INVALID_ARGUMENT,
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
    const HloSharding& sharding,
    const std::vector<int64_t>& available_devices) {
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
  if (sharding.IsTileMaximal()) {
    return sharding;
  }
  int64_t group_count = 1;
  for (int64_t dim : dims_to_replicate) {
    if (sharding.ReplicateOnLastTileDim()) {
      CHECK_LT(dim, sharding.tile_assignment().num_dimensions() - 1);
    } else {
      CHECK_LT(dim, sharding.tile_assignment().num_dimensions() -
                        sharding.subgroup_types().size());
    }
    group_count *= sharding.tile_assignment().dim(dim);
  }
  if (group_count == 1) {
    return sharding;
  }
  if (group_count == sharding.NumTiles() && sharding.subgroup_types().empty()) {
    return HloSharding::Replicate(sharding.metadata());
  }
  std::vector<int64_t> dim_permutation(sharding.TiledDataRank());
  std::iota(dim_permutation.begin(), dim_permutation.end(), 0);
  absl::c_stable_sort(dim_permutation, [&](const int64_t a, const int64_t b) {
    return absl::c_linear_search(dims_to_replicate, a) <
           absl::c_linear_search(dims_to_replicate, b);
  });
  auto new_tile =
      TransposeSharding(sharding, dim_permutation).tile_assignment();
  std::vector<int64_t> new_tile_shape(
      sharding.tile_assignment().dimensions().begin(),
      sharding.tile_assignment().dimensions().end());
  for (int64_t dim : dims_to_replicate) {
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
  if (sharding.IsTileMaximal()) {
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
                                  const std::vector<int64_t>& dims_to_remove) {
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

absl::optional<HloSharding> TransposeShardingWithCollapsedDims(
    const HloSharding& source, absl::Span<int64_t const> src_to_tgt,
    absl::Span<int64_t const> tgt_to_src) {
  if (source.IsTileMaximal()) {
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
      if (i < tgt_non_subgroup_dims) {
        tgt_dims_skipping_new[i] = i - skipped_tgt_dims;
      } else {
        tgt_dims_skipping_new[i] = i;
      }
    }
  }
  int64_t skipped_src_dims = absl::c_count(src_to_tgt, -1);
  std::vector<int64_t> perm(src_to_tgt.size());
  for (int64_t i = 0; i < src_non_subgroup_dims; ++i) {
    if (src_to_tgt[i] < 0) {
      if (source.tile_assignment().dim(i) > 1) {
        return absl::nullopt;
      }
      perm[src_non_subgroup_dims - skipped_src_dims] = i;
      skipped_src_dims--;
    } else {
      perm[tgt_dims_skipping_new[src_to_tgt[i]]] = i;
    }
  }
  for (int64_t i = src_non_subgroup_dims; i < src_to_tgt.size(); ++i) {
    CHECK_GE(tgt_to_src[i], tgt_non_subgroup_dims)
        << "Sharding transpose should not move subgroup dims before data dims.";
    perm[tgt_to_src[i]] = i;
  }
  auto tgt_sharding = hlo_sharding_util::TransposeSharding(source, perm);
  auto reshape_tiles = tgt_sharding.tile_assignment();
  std::vector<int64_t> tgt_tiles(tgt_to_src.size(), 1);
  for (int64_t i = 0; i < tgt_tiles.size(); ++i) {
    if (tgt_to_src[i] >= 0) {
      tgt_tiles[i] = reshape_tiles.dim(tgt_dims_skipping_new[i]);
    }
  }
  reshape_tiles.Reshape(tgt_tiles);
  return source.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(reshape_tiles, source.metadata())
             : HloSharding::Subgroup(reshape_tiles, source.subgroup_types(),
                                     source.metadata());
}

absl::optional<int64_t> GetDimensionForIota(const HloInstruction* maybe_iota) {
  if (auto* iota = DynCast<HloIotaInstruction>(maybe_iota)) {
    return iota->iota_dimension();
  }

  if (maybe_iota->shape().element_type() != S32) {
    return absl::nullopt;
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
    return absl::nullopt;
  }

  if (maybe_iota->opcode() == HloOpcode::kBroadcast) {
    auto operand_dim = GetDimensionForIota(maybe_iota->operand(0));
    if (operand_dim) {
      return maybe_iota->dimensions(*operand_dim);
    }
    return absl::nullopt;
  }
  return absl::nullopt;
}

absl::optional<GatherParallelDims> GetGatherBatchParallelDims(
    const HloInstruction& hlo) {
  const auto& dnums = hlo.gather_dimension_numbers();
  int64_t index_dim = dnums.index_vector_dim();
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
  const HloInstruction* indices = hlo.operand(1);
  const int num_indices = dnums.start_index_map_size();
  std::vector<int64_t> index_parallel_in_dim(num_indices, -1);
  // Handle cases where we concatenate pieces of the indices one at a time.
  if (indices->opcode() == HloOpcode::kConcatenate &&
      indices->concatenate_dimension() == index_dim) {
    int concatenated_dims = 0;
    for (int i = 0; i < indices->operand_count(); ++i) {
      const HloInstruction* op = indices->operand(i);
      const int64_t num_indices_from_element =
          op->shape().dimensions_size() > index_dim
              ? op->shape().dimensions(index_dim)
              : 1;
      if (absl::optional<int64_t> maybe_iota_dim = GetDimensionForIota(op)) {
        if (*maybe_iota_dim != index_dim) {
          for (int j = 0; j < num_indices_from_element; ++j) {
            index_parallel_in_dim[concatenated_dims + j] = *maybe_iota_dim;
          }
        }
      }
      concatenated_dims += num_indices_from_element;
    }
  } else if (absl::optional<int64_t> maybe_iota_dim =
                 GetDimensionForIota(indices)) {
    if (*maybe_iota_dim != index_dim) {
      // This is a case of a single iota with index_dim being out of bounds.
      const int64_t num_indices_from_element =
          indices->shape().dimensions_size() > index_dim
              ? indices->shape().dimensions(index_dim)
              : 1;
      index_parallel_in_dim.assign(num_indices_from_element, *maybe_iota_dim);
    }
  }
  absl::InlinedVector<int64_t, 1> indices_parallel_dims;
  absl::InlinedVector<int64_t, 1> operand_parallel_dims;
  // Map the parallelizable dimension from the iota to the dimensions of the
  // output and the operand. These dimensions are interconnected, but between
  // operands and index they could have different spots in the shape because the
  // position of the index dimension in the operand is determined by
  // start_index_map.
  for (int i = 0; i < index_parallel_in_dim.size(); ++i) {
    int index_parallel_dim = index_parallel_in_dim[i];
    if (index_parallel_dim == -1) {
      continue;
    }
    if (absl::c_linear_search(indices_parallel_dims, index_parallel_dim)) {
      return absl::nullopt;
    }
    // Considered parallel only if the slice is of size 1 over the operand.
    if (hlo.gather_slice_sizes()[dnums.start_index_map(i)] == 1) {
      indices_parallel_dims.push_back(index_parallel_dim);
      operand_parallel_dims.push_back(dnums.start_index_map(i));
    } else {
      index_parallel_in_dim[i] = -1;
    }
  }
  absl::c_sort(indices_parallel_dims);
  if (!indices_parallel_dims.empty()) {
    return GatherParallelDims{indices_parallel_dims, operand_parallel_dims,
                              index_parallel_in_dim};
  }
  return absl::nullopt;
}

absl::InlinedVector<int64_t, 1> GatherParallelOutputDims(
    const HloInstruction& gather, const GatherParallelDims& parallel_dim) {
  absl::InlinedVector<int64_t, 1> output_parallel_dims;
  auto indices_parallel_dims = parallel_dim.indices_parallel_dims;
  const Shape gather_shape = gather.shape();
  auto dnums = gather.gather_dimension_numbers();
  for (int i = 0, idx_dim = 0; i < gather_shape.dimensions_size(); ++i) {
    if (absl::c_linear_search(dnums.offset_dims(), i)) {
      continue;
    }
    const int index_dim =
        idx_dim < dnums.index_vector_dim() ? idx_dim : idx_dim + 1;
    if (absl::c_binary_search(indices_parallel_dims, index_dim)) {
      output_parallel_dims.push_back(i);
    }
    ++idx_dim;
  }
  return output_parallel_dims;
}

absl::InlinedVector<int64_t, 1> GatherOutputAlignedOperandParallelDims(
    const HloInstruction& gather, const GatherParallelDims& parallel_dims) {
  absl::InlinedVector<int64_t, 1> operand_parallel_dim_to_output(
      parallel_dims.operand_parallel_dims.size(), -1);
  auto dnums = gather.gather_dimension_numbers();
  CHECK_LE(parallel_dims.indices_parallel_dims.size(),
           parallel_dims.operand_parallel_dims.size());
  for (int i = 0; i < parallel_dims.index_parallel_in_dim.size(); ++i) {
    // This is the equivalent batch dimension of the indices that corresponds
    // to this index dimension.
    const int64_t index_parallel_dim = parallel_dims.index_parallel_in_dim[i];
    // If it's not an index that is parallel skip.
    if (index_parallel_dim == -1) {
      continue;
    }
    // This is small so just look linearly. Populate the operand parallel
    // dimensions based on the order of the index batch dims (which is the same
    // order as the output).
    for (int j = 0; j < parallel_dims.indices_parallel_dims.size(); ++j) {
      if (parallel_dims.indices_parallel_dims[j] == index_parallel_dim) {
        const int64_t operand_parallel_dim = dnums.start_index_map(i);
        if (operand_parallel_dim_to_output[j] == -1) {
          operand_parallel_dim_to_output[j] = operand_parallel_dim;
        }
        break;
      }
    }
  }
  return operand_parallel_dim_to_output;
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

HloSharding UngroupSharding(const GroupedSharding& grouped_sharding) {
  std::vector<int64_t> tiling_dims;
  bool partial_sharding = false;
  std::vector<OpSharding::Type> subgroup_types;
  Array<int64_t> grouped_tiling = grouped_sharding.sharding.tile_assignment();
  if (grouped_sharding.sharding.IsTileMaximal()) {
    tiling_dims = std::vector<int64_t>(grouped_sharding.data_rank, 1);
    if (grouped_sharding.device_groups[0].size() != 1) {
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

}  // namespace hlo_sharding_util
}  // namespace xla
