/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/collective_opt_utils.h"

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"

namespace xla {
namespace {
bool IsTableLookup(const HloInstruction* hlo) {
  while (hlo->opcode() == HloOpcode::kBitcast ||
         hlo->opcode() == HloOpcode::kReshape ||
         hlo->opcode() == HloOpcode::kCopy) {
    hlo = hlo->operand(0);
  }
  return hlo->opcode() == HloOpcode::kDynamicSlice &&
         (hlo->operand(0)->IsConstant() ||
          hlo->operand(0)->opcode() == HloOpcode::kIota) &&
         hlo->operand(0)->shape().dimensions_size() == 1 &&
         (hlo->operand(0)->shape().element_type() == S32 ||
          hlo->operand(0)->shape().element_type() == U32);
}

std::optional<int64_t> GetScalarInt64Value(const HloInstruction* constant) {
  CHECK_EQ(constant->opcode(), HloOpcode::kConstant);
  CHECK(ShapeUtil::IsEffectiveScalar(constant->shape()));
  absl::InlinedVector<int64_t, 8> multi_index(
      constant->shape().dimensions_size());
  return constant->literal().GetIntegralAsS64(multi_index);
}

// Function to map a replica/partition/global ID to an offset in the offset
// table, based on the given scalar offset HLO. For example, if the HLO is
// kPartitionId but the all-reduce uses global IDs, then the function maps
// global IDs to partition IDs. It returns -1 if the HLO cannot be understood.
using MapIdToTableOffset =
    std::function<int64_t(const HloInstruction*, int64_t)>;

int64_t GetIndexForId(const HloInstruction* index, int64_t id,
                      const MapIdToTableOffset& map_id) {
  // ID itself.
  int64_t maybe_mapped_id = map_id(index, id);
  if (maybe_mapped_id >= 0) {
    return maybe_mapped_id;
  }
  if (!IsTableLookup(index)) {
    VLOG(2) << "Index is not table lookup " << index->ToString();
    return -1;
  }
  while (index->opcode() == HloOpcode::kReshape ||
         index->opcode() == HloOpcode::kBitcast ||
         index->opcode() == HloOpcode::kCopy) {
    index = index->operand(0);
  }
  int64_t inner_index = GetIndexForId(index->operand(1), id, map_id);
  if (inner_index < 0) {
    VLOG(2) << "Failed to get inner index.";
    return -1;
  }
  if (index->operand(0)->opcode() == HloOpcode::kIota) {
    return inner_index;
  }
  // A table lookup.
  const auto& table = index->operand(0)->literal();
  return *table.GetIntegralAsS64({inner_index});
}

bool IsPerIdOffsets(absl::Span<const HloInstruction*> offsets,
                    int64_t shard_size, const MapIdToTableOffset& map_id,
                    std::vector<int64_t> slice_group_sizes,
                    const HloInstruction* instruction, bool is_cross_module,
                    bool use_global_device_ids) {
  if (offsets.size() != slice_group_sizes.size()) {
    return false;
  }
  if (!is_cross_module || !use_global_device_ids) {
    return false;
  }

  int num_groups = instruction->replica_groups().size();
  int num_split_dims = slice_group_sizes.size();

  for (int64_t i = 0; i < num_groups; ++i) {
    for (int64_t j = 0; j < Product(slice_group_sizes); ++j) {
      int64_t final_table_entry = 0;
      int64_t id = instruction->replica_groups()[i].replica_ids(j);
      int64_t slice_group_size = Product(slice_group_sizes);
      for (int dim = 0; dim < num_split_dims; dim++) {
        auto scalar_offset = offsets[dim];
        while (scalar_offset->opcode() == HloOpcode::kReshape ||
               scalar_offset->opcode() == HloOpcode::kBitcast ||
               scalar_offset->opcode() == HloOpcode::kCopy) {
          scalar_offset = scalar_offset->operand(0);
        }
        if (!IsTableLookup(scalar_offset)) {
          return false;
        }
        int64_t table_index =
            GetIndexForId(scalar_offset->operand(1), id, map_id);
        if (table_index < 0) {
          return false;
        }

        int64_t table_entry;
        if (scalar_offset->operand(0)->opcode() == HloOpcode::kIota) {
          table_entry = table_index;
        } else {
          table_entry = *scalar_offset->operand(0)->literal().GetIntegralAsS64(
              {table_index});
        }
        slice_group_size /= slice_group_sizes[dim];
        final_table_entry += table_entry * slice_group_size;
      }
      if (final_table_entry != shard_size * j) {
        return false;
      }
    }
  }

  return true;
}

// Returns if `offset` == shard_size * id.
bool IsPerIdOffset(const HloInstruction* offset, int64_t shard_size,
                   const MapIdToTableOffset& map_id, int64_t group_size,
                   const HloInstruction* instruction, bool is_cross_module,
                   bool use_global_device_ids) {
  const bool iota_group = instruction->replica_groups().empty() ||
                          (is_cross_module && !use_global_device_ids);

  if (offset->opcode() == HloOpcode::kMultiply) {
    // Check if it's constant * IsPerIdOffset(..., shard_size / constant, ...)
    if (!ShapeUtil::IsEffectiveScalar(offset->shape())) {
      VLOG(2) << "Offset is not a scalar " << offset->ToString();
      return false;
    }
    int64_t const_operand = -1;
    if (offset->operand(0)->IsConstant()) {
      const_operand = 0;
    } else if (offset->operand(1)->IsConstant()) {
      const_operand = 1;
    } else {
      VLOG(2) << "Offset is not multiple(const, ...) " << offset->ToString();
      return false;
    }
    auto multiplier = GetScalarInt64Value(offset->operand(const_operand));
    if (!multiplier || shard_size % *multiplier != 0) {
      VLOG(2) << "Multiplier is unknown or cannot evenly divide shard size "
              << offset->operand(const_operand);
      return false;
    }
    return IsPerIdOffset(offset->operand(1 - const_operand),
                         shard_size / *multiplier, map_id, group_size,
                         instruction, is_cross_module, use_global_device_ids);
  }
  if (shard_size == 1 && iota_group) {
    bool id_mapping_is_identity = true;
    for (int64_t id = 0; id < group_size; ++id) {
      int64_t mapped_id = map_id(offset, id);
      if (mapped_id != id) {
        id_mapping_is_identity = false;
        break;
      }
    }
    if (id_mapping_is_identity) {
      return true;
    }
  }
  if (offset->opcode() == HloOpcode::kBitcast ||
      offset->opcode() == HloOpcode::kReshape ||
      offset->opcode() == HloOpcode::kCopy) {
    return IsPerIdOffset(offset->operand(0), shard_size, map_id, group_size,
                         instruction, is_cross_module, use_global_device_ids);
  }

  if (offset->opcode() == HloOpcode::kConvert &&
      offset->operand(0)->shape().AreAllLeavesIntegers() &&
      primitive_util::BitWidth(offset->operand(0)->shape().element_type()) <=
          primitive_util::BitWidth(offset->shape().element_type())) {
    return IsPerIdOffset(offset->operand(0), shard_size, map_id, group_size,
                         instruction, is_cross_module, use_global_device_ids);
  }

  if (offset->opcode() == HloOpcode::kClamp) {
    auto lower_bound = GetScalarInt64Value(offset->operand(0));
    auto upper_bound = GetScalarInt64Value(offset->operand(2));
    if (!lower_bound || !upper_bound || lower_bound != 0 ||
        *upper_bound < (group_size - 1) * shard_size) {
      VLOG(2) << "Boundaries of the clamp are not legal: "
              << offset->ToString();
      return false;
    }
    return IsPerIdOffset(offset->operand(1), shard_size, map_id, group_size,
                         instruction, is_cross_module, use_global_device_ids);
  }

  const int64_t num_groups =
      iota_group ? 1 : instruction->replica_groups().size();
  if (IsTableLookup(offset)) {
    // Check the values of the offset table, and see if they are shard_index *
    // shard_size.
    for (int64_t i = 0; i < num_groups; ++i) {
      for (int64_t j = 0; j < group_size; ++j) {
        int64_t id =
            iota_group ? j : instruction->replica_groups()[i].replica_ids(j);
        int64_t table_index = GetIndexForId(offset->operand(1), id, map_id);
        if (table_index < 0) {
          VLOG(2) << "Failed to infer table index from "
                  << offset->operand(1)->ToString();
          return false;
        }

        int64_t table_entry;
        if (offset->operand(0)->opcode() == HloOpcode::kIota) {
          table_entry = table_index;
        } else {
          table_entry =
              *offset->operand(0)->literal().GetIntegralAsS64({table_index});
        }
        if (table_entry != shard_size * j) {
          VLOG(2) << "Unexpected offset from table.";
          return false;
        }
      }
    }

    // All table entries are good.
    return true;
  }

  // Check if the offset is the id itself and it has the right values.
  for (int64_t i = 0; i < num_groups; ++i) {
    for (int64_t j = 0; j < group_size; ++j) {
      int64_t id =
          iota_group ? j : instruction->replica_groups()[i].replica_ids(j);
      int mapped_id = map_id(offset, id);
      if (mapped_id != shard_size * j) {
        VLOG(2) << "Mapping of " << id << " to " << mapped_id
                << " not matching expected value " << shard_size * j << ": "
                << offset->ToString();
        return false;
      }
    }
  }

  return true;
}

std::optional<ReduceScatterSpec> SpecFromReduceScatterInstr(
    const HloInstruction* rs_instr, int64_t num_partitions,
    int64_t num_replicas, int64_t min_rank, bool is_constrain_layout,
    bool use_global_device_ids, bool is_cross_module) {
  if (rs_instr->shape().dimensions_size() < min_rank) {
    return std::nullopt;
  }
  CHECK(rs_instr->opcode() == HloOpcode::kReduceScatter);
  ReduceScatterSpec spec;
  spec.split_dim = rs_instr->dimensions(0);
  if (!is_cross_module) {
    spec.sharded_replicas = num_replicas;
    spec.group_size = rs_instr->replica_groups().empty()
                          ? num_replicas
                          : rs_instr->replica_groups()[0].replica_ids_size();
  } else if (use_global_device_ids) {
    spec.sharded_replicas = num_replicas;
    spec.sharded_partitions = num_partitions;
    spec.group_size = rs_instr->replica_groups()[0].replica_ids_size();
  } else {
    spec.sharded_partitions = num_partitions;
    spec.group_size = num_partitions;
  }
  spec.original_split_dims = {spec.split_dim};
  spec.dynamic_slice = nullptr;
  return spec;
}

}  // namespace

std::optional<ReduceScatterSpec> MatchReduceScatter(
    const HloAllReduceInstructionBase* ar, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims,
    bool allow_intervening_reshape, int64_t min_rank,
    HloPredicate match_partition_id, HloPredicate match_replica_id,
    bool allow_intervening_bitcast) {
  if (ar->opcode() == HloOpcode::kReduceScatter) {
    return SpecFromReduceScatterInstr(
        ar, num_partitions, num_replicas, min_rank, ar->constrain_layout(),
        ar->use_global_device_ids(), ar->channel_id().has_value());
  }
  auto spec = MatchWithDynamicSlice(
      ar, num_partitions, num_replicas, allow_multiple_split_dims,
      allow_intervening_reshape, min_rank, match_partition_id, match_replica_id,
      ar->constrain_layout(), ar->use_global_device_ids(),
      ar->channel_id() && ar->opcode() == HloOpcode::kAllReduce,
      allow_intervening_bitcast);
  return spec;
}

std::optional<ReduceScatterSpec> AllGatherDynamicSliceCancellation(
    const HloAllGatherInstruction* ag, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims,
    bool allow_intervening_reshape, int64_t min_rank,
    HloPredicate match_partition_id, HloPredicate match_replica_id,
    bool allow_intervening_bitcast, bool allow_multiple_users) {
  auto spec = MatchWithDynamicSlice(
      ag, num_partitions, num_replicas, allow_multiple_split_dims,
      allow_intervening_reshape, min_rank, match_partition_id, match_replica_id,
      ag->constrain_layout(), ag->use_global_device_ids(),
      ag->channel_id() && ag->opcode() == HloOpcode::kAllGather,
      allow_intervening_bitcast, allow_multiple_users);

  if (!spec.has_value()) {
    return std::nullopt;
  }
  if (spec->dynamic_slice && spec->split_dim != ag->all_gather_dimension()) {
    VLOG(2) << "Mismatch AG and DS: AG: " << ag->ToString()
            << ", DS: " << spec->dynamic_slice->ToString()
            << ", ag_dim: " << ag->all_gather_dimension()
            << ", ds_dim: " << spec->split_dim;
    return std::nullopt;
  }
  return spec;
}

std::optional<ReduceScatterSpec> MatchWithDynamicSlice(
    const HloChannelInstruction* instruction, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims,
    bool allow_intervening_reshape, int64_t min_rank,
    HloPredicate match_partition_id, HloPredicate match_replica_id,
    bool is_constrain_layout, bool use_global_device_ids, bool is_cross_module,
    bool allow_intervening_bitcast, bool allow_multiple_users) {
  if (!instruction->shape().IsArray() || is_constrain_layout ||
      (is_cross_module &&
       !instruction->GetModule()->config().use_spmd_partitioning())) {
    VLOG(2) << "Unsupported collective: " << instruction->ToString();
    return std::nullopt;
  }
  if (instruction->shape().dimensions_size() -
          absl::c_count(instruction->shape().dimensions(), 1) <
      min_rank) {
    VLOG(2) << " Should be at least rank-" << min_rank
            << " excluding trivial dimensions " << instruction->ToString();
    return std::nullopt;
  }
  if (!allow_multiple_users && instruction->user_count() != 1) {
    VLOG(2) << "All-gather user_count != 1 " << instruction->ToString();
    return std::nullopt;
  }
  if (instruction->replica_groups().size() > 1) {
    const int64_t size = instruction->replica_groups()[0].replica_ids_size();
    absl::Span<const ReplicaGroup> rgs = instruction->replica_groups();
    const bool has_uniform_size = absl::c_all_of(
        rgs.subspan(1, size - 1), [size](const ReplicaGroup& group) {
          return group.replica_ids_size() == size;
        });
    if (!has_uniform_size) {
      VLOG(2) << "Unsupported non-uniform replica group size "
              << instruction->ToString();
      return std::nullopt;
    }
  }
  // Always assume first user to start.
  HloInstruction* user = instruction->users()[0];
  if (allow_multiple_users) {
    // If we find a reshape or dynamic-slice use that.
    for (auto* some_user : instruction->users()) {
      if ((allow_intervening_reshape &&
           some_user->opcode() == HloOpcode::kReshape) ||
          some_user->opcode() == HloOpcode::kDynamicSlice) {
        user = some_user;
        break;
      }
    }
  }
  HloInstruction* reshape = nullptr;
  if (allow_intervening_reshape && user->opcode() == HloOpcode::kReshape) {
    // Allow the intervening reshape if it reshapes just the non scattered
    // dimension (checked later).
    reshape = user;
    if (reshape->user_count() != 1) {
      VLOG(2) << "Reshape following all-reduce has user count > 1"
              << reshape->ToString();
      return std::nullopt;
    }
    user = reshape->users().front();
  }
  HloInstruction* bitcast = nullptr;
  if (allow_intervening_bitcast && user->opcode() == HloOpcode::kBitcast) {
    VLOG(2) << "Allowing intervening bitcast " << user->ToString();
    bitcast = user;
    if (bitcast->user_count() != 1) {
      VLOG(2) << "Bitcast following all-reduce has user count > 1"
              << bitcast->ToString();
      return std::nullopt;
    }
    user = bitcast->users().front();
  }

  if (user->opcode() != HloOpcode::kDynamicSlice) {
    VLOG(2) << "AG or AR user is not dynamic slice " << user->ToString();
    return std::nullopt;
  }

  ReduceScatterSpec spec;
  int64_t group_size;
  MapIdToTableOffset map_id;
  spec.dynamic_slice = user;
  if (!is_cross_module) {
    spec.sharded_replicas = num_replicas;
    group_size = instruction->replica_groups().empty()
                     ? num_replicas
                     : instruction->replica_groups()[0].replica_ids_size();
    map_id = [&](const HloInstruction* hlo, int64_t id) {
      return match_replica_id(hlo) ? id : -1;
    };
  } else if (use_global_device_ids) {
    spec.sharded_replicas = num_replicas;
    spec.sharded_partitions = num_partitions;
    group_size = instruction->replica_groups()[0].replica_ids_size();
    bool orthogonal_replicas = true;
    std::vector<int64_t> partition_id_to_index(num_partitions, -1);
    for (int64_t g = 0; g < instruction->replica_groups().size(); ++g) {
      const auto& group = instruction->replica_groups()[g];
      for (int64_t i = 0; i < group.replica_ids_size(); ++i) {
        int64_t global_id = group.replica_ids(i);
        int64_t partition_id = global_id % num_partitions;
        if (partition_id_to_index[partition_id] == -1) {
          partition_id_to_index[partition_id] = i;
          continue;
        }
        if (partition_id_to_index[partition_id] != i ||
            global_id / num_partitions !=
                group.replica_ids(0) / num_partitions) {
          orthogonal_replicas = false;
          break;
        }
      }
    }
    map_id = [&, orthogonal_replicas](const HloInstruction* hlo, int64_t id) {
      if (match_replica_id(hlo)) {
        return num_partitions == 1 ? id : -1;
      }
      if (match_partition_id(hlo)) {
        if (num_replicas == 1) {
          return id;
        }
        return orthogonal_replicas ? id % num_partitions : -1;
      }
      auto is_replica_mul_num_partitions = [&](const HloInstruction* operand) {
        return operand->opcode() == HloOpcode::kMultiply &&
               ((operand->operand(0)->opcode() == HloOpcode::kReplicaId &&
                 operand->operand(1)->IsConstant() &&
                 GetScalarInt64Value(operand->operand(1)) == num_partitions) ||
                (operand->operand(1)->opcode() == HloOpcode::kReplicaId &&
                 operand->operand(0)->IsConstant() &&
                 GetScalarInt64Value(operand->operand(0)) == num_partitions));
      };
      if (hlo->opcode() == HloOpcode::kAdd &&
          ((match_partition_id(hlo->operand(0)) &&
            is_replica_mul_num_partitions(hlo->operand(1))) ||
           (match_partition_id(hlo->operand(1)) &&
            is_replica_mul_num_partitions(hlo->operand(0))))) {
        return id;
      }
      return int64_t{-1};
    };
  } else {
    // Right now all cross-partition all-reduces' subgroups refer to replicas
    // unless they use use_global_device_ids.
    if (instruction->replica_groups().size() != num_replicas ||
        instruction->replica_groups()[0].replica_ids_size() != 1) {
      VLOG(2) << "Unsupported size > 1 replica groups for cross-partition, "
                 "non-global ID "
              << instruction->ToString();
      return std::nullopt;
    }
    spec.sharded_partitions = num_partitions;
    group_size = num_partitions;
    map_id = [&](const HloInstruction* hlo, int64_t id) {
      return match_partition_id(hlo) ? id : -1;
    };
  }
  if (group_size < 2) {
    VLOG(2) << "Group_size < 2, nothing to do " << instruction->ToString();
    return std::nullopt;
  }
  spec.group_size = group_size;
  spec.split_dim = -1;
  std::vector<int64_t> split_dims;
  // First find a single dimension where the input and output of dynamic slice
  // differ.
  int num_dims = 0;
  for (int64_t dim = 0; dim < user->operand(0)->shape().dimensions_size();
       ++dim) {
    if (user->operand(0)->shape().dimensions(dim) ==
        user->shape().dimensions(dim)) {
      continue;
    }
    num_dims++;
    VLOG(2) << "select dim: " << dim;
    spec.split_dim = dim;
  }
  if (spec.split_dim != -1) {
    if (num_dims == 1) {
      split_dims.push_back(spec.split_dim);
    } else {
      // Recompute split dim.
      spec.split_dim = -1;
    }
  }
  const Shape& shape = user->operand(0)->shape();
  if (spec.split_dim == -1) {
    for (int64_t dim = 0; dim < shape.dimensions_size(); ++dim) {
      auto offset = user->operand(dim + 1);
      // Skip trivial (1) dimensions or if the index is a constant 0.
      if (shape.dimensions(dim) == 1 ||
          (offset->opcode() == HloOpcode::kConstant &&
           offset->literal().IsZero({}))) {
        continue;
      }
      split_dims.push_back(dim);
      if (spec.split_dim != -1) {
        if (!allow_multiple_split_dims || spec.split_dim != (dim - 1)) {
          VLOG(2) << "Only support split on consecutive dims "
                  << user->ToString();
          return std::nullopt;
        }
        continue;
      }
      spec.split_dim = dim;
    }
  }
  std::vector<int64_t> group_sizes;
  group_sizes.reserve(split_dims.size());
  for (auto dim : split_dims) {
    group_sizes.push_back(user->operand(0)->shape().dimensions(dim) /
                          user->dynamic_slice_sizes()[dim]);
  }

  if (Product(group_sizes) != group_size) {
    VLOG(2) << "Group size mismatch " << user->ToString() << " vs "
            << instruction->ToString();
    return std::nullopt;
  }
  if (split_dims.size() > 1) {
    std::vector<const HloInstruction*> offsets;
    int shard_size = 1;
    for (auto dim : split_dims) {
      offsets.push_back(user->operand(dim + 1));
      shard_size *= user->dynamic_slice_sizes()[dim];
    }
    if (!IsPerIdOffsets(absl::MakeSpan(offsets), shard_size, map_id,
                        group_sizes, instruction, is_cross_module,
                        use_global_device_ids)) {
      VLOG(2) << "IsPerIdOffsets() failed " << instruction->ToString();
      return std::nullopt;
    }
  } else {
    if (!IsPerIdOffset(user->operand(spec.split_dim + 1),
                       user->dynamic_slice_sizes()[spec.split_dim], map_id,
                       group_size, instruction, is_cross_module,
                       use_global_device_ids)) {
      VLOG(2) << "IsPerIdOffset() failed " << instruction->ToString();
      return std::nullopt;
    }
  }

  // If there was a reshape, allow only if the split dims are left unmodified
  // by the reshape. Also rewrite the split dims so that they are in terms of
  // the shape for the all-reduce as opposed to that of the reshape.
  if (reshape) {
    std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
        ShapeUtil::DimensionsUnmodifiedByReshape(reshape->operand(0)->shape(),
                                                 reshape->shape());
    // Map each unmodified output dim of reshape to the corresponding input dim.
    absl::flat_hash_map<int64_t, int64_t> unmodified_output_to_input_map;
    for (const std::pair<int64_t, int64_t>& io_pair : unmodified_dims) {
      unmodified_output_to_input_map.insert({io_pair.second, io_pair.first});
    }

    bool all_split_dims_unmodified =
        absl::c_all_of(split_dims, [&](int64_t out_dim) {
          return unmodified_output_to_input_map.count(out_dim) != 0;
        });
    if (!all_split_dims_unmodified) {
      VLOG(2) << "Split dimensions are modified by reshape";
      return std::nullopt;
    }

    // rewrite the split dim and original_split_dims to be in terms of the
    // shape of the all-reduce.
    spec.split_dim = unmodified_output_to_input_map.at(spec.split_dim);
    for (int64_t& split_dim : split_dims) {
      split_dim = unmodified_output_to_input_map.at(split_dim);
    }
  }

  spec.original_split_dims = split_dims;
  return spec;
}

}  // namespace xla
