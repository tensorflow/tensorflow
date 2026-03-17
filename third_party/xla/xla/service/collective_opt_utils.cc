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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {

namespace m = match;

namespace {

// Table lookup is a specific HLO pattern used to retrieve a value from
// a constant array (the "table") using a dynamic index, which is often derived
// from a device's partition-id/replica-id/flattened-id.
// This mechanism allows for flexible, non-arithmetic mappings from a device ID
// to a specific value, such as a memory offset.

// A table lookup consists of:
//  - The "Table": A 1-dimensional constant array (HloOpcode::kConstant)
//    or an HloOpcode::kIota instruction. This array holds the values to
//    be looked up.
//  - The "Lookup": An HloOpcode::kDynamicSlice instruction that extracts
//    an element from the table. The start index for the slice is computed
//    dynamically, often based on a device identifier.

// The compiler can identify this pattern even if it's wrapped by operations
// that don't change the data representation, e.g. kBitcast/kReshape/kCopy.

// Returns true if the given HLO instruction is a table lookup.
bool IsTableLookup(const HloInstruction* hlo) {
  while (hlo->opcode() == HloOpcode::kBitcast ||
         hlo->opcode() == HloOpcode::kReshape ||
         hlo->opcode() == HloOpcode::kCopy) {
    hlo = hlo->operand(0);
  }
  return hlo->opcode() == HloOpcode::kDynamicSlice &&
         (hlo->operand(0)->IsConstant() ||
          hlo->operand(0)->opcode() == HloOpcode::kIota) &&
         hlo->operand(0)->shape().dimensions().size() == 1 &&
         (hlo->operand(0)->shape().element_type() == S32 ||
          hlo->operand(0)->shape().element_type() == U32);
}

std::optional<int64_t> GetScalarInt64Value(const HloInstruction* constant) {
  CHECK_EQ(constant->opcode(), HloOpcode::kConstant);
  CHECK(ShapeUtil::IsEffectiveScalar(constant->shape()));
  absl::InlinedVector<int64_t, 8> multi_index(
      constant->shape().dimensions().size());
  return constant->literal().GetIntegralAsS64(multi_index);
}

// Computes an index into a lookup table for a given device
// ID (partition-id/replica-id/flattened-id) recursively.
// This function resolves an index value that may be computed directly from a
// device ID or indirectly through one or more table lookups.
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

// Infers the tightest bounds on an instruction's result we can at compile time.
std::optional<std::pair<int64_t, int64_t>> GetKnownRange(
    const HloInstruction* hlo) {
  const HloInstruction* cur = hlo;
  while (HloPredicateIsOp<HloOpcode::kConvert, HloOpcode::kReshape,
                          HloOpcode::kBitcast, HloOpcode::kCopy>(cur)) {
    cur = cur->operand(0);
  }
  if (IsTableLookup(cur)) {
    const HloInstruction* table = cur->operand(0);

    if (table->IsConstant()) {
      int64_t size = table->shape().dimensions(0);
      int64_t min_v = std::numeric_limits<int64_t>::max();
      int64_t max_v = std::numeric_limits<int64_t>::min();
      for (int64_t i = 0; i < size; ++i) {
        std::optional<int64_t> v = table->literal().GetIntegralAsS64({i});
        if (!v.has_value()) {
          return std::nullopt;
        }
        min_v = std::min(min_v, *v);
        max_v = std::max(max_v, *v);
      }
      return std::make_pair(min_v, max_v);
    }
  }
  if (cur->IsConstant() && ShapeUtil::IsEffectiveScalar(cur->shape())) {
    std::optional<int64_t> v = GetScalarInt64Value(cur);
    if (v.has_value()) {
      return std::make_pair(*v, *v);
    }
  }
  return std::nullopt;
}

// Backtracks through no-ops to find the "base" instruction responsible for
// some instruction's value.
const HloInstruction* BacktrackToBase(const HloInstruction* instruction) {
  const HloInstruction* cur = instruction;

  while (true) {
    if (HloPredicateIsOp<HloOpcode::kReshape, HloOpcode::kBitcast,
                         HloOpcode::kCopy>(cur)) {
      cur = cur->operand(0);
      continue;
    }

    if (cur->opcode() == HloOpcode::kConvert) {
      if (cur->operand(0)->shape().AreAllLeavesIntegers() &&
          cur->shape().AreAllLeavesIntegers() &&
          primitive_util::BitWidth(cur->operand(0)->shape().element_type()) <=
              primitive_util::BitWidth(cur->shape().element_type())) {
        cur = cur->operand(0);
        continue;
      }
    }

    if (cur->opcode() == HloOpcode::kClamp) {
      // For some clamp ops it's possible to prove they are no-ops at compile
      // time.
      std::optional<int64_t> lower_bound = GetScalarInt64Value(cur->operand(0));
      std::optional<int64_t> upper_bound = GetScalarInt64Value(cur->operand(2));
      std::optional<std::pair<int64_t, int64_t>> range =
          GetKnownRange(cur->operand(1));
      if (lower_bound.has_value() && upper_bound.has_value() &&
          range.has_value() && range->first >= *lower_bound &&
          range->second <= *upper_bound) {
        cur = cur->operand(1);
        continue;
      }
    }

    break;
  }

  return cur;
}

// Checks that `offset` used in dynamic-slice matches the sequential sharding
// across devices within the same replica group.
// Specifically, it checks if the offset for j-th device in a replica group
// is exactly equal to shard_size * j.
// `shard_size` is the dynamic_slice_sizes on split dimension.
// `group_size` is the number of devices in a replica group.
bool IsPerIdOffset(const HloInstruction* offset, int64_t shard_size,
                   const MapIdToTableOffset& map_id, int64_t group_size,
                   const HloInstruction* instruction, bool is_cross_module,
                   bool use_global_device_ids) {
  const bool iota_group = instruction->replica_groups().empty() ||
                          (is_cross_module && !use_global_device_ids);

  offset = BacktrackToBase(offset);

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

  if (offset->opcode() == HloOpcode::kSubtract) {
    // Handle subtraction pattern: (id * slice_size) - table_lookup[id]
    VLOG(2) << "Checking subtraction pattern: " << offset->ToString();

    // Check if the first operand is a multiplication with partition-id
    if (offset->operand(0)->opcode() == HloOpcode::kMultiply) {
      auto* mult = offset->operand(0);
      const HloInstruction* id_operand = nullptr;
      int64_t slice_size = -1;

      // Find which operand is the ID and which is the slice size
      if (mult->operand(0)->IsConstant()) {
        slice_size = *GetScalarInt64Value(mult->operand(0));
        id_operand = mult->operand(1);
      } else if (mult->operand(1)->IsConstant()) {
        slice_size = *GetScalarInt64Value(mult->operand(1));
        id_operand = mult->operand(0);
      }

      if (slice_size > 0 && id_operand) {
        // Check if the second operand is a table lookup
        if (IsTableLookup(offset->operand(1))) {
          VLOG(2) << "Found subtraction pattern with table lookup";

          // Verify that the table lookup uses the same ID as the multiplication
          const HloInstruction* id_base = BacktrackToBase(id_operand);
          const HloInstruction* table_lookup =
              BacktrackToBase(offset->operand(1));
          if (table_lookup->opcode() != HloOpcode::kDynamicSlice) {
            VLOG(2) << "Table lookup is not dynamic slice: "
                    << table_lookup->ToString();
            return false;
          }
          const HloInstruction* table_base =
              BacktrackToBase(table_lookup->operand(0));
          const HloInstruction* table_idx_base =
              BacktrackToBase(table_lookup->operand(1));
          const bool index_matches = id_base == table_idx_base;

          if (index_matches) {
            const int64_t num_groups =
                iota_group ? 1 : instruction->replica_groups().size();

            // For each partition ID, verify the offset calculation
            for (int64_t i = 0; i < num_groups; ++i) {
              for (int64_t j = 0; j < group_size; ++j) {
                int64_t id =
                    iota_group
                        ? j
                        : instruction->replica_groups()[i].replica_ids(j);

                // Get table lookup value - use the original ID operand for
                // mapping
                int64_t table_index = GetIndexForId(table_idx_base, id, map_id);
                if (table_index < 0) {
                  VLOG(2) << "Failed to get table index for ID " << id;
                  return false;
                }
                int64_t table_value;
                if (table_base->opcode() == HloOpcode::kIota) {
                  table_value = table_index;
                } else {
                  std::optional<int64_t> table_value_opt =
                      table_base->literal().GetIntegralAsS64({table_index});
                  if (!table_value_opt.has_value()) {
                    VLOG(2) << "Failed to get table value for index "
                            << table_index;
                    return false;
                  }
                  table_value = *table_value_opt;
                }

                int64_t mult_id = GetIndexForId(id_base, id, map_id);
                if (mult_id < 0) {
                  VLOG(2) << "Failed to get multiply ID for global ID " << id;
                  return false;
                }
                int64_t expected_offset = (mult_id * slice_size) - table_value;

                // Check if this matches the expected pattern for reduce-scatter
                if (expected_offset != shard_size * j) {
                  VLOG(2) << "Subtraction pattern offset " << expected_offset
                          << " doesn't match expected " << (shard_size * j)
                          << " for ID " << id << ", slice size " << slice_size
                          << ", table value " << table_value;
                  return false;
                }
              }
            }

            VLOG(2) << "Subtraction pattern validation successful";
            return true;
          }
        }
      }
    }

    VLOG(2) << "Subtraction pattern not recognized: " << offset->ToString();
    return false;
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
  if (rs_instr->shape().dimensions().size() < min_rank) {
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

int64_t GetGroupSize(bool is_cross_module, bool use_global_device_ids,
                     int64_t num_partitions, int64_t num_replicas,
                     const HloChannelInstruction* instruction) {
  if (!is_cross_module) {
    return instruction->replica_groups().empty()
               ? num_replicas
               : instruction->replica_groups()[0].replica_ids_size();
  }
  if (use_global_device_ids) {
    return instruction->replica_groups()[0].replica_ids_size();
  }
  return num_partitions;
}

MapIdToTableOffset CreateMapIdFn(bool is_cross_module,
                                 bool use_global_device_ids,
                                 int64_t num_partitions, int64_t num_replicas,
                                 const HloChannelInstruction* instruction,
                                 HloPredicate match_partition_id,
                                 HloPredicate match_replica_id) {
  if (!is_cross_module) {
    return [=](const HloInstruction* hlo, int64_t id) {
      return match_replica_id(hlo) ? id : -1;
    };
  }
  if (use_global_device_ids) {
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
    return [=](const HloInstruction* hlo, int64_t id) -> int64_t {
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
      return -1;
    };
  }
  // is_cross_module && !use_global_device_ids
  return [=](const HloInstruction* hlo, int64_t id) {
    return match_partition_id(hlo) ? id : -1;
  };
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
  bool is_cross_module =
      ar->channel_id() && ar->opcode() == HloOpcode::kAllReduce;
  auto spec = MatchWithDynamicSlice(
      ar, num_partitions, num_replicas, allow_multiple_split_dims,
      allow_intervening_reshape, min_rank, match_partition_id, match_replica_id,
      ar->constrain_layout(), ar->use_global_device_ids(), is_cross_module,
      allow_intervening_bitcast);
  return spec;
}

std::optional<ReduceScatterSpec> AllGatherDynamicSliceCancellation(
    const HloAllGatherInstruction* ag, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims,
    bool allow_intervening_reshape, int64_t min_rank,
    HloPredicate match_partition_id, HloPredicate match_replica_id,
    bool allow_intervening_bitcast, bool allow_multiple_users) {
  bool is_cross_module =
      ag->channel_id() && ag->opcode() == HloOpcode::kAllGather;
  auto spec = MatchWithDynamicSlice(
      ag, num_partitions, num_replicas, allow_multiple_split_dims,
      allow_intervening_reshape, min_rank, match_partition_id, match_replica_id,
      ag->constrain_layout(), ag->use_global_device_ids(), is_cross_module,
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

bool IsDynamicSlicingLocalDeviceFromAllGather(
    HloInstruction* ds, HloAllGatherInstruction* all_gather,
    int64_t num_partitions, int64_t num_replicas, bool is_cross_module,
    bool use_global_device_ids) {
  // Check if the user is a DynamicSlice.
  if (ds->opcode() != HloOpcode::kDynamicSlice) {
    return false;
  }

  const HloInstruction* ag_operand = all_gather->operand(0);
  int64_t all_gather_dimension = all_gather->all_gather_dimension();

  int64_t shard_size = ag_operand->shape().dimensions(all_gather_dimension);

  // The offset operand index in DynamicSlice is 1 + dimension index.
  int64_t operand_index = all_gather_dimension + 1;
  if (operand_index >= ds->operand_count()) {
    VLOG(2) << "DynamicSlice operand index is out of bounds.";
    return false;
  }
  const HloInstruction* offset_hlo = ds->operand(operand_index);

  MapIdToTableOffset map_id = CreateMapIdFn(
      is_cross_module, use_global_device_ids, num_partitions, num_replicas,
      all_gather, HloPredicateIsOp<HloOpcode::kPartitionId>,
      HloPredicateIsOp<HloOpcode::kReplicaId>);
  int64_t group_size = GetGroupSize(is_cross_module, use_global_device_ids,
                                    num_partitions, num_replicas, all_gather);

  // IsPerIdOffset checks if the offset for the j-th device (partition_id)
  // is equal to shard_size * j.
  if (IsPerIdOffset(offset_hlo, shard_size, map_id, group_size, all_gather,
                    is_cross_module, use_global_device_ids)) {
    return true;
  }
  HloInstruction* shard_size_constant;
  HloInstruction* offset_constant;
  int64_t dynamic_slice_size_in_ag_dim =
      ds->dynamic_slice_sizes()[all_gather_dimension];
  if (Match(ds, m::DynamicSlice().WithOperand(
                    operand_index,
                    m::Add(m::MultiplyAnyOrder(
                               m::Convert((m::AnyOf<HloInstruction>(
                                   m::PartitionId(), m::ReplicaId()))),
                               m::Constant(&shard_size_constant)),
                           m::Constant(&offset_constant))))) {
    std::optional<int64_t> shard_size_constant_value =
        GetScalarInt64Value(shard_size_constant);
    if (!shard_size_constant_value.has_value() ||
        shard_size_constant_value != shard_size) {
      VLOG(2) << "Constant does not match the shard size: "
              << shard_size_constant->ToString();
      return false;
    }
    std::optional<int64_t> offset_constant_value =
        GetScalarInt64Value(offset_constant);
    if (!offset_constant_value.has_value()) {
      VLOG(2) << "Offset is not a constant: " << offset_constant->ToString();
      return false;
    }
    if (*offset_constant_value + dynamic_slice_size_in_ag_dim <= shard_size) {
      return true;
    }
    VLOG(2) << "Offset is invalid for shard size: " << shard_size << " "
            << "offset_value: " << *offset_constant_value << " "
            << "dynamic_slice_size_in_ag_dim: " << dynamic_slice_size_in_ag_dim;
  }
  VLOG(2) << "The match pattern is not recognized: " << offset_hlo->ToString();
  return false;
}

std::optional<SplitDimSpec> ExtractSplitDimSpec(
    const HloInstruction& dynamic_slice, bool allow_multiple_split_dims) {
  SplitDimSpec spec;
  // First find a single dimension where the input and output of dynamic slice
  // differ.
  int num_dims = 0;
  for (int64_t dim = 0;
       dim < dynamic_slice.operand(0)->shape().dimensions().size(); ++dim) {
    if (dynamic_slice.operand(0)->shape().dimensions(dim) ==
        dynamic_slice.shape().dimensions(dim)) {
      continue;
    }
    num_dims++;
    VLOG(2) << "select dim: " << dim;
    spec.split_dim = dim;
    spec.split_dim_size = dynamic_slice.dynamic_slice_sizes()[dim];
  }
  if (spec.split_dim != -1 && num_dims == 1) {
    // No recomputation needed if dynamic-slice has unique dimension to slice.
    spec.split_dims.push_back(spec.split_dim);
    return spec;
  }
  // Recompute split dim if dynamic-slice has multiple dimensions to slice.
  spec.split_dim = -1;
  const Shape& shape = dynamic_slice.operand(0)->shape();
  for (int64_t dim = 0; dim < shape.dimensions().size(); ++dim) {
    auto offset = dynamic_slice.operand(dim + 1);
    // Skip trivial (1) dimensions or if the index is a constant 0.
    if (shape.dimensions(dim) == 1 ||
        (offset->opcode() == HloOpcode::kConstant &&
         offset->literal().IsZero({}))) {
      continue;
    }
    spec.split_dims.push_back(dim);
    if (spec.split_dim != -1) {
      if (!allow_multiple_split_dims || spec.split_dim != (dim - 1)) {
        VLOG(2) << "Only support split on consecutive dims "
                << dynamic_slice.ToString();
        return std::nullopt;
      }
      continue;
    }
    spec.split_dim = dim;
    spec.split_dim_size = dynamic_slice.dynamic_slice_sizes()[dim];
  }
  return spec;
}

std::optional<PartitionOffsetSpec> ExtractPartitionOffsetSpec(
    const HloAllGatherInstruction* ag, int64_t num_partitions) {
  VLOG(1) << "Extracting partition offset spec for: " << ag->ToString()
          << " with num_partitions = " << num_partitions;
  PartitionOffsetSpec spec;
  int64_t all_gather_shard_size =
      ag->operand(0)->shape().dimensions(ag->all_gather_dimension());
  VLOG(5) << "AG: " << ag->ToString() << ", num_partitions: " << num_partitions
          << ", all_gather_shard_size: " << all_gather_shard_size;
  if (all_gather_shard_size <= 0) {
    VLOG(5) << "AG does not have valid all gather shard size "
            << ag->ToString();
    return std::nullopt;
  }

  if (ag->replica_groups().empty()) {
    VLOG(5) << "AG " << ag->ToString()
            << " has no replica groups, assuming iota.";
    spec.per_replica_group_offsets.resize(1);
    OffsetToIdMap& offset_map = spec.per_replica_group_offsets[0];
    for (int64_t i = 0; i < num_partitions; ++i) {
      int64_t offset = i * all_gather_shard_size;
      int64_t partition_id = i;
      VLOG(5) << "  - group 0, partition_id " << partition_id << " -> offset "
              << offset;
      if (!offset_map.try_emplace(offset, partition_id).second) {
        VLOG(2) << "Duplicate offset " << offset << " in replica group 0"
                << " for partition " << partition_id << " in AG "
                << ag->ToString();
        return std::nullopt;
      }
    }
    VLOG(3) << "Successfully extracted partition offset spec for "
            << ag->ToString();
    return spec;
  }

  spec.per_replica_group_offsets.resize(ag->replica_groups().size());
  for (int64_t group_idx = 0; group_idx < ag->replica_groups().size();
       ++group_idx) {
    VLOG(5) << "Processing replica group " << group_idx;
    const auto& group = ag->replica_groups()[group_idx];
    for (int64_t replica_idx = 0; replica_idx < group.replica_ids_size();
         ++replica_idx) {
      int64_t offset = replica_idx * all_gather_shard_size;
      int64_t partition_id = group.replica_ids(replica_idx);
      VLOG(5) << "  - group " << group_idx << ", partition_id " << partition_id
              << " -> offset " << offset;
      if (spec.per_replica_group_offsets[group_idx].contains(offset)) {
        VLOG(5) << "Duplicate offset " << offset << " in replica group "
                << group_idx << " in AG " << ag->ToString();
        return std::nullopt;
      }
      spec.per_replica_group_offsets[group_idx].emplace(offset, partition_id);
    }
  }
  VLOG(10) << "Successfully extracted partition offset spec.";
  return spec;
}

std::optional<AllGatherDynamicSliceMatchSpec> MatchAllGatherDynamicSliceOffset(
    const HloAllGatherInstruction* ag, const HloInstruction* ds,
    const PartitionOffsetSpec& ag_shard_offset_spec,
    PartitionOffsetSpec& ds_offset_dest_id_spec, int64_t num_partitions) {
  AllGatherDynamicSliceMatchSpec spec;

  if (ag_shard_offset_spec.per_replica_group_offsets.size() !=
      ds_offset_dest_id_spec.per_replica_group_offsets.size()) {
    VLOG(2) << "AG " << ag->ToString() << " and DS " << ds->ToString()
            << " have different replica group sizes: "
            << ag_shard_offset_spec.per_replica_group_offsets.size() << " vs "
            << ds_offset_dest_id_spec.per_replica_group_offsets.size() << ".";
    return std::nullopt;
  }
  for (int i = 0; i < ag_shard_offset_spec.per_replica_group_offsets.size();
       ++i) {
    const auto& ag_offset_map =
        ag_shard_offset_spec.per_replica_group_offsets[i];
    // `rg_local_offset_to_src_id_map` is the offset->source_partition_id map.
    // `indices_spec` is the offset->target_partition_id map.
    absl::flat_hash_map<int64_t, int64_t> rg_local_offset_to_src_id_map;
    for (const auto& [offset, partition_id] : ag_offset_map) {
      if (!rg_local_offset_to_src_id_map.try_emplace(offset, partition_id)
               .second) {
        VLOG(2) << "Duplicate offset " << offset << " in replica group " << i
                << " in AG " << ag->ToString() << " for partition id "
                << partition_id;
        return std::nullopt;
      }
    }

    if (rg_local_offset_to_src_id_map.size() !=
        ds_offset_dest_id_spec.per_replica_group_offsets[i].size()) {
      VLOG(2) << "AG does not have valid partition offset spec "
              << ag->ToString() << " for replica group " << i;
      return std::nullopt;
    }
    for (const auto& [offset, src_id] : rg_local_offset_to_src_id_map) {
      if (!ds_offset_dest_id_spec.per_replica_group_offsets[i].contains(
              offset)) {
        VLOG(2) << "AG does not have valid partition offset spec "
                << ag->ToString() << " for replica group " << i
                << " for DS offset " << ds->ToString();
        return std::nullopt;
      }
      spec.permutation_pairs.push_back(std::make_pair(
          src_id,
          ds_offset_dest_id_spec.per_replica_group_offsets[i].at(offset)));
    }
  }
  return spec;
}

std::optional<AllGatherDynamicSliceMatchSpec>
MatchPermutedSliceAndPartitionOffset(const HloAllGatherInstruction* ag,
                                     int64_t num_partitions,
                                     int64_t num_replicas,
                                     HloPredicate match_partition_id,
                                     bool allow_multiple_users) {
  // Section 1: basic checks.
  // Only matches for multi-partition cases.
  if (num_replicas > 1 || num_partitions <= 1) {
    VLOG(2) << "Only supports single-replica, multi-partition cases, but got "
            << "num_replicas=" << num_replicas
            << ", num_partitions=" << num_partitions << ".";
    return std::nullopt;
  }

  // Only matches for COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID collective mode.
  absl::StatusOr<CollectiveOpGroupMode> mode = GetCollectiveOpGroupMode(ag);

  if (!mode.ok() ||
      mode.value() !=
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID) {
    VLOG(2) << "AG does not use global device ids or channel id "
            << ag->ToString();
    return std::nullopt;
  }

  // Section 2: Extract the dynamic slice using ag.
  std::optional<CollectiveUsers> collective_users =
      FindUniqueDynamicSliceUserFromCollective(
          ag, allow_multiple_users, /*allow_intervening_reshape*/ true,
          /*allow_intervening_bitcast*/ true);

  if (!collective_users.has_value() || !collective_users->dynamic_slice) {
    VLOG(2) << "AG user is not dynamic slice " << ag->ToString();
    return std::nullopt;
  }
  HloInstruction* dynamic_slice = collective_users->dynamic_slice;

  // Extract the split dim spec from ds.
  // check that we only support single split dymension.
  std::optional<SplitDimSpec> split_dim_spec =
      ExtractSplitDimSpec(*dynamic_slice, /*allow_multiple_split_dims*/ false);
  if (!split_dim_spec.has_value() || split_dim_spec->split_dims.size() > 1) {
    VLOG(2) << "Failed to extract a single split dimension from dynamic-slice "
            << dynamic_slice->ToString();
    return std::nullopt;
  }
  // Check the split dimension matches the all-gather dimension
  // and the dynamic-slice split dimension size matches
  // the all-gather shard size.
  if (split_dim_spec->split_dim != ag->all_gather_dimension() ||
      dynamic_slice->shape().dimensions(split_dim_spec->split_dim) !=
          ag->operand(0)->shape().dimensions(ag->all_gather_dimension())) {
    VLOG(2) << "AG does not have valid split dim spec " << ag->ToString()
            << " for DS " << dynamic_slice->ToString() << " on split_dim"
            << split_dim_spec->split_dim;
    return std::nullopt;
  }
  MapIdToTableOffset map_partition_id = [&](const HloInstruction* hlo,
                                            int64_t id) {
    return HloPredicateIsOp<HloOpcode::kPartitionId>(hlo) ? id : -1;
  };

  VLOG(0) << "dynamic slice: " << dynamic_slice->ToString()
          << " split dim: " << split_dim_spec->split_dim;
  // Section 3: extract the offset spec from dynamic slice and ag.
  std::optional<PartitionOffsetSpec> ds_offset_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          ag, dynamic_slice->operand(split_dim_spec->split_dim + 1),
          map_partition_id, split_dim_spec->split_dim_size);
  if (!ds_offset_spec.has_value()) {
    VLOG(2) << "AG does not have valid indices spec " << ag->ToString()
            << " for DS " << dynamic_slice->ToString() << " on split_dim"
            << split_dim_spec->split_dim;
    return std::nullopt;
  }

  std::optional<PartitionOffsetSpec> ag_offset_spec =
      ExtractPartitionOffsetSpec(ag, num_partitions);

  if (!ag_offset_spec.has_value()) {
    VLOG(2) << "AG does not have valid partition offset spec " << ag->ToString()
            << " for num_partitions " << num_partitions;
    return std::nullopt;
  }

  // Section 4: match the offset spec from dynamic slice and ag.
  return MatchAllGatherDynamicSliceOffset(
      ag, dynamic_slice, ag_offset_spec.value(), ds_offset_spec.value(),
      num_partitions);
}

bool CheckUniformReplicaGroups(const HloChannelInstruction* instruction) {
  CHECK_NE(instruction, nullptr);
  if (instruction->replica_groups().size() <= 1) {
    return true;
  }
  const int64_t size = instruction->replica_groups().front().replica_ids_size();
  absl::Span<const ReplicaGroup> rgs = instruction->replica_groups();
  return absl::c_all_of(rgs.subspan(1), [size](const ReplicaGroup& group) {
    return group.replica_ids_size() == size;
  });
}

std::optional<CollectiveUsers> FindUniqueDynamicSliceUserFromCollective(
    const HloChannelInstruction* absl_nonnull instruction,
    bool allow_multiple_users, bool allow_intervening_reshape,
    bool allow_intervening_bitcast) {
  if (instruction->user_count() == 0) {
    return std::nullopt;
  }

  HloInstruction* user = instruction->users()[0];
  if (allow_multiple_users) {
    for (HloInstruction* some_user : instruction->users()) {
      if ((allow_intervening_reshape &&
           some_user->opcode() == HloOpcode::kReshape) ||
          some_user->opcode() == HloOpcode::kDynamicSlice) {
        user = some_user;
        break;
      }
    }
  }

  CollectiveUsers result;
  if (allow_intervening_reshape) {
    if (user->opcode() == HloOpcode::kReshape) {
      if (user->user_count() != 1) {
        VLOG(2) << "Reshape user count > 1 for " << user->ToString();
        return std::nullopt;
      }
      result.reshape = user;
      user = user->users().front();
    }
  }

  if (allow_intervening_bitcast) {
    if (user->opcode() == HloOpcode::kBitcast) {
      if (user->user_count() != 1) {
        VLOG(2) << "Bitcast user count > 1 for " << user->ToString();
        return std::nullopt;
      }
      result.bitcast = user;
      user = user->users().front();
    }
  }

  if (user->opcode() == HloOpcode::kDynamicSlice) {
    result.dynamic_slice = user;
    return result;
  }
  return std::nullopt;
}

std::optional<ReduceScatterSpec> MatchWithDynamicSlice(
    const HloChannelInstruction* instruction, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims,
    bool allow_intervening_reshape, int64_t min_rank,
    HloPredicate match_partition_id, HloPredicate match_replica_id,
    bool is_constrain_layout, bool use_global_device_ids, bool is_cross_module,
    bool allow_intervening_bitcast, bool allow_multiple_users) {
  if (!instruction->shape().IsArray() || is_constrain_layout) {
    VLOG(2) << "Unsupported collective: " << instruction->ToString();
    return std::nullopt;
  }
  if (instruction->shape().dimensions().size() -
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
  if (!CheckUniformReplicaGroups(instruction)) {
    VLOG(2) << "Non-uniform replica groups " << instruction->ToString();
    return std::nullopt;
  }

  std::optional<CollectiveUsers> ds_user =
      FindUniqueDynamicSliceUserFromCollective(
          instruction, allow_multiple_users, allow_intervening_reshape,
          allow_intervening_bitcast);

  if (!ds_user.has_value()) {
    VLOG(2) << "AG or AR user is not dynamic slice " << instruction->ToString();
    return std::nullopt;
  }

  HloInstruction* user = ds_user->dynamic_slice;
  HloInstruction* reshape = ds_user->reshape;

  MapIdToTableOffset map_id = CreateMapIdFn(
      is_cross_module, use_global_device_ids, num_partitions, num_replicas,
      instruction, match_partition_id, match_replica_id);
  std::optional<int64_t> optional_group_size =
      GetGroupSize(is_cross_module, use_global_device_ids, num_partitions,
                   num_replicas, instruction);
  if (!optional_group_size) {
    return std::nullopt;
  }
  int64_t group_size = *optional_group_size;

  ReduceScatterSpec spec;
  spec.dynamic_slice = user;
  if (!is_cross_module) {
    spec.sharded_replicas = num_replicas;
  } else if (use_global_device_ids) {
    spec.sharded_replicas = num_replicas;
    spec.sharded_partitions = num_partitions;
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
  }
  if (group_size < 2) {
    VLOG(2) << "Group_size < 2, nothing to do " << instruction->ToString();
    return std::nullopt;
  }
  spec.group_size = group_size;
  CHECK_NE(user, nullptr);
  std::optional<SplitDimSpec> split_dim_spec =
      ExtractSplitDimSpec(*user, allow_multiple_split_dims);
  if (!split_dim_spec) {
    return std::nullopt;
  }
  spec.split_dim = split_dim_spec->split_dim;
  std::vector<int64_t> split_dims = std::move(split_dim_spec->split_dims);
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
    for (const auto& io_pair : unmodified_dims) {
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

// Generates a new PartitionOffsetSpec where the offsets are multiplied by the
// multiplier.
PartitionOffsetSpec GenerateMultipliedPartitionOffsetSpec(
    const PartitionOffsetSpec& offset_partition_spec, int64_t multiplier) {
  PartitionOffsetSpec multiplied_indices_spec;
  multiplied_indices_spec.per_replica_group_offsets.resize(
      offset_partition_spec.per_replica_group_offsets.size());
  for (int64_t rg_idx = 0;
       rg_idx < offset_partition_spec.per_replica_group_offsets.size();
       ++rg_idx) {
    for (const auto& [offset, partition_id] :
         offset_partition_spec.per_replica_group_offsets[rg_idx]) {
      multiplied_indices_spec
          .per_replica_group_offsets[rg_idx][offset * multiplier] =
          partition_id;
    }
  }
  return multiplied_indices_spec;
}

std::optional<PartitionOffsetSpec> GetIndicesSpecForDynamicSlice(
    const HloAllGatherInstruction* absl_nonnull ag_instr,
    const HloInstruction* absl_nonnull offset_hlo,
    const std::function<int64_t(const HloInstruction*, int64_t)>& map_id) {
  if (!ag_instr || !offset_hlo) {
    return std::nullopt;
  }
  PartitionOffsetSpec indices_spec;
  if (ag_instr->replica_groups().empty()) {
    return std::nullopt;
  }
  indices_spec.per_replica_group_offsets.resize(
      ag_instr->replica_groups().size());

  if (!IsTableLookup(offset_hlo)) {
    return std::nullopt;
  }
  VLOG(2) << "GetIndicesSpecForDynamicSlice: offset_hlo is a table lookup, "
             "table operand: "
          << offset_hlo->operand(0)->ToString()
          << " offset_hlo: " << offset_hlo->ToString();

  while (offset_hlo->opcode() == HloOpcode::kBitcast ||
         offset_hlo->opcode() == HloOpcode::kReshape ||
         offset_hlo->opcode() == HloOpcode::kCopy) {
    offset_hlo = offset_hlo->operand(0);
  }
  CHECK_EQ(offset_hlo->opcode(), HloOpcode::kDynamicSlice);
  for (int64_t group_idx = 0; group_idx < ag_instr->replica_groups().size();
       ++group_idx) {
    const ReplicaGroup& group = ag_instr->replica_groups()[group_idx];
    for (int64_t partition_id : group.replica_ids()) {
      if (offset_hlo->operand_count() < 2) {
        VLOG(2) << "offset_hlo->operand_count() is "
                << offset_hlo->operand_count();
        return std::nullopt;
      }
      int64_t table_index =
          GetIndexForId(offset_hlo->operand(1), partition_id, map_id);
      VLOG(0) << "offset_hlo: " << offset_hlo->ToString();
      VLOG(0) << "table_index: " << table_index;
      VLOG(0) << offset_hlo->operand(0)->literal().ToString();
      if (table_index < 0) {
        VLOG(2) << "Failed to infer table index from "
                << offset_hlo->operand(1);
        return std::nullopt;
      }

      int64_t slice_offset;
      if (offset_hlo->operand(0)->opcode() == HloOpcode::kIota) {
        slice_offset = table_index;
      } else {
        slice_offset =
            *offset_hlo->operand(0)->literal().GetIntegralAsS64({table_index});
      }
      VLOG(0) << "slice_offset: " << slice_offset;
      if (!indices_spec.per_replica_group_offsets[group_idx]
               .try_emplace(slice_offset, partition_id)
               .second) {
        VLOG(2) << "slice_offset:" << slice_offset
                << " already exists in the map.";
        return std::nullopt;
      }
    }
  }

  return indices_spec;
}

std::optional<PartitionOffsetSpec> GetIndicesSpecForDynamicSliceWithMultiply(
    const HloAllGatherInstruction* absl_nonnull ag_instr,
    const HloInstruction* absl_nonnull offset_hlo,
    const std::function<int64_t(const HloInstruction*, int64_t)>& map_id,
    int64_t split_dim_size) {
  if (!ag_instr || !offset_hlo) {
    return std::nullopt;
  }
  if (ag_instr->replica_groups().empty()) {
    return std::nullopt;
  }
  // Traverses up the instruction graph for layout instructions.
  while (offset_hlo->opcode() == HloOpcode::kBitcast ||
         offset_hlo->opcode() == HloOpcode::kReshape ||
         offset_hlo->opcode() == HloOpcode::kCopy) {
    offset_hlo = offset_hlo->operand(0);
  }
  if (offset_hlo->opcode() == HloOpcode::kMultiply) {
    if (!ShapeUtil::IsEffectiveScalar(offset_hlo->shape())) {
      VLOG(2) << "Offset is not a scalar " << offset_hlo->ToString();
      return std::nullopt;
    }
    int64_t const_operand_idx = -1;
    if (offset_hlo->operand(0)->IsConstant()) {
      const_operand_idx = 0;
    } else if (offset_hlo->operand(1)->IsConstant()) {
      const_operand_idx = 1;
    } else {
      VLOG(2) << "Offset is not multiple(const, ...) "
              << offset_hlo->ToString();
      return std::nullopt;
    }
    std::optional<int64_t> multiplier =
        GetScalarInt64Value(offset_hlo->operand(const_operand_idx));

    if (!multiplier || split_dim_size % *multiplier != 0) {
      VLOG(2)
          << "Multiplier is unknown or cannot evenly divide split_dim_size: "
          << split_dim_size << " multiplier:" << *multiplier << "offset_hlo:"
          << offset_hlo->operand(const_operand_idx)->ToString();
      return std::nullopt;
    }

    VLOG(10) << "detected valid multiplier: " << *multiplier;
    std::optional<PartitionOffsetSpec> offset_partition_spec =
        GetIndicesSpecForDynamicSlice(
            ag_instr, offset_hlo->operand(1 - const_operand_idx), map_id);
    if (!offset_partition_spec.has_value()) {
      VLOG(2) << "Failed to get indices spec for dynamic slice "
              << offset_hlo->operand(1 - const_operand_idx)->ToString();
      return std::nullopt;
    }
    PartitionOffsetSpec multiplied_spec = GenerateMultipliedPartitionOffsetSpec(
        offset_partition_spec.value(), *multiplier);
    return multiplied_spec;
  }
  // Falls back to non-multiply handling when multiply is not found.
  return GetIndicesSpecForDynamicSlice(ag_instr, offset_hlo, map_id);
}

bool MatchDsPadAllGather(HloInstruction* ds_hlo, HloInstruction** pad_hlo,
                         HloInstruction** ag_hlo) {
  namespace m = ::xla::match;
  return Match(ds_hlo,
               m::DynamicSlice().WithOperand(
                   0, m::Pad(pad_hlo, m::AllGather(ag_hlo), m::Constant())));
}

const HloInstruction* FindCanonicalSendRecvStartOp(const HloInstruction* hlo) {
  CHECK(hlo->opcode() == HloOpcode::kSend ||
        hlo->opcode() == HloOpcode::kRecv ||
        hlo->opcode() == HloOpcode::kSendDone ||
        hlo->opcode() == HloOpcode::kRecvDone);
  // If the instruction is wrapped in an async computation, return
  // the instruction itself.
  if (hlo->parent()->IsAsyncComputation()) {
    return hlo;
  }

  // Find container while loop and index for the send/recv case or
  // return canonical start op directly.
  const HloInstruction* while_op = nullptr;
  int64_t i = -1;
  if (hlo->opcode() == HloOpcode::kSend || hlo->opcode() == HloOpcode::kRecv) {
    CHECK_EQ(hlo->users().size(), 1);
    const HloInstruction* unique_user = hlo->users().front();

    // Return send/recv inst directly if this is a simple send/recv
    // pair.
    if (unique_user->opcode() == HloOpcode::kSendDone ||
        unique_user->opcode() == HloOpcode::kRecvDone) {
      return hlo;
    }

    // Find while loop and index, otherwise.
    CHECK(unique_user->opcode() == HloOpcode::kTuple ||
          unique_user->opcode() == HloOpcode::kWhile);
    if (unique_user->IsRoot()) {
      // send/recv op in the loop body.
      auto maybe_while_op =
          unique_user->parent()->GetUniqueCaller(HloOpcode::kWhile);
      CHECK(maybe_while_op);
      while_op = *maybe_while_op;
      i = unique_user->operand_index(hlo);
    } else {
      // send/recv leading into the loop.
      CHECK_EQ(unique_user->users().size(), 1);
      CHECK(unique_user->users().front()->opcode() == HloOpcode::kWhile);
      while_op = unique_user->users().front();
      i = unique_user->operand_index(hlo);
    }
  }

  // Find container while loop and index for the send-done/recv-done
  // case or return canonical start op directly.
  if (hlo->opcode() == HloOpcode::kSendDone ||
      hlo->opcode() == HloOpcode::kRecvDone) {
    const HloInstruction* operand = hlo->operand(0);

    // Return send/recv hlo directly if this is a simple send/recv
    // pair.
    if (operand->opcode() == HloOpcode::kSend ||
        operand->opcode() == HloOpcode::kRecv) {
      return operand;
    }

    // Find while loop and index, otherwise.
    CHECK(operand->opcode() == HloOpcode::kGetTupleElement);
    const auto* gte = Cast<HloGetTupleElementInstruction>(operand);
    const HloInstruction* iter_tuple = operand->operand(0);
    if (iter_tuple->opcode() == HloOpcode::kParameter) {
      // send-done/recv-done in the loop body.
      CHECK(Cast<HloParameterInstruction>(iter_tuple)->parameter_number() == 0);
      auto maybe_while =
          iter_tuple->parent()->GetUniqueCaller(HloOpcode::kWhile);
      CHECK(maybe_while);
      while_op = *maybe_while;
      i = gte->tuple_index();
    } else {
      // send-done/recv-done proceeding the loop.
      CHECK(iter_tuple->opcode() == HloOpcode::kWhile);
      while_op = iter_tuple;
      i = gte->tuple_index();
    }
  }

  // Extract canonical start op from while loop's init.
  CHECK(while_op != nullptr);
  CHECK(0 <= i && i < while_op->shape().tuple_shapes().size());
  const HloInstruction* init = while_op->operand(0);
  const HloInstruction* canonical_start_op = init->operand(i);
  CHECK(canonical_start_op->opcode() == HloOpcode::kSend ||
        canonical_start_op->opcode() == HloOpcode::kRecv);
  return canonical_start_op;
}

}  // namespace xla
