/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALL_GATHER_PAD_DS_SIMPLIFIER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALL_GATHER_PAD_DS_SIMPLIFIER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/shape.h"

namespace xla {

// The offset spec records the position of the all-gather or pad result
// inside the dynamic-slice result.
struct OffsetSpec {
  int64_t start_offset;
  int64_t end_offset;
  int64_t split_dim;

  std::string ToString() const {
    return absl::StrCat("OffsetSpec{ start_offset: ", start_offset,
                        ", end_offset: ", end_offset,
                        ", split_dim: ", split_dim, " }");
  }
};

// Checks and extracts the pad config spec if it matches the pattern for
// optimization.
// The pattern requires:
//  1. Only single edge padding (either low or high), with no interior padding.
//  2. Padding is applied only on the specified `split_dim`.
//  3. The padding value must be zero.
//  4. The AllGather result size must be less than or equal to the DynamicSlice
//     result size on the `split_dim`.
//
// Parameters:
//   pad: The HloPadInstruction to examine.
//   ds_shape: The shape of the DynamicSlice output.
//   ag_shape: The shape of the AllGather output.
//   split_dim: The dimension along which the operations are split.
//
// Returns:
//   An OffsetSpec if the pad instruction matches the criteria, std::nullopt
//   otherwise.
std::optional<OffsetSpec> ExtractValidPadSpec(const HloPadInstruction& pad,
                                              const Shape& ds_shape,
                                              const Shape& ag_shape,
                                              int64_t split_dim);

// Finds the partition ID associated with a given offset in a map.
// The `offset_to_partition_map` stores mappings from start offsets to partition
// IDs. This function returns an iterator to the map element whose offset range
// includes the given `offset`.
//
// Parameters:
//   offset_to_partition_map: A map from start offset to partition ID.
//   offset: The offset to look up.
//
// Returns:
//   An iterator to the correct entry in the map if found, std::nullopt
//   otherwise.
std::optional<OffsetToIdMap::const_iterator> GetPartitionIdForOffset(
    const OffsetToIdMap& offset_to_partition_map, int64_t offset);

// Creates a predicate instruction based on the current partition ID and a
// selection list.
// The resulting predicate is true if the element in `select_list` at the index
// corresponding to the current partition ID is 1, and false otherwise.
//
// Parameters:
//  computation: The HloComputation to which the new instructions will be added.
//  select_list: A vector of integers (0 or 1) indicating selection for each
//               partition.
//
// Returns:
//   A pointer to the newly created HloInstruction (the predicate).
HloInstruction* AddPredInstrBasedOnPartitionIdAndList(
    HloComputation* computation, std::vector<int64_t> select_list);

// A visitor that simplifies all-gather -> dynamic-slice patterns where the
// slices are padded.
class AllGatherPadDsSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status DefaultAction(HloInstruction* hlo_instruction) override {
    return absl::OkStatus();
  }

  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
};

// Replaces a DynamicSlice(Pad(AllGather)) pattern with a more efficient
// sequence of HLOs, primarily using CollectivePermute and Concatenate.
//
// Purpose:
// This pass aims to optimize the case where a sharded AllGather result is
// padded and then sliced by DynamicSlice. Instead of using the large
// padded tensor, it constructs the result for the current partition's slice
// by directly permuting the necessary data from other partitions.
//
// Sample optimization:
// Before:
//   param = f64[1,2,40] parameter(0)
//   all-gather = f64[1,8,40] all-gather(param)
//   pad = f64[1,96,40] pad(all-gather, constant(0)),
//     padding=0_0x0_88x0_0
//   ds_indices = s32[] reshape(s32[1]{0} dynamic-slice(
//     constant({0, 24, 48, 72, 0, 24, 48, 72}), partition-id()))
//   ROOT ds = f64[1,24,40] dynamic-slice(pad, constant(0),
//     ds_indices, s32[] constant(0))
//
// After:
//   compare = pred[] compare(dynamic-slice(
//                            constant({1, 0, 0, 0, 1, 0, 0, 0}),
//                            partition-id()),
//                    constant(1))
//   cp1 = f64[1,2,40] collective-permute(param),
//       source_target_pairs={{5,4},{1,0}}
//   cp2 = f64[1,2,40] collective-permute(param),
//       source_target_pairs={{6,4},{2,0}}
//   cp3 = f64[1,2,40] collective-permute(param),
//       source_target_pairs={{7,4},{3,0}}
//   concatenate = f64[1,24,40] concatenate(
//       param, cp1, cp2, cp3, broadcast(constant(0)))
//   ROOT select = f64[1,24,40] select(compare,
//       concatenate,
//       f64[1,24,40] broadcast(constant(0)))
class AllGatherPadDsSimplifier : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "all-gather-pad-ds-simplifier";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALL_GATHER_PAD_DS_SIMPLIFIER_H_
