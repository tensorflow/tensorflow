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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALL_GATHER_PERMUTED_DS_SIMPLIFIER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALL_GATHER_PERMUTED_DS_SIMPLIFIER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "tsl/platform/statusor.h"

namespace xla {

// Visitor for AllGatherDynamicSlicePermutedOffsetSimplifier.
class AllGatherDynamicSlicePermutedOffsetSimplifierVisitor
    : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
};

// A pass that simplifies a pattern of `all-gather` followed by a permuted
// `dynamic-slice` into a single `collective-permute`.
//
// For example:
//
// Before:
//
// ENTRY entry {
//   p = f32[32,8,128] parameter(0)
//   ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
//     dimensions={0}
//   pid = u32[] partition-id()
//   permuted_idx_list = s32[8]{0} constant({224,192,160,128,96,64,32,0})
//   offset = s32[1] dynamic-slice(permuted_idx_list, pid),
//     dynamic_slice_sizes={1}
//   offset_reshape = s32[] reshape(offset)
//   ...
//   ROOT ds = f32[32,8,128] dynamic-slice(ag, offset_reshape, ...),
//     dynamic_slice_sizes={32,8,128}
// }
//
// After:
//
// ENTRY entry {
//   p = f32[32,8,128] parameter(0)
//   ROOT cp = f32[32,8,128] collective-permute(p),
//     source_target_pairs={{0,7},{1,6},{2,5},{3,4},{4,3},{5,2},{6,1},{7,0}}
// }
class AllGatherDynamicSlicePermutedOffsetSimplifier : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "all-gather-to-collective-permute-simplifier";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_ALL_GATHER_PERMUTED_DS_SIMPLIFIER_H_
