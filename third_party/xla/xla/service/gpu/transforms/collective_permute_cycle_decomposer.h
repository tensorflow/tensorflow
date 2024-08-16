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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_PERMUTE_CYCLE_DECOMPOSER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_PERMUTE_CYCLE_DECOMPOSER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// CollectivePermuteCycleDecomposer is a pass that converts CollectivePermute
// instructions with all participants forming either
// 1. a forward cycle  e.g. {{0,1},{1,2},{2,3},{3,0}
// 2. a backward cycle e.g. {{3,2},{2,1},{1,0},{0,3}}
// into two CollectivePermute instructions if the size of the data is above the
// threshold.
//
// We currently restrict this transformation to CollectivePermute using
// partition mode, with one input, without any context data.
//
// **Example**:
// before transformation:
//  %p = u32[] partition-id()
//  ROOT %start = u32[] collective-permute(u32[] %p), channel_id=1,
//    source_target_pairs={{0,1},{1,2},{2,3},{3,0}})
//
// after transformation:
// %partition-id = u32[] partition-id()
//   %constant = u32[] constant(0)
//   %compare = pred[] compare(u32[] %partition-id, u32[] %constant),
//     direction=EQ
//   %broadcast = pred[] broadcast(pred[] %compare), dimensions={}
//   %p = u32[] partition-id()
//   %collective-permute = u32[] collective-permute(u32[] %p), channel_id=1,
//     source_target_pairs={{3,0}})
//   %collective-permute.1 = u32[] collective-permute(u32[] %p), channel_id=2,
//     source_target_pairs={{0,1},{1,2},{2,3}})
//   ROOT %select = u32[] select(pred[] %broadcast, u32[] %collective-permute,
//     u32[] %collective-permute.1)
class CollectivePermuteCycleDecomposer : public HloModulePass {
 public:
  explicit CollectivePermuteCycleDecomposer(int64_t threshold_in_bytes)
      : threshold_in_bytes_(threshold_in_bytes) {}
  absl::string_view name() const override {
    return "collective-permute-cycle-decomposer";
  }

  using HloPassInterface::Run;
  // Runs CollectivePermuteCycleDecomposer pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Transform only if the size of the CollectivePermute data >= threshold.
  int64_t threshold_in_bytes_;
};

}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_PERMUTE_CYCLE_DECOMPOSER_H_
