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

#ifndef XLA_SERVICE_ALL_REDUCE_SPLITTER_H_
#define XLA_SERVICE_ALL_REDUCE_SPLITTER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Rewrites global AR if it is in the form of AR + DS and matches existing
// replica groups into a logical RS followed by AR.
//
// If the pass detects AR followed by DS, then it checks whether
// it is profitable to break it down into a logical RS (but AR + DS still),
// followed by an AR to keep the rewrite numerically equivalent.
//
// Consider following example:
//
// Input program:
//   HloModule m, num_partitions=8
//     p = partition_id()
//     ar = bf16[32] all-reduce(x), replica_groups={{0,1,2,3,4,5,6,7}}
//     ds = dynamic-slice(ar, pointer(partition_id)), dynamic_slice_sizes={8}
//
// There is a global AR performing a reduction over 8 partitions.
// However DS is performing 8-sized slice of a 32-sized tensor which implies
// only 4 distinct slices of a tensor, which further implies 2 replicas of each
// calculated slice. This can be expressed as RS within the replicas followed by
// AR across the replicas. The transformation limits collectives to the data
// that is actually needed for the requested slice.
//
// Output program:
//   HloModule m, num_partitions=8
//     p = partition_id()
//     ar = bf16[32] all-reduce(x), replica_groups={{0,1,2,3},{4,5,6,7}}
//     ds = dynamic-slice(ar, pointer(partition_id)), dynamic_slice_sizes={8}
//     ar.2 = bf16[32] all-reduce(ds), replica_groups={{0,4},{1,5},{2,6},{3,7}}
//
// In addition the pass does the rewrite only if it finds it profitable to do
// so. The profitability function is simple, and just checks whether there are
// any collectives with same replica groups. If there are then the combiner pass
// can pick it up, and fuse it into the same NCCL call.
//
// While the solution is orthogonal to existing known distribution patterns, in
// practice it is profitable for HSDP style communication pattern.
// https://arxiv.org/pdf/2203.11014
//
class AllReduceSplitter : public HloModulePass {
 public:
  absl::string_view name() const override { return "all-reduce-splitter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_ALL_REDUCE_SPLITTER_H_
