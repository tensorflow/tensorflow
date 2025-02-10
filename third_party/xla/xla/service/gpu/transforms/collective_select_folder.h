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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_SELECT_FOLDER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_SELECT_FOLDER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// If a collective-permute selects its source data based on a partition or
// replica ID and we can prove that the condition is either always true or
// always false, we can fold the redundant select op and use the correct source
// data directly.
//
// Example:
//
//   condition = compare(replica-id(), X), direction=EQ
//   snd_data = select(condition, true_data, false_data)
//   rcv_data = collective-permute(snd_data), source_target_pairs={{X,0}}
//
// The condition is always true for the only relevant replica X and the IR can
// be folded into
//
//   rcv_data = collective-permute(true_data), source_target_pairs={{X,0}}
//
// The pass only supports simple partion/replica-based predicates, comparing
// partition/replica-id with a constant. Only comparison directions {EQ,NE} are
// supported. The predicate may be broadcasted.
//
// This pass is motivated by pipeline parallelism, where it removes undesired
// data dependencies.
//
// Example:
//
//   fwd_data = ...
//   bwd_data =
//   is_first_device = ...
//   is_last_device = ...
//   snd_data = select(is_last_device, bwd_data, fwd_data)
//   rcv_bwd_data = collective-permute(snd_data),
//       source_target_pairs={{LAST_ID,0}}
//   rcv_fwd_data = collective-permute(snd_data),
//       source_target_pairs={{0,1},{1,2},...,{LAST_ID,0}}
//   ROOT rcv_data = select(is_first_device, rcv_bwd_data, rcv_fwd_data)
//
// The select can be removed on both paths resulting in
//
//   fwd_data = ...
//   bwd_data =
//   is_first_device = ...
//   is_last_device = ...
//   rcv_bwd_data = collective-permute(bwd_data),
//       source_target_pairs={{LAST_ID,0}}
//   rcv_fwd_data = collective-permute(fwd_data),
//       source_target_pairs={{0,1},{1,2},...,{LAST_ID,0}}
//   ROOT rcv_data = select(is_first_device, rcv_bwd_data, rcv_fwd_data)
//
class CollectiveSelectFolder : public HloModulePass {
 public:
  absl::string_view name() const override { return "collective-select-folder"; }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_SELECT_FOLDER_H_
