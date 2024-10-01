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

// When collective-permute operates on a comparison to a device id
// and the senders match the condition's branch
// we can link collective-permute to the original data skipping the comparison.
// For example
//   condition = broadcast(compare(replica_id, X), direction=EQ
//   data_snd = select(condition, compare_true_data, compare_false_data)
//   rcv = collective-permute(data_snd compare_true_data), pairs={{X,0}}
// can be transformed to
//   rcv = collective-permute(compare_true_data), pairs={{X,0}}
//
// The pass is *only* handling compare direction={EQ,NE}.
// The pass handles Compare with and without preceding Broadcast.
//
// This pass is particularly useful in the pipeline parallelism generated module
// such as:
//   fwd_data = ...
//   bwd_data =
//   is_first_device = ...
//   is_last_device = ...
//   data_snd = select(is_last_device, bwd_data, fwd_data)
//   bwd_data_rcv = collective-permute(data_snd), pairs={{3,0}}
//   fwd_data_rcv = collective-permute(data_snd), pairs={{0,1},{1,2},{2,3}}
//   ROOT data_rcv = select(is_first_device, bwd_data_rcv, fwd_data_rcv)
//
// After the transformation, the module will become:
//   fwd_data_snd = ...
//   bwd_data_snd = ...
//   is_first_device = ...
//   bwd_data_rcv = collective-permute(bwd_data_snd), pairs={{3,0}}
//   fwd_data_rcv = collective-permute(fwd_data_snd), pairs={{0,1},{1,2},{2,3}}
//   ROOT data_rcv = select(is_first_device, bwd_data_rcv, fwd_data_rcv)
class CollectiveSelectFolder : public HloModulePass {
 public:
  absl::string_view name() const override { return "collective-select-folder"; }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_SELECT_FOLDER_H_
