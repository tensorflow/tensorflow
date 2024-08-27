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
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Removes redundant select instructions in a SPMD generated module
// which is like following for a forward cycle:
//   fwd_data = ...
//   bwd_data =
//   is_first_device = ...
//   is_last_device = ...
//   data_snd = select(is_last_device, bwd_data, fwd_data)
//   bwd_data_rcv = collective-permute(data_snd), pairs={{3,0}}
//   fwd_data_rcv = collective-permute(data_snd), pairs={{0,1},{1,2},{2,3}}
//   ROOT data_rcv = select(is_first_device, bwd_data_rcv, fwd_data_rcv)
//
// After the transformation, the module will become like following:
//   fwd_data_snd = ...
//   bwd_data_snd = ...
//   is_first_device = ...
//   bwd_data_rcv = collective-permute(bwd_data_snd), pairs={{3,0}}
//   fwd_data_rcv = collective-permute(fwd_data_snd), pairs={{0,1},{1,2},{2,3}}
//   ROOT data_rcv = select(is_first_device, bwd_data_rcv, fwd_data_rcv)
//
// TODO (b/359348622) Further generalize to work with just one
// collective-permute and one select.
class CollectiveSelectFolder : public HloModulePass {
 public:
  absl::string_view name() const override { return "collective-select-folder"; }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COLLECTIVE_SELECT_FOLDER_H_
