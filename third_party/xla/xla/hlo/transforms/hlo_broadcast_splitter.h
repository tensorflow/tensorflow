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
#ifndef XLA_HLO_TRANSFORMS_HLO_BROADCAST_SPLITTER_H_
#define XLA_HLO_TRANSFORMS_HLO_BROADCAST_SPLITTER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Splits the broadcast instructions such that they have a single user. This
// aggressively duplicates all broadcasts and relies on DCE to clean up the
// duplicates after propagation and partitioning.
class HloBroadcastSplitter : public HloModulePass {
 public:
  HloBroadcastSplitter() = default;
  absl::string_view name() const override { return "hlo-broadcast-splitter"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_HLO_BROADCAST_SPLITTER_H_
