/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_COMPUTATION_DEDUPLICATOR_H_
#define XLA_SERVICE_HLO_COMPUTATION_DEDUPLICATOR_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Deduplicate computations inside a `HloModule`: If two computations are
// identical then keep the first one (in postorder terms) and remove the rest.
class HloComputationDeduplicator : public HloModulePass {
 public:
  // Setting mark_fusion_duplications to true will only process fusions in the
  // HLO. The comparator in this pass will mark duplicate fusions which is
  // needed for groupings in analysis (e.g. Xprof). Currently, the pass
  // doesn't change the HLO if the flag is set to true.
  explicit HloComputationDeduplicator(bool mark_fusion_duplications = false)
      : mark_fusion_duplications_(mark_fusion_duplications) {}
  absl::string_view name() const override { return "computation-deduplicator"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool ContainsLargeConstants(HloComputation* comp);
  bool mark_fusion_duplications_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_COMPUTATION_DEDUPLICATOR_H_
