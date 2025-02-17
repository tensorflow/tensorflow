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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_COMBINER_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_COMBINER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/collective_permute_key.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Combines small non-dependent CollectivePermute ops into larger combined
// CollectivePermute ops. A single combined op is more efficient as each
// collective op has some inherent overhead including kernel launching.
class CollectivePermuteCombiner : public HloModulePass {
 public:
  CollectivePermuteCombiner(int64_t combine_threshold_in_bytes,
                            int64_t combine_threshold_count);

  absl::string_view name() const override {
    return "collective-permute-combiner";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Combine collective permute ops up to this threshold.
  int64_t combine_threshold_in_bytes_;

  // Combine collective permute ops up to this threshold (number of operands).
  int64_t combine_threshold_count_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_COMBINER_H_
