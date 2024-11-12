/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_REDUCE_SCATTER_COMBINER_H_
#define XLA_SERVICE_REDUCE_SCATTER_COMBINER_H_

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
#include "xla/service/all_reduce_key.h"
#include "xla/service/hlo_domain_map.h"

namespace xla {

// Combines small non-dependent ReduceScatter ops into larger combined
// ReduceScatter ops. A typical ReduceScatter implementation has a minimum
// latency-induced time for a ReduceScatter op so a single combined op can be
// more efficient than many small ones.
class ReduceScatterCombiner : public HloModulePass {
 public:
  ReduceScatterCombiner(int64_t combine_threshold_in_bytes,
                        int64_t combine_threshold_count, bool combine_by_dim,
                        bool combine_while_loops = true);

  absl::string_view name() const override { return "reduce-scatter-combiner"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  using GroupKey = std::tuple<AllReduceKey, /*scatter_dimension*/ int64_t,
                              /*extra_args*/ std::string>;

  static std::string& GetGroupKeyExtraArgs(
      ReduceScatterCombiner::GroupKey& key);

  // Returns a key that will be equal for instructions that might be combined,
  // or different if not.
  static std::optional<ReduceScatterCombiner::GroupKey> CombineKey(
      const HloInstruction* instruction, const HloDomainMap& domain_map,
      bool combine_by_dim);

 protected:
  absl::StatusOr<bool> RunWithKeyCombiner(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      absl::FunctionRef<std::optional<ReduceScatterCombiner::GroupKey>(
          const HloInstruction*, const HloDomainMap&, bool)>
          combine_key);

  // Combine reduce-scatter ops up to this threshold.
  int64_t combine_threshold_in_bytes_;

  // Combine reduce-scatter ops up to this threshold (number of operands).
  int64_t combine_threshold_count_;

  // Combine only reduce-scatter ops with the same dimension.
  bool combine_by_dim_;

  // Combine reduce-scatter ops that are inside while loop body computations.
  bool combine_while_loops_;
};

}  // namespace xla

#endif  // XLA_SERVICE_REDUCE_SCATTER_COMBINER_H_
