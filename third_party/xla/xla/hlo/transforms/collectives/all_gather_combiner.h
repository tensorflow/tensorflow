/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_COMBINER_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_COMBINER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Combines small non-dependent AllGather ops into larger combined
// AllGather ops. A typical AllGather implementation has a minimum
// latency-induced time for a AllGather op so a single combined op can be
// more efficient than many small ones.
class AllGatherCombiner : public HloModulePass {
 public:
  AllGatherCombiner(int64_t combine_threshold_in_bytes,
                    int64_t combine_threshold_count, bool combine_by_dim,
                    bool combine_different_dtypes = true,
                    bool combine_while_loops = true);

  absl::string_view name() const override { return "all-gather-combiner"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // The group key encapsulates all of the properties which must match for it to
  // be possible to combine the instructions.
  // The field of the key corresponds to the following:
  // 1. all_gather_dimension
  // 2. domain_metadata_id
  // 3. channel_id
  // 4. use_global_device_ids
  // 5. data_type
  // 6. replica_groups
  // 7. extra arguments in string format.
  using GroupKey =
      std::tuple<std::optional<int64_t>, int64_t, bool, bool, PrimitiveType,
                 std::vector<std::vector<int64_t>>, std::string>;

  static std::string& GetGroupKeyExtraArgs(GroupKey& key);

  // Returns a key that will be equal for instructions that might be combined,
  // or different if not.
  static std::optional<AllGatherCombiner::GroupKey> CombineKey(
      const HloInstruction* instruction, const HloDomainMap& domain_map,
      bool combine_by_dim, bool combine_different_dtypes = true);

 protected:
  absl::StatusOr<bool> RunWithKeyCombiner(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      absl::FunctionRef<std::optional<AllGatherCombiner::GroupKey>(
          const HloInstruction*, const HloDomainMap&, bool, bool)>
          combine_key);

 protected:
  // Combine all gather ops up to this threshold.
  int64_t combine_threshold_in_bytes_;

  // Combine all gather ops up to this threshold (number of operands).
  int64_t combine_threshold_count_;

  // Combine only all-gather ops with the same gather dimension.
  bool combine_by_dim_;

  // Combine all-gather ops with different dtypes.
  bool combine_different_dtypes_;

  // Combine all-gather ops inside while loop bodies.
  bool combine_while_loops_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_COMBINER_H_
