/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SHARDING_PROPAGATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SHARDING_PROPAGATION_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/custom_call_sharding_helper.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Remove Sharding custom-call instruction by folding the sharding attribute
// to its operand. If the operand already has a different sharding, insert a
// copy node for reshard.
// partially_specified will be populated with the converted copies if the custom
// call is partially specified.
StatusOr<bool> ProcessShardingInstruction(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    bool replace_sharding_with_copy,
    absl::flat_hash_map<const HloInstruction*, std::vector<int64_t>>*
        unspecified_dims);

int64_t ComputeNonRootUsers(const HloInstruction* instr);

// Infers broadcast ops' operand sharding, based on its output sharding.
std::optional<HloSharding> InferBroadcastOperandSharding(
    const HloInstruction& instruction, bool is_spmd = true);

bool InferReduceShardingFromOperand(HloInstruction* instruction,
                                    bool may_combine_partial_sharding,
                                    bool is_spmd);

// Propagates sharding information around the graph. HLOs that have shardings
// are kept as-is, those that do not have shardings are given shardings based on
// a simple local greedy heuristic.
class ShardingPropagation : public HloModulePass {
 public:
  using ComputationMap =
      absl::flat_hash_map<const HloComputation*, HloInstruction*>;
  explicit ShardingPropagation(
      bool is_spmd = false, bool propagate_metadata = false,
      bool allow_spmd_sharding_propagation_to_output = false,
      bool cse_prevention_only = false,
      std::unique_ptr<CustomCallShardingHelper> sharding_helper = nullptr)
      : is_spmd_(is_spmd),
        propagate_metadata_(propagate_metadata),
        allow_spmd_sharding_propagation_to_output_(
            allow_spmd_sharding_propagation_to_output),
        cse_prevention_only_(cse_prevention_only) {
    if (sharding_helper) {
      sharding_helper_ = std::move(sharding_helper);
    } else {
      sharding_helper_ = std::make_unique<CustomCallShardingHelper>();
    }
  }
  absl::string_view name() const override { return "sharding-propagation"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Function which can be used to apply a spatially partitioned sharding onto a
  // given domain. It will apply the sharding into the exit edges of the domain
  // and then rely on the rest of sharding propagation to ensure that the
  // intermediate nodes get the correct sharding.
  static Status NormalizeDomain(const DomainMetadata::Domain& domain,
                                const DomainMetadata* metadata);

  static std::optional<HloSharding> GetShardingFromUser(
      const HloInstruction& instruction, const HloInstruction& user,
      int64_t aggressiveness, bool is_spmd, const CallGraph& call_graph);

  // Canonicalizes entry_computation_layouts by calling
  // module.layout_canonicalization_callback(), which gives canolicalized
  // argument and result layouts based on current module. Currently used by
  // PJRT which assigns layouts based on runtime shapes: see
  // DetermineArgumentLayoutsFromCompileOptions() in
  //     tensorflow/compiler/xla/pjrt/utils.cc
  Status CanonicalizeLayouts(HloModule* module);

 private:
  bool InferShardingFromOperands(HloInstruction* instruction,
                                 const ComputationMap& computation_map,
                                 int64_t aggressiveness,
                                 const CallGraph& call_graph);

  std::unique_ptr<CustomCallShardingHelper> sharding_helper_;
  bool is_spmd_;
  bool propagate_metadata_;
  bool allow_spmd_sharding_propagation_to_output_;
  // If true, the pass keeps the propagation results only on selected
  // instructions to prevent CSE across unrelated subgraphs. (A common case is
  // scalar broadcasts).
  bool cse_prevention_only_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SHARDING_PROPAGATION_H_
