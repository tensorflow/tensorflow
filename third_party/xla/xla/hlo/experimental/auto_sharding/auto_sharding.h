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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_cost_graph.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/shape.h"

namespace xla {

class DummyAutoSharding : public HloModulePass {
 public:
  DummyAutoSharding() = default;
  ~DummyAutoSharding() override = default;
  absl::string_view name() const override { return "dummy_auto_sharding"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

enum class AutoShardingResult {
  kModuleUnchanged,
  kModuleChangedShardingPerformed,
  kModuleUnchangedNoShardingPerformed
};

class AutoShardingImplementation {
 public:
  explicit AutoShardingImplementation(const AutoShardingOption& option);
  ~AutoShardingImplementation() = default;

  absl::StatusOr<AutoShardingResult> RunAutoSharding(
      HloModule* module,
      const absl::flat_hash_set<std::string>& replicated_small_tensors,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      const absl::flat_hash_map<std::string, const HloInstruction*>&
          sharding_propagation_solution = {});

  // Returns sharding annotations that need to be preserved in a map (for
  // verification after auto-sharding is done), and removes any sharding
  // anotations that need to be removed.
  std::pair<absl::flat_hash_map<std::string, std::vector<HloSharding>>, bool>
  SaveAndRemoveShardingAnnotation(
      HloModule* module,
      const absl::flat_hash_set<const HloInstruction*>& instructions_to_shard,
      const absl::flat_hash_set<std::string>& replicated_small_tensors,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Canonicalizes entry_computation_layouts by calling
  // module.layout_canonicalization_callback(), which gives canonicalized
  // argument and result layouts based on current module. Currently used by
  // PJRT which assigns layouts based on runtime shapes: see
  // DetermineArgumentLayoutsFromCompileOptions() in
  //     tensorflow/compiler/xla/pjrt/utils.cc
  absl::Status CanonicalizeLayouts(HloModule* module);

  // Returns the optimal objective value that the ILP solver computes.
  double GetSolverOptimalObjectiveValue() {
    return solver_optimal_objective_value_;
  }

 private:
  AutoShardingOption option_;

  // Stores the optimal value of the objective the solver found. This is used to
  // choose the best mesh shape when the try_multiple_mesh_shapes option is on.
  double solver_optimal_objective_value_ = -1.0;
};

class AutoSharding : public HloModulePass {
 public:
  explicit AutoSharding(const AutoShardingOption& option);
  ~AutoSharding() override = default;
  absl::string_view name() const override { return "auto_sharding"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  double GetSolverOptimalObjectiveValue() {
    return solver_optimal_objective_value_;
  }

  std::vector<int64_t> GetChosenDeviceMeshShape() { return chosen_mesh_shape_; }

 private:
  AutoShardingOption option_;
  // Stores the optimal value of the objective the solver found.
  double solver_optimal_objective_value_ = -1.0;
  // Stores the optimal mesh shape found.
  std::vector<int64_t> chosen_mesh_shape_;
};

namespace spmd {
// Function declarations.
// Their comments can be found in their definitions in *.cc files.
HloSharding Tile(const Shape& shape, absl::Span<const int64_t> tensor_dims,
                 absl::Span<const int64_t> mesh_dims,
                 const Array<int64_t>& device_mesh);

std::vector<double> CommunicationReshardingCostVector(
    const StrategyGroup* strategy_group, const Shape& shape,
    const HloSharding& required_sharding,
    const ClusterEnvironment& cluster_env);

std::vector<double> MemoryReshardingCostVector(
    const StrategyGroup* strategy_group, const Shape& operand_shape,
    const HloSharding& required_sharding,
    const ClusterEnvironment& cluster_env);

std::vector<double> FollowInsCostVector(int64_t source_len, int64_t index);

std::unique_ptr<StrategyGroup> CreateLeafStrategyGroup(
    size_t instruction_id, const HloInstruction* ins,
    const StrategyMap& strategy_map, StrategyGroups& strategy_groups);

void SetInNodesWithInstruction(std::unique_ptr<StrategyGroup>& strategy_group,
                               const HloInstruction* ins,
                               const StrategyMap& strategy_map);

void RemoveDuplicatedStrategy(std::unique_ptr<StrategyGroup>& strategy_group);

absl::Status FilterStrategy(const HloInstruction* ins, const Shape& shape,
                            std::unique_ptr<StrategyGroup>& strategy_group,
                            const ClusterEnvironment& cluster_env,
                            const InstructionBatchDimMap& batch_map,
                            const AutoShardingOption& option);

absl::Status HandleDot(std::unique_ptr<StrategyGroup>& strategy_group,
                       StrategyGroups& strategy_groups,
                       StrategyMap& strategy_map, const HloInstruction* ins,
                       size_t instruction_id,
                       const HloInstructionSequence& instruction_sequence,
                       const HloCostAnalysis& hlo_cost_analysis,
                       const ClusterEnvironment& cluster_env,
                       const InstructionBatchDimMap& batch_map,
                       const AutoShardingOption& option,
                       const CallGraph& call_graph);

absl::Status HandleConv(std::unique_ptr<StrategyGroup>& strategy_group,
                        StrategyGroups& strategy_groups,
                        StrategyMap& strategy_map, const HloInstruction* ins,
                        size_t instruction_id,
                        const HloInstructionSequence& instruction_sequence,
                        const HloCostAnalysis& hlo_cost_analysis,
                        const ClusterEnvironment& cluster_env,
                        const InstructionBatchDimMap& batch_map,
                        const AutoShardingOption& option,
                        const CallGraph& call_graph);

void AnnotateShardingWithSimpleHeuristic(HloModule* module,
                                         const std::string& heuristic,
                                         const AliasMap& alias_map,
                                         const ClusterEnvironment& cluster_env);

// Handle alias: alias pairs must have the same HloSharding.
// To deal with alias, we do special process both before and after
// BuildStrategyAndCost. Because it is easier to handle elementwise
// instructions before BuildStrategyAndCost and it is easier to handle
// dot/conv instructions after BuildStrategyAndCost. Before
// BuildStrategyAndCost, we build an AliasMap to guide the generation of
// strategies. After BuildStrategyAndCost, we use AliasSet to add alias
// constraints in the ILP problem.
AliasMap BuildAliasMap(const HloModule* module);

AliasSet BuildAliasSet(const HloModule* module,
                       const StrategyMap& strategy_map);

absl::Status CheckAliasSetCompatibility(const AliasSet& alias_set,
                                        const StrategyGroups& strategy_groups,
                                        const HloInstructionSequence& sequence,
                                        bool crash_on_error);

absl::Status GenerateReduceScatter(
    const HloInstructionSequence& sequence, const AliasMap& alias_map,
    const InstructionDepthMap& depth_map, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const int64_t> s_val,
    const ClusterEnvironment& cluster_env, const AutoShardingOption& option);

bool HasReduceScatterOpportunity(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const int64_t> s_val,
    const StableHashSet<const HloInstruction*>& modified);

HloSharding GetReduceScatterOutput(const HloInstruction* ins,
                                   const ShardingStrategy& strategy,
                                   const ClusterEnvironment& cluster_env);

// The high-level "recipe" for solving an Auto Sharding problem.
AutoShardingSolverResult Solve(
    const HloModule& hlo_module, const HloLiveRange& hlo_live_range,
    const StrategyMap& strategy_map, const StrategyGroups& strategy_groups,
    const CostGraph& cost_graph, const AliasSet& alias_set,
    const std::vector<std::pair<LivenessIdx, LivenessIdx>>& node_intervals,
    const std::vector<std::pair<LivenessIdx, LivenessIdx>>& edge_intervals,
    const std::vector<absl::btree_set<int64_t>>& node_groups,
    const std::vector<absl::btree_set<int64_t>>& edge_groups,
    const AutoShardingOption& option, absl::string_view request_prefix,
    const absl::flat_hash_map<std::string, const HloInstruction*>&
        sharding_propagation_solution = {});

// Populates temporal distance values.
void PopulateTemporalValues(const CostGraph& cost_graph,
                            AutoShardingSolverRequest& request);

void AddReplicatedStrategy(
    const HloInstruction* ins, const Shape& shape,
    const ClusterEnvironment& cluster_env, const StrategyMap& strategy_map,
    std::unique_ptr<StrategyGroup>& strategy_group, double replicated_penalty,
    absl::flat_hash_set<int64_t> operands_to_consider_all_strategies_for = {});

void CheckMemoryCosts(StrategyGroup* strategy_group, const Shape& shape);

// Choose an operand to follow. We choose to follow the operand with the highest
// priority.
std::pair<int64_t, bool> ChooseOperandToFollow(
    const StrategyMap& strategy_map, const InstructionDepthMap& depth_map,
    const AliasMap& alias_map, int64_t max_depth, const HloInstruction* ins);

void FillAllStrategiesForArray(
    std::unique_ptr<StrategyGroup>& strategy_group, const HloInstruction* ins,
    const Shape& shape, const ClusterEnvironment& cluster_env,
    const StrategyMap& strategy_map, const AutoShardingOption& option,
    double replicated_penalty, const InstructionBatchDimMap& batch_dim_map,
    const CallGraph& call_graph, bool only_allow_divisible,
    bool create_replicated_strategies,
    bool create_partially_replicated_strategies);

absl::StatusOr<std::unique_ptr<StrategyGroup>> CreateAllStrategiesGroup(
    const HloInstruction* ins, const Shape& shape, size_t instruction_id,
    StrategyGroups& strategy_groups, const ClusterEnvironment& cluster_env,
    const StrategyMap& strategy_map, const AutoShardingOption& option,
    double replicated_penalty, const InstructionBatchDimMap& batch_dim_map,
    const CallGraph& call_graph, bool only_allow_divisible,
    bool create_replicated_strategies,
    bool create_partially_replicated_strategies);

// Enumerates sharding strategies for elementwise operators by following
// strategies of an operand of the elementwise op.
std::unique_ptr<StrategyGroup> CreateElementwiseOperatorStrategies(
    size_t instruction_id, const HloInstruction* ins,
    const StrategyMap& strategy_map, const ClusterEnvironment& cluster_env,
    const InstructionDepthMap& depth_map, const AliasMap& alias_map,
    const StableHashMap<int64_t, std::vector<ShardingStrategy>>&
        pretrimmed_strategy_map,
    int64_t max_depth, StrategyGroups& strategy_groups,
    AssociativeDotPairs& associative_dot_pairs);

std::unique_ptr<StrategyGroup> HandleManuallyShardedInstruction(
    const HloInstruction* ins, const Shape& shape, size_t instruction_id,
    StrategyGroups& strategy_groups, StrategyMap& strategy_map);

std::unique_ptr<StrategyGroup> HandlePartialReduce(
    const HloInstruction* ins, size_t instruction_id,
    StrategyGroups& strategy_groups, const ClusterEnvironment& cluster_env,
    StrategyMap& strategy_map, const CallGraph& call_graph);

// Factory functions for StrategyGroup.
std::unique_ptr<StrategyGroup> CreateLeafStrategyGroupWithoutInNodes(
    size_t instruction_id, StrategyGroups& strategy_groups);

// Enumerates sharding strategies for reshape operators. The function does so by
// essentially reshaping the sharding of the operand in a manner similar to the
// tensor reshape itself.
std::unique_ptr<StrategyGroup> CreateReshapeStrategies(
    size_t instruction_id, const HloInstruction* ins,
    const StrategyMap& strategy_map, const ClusterEnvironment& cluster_env,
    bool only_allow_divisible, double replicated_penalty,
    const InstructionBatchDimMap& batch_dim_map,
    const AutoShardingOption& option, StrategyGroups& strategy_groups,
    const CallGraph& call_graph);

std::unique_ptr<StrategyGroup> CreateTupleStrategyGroup(size_t instruction_id);

// Enumerate all 1d partition strategies.
void EnumerateAll1DPartition(const HloInstruction* ins, const Shape& shape,
                             const Array<int64_t>& device_mesh,
                             const ClusterEnvironment& cluster_env,
                             const StrategyMap& strategy_map,
                             std::unique_ptr<StrategyGroup>& strategy_group,
                             bool only_allow_divisible,
                             const std::string& suffix,
                             const CallGraph& call_graph);

// Enumerate all partitions recursively.
void EnumerateAllPartition(const HloInstruction* ins, const Shape& shape,
                           const Array<int64_t>& device_mesh,
                           const ClusterEnvironment& cluster_env,
                           const StrategyMap& strategy_map,
                           std::unique_ptr<StrategyGroup>& strategy_group,
                           const InstructionBatchDimMap& batch_dim_map,
                           bool only_allow_divisible,
                           const CallGraph& call_graph,
                           int64_t partition_dimensions,
                           const std::vector<int64_t>& tensor_dims = {});

absl::StatusOr<std::unique_ptr<StrategyGroup>> FollowReduceStrategy(
    const HloInstruction* ins, const Shape& output_shape,
    const HloInstruction* operand, const HloInstruction* unit,
    size_t instruction_id, StrategyMap& strategy_map,
    StrategyGroups& strategy_groups, const ClusterEnvironment& cluster_env,
    bool allow_mixed_mesh_shape, bool crash_at_error);

void GenerateOutfeedStrategy(const HloInstruction* ins, const Shape& shape,
                             const ClusterEnvironment& cluster_env,
                             const StrategyMap& strategy_map,
                             std::unique_ptr<StrategyGroup>& strategy_group,
                             double replicated_penalty);

std::pair<ReshardingCosts, ReshardingCosts>
GenerateReshardingCostsAndMissingShardingsForAllOperands(
    const HloInstruction* ins, const HloSharding& output_sharding,
    const StrategyMap& strategy_map, const ClusterEnvironment& cluster_env,
    const CallGraph& call_graph,
    std::vector<std::optional<HloSharding>>& input_shardings);

std::unique_ptr<StrategyGroup> MaybeFollowInsStrategyGroup(
    const StrategyGroup* src_strategy_group, const Shape& shape,
    size_t instruction_id, bool have_memory_cost,
    StrategyGroups& strategy_groups, const ClusterEnvironment& cluster_env,
    const StableHashMap<NodeIdx, std::vector<ShardingStrategy>>&
        pretrimmed_strategy_map);

void RemoveShardingsWhereSmallDimsShardedAcrossManyDevices(
    const Shape& shape, StrategyGroup* strategy_group,
    bool instruction_has_user_sharding);

void ScaleCostsWithExecutionCounts(StrategyGroup* strategy_group,
                                   int64_t execution_count);

// Existing shardings refer to the HloSharding field in the given
// HloInstruction.
void TrimOrGenerateStrategiesBasedOnExistingSharding(
    const Shape& output_shape, StrategyGroup* strategy_group,
    const StrategyMap& strategy_map,
    const std::vector<HloInstruction*>& instructions,
    const HloSharding& existing_sharding, const ClusterEnvironment& cluster_env,
    StableHashMap<int64_t, std::vector<ShardingStrategy>>&
        pretrimmed_strategy_map,
    const CallGraph& call_graph, bool strict);

// Build possible sharding strategies and their costs for all instructions.
absl::StatusOr<std::tuple<StrategyMap, StrategyGroups, AssociativeDotPairs>>
BuildStrategyAndCost(
    const HloInstructionSequence& sequence, const HloModule* module,
    const absl::flat_hash_set<const HloInstruction*>& instructions_to_shard,
    const absl::flat_hash_map<const HloInstruction*, int64_t>&
        instruction_execution_counts,
    const InstructionDepthMap& depth_map,
    const InstructionBatchDimMap& batch_dim_map, const AliasMap& alias_map,
    const ClusterEnvironment& cluster_env, AutoShardingOption& option,
    const CallGraph& call_graph, const HloCostAnalysis& hlo_cost_analysis,
    bool trying_multiple_mesh_shapes);

// Computes an approximate lower bound on the per-device memory usage of a
// module once it has been sharded. This quantity is multiplied with
// memory_budget_ratio to obtain the memory budget using in our ILP formulation.
int64_t MemoryBudgetLowerBound(
    const HloModule& module,
    const absl::flat_hash_set<const HloInstruction*>& instructions_to_shard,
    const LivenessSet& liveness_set, const HloAliasAnalysis& alias_analysis,
    int64_t num_devices,
    const absl::flat_hash_map<std::string, std::vector<HloSharding>>&
        preserved_shardings);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_H_
