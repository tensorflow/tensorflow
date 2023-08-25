/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_cost_graph.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver_option.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_live_range.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class DummyAutoSharding : public HloModulePass {
 public:
  DummyAutoSharding() = default;
  ~DummyAutoSharding() override = default;
  absl::string_view name() const override { return "dummy_auto_sharding"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

enum class AutoShardingResult {
  kModuleUnchanged,
  kModuleChangedShardingPerformed,
  kModuleUnchangedNoShardingPerfomed
};

class AutoShardingImplementation {
 public:
  explicit AutoShardingImplementation(const AutoShardingOption& option);
  ~AutoShardingImplementation() = default;

  // using HloPassInterface::Run;
  StatusOr<AutoShardingResult> RunAutoSharding(
      HloModule* module,
      const absl::flat_hash_set<std::string>& replicated_small_tensors,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Removes SPMD annotations (if there are) to test AutoSharding on manually
  // annotated graphs.
  StatusOr<bool> RemoveShardingAnnotation(
      HloModule* module,
      const absl::flat_hash_set<std::string>& replicated_small_tensors = {},

      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Canonicalizes entry_computation_layouts by calling
  // module.layout_canonicalization_callback(), which gives canonicalized
  // argument and result layouts based on current module. Currently used by
  // PJRT which assigns layouts based on runtime shapes: see
  // DetermineArgumentLayoutsFromCompileOptions() in
  //     tensorflow/compiler/xla/pjrt/utils.cc
  Status CanonicalizeLayouts(HloModule* module);

  // Returns the optimal objective value that the ILP solver computes
  double GetSolverOptimalObjectiveValue() {
    return solver_optimal_objective_value_;
  }

 private:
  AutoShardingOption option_;

  // Stores the optimal value of the objective the solver found. This is used to
  // chose the best mesh shape when the try_multiple_mesh_shapes option is on.
  double solver_optimal_objective_value_ = -1.0;
};

class AutoSharding : public HloModulePass {
 public:
  explicit AutoSharding(const AutoShardingOption& option);
  ~AutoSharding() override = default;
  absl::string_view name() const override { return "auto_sharding"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
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
// Function declarations
// Their comments can be found in their definitions in *.cc files.
HloSharding Tile(const Shape& shape, absl::Span<const int64_t> tensor_dims,
                 absl::Span<const int64_t> mesh_dims,
                 const Array<int64_t>& device_mesh);

std::vector<double> ReshardingCostVector(const StrategyVector* strategies,
                                         const Shape& shape,
                                         const HloSharding& required_sharding,
                                         const ClusterEnvironment& cluster_env);

std::vector<double> FollowInsCostVector(int64_t source_len, int64_t index);

std::unique_ptr<StrategyVector> CreateLeafStrategyVector(
    size_t instruction_id, const HloInstruction* ins,
    const StrategyMap& strategy_map, LeafStrategies& leaf_strategies);

void SetInNodesWithInstruction(std::unique_ptr<StrategyVector>& strategies,
                               const HloInstruction* ins,
                               const StrategyMap& strategy_map);

void RemoveDuplicatedStrategy(std::unique_ptr<StrategyVector>& strategies);

Status FilterStrategy(const HloInstruction* ins, const Shape& shape,
                      std::unique_ptr<StrategyVector>& strategies,
                      const ClusterEnvironment& cluster_env,
                      const InstructionBatchDimMap& batch_map,
                      const AutoShardingSolverOption& solver_option);

Status HandleDot(std::unique_ptr<StrategyVector>& strategies,
                 LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                 const HloInstruction* ins, size_t instruction_id,
                 const ClusterEnvironment& cluster_env,
                 const InstructionBatchDimMap& batch_map,
                 const AutoShardingSolverOption& solver_option);

Status HandleConv(std::unique_ptr<StrategyVector>& strategies,
                  LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                  const HloInstruction* ins, size_t instruction_id,
                  const ClusterEnvironment& cluster_env,
                  const InstructionBatchDimMap& batch_map,
                  const AutoShardingSolverOption& solver_option);

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

void CheckAliasSetCompatibility(const AliasSet& alias_set,
                                const LeafStrategies& leaf_strategies,
                                const HloInstructionSequence& sequence);

void GenerateReduceScatter(const HloInstructionSequence& sequence,
                           const AliasMap& alias_map,
                           const InstructionDepthMap& depth_map,
                           const StrategyMap& strategy_map,
                           const CostGraph& cost_graph,
                           absl::Span<const int64_t> s_val,
                           const ClusterEnvironment& cluster_env,
                           const AutoShardingSolverOption& solver_option);

bool HasReduceScatterOpportunity(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, absl::Span<const int64_t> s_val,
    const StableHashSet<const HloInstruction*>& modified);

HloSharding GetReduceScatterOutput(const HloInstruction* ins,
                                   const ShardingStrategy& strategy,
                                   const ClusterEnvironment& cluster_env);

// The high-level "recipe" for solving an Auto Sharding problem.
AutoShardingSolverResult Solve(const HloLiveRange& hlo_live_range,
                               const LivenessSet& liveness_set,
                               const StrategyMap& strategy_map,
                               const LeafStrategies& leaf_strategies,
                               const CostGraph& cost_graph,
                               const AliasSet& alias_set,
                               const AutoShardingOption& option);

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_H_
