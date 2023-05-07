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

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_cost_graph.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver_option.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/tsl/platform/errors.h"
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

static constexpr double kDeviceMeshAlpha = 1.0;
static constexpr double kDeviceMeshBeta = 1.0;

// Options for the autosharding pass
struct AutoShardingOption {
  // Enable the auto sharding pass.
  bool enable = false;

  enum class PreserveShardingsType {
    // AutoSharding constrains the search space using all user shardings.
    kKeepAllShardings,
    // AutoSharding constains the search space using input and output shardings
    // of HloModule's entry computations and remove shardings of all
    // intermediate tensors.
    kKeepInputOutputShardings,
    // Remove all user shardings. This is useful when testing with HLO
    // modules with XLA shardings, so that we can get performance comparison
    // with
    // and without AutoSharding, without changing HLO Modules.
    kRemoveAllShardings
  };

  PreserveShardingsType preserve_shardings =
      PreserveShardingsType::kKeepInputOutputShardings;

  // Simplify the cost graph by merging nodes that should have the same sharding
  // strategy. E.g., an XLAop constructed from an elementwise transformation of
  // another XLAop.
  bool simplify_graph = true;

  // Memory budget (bytes) per device. Default value -1 means no memory budget.
  // Value 0 means setting it to the memory lower bound estimation.
  int64_t memory_budget_per_device = -1;

  // Memory budget =
  //     memory_budget_ratio * (memory lower bound estimation).
  // Enabled when memory_budget_per_device == 0;
  float memory_budget_ratio = 1.1;

  // Overwrite the all gather cost with the input all reduce cost.
  bool force_all_gather_cost = false;
  double all_gather_cost;

  // Overwrite the all gather cost with the input all reduce cost.
  bool force_all_to_all_cost = false;
  double all_to_all_cost;

  // Forcibly split the batch dimension and map it to a mesh dimension.
  // This can force the auto-sharding pass to generate the data parallel
  // strategy.
  int force_batch_dim_to_mesh_dim = -1;

  // If true, allow replicated parameters.
  bool allow_replicated_parameters = true;

  // If true, prefer reduce-scatter + all-gather over all-reduce.
  // A post process will be applied to replace all-reduce with reduce-scater +
  // all-gather if no communication overhead is introduced.
  bool prefer_reduce_scatter = false;

  // If True, generate a gradient-accumulation friendly variant of
  // reduce-scatter
  bool reduce_scatter_grad_acc_friendly = false;

  // If true, aggressively partition more tensors when generating
  // reduce-scatter, even if it introduces more communication.
  bool reduce_scatter_aggressive_partition = false;

  // If true, the batch matmul will always be parallelized on the batch dim in
  // 2d mesh case.
  bool batch_matmul_always_split_batch = false;

  // If true, allow strategies that recompute heavy operators (e.g., dot)
  // to reduce communication.
  bool allow_recompute_heavy_op = false;

  // If true, allow adding 1d strategies in 2d logical mesh.
  bool allow_mixed_mesh_shape = false;

  // The number of micro batches if gradient accumulation is used.
  // If this is not 1, the cost of all-reduce for gradient synchronization
  // is divided by this number.
  int grad_acc_num_micro_batches = 1;

  // If true, load solution vector from PassContext
  bool load_solution_vector = false;

  // If true, N-D sharding (e.g., N maybe be 2 or 3) will be solved in N
  // iterations, where one iteration chooses one tensor dimension to shard. If
  // false, solve N-D sharding directly, i.e., generating all possible sharding
  // strategies for N-D mesh shape.
  bool solve_nd_sharding_iteratively = true;

  // If it is not empty, forcibly use simple heuristic strategies
  // instead of the ILP solver. This is used for ablation study.
  std::string force_simple_heuristic;

  // If true, forcibly set the strategy of some instructions.
  bool force_strategy = false;
  std::vector<int64_t> force_strategy_inst_indices;
  std::vector<std::string> force_strategy_stra_names;

  // Device mesh shape.
  std::vector<int64_t> device_mesh_shape;
  // Device IDs in the mesh.
  std::vector<int64_t> device_mesh_ids;
  // We use an alpha-beta model as the communication model:
  //   latency = alpha + beta * size
  // the following two vectors have the same size as device_mesh_shape and each
  // element models the communication performance along each mesh dimension.
  std::vector<double> device_mesh_alpha;
  std::vector<double> device_mesh_beta;
  // Load the strategy vector instead of solving one.
  bool load_strategy = false;
  // Explore other mesh shapes with the same number of devices as the provided
  // one for a potentially better auto-sharding solution.
  bool try_multiple_mesh_shapes = false;
  std::vector<int64_t> strategy_vector;

  std::string ToString() {
    std::vector<std::string> lines;
    lines.push_back(absl::StrCat("preserve_shardings: ", preserve_shardings));
    lines.push_back(absl::StrCat("simplify_graph: ", simplify_graph));
    if (memory_budget_per_device == -1) {
      lines.push_back("memory_budget_per_device: -1");
    } else {
      lines.push_back(
          absl::StrCat("memory_budget_per_device: ",
                       memory_budget_per_device / (1024 * 1024 * 1024), " GB"));
    }
    lines.push_back(
        absl::StrCat("try_multiple_mesh_shapes: ", try_multiple_mesh_shapes));
    lines.push_back(
        absl::StrCat("force_all_gather_cost: ", force_all_gather_cost));

    if (force_all_gather_cost) {
      lines.push_back(absl::StrCat("all_gather_cost: ", all_gather_cost));
    }
    lines.push_back(
        absl::StrCat("force_all_to_all_cost: ", force_all_to_all_cost));
    if (force_all_to_all_cost) {
      lines.push_back(absl::StrCat("all_to_all_cost: ", all_to_all_cost));
    }
    lines.push_back(absl::StrCat("force_batch_dim_to_mesh_dim: ",
                                 force_batch_dim_to_mesh_dim));
    lines.push_back(absl::StrCat("allow_replicated_parameters: ",
                                 allow_replicated_parameters));
    lines.push_back(
        absl::StrCat("prefer_reduce_scatter: ", prefer_reduce_scatter));
    lines.push_back(absl::StrCat("reduce_scatter_grad_acc_friendly: ",
                                 reduce_scatter_grad_acc_friendly));
    lines.push_back(absl::StrCat("reduce_scatter_aggressive_partition: ",
                                 reduce_scatter_aggressive_partition));
    lines.push_back(absl::StrCat("batch_matmul_always_split_batch: ",
                                 batch_matmul_always_split_batch));
    lines.push_back(
        absl::StrCat("allow_recompute_heavy_op: ", allow_recompute_heavy_op));
    lines.push_back(
        absl::StrCat("allow_mixed_mesh_shape: ", allow_mixed_mesh_shape));
    lines.push_back(absl::StrCat("grad_acc_num_micro_batches: ",
                                 grad_acc_num_micro_batches));
    lines.push_back(
        absl::StrCat("load_solution_vector: ", load_solution_vector));
    lines.push_back(
        absl::StrCat("force_simple_heuristic: ", force_simple_heuristic));
    lines.push_back(absl::StrCat("force_strategy: ", force_strategy));

    if (force_strategy) {
      lines.push_back(
          absl::StrCat("force_strategy_inst_indices: [",
                       absl::StrJoin(force_strategy_inst_indices, ","), "]"));
      lines.push_back(
          absl::StrCat("force_strategy_stra_names: [",
                       absl::StrJoin(force_strategy_stra_names, ","), "]"));
    }

    lines.push_back(absl::StrCat("device_mesh_shape: [",
                                 absl::StrJoin(device_mesh_shape, ","), "]"));
    lines.push_back(absl::StrCat("device_mesh_alpha: [",
                                 absl::StrJoin(device_mesh_alpha, ","), "]"));
    lines.push_back(absl::StrCat("device_mesh_beta: [",
                                 absl::StrJoin(device_mesh_beta, ","), "]"));

    lines.push_back(absl::StrCat("load_strategy: ", load_strategy));
    if (load_strategy) {
      lines.push_back(absl::StrCat("strategy_vector: [",
                                   absl::StrJoin(strategy_vector, ","), "]"));
    }

    return absl::StrJoin(lines, "\n");
  }

  Status CheckAndSetup() {
    if (device_mesh_shape.empty()) {
      return tsl::errors::OutOfRange(
          "device_mesh_shape is empty and it needs to be specified.");
    }
    if (device_mesh_shape.size() > 3) {
      return tsl::errors::OutOfRange(
          absl::StrCat("Not supported: the length of device_mesh_shape is "
                       "greater than 3, actual length: ",
                       device_mesh_shape.size()));
    }
    // All values in device_mesh_shape must be greater than 0.
    if (absl::c_any_of(device_mesh_shape,
                       [](const int64_t i) { return i <= 0; })) {
      return tsl::errors::OutOfRange(
          absl::StrCat("device_mesh_shape values need to be larger than 0: "
                       "device_mesh_shape=",
                       absl::StrJoin(device_mesh_shape, ",")));
    }
    if (spmd::VectorGreaterThanOneElementCount(device_mesh_shape) > 2) {
      return tsl::errors::OutOfRange(
          absl::StrCat("the auto-sharding pass currently does not support ",
                       "more than two shardable dims: device_mesh_shape=",
                       absl::StrJoin(device_mesh_shape, ",")));
    }
    if (device_mesh_alpha.empty()) {
      // Generates simple device_mesh_alpha based on the size of
      // device_mesh_shape.
      device_mesh_alpha =
          std::vector(device_mesh_shape.size(), kDeviceMeshAlpha);
      VLOG(0) << "Using default values for device_mesh_alpha: "
              << absl::StrJoin(device_mesh_alpha, ",");
    }
    if (device_mesh_beta.empty()) {
      // Generates simple device_mesh_beta based on the size of
      // device_mesh_shape.
      device_mesh_beta = std::vector(device_mesh_shape.size(), kDeviceMeshBeta);
      VLOG(0) << "Using default values for device_mesh_beta: "
              << absl::StrJoin(device_mesh_beta, ",");
    }

    // If device_mesh_shape has only one value, append 1 to it
    if (device_mesh_shape.size() == 1) {
      device_mesh_shape.push_back(1);
      device_mesh_alpha.push_back(1.0);
      device_mesh_beta.push_back(1.0);
    }

    if (device_mesh_shape.size() != device_mesh_alpha.size() ||
        device_mesh_shape.size() != device_mesh_beta.size()) {
      return tsl::errors::OutOfRange(absl::StrCat(
          "Sizes do not match: length of device_mesh_shape is ",
          device_mesh_shape.size(), ", length of device_mesh_alpha is ",
          device_mesh_alpha.size(), ", length of device_mesh_beta is ",
          device_mesh_beta.size(),
          ". If not sure how to set device_mesh_alpha and "
          "device_mesh_beta, "
          "please leave them empty and default values will be used."));
    }
    int64_t total_devices = 1;
    for (auto i : device_mesh_shape) {
      total_devices *= i;
    }
    // Set up device_mesh_ids based on device_mesh_shape
    if (device_mesh_ids.empty()) {
      device_mesh_ids = std::vector<int64_t>(total_devices);
      std::iota(device_mesh_ids.begin(), device_mesh_ids.end(), 0);
      VLOG(0) << "Using default values for device_mesh_ids: "
              << absl::StrJoin(device_mesh_ids, ",");
    } else {
      // Checks whether device_mesh_shape and device_mesh_ids are compatible.
      if (total_devices != device_mesh_ids.size()) {
        return tsl::errors::OutOfRange(absl::StrCat(
            "Expect the product of device_mesh_shape to be the same as the "
            "size of device_mesh_ids, but we have total devices = ",
            total_devices,
            " and device_mesh_ids.size()=", device_mesh_ids.size()));
      }
    }
    return OkStatus();
  }
};

class AutoShardingImplementation {
 public:
  explicit AutoShardingImplementation(const AutoShardingOption& option);
  ~AutoShardingImplementation() = default;

  // using HloPassInterface::Run;
  StatusOr<bool> RunAutoSharding(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Removes SPMD annotations (if there are) to test AutoSharding on manually
  // annotated graphs.
  StatusOr<bool> RemoveShardingAnnotation(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Canonicalizes entry_computation_layouts by calling
  // module.layout_canonicalization_callback(), which gives canolicalized
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

 private:
  AutoShardingOption option_;
  // Stores the optimal value of the objective the solver found.
  double solver_optimal_objective_value_ = -1.0;
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
}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_H_
