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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_OPTION_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_OPTION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"

namespace xla {

static constexpr double kDeviceMeshAlpha = 1.0;
static constexpr double kDeviceMeshBeta = 1.0;
static constexpr double kOverbudgetCoeff = 1e6;

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

  // Controls the penalty associated with violating memory constraints; if
  // negative, the memory budget is instead imposed as a hard constraint.
  float memory_overbudget_coeff = kOverbudgetCoeff;

  // Overwrite the all gather cost with the input all reduce cost.
  bool force_override_all_gather_cost = false;
  double all_gather_cost = 0;

  // Overwrite the all gather cost with the input all reduce cost.
  bool force_override_all_to_all_cost = false;
  double all_to_all_cost = 0;

  // Overwrite the all gather cost with the input all reduce cost.
  bool force_override_all_reduce_cost = false;
  double all_reduce_cost = 0;

  // Overwrite the all gather cost with the input all reduce cost.
  bool force_override_reduce_scatter_cost = false;
  double reduce_scatter_cost = 0;

  // Forcibly split the batch dimension and map it to a mesh dimension.
  // This can force the auto-sharding pass to generate the data parallel
  // strategy.
  int force_batch_dim_to_mesh_dim = -1;

  // If true, allow replicated parameters.
  bool allow_replicated_parameters = true;

  // If true, prefer reduce-scatter + all-gather over all-reduce.
  // A post process will be applied to replace all-reduce with reduce-scatter +
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

  // If true, allow strategies that recompute heavy operators (e.g., dot) to
  // reduce communication. This will generate generate replicated or partially
  // replicated strategies for dot/conv ops. Generating these seems to be
  // beneficial for LLM serving models, but can increase the search space, so
  // this feature is exposed as an option.
  bool allow_recompute_heavy_op = true;

  // If true, allow adding 1d strategies in 2d logical mesh.
  bool allow_mixed_mesh_shape = false;

  // The number of micro batches if gradient accumulation is used.
  // If this is not 1, the cost of all-reduce for gradient synchronization
  // is divided by this number.
  int grad_acc_num_micro_batches = 1;

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

  // Whether or not we allow sharding strategies where the tensor dim is
  // indivisible by the #tiles in that dimension.
  bool only_allow_divisible_input_output = true;
  bool only_allow_divisible_intermediate = false;

  // If true, strictly limit the following iterations to use the same number of
  // shards for sharded tensor dimensions; if false, the following iterations
  // can choose different number of shards for sharded tensor dimensions.
  // Enabling it can hurt the performance of dot ops, but can make the search
  // space more scalable. Therefore leaving it as an option.
  bool nd_sharding_iteratively_strict_search_space = false;

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

  // Explore other mesh shapes with the same number of devices as the provided
  // one for a potentially better auto-sharding solution.
  bool try_multiple_mesh_shapes = false;

  // Timeout for the solver. If the solver fails to find an optimal solution
  // before the timeout, we rely on the heuristic-based sharding implemented in
  // sharding_propagation.cc.
  int64_t solver_timeout_in_seconds = 3600;

  // Static estimate for iteration count of a while loop, used in the cost
  // model. This estimate is used when we cannot infer an upper bound on the
  // number of iterations in the loop (as implemented in
  // third_party/tensorflow/compiler/xla/service/while_loop_analysis.h)
  int64_t loop_iteration_count_estimate = 100;

  // Allows the conversion of aliases to followers if their pairwise strategy
  // compatibilities are embodied by the identity matrix (which makes for a
  // smaller Mixed ILP).
  bool allow_alias_to_follower_conversion = true;

  // If greater than zero, tensors with size smaller than or equal to this limit
  // will always be replicated if they don't have a different user-specified
  // sharding.
  int64_t small_tensor_byte_size = 0;

  // In order to obtain default sharding strategies for instructions to limit
  // departures from the defaults, use sharding propagation instead of assuming
  // a simple replicated default.
  bool use_sharding_propagation_for_default_shardings = false;

  // Whether or not to model the memory usage of intermediate tensors, if any,
  // for resharding edges.
  bool model_resharding_memory_costs = true;

  // Whether or not to generate strategies that model the windowed einsum (or
  // collective matmul) optimization
  // TODO(331684721,329508561): Generate windowed-einsum strategies by default
  // once it is fully implemented.
  bool generate_windowed_einsum_strategies = false;

  // Prints a debug string.
  std::string ToString() const;

  // Initializes uninitialized fields with default values, as well as checks the
  // consistency of different options.
  absl::Status CheckAndSetup();
};

}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_OPTION_H_
