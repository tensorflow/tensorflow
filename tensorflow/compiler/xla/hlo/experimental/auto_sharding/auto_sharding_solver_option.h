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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_OPTION_H__
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_OPTION_H__

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xla {
namespace spmd {
// Options for the auto-sharding solver.
struct AutoShardingSolverOption {
  // Forcibly split the batch dimension and map it to a mesh dimension.
  // This can force the auto-sharding pass to generate the data parallel
  // strategy.
  int force_batch_dim_to_mesh_dim;

  // If true, override the cost of all-gather with the given value.
  bool override_all_gather_cost;
  double all_gather_cost;

  // If true, override the cost of all-reduce with the given value.
  bool override_all_reduce_cost;
  double all_reduce_cost;

  // If true, override the cost of reduce-scatter with the given value.
  bool override_reduce_scatter_cost;
  double reduce_scatter_cost;

  // If true, override the cost of all-to-all with the given value.
  bool override_all_to_all_cost;
  double all_to_all_cost;

  // If true, allow replicated parameters.
  bool allow_replicated_parameters;

  // If true, prefer reduce-scatter + all-gather over all-reduce.
  // A post process will be applied to replace all-reduce with reduce-scater +
  // all-gather if no communication overhead is introduced.
  bool prefer_reduce_scatter;

  // If True, generate a gradient-accumulation friendly variant of
  // reduce-scatter
  bool reduce_scatter_grad_acc_friendly;

  // If true, aggressively partition more tensors when generating
  // reduce-scatter, even if it introduces more communication.
  bool reduce_scatter_aggressive_partition;

  // If true, the batch matmul will always be parallelized on the batch dim in
  // 2d mesh case.
  bool batch_matmul_always_split_batch;

  // If true, allow strategies that recompute heavy operators (e.g., dot)
  // to reduce communication.
  bool allow_recompute_heavy_op;

  // If true, allow adding 1d strategies in 2d logical mesh.
  bool allow_mixed_mesh_shape;

  // The number of micro batches if gradient accumulation is used.
  // If this is not 1, the cost of all-reduce for gradient synchronization
  // is divided by this number.
  int grad_acc_num_micro_batches;

  // If true, load solution vector from PassContext
  bool load_solution_vector;

  // If it is not empty, forcibly use simple heuristic strategies
  // instead of the ILP solver. This is used for ablation study.
  std::string force_simple_heuristic;

  // If true, forcibly set the strategy of some instructions
  bool force_strategy;
  std::vector<int64_t> force_strategy_inst_indices;
  std::vector<std::string> force_strategy_stra_names;

  bool only_allow_divisible_input_output;

  bool only_allow_divisible_intermediate;

  // If true, trictly limit the following iterations to use the same number of
  // shards for sharded tensor dimensions; if false, the following iterations
  // can choose different number of shards for sharded tensor dimensions.
  // Enabling it can hurt the performance of dot ops, but can make the search
  // space more scalable. Therefore leaving it as an option.
  bool nd_sharding_iteratively_strict_search_space;
};
}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_OPTION_H_
