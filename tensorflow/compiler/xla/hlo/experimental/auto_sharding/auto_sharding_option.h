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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_OPTION_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_OPTION_H_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_util.h"

namespace xla {

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

  // Timeout for the solver. If the solver fails to find an optimal solution
  // before the timeout, we rely on the heuristic-based sharding implemented in
  // sharding_propagation.cc.
  int64_t solver_timeout_in_seconds = 3600;

  // Static estimate for iteration count of a while loop, used in the cost model
  int64_t loop_iteration_count_estimate = 100;

  // Whether or not to generate replicated strategies for dot/conv
  // ops. Generating these seems to be beneficial for LLM serving models, but
  // can increase the search space, so this feature is exposed as an option.
  bool allow_replicated_strategy_for_dot_and_conv = true;

  // Allows the conversion of aliases to followers if their pairwise strategy
  // compatibilities are embodied by the identity matrix (which makes for a
  // smaller Mixed ILP).
  bool allow_alias_to_follower_conversion = true;

  std::vector<int64_t> strategy_vector;
  // If greater than zero, tensors with size smaller than or equal to this limit
  // will always be replicated if they don't have a different user-specified
  // sharding.
  int64_t small_tensor_byte_size = 0;

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
      return absl::OutOfRangeError(
          "device_mesh_shape is empty and it needs to be specified.");
    }
    std::vector<int64_t> mesh_dims_greater_than_one_indices =
        spmd::VectorGreaterThanOneElementIndices(device_mesh_shape);

    if (mesh_dims_greater_than_one_indices.size() > 3) {
      return absl::OutOfRangeError(absl::StrCat(
          "Not supported: only device_mesh_shapes with 3 or less "
          "dimensions larger than 1 are supported. Instead we have ",
          mesh_dims_greater_than_one_indices.size(),
          " dimensions greater than 1."));
    }
    // All values in device_mesh_shape must be greater than 0.
    if (absl::c_any_of(device_mesh_shape,
                       [](const int64_t i) { return i <= 0; })) {
      return absl::OutOfRangeError(
          absl::StrCat("device_mesh_shape values need to be larger than 0: "
                       "device_mesh_shape=",
                       absl::StrJoin(device_mesh_shape, ",")));
    }
    if (spmd::VectorGreaterThanOneElementCount(device_mesh_shape) > 2) {
      return absl::OutOfRangeError(
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

    if (device_mesh_shape.size() != device_mesh_alpha.size() ||
        device_mesh_shape.size() != device_mesh_beta.size()) {
      return absl::OutOfRangeError(absl::StrCat(
          "Sizes do not match: length of device_mesh_shape is ",
          device_mesh_shape.size(), ", length of device_mesh_alpha is ",
          device_mesh_alpha.size(), ", length of device_mesh_beta is ",
          device_mesh_beta.size(),
          ". If not sure how to set device_mesh_alpha and "
          "device_mesh_beta, "
          "please leave them empty and default values will be used."));
    }

    std::vector<int64_t> compressed_device_mesh_shape;
    std::vector<double> compressed_device_mesh_alpha;
    std::vector<double> compressed_device_mesh_beta;
    int non_zero_counter = 0;
    for (size_t i = 0; i < device_mesh_shape.size(); ++i) {
      if (non_zero_counter < mesh_dims_greater_than_one_indices.size() &&
          i == mesh_dims_greater_than_one_indices[non_zero_counter]) {
        non_zero_counter++;
        compressed_device_mesh_shape.push_back(device_mesh_shape[i]);
        compressed_device_mesh_alpha.push_back(device_mesh_alpha[i]);
        compressed_device_mesh_beta.push_back(device_mesh_beta[i]);
      }
    }
    this->device_mesh_shape = compressed_device_mesh_shape;
    this->device_mesh_alpha = compressed_device_mesh_alpha;
    this->device_mesh_beta = compressed_device_mesh_beta;

    // If device_mesh_shape has only one value, append 1 to it
    if (device_mesh_shape.size() == 1) {
      device_mesh_shape.push_back(1);
      device_mesh_alpha.push_back(1.0);
      device_mesh_beta.push_back(1.0);
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
        return absl::OutOfRangeError(absl::StrCat(
            "Expect the product of device_mesh_shape to be the same as the "
            "size of device_mesh_ids, but we have total devices = ",
            total_devices,
            " and device_mesh_ids.size()=", device_mesh_ids.size()));
      }
    }
    return OkStatus();
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_OPTION_H_
