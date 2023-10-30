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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"

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
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"

namespace xla {
std::string AutoShardingOption::ToString() const {
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

  lines.push_back(absl::StrCat("force_override_all_gather_cost: ",
                               force_override_all_gather_cost));
  if (force_override_all_gather_cost) {
    lines.push_back(absl::StrCat("all_gather_cost: ", all_gather_cost));
  }

  lines.push_back(absl::StrCat("force_override_all_to_all_cost: ",
                               force_override_all_to_all_cost));
  if (force_override_all_to_all_cost) {
    lines.push_back(absl::StrCat("all_to_all_cost: ", all_to_all_cost));
  }

  lines.push_back(absl::StrCat("force_override_all_reduce_cost: ",
                               force_override_all_reduce_cost));
  if (force_override_all_reduce_cost) {
    lines.push_back(absl::StrCat("all_reduce_cost: ", all_reduce_cost));
  }

  lines.push_back(absl::StrCat("force_override_reduce_scatter_cost: ",
                               force_override_reduce_scatter_cost));
  if (force_override_reduce_scatter_cost) {
    lines.push_back(absl::StrCat("reduce_scatter_cost: ", reduce_scatter_cost));
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
  lines.push_back(
      absl::StrCat("grad_acc_num_micro_batches: ", grad_acc_num_micro_batches));
  lines.push_back(absl::StrCat("solve_nd_sharding_iteratively: ",
                               solve_nd_sharding_iteratively));
  lines.push_back(
      absl::StrCat("force_simple_heuristic: ", force_simple_heuristic));
  lines.push_back(absl::StrCat("force_strategy: ", force_strategy));

  if (force_strategy) {
    lines.push_back(
        absl::StrCat("force_strategy_inst_indices: [",
                     absl::StrJoin(force_strategy_inst_indices, ","), "]"));
    lines.push_back(absl::StrCat("force_strategy_stra_names: [",
                                 absl::StrJoin(force_strategy_stra_names, ","),
                                 "]"));
  }

  lines.push_back(absl::StrCat("only_allow_divisible_input_output: ",
                               only_allow_divisible_input_output));

  lines.push_back(absl::StrCat("only_allow_divisible_intermediate: ",
                               only_allow_divisible_intermediate));

  lines.push_back(absl::StrCat("nd_sharding_iteratively_strict_search_space: ",
                               nd_sharding_iteratively_strict_search_space));

  lines.push_back(absl::StrCat("device_mesh_shape: [",
                               absl::StrJoin(device_mesh_shape, ","), "]"));
  lines.push_back(absl::StrCat("device_mesh_alpha: [",
                               absl::StrJoin(device_mesh_alpha, ","), "]"));
  lines.push_back(absl::StrCat("device_mesh_beta: [",
                               absl::StrJoin(device_mesh_beta, ","), "]"));

  lines.push_back(
      absl::StrCat("try_multiple_mesh_shapes: ", try_multiple_mesh_shapes));

  lines.push_back(
      absl::StrCat("solver_timeout_in_seconds: ", solver_timeout_in_seconds));

  lines.push_back(absl::StrCat("loop_iteration_count_estimate: ",
                               loop_iteration_count_estimate));

  lines.push_back(absl::StrCat("allow_alias_to_follower_conversion: ",
                               allow_alias_to_follower_conversion));

  lines.push_back(
      absl::StrCat("small_tensor_byte_size: ", small_tensor_byte_size));

  lines.push_back(
      absl::StrCat("use_sharding_propagation_for_default_shardings: ",
                   use_sharding_propagation_for_default_shardings));

  lines.push_back(absl::StrCat("model_resharding_memory_costs: ",
                               model_resharding_memory_costs));

  lines.push_back(absl::StrCat("generate_windowed_einsum_strategies: ",
                               generate_windowed_einsum_strategies));

  lines.push_back(
      absl::StrCat("allow_shardings_small_dims_across_many_devices: ",
                   allow_shardings_small_dims_across_many_devices));

  return absl::StrJoin(lines, "\n");
}

absl::Status AutoShardingOption::CheckAndSetup() {
  only_allow_divisible_input_output = true;
  only_allow_divisible_intermediate = false;

  if (device_mesh_shape.empty()) {
    return absl::OutOfRangeError(
        "device_mesh_shape is empty and it needs to be specified.");
  }
  std::vector<int64_t> mesh_dims_greater_than_one_indices =
      spmd::VectorGreaterThanOneElementIndices(device_mesh_shape);

  // TODO(pratikf) The device mesh shape handling in this function currently
  // does not work when try_multiple_mesh_shapes is true. Fix it.
  if (mesh_dims_greater_than_one_indices.size() > 3 ||
      (device_mesh_shape.size() > 3 && try_multiple_mesh_shapes)) {
    return absl::OutOfRangeError(
        absl::StrCat("Not supported: only device_mesh_shapes with 3 or less "
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
  if (spmd::VectorGreaterThanOneElementCount(device_mesh_shape) > 3) {
    return absl::OutOfRangeError(
        absl::StrCat("the auto-sharding pass currently does not support ",
                     "more than three shardable dims: device_mesh_shape=",
                     absl::StrJoin(device_mesh_shape, ",")));
  }

  if (device_mesh_alpha.empty()) {
    // Generates simple device_mesh_alpha based on the size of
    // device_mesh_shape.
    device_mesh_alpha = std::vector(device_mesh_shape.size(), kDeviceMeshAlpha);
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

  if (!try_multiple_mesh_shapes) {
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
  }

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
  return absl::OkStatus();
}

}  // namespace xla
