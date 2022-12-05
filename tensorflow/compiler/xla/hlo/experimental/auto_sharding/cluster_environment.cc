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

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/cluster_environment.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace xla {
namespace spmd {

double ClusterEnvironment::AllGatherCost(double num_bytes, int mesh_dim) const {
  if (solver_option_.override_all_gather_cost) {
    return solver_option_.all_gather_cost;
  }

  if (prof_result_.Enabled()) {
    return prof_result_.EstimateAllGatherCost(cached_replica_groups_[mesh_dim],
                                              num_bytes / 4, "float32");
  }

  if (solver_option_.force_batch_dim_to_mesh_dim == mesh_dim) {
    // if data-parallel is forced on this dim, we only allow all-reduce
    // in this dimension.
    return kInfinityCost;
  }

  int64_t num_devices = device_mesh_.dim(mesh_dim);
  return (round(mesh_alpha_[mesh_dim] + mesh_beta_[mesh_dim] *
                                            (num_devices - 1) / num_devices *
                                            num_bytes) +
          0.1);
}

// TODO(zhuohan): distinguish dtype and reduce_op.
double ClusterEnvironment::AllReduceCost(double num_bytes, int32_t mesh_dim,
                                         int32_t mesh_dim_another) const {
  if (solver_option_.override_all_reduce_cost) {
    return solver_option_.all_reduce_cost;
  }

  if (prof_result_.Enabled()) {
    return prof_result_.EstimateAllReduceCost(cached_replica_groups_[mesh_dim],
                                              num_bytes / 4, "float32");
  }
  double alpha, beta;
  int64_t num_devices;
  if (mesh_dim_another == -1) {
    // Only communicating on one mesh dimension.
    alpha = mesh_alpha_[mesh_dim];
    beta = mesh_beta_[mesh_dim];
    num_devices = device_mesh_.dim(mesh_dim);
  } else {
    // Communicating through both mesh dimensions.
    alpha = std::max(mesh_alpha_[mesh_dim], mesh_alpha_[mesh_dim_another]);
    beta = std::max(mesh_beta_[mesh_dim], mesh_beta_[mesh_dim_another]);
    num_devices = device_mesh_.num_elements();
  }
  return (
      round(alpha + beta * 2 * (num_devices - 1) / num_devices * num_bytes) +
      0.01);
}

double ClusterEnvironment::ReduceScatterCost(double num_bytes,
                                             int mesh_dim) const {
  if (solver_option_.override_reduce_scatter_cost) {
    return solver_option_.reduce_scatter_cost;
  }

  if (prof_result_.Enabled()) {
    return prof_result_.EstimateReduceScatterCost(
        cached_replica_groups_[mesh_dim], num_bytes / 4, "float32");
  }

  int64_t num_devices = device_mesh_.dim(mesh_dim);
  return (round(mesh_alpha_[mesh_dim] + mesh_beta_[mesh_dim] *
                                            (num_devices - 1) / num_devices *
                                            num_bytes) +
          0.001);
}

double ClusterEnvironment::AllToAllCost(double num_bytes, int mesh_dim) const {
  if (solver_option_.override_all_to_all_cost) {
    return solver_option_.all_to_all_cost;
  }

  if (prof_result_.Enabled()) {
    return prof_result_.EstimateAllToAllCost(cached_replica_groups_[mesh_dim],
                                             num_bytes / 4, "float32");
  }

  if (solver_option_.force_batch_dim_to_mesh_dim == mesh_dim) {
    // if data-parallel is forced on this dim, we only allow all-reduce
    // in this dimension.
    return kInfinityCost;
  }

  int64_t num_devices = device_mesh_.dim(mesh_dim);
  return AllToAllCostUtil(num_bytes, mesh_dim, num_devices, mesh_alpha_,
                          mesh_beta_);
}

double ClusterEnvironment::DotCost(const Shape& lhs_shape,
                                   const Shape& rhs_shape,
                                   const DotDimensionNumbers& dot_dnums) const {
  if (!solver_option_.allow_recompute_heavy_op) {
    return kInfinityCost;
  }

  // TODO(zhuohan): When profiling data is not available, it is not easy to
  // align the scale of compute cost and communication cost. Here we just use
  // a simple heuristic to compute the compute cost with communication cost.
  double num_bytes = GetBytes(lhs_shape) + GetBytes(rhs_shape);
  return AllReduceCost(num_bytes, 0) + AllReduceCost(num_bytes, 1);
}

// The communication cost of resharding a tensor from src to dst
// TODO(b/238210866) Do not use kInfinityCost.
double ClusterEnvironment::ReshardingCost(const Shape& shape,
                                          const HloSharding& src_spec,
                                          const HloSharding& dst_spec) const {
  // TODO(zhuohan): This function can be wrong and needs more tests.
  if (src_spec == dst_spec || IsUndefined(src_spec)) {
    return 0.0;
  }
  CHECK(!IsUndefined(dst_spec));
  int64_t src_n_dim = NumTileDimensions(src_spec);
  int64_t dst_n_dim = NumTileDimensions(dst_spec);
  // When src_spec and dst_spec are for arrays with different number of
  // dimensions, which could happen when an instruction follows the sharding
  // of an operand with a different shape, we need to use their
  // TiledDataRank().
  size_t src_rank = shape.rank();
  if (src_spec.IsTiled()) {
    src_rank = src_spec.TiledDataRank();
  }
  size_t dst_rank = shape.rank();
  if (dst_spec.IsTiled()) {
    dst_rank = dst_spec.TiledDataRank();
  }
  std::vector<int64_t> src_tensor_dim_to_mesh_dim;
  if (VectorGreaterThanOneElementCount(
          src_spec.tile_assignment().dimensions()) == 1 &&
      VectorGreaterThanOneElementCount(device_mesh_.dimensions()) > 1) {
    // src spec is 1D and device_mesh is 2D or 3D
    src_tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(src_rank, src_spec, device_mesh_1d_);
  } else {
    src_tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(src_rank, src_spec, device_mesh_);
  }
  std::vector<int64_t> dst_tensor_dim_to_mesh_dim;
  if (VectorGreaterThanOneElementCount(
          dst_spec.tile_assignment().dimensions()) == 1 &&
      VectorGreaterThanOneElementCount(device_mesh_.dimensions()) > 1) {
    // src spec is 1D and device_mesh is 2D or 3D
    dst_tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(dst_rank, dst_spec, device_mesh_1d_);
  } else {
    dst_tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(dst_rank, dst_spec, device_mesh_);
  }
  if (src_n_dim != dst_n_dim && src_n_dim != -1 && dst_n_dim != -1) {
    return ReshardingCostMixedMeshShape(
        shape, src_tensor_dim_to_mesh_dim, dst_tensor_dim_to_mesh_dim,
        device_mesh_.num_elements(), mesh_alpha_, mesh_beta_);
  }

  AdjustTensorMeshDimMapping(src_tensor_dim_to_mesh_dim, src_n_dim);
  AdjustTensorMeshDimMapping(dst_tensor_dim_to_mesh_dim, dst_n_dim);

  // Analyze the dims that need to dynamic-sliced or all-gather.
  std::vector<int> slice_dims;
  std::vector<int> all_gather_dims;
  for (int64_t i = 0; i < std::min(src_rank, dst_rank); ++i) {
    int src_mesh_dim = src_tensor_dim_to_mesh_dim[i];
    int dst_mesh_dim = dst_tensor_dim_to_mesh_dim[i];
    if (src_mesh_dim == dst_mesh_dim) {
      continue;
    }
    if (src_mesh_dim == -1) {
      slice_dims.push_back(src_mesh_dim);
      continue;
    }
    if (dst_mesh_dim == -1) {
      all_gather_dims.push_back(src_mesh_dim);
      continue;
    }
    // Do not allow other re-sharding patterns. (e.g., collective-permute)
    return kInfinityCost;
  }

  // Case 1: no communication is required. Only needs dynamic-slice.
  if (all_gather_dims.empty()) {
    return 0;
  }

  // Do not allow some strange re-sharding patterns.
  if (slice_dims.size() > 1 && all_gather_dims.size() > 1) {
    return kInfinityCost;
  }

  // Case 2: all-to-all
  if (slice_dims.size() == 1 && all_gather_dims.size() == 1) {
    if (device_mesh_.dim(0) > 1 && device_mesh_.dim(1) > 1) {
      return kInfinityCost;
    }

    double bytes = GetBytes(shape);
    return AllToAllCost(bytes, all_gather_dims.front());
  }

  // Case 3: all-gather
  double bytes = GetBytes(shape) / src_spec.NumTiles();
  double cost = 0.0;
  for (int dim : all_gather_dims) {
    if (dim >= device_mesh_.num_dimensions()) {
      return kInfinityCost;
    }
    bytes *= device_mesh_.dim(dim);
    cost += AllGatherCost(bytes, dim);
  }
  return cost;
}
}  // namespace spmd
}  // namespace xla
