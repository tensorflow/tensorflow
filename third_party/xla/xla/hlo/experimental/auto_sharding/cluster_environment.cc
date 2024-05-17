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

#include "xla/hlo/experimental/auto_sharding/cluster_environment.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/shape.h"

namespace xla {
namespace spmd {

double ClusterEnvironment::AllGatherCost(double num_bytes, int mesh_dim) const {
  if (auto_sharding_option_.force_override_all_gather_cost) {
    return auto_sharding_option_.all_gather_cost;
  }

  if (prof_result_.Enabled()) {
    return prof_result_.EstimateAllGatherCost(cached_replica_groups_[mesh_dim],
                                              num_bytes / 4, "float32");
  }

  if (auto_sharding_option_.force_batch_dim_to_mesh_dim == mesh_dim) {
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
  if (auto_sharding_option_.force_override_all_reduce_cost) {
    return auto_sharding_option_.all_reduce_cost;
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
  if (auto_sharding_option_.force_override_reduce_scatter_cost) {
    return auto_sharding_option_.reduce_scatter_cost;
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

double ClusterEnvironment::AllToAllCostUtil(double num_bytes, int mesh_dim,
                                            int64_t num_devices) const {
  // A penalty factor to make the theoretical cost match the
  // empirical cost on v100 + nvlink.
  double penalty_factor = static_cast<double>(num_devices) / 2.0;
  return (round(mesh_alpha_[mesh_dim] +
                mesh_beta_[mesh_dim] * (num_devices - 1) / num_devices /
                    num_devices * num_bytes * penalty_factor) +
          0.001);
}

double ClusterEnvironment::AllToAllCost(double num_bytes, int mesh_dim) const {
  if (auto_sharding_option_.force_override_all_to_all_cost) {
    return auto_sharding_option_.all_to_all_cost;
  }

  if (prof_result_.Enabled()) {
    return prof_result_.EstimateAllToAllCost(cached_replica_groups_[mesh_dim],
                                             num_bytes / 4, "float32");
  }

  if (auto_sharding_option_.force_batch_dim_to_mesh_dim == mesh_dim) {
    // if data-parallel is forced on this dim, we only allow all-reduce
    // in this dimension.
    return kInfinityCost;
  }

  int64_t num_devices = device_mesh_.dim(mesh_dim);
  return AllToAllCostUtil(num_bytes, mesh_dim, num_devices);
}

// Do not consider device id changes yet.
double ClusterEnvironment::ReshardingCostMixedMeshShape(
    const Shape& shape, absl::Span<const int64_t> src_tensor_dim_to_mesh_dim,
    absl::Span<const int64_t> dst_tensor_dim_to_mesh_dim) const {
  int64_t num_devices = device_mesh_.num_elements();
  double resharding_costs = 0.0;
  for (size_t i = 0; i < shape.rank(); ++i) {
    // Only consider sharded dimensions, do not consider replicate_on_last_dim.
    if (src_tensor_dim_to_mesh_dim[i] == dst_tensor_dim_to_mesh_dim[i]) {
      continue;
    }
    if (dst_tensor_dim_to_mesh_dim[i] == -1 ||
        src_tensor_dim_to_mesh_dim[i] == -1) {
      // AllToAll cost
      int64_t communication_dim;
      if (dst_tensor_dim_to_mesh_dim[i] != -1) {
        communication_dim = dst_tensor_dim_to_mesh_dim[i];
      } else {
        communication_dim = src_tensor_dim_to_mesh_dim[i];
      }
      int64_t communication_bytes = GetBytes(shape);
      resharding_costs +=
          AllToAllCostUtil(communication_bytes, communication_dim, num_devices);
    } else {
      // Do not support this sharding, assuming it is gonna be very expensive.
      return kInfinityCost;
    }
  }
  return resharding_costs;
}

double ClusterEnvironment::CollectivePermuteCost(
    double num_bytes,
    absl::Span<const std::pair<int64_t, int64_t>> src_dst_pairs) const {
  absl::flat_hash_map<int64_t, std::vector<int64_t>> device_to_index_map;
  device_mesh_.Each([&](absl::Span<const int64_t> indices, int64_t device) {
    std::vector<int64_t> indices_vector;
    for (auto i : indices) {
      indices_vector.push_back(i);
    }
    device_to_index_map[device] = indices_vector;
  });
  double max_cost = 0;
  for (const auto& pair : src_dst_pairs) {
    auto src_device_indices = device_to_index_map[pair.first];
    auto dst_device_indices = device_to_index_map[pair.second];
    CHECK_EQ(src_device_indices.size(), dst_device_indices.size());
    double pair_cost = 0;
    for (size_t i = 0; i < src_device_indices.size(); ++i) {
      pair_cost += (src_device_indices[i] == dst_device_indices[i])
                       ? 0.0
                       : (mesh_alpha_[i] + mesh_beta_[i] * num_bytes);
    }
    max_cost = std::max(pair_cost, max_cost);
  }
  return max_cost;
}

// Overestimate the cost of replicating a tensor by decomposing the resharding
// operation as an all-gather on all mesh dimensions.
double ClusterEnvironment::OverestimateReplicationCost(
    const Shape& shape, const HloSharding& src_spec,
    const Array<int64_t>& device_mesh) const {
  if (src_spec.IsTileMaximal() || src_spec.IsManual()) {
    // TODO(b/238210866) Do not use kInfinityCost.
    return kInfinityCost;
  }
  int64_t bytes_moved = GetBytes(shape) / src_spec.NumTiles();
  double cost = 0.0;
  for (size_t i = 0; i < device_mesh.num_dimensions(); ++i) {
    auto this_cost = this->AllGatherCost(bytes_moved, i);
    cost += this_cost;
    bytes_moved *= device_mesh.dimensions()[i];
  }
  return cost;
}

double ClusterEnvironment::TryCollectivePermuteForResharding(
    const Shape& shape, const HloSharding& src_spec,
    const HloSharding& dst_spec) const {
  auto reshard_with_collective_permute = [&]() {
    std::vector<std::pair<int64_t, int64_t>> src_dst_pairs;
    src_spec.tile_assignment().Each(
        [&](absl::Span<const int64_t> indices, int64_t src_device) {
          int64_t dst_device = dst_spec.tile_assignment()(indices);
          src_dst_pairs.emplace_back(src_device, dst_device);
        });
    return this->CollectivePermuteCost(GetBytes(shape) / src_spec.NumTiles(),
                                       src_dst_pairs);
  };

  if (CanReshardWithCollectivePermute(src_spec, dst_spec)) {
    return reshard_with_collective_permute();
  }
  if (auto src_tgt_dims =
          GetReshardAllToAllSourceTargetDims(src_spec, dst_spec)) {
    if (src_tgt_dims->empty()) {
      // If the device order is different in the target, fix the order with
      // ReshardWithCollectivePermute.
      return reshard_with_collective_permute();
    }
  }
  // We currently do not handle these cases. These cases previously returned an
  // infinite resharding cost. Instead, we now overestimate the actual
  // resharding cost by decomposing the resharding operation, say from sharding
  // s1 to sharding 2, into two steps:
  // 1. Replicate the tensor,
  // 2. Use dynamic-slice to extract the portion of the tensor as per sharding
  // s2.
  // Since we only estimate communication costs here, we only need to consider
  // the cost of step 1, ie. replicating the tensor starting from sharding
  // s2. We estimate this cost by invoking OverestimateReplicationCost.
  return OverestimateReplicationCost(shape, src_spec, device_mesh_);
}

double ClusterEnvironment::ReshardingCost(const Shape& shape,
                                          const HloSharding& src_spec,
                                          const HloSharding& dst_spec) const {
  // TODO(zhuohan): This function can be wrong and needs more tests.
  if (src_spec == dst_spec || IsUndefined(src_spec) ||
      src_spec.IsReplicated()) {
    return 0.0;
  }

  if (src_spec.tile_assignment().num_elements() > device_mesh_.num_elements() ||
      dst_spec.tile_assignment().num_elements() > device_mesh_.num_elements()) {
    LOG(WARNING)
        << "Full device sharding found when solving for the partial mesh "
        << spmd::ToString(device_mesh_.dimensions())
        << ". Overestimating the resharding cost by assuming full replication "
           "on the full device mesh "
        << spmd::ToString(device_mesh_.dimensions()) << ".";
    return OverestimateReplicationCost(shape, src_spec, original_device_mesh_);
  }

  CHECK(!IsUndefined(dst_spec));
  int64_t src_n_dim = NumTileDimensions(src_spec);
  int64_t dst_n_dim = NumTileDimensions(dst_spec);
  // When src_spec and dst_spec are for arrays with different number of
  // dimensions, which could happen when an instruction follows the sharding
  // of an operand with a different shape, we need to use their
  // TiledDataRank().
  size_t src_rank =
      src_spec.IsTiled() ? src_spec.TiledDataRank() : shape.rank();
  size_t dst_rank =
      dst_spec.IsTiled() ? dst_spec.TiledDataRank() : shape.rank();

  auto get_tensor_dim_to_mesh_dim = [&](int64_t rank,
                                        const HloSharding& sharding) {
    if (VectorGreaterThanOneElementCount(
            sharding.tile_assignment().dimensions()) == 1 &&
        VectorGreaterThanOneElementCount(device_mesh_.dimensions()) > 1) {
      // sharding is 1D and device_mesh is 2D or 3D
      return GetTensorDimToMeshDimNoCrash(
          rank, sharding, device_mesh_1d_,
          /* consider_reverse_device_meshes */ false);
    } else {
      return GetTensorDimToMeshDimNoCrash(
          rank, sharding, device_mesh_,
          /* consider_reverse_device_meshes */ false);
    }
  };

  auto src_tensor_dim_to_mesh_dim_or =
      get_tensor_dim_to_mesh_dim(src_rank, src_spec);
  auto dst_tensor_dim_to_mesh_dim_or =
      get_tensor_dim_to_mesh_dim(dst_rank, dst_spec);

  if (!src_tensor_dim_to_mesh_dim_or.ok() && dst_spec.IsReplicated()) {
    auto equivalent_src_spec = HloSharding::IotaTile(
        src_spec.tile_assignment().dimensions(), src_spec.metadata());
    if (auto equivalent_src_tensor_dim_to_mesh_dim_or =
            get_tensor_dim_to_mesh_dim(src_rank, equivalent_src_spec);
        equivalent_src_tensor_dim_to_mesh_dim_or.ok()) {
      src_tensor_dim_to_mesh_dim_or = equivalent_src_tensor_dim_to_mesh_dim_or;
    }
  }

  // TODO(pratikf) Currently, we return kInfinityCost when the input mesh shape
  // and mesh shape in the sharding do not match. This can possibly be better
  // handled.
  if (!src_tensor_dim_to_mesh_dim_or.ok() ||
      !dst_tensor_dim_to_mesh_dim_or.ok()) {
    return TryCollectivePermuteForResharding(shape, src_spec, dst_spec);
  }

  std::vector<int64_t> src_tensor_dim_to_mesh_dim =
      src_tensor_dim_to_mesh_dim_or.value();
  std::vector<int64_t> dst_tensor_dim_to_mesh_dim =
      dst_tensor_dim_to_mesh_dim_or.value();

  if (src_n_dim != dst_n_dim && src_n_dim != -1 && dst_n_dim != -1) {
    return ReshardingCostMixedMeshShape(shape, src_tensor_dim_to_mesh_dim,
                                        dst_tensor_dim_to_mesh_dim);
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
    return TryCollectivePermuteForResharding(shape, src_spec, dst_spec);
  }

  // Case 1: no communication is required. Only needs dynamic-slice.
  if (all_gather_dims.empty()) {
    return 0;
  }

  // Do not allow some strange re-sharding patterns.
  if (slice_dims.size() > 1 && all_gather_dims.size() > 1) {
    return TryCollectivePermuteForResharding(shape, src_spec, dst_spec);
  }

  // Case 2: all-to-all
  if (slice_dims.size() == 1 && all_gather_dims.size() == 1) {
    if (device_mesh_.dim(0) > 1 && device_mesh_.dim(1) > 1) {
      return TryCollectivePermuteForResharding(shape, src_spec, dst_spec);
    }

    double bytes = GetBytes(shape);
    return AllToAllCost(bytes, all_gather_dims.front());
  }

  // Case 3: all-gather
  double bytes = GetBytes(shape) / src_spec.NumTiles();
  double cost = 0.0;
  for (int dim : all_gather_dims) {
    if (dim >= device_mesh_.num_dimensions()) {
      return TryCollectivePermuteForResharding(shape, src_spec, dst_spec);
    }
    bytes *= device_mesh_.dim(dim);
    cost += AllGatherCost(bytes, dim);
  }
  return cost;
}
}  // namespace spmd
}  // namespace xla
