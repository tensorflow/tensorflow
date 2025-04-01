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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_CLUSTER_ENVIRONMENT_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_CLUSTER_ENVIRONMENT_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_device_mesh.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "xla/hlo/experimental/auto_sharding/profiling_result.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape.h"

namespace xla {
namespace spmd {

// The cluster has a multi-dimensional device mesh topology.
// Each mesh dimension has its own latency and bandwidth.
// We use alpha-beta model to model the communication cost.
// If profiling result is provided, we always prefer to use
// the real profiling result.
class ClusterEnvironment {
 public:
  ClusterEnvironment(const DeviceMesh& original_device_mesh,
                     const DeviceMesh& device_mesh,
                     absl::Span<const double> mesh_alpha,
                     absl::Span<const double> mesh_beta,
                     const ProfilingResult& prof_result,
                     const AutoShardingOption& auto_sharding_option)
      : original_device_mesh_(original_device_mesh),
        device_mesh_(device_mesh),
        mesh_alpha_(mesh_alpha.begin(), mesh_alpha.end()),
        mesh_beta_(mesh_beta.begin(), mesh_beta.end()),
        prof_result_(prof_result),
        total_devices_(device_mesh.num_elements()),
        device_mesh_1d_(device_mesh),
        original_device_mesh_1d_(original_device_mesh),
        auto_sharding_option_(auto_sharding_option) {
    // Build replica group for each dimension.
    non_zero_mesh_dims_ =
        VectorGreaterThanOneElementIndices(device_mesh.dimensions());
    GenerateCachedReplicaGroups();

    // Essentially, we want to create a 1D mesh here such that the resharding
    // costs between the original mesh and this 1D mesh are the least. This
    // essentially means we create a 1D shape which stretches along the largest
    // dimension of the original mesh. This will not however for asymmetric
    // values of alpha and beta, I think.
    // TODO(pratikf) Fix this for asymmetric alpha and beta values.
    auto original_device_mesh_shape = original_device_mesh.dimensions();
    auto max_dim_iterator = std::max_element(original_device_mesh_shape.begin(),
                                             original_device_mesh_shape.end());
    size_t largest_dim_idx =
        std::distance(original_device_mesh_shape.begin(), max_dim_iterator);
    std::vector<int64_t> device_mesh_1d_shape(device_mesh.num_dimensions(), 1);
    device_mesh_1d_shape[largest_dim_idx] = device_mesh.num_elements();
    device_mesh_1d_.Reshape(device_mesh_1d_shape);

    std::vector<int64_t> original_device_mesh_1d_shape(
        original_device_mesh.num_dimensions(), 1);
    original_device_mesh_1d_shape[largest_dim_idx] =
        original_device_mesh.num_elements();
    original_device_mesh_1d_.Reshape(original_device_mesh_1d_shape);
  }

  size_t NumDevices() const { return total_devices_; }

  bool IsDeviceMesh3D() const {
    return VectorGreaterThanOneElementCount(device_mesh_.dimensions()) == 3;
  }

  bool IsDeviceMesh2D() const {
    return VectorGreaterThanOneElementCount(device_mesh_.dimensions()) == 2;
  }

  bool IsDeviceMesh1D() const {
    return VectorGreaterThanOneElementCount(device_mesh_.dimensions()) == 1;
  }

  bool IsOriginalDeviceMesh2D() const {
    return VectorGreaterThanOneElementCount(
               original_device_mesh_.dimensions()) == 2;
  }

  // Get the corresponding mesh dimension for every tensor dimension.
  // -1 means replicated on that dimension
  std::vector<int64_t> GetTensorDimToMeshDimWrapper(
      const Shape& shape, const HloSharding& spec,
      bool consider_reverse_device_meshes = false,
      bool crash_at_error = true) const {
    int64_t n_dim = NumTileDimensions(spec);
    std::vector<int64_t> tensor_dim_to_mesh_dim;
    if (crash_at_error) {
      tensor_dim_to_mesh_dim =
          GetTensorDimToMeshDim(shape.dimensions_size(), spec, device_mesh_,
                                consider_reverse_device_meshes);
    } else {
      auto tensor_dim_to_mesh_dim_status = GetTensorDimToMeshDimNoCrash(
          shape.dimensions_size(), spec, device_mesh_,
          consider_reverse_device_meshes);
      if (tensor_dim_to_mesh_dim_status.ok()) {
        tensor_dim_to_mesh_dim = tensor_dim_to_mesh_dim_status.value();
      }
    }
    AdjustTensorMeshDimMapping(tensor_dim_to_mesh_dim, n_dim);
    return tensor_dim_to_mesh_dim;
  }

  double GetDefaultReplicatedPenalty() const {
    double replicated_penalty = 0;
    for (int i = 0; i < device_mesh_.num_dimensions(); ++i) {
      replicated_penalty += AllReduceCost(1, i);
    }
    return std::round(replicated_penalty);
  }

  double AllGatherCost(double num_bytes, int mesh_dim) const;

  double AllReduceCost(double num_bytes, int32_t mesh_dim,
                       int32_t mesh_dim_another = -1) const;

  double ReduceScatterCost(double num_bytes, int mesh_dim) const;

  double AllToAllCost(double num_bytes, int mesh_dim) const;

  double ReshardingCostMixedMeshShape(const Shape& shape,
                                      const HloSharding& src_sharding,
                                      const HloSharding& dst_sharding) const;

  double CollectivePermuteCost(
      double num_bytes,
      absl::Span<const std::pair<int64_t, int64_t>> src_dst_pairs) const;

  double TryCollectivePermuteForResharding(const Shape& shape,
                                           const HloSharding& src_spec,
                                           const HloSharding& dst_spec) const;

  // This function attempts to overestimate the cost of replicating a tensor of
  // shape `shape` sharded according to `src_spec`.
  double OverestimateReplicationCost(const Shape& shape,
                                     const HloSharding& src_spec,
                                     const DeviceMesh& device_mesh) const;

  double ReshardingCost(const Shape& shape, const HloSharding& src_spec,
                        const HloSharding& dst_spec) const;

  // Print the information of this device mesh.
  std::string ToString() {
    std::string str;
    absl::StrAppend(&str, "device_mesh: ", device_mesh_.ToString(), "\n");
    absl::StrAppend(&str, "mesh_alpha: ", absl::StrJoin(mesh_alpha_, " "),
                    "\n");
    absl::StrAppend(&str, "mesh_beta: ", absl::StrJoin(mesh_beta_, " "), "\n");
    return str;
  }

  // The original, complete device mesh shape that describes the hardware.
  const DeviceMesh original_device_mesh_;
  // When solve_nd_sharding_iteratively is true, it is a partial mesh shape from
  // the original_device_mesh_. When solve_nd_sharding_iteratively is false, it
  // is the same as original_device_mesh_.
  const DeviceMesh device_mesh_;
  // Bandwidth of the device mesh
  const std::vector<double> mesh_alpha_;
  const std::vector<double> mesh_beta_;
  const ProfilingResult& prof_result_;
  std::vector<int64_t> non_zero_mesh_dims_;
  const int total_devices_;

  // Cache a flatten 1d version of the device mesh.
  // Used for mixed mesh shape strategies.
  DeviceMesh device_mesh_1d_;

  // Cache a flatten 1d version of the original device mesh.
  // Used for mixed mesh shape strategies.
  DeviceMesh original_device_mesh_1d_;

  // The option may override the cost of communication primitives
  const AutoShardingOption& auto_sharding_option_;

  // Cached replica groups. Shape: [mesh_dim, group_id, ids in this group].
  std::vector<std::vector<std::vector<int64_t>>> cached_replica_groups_;

 private:
  double AllToAllCostUtil(double num_bytes, int mesh_dim,
                          int64_t num_devices) const;

  void GenerateCachedReplicaGroups() {
    // One vector per device_mesh_ dimension.
    cached_replica_groups_.reserve(device_mesh_.num_dimensions());
    for (size_t i = 0; i < device_mesh_.num_dimensions(); i++) {
      cached_replica_groups_.push_back(
          GetReplicaGroupsAlongOneDimension(device_mesh_, i));
    }
  }

  void AdjustTensorMeshDimMapping(std::vector<int64_t>& mapping,
                                  int64_t n_dim) const {
    // Shift the non-zero dim for 1d mesh
    if (n_dim == 1 && non_zero_mesh_dims_.size() == 1) {
      for (size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] == 0) {
          mapping[i] = non_zero_mesh_dims_.front();
        }
      }
    }
  }
};
}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_CLUSTER_ENVIRONMENT_H_
