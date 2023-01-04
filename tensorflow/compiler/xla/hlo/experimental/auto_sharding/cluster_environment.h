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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_CLUSTER_ENVIRONMENT_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_CLUSTER_ENVIRONMENT_H_

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver_option.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/profiling_result.h"

namespace xla {
namespace spmd {

// The cluster has a multi-dimensional device mesh topology.
// Each mesh dimension has its own latency and bandwidth.
// We use alpha-beta model to model the communication cost.
// If profiling result is provided, we always prefer to use
// the real profiling result.
class ClusterEnvironment {
 public:
  ClusterEnvironment(const Array<int64_t>& original_device_mesh,
                     const Array<int64_t>& device_mesh,
                     absl::Span<const double> mesh_alpha,
                     absl::Span<const double> mesh_beta,
                     const ProfilingResult& prof_result,
                     const AutoShardingSolverOption& solver_option)
      : original_device_mesh_(original_device_mesh),
        device_mesh_(device_mesh),
        mesh_alpha_(mesh_alpha.begin(), mesh_alpha.end()),
        mesh_beta_(mesh_beta.begin(), mesh_beta.end()),
        prof_result_(prof_result),
        total_devices_(device_mesh.num_elements()),
        device_mesh_1d_(original_device_mesh),
        solver_option_(solver_option) {
    // Build replica group for each dimension.
    non_zero_mesh_dims_ =
        VectorGreaterThanOneElementIndices(device_mesh.dimensions());
    GenerateCachedReplicaGroups();
    // TODO(yuemmawang) Find the largest dimension in original_device_mesh and
    // create 1d mesh on that dimension.
    device_mesh_1d_.Reshape({original_device_mesh.num_elements(), 1});
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
      const Shape& shape, const HloSharding& spec) const {
    int64_t n_dim = NumTileDimensions(spec);
    std::vector<int64_t> tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(shape.rank(), spec, device_mesh_);
    AdjustTensorMeshDimMapping(tensor_dim_to_mesh_dim, n_dim);
    return tensor_dim_to_mesh_dim;
  }

  double AllGatherCost(double num_bytes, int mesh_dim) const;

  double AllReduceCost(double num_bytes, int32_t mesh_dim,
                       int32_t mesh_dim_another = -1) const;

  double ReduceScatterCost(double num_bytes, int mesh_dim) const;

  double AllToAllCost(double num_bytes, int mesh_dim) const;

  double DotCost(const Shape& lhs_shape, const Shape& rhs_shape,
                 const DotDimensionNumbers& dot_dnums) const;

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
  const Array<int64_t> original_device_mesh_;
  // When solve_nd_sharding_iteratively is true, it is a partial mesh shape from
  // the original_device_mesh_. When solve_nd_sharding_iteratively is false, it
  // is the same as original_device_mesh_.
  const Array<int64_t> device_mesh_;
  // Bandwidth of the device mesh
  const std::vector<double> mesh_alpha_;
  const std::vector<double> mesh_beta_;
  const ProfilingResult& prof_result_;
  std::vector<int64_t> non_zero_mesh_dims_;
  const int total_devices_;

  // Cache a flatten 1d version of the device mesh.
  // Used for mixed mesh shape strategies.
  Array<int64_t> device_mesh_1d_;

  // The solver option may override the cost of communication primitives
  const AutoShardingSolverOption& solver_option_;

  // Cached replica groups. Shape: [mesh_dim, group_id, ids in this group].
  std::vector<std::vector<std::vector<int64_t>>> cached_replica_groups_;

 private:
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

#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_CLUSTER_ENVIRONMENT_H_
