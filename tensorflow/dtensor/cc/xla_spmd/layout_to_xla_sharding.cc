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

#include "tensorflow/dtensor/cc/xla_spmd/layout_to_xla_sharding.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {

namespace {

StatusOr<int64_t> DeviceLocationToLinearIndex(
    absl::Span<const int64_t> mesh_shape, DeviceLocation dev_loc,
    absl::Span<const int32_t> minor_to_major_ordering) {
  if (mesh_shape.size() != dev_loc.size()) {
    return errors::InvalidArgument(
        "Mesh shape size and multi_index size must be equal");
  }
  int64_t scale = 1;
  int64_t linear_index = 0;
  for (auto dimension : minor_to_major_ordering) {
    linear_index += scale * dev_loc[dimension];
    scale *= mesh_shape[dimension];
  }

  return linear_index;
}

// Returns a grouping of devices in major-first to minor-last ordering based on
// groups of devices that have the same piece of sharded tensor.
//
// `sharded_indices` represent the sharded indices from some Layout.
//
// Example:
//  For a mesh with dimensions {'x': 2, 'y': 2} and sharded_indices {0},
//  this means that the Layout is sharded only on the `x` dimension, and thus
//  the mesh coordinates {0, 0} and {0, 1} are in one group and {1, 0} and
//  {1, 1} are in the other group. If we convert mesh coordinates to linearized
//  device index, this returns {{0, 1}, {2, 3}}.
StatusOr<std::vector<std::vector<int64_t>>> ComputeReplicatedGroups(
    const Mesh& mesh, const std::vector<int64_t>& sharded_indices) {
  std::map<DeviceLocation, std::vector<int64_t>> replicated_group_map;

  for (size_t device_id = 0; device_id < mesh.size(); ++device_id) {
    TF_ASSIGN_OR_RETURN(DeviceLocation dev_loc,
                        mesh.device_location(device_id));
    DeviceLocation reduced_dev_loc;
    for (int64_t shard_index : sharded_indices) {
      reduced_dev_loc.push_back(dev_loc[shard_index]);
    }
    replicated_group_map[reduced_dev_loc].push_back(device_id);
  }

  // Reorder these replica groups from the map in major-first to minor-last
  // ordering by going through each device id in increasing ordering.
  std::vector<std::vector<int64_t>> replicated_groups;
  absl::flat_hash_set<int64_t> already_included_devices;

  for (size_t device_id = 0; device_id < mesh.size(); ++device_id) {
    if (already_included_devices.contains(device_id)) {
      continue;
    }
    // Find the replicated group that contains this device_id and add all
    // devices in that group to replicated_groups.
    for (const auto& [unused_hash, group] : replicated_group_map) {
      if (std::find(group.begin(), group.end(), device_id) != group.end()) {
        replicated_groups.push_back(group);

        for (int64_t device : group) {
          already_included_devices.insert(device);
        }
      }
    }
  }
  return replicated_groups;
}

struct MeshDimInfo {
  // Stores the dimension size of a mesh dimension.
  int64_t size;
  // Stores the index of a mesh dimension.
  int64_t index;
};

// Returns a vector of device ids for XLA OpShardings `tile_assignment_devices`
// field, based on the layout_shard_specs and the dimensions of the mesh.
//
// Note that this function assumes that the layouts are fully sharded, i.e
// there is no Layout::UNSHARDED dimension in `layout_shard_specs`.
//
// At a high level, this function is a permutation function that permutes device
// ids from [0, n) to a new ordering based on however the `layout_shard_specs`
// transposes the ordering of `mesh_dims`.
//
// `tile_assignment_devices` in ::xla::OpSharding is a linearized list of
// devices based on a defined `minor_to_major` ordering. The default
// `minor_to_major` ordering of a Mesh is first index major, i.e [n-1, n-2, ...,
// 0]. The `layout_shard_specs` essentially defines a new minor_to_major
// ordering based on the ordering of the shard specs, and is needed  to
// compute `tile_assignment_devices`.
StatusOr<std::vector<int64_t>> ComputeTileAssignmentDevices(
    const std::vector<std::string>& layout_shard_specs,
    const std::vector<MeshDimension>& mesh_dims) {
  if (layout_shard_specs.size() != mesh_dims.size()) {
    return errors::InvalidArgument(
        "Number of shard specs must equal number of mesh dimensions. This "
        "might indicate that Layout is not fully sharded.");
  }

  absl::flat_hash_map<std::string, MeshDimInfo> mesh_spec_to_info;
  int64_t num_devices = 1;

  for (int64_t i = 0; i < mesh_dims.size(); ++i) {
    num_devices *= mesh_dims[i].size;
    MeshDimInfo mesh_dim_info;
    mesh_dim_info.size = mesh_dims[i].size;
    mesh_dim_info.index = i;
    mesh_spec_to_info[mesh_dims[i].name] = mesh_dim_info;
  }

  // Shape of transposed mesh based on the ordering of layout's sharding specs.
  std::vector<int64_t> mesh_shape;
  mesh_shape.reserve(layout_shard_specs.size());
  for (const MeshDimension& mesh_dim : mesh_dims) {
    mesh_shape.push_back(mesh_dim.size);
  }

  // Compute the new minor to major ordering based on the ordering of layout
  // sharding.
  //
  // Example:
  //   For a Mesh with specs ['x', 'y'], the original minor_to_major is [1, 0].
  //   But if the layout is ['y', 'x'], the new minor_to_major is [0, 1].
  std::vector<int32_t> minor_to_major_ordering;
  for (const std::string& shard_spec : layout_shard_specs) {
    if (shard_spec == Layout::kUnshardedDim) {
      return errors::InvalidArgument(
          "Expected a sharded mesh dimension but received an unsharded "
          "dimension.");
    }
    minor_to_major_ordering.insert(minor_to_major_ordering.begin(),
                                   mesh_spec_to_info[shard_spec].index);
  }

  // For each device id increasing from [0, n), compute its multi-dimensional
  // index in the mesh, and then compute its new linear index based on
  // the new minor to major ordering. This will give us the new location
  // in the transposed mesh based on the layout. Intuitively, this is just
  // a permutation function of Layout: Layout can be thought of as how
  // it permutes the pieces of tensors.
  absl::flat_hash_map<int64_t, int64_t> permutation_map;
  for (int device = 0; device < num_devices; ++device) {
    // Compute the multidimensional index from this linear index.
    DeviceLocation dev_loc;

    int offset = device;
    int64 i = mesh_shape.size() - 1;
    while (i >= 0) {
      dev_loc.insert(dev_loc.begin(), offset % mesh_shape[i]);
      offset /= mesh_shape[i];
      --i;
    }

    TF_ASSIGN_OR_RETURN(int64_t linear_index,
                        DeviceLocationToLinearIndex(mesh_shape, dev_loc,
                                                    minor_to_major_ordering));
    permutation_map[linear_index] = device;
  }

  // For each device id increasing from [0, n), use the permutation map to
  // reverse the permutation and linearize the device ordering. This
  // gives us the final tile assignment devices such that it is ordered
  // correctly based on Layout.
  std::vector<int64_t> tile_assignment_devices;
  tile_assignment_devices.reserve(num_devices);
  for (int device = 0; device < num_devices; ++device) {
    tile_assignment_devices.push_back(permutation_map[device]);
  }
  return tile_assignment_devices;
}

}  // namespace

StatusOr<::xla::OpSharding> ConvertLayoutToXlaOpSharding(const Layout& layout) {
  ::xla::OpSharding xla_sharding;

  if (layout.IsFullyReplicated()) {
    xla_sharding.set_type(::xla::OpSharding::REPLICATED);
    return xla_sharding;
  }
  // If not replicated, then this is tile sharded, aka OpSharding::OTHER.
  xla_sharding.set_type(::xla::OpSharding::OTHER);

  // Set Tile Assignment Dimensions by handling both partially sharded and fully
  // sharded.
  int32 product_of_sharded_dimensions = 1;
  for (int32 dim_size : layout.num_shards()) {
    product_of_sharded_dimensions *= dim_size;
    xla_sharding.add_tile_assignment_dimensions(dim_size);
  }

  const Mesh mesh = layout.mesh();

  // Add the (n+1)th dimension representing the replicated group size. This
  // only happens for partially sharded layouts.
  if (product_of_sharded_dimensions != mesh.num_devices()) {
    xla_sharding.add_tile_assignment_dimensions(mesh.num_devices() /
                                                product_of_sharded_dimensions);
    xla_sharding.set_replicate_on_last_tile_dim(true);
  }

  // Set Tile Assignment Devices, handling both partially and fully sharded
  // layouts.
  std::vector<std::string> sharded_layout_specs;
  std::vector<int64_t> sharded_mesh_indices;

  // Extract the non-replicated layout specs and mesh indices.
  for (const std::string& spec : layout.sharding_spec_strs()) {
    if (spec == Layout::kUnshardedDim) continue;
    sharded_layout_specs.push_back(spec);
    sharded_mesh_indices.push_back(mesh.idx_for_dim(spec).value());
  }

  // Create a new sub-mesh based only on the sharded dimensions of `layout`.
  std::vector<MeshDimension> reduced_mesh_dims;
  for (const MeshDimension& mesh_dim : mesh.dims()) {
    if (std::find(sharded_layout_specs.begin(), sharded_layout_specs.end(),
                  mesh_dim.name) != sharded_layout_specs.end()) {
      reduced_mesh_dims.push_back(mesh_dim);
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> tile_assignment_devices,
      ComputeTileAssignmentDevices(sharded_layout_specs, reduced_mesh_dims));

  TF_ASSIGN_OR_RETURN(std::vector<std::vector<int64_t>> replicated_groups,
                      ComputeReplicatedGroups(mesh, sharded_mesh_indices));

  if (tile_assignment_devices.size() != replicated_groups.size()) {
    return errors::Internal(
        "Replicated group size was not equal to the number of tile assignment "
        "devices. Please file a bug to DTensor.",
        "tile_assignment_devices size=", tile_assignment_devices.size(),
        "and replicated_grous size=", replicated_groups.size(),
        "for Layout=", layout.ToString());
  }

  // For partially sharded layouts, we need to expand the
  // tile_assignment_devices based on the replica groups. This is a no-op
  // for fully sharded layouts.
  std::vector<int64_t> expanded_tile_assignment_devices;
  for (int64_t group_index : tile_assignment_devices) {
    for (int64_t device : replicated_groups[group_index]) {
      expanded_tile_assignment_devices.push_back(device);
    }
  }

  // Finally add this to the OpSharding proto.
  for (int64_t device : expanded_tile_assignment_devices) {
    xla_sharding.add_tile_assignment_devices(device);
  }

  return xla_sharding;
}

}  // namespace dtensor
}  // namespace tensorflow
