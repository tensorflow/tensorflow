/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/utils/hlo_sharding_reconstruction_util.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::StatusOr<ManualShardingInfo> FactorManualSharding(
    absl::Span<const ShardTensor> shards, const HloSharding& sharding) {
  ManualShardingInfo result(sharding);

  if (sharding.UseNamedShardingLeaf()) {
    const NamedSharding& ns = sharding.named_sharding();
    const Mesh& mesh = ns.mesh();
    absl::Span<const AxisRef> manual_axes = ns.manual_axes();

    if (manual_axes.empty()) {
      result.manual_shard_groups[0] =
          std::vector<ShardTensor>(shards.begin(), shards.end());
    } else {
      result.has_manual_sharding = true;
      // Build reverse map device_id -> coordinates
      absl::flat_hash_map<int64_t, std::vector<int64_t>> device_to_coords;
      mesh.device_assignment().Each(
          [&](absl::Span<const int64_t> index, int64_t device) {
            device_to_coords[device] = {index.begin(), index.end()};
          });

      // Group by manual linear index
      for (const auto& shard : shards) {
        auto it = device_to_coords.find(shard.logical_shard_id);
        if (it == device_to_coords.end()) {
          continue;
        }
        const std::vector<int64_t>& coords = it->second;

        int64_t manual_id = 0;
        for (const AxisRef& axis : manual_axes) {
          int64_t axis_idx = axis.mesh_axis_index();
          int64_t coord = coords[axis_idx];
          if (axis.sub_axis_info().has_value()) {
            coord = (coord / axis.pre_size()) % axis.size(mesh);
          }
          manual_id = manual_id * axis.size(mesh) + coord;
        }
        result.manual_shard_groups[manual_id].push_back(shard);
      }

      NamedSharding ns_no_manual(ns.mesh(), ns.dim_shardings(),
                                 ns.replicated_axes(), ns.unreduced_axes(),
                                 /*manual_axes=*/{}, ns.metadata());
      result.unshard_sharding = HloSharding(std::move(ns_no_manual));
    }
  } else if (sharding.IsManual()) {
    // Purely manual V1. Each device is independent.
    result.has_manual_sharding = true;
    for (const auto& shard : shards) {
      result.manual_shard_groups[shard.logical_shard_id].push_back(shard);
    }
    result.unshard_sharding = HloSharding::Replicate();
  } else if (!sharding.IsTuple() &&
             absl::c_linear_search(sharding.subgroup_types(),
                                   OpSharding::MANUAL)) {
    // Check for manual subgroups in V2.
    std::vector<int> manual_subgroup_dims;
    const auto& subgroup_types = sharding.subgroup_types();
    for (int i = 0; i < subgroup_types.size(); ++i) {
      if (subgroup_types[i] == OpSharding::MANUAL) {
        manual_subgroup_dims.push_back(i + sharding.TiledDataRank());
      }
    }

    result.has_manual_sharding = true;
    absl::flat_hash_map<int64_t, std::vector<int64_t>> device_to_tile_indices;
    sharding.EachTile([&](absl::Span<const int64_t> index, int64_t device) {
      device_to_tile_indices[device] = {index.begin(), index.end()};
    });

    for (const auto& shard : shards) {
      auto it = device_to_tile_indices.find(shard.logical_shard_id);
      if (it == device_to_tile_indices.end()) {
        continue;
      }
      const std::vector<int64_t>& tile_indices = it->second;
      int64_t manual_id = 0;
      for (int dim : manual_subgroup_dims) {
        manual_id =
            manual_id * sharding.tile_assignment().dim(dim) + tile_indices[dim];
      }
      result.manual_shard_groups[manual_id].push_back(shard);
    }

    auto subgroup_types_copy = sharding.subgroup_types();
    for (int dim_in_subgroup : manual_subgroup_dims) {
      int i = dim_in_subgroup - sharding.TiledDataRank();
      subgroup_types_copy[i] = OpSharding::REPLICATED;
    }
    result.unshard_sharding =
        HloSharding::Subgroup(sharding.tile_assignment(), subgroup_types_copy);
  } else {
    // Replicated or Tiled without manual components.
    result.manual_shard_groups[0] =
        std::vector<ShardTensor>(shards.begin(), shards.end());
  }

  return result;
}

absl::StatusOr<xla::Literal> UnshardLiteral(
    absl::Span<const ShardTensor> shards, const HloSharding& sharding,
    const Shape& unsharded_shape) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("No shards provided for unshard");
  }
  if (sharding.IsManual()) {
    return absl::InvalidArgumentError(
        "One should not call UnshardLiteral on a manual sharding. Please call "
        "FactorManualSharding first to remove manual components.");
  }

  if (sharding.IsReplicatedOrSingleDevice()) {
    return shards.front().data->Clone();
  }

  xla::Literal unsharded_literal =
      xla::Literal::CreateFromShape(unsharded_shape);

  if (sharding.IsTuple()) {
    return absl::UnimplementedError(
        "Tuple sharding not supported for original value recovery");
  }

  absl::flat_hash_map<int64_t, std::vector<int64_t>> device_to_index;
  sharding.EachTile([&](absl::Span<const int64_t> index, int64_t device) {
    device_to_index[device] = {index.begin(), index.end()};
  });

  bool is_manual_subgroup = sharding.IsManualSubgroup();
  int64_t manual_dim = is_manual_subgroup ? sharding.SubgroupManualDim() : -1;

  for (const auto& shard : shards) {
    auto it = device_to_index.find(shard.logical_shard_id);
    if (it == device_to_index.end()) {
      continue;  // Could be a padded device
    }
    const std::vector<int64_t>& tile_indices = it->second;

    // For manual subgroups, we just need one of the replicated manual groups to
    // reconstruct the full tensor structure.
    if (is_manual_subgroup) {
      bool should_process = true;
      for (int i = 0; i < tile_indices.size(); ++i) {
        if (i == manual_dim && tile_indices[i] != 0) {
          should_process = false;
          break;
        }
      }
      if (!should_process) {
        continue;
      }
    }

    std::vector<int64_t> start_indices(unsharded_shape.dimensions().size(), 0);
    // Find the offset for each dimension based on tile indices and tile shapes
    for (int i = 0; i < unsharded_shape.dimensions().size(); ++i) {
      int64_t dim = unsharded_shape.dimensions(i);
      int64_t partitions = sharding.dimension(i);
      int64_t tile_size = (dim + partitions - 1) / partitions;
      start_indices[i] = tile_indices[i] * tile_size;
    }

    // Copy a slice into bounds.
    std::vector<int64_t> zero_start(shard.data->shape().dimensions().size(), 0);

    // Limit dimensions to fit in actual unsharded shape if padded
    std::vector<int64_t> copy_dims(unsharded_shape.dimensions().size(), 0);
    for (int i = 0; i < unsharded_shape.dimensions().size(); ++i) {
      copy_dims[i] = std::min(shard.data->shape().dimensions(i),
                              unsharded_shape.dimensions(i) - start_indices[i]);
    }

    // We can do a dynamic slice and dynamic update slice or just CopySliceFrom
    TF_RETURN_IF_ERROR(
        unsharded_literal.CopySliceFrom(*shard.data,
                                        /*src_base=*/zero_start,
                                        /*dest_base=*/start_indices,
                                        /*copy_size=*/copy_dims));
  }
  return unsharded_literal;
}

}  // namespace xla
