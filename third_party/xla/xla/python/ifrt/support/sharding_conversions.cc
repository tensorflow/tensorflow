/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/support/sharding_conversions.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace support {

namespace {

// Updates the axes permutation, if necessary, according to the re-expansion of
// collapsed axes indicated by `subdim_info`.
std::vector<int> UpdatePermutation(const std::vector<int>& permutation,
                                   const std::vector<SubDimInfo>& subdim_info) {
  // If no axes were expanded, then the permutation is unchanged.
  if (subdim_info.size() == permutation.size()) {
    return permutation;
  }

  // The updated permutation is produced by applying argsort to the original
  // permutation, then sorting the indices of the sub-dimensions according to
  // the result. (Note that if subdim_info.size() == permutation.size(), this
  // would just apply argsort twice, resulting in the original permutation.)
  // For example, if permutation is [2, 0, 1] and subdim_info.size() is 4,
  // and the values of `reshape_dim_index` in subdim_info are [0, 1, 1, 2],
  // then the updated permutation will be [3, 0, 1, 2] (since the original
  // axis 1 was split into two axes).
  std::vector<int> argsort_perm(permutation.size());
  for (int i = 0; i < permutation.size(); ++i) {
    argsort_perm[permutation[i]] = i;
  }
  std::vector<int> subdim_indices(subdim_info.size());
  std::iota(subdim_indices.begin(), subdim_indices.end(), 0);
  std::sort(subdim_indices.begin(), subdim_indices.end(), [&](int a, int b) {
    int i = subdim_info[a].reshape_dim_index;
    int j = subdim_info[b].reshape_dim_index;
    if (argsort_perm[i] != argsort_perm[j]) {
      return argsort_perm[i] < argsort_perm[j];
    }
    // If the reshape dim indices are the same, then the tile dim indices
    // must be different; otherwise they would comprise the same sub-dimension.
    return subdim_info[a].tile_dim_index < subdim_info[b].tile_dim_index;
  });
  return subdim_indices;
}

struct ShardingInfo {
  std::vector<int64_t> dims;
  std::vector<int64_t> reshape_dims;
  std::vector<int> permutation;
  std::vector<OpSharding::Type> last_tile_dims;
};

// Gets the arguments to construct an OpSharding or HloSharding.
absl::StatusOr<ShardingInfo> GetShardingInfo(
    const ShardingParam& sharding_param) {
  // The first dimensions represent data dimensions (not subgroups) and are
  // equivalent to dim_shards.
  int64_t data_dim_size = 1;
  std::vector<int64_t> dims;
  dims.reserve(sharding_param.dim_shards().size() + 2);
  for (const int64_t dim_shard : sharding_param.dim_shards()) {
    data_dim_size *= dim_shard;
    dims.push_back(dim_shard);
  }

  // Convert axes permutation to major to minor order.
  int num_axis = sharding_param.minor_to_major().permutation.size();
  std::vector<int> permutation;
  permutation.reserve(num_axis);
  for (const int axis_id :
       llvm::reverse(sharding_param.minor_to_major().permutation)) {
    permutation.push_back(num_axis - axis_id - 1);
  }

  // Convert axis sizes to major to minor order.
  std::vector<int64_t> reshape_dims;
  reshape_dims.reserve(num_axis);
  for (auto axis_size :
       llvm::reverse(sharding_param.minor_to_major().axis_sizes)) {
    reshape_dims.push_back(axis_size);
  }

  // Convert unreduced axes to major to minor order and compute unreduced_size.
  std::vector<int> unreduced_axes;
  int64_t unreduced_size = 1;
  unreduced_axes.reserve(sharding_param.unreduced_axes().size());
  for (int axis : llvm::reverse(sharding_param.unreduced_axes())) {
    unreduced_axes.push_back(num_axis - 1 - axis);
    unreduced_size *= reshape_dims[unreduced_axes.back()];
  }

  // Compute the total replicated size (including unreduced axes).
  int64_t replicated_and_unreduced_size =
      sharding_param.NumDevices() / data_dim_size;
  if (replicated_and_unreduced_size == 1) {
    // There are no replicated or unreduced axes.
    return ShardingInfo{
        std::move(dims), std::move(reshape_dims), std::move(permutation), {}};
  }

  dims.push_back(replicated_and_unreduced_size);
  int64_t replicated_size = replicated_and_unreduced_size / unreduced_size;
  if (replicated_size == 1) {
    // There are no replicated axes.
    return ShardingInfo{std::move(dims),
                        std::move(reshape_dims),
                        std::move(permutation),
                        {OpSharding::UNREDUCED}};
  }
  if (unreduced_size == 1) {
    // There are no unreduced axes.
    return ShardingInfo{std::move(dims),
                        std::move(reshape_dims),
                        std::move(permutation),
                        {OpSharding::REPLICATED}};
  }

  // In this branch, there are both replicated and unreduced axes.
  TF_ASSIGN_OR_RETURN(std::vector<SubDimInfo> sub_dim_info,
                      GetOrderedSubDims(dims, reshape_dims, permutation));

  // Update the axis sizes, possibly re-expanding collapsed axes.
  std::vector<int64_t> new_reshape_dims;
  new_reshape_dims.reserve(sub_dim_info.size());
  for (const auto& sub_dim : sub_dim_info) {
    new_reshape_dims.push_back(sub_dim.size);
  }

  std::vector<int> perm_new = UpdatePermutation(permutation, sub_dim_info);

  // Separate the unreduced and replicated axes so that they can be grouped
  // consecutively in the final permutation.
  absl::flat_hash_set<int> unreduced_axes_set(unreduced_axes.begin(),
                                              unreduced_axes.end());
  std::vector<int> new_permutation;
  std::vector<int> new_unreduced_permutation;
  std::vector<int> new_replicated_permutation;
  new_permutation.reserve(perm_new.size());
  new_unreduced_permutation.reserve(unreduced_axes.size());
  new_replicated_permutation.reserve(perm_new.size() - unreduced_axes.size());
  for (int i = 0; i < perm_new.size(); ++i) {
    int index = perm_new[i];
    if (sub_dim_info[index].tile_dim_index < dims.size() - 1) {
      new_permutation.push_back(index);
    } else {
      // The last element of dims represents replicated and unreduced axes.
      if (unreduced_axes_set.contains(sub_dim_info[index].reshape_dim_index)) {
        new_unreduced_permutation.push_back(index);
      } else {
        new_replicated_permutation.push_back(index);
      }
    }
  }
  if (new_unreduced_permutation.size() != unreduced_axes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot convert `ShardingParam` to `HloSharding` with an UNREDUCED "
        "subgroup: ",
        sharding_param.DebugString()));
  }

  std::vector<OpSharding::Type> new_last_tile_dims;
  new_last_tile_dims.reserve(2);

  // Remove the last element of dims, which represents replicated and unreduced
  // axes.
  dims.pop_back();

  // Handle unreduced axes.
  new_last_tile_dims.push_back(OpSharding::UNREDUCED);
  new_permutation.insert(new_permutation.end(),
                         new_unreduced_permutation.begin(),
                         new_unreduced_permutation.end());
  dims.push_back(unreduced_size);

  // Handle replicated axes.
  new_last_tile_dims.push_back(OpSharding::REPLICATED);
  new_permutation.insert(new_permutation.end(),
                         new_replicated_permutation.begin(),
                         new_replicated_permutation.end());
  dims.push_back(replicated_size);
  return ShardingInfo{std::move(dims), std::move(new_reshape_dims),
                      std::move(new_permutation),
                      std::move(new_last_tile_dims)};
}

}  // namespace

absl::StatusOr<OpSharding> ToOpSharding(const ShardingParam& sharding_param) {
  OpSharding op_sharding;
  {
    bool all_dim_replicated = true;
    for (const int64_t dim_shard : sharding_param.dim_shards()) {
      if (dim_shard != 1) {
        all_dim_replicated = false;
        break;
      }
    }
    if (all_dim_replicated) {
      if (sharding_param.unreduced_axes().size() ==
          sharding_param.minor_to_major().axis_sizes.size()) {
        op_sharding.set_type(OpSharding::UNREDUCED);
        return op_sharding;
      }
      if (sharding_param.unreduced_axes().empty()) {
        op_sharding.set_type(OpSharding::REPLICATED);
        return op_sharding;
      }
    }
  }
  op_sharding.set_type(OpSharding::OTHER);
  TF_ASSIGN_OR_RETURN(ShardingInfo sharding_info,
                      GetShardingInfo(sharding_param));

  // Populate tile_assignment_dimensions.
  auto* tile_assignment_dims = op_sharding.mutable_tile_assignment_dimensions();
  tile_assignment_dims->Assign(sharding_info.dims.begin(),
                               sharding_info.dims.end());

  // Populate iota_reshape_dims.
  auto* iota_reshape_dims = op_sharding.mutable_iota_reshape_dims();
  iota_reshape_dims->Assign(sharding_info.reshape_dims.begin(),
                            sharding_info.reshape_dims.end());
  // Populate iota_transpose_perm.
  auto* iota_transpose_perm = op_sharding.mutable_iota_transpose_perm();
  iota_transpose_perm->Assign(sharding_info.permutation.begin(),
                              sharding_info.permutation.end());

  if (sharding_info.last_tile_dims.size() == 1 &&
      sharding_info.last_tile_dims[0] == OpSharding::REPLICATED) {
    op_sharding.set_replicate_on_last_tile_dim(true);
  } else {
    auto* last_tile_dims = op_sharding.mutable_last_tile_dims();
    last_tile_dims->Assign(sharding_info.last_tile_dims.begin(),
                           sharding_info.last_tile_dims.end());
  }

  return op_sharding;
}

absl::StatusOr<xla::HloSharding> ToHloSharding(
    const ShardingParam& sharding_param) {
  if (sharding_param.NumDevices() == 1) {
    // Generate single-device sharding as TileMaximal.
    return xla::HloSharding::Replicate();
  }
  if (!sharding_param.unreduced_axes().empty()) {
    if (sharding_param.unreduced_axes().size() ==
        sharding_param.minor_to_major().axis_sizes.size()) {
      return xla::HloSharding::Unreduced();
    }
  }
  TF_ASSIGN_OR_RETURN(ShardingInfo sharding_info,
                      GetShardingInfo(sharding_param));
  if (sharding_info.last_tile_dims.empty()) {
    return xla::HloSharding::IotaTile(sharding_info.dims,
                                      sharding_info.reshape_dims,
                                      sharding_info.permutation);
  }
  // If the only tile dimension is REPLICATED, we use
  // xla::HloSharding::PartialTile directly to avoid unnecessary
  // canonicalization by HloSharding::Subgroup.
  if (sharding_info.last_tile_dims.size() == 1 &&
      sharding_info.last_tile_dims[0] == OpSharding::REPLICATED) {
    return xla::HloSharding::PartialTile(
        TileAssignment(sharding_info.dims, sharding_info.reshape_dims,
                       sharding_info.permutation));
  }
  return xla::HloSharding::Subgroup(
      TileAssignment(sharding_info.dims, sharding_info.reshape_dims,
                     sharding_info.permutation),
      std::move(sharding_info.last_tile_dims));
}

absl::StatusOr<ShardingParam> ToShardingParam(
    const xla::HloSharding& hlo_sharding, int rank, int num_devices) {
  // `dim_shards` has size equal to the rank of the array, with each entry
  // representing the number of shards for the corresponding dimension.
  // `minor_to_major.permutation` and `minor_to_major.axis_sizes` must be
  // of the same size, and specify how the shards are mapped over the axis in
  // `minor_to_major` order.
  ShardingParam::MinorToMajor minor_to_major;
  if (hlo_sharding.IsReplicated() || hlo_sharding.IsUnreduced() ||
      (hlo_sharding.IsSingleDevice() && num_devices == 1)) {
    // Convert replicated, unreduced, or TileMaximal. Only single-device
    // TileMaximal conversion is supported. These shardings are represented
    // as ShardingParam with a single axis (at index 0) of size num_devices.
    std::vector<int> unreduced_axes;
    if (hlo_sharding.IsUnreduced()) {
      unreduced_axes = {0};
    }
    std::vector<int64_t> dim_shards(rank, 1);
    minor_to_major.permutation.push_back(0);
    minor_to_major.axis_sizes.push_back(num_devices);
    return ShardingParam(std::move(dim_shards), std::move(minor_to_major),
                         std::move(unreduced_axes));
  }
  if (!hlo_sharding.IsTiled()) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported conversion to `ShardingParam` from "
                     "`HloSharding`; sharding=",
                     hlo_sharding.ToString()));
  }

  const xla::TileAssignment& tile_assignment = hlo_sharding.tile_assignment();
  if (!tile_assignment.iota()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Conversion from `HloSharding` without `IotaTileAssignment` is not "
        "supported; sharding=",
        hlo_sharding.ToString()));
  }
  if (rank != hlo_sharding.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`TiledData` expected to have %d dimensions, but has %d "
        "dimensions; sharding=%s",
        rank, hlo_sharding.TiledDataRank(), hlo_sharding.ToString()));
  }
  for (const auto& subgroup_type : hlo_sharding.subgroup_types()) {
    if (subgroup_type != OpSharding::UNREDUCED &&
        subgroup_type != OpSharding::REPLICATED) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported conversion to `ShardingParam` from `HloSharding` that "
          "has a subgroup type other than `UNREDUCED` or `REPLICATED`; "
          "sharding=",
          hlo_sharding.ToString()));
    }
  }
  // Get the `dim_shards` from the tile assignment.
  std::vector<int64_t> dim_shards(tile_assignment.dimensions().begin(),
                                  tile_assignment.dimensions().end());
  if (hlo_sharding.ReplicateOnLastTileDim()) {
    dim_shards.pop_back();
  }
  for (int i = 0; i < hlo_sharding.subgroup_types().size(); ++i) {
    dim_shards.pop_back();
  }
  std::vector<int> unreduced_axes;
  if (tile_assignment.iota()->reshape_dims().empty()) {
    // If there are no reshape_dims, then the array is replicated.
    minor_to_major.permutation.push_back(0);
    minor_to_major.axis_sizes.push_back(num_devices);
  } else if (!hlo_sharding.HasNonReplicatedSubgroup()) {
    for (auto reshape_dim :
         llvm::reverse(tile_assignment.iota()->reshape_dims())) {
      minor_to_major.axis_sizes.push_back(reshape_dim);
    }
    // The devices generated by HloSharding
    // np.arange(ndevices).reshape(reshape_dims).transpose(transpose_perm)
    // must be equal to the devices ShardingParam
    // np.arange(ndevices).reshape(reverse(axis_size)).T.transpose(perm).T
    // Step 1: Compute transpose(transpose_perm).T.
    // Step 2: Compute T.transpose(transpose_perm).T.
    int num_axis = tile_assignment.iota()->transpose_perm().size();
    for (int axis_id :
         llvm::reverse(tile_assignment.iota()->transpose_perm())) {
      minor_to_major.permutation.push_back(num_axis - axis_id - 1);
    }
  } else {
    TF_ASSIGN_OR_RETURN(
        std::vector<SubDimInfo> subdim_info,
        GetOrderedSubDims(tile_assignment.iota()->dims(),
                          tile_assignment.iota()->reshape_dims(),
                          tile_assignment.iota()->transpose_perm()));

    for (const auto& sub_dim : llvm::reverse(subdim_info)) {
      minor_to_major.axis_sizes.push_back(sub_dim.size);
    }

    // Populate unreduced_axes according to the corresponding indices for the
    // unreduced subgroup, if any.
    int num_axis = subdim_info.size();
    auto unreduced_dim_index =
        std::find(hlo_sharding.subgroup_types().begin(),
                  hlo_sharding.subgroup_types().end(), OpSharding::UNREDUCED);
    if (unreduced_dim_index != hlo_sharding.subgroup_types().end()) {
      int unreduced_dim = dim_shards.size() +
                          std::distance(hlo_sharding.subgroup_types().begin(),
                                        unreduced_dim_index);
      for (int i = 0; i < subdim_info.size(); ++i) {
        if (subdim_info[i].tile_dim_index == unreduced_dim) {
          unreduced_axes.push_back(num_axis - i - 1);
        }
      }
    }

    std::vector<int> perm(tile_assignment.iota()->transpose_perm().begin(),
                          tile_assignment.iota()->transpose_perm().end());
    std::vector<int> perm_new = UpdatePermutation(perm, subdim_info);
    for (auto i : llvm::reverse(perm_new)) {
      minor_to_major.permutation.push_back(num_axis - i - 1);
    }
  }
  return ShardingParam(std::move(dim_shards), std::move(minor_to_major),
                       std::move(unreduced_axes));
}

}  // namespace support
}  // namespace ifrt
}  // namespace xla
