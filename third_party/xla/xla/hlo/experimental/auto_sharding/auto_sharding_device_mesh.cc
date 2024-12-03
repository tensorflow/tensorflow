/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_device_mesh.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"

namespace xla {
namespace spmd {

void DeviceMesh::SetValues(absl::Span<const int64_t> values) {
  device_array_.SetValues(values);
  is_iota_ = AreValuesIota(values);
}

// Transpose an array of any number of dimensions given any axes order.
// Similar to numpy.transpose(array, axes=()) function.
template <typename T>
Array<T> Transpose(const Array<T> array, std::vector<int64_t> axes) {
  // Computes transposed array's size.
  std::vector<int64_t> transposed_array_dimensions(array.dimensions().begin(),
                                                   array.dimensions().end());
  for (size_t i = 0; i < axes.size(); i++) {
    transposed_array_dimensions[i] = array.dimensions()[axes[i]];
  }
  Array<T> transposed_array(transposed_array_dimensions);
  std::vector<int64_t> transposed_array_indices(axes.size());
  array.Each([&](absl::Span<const int64_t> indices, T value) {
    for (int i = 0; i < axes.size(); ++i) {
      transposed_array_indices[i] = indices[axes[i]];
    }
    transposed_array(transposed_array_indices) = value;
  });
  return transposed_array;
}

absl::StatusOr<std::vector<int64_t>>
DeviceMesh::GetMeshDimPermutationOrderInShardingSpec(
    const HloSharding& sharding, bool consider_reverse_device_meshes) const {
  MeshDimPermutationOrderCacheKey cache_key(sharding,
                                            consider_reverse_device_meshes);
  if (auto it = mesh_dim_permutation_order_cache_.find(cache_key);
      it != mesh_dim_permutation_order_cache_.end()) {
    return it->second;
  }

  auto check_mesh =
      [&](const Array<int64_t>& mesh) -> std::optional<std::vector<int64_t>> {
    // Permute the dimensions (or axes in numpy term), find the transform that
    // makes tile_assignment == device_mesh.
    std::vector<int64_t> axes(mesh.num_dimensions());
    absl::c_iota(axes, 0);
    do {
      Array<int64_t> transposed_mesh = Transpose(mesh, axes);
      if (std::equal(transposed_mesh.begin(), transposed_mesh.end(),
                     sharding.tile_assignment().array().begin())) {
        return axes;
      }
    } while (absl::c_next_permutation(axes));
    return std::nullopt;
  };

  // This is an expensive search, as we try all possible meshes obtained by
  // reversing a subset of the mesh axes. Reversed shardings only occur due to
  // the somewhat rare kReverse HLO op. The hope therefore is that most calls
  // to the function that reach here will find a mapping within the first
  // iteration of the loop below.
  std::vector<int64_t> axes(num_dimensions());
  size_t num_subsets =
      consider_reverse_device_meshes ? (1 << num_dimensions()) : 1;
  std::vector<int64_t> reverse_dimensions;
  for (size_t i = 0; i < num_subsets; ++i) {
    reverse_dimensions.clear();
    for (size_t j = 0; j < num_dimensions(); ++j) {
      if (i & (1 << j)) {
        reverse_dimensions.push_back(j);
      }
    }
    Array<int64_t> new_mesh(dimensions());
    new_mesh.Each([&](absl::Span<const int64_t> indices, int64_t* device) {
      std::vector<int64_t> original_indices(indices.begin(), indices.end());
      for (int64_t d : reverse_dimensions) {
        original_indices[d] = new_mesh.dim(d) - 1 - original_indices[d];
      }
      *device = (*this)(original_indices);
    });
    if (auto result = check_mesh(new_mesh); result.has_value()) {
      return (mesh_dim_permutation_order_cache_[cache_key] = result.value());
    }
  }
  return absl::NotFoundError(absl::StrCat("Could not find mapping for ",
                                          sharding.ToString(),
                                          " with device mesh ", ToString()));
}

}  // namespace spmd
}  // namespace xla
