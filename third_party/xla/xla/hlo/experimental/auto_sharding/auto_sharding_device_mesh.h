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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_DEVICE_MESH_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_DEVICE_MESH_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"

namespace xla {
namespace spmd {
class DeviceMesh {
 public:
  explicit DeviceMesh(absl::Span<const int64_t> sizes)
      : device_array_(sizes), is_iota_(false) {}

  void FillIota(const int64_t value) {
    device_array_.FillIota(value);
    is_iota_ = true;
  }

  void SetValues(absl::Span<const int64_t> values);

  bool IsIota() const { return is_iota_; }

  const Array<int64_t>& DeviceArray() const { return device_array_; }

  int64_t num_dimensions() const { return device_array_.num_dimensions(); }

  // Returns the size of the dimension at the given index.
  int64_t dim(int64_t n) const { return device_array_.dim(n); }

  // Returns a vector containing the dimensions of the array.
  absl::Span<const int64_t> dimensions() const {
    return device_array_.dimensions();
  }

  // Returns the total number of elements in the array.
  int64_t num_elements() const { return device_array_.num_elements(); }

  std::string ToString() const { return device_array_.ToString(); }

  void Reshape(absl::Span<const int64_t> new_dimensions) {
    device_array_.Reshape(new_dimensions);
  }

  void TransposeDimensions(absl::Span<const int> permutation) {
    device_array_.TransposeDimensions(permutation);
    is_iota_ = false;
  }

  const int64_t& operator()(absl::Span<const int64_t> indexes) const {
    return device_array_(indexes);
  }

  int64_t& operator()(absl::Span<const int64_t> indexes) {
    return device_array_(indexes);
  }

  void Each(absl::FunctionRef<void(absl::Span<const int64_t>, int64_t*)> f) {
    device_array_.Each(f);
  }

  void Each(
      absl::FunctionRef<void(absl::Span<const int64_t>, int64_t)> f) const {
    device_array_.Each(f);
  }

  absl::StatusOr<std::vector<int64_t>> GetMeshDimPermutationOrderInShardingSpec(
      const HloSharding& sharding, bool consider_reverse_device_meshes) const;

 private:
  Array<int64_t> device_array_;
  bool is_iota_;

  class MeshDimPermutationOrderCacheKey {
   public:
    MeshDimPermutationOrderCacheKey(const HloSharding& sharding,
                                    bool consider_reverse_device_meshes)
        : sharding_string_(sharding.ToString()),
          consider_reverse_device_meshes_(consider_reverse_device_meshes) {}

    bool operator==(const MeshDimPermutationOrderCacheKey& other) const {
      return this->sharding_string_ == other.sharding_string_ &&
             this->consider_reverse_device_meshes_ ==
                 other.consider_reverse_device_meshes_;
    };

    template <typename H>
    friend H AbslHashValue(H h, const MeshDimPermutationOrderCacheKey& key) {
      return H::combine(std::move(h), key.sharding_string_,
                        key.consider_reverse_device_meshes_);
    }

   private:
    // NB: We use sharding.ToString() instead of key.sharding as the latter will
    // materialize the tile assignment array of the sharding (if it is V2, as
    // are a majority of our sharding objects). This is necessary has a sharding
    // can have a V1 or a V2 representation. Hashing the ToString repr of the
    // sharding is much faster as it won't materialize the tile assignment
    // array. This can, however, mean that equivalent shardings can have
    // different hash values. In this case, this is okay, as a cache miss will
    // merely invoke the function again, and the faster hashing more than
    // compensates for the potentially lower hit rate.
    const std::string sharding_string_;
    bool consider_reverse_device_meshes_;
  };

  mutable absl::flat_hash_map<MeshDimPermutationOrderCacheKey,
                              std::vector<int64_t>>
      mesh_dim_permutation_order_cache_;
};

template <class T>
inline bool AreValuesIota(absl::Span<const T> values) {
  for (int i = 1; i < values.size(); ++i) {
    if (values[i] - values[i - 1] != 1) {
      return false;
    }
  }
  return true;
}

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_DEVICE_MESH_H_
