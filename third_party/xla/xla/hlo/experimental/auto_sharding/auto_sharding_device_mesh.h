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

#include <cstdint>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "xla/array.h"

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

 private:
  Array<int64_t> device_array_;
  bool is_iota_;
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
