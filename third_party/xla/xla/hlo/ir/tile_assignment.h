/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_HLO_IR_TILE_ASSIGNMENT_H_
#define XLA_HLO_IR_TILE_ASSIGNMENT_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/printer.h"

namespace xla {

class TileAssignment;

// Describes a TileAssignment with a device array generated from reshaping and
// transposing an iota array, a.k.a. HloShardingV2. This is a more scalable
// format for large number of devices since it does not materialize the full
// list of devices, while being less general since it cannot represent
// arbitrary sequence of devices. It is however sufficient to represent the
// most commonly generated SPMD shardings from ML frameworks that arrange
// devices using mesh axes.
class IotaTileAssignment {
 public:
  // Create a trivial (i.e. the device array is a trivial iota without reshape
  // and transpose) IotaTileAssignment with given dimensions.
  static IotaTileAssignment Create(absl::Span<const int64_t> dims);
  // Creates an IotaTileAssignment canonicalizing `reshape_dims` and
  // `transpose_perm`. More specifically the tile assignment array is as if it
  // is produced by the following numpy code:
  //   numpy.arange(math.prod(dims)).reshape(reshape_dims)
  //      .transpose(transpose_perm).reshape(math.prod(dims))
  // where:
  // `dims`: is the dimensions of the tile assignment array.
  // `reshape_dims`: is the dimensions the 1D iota array is reshaped to.
  // `transpose_perm`: is the dimension permutation to transpose `reshape_dims`.
  //
  // e.g. dims=[8,8,8] reshape_dims=[4,2,2], transpose_perm=[0,1,2] (no
  // transpose) corresponds to [8,8,8]<=[16] which in full array V1 format is
  // [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15].
  // e.g. dims=[8,8,8] reshape_dims=[4,2,2], transpose_perm=[1,0,2] (swap dim 0
  // and dim 1) corresponds to [8,8,8]<=[4,2,2]T(1,0,2) which in full array V1
  // format is [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15].
  static IotaTileAssignment Create(absl::Span<const int64_t> dims,
                                   absl::Span<const int64_t> reshape_dims,
                                   absl::Span<const int> transpose_perm);

  ~IotaTileAssignment() = default;
  IotaTileAssignment(const IotaTileAssignment& other);
  IotaTileAssignment(IotaTileAssignment&& other) = default;
  IotaTileAssignment& operator=(const IotaTileAssignment& other);
  IotaTileAssignment& operator=(IotaTileAssignment&& other) = default;

  bool operator==(const IotaTileAssignment& other) const {
    return dims() == other.dims() && reshape_dims() == other.reshape_dims() &&
           transpose_perm() == other.transpose_perm();
  }

  int64_t value_at(absl::Span<const int64_t> index) const;

  int64_t ndims() const { return ndims_; }

  absl::Span<const int64_t> dims() const {
    return absl::MakeSpan(dims_ptr(), ndims_);
  }

  int64_t dim(int n) const { return dims_ptr()[n]; }

  absl::Span<const int64_t> reshape_dims() const {
    return absl::MakeSpan(reshape_dims_ptr(), reshape_ndims_);
  }

  absl::Span<const int> transpose_perm() const {
    return absl::MakeSpan(transpose_perm_ptr(), reshape_ndims_);
  }

  int64_t num_elements() const {
    return absl::c_accumulate(dims(), 1LL, std::multiplies<int64_t>());
  }

  // TODO(b/281892190): This should really not return optional, when we are
  // sure we can handle all cases.
  std::optional<IotaTileAssignment> Transpose(absl::Span<const int> perm) const;

  void Print(Printer* printer) const;

  std::string ToString() const;

 private:
  friend class TileAssignment;
  static constexpr int kPerDimBytes = sizeof(int64_t);
  static constexpr int kPerReshapeDimBytes = sizeof(int64_t) + sizeof(int);

  explicit IotaTileAssignment(absl::Span<const int64_t> dims,
                              absl::Span<const int64_t> reshape_dims,
                              absl::Span<const int> transpose_perm);

  explicit IotaTileAssignment(int ndims, int reshape_ndims);

  int64_t* dims_ptr() { return reinterpret_cast<int64_t*>(storage_.get()); }
  const int64_t* dims_ptr() const {
    return reinterpret_cast<const int64_t*>(storage_.get());
  }
  const int64_t* reshape_dims_ptr() const { return dims_ptr() + ndims_; }
  int64_t* reshape_dims_ptr() {
    return const_cast<int64_t*>(
        const_cast<const IotaTileAssignment*>(this)->reshape_dims_ptr());
  }
  const int* transpose_perm_ptr() const {
    return reinterpret_cast<const int*>(reshape_dims_ptr() + reshape_ndims_);
  }
  int* transpose_perm_ptr() {
    return const_cast<int*>(
        const_cast<const IotaTileAssignment*>(this)->transpose_perm_ptr());
  }

  int size_bytes() const {
    return ndims_ * kPerDimBytes + reshape_ndims_ * kPerReshapeDimBytes;
  }

  bool next_index(absl::Span<int64_t> index) const {
    DCHECK_EQ(index.size(), ndims_);
    for (int64_t i = ndims_ - 1; i >= 0; --i) {
      index[i]++;
      if (index[i] < dims_ptr()[i]) {
        return true;
      }
      index[i] = 0;
    }
    return false;
  }
  int32_t ndims_;
  int32_t reshape_ndims_;
  // Contiguous buffer storing `int64_t dims[]`, `int64_t reshape_dims[]`,
  // `int transpose_perm[]` in order.
  std::unique_ptr<char[]> storage_;
};

// Internal class that represents how an ordered list of device IDs are sharded
// along different dimensions. It manages full or compact representation of the
// device IDs without having callers worry about what underlying format is used.
// This class is meant to be included ONLY by HloSharding so it does not return
// error status on invalid arguments but rather assert preconditions in its
// implementation, assuming it should always get valid data.
// NOTE: This class is immutable.
class TileAssignment {
 public:
  TileAssignment() : array_(ReplicatedArray()) {}
  explicit TileAssignment(std::shared_ptr<const Array<int64_t>> array)
      : shared_array_(std::move(array)), array_(shared_array_.get()) {}
  explicit TileAssignment(int64_t device_id)
      : TileAssignment(std::make_shared<const Array<int64_t>>(
            std::initializer_list<int64_t>{1}, device_id)) {}
  explicit TileAssignment(IotaTileAssignment iota) : iota_(std::move(iota)) {}
  explicit TileAssignment(absl::Span<const int64_t> dims)
      : iota_(IotaTileAssignment::Create(dims)) {}
  explicit TileAssignment(absl::Span<const int64_t> dims,
                          absl::Span<const int64_t> reshape_dims,
                          absl::Span<const int> transpose_perm)
      : iota_(IotaTileAssignment::Create(dims, reshape_dims, transpose_perm)) {}

  bool operator==(const TileAssignment& other) const;
  bool operator!=(const TileAssignment& other) const {
    return !operator==(other);
  }
  // Methods that mirrors those of xla::Array<int64_t>.
  template <typename... Dims>
  typename std::enable_if_t<array_impl::pack_is_integral<Dims...>::value,
                            int64_t>
  operator()(Dims... dims) const {
    DCHECK_EQ(sizeof...(dims), num_dimensions());
    std::array<int64_t, sizeof...(dims)> indexes{
        {static_cast<int64_t>(dims)...}};
    return operator()(indexes);
  }
  int64_t operator()(absl::Span<const int64_t> indexes) const;

  absl::Span<const int64_t> dimensions() const;
  int64_t num_dimensions() const;
  int64_t dim(int64_t n) const;
  int64_t num_elements() const;
  int64_t first() const;

  void Each(
      absl::FunctionRef<void(absl::Span<const int64_t>, int64_t)> f) const;

  Status EachStatus(
      absl::FunctionRef<Status(absl::Span<const int64_t>, int64_t)> f) const;

  // Returns a tile assignment reshaped to the given dimensions.
  // REQUIRES: new shape has the same number of elements.
  [[nodiscard]] TileAssignment Reshape(
      absl::Span<const int64_t> new_dimensions) const;

  // Returns a tile assignment transposd using the given dimension permutations.
  // REQUIRES: `perm` must a an array of num_dimensions elements, with unique
  // values within [0, num_dimensions).
  [[nodiscard]] TileAssignment Transpose(absl::Span<const int> perm) const;

  void Print(Printer* printer) const;

  std::string ToString() const;

  bool UsesDevice(int64_t device) const;

  // Returns non-nullopt iota tile assignment iff it holds that format.
  const std::optional<IotaTileAssignment>& iota() const { return iota_; }
  // Returns reference to the full array representation. If it holds iota
  // format, reference to a lazily materialized array is returned.
  const Array<int64_t>& array() const;
  // Similar to array() but returns the underlying shared_ptr to avoid deep
  // copy.
  const std::shared_ptr<const Array<int64_t>>& shared_array() const;
  // Makes a deep copy of shared_array().
  std::shared_ptr<Array<int64_t>> shared_array_clone() const;

  template <typename H>
  friend H AbslHashValue(H h, const TileAssignment& tile) {
    // TODO(b/281892190): Ideally hashing a TileAssignment should not force iota
    // -> full array conversion, but a requirement is that they should have
    // equivalence. Consider providing a faster hash function for iota tile
    // assignment.
    return H::combine(std::move(h), tile.array());
  }

 private:
  friend class HloSharding;
  // TODO(b/281892190): Consider changing int64_t to int32_t since it's unlikely
  // to have so many devices to overflow int32_t in practice.
  explicit TileAssignment(IotaTileAssignment iota,
                          std::shared_ptr<const Array<int64_t>> shared_array)
      : iota_(std::move(iota)),
        shared_array_(std::move(shared_array)),
        array_(shared_array_.get()) {}

  void MaybeMaterializeFullArray() const;

  static const Array<int64_t>* ReplicatedArray() {
    static auto* array = new Array<int64_t>({0});
    return array;
  }

  std::optional<IotaTileAssignment> iota_;
  // If iota_ is set, shared_array_ is a lazy cache of the materialized array.
  mutable std::shared_ptr<const Array<int64_t>> shared_array_;
  // Pointer to the storage of the fully materialized array format.
  mutable const Array<int64_t>* array_ = nullptr;
};

}  // namespace xla

#endif  // XLA_HLO_IR_TILE_ASSIGNMENT_H_
