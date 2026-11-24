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

#include "xla/hlo/ir/tile_assignment.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/overflow_util.h"
#include "xla/printer.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

// Helper function to canonicalize reshape_dims and transpose_perm of an
// IotaTileAssignment, below shows some examples of the process of
// canonicalization, the format is [reshape_dims]T(transpose_perm),
// transpose_perm can be omitted if transpose is noop.
//
// [3,4,5] => [12,1,5] => [12,5] => [60,1] => [60]
//
// [3,4,5]T(2,1,0)
//
// [3,4,5]T(1,2,0) => [3,20]T(1,0)
//
// [3,4,5]T(1,0,2)
//
// [3,4,5]T(2,0,1) => [12,5]T(1,0)
//
// [3,4,5]T(2,1,0)
//
// [1,3,1,4,1,5]T(4,3,2,5,1,0) => [3,4,5]T(1,2,0) => [3,20,1]T(1,0,2) =>
// [3,20]T(1,0)
void CanonicalizeIotaDims(absl::Span<int64_t>& dims, absl::Span<int>& perm) {
  DCHECK_EQ(dims.size(), perm.size());
  if (dims.size() <= 1) {
    return;
  }
  absl::InlinedVector<int, 6> old_to_new_dims(dims.size());
  while (true) {
    bool changed = false;
    // Remove all dimensions of size one.
    int new_ndims = 0;
    for (int i = 0; i < dims.size(); ++i) {
      if (dims[i] == 1) {
        old_to_new_dims[i] = -1;
      } else {
        old_to_new_dims[i] = new_ndims;
        ++new_ndims;
      }
    }
    if (new_ndims != dims.size()) {
      for (int i = 0, new_idx = 0; i < dims.size(); ++i) {
        int new_dim = old_to_new_dims[i];
        if (new_dim >= 0) {
          dims[new_dim] = dims[i];
        }

        int new_perm_dim = old_to_new_dims[perm[i]];
        if (new_perm_dim >= 0) {
          perm[new_idx] = new_perm_dim;
          ++new_idx;
          DCHECK_LE(new_idx, new_ndims);
        }
      }
      perm = perm.subspan(0, new_ndims);
      dims = dims.subspan(0, new_ndims);
    }
    // Merge subranges of dimensions that are major to minor order into single
    // dimensions of size of their product. The merged dimension is placed at
    // the first dimension of the subrange, and the other merged dimensions
    // are set to 1, which are then removed. `remove_one_dims` is always
    // called right before this, so it can assume there is no size one
    // dimension.
    for (int i = 1, base = 0, n = dims.size(); i < n; ++i) {
      const int base_dim = perm[base];
      const int dim = perm[i];
      if (dim - base_dim == i - base) {
        dims[base_dim] *= dims[dim];
        dims[dim] = 1;
        changed = true;
      } else {
        base = i;
      }
    }
    if (!changed) {
      break;
    }
  }
}

enum class TransposeKind {
  kNoop,       // Nothing to do.
  kReshape,    // Flat array is identical but degenerate shapes have moved.
  kTranspose,  // Regular transpose.
};

TransposeKind GetTransposeKind(absl::Span<const int64_t> dims,
                               absl::Span<const int> perm) {
  TransposeKind kind = TransposeKind::kNoop;
  int prev_non_one_dim = -1;
  for (int i = 0; i < perm.size(); ++i) {
    const auto& d = perm[i];
    if (dims[d] == 1) {
      if (d != i && dims[i] != 1) {
        kind = TransposeKind::kReshape;
      }
      continue;
    }
    if (d <= prev_non_one_dim) {
      return TransposeKind::kTranspose;
    }
    prev_non_one_dim = d;
  }
  return kind;
}

std::optional<IotaTileAssignment> TryToConvertV1ToV2(const TileAssignment& T) {
  if (T.iota().has_value()) {
    return T.iota();
  }
  int64_t rank = T.num_dimensions();
  if (rank == 0) {
    return std::nullopt;
  }

  // An iota tile assignment is linear: T(x) = sum(x_d * stride_d).
  // Thus, the physical stride for dimension d is T(0,...,1,...,0)
  // (with 1 at d).
  // Sorting these strides reconstructs the transposed layout's dimension order.
  // Read strides for each dimension.
  std::vector<std::pair<int64_t, int>> dim_strides;
  dim_strides.reserve(rank);
  for (int d = 0; d < rank; ++d) {
    if (T.dim(d) > 1) {
      std::vector<int64_t> idx(rank, 0);
      idx[d] = 1;
      dim_strides.push_back({T(idx), d});
    } else {
      dim_strides.push_back({0, d});
    }
  }

  // Sort dimensions by stride in descending order to get major-to-minor layout.
  absl::c_sort(dim_strides,
               [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<int> perm(rank);
  std::vector<int64_t> transposed_dims(rank);
  for (int i = 0; i < rank; ++i) {
    perm[i] = dim_strides[i].second;
    transposed_dims[i] = T.dim(perm[i]);
  }

  // Compute expected strides for this permutation.
  std::vector<int64_t> expected_strides(rank);
  int64_t current_stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    expected_strides[i] = current_stride;
    current_stride *= transposed_dims[i];
  }

  // Verify the permutation.
  absl::Status matches_iota = T.EachStatus(
      [&](absl::Span<const int64_t> index_of_T, int64_t device_id) {
        int64_t flat_index = 0;
        for (int k = 0; k < perm.size(); ++k) {
          flat_index += index_of_T[perm[k]] * expected_strides[k];
        }
        return device_id == flat_index ? absl::OkStatus()
                                       : absl::InvalidArgumentError("Mismatch");
      });

  if (matches_iota.ok()) {
    IotaTileAssignment transposed_Iota =
        IotaTileAssignment::Create(transposed_dims);
    std::vector<int> inv_perm(rank);
    for (int i = 0; i < rank; ++i) {
      inv_perm[perm[i]] = i;
    }
    return transposed_Iota.Transpose(inv_perm);
  }

  return std::nullopt;
}

absl::StatusOr<int64_t> ValidateIotaDims(absl::Span<const int64_t> dims) {
  constexpr size_t kMaxIotaRank =
      std::min<size_t>(std::numeric_limits<int>::max(),
                       std::numeric_limits<std::ptrdiff_t>::max() /
                           (2 * sizeof(int64_t) + sizeof(int)));
  if (dims.size() > kMaxIotaRank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Iota dimensions rank (%d) exceeds maximum (%d)",
                        dims.size(), kMaxIotaRank));
  }
  bool has_zero = false;
  int64_t nonzero_product = 1;
  for (int64_t dim : dims) {
    if (dim < 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Iota dimensions must be nonnegative, got %d", dim));
    }
    if (dim == 0) {
      has_zero = true;
      continue;
    }
    auto [next, overflow] = OverflowSafeMultiply(nonzero_product, dim);
    if (overflow) {
      return absl::InvalidArgumentError(
          "Product of iota dimensions overflows int64_t");
    }
    nonzero_product = next;
  }
  return has_zero ? 0 : nonzero_product;
}

absl::Status ValidateIotaReshapeAndTranspose(
    absl::Span<const int64_t> reshape_dims,
    absl::Span<const int> transpose_perm, int64_t expected_elements) {
  if (reshape_dims.empty()) {
    return absl::InvalidArgumentError("Iota reshape_dims must not be empty");
  }
  if (reshape_dims.size() != transpose_perm.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Iota reshape_dims size (%d) must match transpose_perm size (%d)",
        reshape_dims.size(), transpose_perm.size()));
  }
  ABSL_ASSIGN_OR_RETURN(const int64_t reshape_elements,
                   ValidateIotaDims(reshape_dims));
  if (reshape_elements != expected_elements) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Product of iota reshape_dims (%d) must match expected element count "
        "(%d)",
        reshape_elements, expected_elements));
  }
  absl::InlinedVector<bool, 6> seen_perm(transpose_perm.size(), false);
  for (int perm_dim : transpose_perm) {
    if (perm_dim < 0 ||
        static_cast<size_t>(perm_dim) >= transpose_perm.size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Iota transpose_perm entry %d out of range [0, %d)",
                          perm_dim, transpose_perm.size()));
    }
    if (seen_perm[perm_dim]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Iota transpose_perm entry %d is not unique", perm_dim));
    }
    seen_perm[perm_dim] = true;
  }
  return absl::OkStatus();
}

}  // namespace

/*static*/ absl::StatusOr<IotaTileAssignment> IotaTileAssignment::TryCreate(
    absl::Span<const int64_t> dims) {
  ABSL_ASSIGN_OR_RETURN(const int64_t product, ValidateIotaDims(dims));
  return IotaTileAssignment(dims, {product}, {0});
}

/*static*/ IotaTileAssignment IotaTileAssignment::Create(
    absl::Span<const int64_t> dims) {
  absl::StatusOr<IotaTileAssignment> iota = TryCreate(dims);
  CHECK_OK(iota);
  return *std::move(iota);
}

/*static*/ absl::StatusOr<IotaTileAssignment> IotaTileAssignment::TryCreate(
    absl::Span<const int64_t> dims, absl::Span<const int64_t> reshape_dims,
    absl::Span<const int> transpose_perm) {
  ABSL_ASSIGN_OR_RETURN(const int64_t expected_elements, ValidateIotaDims(dims));
  ABSL_RETURN_IF_ERROR(ValidateIotaReshapeAndTranspose(reshape_dims, transpose_perm,
                                                  expected_elements));
  if (expected_elements == 0) {
    // All tile assignments without elements are equivalent, canonicalize them
    // the same way as the single-argument `TryCreate`.
    return IotaTileAssignment(dims, {int64_t{0}}, {0});
  }
  absl::InlinedVector<int64_t, 6> canonicalized_dims(reshape_dims.begin(),
                                                     reshape_dims.end());
  absl::InlinedVector<int, 6> canonicalized_perm(transpose_perm.begin(),
                                                 transpose_perm.end());
  auto dims_span = absl::MakeSpan(canonicalized_dims);
  auto perm_span = absl::MakeSpan(canonicalized_perm);
  CanonicalizeIotaDims(dims_span, perm_span);
  if (dims_span.empty()) {
    canonicalized_dims[0] = 1;
    dims_span = absl::MakeSpan(canonicalized_dims.data(), 1);
    canonicalized_perm[0] = 0;
    perm_span = absl::MakeSpan(canonicalized_perm.data(), 1);
  }
  return IotaTileAssignment(dims, dims_span, perm_span);
}

/*static*/ IotaTileAssignment IotaTileAssignment::Create(
    absl::Span<const int64_t> dims, absl::Span<const int64_t> reshape_dims,
    absl::Span<const int> transpose_perm) {
  absl::StatusOr<IotaTileAssignment> iota =
      TryCreate(dims, reshape_dims, transpose_perm);
  CHECK_OK(iota);
  return *std::move(iota);
}

/*static*/ IotaTileAssignment IotaTileAssignment::Create(
    absl::Span<const int64_t> dims, const MeshProto::IotaTransform& transform) {
  // This is an extra copy, but the array will typically be quite small since
  // it scales with the number of dimensions.
  absl::InlinedVector<int, 6> int_perm(transform.transpose_perm().begin(),
                                       transform.transpose_perm().end());
  return Create(dims, transform.reshape_dims(), absl::MakeSpan(int_perm));
}

// Materializes array representation of IotaTileAssignment.
Array<int64_t> ToArray(absl::Span<const int64_t> reshape_dims,
                       absl::Span<const int> transpose_perm,
                       absl::Span<const int64_t> dims) {
  Array<int64_t> array(reshape_dims);
  array.FillIota(0);
  array.TransposeDimensions(transpose_perm);
  array.Reshape(dims);
  return array;
}

Array<int64_t> IotaTileAssignment::ToArray() const {
  return ::xla::ToArray(reshape_dims(), transpose_perm(), dims());
}

IotaTileAssignment::IotaTileAssignment(const IotaTileAssignment& other)
    : IotaTileAssignment(other.ndims_, other.reshape_ndims_) {
  const size_t size = size_bytes();
  if (size > 0) {
    std::memcpy(storage_.get(), other.storage_.get(), size);
  }
}

IotaTileAssignment& IotaTileAssignment::operator=(
    const IotaTileAssignment& other) {
  if (this == &other) {
    return *this;
  }
  const size_t new_size = other.size_bytes();
  if (size_bytes() != new_size) {
    storage_.reset(new_size > 0 ? new char[new_size] : nullptr);
  }
  ndims_ = other.ndims_;
  reshape_ndims_ = other.reshape_ndims_;
  if (new_size > 0) {
    std::memcpy(storage_.get(), other.storage_.get(), new_size);
  }
  return *this;
}

IotaTileAssignment::IotaTileAssignment(absl::Span<const int64_t> dims,
                                       absl::Span<const int64_t> reshape_dims,
                                       absl::Span<const int> transpose_perm)
    : IotaTileAssignment(dims.size(), reshape_dims.size()) {
  DCHECK_EQ(reshape_dims.size(), transpose_perm.size());
  if (ndims_ > 0) {
    std::memcpy(dims_ptr(), dims.data(),
                static_cast<size_t>(ndims_) * sizeof(int64_t));
  }
  DCHECK_EQ(num_elements(), Product(reshape_dims));
  if (reshape_ndims_ > 0) {
    std::memcpy(reshape_dims_ptr(), reshape_dims.data(),
                static_cast<size_t>(reshape_ndims_) * sizeof(int64_t));
    std::memcpy(transpose_perm_ptr(), transpose_perm.data(),
                static_cast<size_t>(reshape_ndims_) * sizeof(int));
  }
}

IotaTileAssignment::IotaTileAssignment(int ndims, int reshape_ndims)
    : ndims_(ndims),
      reshape_ndims_(reshape_ndims),
      storage_(size_bytes() > 0 ? new char[size_bytes()] : nullptr) {
  DCHECK_GE(ndims_, 0);
  DCHECK_GE(reshape_ndims_, 0);
}

std::optional<IotaTileAssignment> IotaTileAssignment::Transpose(
    absl::Span<const int> perm) const {
  DCHECK_EQ(ndims_, perm.size());
  auto dims = this->dims();
  const TransposeKind kind = GetTransposeKind(dims, perm);
  if (kind == TransposeKind::kNoop) {
    return *this;
  }
  absl::InlinedVector<int64_t, 6> new_dims(ndims_);
  for (int64_t i = 0; i < ndims_; ++i) {
    new_dims[i] = dims[perm[i]];
  }
  if (absl::c_linear_search(dims, 0)) {
    // All tile assignments without elements are equivalent.
    return IotaTileAssignment::Create(new_dims);
  }
  if (kind == TransposeKind::kReshape) {
    return IotaTileAssignment::Create(new_dims, reshape_dims(),
                                      transpose_perm());
  }
  if (reshape_ndims_ == 1) {
    return IotaTileAssignment::Create(new_dims, dims, perm);
  }
  bool is_pure_transpose = true;
  absl::InlinedVector<int64_t, 6> non_one_dims;
  absl::InlinedVector<int, 6> one_to_non_one(ndims_);
  non_one_dims.reserve(ndims_);
  auto reshape_dims = this->reshape_dims();
  auto transpose_perm = this->transpose_perm();
  for (int i = 0; i < ndims_; ++i) {
    const int64_t dim = dims[i];
    if (dim == 1) {
      one_to_non_one[i] = -1;
      continue;
    }
    if (non_one_dims.size() >= reshape_ndims_ ||
        reshape_dims[transpose_perm[non_one_dims.size()]] != dim) {
      is_pure_transpose = false;
    }
    one_to_non_one[i] = non_one_dims.size();
    non_one_dims.push_back(dims[i]);
  }
  if (is_pure_transpose) {
    CHECK_EQ(reshape_ndims_, non_one_dims.size());
    absl::InlinedVector<int, 6> new_perm;
    new_perm.reserve(non_one_dims.size());
    for (int i = 0; i < ndims_; ++i) {
      if (dims[perm[i]] == 1) {
        continue;
      }
      new_perm.push_back(transpose_perm[one_to_non_one[perm[i]]]);
    }
    CHECK_EQ(reshape_ndims_, new_perm.size());
    return IotaTileAssignment::Create(new_dims, reshape_dims, new_perm);
  }

  // Find the shortest common refinement of `non_one_dims` and the reshape dims
  // permuted by `transpose_perm`. Each resulting sub-dimension belongs to
  // exactly one reshape dimension and one tile dimension, so the transposed
  // tile assignment can be expressed by permuting the groups of sub-dimensions
  // that make up each tile dimension. This only needs divisibility checks
  // between dimension sizes, so it runs in time linear in the rank and cannot
  // overflow.
  absl::StatusOr<std::vector<SubDimInfo>> ordered_sub_dims =
      GetOrderedSubDims(dims, reshape_dims, transpose_perm);
  if (!ordered_sub_dims.ok()) {
    // TODO(b/341371396): Handle remaining patterns and remove nullopt path.
    // It seems this cannot happen under the valid condition that we generate
    // code with predefined mesh axises, but the C++ API does not restrict
    // people from constructing sharding like `[2,3]<=[2,3]T(1,0]` which
    // breaks the axises and transposing it will result in V1 sharding.
    return std::nullopt;
  }
  const std::vector<SubDimInfo>& sub_dims = *ordered_sub_dims;
  // `sub_dims` is sorted by (reshape_dim_index, tile_dim_index), which is the
  // major-to-minor order of the refined reshape dimensions.
  absl::InlinedVector<int64_t, 6> new_reshape_dims;
  new_reshape_dims.reserve(sub_dims.size());
  absl::InlinedVector<absl::InlinedVector<int, 2>, 6> grouped_reshape_dims(
      ndims_);
  for (int i = 0; i < sub_dims.size(); ++i) {
    const SubDimInfo& sub_dim = sub_dims[i];
    new_reshape_dims.push_back(sub_dim.size);
    grouped_reshape_dims[sub_dim.tile_dim_index].push_back(i);
  }
  absl::InlinedVector<int, 6> new_transpose_perm;
  new_transpose_perm.reserve(sub_dims.size());
  for (int i = 0; i < ndims_; ++i) {
    auto& group = grouped_reshape_dims[perm[i]];
    // Order the sub-dimensions of each tile dimension from major to minor.
    absl::c_sort(group, [&](int a, int b) {
      return sub_dims[a].tile_sub_dim_index > sub_dims[b].tile_sub_dim_index;
    });
    new_transpose_perm.insert(new_transpose_perm.end(), group.begin(),
                              group.end());
  }
  CHECK_EQ(new_transpose_perm.size(), new_reshape_dims.size());
  return IotaTileAssignment::Create(new_dims, new_reshape_dims,
                                    new_transpose_perm);
}

void IotaTileAssignment::PrintArray(Printer* printer) const {
  printer->Append("[");
  AppendJoin(printer, reshape_dims(), ",");
  printer->Append("]");
  if (reshape_ndims_ > 1) {
    printer->Append("T(");
    AppendJoin(printer, transpose_perm(), ",");
    printer->Append(")");
  }
}

void IotaTileAssignment::Print(Printer* printer) const {
  printer->Append("[");
  AppendJoin(printer, dims(), ",");
  printer->Append("]<=");
  PrintArray(printer);
}

std::string IotaTileAssignment::ArrayToString() const {
  StringPrinter printer;
  PrintArray(&printer);
  return std::move(printer).ToString();
}

std::string IotaTileAssignment::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

int64_t IotaTileAssignment::value_at(absl::Span<const int64_t> index) const {
  DCHECK_EQ(index.size(), ndims_);
  int64_t linear_index = 0;
  auto dims = this->dims();
  for (int64_t i = 0; i < ndims_; ++i) {
    linear_index *= dims[i];
    linear_index += index[i];
  }
  auto reshape_dims = this->reshape_dims();
  auto transpose_perm = this->transpose_perm();
  absl::InlinedVector<int64_t, 6> reshape_index(reshape_ndims_);
  for (int64_t i = reshape_ndims_ - 1; i >= 0; --i) {
    const int dim = transpose_perm[i];
    const int64_t dim_size = reshape_dims[dim];
    reshape_index[dim] = linear_index % dim_size;
    linear_index /= dim_size;
  }
  int64_t value = reshape_index[0];
  for (int64_t i = 1; i < reshape_ndims_; ++i) {
    value *= reshape_dims[i];
    value += reshape_index[i];
  }
  return value;
}

std::vector<int64_t> IotaTileAssignment::index_for(int64_t device_index) const {
  const auto dims = this->dims();
  const auto reshape_dims = this->reshape_dims();
  const auto transpose_perm = this->transpose_perm();
  absl::InlinedVector<int64_t, 6> reshaped_index(reshape_dims.size());
  for (int64_t i = reshape_dims.size() - 1; i >= 0; --i) {
    const int64_t dim_size = reshape_dims[i];
    reshaped_index[i] = device_index % dim_size;
    device_index /= dim_size;
  }
  DCHECK_EQ(device_index, 0);
  int64_t flat_index = reshaped_index[transpose_perm[0]];
  for (int64_t i = 1; i < reshape_dims.size(); ++i) {
    const int64_t dim = transpose_perm[i];
    flat_index *= reshape_dims[dim];
    flat_index += reshaped_index[dim];
  }
  std::vector<int64_t> tile_index(ndims_);
  for (int64_t i = ndims_ - 1; i >= 0; --i) {
    const int64_t dim_size = dims[i];
    tile_index[i] = flat_index % dim_size;
    flat_index /= dim_size;
  }
  DCHECK_EQ(flat_index, 0);
  return tile_index;
}

TileAssignment::TileAssignment(const TileAssignment& other) {
  iota_ = other.iota_;
  absl::MutexLock other_lock(other.mu_);
  shared_array_ = other.shared_array_;
  array_.store(other.array_ptr(), std::memory_order_relaxed);
}

TileAssignment::TileAssignment(TileAssignment&& other) {
  absl::MutexLock other_lock(other.mu_);
  iota_ = std::move(other.iota_);
  shared_array_ = std::move(other.shared_array_);
  array_.store(other.array_ptr(), std::memory_order_relaxed);
  other.array_.store(nullptr, std::memory_order_relaxed);
}

TileAssignment& TileAssignment::operator=(const TileAssignment& other) {
  iota_ = other.iota_;
  std::shared_ptr<const Array<int64_t>> shared_array;
  const Array<int64_t>* array;
  {
    absl::MutexLock other_lock(other.mu_);
    shared_array = other.shared_array_;
    array = other.array_ptr();
  }
  absl::MutexLock lock(mu_);
  shared_array_ = std::move(shared_array);
  array_.store(array, std::memory_order_relaxed);
  return *this;
}

TileAssignment& TileAssignment::operator=(TileAssignment&& other) {
  if (this == &other) {
    return *this;
  }
  iota_ = std::move(other.iota_);
  std::shared_ptr<const Array<int64_t>> shared_array;
  const Array<int64_t>* array;
  {
    absl::MutexLock other_lock(other.mu_);
    shared_array = std::move(other.shared_array_);
    array = other.array_ptr();
    other.array_.store(nullptr, std::memory_order_relaxed);
  }
  absl::MutexLock lock(mu_);
  shared_array_ = std::move(shared_array);
  array_.store(array, std::memory_order_relaxed);
  return *this;
}

bool TileAssignment::operator==(const TileAssignment& other) const {
  if (iota_ && other.iota_) {
    return *iota_ == *other.iota_;
  }
  return array() == other.array();
}

int64_t TileAssignment::operator()(absl::Span<const int64_t> indexes) const {
  if (const auto* arr = array_ptr(); arr != nullptr) {
    return (*arr)(indexes);
  }
  return iota_->value_at(indexes);
}

std::vector<int64_t> TileAssignment::index_for(int64_t device) const {
  if (iota_.has_value()) {
    return iota_->index_for(device);
  }

  const Array<int64_t>& arr = array();
  auto it = absl::c_find(arr, device);
  CHECK_NE(it, arr.end());

  int64_t flat_index = it - arr.begin();
  std::vector<int64_t> tile_index(arr.num_dimensions());
  for (int64_t i = arr.num_dimensions() - 1; i >= 0; --i) {
    tile_index[i] = flat_index % arr.dim(i);
    flat_index /= arr.dim(i);
  }
  DCHECK_EQ(flat_index, 0);
  return tile_index;
}

absl::Span<const int64_t> TileAssignment::dimensions() const {
  return iota_.has_value() ? iota_->dims() : array().dimensions();
}

int64_t TileAssignment::num_dimensions() const {
  return iota_.has_value() ? iota_->ndims() : array().num_dimensions();
}

int64_t TileAssignment::dim(int64_t n) const {
  return iota_.has_value() ? iota_->dim(n) : array().dim(n);
}
int64_t TileAssignment::num_elements() const {
  return array_ptr() != nullptr ? array().num_elements()
                                : iota_->num_elements();
}

int64_t TileAssignment::first() const {
  return array_ptr() != nullptr ? *array().begin() : 0;
}

void TileAssignment::Each(
    absl::FunctionRef<void(absl::Span<const int64_t>, int64_t)> f) const {
  array().Each(f);
}

absl::Status TileAssignment::EachStatus(
    absl::FunctionRef<absl::Status(absl::Span<const int64_t>, int64_t)> f)
    const {
  return array().EachStatus(f);
}

[[nodiscard]] TileAssignment TileAssignment::Reshape(
    absl::Span<const int64_t> new_dimensions) const {
  if (iota_) {
    CHECK_EQ(Product(new_dimensions), iota_->num_elements());
    return TileAssignment(new_dimensions, iota_->reshape_dims(),
                          iota_->transpose_perm());
  }
  std::shared_ptr<Array<int64_t>> reshaped = shared_array_clone();
  reshaped->Reshape(new_dimensions);
  return TileAssignment(std::move(reshaped));
}

[[nodiscard]] TileAssignment TileAssignment::Transpose(
    absl::Span<const int> perm) const {
  const TransposeKind kind = GetTransposeKind(dimensions(), perm);
  if (kind == TransposeKind::kNoop) {
    return *this;
  }
  if (iota_) {
    auto transposed = iota_->Transpose(perm);
    if (transposed) {
      return TileAssignment(std::move(*transposed));
    }
  }
  auto cloned_array = shared_array_clone();
  cloned_array->TransposeDimensions(perm);
  return TileAssignment(std::move(cloned_array));
}

void TileAssignment::PrintArray(Printer* printer) const {
  if (iota_) {
    iota_->PrintArray(printer);
  } else {
    AppendJoin(printer, array(), ",");
  }
}

void TileAssignment::Print(Printer* printer) const {
  if (iota_) {
    printer->Append("devices=");
    iota_->Print(printer);
  } else {
    printer->Append("devices=[");
    AppendJoin(printer, array().dimensions(), ",");
    printer->Append("]");
    AppendJoin(printer, array(), ",");
  }
}

std::string TileAssignment::ArrayToString() const {
  StringPrinter printer;
  PrintArray(&printer);
  return std::move(printer).ToString();
}

std::string TileAssignment::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

const Array<int64_t>& TileAssignment::array() const {
  auto* arr = array_ptr();
  if (arr != nullptr) {
    return *arr;
  }

  {
    absl::MutexLock lock(mu_);
    MaybeMaterializeFullArray();
  }

  arr = array_ptr();
  DCHECK_NE(arr, nullptr);
  return *arr;
}
std::shared_ptr<const Array<int64_t>> TileAssignment::shared_array() const {
  absl::MutexLock lock(mu_);
  MaybeMaterializeFullArray();
  return shared_array_;
}

std::shared_ptr<Array<int64_t>> TileAssignment::shared_array_clone() const {
  return std::make_shared<Array<int64_t>>(array());
}

void TileAssignment::MaybeMaterializeFullArray() const {
  if (array_ptr() == nullptr) {
    DCHECK(shared_array_ == nullptr);
    DCHECK(iota_.has_value());
    shared_array_ = std::make_shared<const Array<int64_t>>(iota_->ToArray());
    array_.store(shared_array_.get(), std::memory_order_release);
  }
}

std::vector<int64_t> ExtractCommonFactorSequence(
    absl::Span<const int64_t> array1, absl::Span<const int64_t> array2) {
  std::vector<int64_t> result;
  int64_t index1 = 0;
  int64_t index2 = 0;
  int64_t val1 = 1;
  int64_t val2 = 1;

  while (index1 < array1.size() || index2 < array2.size() || val1 > 1 ||
         val2 > 1) {
    while (val1 == 1 && index1 < array1.size()) {
      val1 = array1[index1++];
    }
    while (val2 == 1 && index2 < array2.size()) {
      val2 = array2[index2++];
    }

    if (val1 == 1 && val2 == 1) {
      break;
    }
    // Mismatched products, or non-positive sizes (which would otherwise lead
    // to a division by zero below).
    if (val1 <= 1 || val2 <= 1) {
      return {};
    }

    const int64_t common = std::min(val1, val2);
    if (val1 % common != 0 || val2 % common != 0) {
      return {};
    }
    result.push_back(common);
    val1 /= common;
    val2 /= common;
  }

  return result;
}

std::optional<std::vector<SubDimInfo>> GetOrderedSubDimsFromIotaTileAssignment(
    const IotaTileAssignment& iota) {
  auto result = GetOrderedSubDims(iota.dims(), iota.reshape_dims(),
                                  iota.transpose_perm());
  if (!result.ok()) {
    return std::nullopt;
  }
  return *std::move(result);
}

absl::StatusOr<std::vector<SubDimInfo>> GetOrderedSubDims(
    absl::Span<const int64_t> dims, absl::Span<const int64_t> reshape_dims,
    absl::Span<const int> transpose_perm) {
  std::vector<int64_t> device_shape;
  device_shape.reserve(transpose_perm.size());
  for (const int perm_index : transpose_perm) {
    device_shape.push_back(reshape_dims[perm_index]);
  }

  const std::vector<int64_t> axis_sizes =
      ExtractCommonFactorSequence(dims, device_shape);
  if (axis_sizes.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to extract common factor sequence: dims: [",
                     absl::StrJoin(dims, ","), "] reshape_dims: [",
                     absl::StrJoin(reshape_dims, ","), "] transpose_perm: T(",
                     absl::StrJoin(transpose_perm, ","), ")"));
  }

  std::vector<SubDimInfo> sub_dims;
  sub_dims.reserve(axis_sizes.size());

  int64_t tile_dim_index = dims.size() - 1;
  int64_t trans_perm_index = transpose_perm.size() - 1;
  int64_t acc_tile_size = 1;
  int64_t acc_device_size = 1;
  int64_t sub_dim = 0;

  for (auto it = axis_sizes.rbegin(); it != axis_sizes.rend(); ++it) {
    int64_t axis_size = *it;
    while (dims[tile_dim_index] == 1) {
      tile_dim_index--;
    }
    sub_dims.push_back(SubDimInfo{
        /* .tile_dim_index = */ tile_dim_index,
        /* .tile_sub_dim_index = */ sub_dim++,
        /* .reshape_dim_index = */ transpose_perm[trans_perm_index],
        /* .size = */ axis_size,
    });
    acc_tile_size *= axis_size;
    acc_device_size *= axis_size;
    if (dims[tile_dim_index] == acc_tile_size) {
      tile_dim_index--;
      acc_tile_size = 1;
      sub_dim = 0;
    }
    if (device_shape[trans_perm_index] == acc_device_size) {
      acc_device_size = 1;
      trans_perm_index--;
    }
  }

  absl::c_sort(sub_dims, [](const SubDimInfo& a, const SubDimInfo& b) {
    return std::forward_as_tuple(a.reshape_dim_index, a.tile_dim_index) <
           std::forward_as_tuple(b.reshape_dim_index, b.tile_dim_index);
  });
  return sub_dims;
}

std::optional<AnalyzeTileAssignmentResult> AnalyzeTileAssignment(
    const TileAssignment& tile_assignment) {
  // If the input has iota tile assignment (the corresponding HloSharding is in
  // V2 format), we use GetOrderedSubDimsFromIotaTileAssignment.
  if (tile_assignment.iota()) {
    std::optional<std::vector<SubDimInfo>> sub_dims =
        GetOrderedSubDimsFromIotaTileAssignment(*tile_assignment.iota());
    CHECK(sub_dims.has_value())
        << "tile assignment: " << tile_assignment.ToString();

    std::vector<int64_t> mesh;
    mesh.reserve(sub_dims->size());
    for (const SubDimInfo& sub_dim_info : *sub_dims) {
      mesh.push_back(sub_dim_info.size);
    }
    return AnalyzeTileAssignmentResult{
        /* .sub_dims = */ std::move(*sub_dims),
        /* .local_mesh = */ std::move(mesh),
        /* .iota = */ *tile_assignment.iota(),
    };
  }

  // If the input is a full array tile assignment (the corresponding HloSharding
  // is in V1 format), we try to detect if it's an iota tile assignment.
  // We try all permutations of the tile assignment dimensions to see if any
  // matches the original array.
  if (!tile_assignment.iota()) {
    // If the input is a full array tile assignment (the corresponding
    // HloSharding is in V1 format), we try to detect if it's a (possibly
    // transposed) iota tile assignment.
    if (auto iota = TryToConvertV1ToV2(tile_assignment)) {
      if (auto sub_dims =
              GetOrderedSubDims(tile_assignment.dimensions(),
                                iota->reshape_dims(), iota->transpose_perm());
          sub_dims.ok()) {
        std::vector<int64_t> mesh;
        mesh.reserve(sub_dims->size());
        absl::c_transform(*sub_dims, std::back_inserter(mesh),
                          [](const SubDimInfo& info) { return info.size; });
        return AnalyzeTileAssignmentResult{
            /* .sub_dims = */ std::move(*sub_dims),
            /* .local_mesh = */ std::move(mesh),
            /* .iota = */ *iota,
        };
      }
    }
  }

  return std::nullopt;
}

}  // namespace xla
