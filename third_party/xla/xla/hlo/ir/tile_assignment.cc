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

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/printer.h"
#include "xla/util.h"

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
      if (base_dim + (i - base) == dim) {
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
      if (d != i && dims[i] != 1) kind = TransposeKind::kReshape;
      continue;
    }
    if (d <= prev_non_one_dim) return TransposeKind::kTranspose;
    prev_non_one_dim = d;
  }
  return kind;
}

// Fully decanonicalizes reshape_dims into prime factors and return the new
// reshape_dims and transpose_perm.
std::pair<absl::InlinedVector<int64_t, 6>, absl::InlinedVector<int, 6>>
FullyDecanonicalize(absl::Span<const int64_t> reshape_dims,
                    absl::Span<const int> transpose_perm) {
  absl::InlinedVector<int64_t, 6> new_reshape_dims;
  absl::InlinedVector<int, 6> old_to_new_dims(reshape_dims.size() + 1);

  for (int i = 0, n = reshape_dims.size(); i < n; ++i) {
    int64_t dim_size = reshape_dims[i];
    while (dim_size % 2 == 0) {
      new_reshape_dims.push_back(2);
      dim_size /= 2;
    }
    for (int i = 3; i * i <= dim_size; i += 2) {
      while (dim_size % i == 0) {
        new_reshape_dims.push_back(i);
        dim_size /= i;
      }
    }
    if (dim_size > 1) {
      CHECK_GT(dim_size, 2);
      new_reshape_dims.push_back(dim_size);
    }
    old_to_new_dims[i + 1] = new_reshape_dims.size();
  }
  absl::InlinedVector<int, 6> new_transpose_perm;
  new_transpose_perm.reserve(new_reshape_dims.size());
  for (int i = 0; i < transpose_perm.size(); ++i) {
    const int old_dim = transpose_perm[i];
    for (int j = old_to_new_dims[old_dim], n = old_to_new_dims[old_dim + 1];
         j < n; ++j) {
      new_transpose_perm.push_back(j);
    }
  }
  return std::make_pair(std::move(new_reshape_dims),
                        std::move(new_transpose_perm));
}

}  // namespace

/*static*/ IotaTileAssignment IotaTileAssignment::Create(
    absl::Span<const int64_t> dims) {
  return IotaTileAssignment(dims, {Product(dims)}, {0});
}

/*static*/ IotaTileAssignment IotaTileAssignment::Create(
    absl::Span<const int64_t> dims, absl::Span<const int64_t> reshape_dims,
    absl::Span<const int> transpose_perm) {
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

Array<int64_t> IotaTileAssignment::ToArray() const {
  Array<int64_t> array(reshape_dims());
  array.FillIota(0);
  array.TransposeDimensions(transpose_perm());
  array.Reshape(dims());
  return array;
}

IotaTileAssignment::IotaTileAssignment(const IotaTileAssignment& other)
    : IotaTileAssignment(other.ndims_, other.reshape_ndims_) {
  std::memcpy(storage_.get(), other.storage_.get(), size_bytes());
}

IotaTileAssignment& IotaTileAssignment::operator=(
    const IotaTileAssignment& other) {
  const int new_size = other.size_bytes();
  if (size_bytes() != new_size) {
    storage_.reset(new char[new_size]);
  }
  ndims_ = other.ndims_;
  reshape_ndims_ = other.reshape_ndims_;
  std::memcpy(storage_.get(), other.storage_.get(), new_size);
  return *this;
}

IotaTileAssignment::IotaTileAssignment(absl::Span<const int64_t> dims,
                                       absl::Span<const int64_t> reshape_dims,
                                       absl::Span<const int> transpose_perm)
    : IotaTileAssignment(dims.size(), reshape_dims.size()) {
  DCHECK_EQ(reshape_dims.size(), transpose_perm.size());
  std::memcpy(dims_ptr(), dims.data(), ndims_ * sizeof(int64_t));
  DCHECK_EQ(num_elements(), Product(reshape_dims));
  std::memcpy(reshape_dims_ptr(), reshape_dims.data(),
              reshape_ndims_ * sizeof(int64_t));
  std::memcpy(transpose_perm_ptr(), transpose_perm.data(),
              reshape_ndims_ * sizeof(int));
}

IotaTileAssignment::IotaTileAssignment(int ndims, int reshape_ndims)
    : ndims_(ndims),
      reshape_ndims_(reshape_ndims),
      storage_(new char[size_bytes()]) {}

std::optional<IotaTileAssignment> IotaTileAssignment::Transpose(
    absl::Span<const int> perm) const {
  DCHECK_EQ(ndims_, perm.size());
  auto dims = this->dims();
  const TransposeKind kind = GetTransposeKind(dims, perm);
  if (kind == TransposeKind::kNoop) return *this;
  absl::InlinedVector<int64_t, 6> new_dims(ndims_);
  for (int64_t i = 0; i < ndims_; ++i) {
    new_dims[i] = dims[perm[i]];
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
      if (dims[perm[i]] == 1) continue;
      new_perm.push_back(transpose_perm[one_to_non_one[perm[i]]]);
    }
    CHECK_EQ(reshape_ndims_, new_perm.size());
    return IotaTileAssignment::Create(new_dims, reshape_dims, new_perm);
  }

  auto [decanonicalized_reshape_dims, decanonicalized_transpose_perm] =
      FullyDecanonicalize(reshape_dims, transpose_perm);
  CHECK_LE(non_one_dims.size(), decanonicalized_reshape_dims.size());
  // Try grouping decanonicalized reshape dimensions together to see if they
  // form the identical tile dimensions, then transpose them in groups.
  absl::InlinedVector<absl::InlinedVector<int, 2>, 6> grouped_reshape_dims(
      non_one_dims.size());
  int transpose_perm_idx = 0;
  for (int i = 0, n = non_one_dims.size(),
           dn = decanonicalized_reshape_dims.size();
       i < n && transpose_perm_idx < dn; ++i) {
    int reshape_dim_idx = decanonicalized_transpose_perm[transpose_perm_idx];
    int64_t cand = decanonicalized_reshape_dims[reshape_dim_idx];
    int64_t target = non_one_dims[i];
    while (target % cand == 0) {
      target /= cand;
      grouped_reshape_dims[i].push_back(reshape_dim_idx);
      if (++transpose_perm_idx >= dn) {
        break;
      }
      reshape_dim_idx = decanonicalized_transpose_perm[transpose_perm_idx];
      cand = decanonicalized_reshape_dims[reshape_dim_idx];
    }
    if (target != 1) {
      // TODO(b/341371396): Handle remaining patterns and remove nullopt path.
      // It seems this cannot happen under the valid condition that we generate
      // code with predefined mesh axises, but the C++ API does not restrict
      // people from constructing sharding like `[2,3]<=[2,3]T(1,0]` which
      // breaks the axises and transposing it will result in V1 sharding.
      return std::nullopt;
    }
  }
  absl::InlinedVector<int, 6> flattened_transpose_perm;
  flattened_transpose_perm.reserve(reshape_ndims_);
  for (int i = 0; i < perm.size(); ++i) {
    const int dim = perm[i];
    if (one_to_non_one[dim] < 0) {
      continue;
    }
    auto& group = grouped_reshape_dims[one_to_non_one[dim]];
    flattened_transpose_perm.insert(flattened_transpose_perm.end(),
                                    group.begin(), group.end());
  }
  CHECK_EQ(flattened_transpose_perm.size(),
           decanonicalized_transpose_perm.size());
  return IotaTileAssignment::Create(new_dims, decanonicalized_reshape_dims,
                                    flattened_transpose_perm);
}

void IotaTileAssignment::Print(Printer* printer) const {
  printer->Append("[");
  AppendJoin(printer, dims(), ",");
  printer->Append("]<=[");
  AppendJoin(printer, reshape_dims(), ",");
  printer->Append("]");
  if (reshape_ndims_ > 1) {
    printer->Append("T(");
    AppendJoin(printer, transpose_perm(), ",");
    printer->Append(")");
  }
}

std::string IotaTileAssignment::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

int64_t IotaTileAssignment::value_at(absl::Span<const int64_t> index) const {
  DCHECK_EQ(index.size(), ndims_);
  int64_t linear_index = index[0];
  auto dims = this->dims();
  for (int64_t i = 1; i < ndims_; ++i) {
    linear_index *= dims[i];
    linear_index += index[i];
  }
  auto reshape_dims = this->reshape_dims();
  auto transpose_perm = this->transpose_perm();
  absl::InlinedVector<int64_t, 6> reshape_index(reshape_ndims_);
  for (int64_t i = reshape_ndims_ - 1; i >= 0; --i) {
    int dim = transpose_perm[i];
    int dim_size = reshape_dims[dim];
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

bool TileAssignment::operator==(const TileAssignment& other) const {
  if (iota_ && other.iota_) {
    return *iota_ == *other.iota_;
  }
  return array() == other.array();
}

int64_t TileAssignment::operator()(absl::Span<const int64_t> indexes) const {
  return array_ ? (*array_)(indexes) : iota_->value_at(indexes);
}

absl::Span<const int64_t> TileAssignment::dimensions() const {
  return array_ ? array_->dimensions() : iota_->dims();
}

int64_t TileAssignment::num_dimensions() const {
  return array_ ? array_->num_dimensions() : iota_->ndims();
}

int64_t TileAssignment::dim(int64_t n) const {
  return array_ ? array_->dim(n) : iota_->dim(n);
}
int64_t TileAssignment::num_elements() const {
  return array_ ? array_->num_elements() : iota_->num_elements();
}

int64_t TileAssignment::first() const { return array_ ? *array_->begin() : 0; }

void TileAssignment::Each(
    absl::FunctionRef<void(absl::Span<const int64_t>, int64_t)> f) const {
  MaybeMaterializeFullArray();
  array_->Each(f);
}

absl::Status TileAssignment::EachStatus(
    absl::FunctionRef<absl::Status(absl::Span<const int64_t>, int64_t)> f)
    const {
  MaybeMaterializeFullArray();
  return array_->EachStatus(f);
}

[[nodiscard]] TileAssignment TileAssignment::Reshape(
    absl::Span<const int64_t> new_dimensions) const {
  if (iota_) {
    CHECK_EQ(Product(new_dimensions), iota_->num_elements());
    return TileAssignment(
        IotaTileAssignment(new_dimensions, iota_->reshape_dims(),
                           iota_->transpose_perm()),
        /*shared_array=*/nullptr);
  }
  auto reshaped = std::make_shared<Array<int64_t>>(*array_);
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

std::string TileAssignment::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

bool TileAssignment::UsesDevice(int64_t device) const {
  return iota_ ? device < iota_->num_elements()
               : absl::c_linear_search(array(), device);
}

const Array<int64_t>& TileAssignment::array() const {
  MaybeMaterializeFullArray();
  return *array_;
}
const std::shared_ptr<const Array<int64_t>>& TileAssignment::shared_array()
    const {
  MaybeMaterializeFullArray();
  return shared_array_;
}

std::shared_ptr<Array<int64_t>> TileAssignment::shared_array_clone() const {
  MaybeMaterializeFullArray();
  return std::make_shared<Array<int64_t>>(*array_);
}

void TileAssignment::MaybeMaterializeFullArray() const {
  if (array_ == nullptr) {
    DCHECK(shared_array_ == nullptr);
    DCHECK(iota_.has_value());
    auto full = std::make_shared<Array<int64_t>>(iota_->ToArray());
    shared_array_ = std::move(full);
    array_ = shared_array_.get();
  }
}

}  // namespace xla
