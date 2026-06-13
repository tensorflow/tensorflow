/* Copyright 2025 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_ARRAY_UTIL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_ARRAY_UTIL_H_

#include <cstdint>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "xla/array.h"
#include "xla/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

namespace internal {

// Returns true if the slice is empty.
// `starts` and `limits` must be the same length.
bool sliceIsEmpty(absl::Span<const int64_t> starts,
                  absl::Span<const int64_t> limits);

// Increments the slice index.
// Returns true if the slice index is in bounds.
// `idx`, `starts` and `limits` must be the same length.
bool incrementSliceIndex(MutableArrayRef<int64_t> idx,
                         absl::Span<const int64_t> starts,
                         absl::Span<const int64_t> limits);

}  // namespace internal

template <typename T>
ArrayRef<T> XlaArrayToFlatArrayRef(const xla::Array<T> &arr) {
  return ArrayRef<T>(arr.data(), arr.num_elements());
}

template <typename T, typename Range>
xla::Array<T> XlaArrayFromShapeAndValues(ArrayRef<int64_t> sizes, Range vals) {
  // TODO(tlongeri): is there no way to avoid default initialization in the
  // constructor?
  xla::Array<T> arr(sizes);
  arr.SetValues(vals);
  return arr;
}

// An alternative to `xla::Array::UpdateSlice` that takes a single value.
template <typename T>
void updateSlice(xla::Array<T> &arr, const T &value,
                 const absl::Span<const int64_t> starts,
                 const absl::Span<const int64_t> limits) {
  if (internal::sliceIsEmpty(starts, limits)) {
    return;
  }
  SmallVector<int64_t> idx(toArrayRef(starts));
  do {
    arr(idx) = value;
  } while (internal::incrementSliceIndex(idx, starts, limits));
}

// An alternative to `xla::Array::UpdateSlice` that takes a range of data.
template <typename T, typename Range>
void updateSliceFromRange(xla::Array<T> &arr, Range data,
                          const absl::Span<const int64_t> starts,
                          const absl::Span<const int64_t> limits) {
  if (internal::sliceIsEmpty(starts, limits)) {
    return;
  }
  SmallVector<int64_t> idx(toArrayRef(starts));
  auto in_bounds = [&] {
    for (int64_t i = 0; i < idx.size(); ++i) {
      if (idx[i] >= arr.dim(i)) {
        return false;
      }
    }
    return true;
  };
  auto data_it = data.begin();
  do {
    if (in_bounds()) {
      arr(idx) = *data_it;
    }
    ++data_it;
  } while (internal::incrementSliceIndex(idx, starts, limits));
  CHECK(data_it == data.end());
}

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_ARRAY_UTIL_H_
