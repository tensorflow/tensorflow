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

#ifndef XLA_MOSAIC_DIALECT_TPU_ARRAY_UTIL_H_
#define XLA_MOSAIC_DIALECT_TPU_ARRAY_UTIL_H_

#include <cstdint>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/util.h"
#include "xla/array.h"

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

bool incrementIndex(MutableArrayRef<int64_t> idx,
                    absl::Span<const int64_t> limits);

// Similar to incrementIndex, but only increments the dimensions in
// `subsequence`, starting with the last dimension in `subsequence` (row-major
// order).
template <typename T>
bool incrementIndexSubsequence(const MutableArrayRef<int64_t> idx,
                               const ArrayRef<T> subsequence,
                               const ArrayRef<int64_t> limits) {
  CHECK_EQ(idx.size(), limits.size());
  for (int64_t i = subsequence.size() - 1; i >= 0; --i) {
    const int64_t d = subsequence[i];
    ++idx[d];
    if (idx[d] < limits[d]) {
      return true;
    }
    idx[d] = 0;
  }
  return false;
}

template <typename T>
ArrayRef<T> XlaArrayToFlatArrayRef(const xla::Array<T>& arr) {
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

// An alternative to xla::Array::Each that returns a LogicalResult
template <typename T, typename Func>
std::enable_if_t<std::is_invocable_v<Func, ArrayRef<int64_t>, T*>,
                 LogicalResult>
Each(xla::Array<T>& arr, Func&& func) {
  SmallVector<int64_t> idx(arr.num_dimensions());
  auto it = arr.begin();
  do {
    RETURN_IF_FAILED(func(ArrayRef(idx), &*it));
    ++it;
  } while (incrementIndex(idx, arr.dimensions()));
  DCHECK(it == arr.end());
  return success();
}
template <typename T, typename Func>
std::enable_if_t<std::is_invocable_v<Func, ArrayRef<int64_t>, T>, LogicalResult>
Each(const xla::Array<T>& arr, Func&& func) {
  SmallVector<int64_t> idx(arr.num_dimensions());
  auto it = arr.begin();
  do {
    RETURN_IF_FAILED(func(ArrayRef(idx), *it));
    ++it;
  } while (incrementIndex(idx, arr.dimensions()));
  DCHECK(it == arr.end());
  return success();
}

// An alternative to `xla::Array::UpdateSlice` that takes a single value.
template <typename T>
void updateSlice(xla::Array<T>& arr, const T& value,
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
void updateSliceFromRange(xla::Array<T>& arr, Range data,
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

template <typename T>
xla::Array<T> broadcast(const xla::Array<T>& arr, ArrayRef<int64_t> new_shape) {
  CHECK_EQ(arr.num_dimensions(), new_shape.size());
  // TODO(tlongeri): Don't default-initialize elements here
  xla::Array<T> result(new_shape);
  SmallVector<int64_t> idx(arr.num_dimensions());
  SmallVector<int64_t> broadcast_dims;
  for (int64_t i = idx.size() - 1; i >= 0; --i) {
    if (arr.dim(i) != new_shape[i]) {
      CHECK_EQ(arr.dim(i), 1);
      broadcast_dims.push_back(i);
    }
  }
  // TODO(tlongeri): This could be more efficient if it didn't recompute the
  // linear offset from the idx every time and instead used a stride.
  do {
    const T element = arr(idx);
    do {
      result(idx) = element;
    } while (incrementIndexSubsequence(idx, ArrayRef(broadcast_dims),
                                       new_shape));
  } while (incrementIndex(idx, arr.dimensions()));
  return result;
}

}  // namespace mlir::tpu

#endif  // XLA_MOSAIC_DIALECT_TPU_ARRAY_UTIL_H_
