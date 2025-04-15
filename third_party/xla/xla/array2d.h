/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_ARRAY2D_H_
#define XLA_ARRAY2D_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "xla/array.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla {

template <typename T>
class Array2D : public Array<T> {
 public:
  Array2D() : Array<T>(std::vector<int64_t>{0, 0}) {}

  Array2D(const int64_t n1, const int64_t n2)
      : Array<T>(std::vector<int64_t>{n1, n2}) {}

  Array2D(const int64_t n1, const int64_t n2, const T value)
      : Array<T>({n1, n2}, value) {}

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension; the inner is the second dimension.
  // For example, {{1, 2, 3}, {4, 5, 6}} results in an array with n1=2 and n2=3.
  Array2D(std::initializer_list<std::initializer_list<T>> values)
      : Array<T>(values) {}

  // Creates an array of a floating-point type (float8, half, bfloat16, float,
  // or double) from the given nested initializer list of float values.
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array2D(std::initializer_list<std::initializer_list<T2>> values)
      : Array<T>(values) {}

  Array2D(const Array2D<T>& other) : Array<T>(other) {}
  Array2D(Array2D<T>&& other) noexcept : Array<T>(std::move(other)) {}

  Array2D& operator=(const Array2D<T>& other) {
    Array<T>::operator=(other);
    return *this;
  }

  Array2D& operator=(Array2D<T>&& other) noexcept {
    Array<T>::operator=(std::move(other));
    return *this;
  }

  int64_t n1() const { return this->dim(0); }
  int64_t n2() const { return this->dim(1); }

  int64_t height() const { return this->dim(0); }
  int64_t width() const { return this->dim(1); }

  // Fills the array with a pattern of values of the form:
  //
  //    (rowno << log2ceil(width) | colno) + start_value
  //
  // This makes it easy to see distinct row/column values in the array.
  void FillUnique(T start_value = 0) {
    int shift = Log2Ceiling<uint64_t>(n2());
    for (int64_t i0 = 0; i0 < n1(); ++i0) {
      for (int64_t i1 = 0; i1 < n2(); ++i1) {
        (*this)(i0, i1) = ((i0 << shift) | i1) + start_value;
      }
    }
  }

  // Applies f to all cells in this array, in row-major order.
  void Each(absl::FunctionRef<void(int64_t, int64_t, T*)> f) {
    for (int64_t i0 = 0; i0 < n1(); ++i0) {
      for (int64_t i1 = 0; i1 < n2(); ++i1) {
        f(i0, i1, &(*this)(i0, i1));
      }
    }
  }
};

// Returns a linspace-populated Array2D in the range [from, to] (inclusive)
// with dimensions n1 x n2.
template <typename NativeT = float>
std::unique_ptr<Array2D<NativeT>> MakeLinspaceArray2D(double from, double to,
                                                      int64_t n1, int64_t n2) {
  auto array = std::make_unique<Array2D<NativeT>>(n1, n2);
  int64_t count = n1 * n2;
  double step =
      static_cast<double>((count > 1) ? (to - from) / (count - 1) : 0);

  auto set = [&array, n2](int64_t index, NativeT value) {
    (*array)(index / n2, index % n2) = value;
  };
  for (int64_t i = 0; i < count - 1; ++i) {
    set(i, (static_cast<NativeT>(from + i * step)));
  }
  set(count - 1, static_cast<NativeT>(to));
  return array;
}
}  // namespace xla

#endif  // XLA_ARRAY2D_H_
