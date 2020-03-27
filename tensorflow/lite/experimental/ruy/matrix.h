/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_MATRIX_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_MATRIX_H_

#include <cstddef>
#include <cstdint>  // IWYU pragma: keep
#include <type_traits>

#include "tensorflow/lite/experimental/ruy/check_macros.h"

namespace ruy {

// Layout storage order. Here and elsewhere, 'col' is short for 'column'.
// 'column-major' means that each column is contiguous in memory.
enum class Order : std::uint8_t { kColMajor, kRowMajor };

// Describes the shape and storage layout of a matrix.
struct Layout final {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  // Stride is the offset between two adjacent matrix elements
  // in the non-contiguous direction.
  std::int32_t stride = 0;
  Order order = Order::kColMajor;
};

namespace detail {

// Thin wrapper around a pointer that tracks its constness dynamically.
//
// This is our take on the C++ problem of enforcing constness of data
// wrapped in a containers class: it's not worth the hassle of trying to
// make it fully work at compile-time.
// Instead, we only enforce constness at runtime, and to make it
// zero-overhead, we only enforce it in debug builds.
template <typename T>
class ConstCheckingPtr final {
 public:
  using element_type = T;

  // Convenience methods. Most `set` calls go through these.
  ConstCheckingPtr& operator=(T* ptr) {
    set(ptr);
    return *this;
  }
  ConstCheckingPtr& operator=(const T* ptr) {
    set(ptr);
    return *this;
  }
  ConstCheckingPtr& operator=(std::nullptr_t) {
    set(static_cast<T*>(nullptr));
    return *this;
  }

  // Core accessors. These encapsulate the main logic:
  // - for `set`, the constness of the argument determines whether internal
  // pointer should be tracked as const/mutable.
  // - for `get`, the constness of `this` determines whether the call
  // counts as a const or mutable use of the internal pointer.
  void set(T* ptr) {
    ptr_ = ptr;
    set_mutable(true);
  }
  void set(const T* ptr) {
    ptr_ = ptr;
    set_mutable(false);
  }
  T* get() /* NOT const */ {
    assert_mutable();
    return const_cast<T*>(ptr_);
  }
  const T* get() const { return ptr_; }

 private:
  static_assert(!std::is_const<T>::value, "");
  const T* ptr_ = nullptr;
#ifndef NDEBUG
  bool is_mutable_ = true;
  void set_mutable(bool val) { is_mutable_ = val; }
  void assert_mutable() { RUY_DCHECK(is_mutable_); }
#else
  void set_mutable(bool) {}
  void assert_mutable() {}
#endif
};

}  // namespace detail

// A Matrix is really what Eigen and gemmlowp would have called a 'matrix map':
// it merely wraps existing data as a matrix. It doesn't own any buffer.
// Scalar may be any floating-point or integral type. When integral, it may be
// signed or unsigned.
template <typename Scalar>
struct Matrix final {
  Matrix& operator=(const Matrix& other) {
    data = other.data;
    cacheable = other.cacheable;
    layout = other.layout;
    zero_point = other.zero_point;
    return *this;
  }

  // The underlying buffer wrapped by this matrix.
  detail::ConstCheckingPtr<Scalar> data;
  // The shape and data layout of this matrix.
  Layout layout;
  // The zero_point, i.e. which Scalar value is to be interpreted as zero.
  // When Scalar is floating-point, this must be 0.
  Scalar zero_point = 0;
  // Clients of Ruy must set this flag to enable any caching behavior. Doesn't
  // impact numerical results, but caching can impact observable metrics like
  // latency, memory usage, power, etc.
  bool cacheable = false;
};

inline void MakeSimpleLayout(int rows, int cols, Order order, Layout* layout) {
  layout->rows = rows;
  layout->cols = cols;
  layout->order = order;
  layout->stride = order == Order::kColMajor ? rows : cols;
}

// Opaque data structure representing a pre-packed matrix, as obtained from
// Ruy's advanced API.
struct PrepackedMatrix {
  void* data = nullptr;
  std::size_t data_size = 0;
  void* sums = nullptr;
  std::size_t sums_size = 0;
};

template <typename StreamType, typename Scalar>
StreamType& operator<<(StreamType& stream, const Matrix<Scalar>& mat) {
  for (int row = 0; row < mat.layout.rows; row++) {
    for (int col = 0; col < mat.layout.cols; col++) {
      stream << static_cast<double>(Element(mat, row, col)) << " ";
    }
    stream << "\n";
  }
  return stream;
}

// Compile-time version of KernelLayout, used to declare kernel layouts in a
// way that can be consumed by compile-time logic.
// See how partial specializations of Kernel use it to declare their layouts.
// The only reason why this is currently part of the public API is to
// allow testing various layouts for the Path::kStandardCpp kernel, as a
// testing-only feature. See Spec::StandardCppKernelLhsLayout.
template <Order tOrder, int tRows, int tCols>
struct FixedKernelLayout {
  static constexpr Order kOrder = tOrder;
  static constexpr int kRows = tRows;
  static constexpr int kCols = tCols;
};

#if (__cplusplus < 201703L)
// A static constexpr data member is automatically inline and should not require
// redeclaration without an initializer. This is actually deprecated from C++17
// onwards. Clang with -O0 without this can fail to link.
template <Order tOrder, int tRows, int tCols>
constexpr int FixedKernelLayout<tOrder, tRows, tCols>::kCols;
template <Order tOrder, int tRows, int tCols>
constexpr int FixedKernelLayout<tOrder, tRows, tCols>::kRows;
#endif

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_MATRIX_H_
