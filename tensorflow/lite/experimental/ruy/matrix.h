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

#include <cstdint>
#include <type_traits>

#include "tensorflow/lite/experimental/ruy/check_macros.h"

namespace ruy {

// Layout storage order. Here and elsewhere, 'col' is short for 'column'.
// 'column-major' means that each column is contiguous in memory.
enum class Order : std::uint8_t { kColMajor, kRowMajor };

// KernelLayout describes small-scale block structure in a matrix layout.
// The default (rows = 1, cols = 1) means no such small-scale block structure,
// since 1x1 blocks is the same as no blocks. In that case, the overall
// matrix layout is just the usual linear row-major or column-major layout
// described by the other members of struct Layout.
struct KernelLayout final {
  Order order = Order::kColMajor;
  std::uint8_t rows = 1;
  std::uint8_t cols = 1;
};

// Describes the shape and storage layout of a matrix.
struct Layout final {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  // Stride is the offset between two adjacent matrix elements
  // in the non-contiguous direction.
  std::int32_t stride = 0;
  Order order = Order::kColMajor;

  // Small scale layout shuffling, potentially departing from
  // linear row-major or column-major storage. See KernelLayout.
  KernelLayout kernel;
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
  void operator=(T* ptr) { set(ptr); }
  void operator=(const T* ptr) { set(ptr); }

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

  void operator=(const Matrix& other) {
    data = other.data;
    layout = other.layout;
    zero_point = other.zero_point;
  }

 private:

 public:
  // The underlying buffer wrapped by this matrix.
  detail::ConstCheckingPtr<Scalar> data;
  // The shape and data layout of this matrix.
  Layout layout;
  // The zero_point, i.e. which Scalar value is to be interpreted as zero.
  // When Scalar is floating-point, this must be 0.
  Scalar zero_point = 0;
  // The row/column sums needed for quantized matrix multiplication when
  // the opposite operand of the multiplication uses a non-symmetric zero
  // point.
  // This member is only relevant for packed matrices.
  // Additionally, Ruy always uses 32-bit signed accumulators for quantized
  // matrix multiplication.
  // For floating point types, there is no quantization, so this pointer
  // will always be null. We still need code referencing it to compile
  // though, even if it is always branched around. Hence we use Scalar*
  // itself as the type in that case.
  using SumsType =
      typename std::conditional<std::is_floating_point<Scalar>::value, Scalar,
                                std::int32_t>::type;
  detail::ConstCheckingPtr<SumsType> sums;
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

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_MATRIX_H_
