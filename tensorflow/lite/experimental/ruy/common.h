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

// Miscellaneous helpers internal library.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_COMMON_H_

#include <atomic>
#include <limits>
#include <type_traits>
#include <utility>

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#if RUY_OPT_SET & RUY_OPT_PREFETCH
#define RUY_PREFETCH(X) X
#else
#define RUY_PREFETCH(X)
#endif

#define RUY_STR(s) RUY_STR_UNEXPANDED(s)
#define RUY_STR_UNEXPANDED(s) #s

namespace ruy {

inline void MakeSimpleLayout(int rows, int cols, Order order, Layout* layout) {
  layout->rows = rows;
  layout->cols = cols;
  layout->order = order;
  layout->stride = order == Order::kColMajor ? rows : cols;
  layout->kernel.order = order;
  layout->kernel.rows = 1;
  layout->kernel.cols = 1;
}

inline bool IsLinear(const Layout& layout) {
  return layout.kernel.rows == 1 && layout.kernel.cols == 1;
}

inline bool IsPacked(const Layout& layout) {
  if (layout.order == Order::kColMajor) {
    return layout.stride == layout.rows;
  } else {
    return layout.stride == layout.cols;
  }
}

inline bool IsPackedLinear(const Layout& layout) {
  return IsPacked(layout) && IsLinear(layout);
}

inline bool IsRowMajor(const Layout& layout) {
  return layout.order == Order::kRowMajor;
}

inline bool IsColMajor(const Layout& layout) {
  return layout.order == Order::kColMajor;
}

inline bool IsLinearColMajor(const Layout& layout) {
  return IsLinear(layout) && IsColMajor(layout);
}

inline bool IsPackedLinearColMajor(const Layout& layout) {
  return IsLinearColMajor(layout) && IsPacked(layout);
}

inline bool IsLinearRowMajor(const Layout& layout) {
  return IsLinear(layout) && IsRowMajor(layout);
}

inline bool IsPackedLinearRowMajor(const Layout& layout) {
  return IsLinearRowMajor(layout) && IsPacked(layout);
}

inline int FlatSize(const Layout& layout) {
  const int outerdim =
      layout.order == Order::kColMajor ? layout.cols : layout.rows;
  return layout.stride * outerdim;
}

// TODO(b/130417400) add a unit test
inline int Offset(const Layout& layout, int row, int col) {
  // TODO(benoitjacob)  - should check this but this make the _slow tests take
  // 5x longer.  Find a mitigation like in Eigen with an 'internal' variant
  // bypassing the check?
  // RUY_DCHECK_GE(row, 0);
  // RUY_DCHECK_GE(col, 0);
  // RUY_DCHECK_LT(row, layout.rows);
  // RUY_DCHECK_LT(col, layout.cols);
  if (IsLinear(layout)) {
    int row_stride = layout.order == Order::kColMajor ? 1 : layout.stride;
    int col_stride = layout.order == Order::kRowMajor ? 1 : layout.stride;
    return row * row_stride + col * col_stride;
  } else {
    RUY_DCHECK(is_pot(layout.kernel.rows));
    RUY_DCHECK(is_pot(layout.kernel.cols));
    int row_outer = row & ~(layout.kernel.rows - 1);
    int col_outer = col & ~(layout.kernel.cols - 1);
    int row_stride_outer =
        layout.order == Order::kColMajor ? layout.kernel.cols : layout.stride;
    int col_stride_outer =
        layout.order == Order::kRowMajor ? layout.kernel.rows : layout.stride;
    int offset_outer =
        row_outer * row_stride_outer + col_outer * col_stride_outer;
    int row_inner = row - row_outer;
    int col_inner = col - col_outer;
    int row_stride_inner =
        layout.kernel.order == Order::kColMajor ? 1 : layout.kernel.cols;
    int col_stride_inner =
        layout.kernel.order == Order::kRowMajor ? 1 : layout.kernel.rows;
    int offset_inner =
        row_inner * row_stride_inner + col_inner * col_stride_inner;
    return offset_outer + offset_inner;
  }
}

template <typename Scalar>
const Scalar* ElementPtr(const Matrix<Scalar>& mat, int row, int col) {
  return mat.data.get() + Offset(mat.layout, row, col);
}

template <typename Scalar>
Scalar* ElementPtr(Matrix<Scalar>* mat, int row, int col) {
  return mat->data.get() + Offset(mat->layout, row, col);
}

template <typename Scalar>
Scalar Element(const Matrix<Scalar>& mat, int row, int col) {
  return *ElementPtr(mat, row, col);
}

// We need this where we have multiple threads potentially writing concurrently
// to the same memory location. That is currently the case for Pack (see
// the comment in TrMulTask where Pack is called) and in tracing.
//
// This is a strict-aliasing violation. For nicer things, see C++20 atomic_ref
// and the defunct N4013. (Thanks to hboehm@).
template <typename T>
void relaxed_atomic_store(T* ptr, T value) {
  static_assert(sizeof(std::atomic<T>) == sizeof(T), "");
  std::atomic<T>* atomic = reinterpret_cast<std::atomic<T>*>(ptr);
  RUY_DCHECK(atomic->is_lock_free());
  atomic->store(value, std::memory_order_relaxed);
}

template <typename Scalar>
Scalar SymmetricZeroPoint() {
  if (std::is_floating_point<Scalar>::value) {
    return 0;
  }
  if (std::is_signed<Scalar>::value) {
    return 0;
  }
  return std::numeric_limits<Scalar>::max() / 2 + 1;
}

template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
struct TrMulImpl;

template <Order tOrder, int tRows, int tCols>
struct FixedKernelLayout {
  static constexpr Order kOrder = tOrder;
  static constexpr int kRows = tRows;
  static constexpr int kCols = tCols;
};

inline void Transpose(Order* order) {
  *order = *order == Order::kColMajor ? Order::kRowMajor : Order::kColMajor;
}

inline void Transpose(Layout* layout) {
  Transpose(&layout->order);
  Transpose(&layout->kernel.order);
  std::swap(layout->rows, layout->cols);
  std::swap(layout->kernel.rows, layout->kernel.cols);
}

template <typename Scalar>
inline void Transpose(Matrix<Scalar>* matrix) {
  Transpose(&matrix->layout);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_COMMON_H_
