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

// Internal types and helpers for matrices.
//
// Ruy has a couple slightly different notions of matrices, besides the
// Matrix<T> class that we expose to the user-facing API.
//
// TODO(silvasean): Put parts of this architecture description somewhere more
// prominent.
//
// The 4 main matrix types are:
// - Matrix<T>: This is a user-facing type on Ruy's external API boundary. It is
// also used internally.
// - DMatrix: This is a type-erased version of Matrix<T>. "D" = "dynamic".
// - PMatrix: This represents a packed matrix, which requires tracking kernel
// layout and row/column sums for quantization. It is type-erased.
// - PackedMatrix<T>: This is a statically typed variant of PMatrix for
// convenience inside typed routines.
//
// Note that Matrix<T> is *not* implemented in terms of the internal types. It
// is an independent, simple, and user-facing type.
//
// The use of type-erasure might seem surprising for a library like Ruy with a
// heavily-templated entry point, but it is motivated by the desire for most of
// Ruy's "middle-end" to be non-templated. Ruy can be thought of as having 3
// main parts:
// - "front-end" (dispatch.h) - this is the highly templated ruy::Mul entry
// point, along with routines that select RunKernel and RunPack implementations
// statically based on those template parameters.
// - "back-end" (kernel.h, pack.h)- this consists of the implementations of
// RunKernel and RunPack, often in assembly code, which are the building blocks
// that Ruy calls to perform matrix multiplication.  These are templated so that
// only the requested types/Path's are actually emitted by the compiler.
// - "middle-end" (trmul.h) - this is the part of Ruy that orchestrates the
// calls to the "back-end" optimized building blocks. This layer has to deal
// with issues like cache locality and low-overhead multi-threading.
//
// There is a desire for the "middle-end" to be non-templated in order to
// simplify the implementation and reduce code-size. We type-erase when going
// from the "front-end" to the "middle-end", and un-type-erase going from the
// "middle-end" to the "back-end". The un-type-erasure is possible because the
// "front-end" is responsible for instantiating the needed "back-end" templates,
// and thus the static type information is still present.
//
// Each layer of Ruy uses matrix types:
// - "front-end": Matrix<T>
// - "middle-end": DMatrix, PMatrix
// - "back-end": Matrix<T>, PackedMatrix<T>
//
// The use of separate types for packed matrices is not essential, but makes it
// obvious at a glance whether a matrix is a packed matrix or not. We would
// reconsider this decision if there was significant duplication between packed
// and unpacked matrices, but that doesn't seem to be the case at the moment.
//
// Another goal is to keep the user-facing Matrix<T> as simple and
// understandable as possible. Ideally, a user should be able to read the struct
// definition for Matrix<T> and see a very simple definition with no internal
// details like sums and kernel block layout.
//
// To present another structured view of our various matrix types, here's a
// table:
//                Plain matrices   Packed matrices
//             +----------------------------------
// Templated   |  Matrix<T>        PackedMatrix<T>
// Type-erased |  DMatrix          PMatrix
//
//
// There is 1 additional matrix type not mentioned above, due to its low
// importance:
// - PrepackedMatrix: This is a user-facing version of PMatrix. It has the bare
// minimum of fields needed for representing the raw data and sums buffers of a
// packed matrix for the "advanced" explicit pre-packing API. This type plays no
// role in Ruy's internals and can generally by ignored. The only reason it
// exists is so that PMatrix is not exposed to users -- we prefer to keep the
// internal matrix types hidden, even from "advanced" users.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_INTERNAL_MATRIX_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_INTERNAL_MATRIX_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"

namespace ruy {

// KernelLayout describes small-scale block structure in a packed matrix layout.
// It's a runtime (as opposed to compile-time-constant) version of the
// FixedKernelLayout struct used to declare kernel layouts.
//
// This is is sometimes known as "tiling" in other contexts.
//
// For example, consider a packed matrix in column-major format with a
// column-major KernelLayout. The matrix logically has a shape of
// `[cols, rows]`. However, the matrix is laid out as though it were a 4D array
// of shape `[cols / kcols, rows / krows, kcols, krows]`.
//
// Note that in the case of kcols=1, krows=1, this degenerates to
// `[cols, rows, 1, 1]` which is equivalent to having no small-scale block
// structure.
struct KernelLayout {
  Order order = Order::kColMajor;
  std::uint8_t rows = 1;
  std::uint8_t cols = 1;
};

// A packed matrix has a small-scale block structure that is not present in in
// the input matrices. This block structure is necessary for the kernels to
// process data efficiently.
//
// This struct is very similar to Layout, but has the extra KernelLayout field.
struct PackedLayout {
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

// Dynamic representation for a type.
//
// The most important field in this struct is the size, which Ruy uses to know
// how much memory to allocate without having to be templated on a type.
// Signed-ness and floating-point-ness are mainly present as debugging checks.
//
// Note: Ruy does not use this struct to to dynamically dispatch between
// different typed implementations. As described in the comment at the top of
// this file, Ruy's "front-end", which is templated, instantiates all the
// necessary "back-end" routines with complete static knowledge of all the
// types.
struct Type {
  template <typename T>
  static Type Create() {
    Type ret;
    ret.is_signed = std::is_signed<T>::value;
    ret.is_floating_point = std::is_floating_point<T>::value;
    ret.size = sizeof(T);
    return ret;
  }

  template <typename T>
  void AssertIs() const {
    RUY_DCHECK_EQ(is_signed, Create<T>().is_signed);
    RUY_DCHECK_EQ(is_floating_point, Create<T>().is_floating_point);
    RUY_DCHECK_EQ(size, Create<T>().size);
  }

  bool is_signed = false;
  bool is_floating_point = false;
  std::uint8_t size = 0;
};

// Type-erased matrix.
struct DMatrix {
  Type data_type;
  void* data = nullptr;
  Layout layout;
  std::int32_t zero_point = 0;
};

// Type-erased packed matrix.
struct PMatrix {
  Type data_type;
  void* data = nullptr;
  Type sums_type;
  void* sums = nullptr;
  PackedLayout layout;
  std::int32_t zero_point = 0;
};

// Convenient typed helper for packed matrices.
template <typename Scalar>
struct PackedMatrix {
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

  Scalar* data = nullptr;
  SumsType* sums = nullptr;
  PackedLayout layout;
  std::int32_t zero_point = 0;
};

template <typename T>
DMatrix ToDMatrix(const Matrix<T>& matrix) {
  DMatrix ret;
  ret.data_type = Type::Create<T>();
  ret.data = ToVoidPtr(matrix.data.get());
  ret.layout = matrix.layout;
  ret.zero_point = matrix.zero_point;
  return ret;
}

template <typename T>
Matrix<T> ToMatrix(const DMatrix& dmatrix) {
  dmatrix.data_type.AssertIs<T>();
  Matrix<T> ret;
  ret.data = static_cast<T*>(dmatrix.data);
  ret.layout = dmatrix.layout;
  ret.zero_point = dmatrix.zero_point;
  return ret;
}

template <typename T>
PackedMatrix<T> ToPackedMatrix(const PMatrix& pmatrix) {
  using SumsType = typename PackedMatrix<T>::SumsType;
  pmatrix.data_type.AssertIs<T>();
  pmatrix.sums_type.AssertIs<SumsType>();
  PackedMatrix<T> ret;
  ret.data = static_cast<T*>(pmatrix.data);
  ret.sums = static_cast<SumsType*>(pmatrix.sums);
  ret.layout = pmatrix.layout;
  ret.zero_point = pmatrix.zero_point;
  return ret;
}

// Helpers for Layout / PackedLayout.

inline bool IsPacked(const Layout& layout) {
  if (layout.order == Order::kColMajor) {
    return layout.stride == layout.rows;
  } else {
    return layout.stride == layout.cols;
  }
}

inline bool IsRowMajor(const Layout& layout) {
  return layout.order == Order::kRowMajor;
}

template <typename LayoutOrPackedLayout>
inline bool IsColMajor(const LayoutOrPackedLayout& layout) {
  return layout.order == Order::kColMajor;
}

template <typename LayoutOrPackedLayout>
inline int FlatSize(const LayoutOrPackedLayout& layout) {
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
  int row_stride = layout.order == Order::kColMajor ? 1 : layout.stride;
  int col_stride = layout.order == Order::kRowMajor ? 1 : layout.stride;
  return row * row_stride + col * col_stride;
}

// TODO(b/130417400) add a unit test
inline int Offset(const PackedLayout& layout, int row, int col) {
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

// Helpers for Matrix<T>.

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

// Helpers for PackedMatrix<T>.
// Duplicated from Matrix<T>, but the duplication seems acceptable.

template <typename Scalar>
const Scalar* ElementPtr(const PackedMatrix<Scalar>& mat, int row, int col) {
  return mat.data + Offset(mat.layout, row, col);
}

template <typename Scalar>
Scalar* ElementPtr(PackedMatrix<Scalar>* mat, int row, int col) {
  return mat->data + Offset(mat->layout, row, col);
}

template <typename Scalar>
Scalar Element(const PackedMatrix<Scalar>& mat, int row, int col) {
  return *ElementPtr(mat, row, col);
}

// Helpers for PMatrix.

inline std::size_t DataSize(const PMatrix& packed) {
  return FlatSize(packed.layout) * packed.data_type.size;
}

inline std::size_t SumsSize(const PMatrix& packed) {
  // Packed matrices are only relevant for Ruy's TrMul implementations. For
  // TrMul, the number of sums is always equal to the number of columns.
  return packed.layout.cols * packed.sums_type.size;
}

// Transpose helpers.

inline void Transpose(Order* order) {
  *order = *order == Order::kColMajor ? Order::kRowMajor : Order::kColMajor;
}

inline void Transpose(Layout* layout) {
  Transpose(&layout->order);
  std::swap(layout->rows, layout->cols);
}

template <typename Scalar>
inline void Transpose(Matrix<Scalar>* matrix) {
  Transpose(&matrix->layout);
}

// Helpers for KernelLayout.

template <typename FixedKernelLayout>
KernelLayout ToKernelLayout() {
  KernelLayout ret;
  ret.order = FixedKernelLayout::kOrder;
  ret.rows = FixedKernelLayout::kRows;
  ret.cols = FixedKernelLayout::kCols;
  return ret;
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_INTERNAL_MATRIX_H_
