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

// # What is "packing"?
//
// Before feeding data to the gemm kernels (the parts of Ruy that do lots
// of multiply-add operations), Ruy first performs a data transformation (which
// we call "packing") on the input matrices. This transformation has two main
// goals:
// - rearrange data into blocks that are a convenient size/layout for the gemm
// kernels to consume. This helps make the memory access pattern of the gemm
// kernel simpler and more contiguous, and puts the data in a layout most
// convenient for specific arithmetic instructions in the gemm kernel.
// - compute row/column sums needed for handling quantization with non-symmetric
// zero points.
//
// # Simplified algorithmic analysis of packing
//
// Packing is a relatively simple transformation which does a small constant
// amount of work on each element of an input matrix, and hence for an NxM
// matrix performs O(N*M) work. If N and M are of the same order, then this is
// O(N^2) work.
//
// A NxKxM matrix multiplication requires N*K*M multiply-accumulate operations.
// Note that if N, K, and M are all the same order, then the number of
// multiply-accumulate operations is O(N^3).
//
// Thus, the O(N^2) cost of packing is small compared to the O(N^3) work, in the
// case of all dimensions being roughly the same order.
//
// # Packing cost can be significant
//
// When matrix * matrix multiplications begin to look more like matrix * vector
// multiplications, packing cost can become significant. We sometimes call these
// cases "gemv-like".
//
// Continuing the algorithmic analysis above, if we consider a case where an
// NxKxM matrix multiplication has either N = O(1) or M = O(1), then the
// situation is different. In this case, the multiply-accumulate work is only
// quadratic, so the quadratic cost of packing can be come significant.
//
// Another way to say this is that the cost of packing an input matrix (either
// the LHS or RHS) is amortized across the non-depth dimension of the opposite
// input matrix. Thus, when the LHS has very few rows or the RHS has very few
// columns, the cost of packing the opposite input matrix can become
// significant.
//
// As a rough rule of thumb, the cost of packing starts to become significant
// when either N or M is below 32 (and other dimensions are hundreds), with very
// significant packing costs at 8 or below. This varies by data type, Path, and
// tuning, so these numbers are only rough guides.
//
// One practical use case that is affected by this is inference of
// fully connected neural network layers with a low batch size. The weight
// matrix (which is a constant for inference) is the one affected by significant
// packing cost.
//
// Ruy provides an API in ruy_advanced.h for advanced users to pre-pack
// input matrices that are affected by significant packing costs.
//
// # Implementation notes
//
// Ruy's packing routines always operate on a range of columns and can be
// applied to either the LHS or RHS. This is possible because Ruy internally
// implements a TrMul, so the accumulation along depth is done along columns of
// both the LHS and RHS (whereas for a normal Mul the accumulation along depth
// for the LHS is along rows). As another example, we are always computing
// column sums for quantization (and never row sums, since the LHS is
// transposed).

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_X86_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_X86_H_

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/pack_common.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

#if RUY_PLATFORM(X86)
// Note that source and zero buffers can be uint8 type, but in the packing
// function are reinterpreted as int8, and are XOR-ed with input_xor.
void Pack8bitAvx2(const std::int8_t* src_ptr, std::int8_t input_xor,
                  const std::int8_t* zerobuf, int src_stride,
                  int remaining_src_cols, int src_rows, std::int8_t* packed_ptr,
                  std::int32_t* sums_ptr);

template <typename Scalar>
struct PackImpl<Path::kAvx2, FixedKernelLayout<Order::kColMajor, 4, 8>, Scalar,
                std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  using Layout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  static constexpr std::int8_t kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    gemmlowp::ScopedProfilingLabel label("Pack (AVX2 8-bit)");

    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[Layout::kCols * Layout::kRows];
    memset(zerobuf, packed_matrix->zero_point ^ kInputXor,
           Layout::kCols * Layout::kRows * sizeof(Scalar));
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      std::int8_t* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      Pack8bitAvx2(reinterpret_cast<const std::int8_t*>(src_ptr), kInputXor,
                   reinterpret_cast<const std::int8_t*>(zerobuf), src_stride,
                   remaining_src_cols, src_matrix.layout.rows, packed_ptr,
                   sums_ptr);
    }
  }
};

void PackFloatAvx2(const float* src_ptr, const float* zerobuf, int src_stride,
                   int remaining_src_cols, int src_rows, float* packed_ptr);

template <>
struct PackImpl<Path::kAvx2, FixedKernelLayout<Order::kRowMajor, 1, 8>, float,
                float, float> {
  using Layout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  static void Run(Tuning, const Matrix<float>& src_matrix,
                  PackedMatrix<float>* packed_matrix, int start_col,
                  int end_col) {
    gemmlowp::ScopedProfilingLabel label("Pack (AVX2 float)");
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    const float zerobuf[Layout::kCols] = {
        0.0f};  // Remainder default inits to 0.0f.
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      float* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      PackFloatAvx2(src_ptr, zerobuf, src_stride, remaining_src_cols,
                    src_matrix.layout.rows, packed_ptr);
    }
  }
};

// Note that source and zero buffers can be uint8 type, but in the packing
// function are reinterpreted as int8, and are XOR-ed with input_xor.
void Pack8bitAvx512(const std::int8_t* src_ptr, std::int8_t input_xor,
                    const std::int8_t* zerobuf, int src_stride,
                    int remaining_src_cols, int src_rows,
                    std::int8_t* packed_ptr, std::int32_t* sums_ptr);

template <typename Scalar>
struct PackImpl<Path::kAvx512, FixedKernelLayout<Order::kColMajor, 4, 16>,
                Scalar, std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  using Layout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  static constexpr int kHalfLayoutCols =
      8;  // Half the number of cols in a block.
  static constexpr std::int8_t kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    gemmlowp::ScopedProfilingLabel label("Pack (AVX-512 8-bit)");

    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    RUY_DCHECK_EQ(kHalfLayoutCols * 2, Layout::kCols);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[kHalfLayoutCols * Layout::kRows];
    memset(zerobuf, packed_matrix->zero_point ^ kInputXor,
           kHalfLayoutCols * Layout::kRows * sizeof(Scalar));
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      std::int8_t* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      Pack8bitAvx512(reinterpret_cast<const std::int8_t*>(src_ptr), kInputXor,
                     reinterpret_cast<const std::int8_t*>(zerobuf), src_stride,
                     remaining_src_cols, src_matrix.layout.rows, packed_ptr,
                     sums_ptr);
    }
  }
};

void PackFloatAvx512(const float* src_ptr, const float* zerobuf, int src_stride,
                     int remaining_src_cols, int src_rows, float* packed_ptr);

template <>
struct PackImpl<Path::kAvx512, FixedKernelLayout<Order::kRowMajor, 1, 16>,
                float, float, float> {
  static void Run(Tuning, const Matrix<float>& src_matrix,
                  PackedMatrix<float>* packed_matrix, int start_col,
                  int end_col) {
    gemmlowp::ScopedProfilingLabel label("Pack (AVX-512 float)");
    using Layout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ((end_col - start_col) % Layout::kCols, 0);
    RUY_DCHECK_EQ(start_col % Layout::kCols, 0);
    const float zerobuf[Layout::kCols] = {
        0.0f};  // Remainder default inits to 0.0f.
    for (int block_col = start_col; block_col < end_col;
         block_col += Layout::kCols) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr = src_matrix.data.get() + src_stride * block_col;
      int remaining_src_cols = src_matrix.layout.cols - block_col;

      static constexpr int block_col_mask = ~(Layout::kCols - 1);  // High bits.
      float* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & block_col_mask);
      PackFloatAvx512(src_ptr, zerobuf, src_stride, remaining_src_cols,
                      src_matrix.layout.rows, packed_ptr);
    }
  }
};
#endif  // RUY_PLATFORM(X86)

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_X86_H_
