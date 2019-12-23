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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_COMMON_H_

#include <cstdint>

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

template <Path ThePath, typename Scalar>
struct PackedTypeImpl {
  using Type = Scalar;
};

#if RUY_PLATFORM(NEON_32)
struct PackParams8bit {
  const void* src_ptr0;
  const void* src_ptr1;
  const void* src_ptr2;
  const void* src_ptr3;
  const std::int32_t* sums_ptr;
  const std::int8_t* packed_ptr;
  int src_inc0;
  int src_inc1;
  int src_inc2;
  int src_inc3;
  int src_rows;
  int src_zero_point;
  int input_xor;
};

inline void MakePackParams8bit(const void* src_ptr0, const void* src_ptr1,
                               const void* src_ptr2, const void* src_ptr3,
                               const std::int32_t* sums_ptr,
                               const std::int8_t* packed_ptr, int src_inc0,
                               int src_inc1, int src_inc2, int src_inc3,
                               int src_rows, int src_zero_point, int input_xor,
                               PackParams8bit* params) {
  params->src_ptr0 = src_ptr0;
  params->src_ptr1 = src_ptr1;
  params->src_ptr2 = src_ptr2;
  params->src_ptr3 = src_ptr3;
  params->sums_ptr = sums_ptr;
  params->packed_ptr = packed_ptr;
  params->src_inc0 = src_inc0;
  params->src_inc1 = src_inc1;
  params->src_inc2 = src_inc2;
  params->src_inc3 = src_inc3;
  params->src_rows = src_rows;
  params->src_zero_point = src_zero_point;
  params->input_xor = input_xor;
}
#endif

#if RUY_PLATFORM(NEON)
template <>
struct PackedTypeImpl<Path::kNeon, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kNeonDotprod, std::uint8_t> {
  using Type = std::int8_t;
};
#elif RUY_PLATFORM(X86)
template <>
struct PackedTypeImpl<Path::kAvx2, std::uint8_t> {
  using Type = std::int8_t;
};
template <>
struct PackedTypeImpl<Path::kAvx512, std::uint8_t> {
  using Type = std::int8_t;
};
#endif

template <Path ThePath, typename Scalar>
using PackedType = typename PackedTypeImpl<ThePath, Scalar>::Type;

template <typename PackedScalar, typename Scalar>
PackedScalar Pack(Scalar x) {
  return x - SymmetricZeroPoint<Scalar>() + SymmetricZeroPoint<PackedScalar>();
}

template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar, typename SumsType>
struct PackImpl {};

#define RUY_INHERIT_PACK(PARENT, CHILD)                                       \
  template <typename FixedKernelLayout, typename Scalar,                      \
            typename PackedScalar, typename SumsType>                         \
  struct PackImpl<CHILD, FixedKernelLayout, Scalar, PackedScalar, SumsType>   \
      : PackImpl<PARENT, FixedKernelLayout, Scalar, PackedScalar, SumsType> { \
  };

template <typename FixedKernelLayout, typename Scalar, typename PackedScalar,
          typename SumsType>
struct PackImpl<Path::kStandardCpp, FixedKernelLayout, Scalar, PackedScalar,
                SumsType> {
  static void Run(Tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<PackedScalar>* packed_matrix, int start_col,
                  int end_col) {
    gemmlowp::ScopedProfilingLabel label("Pack (generic)");
    RUY_DCHECK_EQ((end_col - start_col) % FixedKernelLayout::kCols, 0);
    SumsType* sums = packed_matrix->sums;
    for (int col = start_col; col < end_col; col++) {
      SumsType accum = 0;
      for (int row = 0; row < packed_matrix->layout.rows; row++) {
        PackedScalar packed_val;
        if (col < src_matrix.layout.cols && row < src_matrix.layout.rows) {
          packed_val = Pack<PackedScalar>(Element(src_matrix, row, col));
        } else {
          packed_val = packed_matrix->zero_point;
        }
        accum += packed_val;
        *ElementPtr(packed_matrix, row, col) = packed_val;
      }
      if (sums) {
        sums[col] = accum;
      }
    }
  }
};

#if RUY_PLATFORM(NEON)
RUY_INHERIT_PACK(Path::kStandardCpp, Path::kNeon)
#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
RUY_INHERIT_PACK(Path::kNeon, Path::kNeonDotprod)
#endif
#elif RUY_PLATFORM(X86)
RUY_INHERIT_PACK(Path::kStandardCpp, Path::kAvx2)
RUY_INHERIT_PACK(Path::kAvx2, Path::kAvx512)
#endif

// Main entry point for packing.
template <Path ThePath, typename FixedKernelLayout, typename Scalar,
          typename PackedScalar>
void RunPack(Tuning tuning, const DMatrix& src_matrix, PMatrix* packed_matrix,
             int start_col, int end_col) {
  using SumsType = typename PackedMatrix<PackedScalar>::SumsType;
  Matrix<Scalar> src = ToMatrix<Scalar>(src_matrix);
  PackedMatrix<PackedScalar> packed =
      ToPackedMatrix<PackedScalar>(*packed_matrix);
  PackImpl<ThePath, FixedKernelLayout, Scalar, PackedScalar, SumsType>::Run(
      tuning, src, &packed, start_col, end_col);
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_COMMON_H_
