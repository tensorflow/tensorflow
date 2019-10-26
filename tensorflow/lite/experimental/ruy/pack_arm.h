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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_ARM_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_ARM_H_

#include <cstdint>
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

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void Pack8bitNeonOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                            const void* src_ptr2, const void* src_ptr3,
                            int src_inc0, int src_inc1, int src_inc2,
                            int src_inc3, int src_rows, int src_zero_point,
                            std::int8_t* packed_ptr, int start_col, int end_col,
                            std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonInOrder(const void* src_ptr0, const void* src_ptr1,
                         const void* src_ptr2, const void* src_ptr3,
                         int src_inc0, int src_inc1, int src_inc2, int src_inc3,
                         int src_rows, int src_zero_point,
                         std::int8_t* packed_ptr, int start_col, int end_col,
                         std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonDotprodOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                                   const void* src_ptr2, const void* src_ptr3,
                                   int src_inc0, int src_inc1, int src_inc2,
                                   int src_inc3, int src_rows,
                                   int src_zero_point, std::int8_t* packed_ptr,
                                   int start_col, int end_col,
                                   std::int32_t* sums_ptr, int input_xor);
void Pack8bitNeonDotprodInOrder(const void* src_ptr0, const void* src_ptr1,
                                const void* src_ptr2, const void* src_ptr3,
                                int src_inc0, int src_inc1, int src_inc2,
                                int src_inc3, int src_rows, int src_zero_point,
                                std::int8_t* packed_ptr, int start_col,
                                int end_col, std::int32_t* sums_ptr,
                                int input_xor);

#elif RUY_PLATFORM(NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void Pack8bitNeonOutOfOrder4Cols(const PackParams8bit& params);
void Pack8bitNeonOutOfOrder2Cols(const PackParams8bit& params);
#endif  // (RUY_PLATFORM(NEON_64)&& RUY_OPT_ENABLED(RUY_OPT_ASM)

#if (RUY_PLATFORM(NEON_32) || RUY_PLATFORM(NEON_64)) && \
    RUY_OPT_ENABLED(RUY_OPT_ASM)

template <typename Scalar>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kColMajor, 16, 4>, Scalar,
                std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      int src_inc0 = 16;
      int src_inc1 = 16;
      int src_inc2 = 16;
      int src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * block_col;
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
#if RUY_PLATFORM(NEON_64)
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        Pack8bitNeonInOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      } else {
        Pack8bitNeonOutOfOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      }
#else
      // We have a more limited set of general purpose registers in ARMv7, so
      // we use the "params" struct technique from the kernel code to save
      // registers.
      PackParams8bit params;
      MakePackParams8bit(src_ptr0, src_ptr1, src_ptr2, src_ptr3, sums_ptr,
                         packed_ptr, src_inc0, src_inc1, src_inc2, src_inc3,
                         src_matrix.layout.rows, src_matrix.zero_point,
                         kInputXor, &params);
      Pack8bitNeonOutOfOrder4Cols(params);
#endif  // RUY_PLATFORM(NEON_64)
    }
  }
};

#endif  // (RUY_PLATFORM(NEON_32) || RUY_PLATFORM(NEON_64)) &&
        // RUY_OPT_ENABLED(RUY_OPT_ASM)

#if RUY_PLATFORM(NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)
// The 32-bit float kernel is 4 rows X 2 columns, so we need an additional
// partial specialization for the RHS, which has a FixedKernelLayout with 2
// columns.
template <typename Scalar>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kColMajor, 16, 2>, Scalar,
                std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;
  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 2, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 2) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      int src_inc0 = 16;
      int src_inc1 = 16;
      if (block_col >= src_matrix.layout.cols - 2) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * block_col;
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      PackParams8bit params;
      MakePackParams8bit(src_ptr0, src_ptr1, nullptr, nullptr, sums_ptr,
                         packed_ptr, src_inc0, src_inc1, -1, -1,
                         src_matrix.layout.rows, src_matrix.zero_point,
                         kInputXor, &params);
      Pack8bitNeonOutOfOrder2Cols(params);
    }
  }
};
#endif  // (RUY_PLATFORM(NEON_32)) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
template <typename Scalar>
struct PackImpl<Path::kNeonDotprod, FixedKernelLayout<Order::kColMajor, 4, 8>,
                Scalar, std::int8_t, std::int32_t> {
  static_assert(std::is_same<Scalar, std::int8_t>::value ||
                    std::is_same<Scalar, std::uint8_t>::value,
                "");
  static constexpr int kInputXor =
      std::is_same<Scalar, std::int8_t>::value ? 0 : 0x80;

  static void Run(Tuning tuning, const Matrix<Scalar>& src_matrix,
                  PackedMatrix<std::int8_t>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    std::int32_t* sums = packed_matrix->sums;
    Scalar zerobuf[16];
    memset(zerobuf, src_matrix.zero_point, sizeof(zerobuf));
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const Scalar* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const Scalar* src_ptr1 = src_ptr0 + src_stride;
      const Scalar* src_ptr2 = src_ptr1 + src_stride;
      const Scalar* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      std::int8_t* packed_ptr =
          packed_matrix->data +
          packed_matrix->layout.stride * (block_col & ~7) +
          ((block_col & 4) * 4);
      std::int32_t* sums_ptr = sums ? sums + block_col : nullptr;
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        Pack8bitNeonDotprodInOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      } else {
        Pack8bitNeonDotprodOutOfOrder(
            src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0, src_inc1,
            src_inc2, src_inc3, src_matrix.layout.rows, src_matrix.zero_point,
            packed_ptr, start_col, end_col, sums_ptr, kInputXor);
      }
    }
  }
};
#endif  // (RUY_PLATFORM(NEON_64)&& RUY_OPT_ENABLED(RUY_OPT_ASM)

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void PackFloatNeonOutOfOrder(const float* src_ptr0, const float* src_ptr1,
                             const float* src_ptr2, const float* src_ptr3,
                             int src_inc0, int src_inc1, int src_inc2,
                             int src_inc3, int src_rows, int src_zero_point,
                             float* packed_ptr, int start_col, int end_col);
void PackFloatNeonInOrder(const float* src_ptr0, const float* src_ptr1,
                          const float* src_ptr2, const float* src_ptr3,
                          int src_inc0, int src_inc1, int src_inc2,
                          int src_inc3, int src_rows, int src_zero_point,
                          float* packed_ptr, int start_col, int end_col);

#elif RUY_PLATFORM(NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void PackFloatNeonOutOfOrder(const float* src_ptr0, const float* src_ptr1,
                             const float* src_ptr2, const float* src_ptr3,
                             int src_inc, int src_rows, int src_zero_point,
                             float* packed_ptr, int start_col, int end_col,
                             int stride);
#endif  // (RUY_PLATFORM(NEON_64)&& RUY_OPT_ENABLED(RUY_OPT_ASM)

#if (RUY_PLATFORM(NEON_32) || RUY_PLATFORM(NEON_64)) && \
    RUY_OPT_ENABLED(RUY_OPT_ASM)

template <>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kRowMajor, 1, 8>, float,
                float, float> {
  static void Run(Tuning tuning, const Matrix<float>& src_matrix,
                  PackedMatrix<float>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 8, 0);
    const float zerobuf[4] = {0};
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const float* src_ptr1 = src_ptr0 + src_stride;
      const float* src_ptr2 = src_ptr1 + src_stride;
      const float* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      float* packed_ptr = packed_matrix->data +
                          packed_matrix->layout.stride * (block_col & ~7) +
                          ((block_col & 4));
#if RUY_PLATFORM(NEON_64)
      if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
        PackFloatNeonInOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc0,
                             src_inc1, src_inc2, src_inc3,
                             src_matrix.layout.rows, src_matrix.zero_point,
                             packed_ptr, start_col, end_col);
      } else {
        PackFloatNeonOutOfOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3,
                                src_inc0, src_inc1, src_inc2, src_inc3,
                                src_matrix.layout.rows, src_matrix.zero_point,
                                packed_ptr, start_col, end_col);
      }
#else
      // Encode each of src_inc0, ..., src_inc3 in lowest 4 bits of src_inc
      // to save on registers (we have fewer general purpose registers in
      // 32-bit ARM than in 64-bit ARM). For the 64-bit case, we pass four
      // values that are each either 16 or 0 and use them directly. For the
      // 32-bit case, bits 0, 1, 2, and 3 are used to determine if we should
      // use the value 16 (bit is set) or 0 (bit is not set) for the
      // respective increment value.
      std::int64_t src_inc = 0;
      src_inc += src_inc0 == 16 ? 1 : 0;
      src_inc += src_inc1 == 16 ? 2 : 0;
      src_inc += src_inc2 == 16 ? 4 : 0;
      src_inc += src_inc3 == 16 ? 8 : 0;
      const int kOutputStride = 32;
      PackFloatNeonOutOfOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc,
                              src_matrix.layout.rows, src_matrix.zero_point,
                              packed_ptr, start_col, end_col, kOutputStride);
#endif  // RUY_PLATFORM(NEON_64)
    }
  }
};

#if RUY_PLATFORM(NEON_32)
// The 32-bit float kernel is 8 rows X 4 columns, so we need an additional
// specialization for a FixedKernelLayout with 4 columns.
template <>
struct PackImpl<Path::kNeon, FixedKernelLayout<Order::kRowMajor, 1, 4>, float,
                float, float> {
  static void Run(Tuning tuning, const Matrix<float>& src_matrix,
                  PackedMatrix<float>* packed_matrix, int start_col,
                  int end_col) {
    RUY_DCHECK(IsColMajor(src_matrix.layout));
    RUY_DCHECK(IsColMajor(packed_matrix->layout));
    RUY_DCHECK_EQ(start_col % 4, 0);
    const float zerobuf[4] = {0};
    for (int block_col = start_col; block_col < end_col; block_col += 4) {
      int src_stride = src_matrix.layout.stride;
      const float* src_ptr0 = src_matrix.data.get() + src_stride * block_col;
      const float* src_ptr1 = src_ptr0 + src_stride;
      const float* src_ptr2 = src_ptr1 + src_stride;
      const float* src_ptr3 = src_ptr2 + src_stride;
      std::int64_t src_inc0 = 16;
      std::int64_t src_inc1 = 16;
      std::int64_t src_inc2 = 16;
      std::int64_t src_inc3 = 16;
      if (block_col >= src_matrix.layout.cols - 3) {
        if (block_col >= src_matrix.layout.cols - 0) {
          src_ptr0 = zerobuf;
          src_inc0 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 1) {
          src_ptr1 = zerobuf;
          src_inc1 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 2) {
          src_ptr2 = zerobuf;
          src_inc2 = 0;
        }
        if (block_col >= src_matrix.layout.cols - 3) {
          src_ptr3 = zerobuf;
          src_inc3 = 0;
        }
      }
      float* packed_ptr =
          packed_matrix->data + packed_matrix->layout.stride * (block_col);
      // Encode each of src_inc0, ..., src_inc1 in lowest 4 bits of scrc_inc
      // to save registers.
      std::int64_t src_inc = 0;
      src_inc += src_inc0 == 16 ? 1 : 0;
      src_inc += src_inc1 == 16 ? 2 : 0;
      src_inc += src_inc2 == 16 ? 4 : 0;
      src_inc += src_inc3 == 16 ? 8 : 0;
      const int kOutputStride = 16;
      PackFloatNeonOutOfOrder(src_ptr0, src_ptr1, src_ptr2, src_ptr3, src_inc,
                              src_matrix.layout.rows, src_matrix.zero_point,
                              packed_ptr, start_col, end_col, kOutputStride);
    }
  }
};
#endif  // (RUY_PLATFORM(NEON_32))
#endif  // (RUY_PLATFORM(NEON_64) || RUY_PLATFORM(NEON_32)) && \
        // RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_ARM_H_
