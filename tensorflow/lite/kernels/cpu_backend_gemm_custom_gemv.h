/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Fast Gemv (i.e. matrix*vector multiplication) paths.
// TODO(b/132094390): remove when GEMM performance is good enough on GEMV cases.

// TFLite's runtime ops concentrate as much as possible the matrix*vector
// use cases on the (matrix) * (column-vector) case, as opposed to
// (row-vector) * (matrix).  So that is what we focus on optimizing here.
// Accordingly, the public cpu_backend_gemm::Gemm() entry point checks
// if we are in this (matrix) * (column-vector) case, and if so calls
// CustomGemv.
//
// cpu_backend_gemm::Gemm is also currently restricted (as enforced in
// ValidateParams) to the case where the left-hand side matrix is row-major.
//
// So the current scope of this CustomGemv function really is:
// (row-major matrix) * (column-vector).

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_CUSTOM_GEMV_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_CUSTOM_GEMV_H_

#include <stdint.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

// CustomGemvImpl is what needs to be specialized for each custom GEMV path.
//
// It does not deal with any multi-threaded implementation detail. Rather,
// it provides the single-thread implementation to be run by each thread.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct CustomGemvImpl {
  // The number of rows of the left-hand-side matrix (and equivalently of the
  // destination column-vector) that the kernel processes at a time.
  // This will also be the minimum required number of rows for a Gemv shape
  // to be supported by this path.
  //
  // Gemv implementations are expected to be able to deal with numbers of
  // rows that aren't multiples of kKernelRows by possibly running the kernel
  // again at an odd row_start, e.g. if kKernelRows==4, Run() should still
  // support running on 7 rows by running twice: once with row_start=0 and then
  // another time with row_start=3.
  //
  // On the other hand, gemv implementations are not expected to support
  // running on fewer than kKernelRows rows. There is no interest in
  // optimizing such narrow Gemv's that they are just a few dot-products.
  // Supporting that would require custom kernel code only for that case.
  static constexpr int kKernelRows = 1;

  // Returns true if the Gemv shape is supported by Run(), provided that
  // (row_end - row_start) > kKernelRows.
  static bool IsSupportedGivenSufficientlyManyRows(
      const MatrixParams<LhsScalar>& lhs_params,
      const MatrixParams<RhsScalar>& rhs_params,
      const MatrixParams<DstScalar>& dst_params,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params) {
    return false;
  }

  // Performs the Gemv.
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      int row_start, int row_end) {}
};

// Wraps CustomGemvImpl for multi-threaded operation.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
class CustomGemvTask : public cpu_backend_threadpool::Task {
 public:
  CustomGemvTask(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      int row_start, int row_end)
      : lhs_params_(lhs_params),
        lhs_data_(lhs_data),
        rhs_params_(rhs_params),
        rhs_data_(rhs_data),
        dst_params_(dst_params),
        dst_data_(dst_data),
        params_(params),
        row_start_(row_start),
        row_end_(row_end) {}

  void Run() override {
    using Impl = CustomGemvImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                                quantization_flavor>;
    Impl::Run(lhs_params_, lhs_data_, rhs_params_, rhs_data_, dst_params_,
              dst_data_, params_, row_start_, row_end_);
  }

 private:
  const MatrixParams<LhsScalar>& lhs_params_;
  const LhsScalar* lhs_data_;
  const MatrixParams<RhsScalar>& rhs_params_;
  const RhsScalar* rhs_data_;
  const MatrixParams<DstScalar>& dst_params_;
  DstScalar* dst_data_;
  const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params_;
  int row_start_;
  int row_end_;
};

// Either performs the requested Gemv operation and returns true,
// or immediately returns false.
//
// See the comment at the top of the file for the scope of what this handles.
// In summary: (row-major matrix) * (column-vector).
//
// Here is only high-level logic.
// The actual implementation details are in specializations of
// CustomGemvImpl.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
bool CustomGemv(
    const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
    const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
    const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
    CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm: CustomGemv");
  using Impl = CustomGemvImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                              quantization_flavor>;
  if (lhs_params.rows < Impl::kKernelRows) {
    return false;
  }
  if (!Impl::IsSupportedGivenSufficientlyManyRows(lhs_params, rhs_params,
                                                  dst_params, params)) {
    return false;
  }
  TFLITE_DCHECK_GE(lhs_params.rows, Impl::kKernelRows);
  int thread_count = LegacyHowManyThreads<Impl::kKernelRows>(
      context->max_num_threads(), dst_params.rows, dst_params.cols,
      lhs_params.cols);
  if (thread_count == 1) {
    Impl::Run(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data,
              params, 0, lhs_params.rows);
  } else {
    using Task = CustomGemvTask<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                                quantization_flavor>;
    std::vector<Task> tasks;
    tasks.reserve(thread_count);
    const int kRowsPerThread =
        RoundUp<Impl::kKernelRows>(CeilQuotient(dst_params.rows, thread_count));
    int row_start = 0;
    for (int i = 0; i < thread_count; i++) {
      int row_end = std::min(dst_params.rows, row_start + kRowsPerThread);
      tasks.emplace_back(lhs_params, lhs_data, rhs_params, rhs_data, dst_params,
                         dst_data, params, row_start, row_end);
      row_start = row_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(), context);
  }
  return true;
}

// USE_NEON still allows for x86 where we may be using the arm_neon_sse.h
// wrapper implementing NEON intrinsics on top of SSE4 intrinsics.
#ifdef USE_NEON

// Some NEON helper functions used by CustomGemvImpl specializations below,
// allowing for some type genericity in them.

inline int16x8x2_t Load16AndSubtractZeroPoint(const std::uint8_t* src,
                                              std::uint8_t zero_point) {
  uint8x16_t src_u8 = vld1q_u8(src);
  int16x8_t src_s16_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_u8)));
  int16x8_t src_s16_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src_u8)));
  int16x8x2_t result;
  int16x8_t zero_point_vec = vdupq_n_s16(zero_point);
  result.val[0] = vsubq_s16(src_s16_0, zero_point_vec);
  result.val[1] = vsubq_s16(src_s16_1, zero_point_vec);
  return result;
}

inline int16x8x2_t Load16AndSubtractZeroPoint(const std::int8_t* src,
                                              std::int8_t zero_point) {
  int8x16_t src_s8 = vld1q_s8(src);
  int16x8_t src_s16_0 = vmovl_s8(vget_low_s8(src_s8));
  int16x8_t src_s16_1 = vmovl_s8(vget_high_s8(src_s8));
  int16x8x2_t result;
  int16x8_t zero_point_vec = vdupq_n_s16(zero_point);
  result.val[0] = vsubq_s16(src_s16_0, zero_point_vec);
  result.val[1] = vsubq_s16(src_s16_1, zero_point_vec);
  return result;
}

inline int16x8_t Load8AndSubtractZeroPoint(const std::uint8_t* src,
                                           std::uint8_t zero_point) {
  uint8x8_t src_u8 = vld1_u8(src);
  int16x8_t src_s16 = vreinterpretq_s16_u16(vmovl_u8(src_u8));
  int16x8_t zero_point_vec = vdupq_n_s16(zero_point);
  return vsubq_s16(src_s16, zero_point_vec);
}

inline int16x8_t Load8AndSubtractZeroPoint(const std::int8_t* src,
                                           std::int8_t zero_point) {
  int8x8_t src_s8 = vld1_s8(src);
  int16x8_t src_s16 = vmovl_s8(src_s8);
  int16x8_t zero_point_vec = vdupq_n_s16(zero_point);
  return vsubq_s16(src_s16, zero_point_vec);
}

inline void ClampAndStore(int32x4_t src, std::uint8_t clamp_min,
                          std::uint8_t clamp_max, std::uint8_t* dst) {
  // Narrow values down to 16 bit signed.
  const int16x4_t res16 = vqmovn_s32(src);
  // Narrow values down to 8 bit unsigned, saturating.
  uint8x8_t res8 = vqmovun_s16(vcombine_s16(res16, res16));
  // Apply the clamping from the activation function
  res8 = vmax_u8(res8, vdup_n_u8(clamp_min));
  res8 = vmin_u8(res8, vdup_n_u8(clamp_max));
  // Store results to destination.
  vst1_lane_u8(dst + 0, res8, 0);
  vst1_lane_u8(dst + 1, res8, 1);
  vst1_lane_u8(dst + 2, res8, 2);
  vst1_lane_u8(dst + 3, res8, 3);
}

inline void ClampAndStore(int32x4_t src, std::int8_t clamp_min,
                          std::int8_t clamp_max, std::int8_t* dst) {
  // Narrow values down to 16 bit signed.
  const int16x4_t res16 = vqmovn_s32(src);
  // Narrow values down to 8 bit unsigned, saturating.
  int8x8_t res8 = vqmovn_s16(vcombine_s16(res16, res16));
  // Apply the clamping from the activation function
  res8 = vmax_s8(res8, vdup_n_s8(clamp_min));
  res8 = vmin_s8(res8, vdup_n_s8(clamp_max));
  // Store results to destination.
  vst1_lane_s8(dst + 0, res8, 0);
  vst1_lane_s8(dst + 1, res8, 1);
  vst1_lane_s8(dst + 2, res8, 2);
  vst1_lane_s8(dst + 3, res8, 3);
}

inline void ClampAndStore(int32x4_t src, std::int16_t clamp_min,
                          std::int16_t clamp_max, std::int16_t* dst) {
  // Narrow values down to 16 bit signed.
  int16x4_t res16 = vqmovn_s32(src);
  // Apply the clamping from the activation function
  res16 = vmax_s16(res16, vdup_n_s16(clamp_min));
  res16 = vmin_s16(res16, vdup_n_s16(clamp_max));
  // Store results to destination.
  vst1_lane_s16(dst + 0, res16, 0);
  vst1_lane_s16(dst + 1, res16, 1);
  vst1_lane_s16(dst + 2, res16, 2);
  vst1_lane_s16(dst + 3, res16, 3);
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct CustomGemvImpl<LhsScalar, RhsScalar, std::int32_t, DstScalar,
                      quantization_flavor> {
  // This partial template specialization is less generic than its declaration
  // implies: it assumes the following constraints on its free template
  // parameters. We guard these assumptions in the following static_assert's.
  static_assert(std::is_same<LhsScalar, std::uint8_t>::value ||
                    std::is_same<LhsScalar, std::int8_t>::value,
                "");
  static_assert(std::is_same<RhsScalar, std::uint8_t>::value ||
                    std::is_same<RhsScalar, std::int8_t>::value,
                "");
  static_assert(std::is_same<DstScalar, std::uint8_t>::value ||
                    std::is_same<DstScalar, std::int8_t>::value ||
                    std::is_same<DstScalar, std::int16_t>::value,
                "");
  static_assert(quantization_flavor ==
                        QuantizationFlavor::kIntegerWithUniformMultiplier ||
                    quantization_flavor ==
                        QuantizationFlavor::kIntegerWithPerRowMultiplier,
                "");

  // This implementation's inner loop processes 4 rows of the left-hand side
  // matrix at a time.
  static constexpr int kKernelRows = 4;

  static bool IsSupportedGivenSufficientlyManyRows(
      const MatrixParams<LhsScalar>& lhs_params,
      const MatrixParams<RhsScalar>& rhs_params,
      const MatrixParams<DstScalar>& dst_params,
      const GemmParams<std::int32_t, DstScalar, quantization_flavor>& params) {
    // The kernel processes at least 8 LHS columns at once to fill NEON
    // registers. The leftovers-handling code at the end works by loading a
    // partially overlapping final register by walking back by a few (<8) values
    // to avoid running past the row's end. This relies on there being
    // at least 8 LHS columns.
    return lhs_params.cols >= 8;
  }

  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<std::int32_t, DstScalar, quantization_flavor>& params,
      int row_start, int row_end) {
    // Handle kKernelRows ( == 4) rows of the left-hand side matrix at each
    // iteration of this for loop.
    TFLITE_DCHECK_GE(row_end - row_start, kKernelRows);
    for (int row = row_start; row < row_end; row += kKernelRows) {
      // Here is the magic where we allow this kernel to handle any odd number
      // of rows as long as it's >= kKernelRows: the last group of `kKernelRows`
      // rows will be nudged to fit, possibly by starting at an odd value of
      // `row`.
      row = std::min(row, row_end - kKernelRows);
      const LhsScalar* filter_ptr = lhs_data + row * lhs_params.cols;

      static constexpr int kCacheLineSize = 64;
      for (int k = 0; k < rhs_params.rows;
           k += kCacheLineSize / sizeof(RhsScalar)) {
        optimized_ops_preload_l1_keep(rhs_data + k);
      }

      // kPreloadAhead is empirically determined.
      // End-to-end latency (ms) on mobilenet_v2_0.35_96_8bit, 1 thread,
      // on Qualcomm S855:
      //
      // kPreloadAhead | big core | little core
      // --------------+----------+------------
      // 64            | 1.26     | 5.45
      // 128           | 1.23     | 5.01
      // 256           | 1.18     | 4.9
      // 512           | 1.18     | 5.45
      // 1024          | 1.18     | 6.5
      // no prefetch   | 1.25     | 8.1
      static constexpr int kPreloadAhead = 256;

      // 4 accumulator registers, one for each row being processed.
      // Each has 4 int32 lanes that corresponds to columns modulo 4, and
      // will need to be horizontally reduced at the end.
      int32x4_t acc0 = vdupq_n_s32(0);
      int32x4_t acc1 = acc0;
      int32x4_t acc2 = acc0;
      int32x4_t acc3 = acc0;
      int in = 0;
      // As much as possible, handle 16 columns of the left-hand side matrix
      // at a time. This allows for decent NEON implementation.
      for (; in <= lhs_params.cols - 16; in += 16) {
        const LhsScalar* local_filter_ptr = filter_ptr;
        int16x8x2_t input_val =
            Load16AndSubtractZeroPoint(rhs_data + in, rhs_params.zero_point);
        int16x8x2_t filter_val_0 =
            Load16AndSubtractZeroPoint(local_filter_ptr, lhs_params.zero_point);
        optimized_ops_preload_l1_stream(local_filter_ptr +
                                        kPreloadAhead / sizeof(LhsScalar));
        local_filter_ptr += lhs_params.cols;
        int16x8x2_t filter_val_1 =
            Load16AndSubtractZeroPoint(local_filter_ptr, lhs_params.zero_point);
        optimized_ops_preload_l1_stream(local_filter_ptr +
                                        kPreloadAhead / sizeof(LhsScalar));
        local_filter_ptr += lhs_params.cols;
        int16x8x2_t filter_val_2 =
            Load16AndSubtractZeroPoint(local_filter_ptr, lhs_params.zero_point);
        optimized_ops_preload_l1_stream(local_filter_ptr +
                                        kPreloadAhead / sizeof(LhsScalar));
        local_filter_ptr += lhs_params.cols;
        int16x8x2_t filter_val_3 =
            Load16AndSubtractZeroPoint(local_filter_ptr, lhs_params.zero_point);
        optimized_ops_preload_l1_stream(local_filter_ptr +
                                        kPreloadAhead / sizeof(LhsScalar));
        filter_ptr += 16;
        acc0 = vmlal_s16(acc0, vget_low_s16(filter_val_0.val[0]),
                         vget_low_s16(input_val.val[0]));
        acc1 = vmlal_s16(acc1, vget_low_s16(filter_val_1.val[0]),
                         vget_low_s16(input_val.val[0]));
        acc2 = vmlal_s16(acc2, vget_low_s16(filter_val_2.val[0]),
                         vget_low_s16(input_val.val[0]));
        acc3 = vmlal_s16(acc3, vget_low_s16(filter_val_3.val[0]),
                         vget_low_s16(input_val.val[0]));
        acc0 = vmlal_s16(acc0, vget_low_s16(filter_val_0.val[1]),
                         vget_low_s16(input_val.val[1]));
        acc1 = vmlal_s16(acc1, vget_low_s16(filter_val_1.val[1]),
                         vget_low_s16(input_val.val[1]));
        acc2 = vmlal_s16(acc2, vget_low_s16(filter_val_2.val[1]),
                         vget_low_s16(input_val.val[1]));
        acc3 = vmlal_s16(acc3, vget_low_s16(filter_val_3.val[1]),
                         vget_low_s16(input_val.val[1]));
        acc0 = vmlal_s16(acc0, vget_high_s16(filter_val_0.val[0]),
                         vget_high_s16(input_val.val[0]));
        acc1 = vmlal_s16(acc1, vget_high_s16(filter_val_1.val[0]),
                         vget_high_s16(input_val.val[0]));
        acc2 = vmlal_s16(acc2, vget_high_s16(filter_val_2.val[0]),
                         vget_high_s16(input_val.val[0]));
        acc3 = vmlal_s16(acc3, vget_high_s16(filter_val_3.val[0]),
                         vget_high_s16(input_val.val[0]));
        acc0 = vmlal_s16(acc0, vget_high_s16(filter_val_0.val[1]),
                         vget_high_s16(input_val.val[1]));
        acc1 = vmlal_s16(acc1, vget_high_s16(filter_val_1.val[1]),
                         vget_high_s16(input_val.val[1]));
        acc2 = vmlal_s16(acc2, vget_high_s16(filter_val_2.val[1]),
                         vget_high_s16(input_val.val[1]));
        acc3 = vmlal_s16(acc3, vget_high_s16(filter_val_3.val[1]),
                         vget_high_s16(input_val.val[1]));
      }
      // Less that 16 values remain. Try to handle 8 more.
      if (in <= lhs_params.cols - 8) {
        int16x8_t input_val =
            Load8AndSubtractZeroPoint(rhs_data + in, rhs_params.zero_point);
        int16x8_t filter_val_0 = Load8AndSubtractZeroPoint(
            filter_ptr + 0 * lhs_params.cols, lhs_params.zero_point);
        int16x8_t filter_val_1 = Load8AndSubtractZeroPoint(
            filter_ptr + 1 * lhs_params.cols, lhs_params.zero_point);
        int16x8_t filter_val_2 = Load8AndSubtractZeroPoint(
            filter_ptr + 2 * lhs_params.cols, lhs_params.zero_point);
        int16x8_t filter_val_3 = Load8AndSubtractZeroPoint(
            filter_ptr + 3 * lhs_params.cols, lhs_params.zero_point);
        filter_ptr += 8;
        acc0 = vmlal_s16(acc0, vget_low_s16(filter_val_0),
                         vget_low_s16(input_val));
        acc1 = vmlal_s16(acc1, vget_low_s16(filter_val_1),
                         vget_low_s16(input_val));
        acc2 = vmlal_s16(acc2, vget_low_s16(filter_val_2),
                         vget_low_s16(input_val));
        acc3 = vmlal_s16(acc3, vget_low_s16(filter_val_3),
                         vget_low_s16(input_val));
        acc0 = vmlal_s16(acc0, vget_high_s16(filter_val_0),
                         vget_high_s16(input_val));
        acc1 = vmlal_s16(acc1, vget_high_s16(filter_val_1),
                         vget_high_s16(input_val));
        acc2 = vmlal_s16(acc2, vget_high_s16(filter_val_2),
                         vget_high_s16(input_val));
        acc3 = vmlal_s16(acc3, vget_high_s16(filter_val_3),
                         vget_high_s16(input_val));
        in += 8;
      }
      // Less than 8 values remain. Handle the remaining values
      // in one more copy of the above code handling 8, where we
      // walk back a few values to be able to load 8 values without
      // overrunning the buffer. This is where we make use of the requirement
      // (see IsSupportedGivenSufficientlyManyRows) that there at least
      // 8 LHS columns.
      if (in < lhs_params.cols) {
        // `back` is how many entries to walk back by.
        // Its value is necessarily between 1 and 7.
        const int back = in + 8 - lhs_params.cols;
        TFLITE_DCHECK_GE(back, 1);
        TFLITE_DCHECK_LE(back, 7);
        // Load 8 values as usual.
        int16x8_t input_val = Load8AndSubtractZeroPoint(
            rhs_data + lhs_params.cols - 8, rhs_params.zero_point);
        const LhsScalar* local_filter_ptr = filter_ptr - back;
        filter_ptr += lhs_params.cols - in;
        int16x8_t filter_val_0 =
            Load8AndSubtractZeroPoint(local_filter_ptr, lhs_params.zero_point);
        local_filter_ptr += lhs_params.cols;
        int16x8_t filter_val_1 =
            Load8AndSubtractZeroPoint(local_filter_ptr, lhs_params.zero_point);
        local_filter_ptr += lhs_params.cols;
        int16x8_t filter_val_2 =
            Load8AndSubtractZeroPoint(local_filter_ptr, lhs_params.zero_point);
        local_filter_ptr += lhs_params.cols;
        int16x8_t filter_val_3 =
            Load8AndSubtractZeroPoint(local_filter_ptr, lhs_params.zero_point);
        // Now zero out the `back` first entries of input_val.
        // vsetq_lane_s16 takes a literal index, so we need unrolled code.
        switch (back) {
          case 7:
            input_val = vsetq_lane_s16(0, input_val, 6);
            [[clang::fallthrough]];
          case 6:
            input_val = vsetq_lane_s16(0, input_val, 5);
            [[clang::fallthrough]];
          case 5:
            input_val = vsetq_lane_s16(0, input_val, 4);
            [[clang::fallthrough]];
          case 4:
            input_val = vsetq_lane_s16(0, input_val, 3);
            [[clang::fallthrough]];
          case 3:
            input_val = vsetq_lane_s16(0, input_val, 2);
            [[clang::fallthrough]];
          case 2:
            input_val = vsetq_lane_s16(0, input_val, 1);
            [[clang::fallthrough]];
          default:
            input_val = vsetq_lane_s16(0, input_val, 0);
        }
        // Multiply-accumulate 8 values as usual. The `back` first lanes
        // of filter_val_* are junk, but it doesn't matter since they get
        // multiplied by the zeros that we just wrote in the corresponding
        // lanes of input_val.
        acc0 = vmlal_s16(acc0, vget_low_s16(filter_val_0),
                         vget_low_s16(input_val));
        acc1 = vmlal_s16(acc1, vget_low_s16(filter_val_1),
                         vget_low_s16(input_val));
        acc2 = vmlal_s16(acc2, vget_low_s16(filter_val_2),
                         vget_low_s16(input_val));
        acc3 = vmlal_s16(acc3, vget_low_s16(filter_val_3),
                         vget_low_s16(input_val));
        acc0 = vmlal_s16(acc0, vget_high_s16(filter_val_0),
                         vget_high_s16(input_val));
        acc1 = vmlal_s16(acc1, vget_high_s16(filter_val_1),
                         vget_high_s16(input_val));
        acc2 = vmlal_s16(acc2, vget_high_s16(filter_val_2),
                         vget_high_s16(input_val));
        acc3 = vmlal_s16(acc3, vget_high_s16(filter_val_3),
                         vget_high_s16(input_val));
      }

      // Horizontally reduce accumulators
      int32x2_t pairwise_reduced_acc_0 =
          vpadd_s32(vget_low_s32(acc0), vget_high_s32(acc0));
      int32x2_t pairwise_reduced_acc_1 =
          vpadd_s32(vget_low_s32(acc1), vget_high_s32(acc1));
      int32x2_t pairwise_reduced_acc_2 =
          vpadd_s32(vget_low_s32(acc2), vget_high_s32(acc2));
      int32x2_t pairwise_reduced_acc_3 =
          vpadd_s32(vget_low_s32(acc3), vget_high_s32(acc3));
      const int32x2_t reduced_lo =
          vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
      const int32x2_t reduced_hi =
          vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
      int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
      // End of horizontal reduction: now `reduced` is a single int32x4
      // containing the 4 int32 accumulators corresponding to the 4 rows
      // being processed.

      // Add bias values.
      if (params.bias) {
        int32x4_t bias_vec = vld1q_s32(params.bias + row);
        reduced = vaddq_s32(reduced, bias_vec);
      }

      // Get multiplier parameters.
      int32x4_t multiplier_fixedpoint;
      int32x4_t multiplier_exponent;
      if (quantization_flavor ==
          QuantizationFlavor::kIntegerWithPerRowMultiplier) {
        multiplier_exponent =
            vld1q_s32(params.multiplier_exponent_perchannel + row);
        multiplier_fixedpoint =
            vld1q_s32(params.multiplier_fixedpoint_perchannel + row);
      } else {
        multiplier_exponent = vdupq_n_s32(params.multiplier_exponent);
        multiplier_fixedpoint = vdupq_n_s32(params.multiplier_fixedpoint);
      }

      // If positive exponent, shift left.
      int32x4_t exponent_positive_part =
          vmaxq_s32(multiplier_exponent, vdupq_n_s32(0));
      reduced = vshlq_s32(reduced, exponent_positive_part);
      // Multiply by the fixed-point multiplier.
      reduced = vqrdmulhq_s32(reduced, multiplier_fixedpoint);
      // If negative exponent, rounding-shift-right.
      int32x4_t exponent_negative_part =
          vminq_s32(multiplier_exponent, vdupq_n_s32(0));
      reduced = vrshlq_s32(reduced, exponent_negative_part);

      // Add the output offset.
      const int32x4_t output_offset_vec = vdupq_n_s32(dst_params.zero_point);
      reduced = vaddq_s32(reduced, output_offset_vec);

      // Finally, clamp and store to the destination.
      ClampAndStore(reduced, params.clamp_min, params.clamp_max,
                    dst_data + row);
    }
  }
};

// The float specialization below is unconditionally faster than ruy
// because ruy does not currently have any Gemv path.
// But it is not unconditionally faster than Eigen, which is what is used
// unless TFLITE_WITH_RUY is defined. Indeed, Eigen has decently efficient
// Gemv paths, and they may use AVX instructions, while the present
// NEON intrinsics code maps at best to SSE4 on x86.
#ifdef TFLITE_WITH_RUY

// We want to use fused multiply-add when it's available (that is, on A64
// unconditionally and on A32 with VFPv4) because it's often faster, and
// because non-fused seems not to be available in A64 so a conscientious
// compiler might emit slow code (separate mul and add instructions) in order to
// implement the vmlaq_f32 intrinsic with strict bit-for-bit exactness on A64.
// (Compilers seem to be generating a fused fmla instruction at the moment,
// but that could change).
//
// We still want to support building for A32 without VFPv4.
inline float32x4_t mul_add(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
#ifdef __ARM_FEATURE_FMA
  return vfmaq_f32(acc, lhs, rhs);
#else
  return vmlaq_f32(acc, lhs, rhs);
#endif
}

template <>
struct CustomGemvImpl<float, float, float, float,
                      QuantizationFlavor::kFloatingPoint> {
  // This implementation's inner loop processes 4 rows of the left-hand side
  // matrix at a time.
  static constexpr int kKernelRows = 4;

  static bool IsSupportedGivenSufficientlyManyRows(
      const MatrixParams<float>& lhs_params,
      const MatrixParams<float>& rhs_params,
      const MatrixParams<float>& dst_params,
      const GemmParams<float, float>& params) {
    // The kernel processes 4 LHS columns at once to fill float32x4 registers.
    // The leftovers-handling code at the end works by loading a partially
    // overlapping final register by walking back by a few (<4) floats
    // to avoid running past the row's end. This relies on there being
    // at least 4 LHS columns.
    return lhs_params.cols >= 4;
  }
  static void Run(const MatrixParams<float>& lhs_params, const float* lhs_data,
                  const MatrixParams<float>& rhs_params, const float* rhs_data,
                  const MatrixParams<float>& dst_params, float* dst_data,
                  const GemmParams<float, float>& params, int row_start,
                  int row_end) {
    // Handle kKernelRows ( == 4) rows of the left-hand side matrix at each
    // iteration of this for loop.
    TFLITE_DCHECK_GE(row_end - row_start, kKernelRows);
    for (int row = row_start; row < row_end; row += kKernelRows) {
      // Here is the magic where we allow this kernel to handle any odd number
      // of rows as long as it's >= kKernelRows: the last group of `kKernelRows`
      // rows will be nudged to fit, possibly by starting at an odd value of
      // `row`.
      row = std::min(row, row_end - kKernelRows);
      const float* filter_ptr = lhs_data + row * lhs_params.cols;

      static constexpr int kCacheLineSize = 64;
      for (int k = 0; k < rhs_params.rows;
           k += kCacheLineSize / sizeof(float)) {
        optimized_ops_preload_l1_keep(rhs_data + k);
      }

      // kPreloadAhead is empirically determined.
      // End-to-end latency (ms) on mobilenet_v2_0.35_96_float, 1 thread,
      // on Qualcomm S855:
      //
      // kPreloadAhead | big core | little core
      // --------------+----------+------------
      // 64            | 2.4      | 15.2
      // 128           | 2.15     | 12.9
      // 256           | 2        | 12.9
      // 512           | 2.08     | 13.3
      // 1024          | 2.05     | 14.7
      // no prefetch   | 2.1      | 28
      static constexpr int kPreloadAhead = 256;

      // 4 accumulator registers, one for each row being processed.
      // Each has 4 float32 lanes that corresponds to columns modulo 4, and
      // will need to be horizontally reduced at the end.
      float32x4_t acc0 = vdupq_n_f32(0);
      float32x4_t acc1 = acc0;
      float32x4_t acc2 = acc0;
      float32x4_t acc3 = acc0;
      int in = 0;
      // As much as possible, handle 4 columns of the left-hand side matrix
      // at a time. This allows for decent NEON implementation.
      for (; in <= lhs_params.cols - 4; in += 4) {
        float32x4_t input_val = vld1q_f32(rhs_data + in);
        const float* local_filter_ptr = filter_ptr;
        float32x4_t filter_val_0 = vld1q_f32(local_filter_ptr);
        optimized_ops_preload_l1_stream(local_filter_ptr +
                                        kPreloadAhead / sizeof(float));
        local_filter_ptr += lhs_params.cols;
        float32x4_t filter_val_1 = vld1q_f32(local_filter_ptr);
        optimized_ops_preload_l1_stream(local_filter_ptr +
                                        kPreloadAhead / sizeof(float));
        local_filter_ptr += lhs_params.cols;
        float32x4_t filter_val_2 = vld1q_f32(local_filter_ptr);
        optimized_ops_preload_l1_stream(local_filter_ptr +
                                        kPreloadAhead / sizeof(float));
        local_filter_ptr += lhs_params.cols;
        float32x4_t filter_val_3 = vld1q_f32(local_filter_ptr);
        optimized_ops_preload_l1_stream(local_filter_ptr +
                                        kPreloadAhead / sizeof(float));
        filter_ptr += 4;
        acc0 = mul_add(acc0, filter_val_0, input_val);
        acc1 = mul_add(acc1, filter_val_1, input_val);
        acc2 = mul_add(acc2, filter_val_2, input_val);
        acc3 = mul_add(acc3, filter_val_3, input_val);
      }
      // Less than 4 values remain. Handle the remaining values
      // in one more copy of the above code handling 4, where we
      // walk back a few values to be able to load 4 values without
      // overrunning the buffer. This is where we make use of the requirement
      // (see IsSupportedGivenSufficientlyManyRows) that there at least
      // 4 LHS columns.
      if (in < lhs_params.cols) {
        // `back` is how many entries to walk back by.
        // Its value is necessarily between 1 and 3.
        const int back = in + 4 - lhs_params.cols;
        TFLITE_DCHECK_GE(back, 1);
        TFLITE_DCHECK_LE(back, 3);
        // Load 4 values as usual.
        float32x4_t input_val = vld1q_f32(rhs_data + lhs_params.cols - 4);
        const float* local_filter_ptr = filter_ptr - back;
        filter_ptr += lhs_params.cols - in;
        float32x4_t filter_val_0 = vld1q_f32(local_filter_ptr);
        local_filter_ptr += lhs_params.cols;
        float32x4_t filter_val_1 = vld1q_f32(local_filter_ptr);
        local_filter_ptr += lhs_params.cols;
        float32x4_t filter_val_2 = vld1q_f32(local_filter_ptr);
        local_filter_ptr += lhs_params.cols;
        float32x4_t filter_val_3 = vld1q_f32(local_filter_ptr);
        // Now zero out the `back` first entries of input_val.
        // vsetq_lane_f32 takes a literal index, so we need unrolled code.
        switch (back) {
          case 3:
            input_val = vsetq_lane_f32(0, input_val, 2);
            [[clang::fallthrough]];
          case 2:
            input_val = vsetq_lane_f32(0, input_val, 1);
            [[clang::fallthrough]];
          default:
            input_val = vsetq_lane_f32(0, input_val, 0);
        }
        // Multiply-accumulate 4 values as usual. The `back` first lanes
        // of filter_val_* are junk, but it doesn't matter since they get
        // multiplied by the zeros that we just wrote in the corresponding
        // lanes of input_val.
        acc0 = mul_add(acc0, filter_val_0, input_val);
        acc1 = mul_add(acc1, filter_val_1, input_val);
        acc2 = mul_add(acc2, filter_val_2, input_val);
        acc3 = mul_add(acc3, filter_val_3, input_val);
      }

      // Horizontally reduce accumulators
      float32x2_t pairwise_reduced_acc_0 =
          vpadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
      float32x2_t pairwise_reduced_acc_1 =
          vpadd_f32(vget_low_f32(acc1), vget_high_f32(acc1));
      float32x2_t pairwise_reduced_acc_2 =
          vpadd_f32(vget_low_f32(acc2), vget_high_f32(acc2));
      float32x2_t pairwise_reduced_acc_3 =
          vpadd_f32(vget_low_f32(acc3), vget_high_f32(acc3));
      float32x2_t reduced_lo =
          vpadd_f32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
      float32x2_t reduced_hi =
          vpadd_f32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
      float32x4_t reduced = vcombine_f32(reduced_lo, reduced_hi);
      // End of horizontal reduction: now `reduced` is a single float32x4
      // containing the 4 float32 accumulators corresponding to the 4 rows
      // being processed.

      if (params.bias) {
        // Add bias values.
        reduced = vaddq_f32(reduced, vld1q_f32(params.bias + row));
      }

      // Clamp and store to destination.
      reduced = vminq_f32(reduced, vdupq_n_f32(params.clamp_max));
      reduced = vmaxq_f32(reduced, vdupq_n_f32(params.clamp_min));
      vst1q_f32(dst_data + row, reduced);
    }
  }
};

#endif  // TFLITE_WITH_RUY

#endif  // USE_NEON

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_CUSTOM_GEMV_H_
