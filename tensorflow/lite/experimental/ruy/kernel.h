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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "fixedpoint/fixedpoint.h"
#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"
#include "tensorflow/lite/experimental/ruy/spec.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
struct Kernel {};

template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void RunKernelTyped(Tuning tuning, const PackedMatrix<LhsScalar>& lhs,
                    const PackedMatrix<RhsScalar>& rhs, const Spec& spec,
                    int start_row, int start_col, int end_row, int end_col,
                    Matrix<DstScalar>* dst) {
  using Kernel = Kernel<ThePath, LhsScalar, RhsScalar, DstScalar, Spec>;
  Kernel kernel(tuning);
  using LhsLayout = typename Kernel::LhsLayout;
  using RhsLayout = typename Kernel::RhsLayout;
  // end_row and end_col may be larger than dst dimensions.
  // that is because kernels write directly to the destination matrix, whose
  // dimensions may not be a multiple of the kernel dimensions, and we try to
  // keep this annoyance localized as an implementation detail in kernels,
  // by allowing to pass rounded-up values down as far as possible.
  // These assertions encode the contract.
  RUY_DCHECK_LE(0, start_row);
  RUY_DCHECK_LE(start_row, end_row);
  RUY_DCHECK_LT(end_row, dst->layout.rows + LhsLayout::kCols);
  RUY_DCHECK_EQ((end_row - start_row) % LhsLayout::kCols, 0);
  RUY_DCHECK_LE(0, start_col);
  RUY_DCHECK_LE(start_col, end_col);
  RUY_DCHECK_LT(end_col, dst->layout.cols + RhsLayout::kCols);
  RUY_DCHECK_EQ((end_col - start_col) % RhsLayout::kCols, 0);
#if RUY_OPT_SET & RUY_OPT_FAT_KERNEL
  kernel.Run(lhs, rhs, spec, start_row, start_col, end_row, end_col, dst);
#else
  for (int col = start_col; col < end_col; col += RhsLayout::kCols) {
    int block_end_col = std::min(col + RhsLayout::kCols, end_col);
    for (int row = start_row; row < end_row; row += LhsLayout::kCols) {
      int block_end_row = std::min(row + LhsLayout::kCols, end_row);
      kernel.Run(lhs, rhs, spec, row, col, block_end_row, block_end_col, dst);
    }
  }
#endif
}

// Main entry point for kernels.
template <Path ThePath, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
void RunKernel(Tuning tuning, const PMatrix& lhs, const PMatrix& rhs,
               void* spec, int start_row, int start_col, int end_row,
               int end_col, DMatrix* dst) {
  Matrix<DstScalar> mdst = ToMatrix<DstScalar>(*dst);
  RunKernelTyped<ThePath, LhsScalar, RhsScalar, DstScalar, Spec>(
      tuning, ToPackedMatrix<LhsScalar>(lhs), ToPackedMatrix<RhsScalar>(rhs),
      *static_cast<const Spec*>(spec), start_row, start_col, end_row, end_col,
      &mdst);
}

// The signature of RunKernel is the same, regardless of template parameters.
using RunKernelFn =
    decltype(RunKernel<Path::kStandardCpp, std::int8_t, std::int8_t,
                       std::int8_t, BasicSpec<std::int32_t, std::int8_t>>);

// Copied from TF Lite code.
inline std::int32_t MultiplyByQuantizedMultiplier(
    std::int32_t x, std::int32_t quantized_multiplier, int shift) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                                 x * (1 << left_shift), quantized_multiplier),
                             right_shift);
}

// Helper to apply a fixed-point multiplier.  Only 'applicable' if AccumScalar
// is int32 (i.e. in all cases except floating-point) and if the destination is
// not int32 (i.e. unless the user wants to get raw accumulators).
template <typename Spec,
          bool IsApplicable =
              std::is_same<typename Spec::AccumScalar, std::int32_t>::value &&
              !std::is_same<typename Spec::DstScalar, std::int32_t>::value>
struct ApplyMultiplierImpl {};

// Specialization in non-applicable case: do nothing, just check that values
// are default.
template <typename Spec>
struct ApplyMultiplierImpl<Spec, false> {
  using AccumScalar = typename Spec::AccumScalar;
  using DstScalar = typename Spec::DstScalar;
  static void Run(const Spec& spec, int row, AccumScalar* accum) {
    RUY_DCHECK_EQ(spec.multiplier_fixedpoint, 0);
    RUY_DCHECK_EQ(spec.multiplier_exponent, 0);
  }
};

template <typename Spec>
struct ApplyMultiplierImpl<Spec, true> {
  using AccumScalar = typename Spec::AccumScalar;
  using DstScalar = typename Spec::DstScalar;
  static void Run(const Spec& spec, int row, AccumScalar* accum) {
    AccumScalar m = spec.multiplier_fixedpoint_perchannel
                        ? spec.multiplier_fixedpoint_perchannel[row]
                        : spec.multiplier_fixedpoint;
    int e = spec.multiplier_exponent_perchannel
                ? spec.multiplier_exponent_perchannel[row]
                : spec.multiplier_exponent;
    *accum = MultiplyByQuantizedMultiplier(*accum, m, e);
  }
};

template <typename Spec>
void ApplyMultiplier(const Spec& spec, int row,
                     typename Spec::AccumScalar* accum) {
  ApplyMultiplierImpl<Spec>::Run(spec, row, accum);
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct Kernel<Path::kStandardCpp, LhsScalar, RhsScalar, DstScalar, Spec> {
  using AccumScalar = typename Spec::AccumScalar;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  explicit Kernel(Tuning) {}
  void Run(const PackedMatrix<LhsScalar>& lhs,
           const PackedMatrix<RhsScalar>& rhs, const Spec& spec, int start_row,
           int start_col, int end_row, int end_col,
           Matrix<DstScalar>* dst) const {
    // See the comment in RunKernelTyped. end_row may be larger than
    // dst->layout.rows. It's the responsibility of the kernel to avoid
    // overrunning dst boundaries, which we do here by computing
    // clamped_end_row.
    int clamped_end_row = std::min(end_row, dst->layout.rows);
    int clamped_end_col = std::min(end_col, dst->layout.cols);
    RUY_DCHECK_LE(0, start_row);
    RUY_DCHECK_LE(start_row, clamped_end_row);
    RUY_DCHECK_LE(clamped_end_row, dst->layout.rows);
    RUY_DCHECK_LE(clamped_end_row, end_row);
    RUY_DCHECK_LE(end_row - clamped_end_row, LhsLayout::kCols);
    RUY_DCHECK_LE(0, start_col);
    RUY_DCHECK_LE(start_col, clamped_end_col);
    RUY_DCHECK_LE(clamped_end_col, dst->layout.cols);
    RUY_DCHECK_LE(clamped_end_col, end_col);
    RUY_DCHECK_LE(end_col - clamped_end_col, RhsLayout::kCols);
    gemmlowp::ScopedProfilingLabel label("Kernel (Standard Cpp)");
    const int depth = lhs.layout.rows;
    for (int i = start_row; i < clamped_end_row; i++) {
      for (int j = start_col; j < clamped_end_col; j++) {
        using AccumScalar = typename Spec::AccumScalar;
        AccumScalar accum = 0;
        for (int k = 0; k < depth; k++) {
          AccumScalar lhs_val = Element(lhs, k, i);
          AccumScalar rhs_val = Element(rhs, k, j);
          accum += lhs_val * rhs_val;
        }
        if (spec.bias) {
          accum += spec.bias[i];
        }
        if (lhs.zero_point) {
          accum -= lhs.zero_point * rhs.sums[j];
        }
        if (rhs.zero_point) {
          accum -= rhs.zero_point * lhs.sums[i];
        }
        if (lhs.zero_point && rhs.zero_point) {
          accum += lhs.zero_point * rhs.zero_point * depth;
        }
        ApplyMultiplier(spec, i, &accum);
        accum += dst->zero_point;
        accum = std::min<AccumScalar>(accum, spec.clamp_max);
        accum = std::max<AccumScalar>(accum, spec.clamp_min);
        relaxed_atomic_store(ElementPtr(dst, i, j),
                             static_cast<DstScalar>(accum));
      }
    }
  }
};

#define RUY_INHERIT_KERNEL(PARENT, CHILD)                                  \
  template <typename LhsScalar, typename RhsScalar, typename DstScalar,    \
            typename Spec>                                                 \
  struct Kernel<CHILD, LhsScalar, RhsScalar, DstScalar, Spec>              \
      : Kernel<PARENT, LhsScalar, RhsScalar, DstScalar, Spec> {            \
    explicit Kernel(Tuning tuning)                                         \
        : Kernel<PARENT, LhsScalar, RhsScalar, DstScalar, Spec>(tuning) {} \
  };

RUY_INHERIT_KERNEL(Path::kStandardCpp, Path::kNeon)
RUY_INHERIT_KERNEL(Path::kNeon, Path::kNeonDotprod)

#if (defined __aarch64__) && (RUY_OPT_SET & RUY_OPT_ASM)

#define RUY_ASM_FLAG_HAS_BIAS 0x1
#define RUY_ASM_FLAG_HAS_LHS_SUMS 0x2
#define RUY_ASM_FLAG_HAS_RHS_SUMS 0x4
#define RUY_ASM_FLAG_HAS_PERCHANNEL 0x8

#define RUY_ASM_TYPE_ID_UINT8 1
#define RUY_ASM_TYPE_ID_INT8 2
#define RUY_ASM_TYPE_ID_INT16 3
#define RUY_ASM_TYPE_ID_INT32 4

template <typename DstScalar>
struct DstTypeId {};

template <>
struct DstTypeId<std::uint8_t> {
  static constexpr int kValue = RUY_ASM_TYPE_ID_UINT8;
};

template <>
struct DstTypeId<std::int8_t> {
  static constexpr int kValue = RUY_ASM_TYPE_ID_INT8;
};

template <>
struct DstTypeId<std::int16_t> {
  static constexpr int kValue = RUY_ASM_TYPE_ID_INT16;
};

template <>
struct DstTypeId<std::int32_t> {
  static constexpr int kValue = RUY_ASM_TYPE_ID_INT32;
};

template <int LhsCols, int RhsCols>
struct KernelParams8bit {
  static constexpr int kMaxDstTypeSize = 4;

  const std::int32_t* bias;
  const std::int32_t* lhs_sums;
  const std::int32_t* rhs_sums;
  const std::int8_t* lhs_base_ptr;
  const std::int32_t* multiplier_fixedpoint;
  const std::int32_t* multiplier_exponent;
  const std::int8_t* rhs_base_ptr;
  void* dst_base_ptr;
  std::int32_t lhs_zero_point;
  std::int32_t rhs_zero_point;
  std::int32_t dst_zero_point;
  std::int32_t prod_zp_depth;
  std::int32_t start_row;
  std::int32_t start_col;
  std::int32_t last_row;
  std::int32_t last_col;
  std::int32_t dst_rows;
  std::int32_t dst_cols;
  std::int32_t lhs_stride;
  std::int32_t rhs_stride;
  std::int32_t dst_stride;
  std::int32_t depth;
  std::int32_t clamp_min;
  std::int32_t clamp_max;
  std::uint8_t flags;
  std::uint8_t dst_type_id;
  const std::int32_t zero_data[LhsCols] = {0};
  std::uint8_t dst_tmp_buf[LhsCols * RhsCols * kMaxDstTypeSize];
  std::int32_t multiplier_fixedpoint_buf[LhsCols];
  std::int32_t multiplier_exponent_buf[LhsCols];
};

template <typename DstScalar, int LhsCols, int RhsCols>
void MakeKernelParams8bit(const PackedMatrix<std::int8_t>& lhs,
                          const PackedMatrix<std::int8_t>& rhs,
                          const BasicSpec<std::int32_t, DstScalar>& spec,
                          int start_row, int start_col, int end_row,
                          int end_col, Matrix<DstScalar>* dst,
                          KernelParams8bit<LhsCols, RhsCols>* params) {
  using Params = KernelParams8bit<LhsCols, RhsCols>;

  static_assert(sizeof(DstScalar) <= Params::kMaxDstTypeSize, "");

  const int depth = lhs.layout.rows;
  RUY_DCHECK_EQ(start_row % LhsCols, 0);
  RUY_DCHECK_EQ(start_col % RhsCols, 0);
  RUY_DCHECK_EQ(end_row % LhsCols, 0);
  RUY_DCHECK_EQ(end_col % RhsCols, 0);

  params->lhs_base_ptr = lhs.data + start_row * lhs.layout.stride;
  params->rhs_base_ptr = rhs.data + start_col * rhs.layout.stride;
  params->flags = 0;
  params->bias = params->zero_data;
  if (spec.bias) {
    params->bias = spec.bias;
    params->flags |= RUY_ASM_FLAG_HAS_BIAS;
  }
  if (lhs.sums) {
    params->lhs_sums = lhs.sums;
    params->flags |= RUY_ASM_FLAG_HAS_LHS_SUMS;
  }
  if (rhs.sums) {
    params->rhs_sums = rhs.sums;
    params->flags |= RUY_ASM_FLAG_HAS_RHS_SUMS;
  }
  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = lhs.layout.stride;
  params->rhs_stride = rhs.layout.stride;
  params->dst_stride = sizeof(DstScalar) * dst->layout.stride;
  params->lhs_zero_point = lhs.zero_point;
  params->rhs_zero_point = rhs.zero_point;
  params->dst_zero_point = dst->zero_point;
  params->depth = depth;
  params->prod_zp_depth = lhs.zero_point * rhs.zero_point * depth;
  if (spec.multiplier_fixedpoint_perchannel) {
    params->flags |= RUY_ASM_FLAG_HAS_PERCHANNEL;
    params->multiplier_fixedpoint = spec.multiplier_fixedpoint_perchannel;
    params->multiplier_exponent = spec.multiplier_exponent_perchannel;
  } else {
    params->multiplier_fixedpoint = params->multiplier_fixedpoint_buf;
    params->multiplier_exponent = params->multiplier_exponent_buf;
    for (int i = 0; i < LhsCols; i++) {
      params->multiplier_fixedpoint_buf[i] = spec.multiplier_fixedpoint;
      params->multiplier_exponent_buf[i] = spec.multiplier_exponent;
    }
  }
  params->clamp_min = spec.clamp_min;
  params->clamp_max = spec.clamp_max;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);

  params->dst_type_id = DstTypeId<DstScalar>::kValue;
  params->dst_base_ptr =
      dst->data.get() + start_col * dst->layout.stride + start_row;
}

void Kernel8bitNeonOutOfOrder(const KernelParams8bit<4, 4>& params);
void Kernel8bitNeonInOrder(const KernelParams8bit<4, 4>& params);
void Kernel8bitNeonDotprodOutOfOrder(const KernelParams8bit<8, 8>& params);
void Kernel8bitNeonDotprodInOrder(const KernelParams8bit<8, 8>& params);

template <typename DstScalar>
struct Kernel<Path::kNeon, std::int8_t, std::int8_t, DstScalar,
              BasicSpec<std::int32_t, DstScalar>> {
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 16, 4>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 16, 4>;
  Tuning tuning = Tuning::kAuto;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PackedMatrix<std::int8_t>& lhs,
           const PackedMatrix<std::int8_t>& rhs,
           const BasicSpec<std::int32_t, DstScalar>& spec, int start_row,
           int start_col, int end_row, int end_col,
           Matrix<DstScalar>* dst) const {
    KernelParams8bit<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParams8bit(lhs, rhs, spec, start_row, start_col, end_row, end_col,
                         dst, &params);
    if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
      Kernel8bitNeonInOrder(params);
    } else {
      Kernel8bitNeonOutOfOrder(params);
    }
  }
};

template <typename DstScalar>
struct Kernel<Path::kNeonDotprod, std::int8_t, std::int8_t, DstScalar,
              BasicSpec<std::int32_t, DstScalar>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 8>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PackedMatrix<std::int8_t>& lhs,
           const PackedMatrix<std::int8_t>& rhs,
           const BasicSpec<std::int32_t, DstScalar>& spec, int start_row,
           int start_col, int end_row, int end_col,
           Matrix<DstScalar>* dst) const {
    KernelParams8bit<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParams8bit(lhs, rhs, spec, start_row, start_col, end_row, end_col,
                         dst, &params);
    if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
      Kernel8bitNeonDotprodInOrder(params);
    } else {
      Kernel8bitNeonDotprodOutOfOrder(params);
    }
  }
};

template <int LhsCols, int RhsCols>
struct KernelParamsFloat {
  const float* lhs_base_ptr;
  const float* rhs_base_ptr;
  float* dst_base_ptr;
  const float* bias;
  std::int32_t start_row;
  std::int32_t start_col;
  std::int32_t last_row;
  std::int32_t last_col;
  std::int32_t dst_rows;
  std::int32_t dst_cols;
  std::int32_t lhs_stride;
  std::int32_t rhs_stride;
  std::int32_t dst_stride;
  std::int32_t depth;
  float clamp_min;
  float clamp_max;
  std::uint8_t flags;
  const float zero_data[LhsCols] = {0};
  float dst_tmp_buf[LhsCols * RhsCols];
};

template <int LhsCols, int RhsCols>
inline void MakeKernelParamsFloat(const PackedMatrix<float>& lhs,
                                  const PackedMatrix<float>& rhs,
                                  const BasicSpec<float, float>& spec,
                                  int start_row, int start_col, int end_row,
                                  int end_col, Matrix<float>* dst,
                                  KernelParamsFloat<LhsCols, RhsCols>* params) {
  using Params = KernelParamsFloat<LhsCols, RhsCols>;

  const int depth = lhs.layout.rows;
  RUY_DCHECK_EQ(start_row % LhsCols, 0);
  RUY_DCHECK_EQ(start_col % RhsCols, 0);
  RUY_DCHECK_EQ(end_row % LhsCols, 0);
  RUY_DCHECK_EQ(end_col % RhsCols, 0);

  params->lhs_base_ptr = lhs.data + start_row * lhs.layout.stride;
  params->rhs_base_ptr = rhs.data + start_col * rhs.layout.stride;
  params->dst_base_ptr =
      dst->data.get() + start_col * dst->layout.stride + start_row;

  std::uint8_t flags = 0;
  params->bias = params->zero_data;
  if (spec.bias) {
    params->bias = spec.bias;
    flags |= RUY_ASM_FLAG_HAS_BIAS;
  }
  params->flags = flags;
  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = sizeof(float) * lhs.layout.stride;
  params->rhs_stride = sizeof(float) * rhs.layout.stride;
  params->dst_stride = sizeof(float) * dst->layout.stride;
  params->depth = depth;
  params->clamp_min = spec.clamp_min;
  params->clamp_max = spec.clamp_max;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

void KernelFloatNeonOutOfOrder(const KernelParamsFloat<8, 8>& params);
void KernelFloatNeonInOrder(const KernelParamsFloat<8, 8>& params);
void KernelFloatNeonDotprodInOrder(const KernelParamsFloat<8, 8>& params);

template <>
struct Kernel<Path::kNeon, float, float, float, BasicSpec<float, float>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PackedMatrix<float>& lhs, const PackedMatrix<float>& rhs,
           const BasicSpec<float, float>& spec, int start_row, int start_col,
           int end_row, int end_col, Matrix<float>* dst) const {
    KernelParamsFloat<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParamsFloat(lhs, rhs, spec, start_row, start_col, end_row,
                          end_col, dst, &params);
    if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
      KernelFloatNeonInOrder(params);
    } else {
      KernelFloatNeonOutOfOrder(params);
    }
  }
};

// While the dotprod NEON extension does not concern floating-point arithmetic,
// its presence allows us to distinguish, in the in-order tuning case, between
// A53 and A55r1. TODO: should this be folded into tuning?
template <>
struct Kernel<Path::kNeonDotprod, float, float, float, BasicSpec<float, float>>
    : Kernel<Path::kNeon, float, float, float, BasicSpec<float, float>> {
  using Base =
      Kernel<Path::kNeon, float, float, float, BasicSpec<float, float>>;
  explicit Kernel(Tuning tuning_) : Base(tuning_) {}
  void Run(const PackedMatrix<float>& lhs, const PackedMatrix<float>& rhs,
           const BasicSpec<float, float>& spec, int start_row, int start_col,
           int end_row, int end_col, Matrix<float>* dst) const {
    KernelParamsFloat<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParamsFloat(lhs, rhs, spec, start_row, start_col, end_row,
                          end_col, dst, &params);
    if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
      KernelFloatNeonDotprodInOrder(params);
    } else {
      KernelFloatNeonOutOfOrder(params);
    }
  }
};

#endif  // (defined __aarch64__) && (RUY_OPT_SET & RUY_OPT_ASM)

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_H_
