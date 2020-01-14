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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_ARM_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_ARM_H_

#include <cstddef>
#include <cstdint>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/kernel_common.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"
#include "tensorflow/lite/experimental/ruy/spec.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

#if RUY_PLATFORM(NEON) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#if RUY_PLATFORM(NEON_64)
void Kernel8bitNeonOutOfOrder(const KernelParams8bit<4, 4>& params);
void Kernel8bitNeonOutOfOrder1Col(const KernelParams8bit<4, 4>& params);
#elif RUY_PLATFORM(NEON_32)
void Kernel8bitNeonOutOfOrder(const KernelParams8bit<4, 2>& params);
void Kernel8bitNeonOutOfOrder1Col(const KernelParams8bit<4, 2>& params);
#endif
void Kernel8bitNeonInOrder(const KernelParams8bit<4, 4>& params);
void Kernel8bitNeonDotprodOutOfOrder(const KernelParams8bit<8, 8>& params);
void Kernel8bitNeonDotprodOutOfOrder1Col(const KernelParams8bit<8, 8>& params);
void Kernel8bitNeonDotprodInOrder(const KernelParams8bit<8, 8>& params);

#if RUY_PLATFORM(NEON_64)
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
    if (dst->layout.cols == 1) {
      Kernel8bitNeonOutOfOrder1Col(params);
      return;
    }
    if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
      Kernel8bitNeonInOrder(params);
    } else {
      Kernel8bitNeonOutOfOrder(params);
    }
  }
};
#endif

#if RUY_PLATFORM(NEON_32)
template <typename DstScalar>
struct Kernel<Path::kNeon, std::int8_t, std::int8_t, DstScalar,
              BasicSpec<std::int32_t, DstScalar>> {
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 16, 4>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 16, 2>;
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
    if (dst->layout.cols == 1) {
      Kernel8bitNeonOutOfOrder1Col(params);
      return;
    }
    Kernel8bitNeonOutOfOrder(params);
  }
};
#endif

#if RUY_PLATFORM(NEON_64)
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
    if (dst->layout.cols == 1) {
      Kernel8bitNeonDotprodOutOfOrder1Col(params);
    } else if (__builtin_expect(tuning == Tuning::kInOrder, true)) {
      Kernel8bitNeonDotprodInOrder(params);
    } else {
      Kernel8bitNeonDotprodOutOfOrder(params);
    }
  }
};
#endif

void KernelFloatNeonOutOfOrder(const KernelParamsFloat<8, 8>& params);
void KernelFloatNeonInOrder(const KernelParamsFloat<8, 8>& params);
void KernelFloat32NeonOutOfOrder(const KernelParamsFloat<8, 4>& params);
void KernelFloatNeonDotprodInOrder(const KernelParamsFloat<8, 8>& params);

#if RUY_PLATFORM(NEON_64)
// A Float kernel for ARM64 Neon.
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
#endif

#if RUY_PLATFORM(NEON_32)
// A Float kernel for ARM32 Neon.
template <>
struct Kernel<Path::kNeon, float, float, float, BasicSpec<float, float>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 4>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PackedMatrix<float>& lhs, const PackedMatrix<float>& rhs,
           const BasicSpec<float, float>& spec, int start_row, int start_col,
           int end_row, int end_col, Matrix<float>* dst) const {
    KernelParamsFloat<8, 4> params;

    MakeKernelParamsFloat(lhs, rhs, spec, start_row, start_col, end_row,
                          end_col, dst, &params);

    KernelFloat32NeonOutOfOrder(params);
  }
};
#endif

// While the dotprod NEON extension does not concern floating-point arithmetic,
// its presence allows us to distinguish, in the in-order tuning case, between
// A53 and A55r1. TODO: should this be folded into tuning?
template <>
struct Kernel<Path::kNeonDotprod, float, float, float,
              BasicSpec<float, float>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 8>;
  using Base =
      Kernel<Path::kNeon, float, float, float, BasicSpec<float, float>>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
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

#endif  // RUY_PLATFORM(NEON) && RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_ARM_H_
