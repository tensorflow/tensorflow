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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_X86_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_X86_H_

#include <cstdint>

#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/kernel_common.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/spec.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

#if RUY_PLATFORM(X86)
void Kernel8bitAvx512(const KernelParams8bit<16, 16>& params);

template <typename DstScalar>
struct Kernel<Path::kAvx512, std::int8_t, std::int8_t, DstScalar,
              BasicSpec<std::int32_t, DstScalar>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 16>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PackedMatrix<std::int8_t>& lhs,
           const PackedMatrix<std::int8_t>& rhs,
           const BasicSpec<std::int32_t, DstScalar>& spec, int start_row,
           int start_col, int end_row, int end_col,
           Matrix<DstScalar>* dst) const {
    KernelParams8bit<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParams8bit(lhs, rhs, spec, start_row, start_col, end_row, end_col,
                         dst, &params);
    Kernel8bitAvx512(params);
  }
};

void KernelFloatAvx512(const KernelParamsFloat<16, 16>& params);

template <>
struct Kernel<Path::kAvx512, float, float, float, BasicSpec<float, float>> {
  Tuning tuning = Tuning::kAuto;
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  using RhsLayout = FixedKernelLayout<Order::kRowMajor, 1, 16>;
  explicit Kernel(Tuning tuning_) : tuning(tuning_) {}
  void Run(const PackedMatrix<float>& lhs, const PackedMatrix<float>& rhs,
           const BasicSpec<float, float>& spec, int start_row, int start_col,
           int end_row, int end_col, Matrix<float>* dst) const {
    KernelParamsFloat<LhsLayout::kCols, RhsLayout::kCols> params;
    MakeKernelParamsFloat(lhs, rhs, spec, start_row, start_col, end_row,
                          end_col, dst, &params);
    KernelFloatAvx512(params);
  }
};

void Kernel8bitAvx2(const KernelParams8bit<8, 8>& params);

template <typename DstScalar>
struct Kernel<Path::kAvx2, std::int8_t, std::int8_t, DstScalar,
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
    Kernel8bitAvx2(params);
  }
};

void KernelFloatAvx2(const KernelParamsFloat<8, 8>& params);

template <>
struct Kernel<Path::kAvx2, float, float, float, BasicSpec<float, float>> {
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
    KernelFloatAvx2(params);
  }
};
#endif  // RUY_PLATFORM(X86)

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_KERNEL_X86_H_
