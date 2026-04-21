/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_REFERENCE_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_REFERENCE_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

template <typename T>
T GetElement(const MatrixParams<T>& params, const T* data, int r, int c) {
  if (params.order == Order::kRowMajor) {
    return data[r * params.cols + c];
  } else {
    return data[c * params.rows + r];
  }
}

template <typename T>
void SetElement(const MatrixParams<T>& params, T* data, int r, int c, T val) {
  if (params.order == Order::kRowMajor) {
    data[r * params.cols + c] = val;
  } else {
    data[c * params.rows + r] = val;
  }
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingReference {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
    int M = lhs_params.rows;
    int K = lhs_params.cols;
    int N = rhs_params.cols;

    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        AccumScalar acc = 0;
        for (int k = 0; k < K; ++k) {
          AccumScalar lhs_val;
          AccumScalar rhs_val;
          if constexpr (quantization_flavor ==
                        QuantizationFlavor::kFloatingPoint) {
            TFLITE_DCHECK_EQ(lhs_params.zero_point, 0);
            TFLITE_DCHECK_EQ(rhs_params.zero_point, 0);
            lhs_val = GetElement(lhs_params, lhs_data, m, k);
            rhs_val = GetElement(rhs_params, rhs_data, k, n);
          } else {
            lhs_val =
                GetElement(lhs_params, lhs_data, m, k) - lhs_params.zero_point;
            rhs_val =
                GetElement(rhs_params, rhs_data, k, n) - rhs_params.zero_point;
          }
          acc += lhs_val * rhs_val;
        }

        if (params.bias) {
          acc += params.bias[m];
        }

        DstScalar dst_val;
        if constexpr (quantization_flavor ==
                      QuantizationFlavor::kFloatingPoint) {
          dst_val = acc;
        } else if constexpr (quantization_flavor ==
                             QuantizationFlavor::
                                 kIntegerWithUniformMultiplier) {
          if constexpr (std::is_same_v<DstScalar, int32_t>) {
            // Raw integer case, no multiplier
            dst_val = acc;
          } else {
            int32_t scaled_acc = MultiplyByQuantizedMultiplier(
                acc, params.multiplier_fixedpoint, params.multiplier_exponent);
            scaled_acc += dst_params.zero_point;
            dst_val = static_cast<DstScalar>(scaled_acc);
          }
        } else if constexpr (quantization_flavor ==
                             QuantizationFlavor::kIntegerWithPerRowMultiplier) {
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, params.multiplier_fixedpoint_perchannel[m],
              params.multiplier_exponent_perchannel[m]);
          scaled_acc += dst_params.zero_point;
          dst_val = static_cast<DstScalar>(scaled_acc);
        }

        dst_val = std::max(dst_val, params.clamp_min);
        dst_val = std::min(dst_val, params.clamp_max);

        SetElement(dst_params, dst_data, m, n, dst_val);
      }
    }
  }
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_REFERENCE_H_
