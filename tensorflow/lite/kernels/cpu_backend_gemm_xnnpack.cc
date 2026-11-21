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

#include "tensorflow/lite/kernels/cpu_backend_gemm_xnnpack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#ifdef TFLITE_KERNEL_USE_XNNPACK
#include "xnnpack.h"  // from @XNNPACK
#include "pthreadpool.h"  // from @pthreadpool
#endif

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

#ifdef TFLITE_KERNEL_USE_XNNPACK
namespace {

// Matches XNNPACK's internal XNN_ALLOCATION_ALIGNMENT (64 bytes) for operator
// workspaces without including internal XNNPACK headers.
constexpr size_t kXnnWorkspaceAlignment = 64;

// XNNPACK requantization kernels require scale in [2^-32, 256.0).
constexpr float kMinXnnRequantizationScale = 0x1.0p-32f;
constexpr float kMaxXnnRequantizationScale = 256.0f;

struct XnnOperatorDeleter {
  void operator()(xnn_operator_t op) const {
    if (op != nullptr) {
      xnn_delete_operator(op);
    }
  }
};
using UniqueXnnOperator = std::unique_ptr<xnn_operator, XnnOperatorDeleter>;

struct AlignedWorkspaceDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      ::operator delete(ptr, std::align_val_t{kXnnWorkspaceAlignment});
    }
  }
};
using AlignedWorkspace = std::unique_ptr<void, AlignedWorkspaceDeleter>;

AlignedWorkspace AllocateWorkspace(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  return AlignedWorkspace(::operator new(
      size + XNN_EXTRA_BYTES, std::align_val_t{kXnnWorkspaceAlignment}));
}

// XNNPACK SIMD microkernels may read up to XNN_EXTRA_BYTES past the end of
// unpacked input buffers. Callers of cpu_backend_gemm pass raw pointers (e.g.
// std::vector or sub-tensor slices) that are not guaranteed to carry tail
// padding, so pad the unpacked input operand before invoking XNNPACK.
template <typename T>
class PaddedInputBuffer {
 public:
  PaddedInputBuffer(const T* src, size_t count) {
    const size_t bytes = count * sizeof(T);
    data_ = std::make_unique<uint8_t[]>(bytes + XNN_EXTRA_BYTES);
    std::memcpy(data_.get(), src, bytes);
    std::memset(data_.get() + bytes, 0, XNN_EXTRA_BYTES);
  }

  const T* data() const { return reinterpret_cast<const T*>(data_.get()); }

 private:
  std::unique_ptr<uint8_t[]> data_;
};

bool ConvertQuantizedMultiplierToScale(int32_t multiplier_fixedpoint,
                                       int multiplier_exponent,
                                       float* scale_out) {
  // TFLite's MultiplyByQuantizedMultiplier computes
  // round(acc * q * 2^(e - 31)).
  const float scale = std::ldexp(static_cast<float>(multiplier_fixedpoint),
                                 multiplier_exponent - 31);
  if (!std::isnormal(scale) || scale < kMinXnnRequantizationScale ||
      scale >= kMaxXnnRequantizationScale) {
    return false;
  }
  *scale_out = scale;
  return true;
}

pthreadpool_t GetThreadpool(CpuBackendContext* context) {
  return context != nullptr ? context->get_xnnpack_threadpool() : nullptr;
}

}  // namespace
#endif  // TFLITE_KERNEL_USE_XNNPACK

bool GemmImplUsingXnnpack<float, float, float, float,
                          QuantizationFlavor::kFloatingPoint>::
    Run(const MatrixParams<float>& lhs_params, const float* lhs_data,
        const MatrixParams<float>& rhs_params, const float* rhs_data,
        const MatrixParams<float>& dst_params, float* dst_data,
        const GemmParams<float, float, QuantizationFlavor::kFloatingPoint>&
            params,
        CpuBackendContext* context) {
#ifdef TFLITE_KERNEL_USE_XNNPACK
  if (lhs_params.rows != dst_params.rows ||
      lhs_params.cols != rhs_params.rows ||
      rhs_params.cols != dst_params.cols) {
    return false;
  }
  if (std::isnan(params.clamp_min) || std::isnan(params.clamp_max) ||
      params.clamp_min > params.clamp_max) {
    return false;
  }
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return false;
  }

  const size_t m = static_cast<size_t>(lhs_params.rows);
  const size_t k = static_cast<size_t>(lhs_params.cols);
  const size_t n = static_cast<size_t>(rhs_params.cols);
  pthreadpool_t threadpool = GetThreadpool(context);

  // Canonical TFLite layout: Dst is ColMajor[M, N] (= RowMajor[N, M]) and Rhs
  // is ColMajor[K, N] (= RowMajor[N, K]). This maps directly to XNNPACK
  // dynamic_fully_connected_nc with batch_size=N, input=Rhs, kernel=Lhs.
  if (dst_params.order == Order::kColMajor &&
      rhs_params.order == Order::kColMajor) {
    const uint32_t flags =
        lhs_params.order == Order::kColMajor ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0;
    xnn_operator_t raw_op = nullptr;
    if (xnn_create_dynamic_fully_connected_nc_f32(
            params.clamp_min, params.clamp_max, flags, &raw_op) !=
        xnn_status_success) {
      return false;
    }
    UniqueXnnOperator op(raw_op);

    size_t workspace_size = 0;
    if (xnn_reshape_dynamic_fully_connected_nc_f32(
            op.get(), /*batch_size=*/n, /*input_channels=*/k,
            /*output_channels=*/m, /*input_stride=*/k, /*output_stride=*/m,
            &workspace_size, threadpool) != xnn_status_success) {
      return false;
    }
    AlignedWorkspace workspace = AllocateWorkspace(workspace_size);
    PaddedInputBuffer<float> padded_rhs(rhs_data, n * k);

    if (xnn_setup_dynamic_fully_connected_nc_f32(
            op.get(), workspace.get(), padded_rhs.data(), lhs_data, params.bias,
            dst_data) != xnn_status_success) {
      return false;
    }
    return xnn_run_operator(op.get(), threadpool) == xnn_status_success;
  }

  // RowMajor Dst[M, N] = RowMajor Lhs[M, K] * Rhs[K, N] via
  // batch_matrix_multiply.
  if (dst_params.order == Order::kRowMajor &&
      lhs_params.order == Order::kRowMajor) {
    const uint32_t flags =
        rhs_params.order == Order::kColMajor ? XNN_FLAG_TRANSPOSE_B : 0;
    xnn_operator_t raw_op = nullptr;
    if (xnn_create_batch_matrix_multiply_nc_f32(flags, &raw_op) !=
        xnn_status_success) {
      return false;
    }
    UniqueXnnOperator op(raw_op);

    const size_t batch_dim = 1;
    size_t workspace_size = 0;
    if (xnn_reshape_batch_matrix_multiply_nc_f32(
            op.get(), /*num_batch_dims=*/1, &batch_dim, &batch_dim, m, k, n,
            &workspace_size, threadpool) != xnn_status_success) {
      return false;
    }
    AlignedWorkspace workspace = AllocateWorkspace(workspace_size);
    PaddedInputBuffer<float> padded_lhs(lhs_data, m * k);
    PaddedInputBuffer<float> padded_rhs(rhs_data, k * n);

    if (xnn_setup_batch_matrix_multiply_nc_f32(
            op.get(), workspace.get(), padded_lhs.data(), padded_rhs.data(),
            dst_data) != xnn_status_success) {
      return false;
    }
    if (xnn_run_operator(op.get(), threadpool) != xnn_status_success) {
      return false;
    }

    const bool has_clamp =
        params.clamp_min > -std::numeric_limits<float>::infinity() ||
        params.clamp_max < std::numeric_limits<float>::infinity();
    if (params.bias != nullptr || has_clamp) {
      for (size_t r = 0; r < m; ++r) {
        const float bias_val = params.bias != nullptr ? params.bias[r] : 0.0f;
        float* row_ptr = dst_data + r * n;
        for (size_t c = 0; c < n; ++c) {
          float v = row_ptr[c] + bias_val;
          v = std::max(v, params.clamp_min);
          v = std::min(v, params.clamp_max);
          row_ptr[c] = v;
        }
      }
    }
    return true;
  }
#endif  // TFLITE_KERNEL_USE_XNNPACK
  return false;
}

bool GemmImplUsingXnnpack<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                          QuantizationFlavor::kIntegerWithUniformMultiplier>::
    Run(const MatrixParams<std::int8_t>& lhs_params,
        const std::int8_t* lhs_data,
        const MatrixParams<std::int8_t>& rhs_params,
        const std::int8_t* rhs_data,
        const MatrixParams<std::int8_t>& dst_params, std::int8_t* dst_data,
        const GemmParams<std::int32_t, std::int8_t,
                         QuantizationFlavor::kIntegerWithUniformMultiplier>&
            params,
        CpuBackendContext* context) {
#ifdef TFLITE_KERNEL_USE_XNNPACK
  if (lhs_params.rows != dst_params.rows ||
      lhs_params.cols != rhs_params.rows ||
      rhs_params.cols != dst_params.cols) {
    return false;
  }
  if (dst_params.order != Order::kColMajor ||
      rhs_params.order != Order::kColMajor || lhs_params.zero_point != 0 ||
      params.clamp_min > params.clamp_max) {
    return false;
  }

  float scale = 0.0f;
  if (!ConvertQuantizedMultiplierToScale(params.multiplier_fixedpoint,
                                         params.multiplier_exponent, &scale)) {
    return false;
  }
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return false;
  }

  const size_t m = static_cast<size_t>(lhs_params.rows);
  const size_t k = static_cast<size_t>(lhs_params.cols);
  const size_t n = static_cast<size_t>(rhs_params.cols);
  const uint32_t flags =
      lhs_params.order == Order::kColMajor ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0;
  pthreadpool_t threadpool = GetThreadpool(context);

  xnn_operator_t raw_op = nullptr;
  if (xnn_create_fully_connected_nc_qs8(
          /*input_channels=*/k, /*output_channels=*/m, /*input_stride=*/k,
          /*output_stride=*/m, /*input_zero_point=*/rhs_params.zero_point,
          /*input_scale=*/1.0f, /*kernel_scale=*/scale, lhs_data, params.bias,
          /*output_zero_point=*/dst_params.zero_point, /*output_scale=*/1.0f,
          params.clamp_min, params.clamp_max, flags,
          /*weights_cache=*/nullptr, &raw_op) != xnn_status_success) {
    return false;
  }
  UniqueXnnOperator op(raw_op);

  if (xnn_reshape_fully_connected_nc_qs8(op.get(), /*batch_size=*/n,
                                         threadpool) != xnn_status_success) {
    return false;
  }
  PaddedInputBuffer<std::int8_t> padded_rhs(rhs_data, n * k);
  if (xnn_setup_fully_connected_nc_qs8(op.get(), padded_rhs.data(), dst_data) !=
      xnn_status_success) {
    return false;
  }
  return xnn_run_operator(op.get(), threadpool) == xnn_status_success;
#else
  return false;
#endif  // TFLITE_KERNEL_USE_XNNPACK
}

bool GemmImplUsingXnnpack<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                          QuantizationFlavor::kIntegerWithPerRowMultiplier>::
    Run(const MatrixParams<std::int8_t>& lhs_params,
        const std::int8_t* lhs_data,
        const MatrixParams<std::int8_t>& rhs_params,
        const std::int8_t* rhs_data,
        const MatrixParams<std::int8_t>& dst_params, std::int8_t* dst_data,
        const GemmParams<std::int32_t, std::int8_t,
                         QuantizationFlavor::kIntegerWithPerRowMultiplier>&
            params,
        CpuBackendContext* context) {
#ifdef TFLITE_KERNEL_USE_XNNPACK
  if (lhs_params.rows != dst_params.rows ||
      lhs_params.cols != rhs_params.rows ||
      rhs_params.cols != dst_params.cols) {
    return false;
  }
  if (dst_params.order != Order::kColMajor ||
      rhs_params.order != Order::kColMajor || lhs_params.zero_point != 0 ||
      params.clamp_min > params.clamp_max ||
      params.multiplier_fixedpoint_perchannel == nullptr ||
      params.multiplier_exponent_perchannel == nullptr) {
    return false;
  }

  const size_t m = static_cast<size_t>(lhs_params.rows);
  const size_t k = static_cast<size_t>(lhs_params.cols);
  const size_t n = static_cast<size_t>(rhs_params.cols);

  std::vector<float> kernel_scales(m);
  for (size_t i = 0; i < m; ++i) {
    if (!ConvertQuantizedMultiplierToScale(
            params.multiplier_fixedpoint_perchannel[i],
            params.multiplier_exponent_perchannel[i], &kernel_scales[i])) {
      return false;
    }
  }
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return false;
  }

  const uint32_t flags =
      lhs_params.order == Order::kColMajor ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0;
  pthreadpool_t threadpool = GetThreadpool(context);

  xnn_operator_t raw_op = nullptr;
  if (xnn_create_fully_connected_nc_qs8_qc8w(
          /*input_channels=*/k, /*output_channels=*/m, /*input_stride=*/k,
          /*output_stride=*/m, /*input_zero_point=*/rhs_params.zero_point,
          /*input_scale=*/1.0f, kernel_scales.data(), lhs_data, params.bias,
          /*output_zero_point=*/dst_params.zero_point, /*output_scale=*/1.0f,
          params.clamp_min, params.clamp_max, flags,
          /*weights_cache=*/nullptr, &raw_op) != xnn_status_success) {
    return false;
  }
  UniqueXnnOperator op(raw_op);

  if (xnn_reshape_fully_connected_nc_qs8_qc8w(
          op.get(), /*batch_size=*/n, threadpool) != xnn_status_success) {
    return false;
  }
  PaddedInputBuffer<std::int8_t> padded_rhs(rhs_data, n * k);
  if (xnn_setup_fully_connected_nc_qs8_qc8w(op.get(), padded_rhs.data(),
                                            dst_data) != xnn_status_success) {
    return false;
  }
  return xnn_run_operator(op.get(), threadpool) == xnn_status_success;
#else
  return false;
#endif  // TFLITE_KERNEL_USE_XNNPACK
}

bool GemmImplUsingXnnpack<std::uint8_t, std::uint8_t, std::int32_t,
                          std::uint8_t,
                          QuantizationFlavor::kIntegerWithUniformMultiplier>::
    Run(const MatrixParams<std::uint8_t>& lhs_params,
        const std::uint8_t* lhs_data,
        const MatrixParams<std::uint8_t>& rhs_params,
        const std::uint8_t* rhs_data,
        const MatrixParams<std::uint8_t>& dst_params, std::uint8_t* dst_data,
        const GemmParams<std::int32_t, std::uint8_t,
                         QuantizationFlavor::kIntegerWithUniformMultiplier>&
            params,
        CpuBackendContext* context) {
#ifdef TFLITE_KERNEL_USE_XNNPACK
  if (lhs_params.rows != dst_params.rows ||
      lhs_params.cols != rhs_params.rows ||
      rhs_params.cols != dst_params.cols) {
    return false;
  }
  if (dst_params.order != Order::kColMajor ||
      rhs_params.order != Order::kColMajor ||
      params.clamp_min > params.clamp_max) {
    return false;
  }

  float scale = 0.0f;
  if (!ConvertQuantizedMultiplierToScale(params.multiplier_fixedpoint,
                                         params.multiplier_exponent, &scale)) {
    return false;
  }
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return false;
  }

  const size_t m = static_cast<size_t>(lhs_params.rows);
  const size_t k = static_cast<size_t>(lhs_params.cols);
  const size_t n = static_cast<size_t>(rhs_params.cols);
  const uint32_t flags =
      lhs_params.order == Order::kColMajor ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0;
  pthreadpool_t threadpool = GetThreadpool(context);

  xnn_operator_t raw_op = nullptr;
  if (xnn_create_fully_connected_nc_qu8(
          /*input_channels=*/k, /*output_channels=*/m, /*input_stride=*/k,
          /*output_stride=*/m, /*input_zero_point=*/rhs_params.zero_point,
          /*input_scale=*/1.0f, /*kernel_zero_point=*/lhs_params.zero_point,
          /*kernel_scale=*/scale, lhs_data, params.bias,
          /*output_zero_point=*/dst_params.zero_point, /*output_scale=*/1.0f,
          params.clamp_min, params.clamp_max, flags,
          /*weights_cache=*/nullptr, &raw_op) != xnn_status_success) {
    return false;
  }
  UniqueXnnOperator op(raw_op);

  if (xnn_reshape_fully_connected_nc_qu8(op.get(), /*batch_size=*/n,
                                         threadpool) != xnn_status_success) {
    return false;
  }
  PaddedInputBuffer<std::uint8_t> padded_rhs(rhs_data, n * k);
  if (xnn_setup_fully_connected_nc_qu8(op.get(), padded_rhs.data(), dst_data) !=
      xnn_status_success) {
    return false;
  }
  return xnn_run_operator(op.get(), threadpool) == xnn_status_success;
#else
  return false;
#endif  // TFLITE_KERNEL_USE_XNNPACK
}

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite
