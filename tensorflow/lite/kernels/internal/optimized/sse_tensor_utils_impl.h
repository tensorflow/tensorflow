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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_IMPL_H_

#include <cstdint>

#include "tensorflow/lite/kernels/cpu_backend_context.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

#ifdef __SSSE3__

// Matrix multiplication for quantized values using symmetric quantization.
void SseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result);

// Matrix multiplication for quantized values using symmetric quantization
// with additional scratch memory for GEMM operation prior to scaling.
void SseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch, int32_t* scratch,
    float* __restrict__ result, CpuBackendContext* context);

// Matrix multiplication for quantized values using asymmetric quantization.
void SseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context);

// Matrix multiplication for quantized values using symmetric quantization.
// Sparse version.
void SseSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    const int m_rows, const int m_cols, const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result);

void SseReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                           const int output_size, const int reduction_size);

#endif  // __SSSE3__

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_IMPL_H_
