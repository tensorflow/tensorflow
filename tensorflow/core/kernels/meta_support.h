/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_QUANTIZATION_KERNELS_META_SUPPORT_H_
#define TENSORFLOW_CONTRIB_QUANTIZATION_KERNELS_META_SUPPORT_H_

#include "meta/multi_thread_gemm.h"
#include "meta/multi_thread_transform.h"
#include "meta/quantized_mul_kernels.h"
#include "meta/streams.h"
#include "meta/transform_kernels.h"

#include "tensorflow/core/framework/numeric_types.h"

namespace tensorflow {

class OpKernelContext;

namespace meta {

// Gemmlowp/meta is a small library of optimized Arm32/64 kernels for quantized
// matrix multiplication and other quantized computations.

// Set the maximum number of threads of computation that the internal workers
// pool can use. If num_threads is 0, then use intra_op_parallelism_threads.
void SetNumThreads(int num_threads);

int GetNumThreads();

// Toggle the internal workers pool. If set to false, the computations will
// use the worker pool passed each time in the OpKernelContext. If set to true
// then the OpKernelContext will be ignored, and the internal optimized workers
// pool will be used.
//
// The internal workers pool is disabled by default (false).
void SetUseLocalContext(bool use_local_context);

bool GetUseLocalContext();

// Toggles the codepath. Enabled by default (true) on supported platforms.
void SetEnabled(bool enabled);

// Returns true if the codepath is supported and is enabled. Use this call
// before calling the compute functions. If the codepath is not supported, and
// any of the compute function is called, the library will log a FATAL error.
bool IsSupportedAndEnabled();

// Calculate the quantized matrix multiplication:
//
// for (i, j) in [0, m) x [0, n) do
//   c_data[i, j] :=
//     sum((a_data[i, l] + offset_a) * (b_data[l, j] + offset_b)) : l in [0, k)
//
// If transpose_a is false the lhs operand has row major layout, otherwise
// column major. Similarly transpose_b describes the layout of the rhs operand.
// lda, ldb, and ldc are the strides of the lhs operand, rhs operand and the
// result arrays.
void QuantizedGemm(OpKernelContext* context, bool transpose_a, bool transpose_b,
                   const quint8* a_data, const quint8* b_data, qint32* c_data,
                   int m, int n, int k, int offset_a, int offset_b, int lda,
                   int ldb, int ldc);

// Take an array of numbers from the range [input_min, input_max] quantized
// uniformly to int32 values, recover their float values, and then quantize
// them back uniformly to the range [output_min, output_max] as uint8.
// Saturate the uint8 values.
void Requantize(OpKernelContext* context, const qint32* input, int count,
                float input_min, float input_max, float output_min,
                float output_max, quint8* output);

// Take an array of numbers from the range [range_min, range_max] quantized
// uniformly to uint8 values and recover their float values.
void Dequantize(OpKernelContext* context, const quint8* input, int count,
                float range_min, float range_max, float* output);

// Take an array of float values and quantize them uniformly to the range
// [range_min, range_max] expressed as uint8. Saturate the uint8 values.
void Quantize(OpKernelContext*, const float* input, int count, float range_min,
              float range_max, quint8* output);

// Take two arrays: the inputs and the bias quantized uniformly in the ranges
// [input_min, input_max], and [bias_min, bias_max] accordingly, as uint8
// values. Recover their float values. Add the values. Quantize them back
// uniformly to the range [output_min, output_max] as int32. Saturate the
// int32 values.
void QuantizedBiasAdd(OpKernelContext* context, const quint8* input,
                      int input_count, const quint8* bias, int bias_count,
                      float input_min, float input_max, float bias_min,
                      float bias_max, float output_min, float output_max,
                      qint32* output);

// Take an array of uint8 values and clamp them to the range [clamp_min,
// clamp_max].
void Clamp(OpKernelContext* context, const quint8* input, int input_count,
           quint8 clamp_min, quint8 clamp_max, quint8* output);

}  // namespace meta
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_QUANTIZATION_KERNELS_META_SUPPORT_H_
