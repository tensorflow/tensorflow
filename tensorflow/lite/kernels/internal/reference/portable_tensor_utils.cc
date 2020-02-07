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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/round.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

namespace {
const int32_t kInt16Max = std::numeric_limits<int16_t>::max();
const int32_t kInt16Min = std::numeric_limits<int16_t>::min();
}  // namespace

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor) {
  auto minmax = std::minmax_element(values, values + size);
  *min_value = *minmax.first;
  *max_value = *minmax.second;

  PortableSymmetricQuantizeFloats(values, size, quantized_values, *min_value,
                                  *max_value, scaling_factor);
}

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float min_value,
                                     float max_value, float* scaling_factor) {
  const int kScale = 127;
  const float range = std::max(std::abs(min_value), std::abs(max_value));
  if (range == 0) {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value =
        static_cast<int32_t>(TfLiteRound(values[i] * scaling_factor_inv));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
}

void PortableAsymmetricQuantizeFloats(const float* values, const int size,
                                      int8_t* quantized_values,
                                      float* scaling_factor, int32_t* offset) {
  const int32_t kMinScale = -128;
  const int32_t kMaxScale = 127;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  float rmin = 0.0, rmax = 0.0;
  const auto minmax = std::minmax_element(values, values + size);
  rmin = rmin < *minmax.first ? rmin : *minmax.first;
  rmax = rmax > *minmax.second ? rmax : *minmax.second;
  if (rmin == rmax) {
    *scaling_factor = 0;
    *offset = 0;
  } else {
    const double scale = (rmax - rmin) / (qmax_double - qmin_double);
    const double zero_point_from_min = qmin_double - rmin / scale;
    const double zero_point_from_max = qmax_double - rmax / scale;
    const double zero_point_from_min_error =
        std::abs(qmin_double) + std::abs(rmin / scale);
    const double zero_point_from_max_error =
        std::abs(qmax_double) + std::abs(rmax / scale);
    const double zero_point_double =
        zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;
    int8 nudged_zero_point = 0;
    if (zero_point_double < qmin_double) {
      nudged_zero_point = kMinScale;
    } else if (zero_point_double > qmax_double) {
      nudged_zero_point = kMaxScale;
    } else {
      nudged_zero_point = static_cast<int8>(round(zero_point_double));
    }
    *scaling_factor = scale;
    *offset = nudged_zero_point;
  }
  const float scaling_factor_inv =
      *scaling_factor == 0 ? 0 : 1.0 / *scaling_factor;
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value = static_cast<int32_t>(
        TfLiteRound(*offset + values[i] * scaling_factor_inv));
    quantized_values[i] =
        std::min(kMaxScale, std::max(kMinScale, quantized_value));
  }
}

void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    for (int r = 0; r < m_rows; r++) {
      float dot_prod = 0.0f;
      const float* vector_in_batch = vector + b * m_cols;
      for (int c = 0; c < m_cols; c++) {
        dot_prod += *matrix_ptr++ * *vector_in_batch++;
      }
      *result_in_batch += dot_prod;
      result_in_batch += result_stride;
    }
  }
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride) {
  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Get the address of the first row.
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      // Initialize the dot product sum for the row to 0.
      int32_t dotprod = 0;
#if defined(__GNUC__)
      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);
#endif
      for (int col = 0; col < m_cols; ++col, ++row_ptr) {
        dotprod += (*row_ptr) * (vectors[col]);
      }  // for col
      *result += dotprod * batch_scaling_factor;
    }  // for row
  }    // for batch
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride,
    const float* per_channel_scale, const int32_t* input_offset) {
  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    const float batch_offset = input_offset[batch];
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      int32_t dotprod = 0;
#if defined(__GNUC__)
      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);
#endif
      for (int col = 0; col < m_cols; ++col, ++row_ptr) {
        dotprod += (*row_ptr) * (vectors[col] - batch_offset);
      }  // for col
      *result += dotprod * batch_scaling_factor * per_channel_scale[row];
    }  // for row
  }    // for batch
}

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result, int result_stride) {
  const int kBlockSize = 16;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);
  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    const uint8_t* ledger_ptr = ledger;
    for (int r = 0; r < m_rows; r++) {
      float dot_prod = 0.0f;
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        const float* vector_in_batch = vector + b * m_cols;
        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int block_start_index = *ledger_ptr++ * kBlockSize;
          const float* vector_block_in_batch_ptr =
              vector_in_batch + block_start_index;
          for (int c = 0; c < kBlockSize; c++) {
            dot_prod += *matrix_ptr++ * *vector_block_in_batch_ptr++;
          }
        }
      }
      *result_in_batch += dot_prod;
      result_in_batch += result_stride;
    }
  }
}

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride) {
  static const int kBlockSize = 16;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);
  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    const uint8_t* ledger_ptr = ledger;
    // Get the address of the first row.
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      // Initialize the dot product sum for the row to 0.
      int32_t dotprod = 0;
#if defined(__GNUC__)
      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);
#endif
      int num_nonzero_blocks = *ledger_ptr++;
      for (int i = 0; i < num_nonzero_blocks; i++) {
        const int block_start_index = *ledger_ptr++ * kBlockSize;
        const int8_t* vector_block_ptr = vectors + block_start_index;
        for (int c = 0; c < kBlockSize; c++) {
          dotprod += (*row_ptr++) * (*vector_block_ptr++);
        }  // for block
      }    // for num_nonzero_blocks
      *result += dotprod * batch_scaling_factor;
    }  // for row
  }    // for batch
}

template <typename T>
void PortableMatrixBatchVectorMultiplyAccumulateImpl(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    T* output) {
  const int16_t output_max = std::numeric_limits<T>::max();
  const int16_t output_min = std::numeric_limits<T>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int row = 0; row < n_output; ++row) {
      int32_t acc = bias[row];
      for (int col = 0; col < n_input; ++col) {
        int8 input_val = input[batch * n_input + col];
        int8 weights_val = input_to_gate_weights[row * n_input + col];
        acc += input_val * weights_val;
      }
      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc += output_zp;
      acc += output[batch * n_output + row];
      if (acc > output_max) {
        acc = output_max;
      }
      if (acc < output_min) {
        acc = output_min;
      }
      output[batch * n_output + row] = static_cast<T>(acc);
    }
  }
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulateImpl(
      input, bias, input_to_gate_weights, multiplier, shift, n_batch, n_input,
      n_output, output_zp, output);
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulateImpl(
      input, bias, input_to_gate_weights, multiplier, shift, n_batch, n_input,
      n_output, output_zp, output);
}

void PortableApplyLayerNorm(const int16_t* input,
                            const int16_t* layer_norm_weights,
                            const int32_t* bias, int32_t layer_norm_scale_a,
                            int32_t layer_norm_scale_b, int32_t variance_limit,
                            int n_batch, int n_input, int16_t* output) {
  // The square of std::pow(2, 10), which is the extra factor that makes sure
  // normalized values has enough resolution.
  static const int kTwoToPower20 = 1 << 20;
  for (int i = 0; i < n_batch; ++i) {
    int64_t sum = 0;
    int64_t sum_sq = 0;
    for (int j = 0; j < n_input; ++j) {
      const int32_t index = i * n_input + j;
      int32_t val = static_cast<int32_t>(input[index]);
      sum += val;
      sum_sq += val * val;
    }
    int32_t mean =
        static_cast<int32_t>(static_cast<int64_t>(sum) * 1024 / n_input);
    // TODO(jianlijianli): Avoids overflow but only works for POT n_input.
    int32 temp = kTwoToPower20 / n_input;
    int64_t variance =
        sum_sq * temp - static_cast<int64_t>(mean) * static_cast<int64_t>(mean);
    int32_t variance2 = static_cast<int32>(variance / kTwoToPower20);
    if (variance2 < 1) {
      variance2 = variance_limit;
    }
    int32_t stddev_inverse_a;
    int stddev_inverse_b;
    GetInvSqrtQuantizedMultiplierExp(variance2, /*reverse_shift*/ -1,
                                     &stddev_inverse_a, &stddev_inverse_b);

    for (int j = 0; j < n_input; ++j) {
      const int32 index = i * n_input + j;
      int32 val = static_cast<int32_t>(input[index]);
      int32 shifted = 1024 * val - mean;
      int32 rescaled = MultiplyByQuantizedMultiplier(shifted, stddev_inverse_a,
                                                     stddev_inverse_b);
      // TODO(jianlijianli): Saturate this.
      int64_t val3 = rescaled * layer_norm_weights[j] + bias[j];
      int32 val4 =
          static_cast<int32>((val3 > 0 ? val3 + 512 : val3 - 512) / 1024);
      int32 val5 = MultiplyByQuantizedMultiplier(val4, layer_norm_scale_a,
                                                 layer_norm_scale_b + 12);
      val5 = std::min(std::max(kInt16Min, val5), kInt16Max);
      output[index] = static_cast<int16_t>(val5);
    }
  }
}

void PortableMatrixScalarMultiplyAccumulate(const int8_t* matrix,
                                            int32_t scalar, int32_t n_row,
                                            int32_t n_col, int32_t* output) {
  for (int i = 0; i < n_row; ++i) {
    int32_t row_sum = 0;
    for (int j = 0; j < n_col; ++j) {
      row_sum += *matrix++;
    }
    output[i] += row_sum * scalar;
  }
}

void PortableApplySigmoid(const int16_t* input, int32_t n_batch,
                          int32_t n_input, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int c = 0; c < n_input; c++) {
      using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;
      using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
      const int index = batch * n_input + c;
      F3 sigmoid_input = F3::FromRaw(input[index]);
      F0 sigmoid_output = gemmlowp::logistic(sigmoid_input);
      output[index] = sigmoid_output.raw();
    }
  }
}

template <int IntegerBits>
void PortableApplyTanhImpl(const int16_t* input, int32_t n_batch,
                           int32_t n_input, int16_t* output) {
  using FX = gemmlowp::FixedPoint<std::int16_t, IntegerBits>;
  using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      FX tanh_input = FX::FromRaw(input[index]);
      F0 tanh_output = gemmlowp::tanh(tanh_input);
      output[index] = tanh_output.raw();
    }
  }
}

void PortableApplyTanh(int32_t integer_bits, const int16_t* input,
                       int32_t n_batch, int32_t n_input, int16_t* output) {
  assert(integer_bits <= 6);
#define DISPATCH_TANH(i)                                       \
  case i:                                                      \
    PortableApplyTanhImpl<i>(input, n_batch, n_input, output); \
    break;
  switch (integer_bits) {
    DISPATCH_TANH(0);
    DISPATCH_TANH(1);
    DISPATCH_TANH(2);
    DISPATCH_TANH(3);
    DISPATCH_TANH(4);
    DISPATCH_TANH(5);
    DISPATCH_TANH(6);
    default:
      return;
  }
#undef DISPATCH_TANH
}

void PortableCwiseMul(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int shift, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const int16_t a = input_1[index];
      const int16_t b = input_2[index];
      const int32_t value = static_cast<int32_t>(a) * static_cast<int32_t>(b);
      output[index] =
          static_cast<int16_t>(gemmlowp::RoundingDivideByPOT(value, shift));
    }
  }
}

void PortableCwiseMul(const int16_t* input_1, const int16_t* input_2,
                      int32_t multiplier, int32_t shift, int32_t n_batch,
                      int32_t n_input, int32_t output_zp, int8_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const int16_t a = input_1[index];
      const int16_t b = input_2[index];
      int32_t value = static_cast<int32_t>(a) * static_cast<int32_t>(b);
      value = MultiplyByQuantizedMultiplier(value, multiplier, shift);
      value -= output_zp;
      value = std::min(std::max(-128, value), 127);

      output[index] = static_cast<int8>(value);
    }
  }
}

void PortableCwiseAdd(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      int32_t sum = input_1[index] + input_2[index];
      const int32 sum_clamped = std::min(kInt16Max, std::max(kInt16Min, sum));
      output[index] = static_cast<int16_t>(sum_clamped);
    }
  }
}

void PortableCwiseClipping(int16_t* input, const int16_t clipping_value,
                           int32_t n_batch, int32_t n_input) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      if (input[index] > clipping_value) {
        input[index] = clipping_value;
      }
      if (input[index] < -clipping_value) {
        input[index] = -clipping_value;
      }
    }
  }
}

void PortableCwiseClipping(int8_t* input, const int8_t clipping_value,
                           int32_t n_batch, int32_t n_input) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      if (input[index] > clipping_value) {
        input[index] = clipping_value;
      }
      if (input[index] < -clipping_value) {
        input[index] = -clipping_value;
      }
    }
  }
}

float PortableVectorVectorDotProduct(const float* vector1, const float* vector2,
                                     int v_size) {
  float result = 0.0;
  for (int v = 0; v < v_size; v++) {
    result += *vector1++ * *vector2++;
  }
  return result;
}

namespace {
inline int32_t VectorVectorDotProduct(const int16_t* vector1,
                                      const int16_t* vector2, int v_size) {
  int32_t result = 0;
  for (int v = 0; v < v_size; v++) {
    result += *vector1++ * *vector2++;
  }
  return result;
}
}  // namespace

void PortableBatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                              const int16_t* vector2,
                                              int v_size, int n_batch,
                                              int32_t* result,
                                              int result_stride) {
  for (int b = 0; b < n_batch; b++) {
    *result = VectorVectorDotProduct(vector1, vector2, v_size);
    vector1 += v_size;
    vector2 += v_size;
    result += result_stride;
  }
}

void PortableVectorBatchVectorCwiseProductAccumulate(
    const int16_t* vector, int v_size, const int16_t* batch_vector, int n_batch,
    int32_t multiplier, int shift, int16_t* result) {
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < v_size; v++) {
      int32_t prod = vector[v] * *batch_vector++;
      prod = MultiplyByQuantizedMultiplier(prod, multiplier, shift);
      int32_t output = prod + *result;
      output = std::max(std::min(32767, output), -32768);
      *result++ = output;
    }
  }
}

void PortableVectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                                  float* batch_vector) {
  for (int b = 0; b < n_batch; b++) {
    for (int i = 0; i < v_size; ++i) {
      batch_vector[i] += vector[i];
    }
    batch_vector += v_size;
  }
}

void PortableSub1Vector(const float* vector, int v_size, float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = 1.0f - *vector++;
  }
}

void PortableSub1Vector(const int16_t* vector, int v_size, int16_t* result) {
  static const int16_t kOne = 32767;
  for (int v = 0; v < v_size; v++) {
    *result++ = kOne - *vector++;
  }
}

void PortableVectorScalarMultiply(const int8_t* vector, const int v_size,
                                  const float scale, float* result) {
  for (int v = 0; v < v_size; ++v) {
    *result++ = scale * *vector++;
  }
}

void PortableClipVector(const float* vector, int v_size, float abs_limit,
                        float* result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(std::min(abs_limit, vector[v]), -abs_limit);
  }
}

void PortableReductionSumVector(const float* input_vector, float* output_vector,
                                int output_size, int reduction_size) {
  const float* input_vector_ptr = input_vector;
  for (int o = 0; o < output_size; o++) {
    for (int r = 0; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }
}

void PortableReductionSumVector(const int32_t* input_vector,
                                int32_t* output_vector, int output_size,
                                int reduction_size) {
  const int32_t* input_vector_ptr = input_vector;
  for (int o = 0; o < output_size; o++) {
    for (int r = 0; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }
}

void PortableMeanStddevNormalization(const float* input_vector,
                                     float* output_vector, int v_size,
                                     int n_batch) {
  for (int batch = 0; batch < n_batch; ++batch) {
    float sum = 0.0f;
    for (int i = 0; i < v_size; ++i) {
      sum += input_vector[i];
    }
    const float mean = sum / v_size;
    float sum_diff_sq = 0.0f;
    for (int i = 0; i < v_size; ++i) {
      const float diff = input_vector[i] - mean;
      sum_diff_sq += diff * diff;
    }
    const float variance = sum_diff_sq / v_size;
    constexpr float kNormalizationConstant = 1e-8f;
    const float stddev_inv =
        1.0f / std::sqrt(variance + kNormalizationConstant);
    for (int i = 0; i < v_size; ++i) {
      output_vector[i] = (input_vector[i] - mean) * stddev_inv;
    }
    input_vector += v_size;
    output_vector += v_size;
  }
}

}  // namespace tensor_utils
}  // namespace tflite
