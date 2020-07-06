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
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h"

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
  const int32_t kScale = 127;
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
    quantized_values[i] = static_cast<int8_t>(
        std::min(kScale, std::max(-kScale, quantized_value)));
  }
}

void PortableAsymmetricQuantizeFloats(const float* values, const int size,
                                      int8_t* quantized_values,
                                      float* scaling_factor, int32_t* offset) {
  const int32_t kMinScale = -128;
  const int32_t kMaxScale = 127;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const auto minmax = std::minmax_element(values, values + size);
  const double rmin = std::fmin(0, *minmax.first);
  const double rmax = std::fmax(0, *minmax.second);
  if (rmin == rmax) {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    *offset = 0;
    return;
  } else {
    double scale = (rmax - rmin) / (qmax_double - qmin_double);
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
    if (zero_point_double <= qmin_double) {
      nudged_zero_point = kMinScale;
    } else if (zero_point_double >= qmax_double) {
      nudged_zero_point = kMaxScale;
    } else {
      nudged_zero_point = static_cast<int8>(round(zero_point_double));
    }
    *scaling_factor = scale;
    *offset = nudged_zero_point;
  }
  const float scaling_factor_inv = 1.0 / *scaling_factor;
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
                                                 int n_batch, float* result) {
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
      ++result_in_batch;
    }
  }
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result) {
  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Get the address of the first row.
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
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
      ++result;
    }  // for row
  }    // for batch
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context) {
  if (input_offset == nullptr) {
    PortableMatrixBatchVectorMultiplyAccumulate(
        matrix, m_rows, m_cols, vectors, scaling_factors, n_batch, result);
    return;
  }
  if (!compute_row_sums || *compute_row_sums) {
    memset(row_sums, 0, sizeof(int32_t) * m_rows);
    PortableReductionSumVector(matrix, row_sums, m_rows, m_cols);
    if (compute_row_sums) {
      *compute_row_sums = false;
    }
  }

  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    const float batch_offset = input_offset[batch];
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
      int32_t dotprod = 0;
      float scale = batch_scaling_factor;
      if (per_channel_scale) {
        scale *= per_channel_scale[row];
      }
#if defined(__GNUC__)
      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);
#endif
      for (int col = 0; col < m_cols; ++col, ++row_ptr) {
        dotprod += (*row_ptr) * vectors[col];
      }  // for col
      dotprod -= row_sums[row] * batch_offset;
      *result += dotprod * scale;
      ++result;
    }  // for row
  }    // for batch
}

void PortableSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result) {
  const int kBlockSize = 4;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);
  for (int batch = 0; batch < n_batch; batch++) {
    const float* matrix_ptr = matrix;
    for (int row = 0; row < m_rows; row++) {
      float dot_prod = 0.0f;
      const float* vector_in_batch = vector + batch * m_cols;
      for (int i = segments[row]; i < segments[row + 1]; i++) {
        const int block_start_index = indices[i] * kBlockSize;
        const float* vector_block_in_batch_ptr =
            vector_in_batch + block_start_index;
        for (int c = 0; c < kBlockSize; c++) {
          dot_prod += *matrix_ptr++ * *vector_block_in_batch_ptr++;
        }
      }
      result[batch * m_rows + row] += dot_prod;
    }
  }
}

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result) {
  const int kBlockSize = 16;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);
  for (int batch = 0; batch < n_batch; batch++) {
    const float* matrix_ptr = matrix;
    const uint8_t* ledger_ptr = ledger;
    for (int row = 0; row < m_rows; row++) {
      float dot_prod = 0.0f;
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        const float* vector_in_batch = vector + batch * m_cols;
        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int block_start_index = *ledger_ptr++ * kBlockSize;
          const float* vector_block_in_batch_ptr =
              vector_in_batch + block_start_index;
          for (int c = 0; c < kBlockSize; c++) {
            dot_prod += *matrix_ptr++ * *vector_block_in_batch_ptr++;
          }
        }
      }
      result[batch * m_rows + row] += dot_prod;
    }
  }
}

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result) {
  static const int kBlockSize = 16;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);
  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    const uint8_t* ledger_ptr = ledger;
    // Get the address of the first row.
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row) {
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
      result[batch * m_rows + row] += dotprod * batch_scaling_factor;
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

void PortableMatrixBatchVectorMultiply(const int8_t* input,
                                       int32_t input_zeropoint,
                                       const int8_t* input_to_gate_weights,
                                       int32_t input_to_gate_effective_scale_a,
                                       int32_t input_to_gate_effective_scale_b,
                                       int32_t n_batch, int32_t n_input,
                                       int32_t n_cell, int8_t* gate_output,
                                       int8_t gate_output_zp) {
  const int32_t int8_max = std::numeric_limits<int8>::max();
  const int32_t int8_min = std::numeric_limits<int8>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int row = 0; row < n_cell; ++row) {
      int32_t acc = 0;
      for (int col = 0; col < n_input; ++col) {
        int32_t input_val = input[batch * n_input + col];
        int8_t weights_val = input_to_gate_weights[row * n_input + col];
        acc += (input_val - input_zeropoint) * weights_val;
      }
      acc = MultiplyByQuantizedMultiplier(acc, input_to_gate_effective_scale_a,
                                          input_to_gate_effective_scale_b);
      acc += gate_output_zp;
      if (acc > int8_max) {
        acc = int8_max;
      }
      if (acc < int8_min) {
        acc = int8_min;
      }
      gate_output[batch * n_cell + row] = static_cast<int8_t>(acc);
    }
  }
}

void PortableMatrixBatchVectorMultiply(
    const int16_t* hidden, const int8_t* hidden_to_output_weights,
    int32_t proj_effective_scale_a, int32_t proj_effective_scale_b,
    const int32_t* gate_bias, int32_t n_batch, int32_t n_hidden,
    int32_t n_output, int32_t output_zp, int8_t* proj_output) {
  const int16_t int8_max = std::numeric_limits<int8>::max();
  const int16_t int8_min = std::numeric_limits<int8>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int row = 0; row < n_output; ++row) {
      int64_t acc = gate_bias[row];
      for (int col = 0; col < n_hidden; ++col) {
        int16_t input_val = hidden[batch * n_hidden + col];
        int8_t weights_val = hidden_to_output_weights[row * n_hidden + col];
        int64_t curr = acc;
        acc += input_val * weights_val;
        if (input_val * weights_val > 0 && acc < curr) {
          acc = std::numeric_limits<int32>::max();
        }
        if (input_val * weights_val < 0 && acc > curr) {
          acc = std::numeric_limits<int32>::min();
        }
      }
      acc = MultiplyByQuantizedMultiplier(acc, proj_effective_scale_a,
                                          proj_effective_scale_b);
      acc += output_zp;
      if (acc > int8_max) {
        acc = int8_max;
      }
      if (acc < int8_min) {
        acc = int8_min;
      }
      proj_output[batch * n_output + row] = acc;
    }
  }
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

void PortableApplyLayerNormFloat(const int16_t* input,
                                 const int16_t* layer_norm_weights,
                                 int32_t layer_norm_scale_a,
                                 int32_t layer_norm_scale_b,
                                 const int32_t* bias, int n_batch, int n_input,
                                 int16_t* output) {
  const int32_t int16_max = std::numeric_limits<int16>::max();
  const int32_t int16_min = std::numeric_limits<int16>::min();
  // This is to surpress a lint warning.
  const double two = 2.0;
  const float layer_norm_scale =
      layer_norm_scale_a *
      std::pow(two, static_cast<double>(layer_norm_scale_b - 31));
  const float bias_scale = std::pow(two, -10) * layer_norm_scale;

  for (int batch = 0; batch < n_batch; ++batch) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const float value = static_cast<float>(input[index]);
      sum += value;
      sum_sq += value * value;
    }
    const float mean = sum / n_input;
    float stddev_inv = 0.0f;
    const float variance = sum_sq / n_input - mean * mean;
    if (variance == 0) {
      stddev_inv = 1.0f / sqrt(1e-8);
    } else {
      stddev_inv = 1.0f / sqrt(variance);
    }
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const float normalized_value =
          (static_cast<float>(input[index]) - mean) * stddev_inv;
      const float weighted_normalized_value =
          normalized_value * layer_norm_weights[i] * layer_norm_scale +
          bias[i] * bias_scale;
      const int32_t quant_output = static_cast<int32>(
          std::round(weighted_normalized_value * std::pow(2, 12)));
      output[index] = std::min(int16_max, std::max(int16_min, quant_output));
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

void PortableApplySigmoidFloat(const int16_t* input, int32_t n_batch,
                               int32_t n_input, int16_t* output) {
  const int32_t int16_max = std::numeric_limits<int16>::max();
  const int32_t int16_min = std::numeric_limits<int16>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const float float_input = input[index] * std::pow(2, -12);
      const float float_output = 1.0f / (1.0f + std::exp(-float_input));
      const int32_t quant_output =
          static_cast<int32>(float_output * std::pow(2, 15));
      const int32_t quant_output_clamped =
          std::min(int16_max, std::max(int16_min, quant_output));
      output[index] = static_cast<int16>(quant_output_clamped);
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

void PortableApplyTanhFloat(const int16_t* input, int32_t n_batch,
                            int32_t n_input, int32_t integer_bits,
                            int16_t* output) {
  const int32_t int16_max = std::numeric_limits<int16>::max();
  const int32_t int16_min = std::numeric_limits<int16>::min();
  const double two = 2.0;
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const float float_input =
          input[index] * std::pow(two, static_cast<double>(integer_bits));
      const float float_output = std::tanh(float_input);
      const int32_t quant_output =
          static_cast<int32>(float_output * std::pow(2, 15));
      const int32_t quant_output_clamped =
          std::min(int16_max, std::max(int16_min, quant_output));
      output[index] = static_cast<int16>(quant_output_clamped);
    }
  }
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
      value = std::min(std::max(static_cast<int32_t>(-128), value),
                       static_cast<int32_t>(127));

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
                                              int32_t* result) {
  for (int b = 0; b < n_batch; b++) {
    result[b] = VectorVectorDotProduct(vector1, vector2, v_size);
    vector1 += v_size;
    vector2 += v_size;
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
      output = std::max(std::min(static_cast<int32_t>(32767), output),
                        static_cast<int32_t>(-32768));
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

void PortableReductionSumVector(const int8_t* input_vector,
                                int32_t* output_vector, int output_size,
                                int reduction_size) {
  const int8_t* input_vector_ptr = input_vector;
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

void PortableTwoGateSaturatingAdd(const int8_t* input, int8_t input_zp,
                                  const int8_t* recurrent, int8_t recurrent_zp,
                                  int32_t input_effective_scale_a,
                                  int32_t input_effective_scale_b,
                                  int32_t recurrent_effective_scale_a,
                                  int32_t recurrent_effective_scale_b,
                                  int32_t n_batch, int32_t n_cell,
                                  int16_t* output) {
  const int32_t int16_max = std::numeric_limits<int16>::max();
  const int32_t int16_min = std::numeric_limits<int16>::min();
  for (int i = 0; i < n_batch * n_cell; ++i) {
    int32_t x = static_cast<int32>(input[i]) - static_cast<int32>(input_zp);
    int32_t h =
        static_cast<int32>(recurrent[i]) - static_cast<int32>(recurrent_zp);
    int32_t x_scaled = MultiplyByQuantizedMultiplier(x, input_effective_scale_a,
                                                     input_effective_scale_b);
    int32_t h_scaled = MultiplyByQuantizedMultiplier(
        h, recurrent_effective_scale_a, recurrent_effective_scale_b);
    int32_t y = h_scaled + x_scaled;
    if (y > int16_max) {
      y = int16_max;
    }
    if (y < int16_min) {
      y = int16_min;
    }
    output[i] = static_cast<int16_t>(y);
  }
}

}  // namespace tensor_utils
}  // namespace tflite
