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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "fixedpoint/fixedpoint.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const int8* input_data, const RuntimeShape& output_shape,
                    int8* output_data) {
  ruy::profiler::ScopeLabel label("MaxPool/8bit");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int8 acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          memset(acc, params.quantized_activation_min,
                 tranche_depth * sizeof(acc[0]));
          const int8* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const int8* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const int8* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                int8x16_t acc_reg = vld1q_s8(acc + channel);
                int8x16_t input_reg = vld1q_s8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg = vmaxq_s8(acc_reg, input_reg);
                vst1q_s8(acc + channel, acc_reg);
              }

              for (; channel <= tranche_depth - 8; channel += 8) {
                int8x8_t acc_reg = vld1_s8(acc + channel);
                int8x8_t input_reg = vld1_s8(input_channel_ptr);
                input_channel_ptr += 8;
                acc_reg = vmax_s8(acc_reg, input_reg);
                vst1_s8(acc + channel, acc_reg);
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] = std::max(acc[channel], *input_channel_ptr++);
              }
              input_row_ptr += depth;
            }
          }
          int8* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                  out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
          for (; channel <= tranche_depth - 16; channel += 16) {
            int8x16_t a = vld1q_s8(acc + channel);
            a = vminq_s8(a, vdupq_n_s8(params.quantized_activation_max));
            a = vmaxq_s8(a, vdupq_n_s8(params.quantized_activation_min));
            vst1q_s8(output_ptr + channel, a);
          }
          for (; channel <= tranche_depth - 8; channel += 8) {
            int8x8_t a = vld1_s8(acc + channel);
            a = vmin_s8(a, vdup_n_s8(params.quantized_activation_max));
            a = vmax_s8(a, vdup_n_s8(params.quantized_activation_min));
            vst1_s8(output_ptr + channel, a);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            int8 a = acc[channel];
            a = std::max<int8>(a, params.quantized_activation_min);
            a = std::min<int8>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<int8>(a);
          }
        }
      }
    }
  }
}

inline void AveragePool16(const PoolParams& params,
                          const RuntimeShape& input_shape,
                          const int8* input_data,
                          const RuntimeShape& output_shape, int8* output_data) {
  ruy::profiler::ScopeLabel label("AveragePool/8bitWith16bitAccumulator");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int16 acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          const int filter_count =
              (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const int8* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const int8* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const int8* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                int16x8_t acc_reg[2];
                for (int i = 0; i < 2; i++) {
                  acc_reg[i] = vld1q_s16(acc + channel + 8 * i);
                }
                int8x16_t input_reg = vld1q_s8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg[0] = vaddw_s8(acc_reg[0], vget_low_s8(input_reg));
                acc_reg[1] = vaddw_s8(acc_reg[1], vget_high_s8(input_reg));
                for (int i = 0; i < 2; i++) {
                  vst1q_s16(acc + channel + 8 * i, acc_reg[i]);
                }
              }
              for (; channel <= tranche_depth - 8; channel += 8) {
                int16x8_t acc_reg = vld1q_s16(acc + channel);
                int8x8_t input_reg = vld1_s8(input_channel_ptr);
                input_channel_ptr += 8;
                acc_reg = vaddw_s8(acc_reg, input_reg);
                vst1q_s16(acc + channel, acc_reg);
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] += *input_channel_ptr++;
              }
              input_row_ptr += depth;
            }
          }
          int8* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                  out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
#define AVGPOOL_DIVIDING_BY(FILTER_COUNT)                                    \
  if (filter_count == FILTER_COUNT) {                                        \
    for (; channel <= tranche_depth - 8; channel += 8) {                     \
      int16 buf[8];                                                          \
      for (int i = 0; i < 8; i++) {                                          \
        buf[i] = acc[channel + i] > 0                                        \
                     ? (acc[channel + i] + FILTER_COUNT / 2) / FILTER_COUNT  \
                     : (acc[channel + i] - FILTER_COUNT / 2) / FILTER_COUNT; \
      }                                                                      \
      int8x8_t buf8 = vqmovn_s16(vld1q_s16(buf));                            \
      buf8 = vmin_s8(buf8, vdup_n_s8(params.quantized_activation_max));      \
      buf8 = vmax_s8(buf8, vdup_n_s8(params.quantized_activation_min));      \
      vst1_s8(output_ptr + channel, buf8);                                   \
    }                                                                        \
  }
          AVGPOOL_DIVIDING_BY(9)
          AVGPOOL_DIVIDING_BY(15)
#undef AVGPOOL_DIVIDING_BY
          for (; channel <= tranche_depth - 8; channel += 8) {
            int16 buf[8];
            for (int i = 0; i < 8; i++) {
              buf[i] =
                  acc[channel + i] > 0
                      ? (acc[channel + i] + filter_count / 2) / filter_count
                      : (acc[channel + i] - filter_count / 2) / filter_count;
            }
            int8x8_t buf8 = vqmovn_s16(vld1q_s16(buf));
            buf8 = vmin_s8(buf8, vdup_n_s8(params.quantized_activation_max));
            buf8 = vmax_s8(buf8, vdup_n_s8(params.quantized_activation_min));
            vst1_s8(output_ptr + channel, buf8);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            int16 a = acc[channel] > 0
                          ? (acc[channel] + filter_count / 2) / filter_count
                          : (acc[channel] - filter_count / 2) / filter_count;
            a = std::max<int16>(a, params.quantized_activation_min);
            a = std::min<int16>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<int8>(a);
          }
        }
      }
    }
  }
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape, const int8* input_data,
                        const RuntimeShape& output_shape, int8* output_data) {
  if (params.filter_height * params.filter_width > 16 * 16) {
    reference_integer_ops::AveragePool(params, input_shape, input_data,
                                       output_shape, output_data);
  } else {
    AveragePool16(params, input_shape, input_data, output_shape, output_data);
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_
