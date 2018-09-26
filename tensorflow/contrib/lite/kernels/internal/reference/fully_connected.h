/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_FULLY_CONNECTED_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_FULLY_CONNECTED_H_

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/round.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

const int kReverseShift = -1;

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data) {
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      float total = 0.f;
      for (int d = 0; d < accum_depth; ++d) {
        total += input_data[b * accum_depth + d] *
                 weights_data[out_c * accum_depth + d];
      }
      float bias_value = 0.0f;
      if (bias_data) {
        bias_value = bias_data[out_c];
      }
      output_data[out_c + output_depth * b] = ActivationFunctionWithMinMax(
          total + bias_value, output_activation_min, output_activation_max);
    }
  }
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// Legacy.
inline void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                           const float* weights_data,
                           const Dims<4>& weights_dims, const float* bias_data,
                           const Dims<4>& bias_dims,
                           float output_activation_min,
                           float output_activation_max, float* output_data,
                           const Dims<4>& output_dims) {
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(weights_dims), weights_data,
                 DimsToShape(bias_dims), bias_data, DimsToShape(output_dims),
                 output_data);
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                    const float* weights_data, const Dims<4>& weights_dims,
                    const float* bias_data, const Dims<4>& bias_dims,
                    float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  FullyConnected(input_data, input_dims, weights_data, weights_dims, bias_data,
                 bias_dims, output_activation_min, output_activation_max,
                 output_data, output_dims);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, void* gemm_context) {
  (void)gemm_context;  // only used in optimized code.
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32 acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32 input_val = input_data[b * accum_depth + d];
        int32 filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<uint8>(acc);
    }
  }
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// Legacy.
inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, uint8* output_data,
                           const Dims<4>& output_dims, void* gemm_context) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                 bias_data, DimsToShape(output_dims), output_data,
                 gemm_context);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    int16* output_data, void* gemm_context) {
  (void)gemm_context;  // only used in optimized code.
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(output_offset, 0);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum = bias_data[out_c];
      // Accumulation loop.
      for (int d = 0; d < accum_depth; ++d) {
        int16 input_val = input_data[b * accum_depth + d] + input_offset;
        int16 filter_val = filter_data[out_c * accum_depth + d] + filter_offset;
        accum += filter_val * input_val;
      }
      // Down-scale the final int32 accumulator to the scale used by our
      // (16-bit, typically 3 integer bits) fixed-point format. The quantized
      // multiplier and shift here have been pre-computed offline
      // (e.g. by toco).
      accum =
          MultiplyByQuantizedMultiplier(accum, output_multiplier, output_shift);
      // Saturate, cast to int16, and store to output array.
      accum = std::max(accum, output_activation_min - output_offset);
      accum = std::min(accum, output_activation_max - output_offset);
      accum += output_offset;
      output_data[out_c + output_depth * b] = accum;
    }
  }
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// Legacy.
inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, int16* output_data,
                           const Dims<4>& output_dims, void* gemm_context) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  FullyConnected(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(filter_dims), filter_data, DimsToShape(bias_dims),
                 bias_data, DimsToShape(output_dims), output_data,
                 gemm_context);
}

inline void ShuffledFullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& weights_shape,
    const uint8* shuffled_weights_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    int16* output_data, uint8* shuffled_input_workspace_data,
    void* gemm_context) {
  (void)gemm_context;  // only used in optimized code.
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dim_count - 1);
  TFLITE_DCHECK((accum_depth % 16) == 0);
  TFLITE_DCHECK((output_depth % 4) == 0);

  // Shuffling and xoring of input activations into the workspace buffer
  uint8* shuffled_input_workspace_ptr = shuffled_input_workspace_data;
  if (batches == 1) {
    for (int i = 0; i < accum_depth; i++) {
      shuffled_input_workspace_data[i] = input_data[i] ^ 0x80;
    }
  } else if (batches == 4) {
    for (int c = 0; c < accum_depth; c += 16) {
      for (int b = 0; b < 4; b++) {
        const uint8* src_data_ptr = input_data + b * accum_depth + c;
        for (int j = 0; j < 16; j++) {
          uint8 src_val = *src_data_ptr++;
          // Flip the sign bit, so that the kernel will only need to
          // reinterpret these uint8 values as int8, getting for free the
          // subtraction of the zero_point value 128.
          uint8 dst_val = src_val ^ 0x80;
          *shuffled_input_workspace_ptr++ = dst_val;
        }
      }
    }
  } else {
    TFLITE_DCHECK(false);
    return;
  }

  // Actual computation
  if (batches == 1) {
    int16* output_ptr = output_data;
    // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
    // so that just reinterpreting them as int8 values is equivalent to
    // subtracting 128 from them, thus implementing for free the subtraction of
    // the zero_point value 128.
    const int8* shuffled_weights_ptr =
        reinterpret_cast<const int8*>(shuffled_weights_data);
    // Likewise, we preshuffled and pre-xored the input data above.
    const int8* shuffled_input_data =
        reinterpret_cast<const int8*>(shuffled_input_workspace_data);
    for (int c = 0; c < output_depth; c += 4) {
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum[4] = {0};
      // Accumulation loop.
      for (int d = 0; d < accum_depth; d += 16) {
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 16; j++) {
            int8 input_val = shuffled_input_data[d + j];
            int8 weights_val = *shuffled_weights_ptr++;
            accum[i] += weights_val * input_val;
          }
        }
      }
      for (int i = 0; i < 4; i++) {
        // Add bias value
        int32 acc = accum[i] + bias_data[c + i];
        // Down-scale the final int32 accumulator to the scale used by our
        // (16-bit, typically 3 integer bits) fixed-point format. The quantized
        // multiplier and shift here have been pre-computed offline
        // (e.g. by toco).
        acc =
            MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
        // Saturate, cast to int16, and store to output array.
        acc = std::max(acc, output_activation_min);
        acc = std::min(acc, output_activation_max);
        output_ptr[c + i] = acc;
      }
    }
  } else if (batches == 4) {
    int16* output_ptr = output_data;
    // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
    // so that just reinterpreting them as int8 values is equivalent to
    // subtracting 128 from them, thus implementing for free the subtraction of
    // the zero_point value 128.
    const int8* shuffled_weights_ptr =
        reinterpret_cast<const int8*>(shuffled_weights_data);
    // Likewise, we preshuffled and pre-xored the input data above.
    const int8* shuffled_input_data =
        reinterpret_cast<const int8*>(shuffled_input_workspace_data);
    for (int c = 0; c < output_depth; c += 4) {
      const int8* shuffled_input_ptr = shuffled_input_data;
      // Accumulation loop.
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum[4][4];
      for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 4; b++) {
          accum[i][b] = 0;
        }
      }
      for (int d = 0; d < accum_depth; d += 16) {
        for (int i = 0; i < 4; i++) {
          for (int b = 0; b < 4; b++) {
            for (int j = 0; j < 16; j++) {
              int8 input_val = shuffled_input_ptr[16 * b + j];
              int8 weights_val = shuffled_weights_ptr[16 * i + j];
              accum[i][b] += weights_val * input_val;
            }
          }
        }
        shuffled_input_ptr += 64;
        shuffled_weights_ptr += 64;
      }
      for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 4; b++) {
          // Add bias value
          int32 acc = accum[i][b] + bias_data[c + i];
          // Down-scale the final int32 accumulator to the scale used by our
          // (16-bit, typically 3 integer bits) fixed-point format. The
          // quantized multiplier and shift here have been pre-computed offline
          // (e.g. by toco).
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              output_shift);
          // Saturate, cast to int16, and store to output array.
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_ptr[b * output_depth + c + i] = acc;
        }
      }
    }
  } else {
    TFLITE_DCHECK(false);
    return;
  }
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// Legacy.
inline void ShuffledFullyConnected(
    const uint8* input_data, const Dims<4>& input_dims,
    const uint8* shuffled_weights_data, const Dims<4>& weights_dims,
    const int32* bias_data, const Dims<4>& bias_dims, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    int16* output_data, const Dims<4>& output_dims,
    uint8* shuffled_input_workspace_data, void* gemm_context) {
  tflite::FullyConnectedParams op_params;
  op_params.output_multiplier = output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  ShuffledFullyConnected(op_params, DimsToShape(input_dims), input_data,
                         DimsToShape(weights_dims), shuffled_weights_data,
                         DimsToShape(bias_dims), bias_data,
                         DimsToShape(output_dims), output_data,
                         shuffled_input_workspace_data, gemm_context);
}

// TODO(b/80418076): Move to legacy ops file, update invocations.
// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_offset, const uint8* filter_data,
                    const Dims<4>& filter_dims, int32 filter_offset,
                    const int32* bias_data, const Dims<4>& bias_dims,
                    int32 output_offset, int32 output_multiplier,
                    int output_shift, int32 output_activation_min,
                    int32 output_activation_max, uint8* output_data,
                    const Dims<4>& output_dims, void* gemm_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  FullyConnected(input_data, input_dims, input_offset, filter_data, filter_dims,
                 filter_offset, bias_data, bias_dims, output_offset,
                 output_multiplier, output_shift, output_activation_min,
                 output_activation_max, output_data, output_dims, gemm_context);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_FULLY_CONNECTED_H_
