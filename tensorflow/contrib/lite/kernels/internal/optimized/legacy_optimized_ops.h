/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_

#include <stdint.h>
#include <sys/types.h>

#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/legacy_reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

// Unoptimized reference ops:
using reference_ops::ArgMax;
using reference_ops::Relu1;
using reference_ops::Relu6;
using reference_ops::SpaceToBatchND;

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const RuntimeShape& input_shape,
                     float* output_data, const RuntimeShape& output_shape) {
  static_assert(Ac == FusedActivationFunctionType::kNone, "");
  tflite::L2NormalizationParams op_params;
  // No params need to be set for float, but reserved in signature for future
  // activations.

  L2Normalization(op_params, input_shape, input_data, output_shape,
                  output_data);
}

inline void L2Normalization(const uint8* input_data,
                            const RuntimeShape& input_shape,
                            int32 input_zero_point, uint8* output_data,
                            const RuntimeShape& output_shape) {
  tflite::L2NormalizationParams op_params;
  op_params.input_zero_point = input_zero_point;

  L2Normalization(op_params, input_shape, input_data, output_shape,
                  output_data);
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  L2Normalization<Ac>(input_data, DimsToShape(input_dims), output_data,
                      DimsToShape(output_dims));
}

inline void L2Normalization(const uint8* input_data, const Dims<4>& input_dims,
                            int32 input_zero_point, uint8* output_data,
                            const Dims<4>& output_dims) {
  L2Normalization(input_data, DimsToShape(input_dims), input_zero_point,
                  output_data, DimsToShape(output_dims));
}

inline void Relu(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  Relu(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Add(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
inline void Add(int left_shift, const uint8* input1_data,
                const Dims<4>& input1_dims, int32 input1_offset,
                int32 input1_multiplier, int input1_shift,
                const uint8* input2_data, const Dims<4>& input2_dims,
                int32 input2_offset, int32 input2_multiplier, int input2_shift,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }

  tflite::ArithmeticParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Add(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Add/int32");
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);

  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = std::numeric_limits<int32>::min();
  op_params.quantized_activation_max = std::numeric_limits<int32>::max();
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  BroadcastAdd4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAdd(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }

  tflite::ArithmeticParams op_params;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  BroadcastAdd4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAddFivefold(
    int y0, int y1, int y2, int y3, int y4, int left_shift,
    const uint8* input1_data, const Dims<4>& input1_dims, int32 input1_offset,
    int32 input1_multiplier, int input1_shift, const uint8* input2_data,
    const Dims<4>& input2_dims, int32 input2_offset, int32 input2_multiplier,
    int input2_shift, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  tflite::ArithmeticParams op_params;
  op_params.broadcast_category =
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;
  op_params.left_shift = left_shift;
  op_params.input1_offset = input1_offset;
  op_params.input1_multiplier = input1_multiplier;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_offset = input2_offset;
  op_params.input2_multiplier = input2_multiplier;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.broadcast_shape[4] = y0;
  op_params.broadcast_shape[3] = y1;
  op_params.broadcast_shape[2] = y2;
  op_params.broadcast_shape[1] = y3;
  op_params.broadcast_shape[0] = y4;
  BroadcastAddFivefold(op_params, DimsToShape(input1_dims), input1_data,
                       DimsToShape(input2_dims), input2_data,
                       DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  BroadcastAdd(input1_data, input1_dims, input2_data, input2_dims,
               output_activation_min, output_activation_max, output_data,
               output_dims);
}

template <FusedActivationFunctionType Ac>
inline void Add(const int16* input1_data, const Dims<4>& input1_dims,
                int input1_shift, const int16* input2_data,
                const Dims<4>& input2_dims, int input2_shift,
                int16 output_activation_min, int16 output_activation_max,
                int16* output_data, const Dims<4>& output_dims) {
  constexpr int kReverseShift = -1;
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, -32768);
    TFLITE_DCHECK_EQ(output_activation_max, 32767);
  }

  tflite::ArithmeticParams op_params;
  op_params.input1_shift = kReverseShift * input1_shift;
  op_params.input2_shift = kReverseShift * input2_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Add(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void Sub(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(FusedActivationFunctionType::kNone,
                      &output_activation_min, &output_activation_max);
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void Sub(const T* input1_data, const Dims<4>& input1_dims, const T* input2_data,
         const Dims<4>& input2_dims, T* output_data,
         const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(FusedActivationFunctionType::kNone,
                      &output_activation_min, &output_activation_max);
  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  Sub(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);
  op_params.input1_offset = input1_offset;
  op_params.input2_offset = input2_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = kReverseShift * output_shift;

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  BroadcastMul(input1_data, input1_dims, input1_offset, input2_data,
               input2_dims, input2_offset, output_offset, output_multiplier,
               output_shift, output_activation_min, output_activation_max,
               output_data, output_dims);
}

inline void AveragePool(const float* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int kwidth, int kheight,
                        float output_activation_min,
                        float output_activation_max, float* output_data,
                        const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = kheight;
  params.filter_width = kwidth;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  AveragePool(params, DimsToShape(input_dims), input_data,
              DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const float* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int kwidth, int kheight, float* output_data,
                 const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  AveragePool(input_data, input_dims, stride_width, stride_height, pad_width,
              pad_height, kwidth, kheight, output_activation_min,
              output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const float* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, float* output_data,
                 const Dims<4>& output_dims) {
  AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
                  filter_width, filter_height, output_data, output_dims);
}

inline void AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int filter_width, int filter_height,
                        int32 output_activation_min,
                        int32 output_activation_max, uint8* output_data,
                        const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  AveragePool(params, DimsToShape(input_dims), input_data,
              DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int filter_width, int filter_height,
                 int32 output_activation_min, int32 output_activation_max,
                 uint8* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  AveragePool(input_data, input_dims, stride_width, stride_height, pad_width,
              pad_height, filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const uint8* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims) {
  AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
                  filter_width, filter_height, output_activation_min,
                  output_activation_max, output_data, output_dims);
}

inline void MaxPool(const float* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int kwidth, int kheight,
                    float output_activation_min, float output_activation_max,
                    float* output_data, const Dims<4>& output_dims) {
  tflite::PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = kheight;
  params.filter_width = kwidth;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  MaxPool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
          output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int kwidth, int kheight, float* output_data,
             const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, kwidth, kheight, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             float* output_data, const Dims<4>& output_dims) {
  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_data, output_dims);
}

inline void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int filter_width, int filter_height,
                    int32 output_activation_min, int32 output_activation_max,
                    uint8* output_data, const Dims<4>& output_dims) {
  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.quantized_activation_min = output_activation_min;
  params.quantized_activation_max = output_activation_max;
  MaxPool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
          output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int filter_width, int filter_height, int32 output_activation_min,
             int32 output_activation_max, uint8* output_data,
             const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, filter_width, filter_height, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             int32 output_activation_min, int32 output_activation_max,
             uint8* output_data, const Dims<4>& output_dims) {
  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

inline void L2Pool(const float* input_data, const Dims<4>& input_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int filter_width, int filter_height,
                   float output_activation_min, float output_activation_max,
                   float* output_data, const Dims<4>& output_dims) {
  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.padding_values.height = pad_height;
  params.padding_values.width = pad_width;
  params.float_activation_min = output_activation_min;
  params.float_activation_max = output_activation_max;
  L2Pool(params, DimsToShape(input_dims), input_data, DimsToShape(output_dims),
         output_data);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims,
            int stride_width, int stride_height, int pad_width, int pad_height,
            int filter_width, int filter_height, float* output_data,
            const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  L2Pool(input_data, input_dims, stride_width, stride_height, pad_width,
         pad_height, filter_width, filter_height, output_activation_min,
         output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int filter_width, int filter_height,
            float* output_data, const Dims<4>& output_dims) {
  L2Pool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
             filter_width, filter_height, output_data, output_dims);
}

inline void Softmax(const float* input_data, const Dims<4>& input_dims,
                    float beta, float* output_data,
                    const Dims<4>& output_dims) {
  Softmax(input_data, DimsToShape(input_dims), beta, output_data,
          DimsToShape(output_dims));
}

inline void Softmax(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const Dims<4>& output_dims) {
  Softmax(input_data, DimsToShape(input_dims), input_beta_multiplier,
          input_beta_left_shift, diff_min, output_data,
          DimsToShape(output_dims));
}

inline void LogSoftmax(const float* input_data, const Dims<4>& input_dims,
                       float* output_data, const Dims<4>& output_dims) {
  LogSoftmax(input_data, DimsToShape(input_dims), output_data,
             DimsToShape(output_dims));
}

inline void LogSoftmax(const uint8* input_data, const Dims<4>& input_dims,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const Dims<4>& output_dims) {
  LogSoftmax(input_data, DimsToShape(input_dims), input_multiplier,
             input_left_shift, reverse_scaling_divisor,
             reverse_scaling_right_shift, diff_min, output_data,
             DimsToShape(output_dims));
}

inline void Logistic(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  Logistic(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
           output_data);
}

inline void Logistic(const uint8* input_data, const Dims<4>& input_dims,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const Dims<4>& output_dims) {
  Logistic(input_data, DimsToShape(input_dims), input_zero_point,
           input_range_radius, input_multiplier, input_left_shift, output_data,
           DimsToShape(output_dims));
}

inline void Logistic(const int16* input_data, const Dims<4>& input_dims,
                     int16* output_data, const Dims<4>& output_dims) {
  Logistic(input_data, DimsToShape(input_dims), output_data,
           DimsToShape(output_dims));
}

inline void Tanh(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  Tanh(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

inline void Tanh(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const Dims<4>& output_dims) {
  Tanh(input_data, DimsToShape(input_dims), input_zero_point,
       input_range_radius, input_multiplier, input_left_shift, output_data,
       DimsToShape(output_dims));
}

inline void Tanh(const int16* input_data, const Dims<4>& input_dims,
                 int input_left_shift, int16* output_data,
                 const Dims<4>& output_dims) {
  Tanh(input_data, DimsToShape(input_dims), input_left_shift, output_data,
       DimsToShape(output_dims));
}

template <typename T>
inline void DepthToSpace(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  tflite::DepthToSpaceParams op_params;
  op_params.block_size = block_size;

  DepthToSpace(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

template <typename T>
inline void SpaceToDepth(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  tflite::SpaceToDepthParams op_params;
  op_params.block_size = block_size;

  SpaceToDepth(op_params, DimsToShape(input_dims), input_data,
               DimsToShape(output_dims), output_data);
}

inline void Mul(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float output_activation_min, float output_activation_max,
                float* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Mul(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  Mul(input1_data, input1_dims, input2_data, input2_dims, output_activation_min,
      output_activation_max, output_data, output_dims);
}

inline void Mul(const int32* input1_data, const Dims<4>& input1_dims,
                const int32* input2_data, const Dims<4>& input2_dims,
                int32 output_activation_min, int32 output_activation_max,
                int32* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <FusedActivationFunctionType Ac>
void Mul(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);
  tflite::ArithmeticParams op_params;
  // No parameters needed.

  MulNoActivation(op_params, DimsToShape(input1_dims), input1_data,
                  DimsToShape(input2_dims), input2_data,
                  DimsToShape(output_dims), output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int16* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  // No parameters needed.

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int32 output_offset, int32 output_activation_min,
                int32 output_activation_max, uint8* output_data,
                const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  op_params.output_offset = output_offset;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  Mul(op_params, DimsToShape(input1_dims), input1_data,
      DimsToShape(input2_dims), input2_data, DimsToShape(output_dims),
      output_data);
}

template <typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

// For compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const float* input1_data, const Dims<4>& input1_dims,
                         const float* input2_data, const Dims<4>& input2_dims,
                         float* output_data, const Dims<4>& output_dims) {
  tflite::ArithmeticParams op_params;
  float float_activation_min;
  float float_activation_max;
  GetActivationMinMax(Ac, &float_activation_min, &float_activation_max);
  SetActivationParams(float_activation_min, float_activation_max, &op_params);

  BroadcastMul4DSlow(op_params, DimsToShape(input1_dims), input1_data,
                     DimsToShape(input2_dims), input2_data,
                     DimsToShape(output_dims), output_data);
}

// Legacy Dims<4>.
inline void LocalResponseNormalization(const float* input_data,
                                       const Dims<4>& input_dims, int range,
                                       float bias, float alpha, float beta,
                                       float* output_data,
                                       const Dims<4>& output_dims) {
  tflite::LocalResponseNormalizationParams op_params;
  op_params.range = range;
  op_params.bias = bias;
  op_params.alpha = alpha;
  op_params.beta = beta;

  LocalResponseNormalization(op_params, DimsToShape(input_dims), input_data,
                             DimsToShape(output_dims), output_data);
}

// Legacy Dims<4> version.
template <typename SrcT, typename DstT>
void Cast(const SrcT* input_data, const Dims<4>& input_dims, DstT* output_data,
          const Dims<4>& output_dims) {
  Cast(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
       output_data);
}

// Legacy Dims<4> version.
inline void Floor(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  Floor(DimsToShape(input_dims), input_data, DimsToShape(output_dims),
        output_data);
}

// Legacy Dims<4>
inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = align_corners;
  ResizeBilinear(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(output_size_dims), output_size_data,
                 DimsToShape(output_dims), output_data);
}

// Legacy Dims<4>
inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
  tflite::ResizeBilinearParams op_params;
  op_params.align_corners = align_corners;
  ResizeBilinear(op_params, DimsToShape(input_dims), input_data,
                 DimsToShape(output_size_dims), output_size_data,
                 DimsToShape(output_dims), output_data);
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear(input_data, input_dims, output_size_data, output_size_dims,
                 output_data, output_dims, /*align_corners=*/false);
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const uint8* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, uint8* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear(input_data, input_dims, output_size_data, output_size_dims,
                 output_data, output_dims, /*align_corners=*/false);
}

// Legacy Dims<4>.
template <typename T>
inline void BatchToSpaceND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* crops_data, const Dims<4>& crops_dims,
                           T* output_data, const Dims<4>& output_dims) {
  BatchToSpaceND(DimsToShape(input_dims), input_data,
                 DimsToShape(block_shape_dims), block_shape_data,
                 DimsToShape(crops_dims), crops_data, DimsToShape(output_dims),
                 output_data);
}

// Legacy signature, function covered both Pad and PadV2.
template <typename T>
inline void PadV2(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& left_paddings,
                  const std::vector<int>& right_paddings, T* output_data,
                  const Dims<4>& output_dims, const T pad_value) {
  TFLITE_DCHECK_EQ(left_paddings.size(), 4);
  TFLITE_DCHECK_EQ(right_paddings.size(), 4);
  tflite::PadParams op_params;
  op_params.left_padding_count = 4;
  op_params.right_padding_count = 4;
  for (int i = 0; i < 4; ++i) {
    op_params.left_padding[i] = left_paddings[3 - i];
    op_params.right_padding[i] = right_paddings[3 - i];
  }
  const T pad_value_copy = pad_value;

  Pad(op_params, DimsToShape(input_dims), input_data, &pad_value_copy,
      DimsToShape(output_dims), output_data);
}

// Old Pad that calls legacy PadV2.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims, const int32_t pad_value) {
  const T converted_pad_value = static_cast<T>(pad_value);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, converted_pad_value);
}

// Old Pad that only padded with 0.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims) {
  const T pad_value = static_cast<T>(0);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, pad_value);
}

template <typename T>
inline void Slice(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& begin, const std::vector<int>& size,
                  T* output_data, const Dims<4>& output_dims) {
  tflite::SliceParams op_params;
  op_params.begin_count = 4;
  op_params.size_count = 4;
  for (int i = 0; i < 4; ++i) {
    op_params.begin[i] = begin[3 - i];
    op_params.size[i] = size[3 - i];
  }

  Slice(op_params, DimsToShape(input_dims), input_data,
        DimsToShape(output_dims), output_data);
}

template <typename T>
void TensorFlowMinimum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  Minimum(DimsToShape(input1_dims), input1_data, input2_data,
          DimsToShape(output_dims), output_data);
}

template <typename T>
void TensorFlowMaximum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  Maximum(DimsToShape(input1_dims), input1_data, input2_data,
          DimsToShape(output_dims), output_data);
}

}  // namespace optimized_ops
}  // namespace tflite
#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_LEGACY_OPTIMIZED_OPS_H_
