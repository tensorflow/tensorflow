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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_

#include <stdint.h>
#include <sys/types.h>

#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

inline RuntimeShape DimsToShape(const tflite::Dims<4>& dims) {
  return RuntimeShape(
      {dims.sizes[3], dims.sizes[2], dims.sizes[1], dims.sizes[0]});
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
  Relu(input_data, DimsToShape(input_dims), output_data,
       DimsToShape(output_dims));
}

inline void Relu1(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  Relu1(input_data, DimsToShape(input_dims), output_data,
        DimsToShape(output_dims));
}

inline void Relu6(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  Relu6(input_data, DimsToShape(input_dims), output_data,
        DimsToShape(output_dims));
}

inline void AveragePool(const float* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int kwidth, int kheight,
                        float output_activation_min,
                        float output_activation_max, float* output_data,
                        const Dims<4>& output_dims) {
  AveragePool(input_data, DimsToShape(input_dims), stride_width, stride_height,
              pad_width, pad_height, kwidth, kheight, output_activation_min,
              output_activation_max, output_data, DimsToShape(output_dims));
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
  AveragePool(input_data, DimsToShape(input_dims), stride_width, stride_height,
              pad_width, pad_height, filter_width, filter_height,
              output_activation_min, output_activation_max, output_data,
              DimsToShape(output_dims));
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
  MaxPool(input_data, DimsToShape(input_dims), stride_width, stride_height,
          pad_width, pad_height, kwidth, kheight, output_activation_min,
          output_activation_max, output_data, DimsToShape(output_dims));
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
  MaxPool(input_data, DimsToShape(input_dims), stride_width, stride_height,
          pad_width, pad_height, filter_width, filter_height,
          output_activation_min, output_activation_max, output_data,
          DimsToShape(output_dims));
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
  L2Pool(input_data, DimsToShape(input_dims), stride_width, stride_height,
         pad_width, pad_height, filter_width, filter_height,
         output_activation_min, output_activation_max, output_data,
         DimsToShape(output_dims));
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
  Logistic(input_data, DimsToShape(input_dims), output_data,
           DimsToShape(output_dims));
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
  Tanh(input_data, DimsToShape(input_dims), output_data,
       DimsToShape(output_dims));
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

}  // namespace reference_ops
}  // namespace tflite
#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_LEGACY_REFERENCE_OPS_H_
