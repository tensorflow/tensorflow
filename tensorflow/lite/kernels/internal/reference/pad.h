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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PAD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PAD_H_

#include <vector>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// TFLite Pad supports activation tensors with up to 4 dimensions.
constexpr int PadKernelMaxDimensionCount() { return 4; }

// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32 is considered a
// specialization distinct from P=int32.
template <typename T, typename P>
inline void PadImpl(const tflite::PadParams& op_params,
                    const RuntimeShape& input_shape, const T* input_data,
                    const P* pad_value_ptr, const RuntimeShape& output_shape,
                    T* output_data) {
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(PadKernelMaxDimensionCount(), input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(PadKernelMaxDimensionCount(), output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, PadKernelMaxDimensionCount());
  TFLITE_DCHECK_LE(op_params.right_padding_count, PadKernelMaxDimensionCount());

  // Runtime calls are currently fixed at 4 dimensions. Copy inputs so we can
  // pad them to 4 dims (yes, we are "padding the padding").
  int left_padding_copy[PadKernelMaxDimensionCount()];
  for (int i = 0; i < PadKernelMaxDimensionCount(); i++) {
    left_padding_copy[i] = 0;
  }
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[i + PadKernelMaxDimensionCount() -
                      op_params.left_padding_count] = op_params.left_padding[i];
  }
  int right_padding_copy[PadKernelMaxDimensionCount()];
  for (int i = 0; i < PadKernelMaxDimensionCount(); i++) {
    right_padding_copy[i] = 0;
  }
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[i + PadKernelMaxDimensionCount() -
                       op_params.right_padding_count] =
        op_params.right_padding[i];
  }

  const int output_batch = ext_output_shape.Dims(0);
  const int output_height = ext_output_shape.Dims(1);
  const int output_width = ext_output_shape.Dims(2);
  const int output_depth = ext_output_shape.Dims(3);

  const int left_b_padding = left_padding_copy[0];
  const int left_h_padding = left_padding_copy[1];
  const int left_w_padding = left_padding_copy[2];
  const int left_d_padding = left_padding_copy[3];

  const int right_b_padding = right_padding_copy[0];
  const int right_h_padding = right_padding_copy[1];
  const int right_w_padding = right_padding_copy[2];
  const int right_d_padding = right_padding_copy[3];

  const T pad_value = *pad_value_ptr;

  const T* in_ptr = input_data;
  T* out_ptr = output_data;
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
          if (out_b < left_b_padding ||
              out_b >= output_batch - right_b_padding ||
              out_h < left_h_padding ||
              out_h >= output_height - right_h_padding ||
              out_w < left_w_padding ||
              out_w >= output_width - right_w_padding ||
              out_d < left_d_padding ||
              out_d >= output_depth - right_d_padding) {
            *out_ptr++ = pad_value;
          } else {
            *out_ptr++ = *in_ptr++;
          }
        }
      }
    }
  }
}

template <typename T, typename P>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const P* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

// The second (pad-value) input can be int32 when, say, the first is uint8.
template <typename T>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const int32* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  const T converted_pad_value = static_cast<T>(*pad_value_ptr);
  PadImpl(op_params, input_shape, input_data, &converted_pad_value,
          output_shape, output_data);
}

// This version avoids conflicting template matching.
template <>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const int32* input_data,
                const int32* pad_value_ptr, const RuntimeShape& output_shape,
                int32* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

// One could make all PadImageStyle calls simply delegate the work to the
// ordinary Pad.  However, it is better that the reference code asserts false in
// similar cases.
template <typename T, typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const P* pad_value_ptr,
                          const RuntimeShape& output_shape, T* output_data) {
  TFLITE_ASSERT_FALSE;
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const uint8* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          uint8* output_data) {
  Pad(op_params, input_shape, input_data, pad_value_ptr, output_shape,
      output_data);
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const int8_t* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          int8_t* output_data) {
  Pad(op_params, input_shape, input_data, pad_value_ptr, output_shape,
      output_data);
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const int16_t* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          int16_t* output_data) {
  Pad(op_params, input_shape, input_data, pad_value_ptr, output_shape,
      output_data);
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const float* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          float* output_data) {
  Pad(op_params, input_shape, input_data, pad_value_ptr, output_shape,
      output_data);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PAD_H_
