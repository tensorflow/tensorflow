/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_PAD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_PAD_H_

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

template <typename T>
void TypedMemset(void* ptr, T value, size_t num) {
  // Optimization for common cases where memset() will suffice.
  if (value == 0 || std::is_same<T, uint8_t>::value) {
    memset(ptr, value, num * sizeof(T));
  } else {
    // Default implementation for cases where memset() will not preserve the
    // bytes, e.g., typically when sizeof(T) > sizeof(uint8_t).
    char* pos = static_cast<char*>(ptr);
    for (size_t i = 0; i < num; ++i) {
      memcpy(pos, &value, sizeof(T));
      pos = pos + sizeof(T);
    }
  }
}

// This makes heavy use of Offset, along with conditional branches. There may be
// opportunities for improvement.
//
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
      RuntimeShape::ExtendedShape(4, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, 4);
  TFLITE_DCHECK_LE(op_params.right_padding_count, 4);

  // Pad kernels are limited to max 4 dimensions. Copy inputs so we can pad them
  // to 4 dims (yes, we are "padding the padding").
  std::vector<int> left_padding_copy(4, 0);
  const int left_padding_extend = 4 - op_params.left_padding_count;
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[left_padding_extend + i] = op_params.left_padding[i];
  }
  std::vector<int> right_padding_copy(4, 0);
  const int right_padding_extend = 4 - op_params.right_padding_count;
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[right_padding_extend + i] = op_params.right_padding[i];
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

  const int input_depth = ext_input_shape.Dims(3);
  const T pad_value = *pad_value_ptr;

  if (left_b_padding != 0) {
    TypedMemset<T>(
        output_data, pad_value,
        left_b_padding * output_height * output_width * output_depth);
  }
  for (int out_b = left_b_padding; out_b < output_batch - right_b_padding;
       ++out_b) {
    if (left_h_padding != 0) {
      TypedMemset<T>(output_data + Offset(ext_output_shape, out_b, 0, 0, 0),
                     pad_value, left_h_padding * output_width * output_depth);
    }
    for (int out_h = left_h_padding; out_h < output_height - right_h_padding;
         ++out_h) {
      if (left_w_padding != 0) {
        TypedMemset<T>(
            output_data + Offset(ext_output_shape, out_b, out_h, 0, 0),
            pad_value, left_w_padding * output_depth);
      }
      for (int out_w = left_w_padding; out_w < output_width - right_w_padding;
           ++out_w) {
        if (left_d_padding != 0) {
          TypedMemset<T>(
              output_data + Offset(ext_output_shape, out_b, out_h, out_w, 0),
              pad_value, left_d_padding);
        }

        T* out = output_data +
                 Offset(ext_output_shape, out_b, out_h, out_w, left_d_padding);
        const T* in = input_data +
                      Offset(ext_input_shape, out_b - left_b_padding,
                             out_h - left_h_padding, out_w - left_w_padding, 0);
        memcpy(out, in, input_depth * sizeof(T));

        if (right_d_padding != 0) {
          TypedMemset<T>(
              output_data + Offset(ext_output_shape, out_b, out_h, out_w,
                                   output_depth - right_d_padding),
              pad_value, right_d_padding);
        }
      }
      if (right_w_padding != 0) {
        TypedMemset<T>(output_data + Offset(ext_output_shape, out_b, out_h,
                                            output_width - right_w_padding, 0),
                       pad_value, right_w_padding * output_depth);
      }
    }
    if (right_h_padding != 0) {
      TypedMemset<T>(
          output_data + Offset(ext_output_shape, out_b,
                               output_height - right_h_padding, 0, 0),
          pad_value, right_h_padding * output_width * output_depth);
    }
  }
  if (right_b_padding != 0) {
    TypedMemset<T>(
        output_data +
            Offset(ext_output_shape, output_batch - right_b_padding, 0, 0, 0),
        pad_value,
        right_b_padding * output_height * output_width * output_depth);
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

// TODO(b/117643175): Optimize. (This is an introductory copy of standard Pad.)
//
// This pad requires that (a) left and right paddings are in the 4D patterns
// {0, h_pad, w_pad, 0}, and (b) memset can be used: *pad_value_ptr == 0 and/or
// T is uint8.
//
// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32 is considered a
// specialization distinct from P=int32.
template <typename T, typename P>
inline void PadImageStyleMemset(const tflite::PadParams& op_params,
                                const RuntimeShape& input_shape,
                                const T* input_data, const P* pad_value_ptr,
                                const RuntimeShape& output_shape,
                                T* output_data) {
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(4, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, 4);
  TFLITE_DCHECK_LE(op_params.right_padding_count, 4);

  // Pad kernels are limited to max 4 dimensions. Copy inputs so we can pad them
  // to 4 dims (yes, we are "padding the padding").
  std::vector<int> left_padding_copy(4, 0);
  const int left_padding_extend = 4 - op_params.left_padding_count;
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[left_padding_extend + i] = op_params.left_padding[i];
  }
  std::vector<int> right_padding_copy(4, 0);
  const int right_padding_extend = 4 - op_params.right_padding_count;
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[right_padding_extend + i] = op_params.right_padding[i];
  }
  // The following padding restrictions are contractual requirements, and
  // embody what it means for a padding op to be "image-style".
  TFLITE_DCHECK_EQ(left_padding_copy[0], 0);
  TFLITE_DCHECK_EQ(left_padding_copy[3], 0);
  TFLITE_DCHECK_EQ(right_padding_copy[0], 0);
  TFLITE_DCHECK_EQ(right_padding_copy[3], 0);

  const int batch = MatchingDim(ext_input_shape, 0, ext_output_shape, 0);
  const int output_height = ext_output_shape.Dims(1);
  const int output_width = ext_output_shape.Dims(2);
  const int input_height = ext_input_shape.Dims(1);
  const int input_width = ext_input_shape.Dims(2);
  const int depth = MatchingDim(ext_input_shape, 3, ext_output_shape, 3);

  const int left_h_padding = left_padding_copy[1];
  const int left_w_padding = left_padding_copy[2];
  const int right_h_padding = right_padding_copy[1];
  const int right_w_padding = right_padding_copy[2];

  TFLITE_DCHECK_EQ(output_height,
                   input_height + left_h_padding + right_h_padding);
  TFLITE_DCHECK_EQ(output_width,
                   input_width + left_w_padding + right_w_padding);

  const T pad_value = *pad_value_ptr;
  const int top_block_size = left_h_padding * output_width * depth;
  const size_t num_top_block_bytes = top_block_size * sizeof(T);
  const int bottom_block_size = right_h_padding * output_width * depth;
  const size_t num_bottom_block_bytes = bottom_block_size * sizeof(T);
  const int left_blocks_size = left_w_padding * depth;
  const size_t num_left_block_bytes = left_blocks_size * sizeof(T);
  const int right_blocks_size = right_w_padding * depth;
  const size_t num_right_block_bytes = right_blocks_size * sizeof(T);
  const int inner_line_size = input_width * depth;
  const size_t num_inner_line_bytes = inner_line_size * sizeof(T);

  if (input_height == 0) {
    memset(output_data, pad_value,
           num_top_block_bytes + num_bottom_block_bytes);
  } else {
    for (int i = 0; i < batch; ++i) {
      // For each image in the batch, apply the top padding, then iterate
      // through rows, then apply the bottom padding.
      //
      // By unwinding one iteration, we can combine the first left-margin
      // padding with the top padding, and the last right-margin padding with
      // the bottom padding.
      memset(output_data, pad_value,
             num_top_block_bytes + num_left_block_bytes);
      output_data += top_block_size + left_blocks_size;
      memcpy(output_data, input_data, num_inner_line_bytes);
      input_data += inner_line_size;
      output_data += inner_line_size;
      // One iteration unwound.
      // Unwinding this loop affords the opportunity to reorder the loop work
      // and hence combine memset() calls.
      //
      // Before unwinding:
      // for (int j = 0; j < input_height; ++j) {
      //   // Pad on left, copy central data, pad on right.
      //   memset(output_data, pad_value, num_left_block_bytes);
      //   output_data += left_blocks_size;
      //   memcpy(output_data, input_data, num_inner_line_bytes);
      //   input_data += inner_line_size;
      //   output_data += inner_line_size;
      //   memset(output_data, pad_value, num_right_block_bytes);
      //   output_data += right_blocks_size;
      // }
      for (int j = 1; j < input_height; ++j) {
        memset(output_data, pad_value,
               num_right_block_bytes + num_left_block_bytes);
        output_data += right_blocks_size + left_blocks_size;
        memcpy(output_data, input_data, num_inner_line_bytes);
        input_data += inner_line_size;
        output_data += inner_line_size;
      }
      memset(output_data, pad_value,
             num_right_block_bytes + num_bottom_block_bytes);
      output_data += right_blocks_size + bottom_block_size;
    }
  }
}

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
  PadImageStyleMemset(op_params, input_shape, input_data, pad_value_ptr,
                      output_shape, output_data);
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const float* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          float* output_data) {
  const float converted_pad_value = static_cast<float>(*pad_value_ptr);
  if (converted_pad_value == 0.0f) {
    PadImageStyleMemset(op_params, input_shape, input_data, pad_value_ptr,
                        output_shape, output_data);
  } else {
    PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
            output_data);
  }
}
}  // namespace optimized_ops
}  // namespace tflite

#endif
