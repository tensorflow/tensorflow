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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_IM2COL_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_IM2COL_UTILS_H_

#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

template <typename T>
inline void ExtractPatchIntoBufferColumn(const RuntimeShape& input_shape, int w,
                                         int h, int b, int kheight, int kwidth,
                                         int stride_width, int stride_height,
                                         int pad_width, int pad_height,
                                         int in_width, int in_height,
                                         int in_depth, int single_buffer_length,
                                         int buffer_id, const T* in_data,
                                         T* conv_buffer_data, uint8 zero_byte) {
  gemmlowp::ScopedProfilingLabel label("ExtractPatchIntoBufferColumn");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  // This chunk of code reshapes all the inputs corresponding to
  // output (b, h, w) to a column vector in conv_buffer(:, buffer_id).
  const int kwidth_times_indepth = kwidth * in_depth;
  const int inwidth_times_indepth = in_width * in_depth;
  const int ih_ungated_start = h * stride_height - pad_height;
  const int ih_ungated_end = (ih_ungated_start + kheight);
  const int ih_end = std::min(ih_ungated_end, in_height);
  const int iw_ungated_start = w * stride_width - pad_width;
  const int iw_ungated_end = (iw_ungated_start + kwidth);
  const int iw_end = std::min(iw_ungated_end, in_width);
  // If the patch is off the edge of the input image, skip writing those rows
  // and columns from the patch into the output array.
  const int h_offset = std::max(0, -ih_ungated_start);
  const int w_offset = std::max(0, -iw_ungated_start);
  const int ih_start = std::max(0, ih_ungated_start);
  const int iw_start = std::max(0, iw_ungated_start);
  const int single_row_num =
      std::min(kwidth - w_offset, in_width - iw_start) * in_depth;
  const int output_row_offset = (buffer_id * single_buffer_length);
  int out_offset =
      output_row_offset + (h_offset * kwidth + w_offset) * in_depth;
  int in_offset = Offset(input_shape, b, ih_start, iw_start, 0);

  // Express all of the calculations as padding around the input patch.
  const int top_padding = h_offset;
  const int bottom_padding = (ih_ungated_end - ih_end);
  const int left_padding = w_offset;
  const int right_padding = (iw_ungated_end - iw_end);
  assert(single_row_num ==
         ((kwidth - (left_padding + right_padding)) * in_depth));

  // Write out zeroes to the elements representing the top rows of the input
  // patch that are off the edge of the input image.
  if (top_padding > 0) {
    const int top_row_elements = (top_padding * kwidth * in_depth);
    memset(conv_buffer_data + output_row_offset, zero_byte,
           (top_row_elements * sizeof(T)));
  }

  // If the patch is on the interior of the input image horizontally, just copy
  // over the rows sequentially, otherwise add zero padding at the start or end.
  if ((left_padding == 0) && (right_padding == 0)) {
    for (int ih = ih_start; ih < ih_end; ++ih) {
      memcpy(conv_buffer_data + out_offset, in_data + in_offset,
             single_row_num * sizeof(T));
      out_offset += kwidth_times_indepth;
      in_offset += inwidth_times_indepth;
    }
  } else {
    for (int ih = ih_start; ih < ih_end; ++ih) {
      if (left_padding > 0) {
        const int left_start = (out_offset - (left_padding * in_depth));
        memset(conv_buffer_data + left_start, zero_byte,
               (left_padding * in_depth * sizeof(T)));
      }
      memcpy(conv_buffer_data + out_offset, in_data + in_offset,
             single_row_num * sizeof(T));
      if (right_padding > 0) {
        const int right_start = (out_offset + single_row_num);
        memset(conv_buffer_data + right_start, zero_byte,
               (right_padding * in_depth * sizeof(T)));
      }
      out_offset += kwidth_times_indepth;
      in_offset += inwidth_times_indepth;
    }
  }

  // If the bottom of the patch falls off the input image, pad the values
  // representing those input rows with zeroes.
  if (bottom_padding > 0) {
    const int bottom_row_elements = (bottom_padding * kwidth * in_depth);
    const int bottom_start =
        output_row_offset +
        ((top_padding + (ih_end - ih_start)) * kwidth * in_depth);
    memset(conv_buffer_data + bottom_start, zero_byte,
           (bottom_row_elements * sizeof(T)));
  }
}

template <typename T>
void DilatedIm2col(const ConvParams& params, uint8 zero_byte,
                   const RuntimeShape& input_shape, const T* input_data,
                   const RuntimeShape& filter_shape,
                   const RuntimeShape& output_shape, T* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  // For dilated convolution, the input pixels are not contiguous therefore we
  // can't use the same opitimizations as Im2Col(). Though note this code would
  // work fine for the non-dilated case too (though likely a bit slower).
  gemmlowp::ScopedProfilingLabel label("DilatedIm2col");
  TFLITE_DCHECK(dilation_width_factor != 1 || dilation_height_factor != 1);
  TFLITE_DCHECK(im2col_data);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  MatchingDim(output_shape, 3, filter_shape, 0);

  // Construct the MxN sized im2col matrix.
  // The rows M, are sub-ordered B x H x W
  const RuntimeShape row_shape({1, batches, output_height, output_width});
  // The columns, N, are sub-ordered Kh x Kw x Din
  const RuntimeShape col_shape({1, filter_height, filter_width, input_depth});
  // Use dimensions M and N to construct dims for indexing directly into im2col
  const RuntimeShape im2col_shape(
      {1, 1, row_shape.FlatSize(), col_shape.FlatSize()});

  // Loop through the output rows (B x H x W)
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        // Each im2col row is an output pixel. Arrange the input data in this
        // row in an order we can conveniently multiply with the filter data.
        int row_offset = Offset(row_shape, 0, batch, out_y, out_x);
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int in_y_origin = (out_y * stride_height) - pad_height;
        // Loop through all the pixels of the filter (Kh x Kw)
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + dilation_height_factor * filter_y;
          if ((in_y >= 0) && (in_y < input_height)) {
            // Filter row is within the input data.
            // Loop through all the filter pixels in this row.
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              int col_offset = Offset(col_shape, 0, filter_y, filter_x, 0);
              T* dst = im2col_data +
                       Offset(im2col_shape, 0, 0, row_offset, col_offset);
              if ((in_x >= 0) && (in_x < input_width)) {
                // Filter pixel is within the input, copy the input data.
                T const* src =
                    input_data + Offset(input_shape, batch, in_y, in_x, 0);
                memcpy(dst, src, input_depth * sizeof(T));
              } else {
                // Filter pixel is outside the input, zero it out.
                memset(dst, zero_byte, input_depth * sizeof(T));
              }
            }
          } else {
            // Filter row is outside the input, zero out the entire filter row.
            int col_offset = Offset(col_shape, 0, filter_y, 0, 0);
            T* dst = im2col_data +
                     Offset(im2col_shape, 0, 0, row_offset, col_offset);
            memset(dst, zero_byte, filter_width * input_depth * sizeof(T));
          }
        }
      }
    }
  }
}

template <typename T>
void Im2col(const ConvParams& params, int kheight, int kwidth, uint8 zero_byte,
            const RuntimeShape& input_shape, const T* input_data,
            const RuntimeShape& output_shape, T* output_data) {
  gemmlowp::ScopedProfilingLabel label("Im2col");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);

  int buffer_id = 0;
  // Loop over the output nodes.
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        ExtractPatchIntoBufferColumn(
            input_shape, w, h, b, kheight, kwidth, stride_width, stride_height,
            pad_width, pad_height, input_width, input_height, input_depth,
            output_depth, buffer_id, input_data, output_data, zero_byte);
        ++buffer_id;
      }
    }
  }
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_IM2COL_UTILS_H_
