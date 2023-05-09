/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_common.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_reference_impl.h"

namespace tflite {
namespace optimized_4bit {

void ReferencePackInner(const int8_t* src, uint8_t* box, int src_rows,
                        int src_cols, int outer_row, int outer_col,
                        int outer_rows, int outer_cols, int inner_rows,
                        int inner_cols) {
  // Create a kernel-specific layout for a packed unit.
  const int width = inner_rows;
  const int depth = inner_cols;
  const int real_depth = depth / 2;
  const int real_src_cols = src_cols / 2;
  // Determine row and column of source tensor.
  const int row = outer_row * inner_rows;
  const int col = outer_col * inner_cols;
  // The source width  (rows) and depth (columns).
  int src_width = std::min(width, src_rows - row);
  int src_depth = std::min(depth, src_cols - col);
  int real_col = col / 2;
  const int8_t* src_data = src + row * real_src_cols + real_col;
  int real_src_depth = src_depth / 2;
  // Src is [rows / src_rows, cols / src_depth, src_rows, src_cols]
  // Reshape and pad to [outer_rows, outer_cols, width, depth]
  // Interleave values [u1,u2,...,u_depth] to
  // [u1,u_{depth/2+1},u2,u_{depth/2+2},.., u_{depth/2},u_depth]
  // So that after shifting, we get [u1,u2...u_{depth/2}] and
  // [u_{depth/2} + 1, ... u_{depth}].
  for (int m = 0; m < src_width; ++m) {
    int i = 0;
    int k = 0;
    int half_depth = depth / 2;
    int half_half_depth = half_depth / 2;
    for (; i < (real_src_depth & (~(half_depth - 1))); i += half_depth) {
      for (int j = 0; j < half_half_depth; ++j) {
        const int8_t v1 = (int8_t)src_data[i + j];
        int8_t uv1 = upper(v1);
        int8_t lv1 = lower(v1);
        const int8_t v2 = (int8_t)src_data[i + j + half_half_depth];
        int8_t uv2 = upper(v2);
        int8_t lv2 = lower(v2);
        box[k] = merge(lv1, lv2);
        box[k + 1] = merge(uv1, uv2);
        k += 2;
      }
    }
    // Handle remaining 16 values
    for (; i < (real_src_depth & (~(half_half_depth - 1)));
         i += half_half_depth) {
      for (int j = 0; j < 8; ++j) {
        const int8_t v1 = (int8_t)src_data[i + j];
        int8_t uv1 = upper(v1);
        int8_t lv1 = lower(v1);
        box[k] = merge(lv1, 0);
        box[k + 1] = merge(uv1, 0);
        k += 2;
      }
    }
    // Any remaining values are just interleaved with 0.
    for (; i < real_src_depth; i++) {
      const int8_t v1 = (int8_t)src_data[i];
      int8_t uv1 = upper(v1);
      int8_t lv1 = lower(v1);
      box[k] = merge(lv1, 0);
      box[k + 1] = merge(uv1, 0);
      k += 2;
    }
    box += real_depth;
    src_data += real_src_cols;
  }
}

void ReferencePrepack(uint8_t** dest, const int8_t* tensor, int layout_rows,
                      int layout_cols, int src_rows, int src_cols, int width,
                      int depth) {
  size_t size = layout_rows * layout_cols / 2;
  *dest = reinterpret_cast<uint8_t*>(malloc(size));
  memset(*dest, static_cast<uint8_t>(0x77), sizeof(uint8_t) * size);
  int outer_cols = layout_cols / depth;
  int outer_rows = layout_rows / width;
  int inner_cols = depth;
  int inner_rows = width;
  for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
    for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
      // Each outer row x outer col contains width x depth, copied
      // from tensor at the cluster_index.
      const int cluster_index = outer_row * outer_cols + outer_col;
      const int real_depth = inner_cols / 2;
      uint8_t* box = *dest + cluster_index * real_depth * inner_rows;
      ReferencePackInner(tensor, box, src_rows, src_cols, outer_row, outer_col,
                         outer_rows, outer_cols, inner_rows, inner_cols);
    }
  }
}

void ReferenceBatchQuantizeFloats4Bit(const float* float_data_ptr, int n_batch,
                                      int n_data, int8_t* quantized_data_ptr,
                                      float* scaling_factors, int width,
                                      int depth, int32_t* input_offsets) {
  const int rows = n_batch;
  const int cols = n_data;
  // depth is always cols
  const int layout_rows = (rows + (width - 1)) & ~(width - 1);
  const int layout_cols = (cols + (depth - 1)) & ~(depth - 1);
  const int size = layout_rows * layout_cols;
  int8_t* data = quantized_data_ptr;
  memset(data, 0, sizeof(int8_t) * size);
  memset(input_offsets, 0, sizeof(int32_t) * layout_rows);
  const float* tensor_data = float_data_ptr;
  // basically, we need to make a new 4D matrix
  // [rows / width, cols / depth, width, depth] in depth-first
  const int outer_cols = layout_cols / depth;
  const int outer_rows = layout_rows / width;
  float* scaling_factors_ptr = scaling_factors;
  for (int outer_row = 0; outer_row < outer_rows; outer_row++) {
    std::vector<float> scale(width);
    const int row = width * outer_row;
    scaling_factors_ptr = scaling_factors + row;
    for (int w = 0; w < width; ++w) {
      if ((row + w) >= rows) {
        continue;
      }
      const float* start = tensor_data + (row + w) * cols;
      float scale_denom = 0;
      for (int c = 0; c < cols; ++c) {
        scale_denom = std::max(scale_denom, std::abs(*(start++)));
      }
      scale[w] = 127.0 / scale_denom;
      scaling_factors_ptr[w] = scale_denom / 127.0;
    }
    for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
      const int col = depth * outer_col;
      const int src_width = std::min(width, rows - row);
      const int src_depth = std::min(depth, cols - col);
      const int cluster_index = outer_row * outer_cols + outer_col;
      int8_t* box = data + cluster_index * depth * width;
      for (int w = 0; w < src_width; ++w) {
        const float* float_data = tensor_data + (row + w) * cols + col;
        for (int d = 0; d < src_depth; ++d) {
          int8_t q = static_cast<int8_t>(TfLiteRound(float_data[d] * scale[w]));
          box[w * depth + d] = q;
          input_offsets[row + w] += q;
        }
      }
    }
  }
  for (int r = 0; r < layout_rows; ++r) {
    // Multiply the input by zero-point so that we don't have to calculate
    // later.
    input_offsets[r] = input_offsets[r] * zero_point_4bit;
  }
}

void ReferenceAssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                          const float* batch_scales,
                                          const float* filter_scales,
                                          const float* bias_ptr,
                                          float* output_ptr, int output_depth,
                                          int batch_size) {
  if (bias_ptr) {
    for (int b = 0; b < batch_size; ++b) {
      const float val = *input_offsets++ * *batch_scales++;
      const float* filter_scales_ptr = filter_scales;
      const float* bias_ptr_tmp = bias_ptr;
      for (int i = 0; i < output_depth; i++) {
        *output_ptr++ = (val * *filter_scales_ptr++) + *bias_ptr_tmp++;
      }
    }
    return;
  }
  for (int b = 0; b < batch_size; ++b) {
    const float val = *input_offsets++ * *batch_scales++;
    const float* filter_scales_ptr = filter_scales;
    for (int i = 0; i < output_depth; i++) {
      *output_ptr++ = (val * *filter_scales_ptr++);
    }
  }
}

/* Unpack the accumulated scratch buffer by transposing and multiplying
 * by input and filter scales.
 * Before, dst contains integer accumulated values with layout:
 *   [rhs_layout_rows // rhs_width, lhs_layout_rows // lhs_width,
 *       rhs_width, lhs_width]
 * Transpose and dequantize to [batch_size, num_units].
 */
template <int Depth, int Width>
void ReferenceUnpack(float* output_ptr, const int32_t* dst, int batch_size,
                     int num_units, const float* scaling_factors,
                     const float* filter_scales, int dst_layout_rows,
                     int dst_layout_cols) {
  // Width == 1 is when batch size == 1, the most frequent case.
  // No need to iterate over outer rows.
  if (Width == 1) {
    const int outer_rows = dst_layout_rows / Width;
    const int outer_cols = dst_layout_cols / Depth;
    const int32_t* dst_ptr = dst;
    int unit = 0;
    for (int outer_col = 0; outer_col < outer_cols;
         ++outer_col, unit += Depth) {
      float* tmp_output_ptr = output_ptr + unit;
      int len = num_units - unit < Depth ? num_units - unit : Depth;
      const float* scaling_factors_ptr = scaling_factors;
      for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
        const float scale = *scaling_factors_ptr;
        const float* filter_scales_ptr = filter_scales + unit;
        for (int i = 0; i < len; ++i) {
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
        }
        dst_ptr += (Depth - len);
        scaling_factors_ptr += Width;
        tmp_output_ptr += (num_units - len);
      }
    }
    return;
  }
  const int outer_rows = dst_layout_rows / Width;
  const int outer_cols = dst_layout_cols / Depth;
  for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
    const int unit = outer_col * Depth;
    const int remaining_units = std::min(num_units - unit, Depth);
    const int depth_offset = Depth - remaining_units;
    const int width_offset = num_units - remaining_units;
    int outer_row = 0;
    for (; outer_row < outer_rows; ++outer_row) {
      const int batch = outer_row * Width;
      const int remaining_width = std::min(batch_size - batch, Width);
      const int cluster_index = outer_col * outer_rows + outer_row;
      const int32_t* dst_ptr = dst + cluster_index * Depth * Width;
      float* tmp_output_ptr = output_ptr + batch * num_units + unit;
      const float* scale = scaling_factors + batch;
      int w = remaining_width;
      for (; w > 0; --w, scale++) {
        int d = remaining_units;
        const float* filter_scales_ptr = filter_scales + unit;
        for (; d > 0; --d) {
          *tmp_output_ptr++ += *dst_ptr++ * (*scale) * (*filter_scales_ptr++);
        }
        dst_ptr += depth_offset;
        tmp_output_ptr += width_offset;
      }
    }
  }
}

template <int RowsLeft, int RowsRight, int Cols>
void ReferenceRunKernel(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
                        int lhs_layout_rows, int lhs_layout_cols,
                        int rhs_layout_rows, int rhs_layout_cols,
                        int dst_layout_rows, int dst_layout_cols) {
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  int32_t* elementPtr = dst;
  const int outer_rows = (clamped_end_row + RowsLeft - 1) / RowsLeft;
  const int outer_cols = (clamped_end_col + RowsRight - 1) / RowsRight;
  const int depth = std::min(lhs_layout_cols / Cols, rhs_layout_cols / Cols);
  for (int i = start_row; i < outer_rows; ++i) {
    int left_index = i * RowsLeft * lhs_layout_cols / 2;
    const uint8_t* lhs_val_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_val = lhs_val_data;
      int right_index = j * RowsRight * rhs_layout_cols;
      const int8_t* rhs_val = rhs + right_index;
      int32_t accum[RowsLeft * RowsRight];
      memset(accum, 0, sizeof(int32_t) * RowsLeft * RowsRight);
      for (int k = 0; k < depth; ++k) {
        uint8_t lhs_[RowsLeft][Cols];
        for (int m = 0; m < RowsLeft; ++m) {
          for (int n = 0; n < Cols / 2; ++n) {
            uint8_t val = *(lhs_val++);
            lhs_[m][n] = (val >> 4 & 15);
            lhs_[m][n + (Cols / 2)] = (val & 15);
          }
        }
        int8_t rhs_[RowsRight][Cols];
        for (int m = 0; m < RowsRight; ++m) {
          for (int n = 0; n < Cols; ++n) {
            rhs_[m][n] = *(rhs_val++);
          }
        }
        for (int r = 0; r < RowsRight; ++r) {
          for (int l = 0; l < RowsLeft; ++l) {
            for (int i = 0; i < Cols; ++i) {
              accum[r * RowsLeft + l] += lhs_[l][i] * rhs_[r][i];
            }
          }
        }
      }  // end depth
      for (int r = 0; r < RowsRight; ++r) {
        for (int l = 0; l < RowsLeft; ++l) {
          int32_t q = accum[r * RowsLeft + l];
          *(elementPtr++) = q;
        }
      }
    }
  }
}

template void ReferenceUnpack<4, 1>(float* output_ptr, const int32_t* dst,
                                    int batch_size, int num_units,
                                    const float* scaling_factors,
                                    const float* filter_scales,
                                    int dst_layout_rows, int dst_layout_cols);

template void ReferenceUnpack<4, 2>(float* output_ptr, const int32_t* dst,
                                    int batch_size, int num_units,
                                    const float* scaling_factors,
                                    const float* filter_scales,
                                    int dst_layout_rows, int dst_layout_cols);

template void ReferenceUnpack<4, 4>(float* output_ptr, const int32_t* dst,
                                    int batch_size, int num_units,
                                    const float* scaling_factors,
                                    const float* filter_scales,
                                    int dst_layout_rows, int dst_layout_cols);

template void ReferenceRunKernel<4, 1, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols);

template void ReferenceRunKernel<4, 2, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols);

template void ReferenceRunKernel<4, 4, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols);

}  // namespace optimized_4bit
}  // namespace tflite
