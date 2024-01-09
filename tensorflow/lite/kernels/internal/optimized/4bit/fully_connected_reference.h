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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_REFERENCE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_REFERENCE_H_

#include <cstdint>

#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_reference_impl.h"

namespace tflite {
namespace optimized_4bit {

/* Returns the maximum number of rhs rows supported/compiled.
 *
 * In general 4 is the most we can do without running out of registers on
 * aarch64. For x86/aarch64, 4x32 4bit = 64 bytes can be held in cache line
 * size. This is required to set the packing layout for the rhs.
 *
 * For reference, return 1.
 */
inline int GetMaxSupportedRows() { return 1; }

/* Pack a 4bit inner_rows x inner_cols array from src.
 * This is called as an inner function for Prepack.
 */
inline void PackInner(const int8_t* src, uint8_t* box, int src_rows,
                      int src_cols, int outer_row, int outer_col,
                      int outer_rows, int outer_cols, int inner_rows,
                      int inner_cols) {
  ReferencePackInner(src, box, src_rows, src_cols, outer_row, outer_col,
                     outer_rows, outer_cols, inner_rows, inner_cols);
}

/* Prepack lhs matrix into dest.
 * Transform tensor from (src_rows, src_cols) to
 * (layout_rows / width, layout_cols / depth, width, depth) with possibly
 * padding, and interleaving values along depth / 2 dimensions.
 * dest should be aligned and allocated before prepack.
 */
inline void Prepack(uint8_t* dest, const int8_t* tensor, int layout_rows,
                    int layout_cols, int src_rows, int src_cols, int width,
                    int depth) {
  ReferencePrepack(dest, tensor, layout_rows, layout_cols, src_rows, src_cols,
                   width, depth);
}

/* Quantize input floats to 8bit and calculate sum of each column.
 *  Data in float_data_ptr of shape (n_batch x n_data), is quantized and
 * packed into (n_batch / width, n_data / depth, width, data) into
 * quantized_data_ptr and input_offsets will contain the product of filter
 * zero_point and input.
 */
inline void BatchQuantizeFloats4Bit(const float* float_data_ptr, int n_batch,
                                    int n_data, int8_t* quantized_data_ptr,
                                    float* scaling_factors, int width,
                                    int depth, int32_t* input_offsets) {
  ReferenceBatchQuantizeFloats4Bit(float_data_ptr, n_batch, n_data,
                                   quantized_data_ptr, scaling_factors, width,
                                   depth, input_offsets);
}

/* Write bias + input offset * filter_scale to output_ptr.
 * output_ptr of size (batch_size, output_depth) will have
 * output_ptr[output_depth * b + o] =
 *     bias_ptr[o] + input_offsets[b] * batch_scales[b] * filter_scale[o]
 */
inline void AssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                        const float* batch_scales,
                                        float* filter_scales,
                                        const float* bias_ptr,
                                        float* output_ptr, int output_depth,
                                        int batch_size) {
  ReferenceAssignBiasAndComputeOffsets(input_offsets, batch_scales,
                                       filter_scales, bias_ptr, output_ptr,
                                       output_depth, batch_size);
}

/* Add accumulated integer sums in dst to float output.
 * output_ptr of size (batch_size, output_depth) will have
 * output_ptr[b * output_depth + o] = \
 *   dst[b / dst_layout_rows, o / dst_layout_cols,
 *       b % dst_layout_rows, o % dst_layout_cols] * scaling_filters[b] *
 *       filter_scales[o]
 */
template <int Depth, int Width>
void Unpack(float* output_ptr, const int32_t* dst, int batch_size,
            int num_units, const float* scaling_factors,
            const float* filter_scales, int dst_layout_rows,
            int dst_layout_cols) {
  ReferenceUnpack<Depth, Width>(output_ptr, dst, batch_size, num_units,
                                scaling_factors, filter_scales, dst_layout_rows,
                                dst_layout_cols);
}

/* Computes dst = (lchd,rchd->lr, lhs, rhs)
 * Where l = lhs_layout_rows, r = rhs_layout_rows,
 * c = rhs_layout_cols = lhs_layout_cols.
 */
template <int RowsLeft, int RowsRight, int Cols>
void RunKernel(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
               int lhs_layout_rows, int lhs_layout_cols, int rhs_layout_rows,
               int rhs_layout_cols, int dst_layout_rows, int dst_layout_cols) {
  ReferenceRunKernel<RowsLeft, RowsRight, Cols>(
      lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols, rhs_layout_rows,
      rhs_layout_cols, dst_layout_rows, dst_layout_cols);
}

// Begin template specializations.
template <>
inline void Unpack<4, 1>(float* output_ptr, const int32_t* dst, int batch_size,
                         int num_units, const float* scaling_factors,
                         const float* filter_scales, int dst_layout_rows,
                         int dst_layout_cols) {
  ReferenceUnpack<4, 1>(output_ptr, dst, batch_size, num_units, scaling_factors,
                        filter_scales, dst_layout_rows, dst_layout_cols);
}

template <>
inline void RunKernel<4, 1, 32>(const uint8_t* lhs, const int8_t* rhs,
                                int32_t* dst, int lhs_layout_rows,
                                int lhs_layout_cols, int rhs_layout_rows,
                                int rhs_layout_cols, int dst_layout_rows,
                                int dst_layout_cols) {
  ReferenceRunKernel<4, 1, 32>(lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols,
                               rhs_layout_rows, rhs_layout_cols,
                               dst_layout_rows, dst_layout_cols);
}

// End template specializations.

// Compute sum of lhs * rhs columnwise and write output to output_ptr.
inline void RunAndUnpack(int rhs_width, const uint8_t* lhs, const int8_t* rhs,
                         int32_t* dst, int output_depth, int batch_size,
                         int lhs_layout_rows, int lhs_layout_cols,
                         int rhs_layout_rows, int rhs_layout_cols,
                         int dst_layout_rows, int dst_layout_cols,
                         float* output_ptr, const float* scaling_factors,
                         const float* filter_scales) {
  ReferenceRunKernel<4, 1, 32>(lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols,
                               rhs_layout_rows, rhs_layout_cols,
                               dst_layout_rows, dst_layout_cols);
  ReferenceUnpack<4, 1>(output_ptr, dst, batch_size, output_depth,
                        scaling_factors, filter_scales, dst_layout_rows,
                        dst_layout_cols);
}

}  // namespace optimized_4bit
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_REFERENCE_H_
