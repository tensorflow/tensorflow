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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_REFERENCE_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_REFERENCE_IMPL_H_

#include <stdint.h>

namespace tflite {
namespace optimized_4bit {

void ReferencePackInner(const int8_t* src, uint8_t* box, int src_rows,
                        int src_cols, int outer_row, int outer_col,
                        int outer_rows, int outer_cols, int inner_rows,
                        int inner_cols);

void ReferencePrepack(uint8_t** dest, const int8_t* tensor, int layout_rows,
                      int layout_cols, int src_rows, int src_cols, int width,
                      int depth);

void ReferenceBatchQuantizeFloats4Bit(const float* float_data_ptr, int n_batch,
                                      int n_data, int8_t* quantized_data_ptr,
                                      float* scaling_factors, int width,
                                      int depth, int32_t* input_offsets);

void ReferenceAssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                          const float* batch_scales,
                                          const float* filter_scales,
                                          const float* bias_ptr,
                                          float* output_ptr, int output_depth,
                                          int batch_size);

template <int Depth, int Width>
extern void ReferenceUnpack(float* output_ptr, const int32_t* dst,
                            int batch_size, int num_units,
                            const float* scaling_factors,
                            const float* filter_scales, int dst_layout_rows,
                            int dst_layout_cols);

template <int RowsLeft, int RowsRight, int Cols>
extern void ReferenceRunKernel(const uint8_t* lhs, const int8_t* rhs,
                               int32_t* dst, int lhs_layout_rows,
                               int lhs_layout_cols, int rhs_layout_rows,
                               int rhs_layout_cols, int dst_layout_rows,
                               int dst_layout_cols);

}  // namespace optimized_4bit
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_FULLY_CONNECTED_REFERENCE_IMPL_H_
