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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_FULLY_CONNECTED_4BIT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_FULLY_CONNECTED_4BIT_H_
#include <stdint.h>

#include <cstdlib>
#include <memory>

#if defined(FC_4BIT_SSE) && defined(__SSSE3__)
#include "tensorflow/lite/kernels/internal/optimized/4bit/sse_fully_connected.h"
#elif defined(FC_4BIT_NEON) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#include "tensorflow/lite/kernels/internal/optimized/4bit/neon_fully_connected.h"
#else
#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_reference.h"
#endif

namespace tflite {
namespace optimized_4bit {

// Define 4-bit filter block size: 4x32 (64 bytes)
constexpr int FilterWidth = 4;
constexpr int FilterDepth = 32;
constexpr int kDefaultAlignmentPadding = 63;

struct OpData4Bit {
  int rows_right = 1;
  int batch_size = 0;
  bool needs_prepack = true;
  uint8_t* prepacked_cache = nullptr;
  std::unique_ptr<uint8_t[]> prepacked_cache_buffer;

  void AllocatePackedRegion(size_t required_size) {
    prepacked_cache_buffer = std::make_unique<uint8_t[]>(required_size);
    prepacked_cache = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(prepacked_cache_buffer.get()) +
         kDefaultAlignmentPadding) &
        ~kDefaultAlignmentPadding);
  }
};

namespace api {
/* Prepack lhs matrix into dest.
 * Transform tensor from (src_rows, src_cols) to
 * (layout_rows / width, layout_cols / depth, width, depth) with possibly
 * padding, and interleaving values along depth / 2 dimensions.
 * dest should be aligned and allocated before prepack.
 */
inline void Prepack(uint8_t* dest, const int8_t* tensor, int layout_rows,
                    int layout_cols, int src_rows, int src_cols, int width,
                    int depth) {
  optimized_4bit::Prepack(dest, tensor, layout_rows, layout_cols, src_rows,
                          src_cols, width, depth);
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
  optimized_4bit::BatchQuantizeFloats4Bit(float_data_ptr, n_batch, n_data,
                                          quantized_data_ptr, scaling_factors,
                                          width, depth, input_offsets);
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
  optimized_4bit::AssignBiasAndComputeOffsets(
      input_offsets, batch_scales, filter_scales, bias_ptr, output_ptr,
      output_depth, batch_size);
}

// Compute sum of lhs * rhs columnwise and write output to output_ptr.
inline void RunAndUnpack(int rhs_width, const uint8_t* lhs, const int8_t* rhs,
                         int32_t* dst, int output_depth, int batch_size,
                         int lhs_layout_rows, int lhs_layout_cols,
                         int rhs_layout_rows, int rhs_layout_cols,
                         int dst_layout_rows, int dst_layout_cols,
                         float* output_ptr, const float* scaling_factors,
                         const float* filter_scales) {
  optimized_4bit::RunAndUnpack(
      rhs_width, lhs, rhs, dst, output_depth, batch_size, lhs_layout_rows,
      lhs_layout_cols, rhs_layout_rows, rhs_layout_cols, dst_layout_rows,
      dst_layout_cols, output_ptr, scaling_factors, filter_scales);
}

}  // namespace api
}  // namespace optimized_4bit
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_FULLY_CONNECTED_4BIT_H_
