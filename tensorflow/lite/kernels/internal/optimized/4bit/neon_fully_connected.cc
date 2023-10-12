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

#if defined(FC_4BIT_NEON) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#include <arm_neon.h>
#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "include/cpuinfo.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_common.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/neon_fully_connected_impl.h"

namespace tflite {
namespace optimized_4bit {

#ifndef __aarch64__
inline int16x8_t vqmovn_high_s32(int16x4_t a_s16x4, int32x4_t b_s32x4) {
  return vcombine_s16(a_s16x4, vqmovn_s32(b_s32x4));
}

inline int8x16_t vqmovn_high_s16(int8x8_t a_s8x8, int16x8_t b_s16x8) {
  return vcombine_s8(a_s8x8, vqmovn_s16(b_s16x8));
}

inline int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
  int32x4x2_t deinterleaved = vuzpq_s32(a, b);
  return vqaddq_s32(deinterleaved.val[0], deinterleaved.val[1]);
}

inline float vmaxvq_f32(float32x4_t max_f32x4) {
  float32x2_t max_f32x2 =
      vmax_f32(vget_low_f32(max_f32x4), vget_high_f32(max_f32x4));
  max_f32x2 = vpmax_f32(max_f32x2, max_f32x2);
  return vget_lane_f32(max_f32x2, 0);
}

inline int32x4_t vcvtaq_s32_f32(float32x4_t a_f32x4) {
  float32x4_t half = vdupq_n_f32(.5);
  float32x4_t sign =
      vcvtq_f32_u32(vshrq_n_u32(vreinterpretq_u32_f32(a_f32x4), 31));
  float32x4_t add_half = vaddq_f32(a_f32x4, half);
  float32x4_t round = vsubq_f32(add_half, sign);
  return vcvtq_s32_f32(round);
}

void NeonAssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                     const float* batch_scales,
                                     float* filter_scales,
                                     const float* bias_ptr, float* output_ptr,
                                     int output_depth, int batch_size) {
  if (bias_ptr) {
    for (int b = 0; b < batch_size; ++b) {
      const float* filter_scales_ptr = filter_scales;
      const float* tmp_bias_ptr = bias_ptr;
      float val = input_offsets[b] * batch_scales[b];
      int o = output_depth;
      const float32x4_t v4_f32x4 = vdupq_n_f32(val);
      for (; o >= 4; o -= 4) {
        float32x4_t v0_f32x4 = vld1q_f32(filter_scales_ptr);
        filter_scales_ptr += 4;
        float32x4_t v5_f32x4 = vld1q_f32(tmp_bias_ptr);
        tmp_bias_ptr += 4;
        v5_f32x4 = vmlaq_f32(v5_f32x4, v0_f32x4, v4_f32x4);
        vst1q_f32(output_ptr, v5_f32x4);
        output_ptr += 4;
      }
      for (; o > 0; --o) {
        *output_ptr++ = val * (*filter_scales_ptr++) + (*tmp_bias_ptr++);
      }
    }
    return;
  }
  for (int b = 0; b < batch_size; ++b) {
    const float* filter_scales_ptr = filter_scales;
    float val = input_offsets[b] * batch_scales[b];
    int o = output_depth;
    const float32x4_t v4_f32x4 = vdupq_n_f32(val);
    for (; o >= 4; o -= 4) {
      float32x4_t v0_f32x4 = vld1q_f32(filter_scales_ptr);
      filter_scales_ptr += 4;
      float32x4_t v13_f32x4 = vmulq_f32(v0_f32x4, v4_f32x4);
      vst1q_f32(output_ptr, v13_f32x4);
      output_ptr += 4;
    }
    for (; o > 0; --o) {
      *output_ptr++ = val * (*filter_scales_ptr++);
    }
  }
}

#else

void NeonAssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                     const float* batch_scales,
                                     float* filter_scales,
                                     const float* bias_ptr, float* output_ptr,
                                     int output_depth, int batch_size) {
  if (bias_ptr) {
    for (int b = 0; b < batch_size; ++b) {
      const float* filter_scales_ptr = filter_scales;
      const float* tmp_bias_ptr = bias_ptr;
      float val = input_offsets[b] * batch_scales[b];
      int o = output_depth;
      const float32x4_t v4_f32x4 = vdupq_n_f32(val);
      for (; o >= 16; o -= 16) {
        float32x4x4_t v0_to_v3_f32x4x4 = vld1q_f32_x4(filter_scales_ptr);
        filter_scales_ptr += 16;
        float32x4x4_t v5_to_v8_f32x4x4 = vld1q_f32_x4(tmp_bias_ptr);
        tmp_bias_ptr += 16;
        v5_to_v8_f32x4x4.val[0] = vfmaq_f32(v5_to_v8_f32x4x4.val[0],
                                            v0_to_v3_f32x4x4.val[0], v4_f32x4);
        v5_to_v8_f32x4x4.val[1] = vfmaq_f32(v5_to_v8_f32x4x4.val[1],
                                            v0_to_v3_f32x4x4.val[1], v4_f32x4);
        v5_to_v8_f32x4x4.val[2] = vfmaq_f32(v5_to_v8_f32x4x4.val[2],
                                            v0_to_v3_f32x4x4.val[2], v4_f32x4);
        v5_to_v8_f32x4x4.val[3] = vfmaq_f32(v5_to_v8_f32x4x4.val[3],
                                            v0_to_v3_f32x4x4.val[3], v4_f32x4);
        vst1q_f32_x4(output_ptr, v5_to_v8_f32x4x4);
        output_ptr += 16;
      }
      if (o >= 8) {
        float32x4x2_t v0_to_v1_f32x4x2 = vld1q_f32_x2(filter_scales_ptr);
        filter_scales_ptr += 8;
        float32x4x2_t v5_to_v6_f32x4x2 = vld1q_f32_x2(tmp_bias_ptr);
        tmp_bias_ptr += 8;
        v5_to_v6_f32x4x2.val[0] = vfmaq_f32(v5_to_v6_f32x4x2.val[0],
                                            v0_to_v1_f32x4x2.val[0], v4_f32x4);
        v5_to_v6_f32x4x2.val[1] = vfmaq_f32(v5_to_v6_f32x4x2.val[1],
                                            v0_to_v1_f32x4x2.val[1], v4_f32x4);
        vst1q_f32_x2(output_ptr, v5_to_v6_f32x4x2);
        output_ptr += 8;
        o -= 8;
      }
      if (o >= 4) {
        float32x4_t v0_f32x4 = vld1q_f32(filter_scales_ptr);
        filter_scales_ptr += 4;
        float32x4_t v5_f32x4 = vld1q_f32(tmp_bias_ptr);
        tmp_bias_ptr += 4;
        v5_f32x4 = vfmaq_f32(v5_f32x4, v0_f32x4, v4_f32x4);
        vst1q_f32(output_ptr, v5_f32x4);
        output_ptr += 4;
        o -= 4;
      }
      for (; o > 0; --o) {
        *output_ptr++ = val * (*filter_scales_ptr++) + (*tmp_bias_ptr++);
      }
    }
    return;
  }
  for (int b = 0; b < batch_size; ++b) {
    const float* filter_scales_ptr = filter_scales;
    float val = input_offsets[b] * batch_scales[b];
    int o = output_depth;
    const float32x4_t v4_f32x4 = vdupq_n_f32(val);
    for (; o >= 16; o -= 16) {
      float32x4x4_t v0_to_v3_f32x4x4 = vld1q_f32_x4(filter_scales_ptr);
      filter_scales_ptr += 16;
      float32x4x4_t v13_to_v16_f32x4x4;
      v13_to_v16_f32x4x4.val[0] = vmulq_f32(v0_to_v3_f32x4x4.val[0], v4_f32x4);
      v13_to_v16_f32x4x4.val[1] = vmulq_f32(v0_to_v3_f32x4x4.val[1], v4_f32x4);
      v13_to_v16_f32x4x4.val[2] = vmulq_f32(v0_to_v3_f32x4x4.val[2], v4_f32x4);
      v13_to_v16_f32x4x4.val[3] = vmulq_f32(v0_to_v3_f32x4x4.val[3], v4_f32x4);
      vst1q_f32_x4(output_ptr, v13_to_v16_f32x4x4);
      output_ptr += 16;
    }
    if (o >= 8) {
      float32x4x2_t v0_to_v1_f32x4x2 = vld1q_f32_x2(filter_scales_ptr);
      filter_scales_ptr += 8;
      float32x4x2_t v11_to_v12_f32x4x2;
      v11_to_v12_f32x4x2.val[0] = vmulq_f32(v0_to_v1_f32x4x2.val[0], v4_f32x4);
      v11_to_v12_f32x4x2.val[1] = vmulq_f32(v0_to_v1_f32x4x2.val[1], v4_f32x4);
      vst1q_f32_x2(output_ptr, v11_to_v12_f32x4x2);
      output_ptr += 8;
      o -= 8;
    }
    if (o >= 4) {
      float32x4_t v0_f32x4 = vld1q_f32(filter_scales_ptr);
      filter_scales_ptr += 4;
      float32x4_t v13_f32x4 = vmulq_f32(v0_f32x4, v4_f32x4);
      vst1q_f32(output_ptr, v13_f32x4);
      output_ptr += 4;
      o -= 4;
    }
    for (; o > 0; --o) {
      *output_ptr++ = val * (*filter_scales_ptr++);
    }
  }
}
#endif

void NeonPackInner(const int8_t* src, uint8_t* box, int src_rows, int src_cols,
                   int outer_row, int outer_col, int outer_rows, int outer_cols,
                   int inner_rows, int inner_cols) {
  // create a kernel-specific layout and store it into packed
  const int width = inner_rows;
  const int depth = inner_cols;
  const int real_depth = depth / 2;
  const int real_src_cols = src_cols / 2;
  // which virtual row
  const int row = outer_row * inner_rows;
  const int col = outer_col * inner_cols;
  int src_width = std::min(width, src_rows - row);
  int src_depth = std::min(depth, src_cols - col);
  int real_col = col / 2;
  const int8_t* src_data = src + row * real_src_cols + real_col;
  int real_src_depth = src_depth / 2;
  const int8x16_t seven = vdupq_n_s8(7);
  const int8x8_t seven8 = vdup_n_s8(7);
  for (int m = 0; m < src_width; ++m) {
    int i = 0;
    int k = 0;
    for (; i < (real_src_depth & (~15)); i += 16) {
      int8x16_t values_16x8 = vld1q_s8(src_data + i);
      int8x16_t uv1 = vshrq_n_s8(values_16x8, 4);
      int8x16_t lv1 = vshlq_n_s8(values_16x8, 4);
      uv1 = vaddq_s8(uv1, seven);
      lv1 = vshrq_n_s8(lv1, 4);
      lv1 = vaddq_s8(lv1, seven);
      int8x8_t iuvl = vget_low_s8(uv1);
      int8x8_t iuvh = vget_high_s8(uv1);
      int8x8_t ilvl = vget_low_s8(lv1);
      int8x8_t ilvh = vget_high_s8(lv1);
      uint8x8_t uvl = vshl_n_u8(vreinterpret_u8_s8(iuvl), 4);
      uint8x8_t lvl = vshl_n_u8(vreinterpret_u8_s8(ilvl), 4);
      uint8x8_t uv = vorr_u8(uvl, vreinterpret_u8_s8(iuvh));
      uint8x8_t lv = vorr_u8(lvl, vreinterpret_u8_s8(ilvh));
      uint8x8x2_t zipped = vzip_u8(lv, uv);
      uint8x16_t combined = vcombine_u8(zipped.val[0], zipped.val[1]);
      vst1q_u8(box + k, combined);
      k += 16;
    }
    // If exactly 16 values remaining, use fast path
    if (real_src_depth == (real_src_depth & (~7))) {
      for (; i < (real_src_depth & (~7)); i += 8) {
        int8x8_t values_8x8 = vld1_s8(src_data + i);
        int8x8_t uv1 = vshr_n_s8(values_8x8, 4);
        int8x8_t lv1 = vshl_n_s8(values_8x8, 4);
        uv1 = vadd_s8(uv1, seven8);
        lv1 = vshr_n_s8(lv1, 4);
        lv1 = vadd_s8(lv1, seven8);
        uint8x8_t uvl = vshl_n_u8(vreinterpret_u8_s8(uv1), 4);
        uint8x8_t lvl = vshl_n_u8(vreinterpret_u8_s8(lv1), 4);
        uint8x8x2_t zipped = vzip_u8(lvl, uvl);
        uint8x16_t combined = vcombine_u8(zipped.val[0], zipped.val[1]);
        vst1q_u8(box + k, combined);
        k += 16;
      }
    }
    // Handle remaining values -- if greater than 16 values,
    // shuffle.
    if (i < real_src_depth) {
      int remaining = 8;
      remaining =
          remaining < (real_src_depth - i) ? remaining : real_src_depth - i;
      for (int j = 0; j < remaining; ++j) {
        const int8_t v1 = (int8_t)src_data[i + j];
        int8_t uv1 = upper(v1);
        int8_t lv1 = lower(v1);
        int8_t uv2 = 0;
        int8_t lv2 = 0;
        if ((i + j + 8) < real_src_depth) {
          const int8_t v2 = (int8_t)src_data[i + j + 8];
          uv2 = upper(v2);
          lv2 = lower(v2);
        }
        box[k] = merge(lv1, lv2);
        box[k + 1] = merge(uv1, uv2);
        k += 2;
      }
    }
    box += real_depth;
    src_data += real_src_cols;
  }
}

void NeonPrepack(uint8_t* dest, const int8_t* tensor, int layout_rows,
                 int layout_cols, int src_rows, int src_cols, int width,
                 int depth) {
  // depth is always cols
  size_t size = layout_rows * layout_cols / 2;
  memset(dest, static_cast<uint8_t>(119), sizeof(uint8_t) * size);
  // basically, we need to make a new 4D matrix
  // [rows / width, cols / depth, width, depth] in depth-first
  int outer_cols = layout_cols / depth;
  int outer_rows = layout_rows / width;
  int inner_cols = depth;
  int inner_rows = width;
  for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
    for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
      const int cluster_index = outer_row * outer_cols + outer_col;
      const int real_depth = inner_cols / 2;
      uint8_t* box = dest + cluster_index * real_depth * inner_rows;
      NeonPackInner(tensor, box, src_rows, src_cols, outer_row, outer_col,
                    outer_rows, outer_cols, inner_rows, inner_cols);
    }
  }
}

void NeonBatchQuantizeFloats4Bit(const float* float_data_ptr, int n_batch,
                                 int n_data, int8_t* quantized_data_ptr,
                                 float* scaling_factors, int width, int depth,
                                 int32_t* input_offsets) {
  const int rows = n_batch;
  const int cols = n_data;
  // depth is alpways cols
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
      int c = 0;
      const float* start = tensor_data + (row + w) * cols;
      float32x4_t v1_f32x4 = vdupq_n_f32(0);
      for (; c < (cols & ~3); c += 4) {
        float32x4_t v0_f32x4 = vld1q_f32(start);
        v0_f32x4 = vabsq_f32(v0_f32x4);
        start += 4;
        v1_f32x4 = vmaxq_f32(v0_f32x4, v1_f32x4);
      }
      float scale_denom = vmaxvq_f32(v1_f32x4);
      for (; c < cols; ++c) {
        scale_denom = std::max(scale_denom, std::abs(*(start++)));
      }
      if (scale_denom == 0) {
        scale_denom = 127.0;
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
      __builtin_prefetch(box, 1, 3);
      for (int w = 0; w < src_width; ++w) {
        const float scale_w = scale[w];
        const float* float_data = tensor_data + (row + w) * cols + col;
        __builtin_prefetch(float_data, 0, 3);
        int32_t* input_offsets_ptr = input_offsets + (row + w);
        __builtin_prefetch(input_offsets_ptr, 1, 3);
        int8_t* x0 = box + w * depth;
        const float* x1 = float_data;
        int16x8_t v12_s16x8 = vdupq_n_s16(0);
        int32x4_t v13_s32x4 = vdupq_n_s32(0);
        size_t run_depth = 0;
        float32x4_t v0_f32x4 = vdupq_n_f32(scale_w);
#ifdef __aarch64__
        for (; run_depth < (src_depth & ~15); run_depth += 16) {
          const float32x4x4_t v1_f32x4x4 = vld1q_f32_x4(x1);
          x1 += 16;
          const float32x4_t v5_f32x4 = vmulq_f32(v1_f32x4x4.val[0], v0_f32x4);
          const float32x4_t v6_f32x4 = vmulq_f32(v1_f32x4x4.val[1], v0_f32x4);
          const float32x4_t v7_f32x4 = vmulq_f32(v1_f32x4x4.val[2], v0_f32x4);
          const float32x4_t v8_f32x4 = vmulq_f32(v1_f32x4x4.val[3], v0_f32x4);
          const int32x4_t v5_s32x4 = vcvtaq_s32_f32(v5_f32x4);
          const int32x4_t v6_s32x4 = vcvtaq_s32_f32(v6_f32x4);
          const int32x4_t v7_s32x4 = vcvtaq_s32_f32(v7_f32x4);
          const int32x4_t v8_s32x4 = vcvtaq_s32_f32(v8_f32x4);
          const int16x4_t v9_low_s16x4 = vqmovn_s32(v5_s32x4);
          const int16x8_t v9_s16x8 = vqmovn_high_s32(v9_low_s16x4, v6_s32x4);
          const int16x4_t v10_low_s16x4 = vqmovn_s32(v7_s32x4);
          const int16x8_t v10_s16x8 = vqmovn_high_s32(v10_low_s16x4, v8_s32x4);
          const int8x8_t v11_low_s8x8 = vqmovn_s16(v9_s16x8);
          const int8x16_t v11_s8x16 = vqmovn_high_s16(v11_low_s8x8, v10_s16x8);
          v12_s16x8 = vaddq_s16(v12_s16x8, v9_s16x8);
          v12_s16x8 = vaddq_s16(v12_s16x8, v10_s16x8);
          vst1q_s8(x0, v11_s8x16);
          x0 += 16;
        }
        for (; run_depth < (src_depth & ~7); run_depth += 8) {
          const float32x4x2_t v1_f32x4x2 = vld1q_f32_x2(x1);
          x1 += 8;
          const float32x4_t v5_f32x4 = vmulq_f32(v1_f32x4x2.val[0], v0_f32x4);
          const float32x4_t v6_f32x4 = vmulq_f32(v1_f32x4x2.val[1], v0_f32x4);
          const int32x4_t v5_s32x4 = vcvtaq_s32_f32(v5_f32x4);
          const int32x4_t v6_s32x4 = vcvtaq_s32_f32(v6_f32x4);
          const int16x4_t v9_low_s16x4 = vqmovn_s32(v5_s32x4);
          const int16x8_t v9_s16x8 = vqmovn_high_s32(v9_low_s16x4, v6_s32x4);
          const int8x8_t v11_low_s8x8 = vqmovn_s16(v9_s16x8);
          vst1_s8(x0, v11_low_s8x8);
          x0 += 8;
          v12_s16x8 = vaddq_s16(v12_s16x8, v9_s16x8);
        }
#else
        for (; run_depth < (src_depth & ~7); run_depth += 8) {
          const float32x4_t v1_f32x4_0 = vld1q_f32(x1);
          x1 += 4;
          const float32x4_t v1_f32x4_1 = vld1q_f32(x1);
          x1 += 4;
          const float32x4_t v5_f32x4 = vmulq_f32(v1_f32x4_0, v0_f32x4);
          const float32x4_t v6_f32x4 = vmulq_f32(v1_f32x4_1, v0_f32x4);
          const int32x4_t v5_s32x4 = vcvtaq_s32_f32(v5_f32x4);
          const int32x4_t v6_s32x4 = vcvtaq_s32_f32(v6_f32x4);
          const int16x4_t v9_low_s16x4 = vqmovn_s32(v5_s32x4);
          const int16x8_t v9_s16x8 = vqmovn_high_s32(v9_low_s16x4, v6_s32x4);
          const int8x8_t v11_low_s8x8 = vqmovn_s16(v9_s16x8);
          vst1_s8(x0, v11_low_s8x8);
          x0 += 8;
          v12_s16x8 = vaddq_s16(v12_s16x8, v9_s16x8);
        }
#endif
        int32_t row_sum = 0;
        if (run_depth > 0) {
          v13_s32x4 = vpadalq_s16(v13_s32x4, v12_s16x8);
          v13_s32x4 = vpaddq_s32(v13_s32x4, v13_s32x4);
          v13_s32x4 = vpaddq_s32(v13_s32x4, v13_s32x4);
          row_sum += vgetq_lane_s32(v13_s32x4, 0);
        }
        for (; run_depth < src_depth; run_depth++) {
          const float f = *x1++;
          const int8_t q =
              static_cast<int8_t>(::tflite::TfLiteRound(f * scale_w));
          *x0++ = q;
          row_sum += q;
        }
        input_offsets[row + w] += row_sum;
      }
    }
  }
  for (int r = 0; r < layout_rows; ++r) {
    input_offsets[r] = input_offsets[r] * zero_point_4bit;
  }
}

void NeonAssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                     const float* batch_scales,
                                     float* filter_scales,
                                     const float* bias_ptr, float* output_ptr,
                                     int output_depth, int batch_size);

template <int Depth, int Width>
void NeonUnpack(float* output_ptr, const int32_t* dst, int batch_size,
                int num_units, const float* scaling_factors,
                const float* filter_scales, int dst_layout_rows,
                int dst_layout_cols) {
  if (Width == 1) {
    const int outer_rows = dst_layout_rows / Width;
    const int outer_cols = dst_layout_cols / Depth;
    const int32_t* dst_ptr = dst;
    int unit = 0;
    for (int outer_col = 0; outer_col < outer_cols;
         ++outer_col, unit += Depth) {
      float* tmp_output_ptr = output_ptr + unit;
      int len = num_units - unit < Depth ? num_units - unit : Depth;
      int cond = len & ~3;
      const float* scaling_factors_ptr = scaling_factors;
      for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
        const float scale = *scaling_factors_ptr;
        const float* filter_scales_ptr = filter_scales + unit;
        int i = 0;
        for (; i < cond; i += 4) {
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
          *(tmp_output_ptr++) += *(dst_ptr++) * scale * (*filter_scales_ptr++);
        }
        for (; i < len; ++i) {
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
        float32x4_t v3_f32x4 = vld1q_dup_f32(scale);
        for (; d > 3; d -= 4) {
          int32x4_t v0_s32x4 = vld1q_s32(dst_ptr);
          dst_ptr += 4;
          float32x4_t v1_f32x4 = vcvtq_f32_s32(v0_s32x4);
          float32x4_t v2_f32x4 = vld1q_f32(filter_scales_ptr);
          filter_scales_ptr += 4;
          float32x4_t v5_f32x4 = vmulq_f32(v1_f32x4, v3_f32x4);
          float32x4_t v6_f32x4 = vmulq_f32(v5_f32x4, v2_f32x4);
          float32x4_t v7_f32x4 = vld1q_f32(tmp_output_ptr);
          float32x4_t v8_f32x4 = vaddq_f32(v6_f32x4, v7_f32x4);
          vst1q_f32(tmp_output_ptr, v8_f32x4);
          tmp_output_ptr += 4;
        }
        for (; d > 0; --d) {
          *tmp_output_ptr++ += *dst_ptr++ * (*scale) * (*filter_scales_ptr++);
        }
        dst_ptr += depth_offset;
        tmp_output_ptr += width_offset;
      }
    }
  }
}

inline bool HasSDot() { return cpuinfo_has_arm_neon_dot(); }

template <int RowsLeft, int RowsRight, int Cols>
void NeonRunKernel(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
                   int lhs_layout_rows, int lhs_layout_cols,
                   int rhs_layout_rows, int rhs_layout_cols,
                   int dst_layout_rows, int dst_layout_cols) {
#ifdef __aarch64__
  if (HasSDot()) {
    NeonRunKernelSDot<RowsLeft, RowsRight, Cols>(
        lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols, rhs_layout_rows,
        rhs_layout_cols, dst_layout_rows, dst_layout_cols);
    return;
  }
#endif
  NeonRunKernelNoSDot<RowsLeft, RowsRight, Cols>(
      lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols, rhs_layout_rows,
      rhs_layout_cols, dst_layout_rows, dst_layout_cols);
}

template void NeonUnpack<4, 1>(float* output_ptr, const int32_t* dst,
                               int batch_size, int num_units,
                               const float* scaling_factors,
                               const float* filter_scales, int dst_layout_rows,
                               int dst_layout_cols);

template void NeonRunKernel<4, 1, 32>(const uint8_t* lhs, const int8_t* rhs,
                                      int32_t* dst, int lhs_layout_rows,
                                      int lhs_layout_cols, int rhs_layout_rows,
                                      int rhs_layout_cols, int dst_layout_rows,
                                      int dst_layout_cols);

#ifdef __aarch64__
template void NeonUnpack<4, 2>(float* output_ptr, const int32_t* dst,
                               int batch_size, int num_units,
                               const float* scaling_factors,
                               const float* filter_scales, int dst_layout_rows,
                               int dst_layout_cols);

template void NeonRunKernel<4, 2, 32>(const uint8_t* lhs, const int8_t* rhs,
                                      int32_t* dst, int lhs_layout_rows,
                                      int lhs_layout_cols, int rhs_layout_rows,
                                      int rhs_layout_cols, int dst_layout_rows,
                                      int dst_layout_cols);

template void NeonUnpack<4, 4>(float* output_ptr, const int32_t* dst,
                               int batch_size, int num_units,
                               const float* scaling_factors,
                               const float* filter_scales, int dst_layout_rows,
                               int dst_layout_cols);

template void NeonRunKernel<4, 4, 32>(const uint8_t* lhs, const int8_t* rhs,
                                      int32_t* dst, int lhs_layout_rows,
                                      int lhs_layout_cols, int rhs_layout_rows,
                                      int rhs_layout_cols, int dst_layout_rows,
                                      int dst_layout_cols);
#endif

}  // namespace optimized_4bit
}  // namespace tflite
#endif  // defined(FC_4BIT_NEON)...
