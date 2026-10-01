/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_MOE_BLOCK_SCALE_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_MOE_BLOCK_SCALE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Scale layout and dequantization helpers shared by the custom "moe" XNNPACK
// delegate kernel. They live in their own header so they can be unit tested
// without standing up a delegate.
//
// MoE expert weights are laid out as `[output_channels, num_experts, 1,
// input_channels]`, i.e. one row per (output channel, expert) pair. The
// matching scale tensor is `[output_channels, num_experts, 1, blocks_per_row]`.
// `blocks_per_row == 1` is per-output-channel quantization; larger values split
// the input axis into that many equally sized blocks, each with its own scale.
namespace tflite {
namespace xnnpack {

// How a single row's scales map onto its input channels.
struct BlockScaleLayout {
  // Number of scales per row. 1 means per-output-channel quantization.
  size_t groups_per_row;
  // Number of input channels covered by one scale.
  size_t group_size;
};

// Derives the layout from the total number of scale elements. `scale_elements`
// is expected to be an exact multiple of `num_rows`; callers validate that
// before reaching here, and a ragged value degrades to treating the row as
// per-channel rather than reading out of bounds.
inline BlockScaleLayout ResolveBlockScaleLayout(size_t scale_elements,
                                                size_t num_rows,
                                                size_t input_channels) {
  const size_t groups_per_row =
      (num_rows > 0) ? (scale_elements / num_rows) : 0;
  const size_t group_size =
      (groups_per_row > 0 && groups_per_row <= input_channels)
          ? (input_channels / groups_per_row)
          : 1;
  return {groups_per_row, group_size};
}

// Index of the scale covering input channel `in`. Clamped to the last block so
// an input axis that does not divide evenly cannot read past the row.
inline size_t BlockScaleIndex(const BlockScaleLayout& layout, size_t in) {
  if (layout.groups_per_row <= 1 || layout.group_size == 0) {
    return 0;
  }
  return std::min(in / layout.group_size, layout.groups_per_row - 1);
}

// Dequantizes a single int8 row `src_row` (`[input_channels]`) using
// `row_scales` (`[layout.groups_per_row]`) into `dst_row`.
inline void CopyAndDequantizeExpertWeightRowInt8(const int8_t* src_row,
                                                 const float* row_scales,
                                                 const BlockScaleLayout& layout,
                                                 size_t input_channels,
                                                 float* dst_row) {
  if (layout.groups_per_row <= 1 || layout.group_size == 0) {
    const float scale_val = row_scales[0];
    for (size_t in = 0; in < input_channels; ++in) {
      dst_row[in] = static_cast<float>(src_row[in]) * scale_val;
    }
    return;
  }
  const size_t num_groups = layout.groups_per_row;
  const size_t group_size = layout.group_size;
  size_t in = 0;
  for (size_t g = 0; g + 1 < num_groups && in < input_channels; ++g) {
    const float scale_val = row_scales[g];
    const size_t block_end = std::min(in + group_size, input_channels);
    for (; in < block_end; ++in) {
      dst_row[in] = static_cast<float>(src_row[in]) * scale_val;
    }
  }
  const float last_scale_val = row_scales[num_groups - 1];
  for (; in < input_channels; ++in) {
    dst_row[in] = static_cast<float>(src_row[in]) * last_scale_val;
  }
}

// Dequantizes output channel range `[out_begin, out_end)` belonging to `expert`
// into `dst` (`[output_channels, input_channels]`).
inline void CopyAndDequantizeExpertWeightRowsInt8Range(
    const int8_t* weight_i8, const float* scale, size_t scale_elements,
    size_t num_experts, size_t expert, size_t output_channels, size_t out_begin,
    size_t out_end, size_t input_channels, float* dst) {
  const size_t num_rows = output_channels * num_experts;
  const BlockScaleLayout layout =
      ResolveBlockScaleLayout(scale_elements, num_rows, input_channels);

  for (size_t out = out_begin; out < out_end; ++out) {
    const size_t row_idx = out * num_experts + expert;
    const int8_t* src_row = weight_i8 + row_idx * input_channels;
    float* dst_row = dst + out * input_channels;
    const float* row_scales = scale + row_idx * layout.groups_per_row;
    CopyAndDequantizeExpertWeightRowInt8(src_row, row_scales, layout,
                                         input_channels, dst_row);
  }
}

// Dequantizes the rows belonging to `expert` into `dst`, which is a dense
// `[output_channels, input_channels]` buffer.
inline void CopyAndDequantizeExpertWeightRowsInt8(
    const int8_t* weight_i8, const float* scale, size_t scale_elements,
    size_t num_experts, size_t expert, size_t output_channels,
    size_t input_channels, float* dst) {
  CopyAndDequantizeExpertWeightRowsInt8Range(
      weight_i8, scale, scale_elements, num_experts, expert, output_channels,
      /*out_begin=*/0, /*out_end=*/output_channels, input_channels, dst);
}

// Dequantizes a span `[in_begin, in_end)` of packed int4 weights (two nibbles
// per byte, low nibble first) with a constant `scale_val`.
inline void DequantizeInt4Span(const int8_t* src_row_packed, size_t in_begin,
                               size_t in_end, float scale_val, float* dst_row) {
  if (in_begin >= in_end) {
    return;
  }
  size_t in = in_begin;
  if ((in & 1) != 0) {
    const int8_t byte_val = src_row_packed[in >> 1];
    const int8_t high = static_cast<int8_t>(byte_val >> 4);
    dst_row[in] = static_cast<float>(high) * scale_val;
    ++in;
  }
  const size_t byte_begin = in >> 1;
  const size_t byte_end = in_end >> 1;
  for (size_t b = byte_begin; b < byte_end; ++b) {
    const int8_t byte_val = src_row_packed[b];
    const int8_t low = static_cast<int8_t>(byte_val << 4) >> 4;
    const int8_t high = static_cast<int8_t>(byte_val >> 4);
    dst_row[2 * b] = static_cast<float>(low) * scale_val;
    dst_row[2 * b + 1] = static_cast<float>(high) * scale_val;
  }
  in = byte_end * 2;
  if (in < in_end) {
    const int8_t byte_val = src_row_packed[in >> 1];
    const int8_t low = static_cast<int8_t>(byte_val << 4) >> 4;
    dst_row[in] = static_cast<float>(low) * scale_val;
  }
}

// Dequantizes a single packed int4 row `src_row_packed` into `dst_row`.
inline void CopyAndDequantizeExpertWeightRowInt4(const int8_t* src_row_packed,
                                                 const float* row_scales,
                                                 const BlockScaleLayout& layout,
                                                 size_t input_channels,
                                                 float* dst_row) {
  if (layout.groups_per_row <= 1 || layout.group_size == 0) {
    DequantizeInt4Span(src_row_packed, 0, input_channels, row_scales[0],
                       dst_row);
    return;
  }
  const size_t num_groups = layout.groups_per_row;
  const size_t group_size = layout.group_size;
  size_t in = 0;
  for (size_t g = 0; g + 1 < num_groups && in < input_channels; ++g) {
    const size_t block_end = std::min(in + group_size, input_channels);
    DequantizeInt4Span(src_row_packed, in, block_end, row_scales[g], dst_row);
    in = block_end;
  }
  if (in < input_channels) {
    DequantizeInt4Span(src_row_packed, in, input_channels,
                       row_scales[num_groups - 1], dst_row);
  }
}

// Dequantizes output channel range `[out_begin, out_end)` of packed int4
// weights belonging to `expert` into `dst` (`[output_channels,
// input_channels]`).
inline void CopyAndDequantizeExpertWeightRowsInt4Range(
    const int8_t* weight_i4_packed, const float* scale, size_t scale_elements,
    size_t num_experts, size_t expert, size_t output_channels, size_t out_begin,
    size_t out_end, size_t input_channels, float* dst) {
  const size_t num_rows = output_channels * num_experts;
  const BlockScaleLayout layout =
      ResolveBlockScaleLayout(scale_elements, num_rows, input_channels);

  for (size_t out = out_begin; out < out_end; ++out) {
    const size_t row_idx = out * num_experts + expert;
    const int8_t* src_row_packed =
        weight_i4_packed + (row_idx * input_channels) / 2;
    float* dst_row = dst + out * input_channels;
    const float* row_scales = scale + row_idx * layout.groups_per_row;
    CopyAndDequantizeExpertWeightRowInt4(src_row_packed, row_scales, layout,
                                         input_channels, dst_row);
  }
}

// As above, for int4 weights packed two nibbles per byte, low nibble first.
inline void CopyAndDequantizeExpertWeightRowsInt4(
    const int8_t* weight_i4_packed, const float* scale, size_t scale_elements,
    size_t num_experts, size_t expert, size_t output_channels,
    size_t input_channels, float* dst) {
  CopyAndDequantizeExpertWeightRowsInt4Range(
      weight_i4_packed, scale, scale_elements, num_experts, expert,
      output_channels, /*out_begin=*/0, /*out_end=*/output_channels,
      input_channels, dst);
}

// The dot products below ask clang to vectorize float reductions, which it only
// does when the loop hint allows reordering them. Instrumented builds (e.g.
// --config=ubsan) can block that transformation, and clang reports it as
// -Wpass-failed, which -Werror turns into a build break. The loops stay correct
// when left scalar, so the warning is silenced for this section only.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

// Computes the dot product of a single int8 weight row with `input`
// (`[input_channels]`) without materializing a dequantized FP32 row buffer.
inline float DotDequantizeExpertWeightRowInt8(const int8_t* src_row,
                                              const float* row_scales,
                                              const BlockScaleLayout& layout,
                                              size_t input_channels,
                                              const float* input) {
  if (layout.groups_per_row <= 1 || layout.group_size == 0) {
    float sum = 0.0f;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
    for (size_t in = 0; in < input_channels; ++in) {
      sum += static_cast<float>(src_row[in]) * input[in];
    }
    return sum * row_scales[0];
  }
  const size_t num_groups = layout.groups_per_row;
  const size_t group_size = layout.group_size;
  float total = 0.0f;
  size_t in = 0;
  for (size_t g = 0; g + 1 < num_groups && in < input_channels; ++g) {
    const size_t block_end = std::min(in + group_size, input_channels);
    float block_sum = 0.0f;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
    for (size_t i = in; i < block_end; ++i) {
      block_sum += static_cast<float>(src_row[i]) * input[i];
    }
    total += block_sum * row_scales[g];
    in = block_end;
  }
  if (in < input_channels) {
    float block_sum = 0.0f;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
    for (size_t i = in; i < input_channels; ++i) {
      block_sum += static_cast<float>(src_row[i]) * input[i];
    }
    total += block_sum * row_scales[num_groups - 1];
  }
  return total;
}

// Computes the unscaled dot product of a span `[in_begin, in_end)` of packed
// int4 weights with `input`.
inline float DotInt4Span(const int8_t* src_row_packed, size_t in_begin,
                         size_t in_end, const float* input) {
  if (in_begin >= in_end) {
    return 0.0f;
  }
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  size_t in = in_begin;
  if ((in & 1) != 0) {
    const int8_t byte_val = src_row_packed[in >> 1];
    const int8_t high = static_cast<int8_t>(byte_val >> 4);
    sum0 += static_cast<float>(high) * input[in];
    ++in;
  }
  const size_t byte_begin = in >> 1;
  const size_t byte_end = in_end >> 1;
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
  for (size_t b = byte_begin; b < byte_end; ++b) {
    const int8_t byte_val = src_row_packed[b];
    const int8_t low = static_cast<int8_t>(byte_val << 4) >> 4;
    const int8_t high = static_cast<int8_t>(byte_val >> 4);
    sum0 += static_cast<float>(low) * input[2 * b];
    sum1 += static_cast<float>(high) * input[2 * b + 1];
  }
  in = byte_end * 2;
  if (in < in_end) {
    const int8_t byte_val = src_row_packed[in >> 1];
    const int8_t low = static_cast<int8_t>(byte_val << 4) >> 4;
    sum0 += static_cast<float>(low) * input[in];
  }
  return sum0 + sum1;
}

// Computes the dot product of a single packed int4 weight row with `input`
// (`[input_channels]`) without materializing a dequantized FP32 row buffer.
inline float DotDequantizeExpertWeightRowInt4(const int8_t* src_row_packed,
                                              const float* row_scales,
                                              const BlockScaleLayout& layout,
                                              size_t input_channels,
                                              const float* input) {
  if (layout.groups_per_row <= 1 || layout.group_size == 0) {
    return DotInt4Span(src_row_packed, 0, input_channels, input) *
           row_scales[0];
  }
  const size_t num_groups = layout.groups_per_row;
  const size_t group_size = layout.group_size;
  float total = 0.0f;
  size_t in = 0;
  for (size_t g = 0; g + 1 < num_groups && in < input_channels; ++g) {
    const size_t block_end = std::min(in + group_size, input_channels);
    total += DotInt4Span(src_row_packed, in, block_end, input) * row_scales[g];
    in = block_end;
  }
  if (in < input_channels) {
    total += DotInt4Span(src_row_packed, in, input_channels, input) *
             row_scales[num_groups - 1];
  }
  return total;
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_MOE_BLOCK_SCALE_H_
