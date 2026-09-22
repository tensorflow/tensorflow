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

// Dequantizes the rows belonging to `expert` into `dst`, which is a dense
// `[output_channels, input_channels]` buffer.
inline void CopyAndDequantizeExpertWeightRowsInt8(
    const int8_t* weight_i8, const float* scale, size_t scale_elements,
    size_t num_experts, size_t expert, size_t output_channels,
    size_t input_channels, float* dst) {
  const size_t num_rows = output_channels * num_experts;
  const BlockScaleLayout layout =
      ResolveBlockScaleLayout(scale_elements, num_rows, input_channels);

  for (size_t out = 0; out < output_channels; ++out) {
    const size_t row_idx = out * num_experts + expert;
    const int8_t* src_row = weight_i8 + row_idx * input_channels;
    float* dst_row = dst + out * input_channels;
    const float* row_scales = scale + row_idx * layout.groups_per_row;
    for (size_t in = 0; in < input_channels; ++in) {
      const float scale_val = row_scales[BlockScaleIndex(layout, in)];
      dst_row[in] = static_cast<float>(src_row[in]) * scale_val;
    }
  }
}

// As above, for int4 weights packed two nibbles per byte, low nibble first.
inline void CopyAndDequantizeExpertWeightRowsInt4(
    const int8_t* weight_i4_packed, const float* scale, size_t scale_elements,
    size_t num_experts, size_t expert, size_t output_channels,
    size_t input_channels, float* dst) {
  const size_t num_rows = output_channels * num_experts;
  const BlockScaleLayout layout =
      ResolveBlockScaleLayout(scale_elements, num_rows, input_channels);

  for (size_t out = 0; out < output_channels; ++out) {
    const size_t row_idx = out * num_experts + expert;
    const int8_t* src_row_packed =
        weight_i4_packed + (row_idx * input_channels) / 2;
    float* dst_row = dst + out * input_channels;
    const float* row_scales = scale + row_idx * layout.groups_per_row;

    for (size_t in = 0; in < input_channels; ++in) {
      const size_t byte_idx = in / 2;
      const int8_t byte_val = src_row_packed[byte_idx];
      const int8_t nibble = (in % 2 == 0)
                                ? static_cast<int8_t>(byte_val << 4) >> 4
                                : static_cast<int8_t>(byte_val >> 4);
      const float scale_val = row_scales[BlockScaleIndex(layout, in)];
      dst_row[in] = static_cast<float>(nibble) * scale_val;
    }
  }
}

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_MOE_BLOCK_SCALE_H_
