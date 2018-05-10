/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_

#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

#ifdef __aarch64__

#define DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE 10 * 10 * 64

template <int kDepth, int kStrideWidth, int kStrideHeight>
struct DepthwiseConvWindow {};

// clang-format gets confused with this file and ends up formatting lines to
// be larger than 80 characters. Turn off here and back on at the end of the
// file.

// clang-format off
template <>
struct DepthwiseConvWindow<8, 1, 1> {
 public:
  static inline void Run(const uint8* input_ptr, int64_t input_depth,
                         int32 input_offset, int64_t input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int64_t output_depth, int output_width,
                         int output_window_height,
                         int output_window_width) {
    const int64_t output_row_size = output_depth * output_width;
    const int64_t input_width_increment = 2 * input_depth;
    const int64_t input_height_increment = 2 * input_row_size;
    const int64_t output_height_increment = 2 * output_row_size;

#define DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "1"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "2"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1 "3"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "4"
#define DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "5"
#define DEPTHWISECONV_LABEL_HEIGHT_1 "6"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "7"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1 "8"
#define DEPTHWISECONV_LABEL_HEIGHT_1_END "9"

    asm volatile(
        // Performs depthwise convolutions for a window specified by
        // |output_window_height| and |output_window_width|. The inner-most loop
        // processes 2x2 outputs, and any leftovers at the end.
        //
        // Algorithm works as follows:
        //
        //   1. Load filters of 8 depth (8x3x3). Registers v0--v8 hold filter
        //      values.
        //   2. For 2 output heights at a time:
        //        i.  For 2 output widths at a time, load inputs for a 2x1 (2
        //            height, 1 width) output window (4x3 input window).
        //            Registers v9--v20 hold input values. Mul-add with
        //            accumulators v21--v24. Then run activation, downquantize
        //            and store. Repeat for the next 2x1 output window,
        //            leveraging overlapping inputs.
        //        ii. Handle single leftover width if exists.
        //   3. Handle single leftover height if exists.
        //        i.  For 2 output widths at a time, load inputs for a 1x2 (1
        //            height, 2 width) output window (3x4 input window).
        //            Registers v9--v20 hold input values. Mul-add with
        //            accumulators v21--v24. Then run activation, downquantize
        //            and store. Repeat for the next 1x2 output window,
        //            leveraging overlapping inputs.
        //        ii. Handle single leftover width if exists.
        //
        // Loads are placed as soon as the register is no longer needed and
        // interleaved with arithmetic operations to take advantage of
        // dual-issue pipelines. We also add input offsets as far from the loads
        // as possible to give loads enough cycles to fetch data from memory.

        // Set "constant" registers. These registers may be replaced with temp
        // values from time to time when there are not enough NEON registers.
        "dup v26.8h, %w[input_offset]\n"
        "cmp %w[output_window_height], #2\n"
        "dup v27.4s, %w[output_multiplier]\n"

        "neg w5, %w[output_shift]\n"
        "dup v28.4s, w5\n"

        "dup v29.4s, %w[output_offset]\n"
        "dup v30.4s, %w[output_activation_min]\n"
        "dup v31.4s, %w[output_activation_max]\n"

        "add x5, %[bias_ptr], #16\n"
        "dup v9.8h, %w[filter_offset]\n"

        // Load filters and add offsets.
        "ld1 {v0.8b}, [%[filter_ptr]], %[output_depth]\n"
        "ld1 {v1.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v0.8h, v9.8h, v0.8b\n"
        "ld1 {v2.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v1.8h, v9.8h, v1.8b\n"
        "ld1 {v3.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v2.8h, v9.8h, v2.8b\n"
        "ld1 {v4.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v3.8h, v9.8h, v3.8b\n"
        "ld1 {v5.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v4.8h, v9.8h, v4.8b\n"
        "ld1 {v6.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v5.8h, v9.8h, v5.8b\n"
        "ld1 {v7.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v6.8h, v9.8h, v6.8b\n"
        "ld1 {v8.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v7.8h, v9.8h, v7.8b\n"
        "uaddw v8.8h, v9.8h, v8.8b\n"

        "blt " DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_2_LOOP ":\n"
          // This loop processes 2x2 outputs. To avoid register exhaustion,
          // inputs for the left 2 outputs are loaded first, then the right
          // two outputs.
          "mov x6, %[input_ptr]\n"
          "mov x4, x6\n"
          "ld1 {v9.8b}, [x4], %[input_depth]\n"
          "add x0, x6, %[input_row_size]\n"
          "ld1 {v10.8b}, [x4], %[input_depth]\n"
          "add x1, x0, %[input_row_size]\n"
          "ld1 {v11.8b}, [x4], %[input_depth]\n"
          "add x7, x1, %[input_row_size]\n"
          "ld1 {v12.8b}, [x0], %[input_depth]\n"
          "mov w8, %w[output_window_width]\n"
          "ld1 {v13.8b}, [x0], %[input_depth]\n"
          "mov x2, %[output_ptr]\n"
          "ld1 {v14.8b}, [x0], %[input_depth]\n"
          "add x3, %[output_ptr], %[output_row_size]\n"
          "ld1 {v15.8b}, [x1], %[input_depth]\n"
          "cmp w8, #2\n"
          "ld1 {v16.8b}, [x1], %[input_depth]\n"
          "ld1 {v17.8b}, [x1], %[input_depth]\n"
          "ld1 {v18.8b}, [x7], %[input_depth]\n"
          "ld1 {v19.8b}, [x7], %[input_depth]\n"
          "ld1 {v20.8b}, [x7], %[input_depth]\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "ld1 {v22.4s}, [x5]\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "ld1 {v24.4s}, [x5]\n"

          "uaddw v9.8h, v26.8h, v9.8b\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"
          "uaddw v14.8h, v26.8h, v14.8b\n"
          "uaddw v15.8h, v26.8h, v15.8b\n"
          "uaddw v16.8h, v26.8h, v16.8b\n"
          "uaddw v17.8h, v26.8h, v17.8b\n"
          "uaddw v18.8h, v26.8h, v18.8b\n"
          "uaddw v19.8h, v26.8h, v19.8b\n"
          "uaddw v20.8h, v26.8h, v20.8b\n"

          "blt " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1 "f\n"

          //"loop_%=:\n"
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP ":\n"
            // Mul-add left outputs.
            "smlal v21.4s, v0.4h, v9.4h\n"
            "subs w8, w8, #2\n"
            "smlal2 v22.4s, v0.8h, v9.8h\n"
            "cmp w8, #2\n"
            "smlal v23.4s, v0.4h, v12.4h\n"
            "ld1 {v9.8b}, [x4]\n"
            "smlal2 v24.4s, v0.8h, v12.8h\n"
            "smlal v21.4s, v1.4h, v10.4h\n"
            "smlal2 v22.4s, v1.8h, v10.8h\n"
            "smlal v23.4s, v1.4h, v13.4h\n"
            "smlal2 v24.4s, v1.8h, v13.8h\n"
            "smlal v21.4s, v2.4h, v11.4h\n"
            "smlal2 v22.4s, v2.8h, v11.8h\n"
            "smlal v23.4s, v2.4h, v14.4h\n"
            "smlal2 v24.4s, v2.8h, v14.8h\n"
            "smlal v21.4s, v3.4h, v12.4h\n"
            "smlal2 v22.4s, v3.8h, v12.8h\n"
            "ld1 {v12.8b}, [x0]\n"
            "smlal v23.4s, v3.4h, v15.4h\n"
            "smlal2 v24.4s, v3.8h, v15.8h\n"
            "smlal v21.4s, v4.4h, v13.4h\n"
            "smlal2 v22.4s, v4.8h, v13.8h\n"
            "smlal v23.4s, v4.4h, v16.4h\n"
            "smlal2 v24.4s, v4.8h, v16.8h\n"
            "smlal v21.4s, v5.4h, v14.4h\n"
            "smlal2 v22.4s, v5.8h, v14.8h\n"
            "smlal v23.4s, v5.4h, v17.4h\n"
            "smlal2 v24.4s, v5.8h, v17.8h\n"
            "smlal v21.4s, v6.4h, v15.4h\n"
            "smlal2 v22.4s, v6.8h, v15.8h\n"
            "ld1 {v15.8b}, [x1]\n"
            "smlal v23.4s, v6.4h, v18.4h\n"
            "smlal2 v24.4s, v6.8h, v18.8h\n"
            "ld1 {v18.8b}, [x7]\n"
            "smlal v21.4s, v7.4h, v16.4h\n"
            "smlal2 v22.4s, v7.8h, v16.8h\n"
            "smlal v23.4s, v7.4h, v19.4h\n"
            "smlal2 v24.4s, v7.8h, v19.8h\n"
            "smlal v21.4s, v8.4h, v17.4h\n"
            "smlal2 v22.4s, v8.8h, v17.8h\n"
            "smlal v23.4s, v8.4h, v20.4h\n"
            "smlal2 v24.4s, v8.8h, v20.8h\n"

            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v25.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v25.4s, v25.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v25.4s\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, %w[output_offset]\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, %w[output_activation_min]\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, %w[output_activation_max]\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x5]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x5]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v26.8h, v9.8b\n"
            "st1 {v21.8b}, [x2], %[output_depth]\n"
            "uaddw v12.8h, v26.8h, v12.8b\n"
            "st1 {v23.8b}, [x3], %[output_depth]\n"
            "uaddw v15.8h, v26.8h, v15.8b\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "uaddw v18.8h, v26.8h, v18.8b\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"

            // Mul-add right outputs.
            "smlal v21.4s, v0.4h, v10.4h\n"
            "add x6, x6, %[input_width_increment]\n"
            "smlal2 v22.4s, v0.8h, v10.8h\n"
            "mov x4, x6\n"
            "smlal v23.4s, v0.4h, v13.4h\n"
            "add x0, x6, %[input_row_size]\n"
            "smlal2 v24.4s, v0.8h, v13.8h\n"
            "add x1, x0, %[input_row_size]\n"
            "smlal v21.4s, v1.4h, v11.4h\n"
            "add x7, x1, %[input_row_size]\n"
            "smlal2 v22.4s, v1.8h, v11.8h\n"
            "smlal v23.4s, v1.4h, v14.4h\n"
            "smlal2 v24.4s, v1.8h, v14.8h\n"
            "smlal v21.4s, v2.4h, v9.4h\n"
            "smlal2 v22.4s, v2.8h, v9.8h\n"
            "ld1 {v9.8b}, [x4], %[input_depth]\n"
            "smlal v23.4s, v2.4h, v12.4h\n"
            "ld1 {v10.8b}, [x4], %[input_depth]\n"
            "smlal2 v24.4s, v2.8h, v12.8h\n"
            "ld1 {v11.8b}, [x4], %[input_depth]\n"
            "smlal v21.4s, v3.4h, v13.4h\n"
            "smlal2 v22.4s, v3.8h, v13.8h\n"
            "smlal v23.4s, v3.4h, v16.4h\n"
            "smlal2 v24.4s, v3.8h, v16.8h\n"
            "smlal v21.4s, v4.4h, v14.4h\n"
            "smlal2 v22.4s, v4.8h, v14.8h\n"
            "smlal v23.4s, v4.4h, v17.4h\n"
            "smlal2 v24.4s, v4.8h, v17.8h\n"
            "smlal v21.4s, v5.4h, v12.4h\n"
            "smlal2 v22.4s, v5.8h, v12.8h\n"
            "ld1 {v12.8b}, [x0], %[input_depth]\n"
            "smlal v23.4s, v5.4h, v15.4h\n"
            "ld1 {v13.8b}, [x0], %[input_depth]\n"
            "smlal2 v24.4s, v5.8h, v15.8h\n"
            "ld1 {v14.8b}, [x0], %[input_depth]\n"
            "smlal v21.4s, v6.4h, v16.4h\n"
            "smlal2 v22.4s, v6.8h, v16.8h\n"
            "smlal v23.4s, v6.4h, v19.4h\n"
            "smlal2 v24.4s, v6.8h, v19.8h\n"
            "smlal v21.4s, v7.4h, v17.4h\n"
            "smlal2 v22.4s, v7.8h, v17.8h\n"
            "smlal v23.4s, v7.4h, v20.4h\n"
            "smlal2 v24.4s, v7.8h, v20.8h\n"
            "smlal v21.4s, v8.4h, v15.4h\n"
            "smlal2 v22.4s, v8.8h, v15.8h\n"
            "ld1 {v15.8b}, [x1], %[input_depth]\n"
            "smlal v23.4s, v8.4h, v18.4h\n"
            "ld1 {v16.8b}, [x1], %[input_depth]\n"
            "smlal2 v24.4s, v8.8h, v18.8h\n"
            "ld1 {v17.8b}, [x1], %[input_depth]\n"

            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "ld1 {v18.8b}, [x7], %[input_depth]\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "ld1 {v19.8b}, [x7], %[input_depth]\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "ld1 {v20.8b}, [x7], %[input_depth]\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v25.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v25.4s, v25.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v25.4s\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, %w[output_offset]\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, %w[output_activation_min]\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, %w[output_activation_max]\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x5]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x5]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v26.8h, v9.8b\n"
            "st1 {v21.8b}, [x2], %[output_depth]\n"
            "uaddw v10.8h, v26.8h, v10.8b\n"
            "st1 {v23.8b}, [x3], %[output_depth]\n"
            "uaddw v11.8h, v26.8h, v11.8b\n"
            "uaddw v12.8h, v26.8h, v12.8b\n"
            "uaddw v13.8h, v26.8h, v13.8b\n"
            "uaddw v14.8h, v26.8h, v14.8b\n"
            "uaddw v15.8h, v26.8h, v15.8b\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "uaddw v16.8h, v26.8h, v16.8b\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"
            "uaddw v17.8h, v26.8h, v17.8b\n"
            "uaddw v18.8h, v26.8h, v18.8b\n"
            "uaddw v19.8h, v26.8h, v19.8b\n"
            "uaddw v20.8h, v26.8h, v20.8b\n"

            "bge " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "b\n"

          // Do last width column if exists.
          "cmp w8, #1\n"
          "blt " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "f\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1 ":\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "smlal v23.4s, v0.4h, v12.4h\n"
          "smlal2 v24.4s, v0.8h, v12.8h\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "smlal v23.4s, v1.4h, v13.4h\n"
          "smlal2 v24.4s, v1.8h, v13.8h\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "smlal v23.4s, v2.4h, v14.4h\n"
          "smlal2 v24.4s, v2.8h, v14.8h\n"
          "smlal v21.4s, v3.4h, v12.4h\n"
          "smlal2 v22.4s, v3.8h, v12.8h\n"
          "smlal v23.4s, v3.4h, v15.4h\n"
          "smlal2 v24.4s, v3.8h, v15.8h\n"
          "smlal v21.4s, v4.4h, v13.4h\n"
          "smlal2 v22.4s, v4.8h, v13.8h\n"
          "smlal v23.4s, v4.4h, v16.4h\n"
          "smlal2 v24.4s, v4.8h, v16.8h\n"
          "smlal v21.4s, v5.4h, v14.4h\n"
          "smlal2 v22.4s, v5.8h, v14.8h\n"
          "smlal v23.4s, v5.4h, v17.4h\n"
          "smlal2 v24.4s, v5.8h, v17.8h\n"
          "smlal v21.4s, v6.4h, v15.4h\n"
          "smlal2 v22.4s, v6.8h, v15.8h\n"
          "smlal v23.4s, v6.4h, v18.4h\n"
          "smlal2 v24.4s, v6.8h, v18.8h\n"
          "smlal v21.4s, v7.4h, v16.4h\n"
          "smlal2 v22.4s, v7.8h, v16.8h\n"
          "smlal v23.4s, v7.4h, v19.4h\n"
          "smlal2 v24.4s, v7.8h, v19.8h\n"
          "smlal v21.4s, v8.4h, v17.4h\n"
          "smlal2 v22.4s, v8.8h, v17.8h\n"
          "smlal v23.4s, v8.4h, v20.4h\n"
          "smlal2 v24.4s, v8.8h, v20.8h\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v9.16b, v21.16b, v28.16b\n"
          "and v12.16b, v22.16b, v28.16b\n"
          "and v15.16b, v23.16b, v28.16b\n"
          "and v18.16b, v24.16b, v28.16b\n"
          "sshr v9.4s, v9.4s, #31\n"
          "sshr v12.4s, v12.4s, #31\n"
          "sshr v15.4s, v15.4s, #31\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sqadd v21.4s, v21.4s, v9.4s\n"
          "sqadd v22.4s, v22.4s, v12.4s\n"
          "sqadd v23.4s, v23.4s, v15.4s\n"
          "sqadd v24.4s, v24.4s, v18.4s\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "st1 {v21.8b}, [x2], %[output_depth]\n"
          "st1 {v23.8b}, [x3], %[output_depth]\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP ":\n"
          "subs %w[output_window_height], %w[output_window_height], #2\n"
          "add %[input_ptr], %[input_ptr], %[input_height_increment]\n"
          "cmp %w[output_window_height], #2\n"
          "add %[output_ptr], %[output_ptr], %[output_height_increment]\n"
          "bge " DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "b\n"

        DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP ":\n"
        "cmp %w[output_window_height], #1\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        DEPTHWISECONV_LABEL_HEIGHT_1 ":\n"
        // Load inputs for 3x4 input window which corresponds to a 1x2 output
        // window.
        "mov x4, %[input_ptr]\n"
        "ld1 {v9.8b}, [x4], %[input_depth]\n"
        "add x0, %[input_ptr], %[input_row_size]\n"
        "ld1 {v10.8b}, [x4], %[input_depth]\n"
        "add x1, x0, %[input_row_size]\n"
        "ld1 {v11.8b}, [x4], %[input_depth]\n"
        "add x7, x1, %[input_row_size]\n"
        "ld1 {v12.8b}, [x4], %[input_depth]\n"
        "mov w8, %w[output_window_width]\n"
        "ld1 {v13.8b}, [x0], %[input_depth]\n"
        "mov x2, %[output_ptr]\n"
        "ld1 {v14.8b}, [x0], %[input_depth]\n"
        "add x3, %[output_ptr], %[output_row_size]\n"
        "ld1 {v15.8b}, [x0], %[input_depth]\n"
        "cmp w8, #2\n"
        "ld1 {v16.8b}, [x0], %[input_depth]\n"
        "ld1 {v17.8b}, [x1], %[input_depth]\n"
        "ld1 {v18.8b}, [x1], %[input_depth]\n"
        "ld1 {v19.8b}, [x1], %[input_depth]\n"
        "ld1 {v20.8b}, [x1], %[input_depth]\n"
        "ld1 {v21.4s}, [%[bias_ptr]]\n"
        "ld1 {v22.4s}, [x5]\n"
        "ld1 {v23.4s}, [%[bias_ptr]]\n"
        "ld1 {v24.4s}, [x5]\n"

        "uaddw v9.8h, v26.8h, v9.8b\n"
        "uaddw v10.8h, v26.8h, v10.8b\n"
        "uaddw v11.8h, v26.8h, v11.8b\n"
        "uaddw v12.8h, v26.8h, v12.8b\n"
        "uaddw v13.8h, v26.8h, v13.8b\n"
        "uaddw v14.8h, v26.8h, v14.8b\n"
        "uaddw v15.8h, v26.8h, v15.8b\n"
        "uaddw v16.8h, v26.8h, v16.8b\n"
        "uaddw v17.8h, v26.8h, v17.8b\n"
        "uaddw v18.8h, v26.8h, v18.8b\n"
        "uaddw v19.8h, v26.8h, v19.8b\n"
        "uaddw v20.8h, v26.8h, v20.8b\n"

        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1 "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP ":\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "subs w8, w8, #2\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "cmp w8, #2\n"
          "smlal v23.4s, v0.4h, v10.4h\n"
          "add %[input_ptr], %[input_ptr], %[input_width_increment]\n"
          "smlal2 v24.4s, v0.8h, v10.8h\n"
          "mov x4, %[input_ptr]\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "ld1 {v9.8b}, [x4], %[input_depth]\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x4], %[input_depth]\n"
          "smlal v23.4s, v1.4h, v11.4h\n"
          "add x0, %[input_ptr], %[input_row_size]\n"
          "smlal2 v24.4s, v1.8h, v11.8h\n"
          "add x1, x0, %[input_row_size]\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "add x7, x1, %[input_row_size]\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x4], %[input_depth]\n"
          "smlal v23.4s, v2.4h, v12.4h\n"
          "smlal2 v24.4s, v2.8h, v12.8h\n"
          "ld1 {v12.8b}, [x4], %[input_depth]\n"
          "smlal v21.4s, v3.4h, v13.4h\n"
          "smlal2 v22.4s, v3.8h, v13.8h\n"
          "ld1 {v13.8b}, [x0], %[input_depth]\n"
          "smlal v23.4s, v3.4h, v14.4h\n"
          "smlal2 v24.4s, v3.8h, v14.8h\n"
          "smlal v21.4s, v4.4h, v14.4h\n"
          "smlal2 v22.4s, v4.8h, v14.8h\n"
          "ld1 {v14.8b}, [x0], %[input_depth]\n"
          "smlal v23.4s, v4.4h, v15.4h\n"
          "smlal2 v24.4s, v4.8h, v15.8h\n"
          "smlal v21.4s, v5.4h, v15.4h\n"
          "smlal2 v22.4s, v5.8h, v15.8h\n"
          "ld1 {v15.8b}, [x0], %[input_depth]\n"
          "smlal v23.4s, v5.4h, v16.4h\n"
          "smlal2 v24.4s, v5.8h, v16.8h\n"
          "ld1 {v16.8b}, [x0], %[input_depth]\n"
          "smlal v21.4s, v6.4h, v17.4h\n"
          "smlal2 v22.4s, v6.8h, v17.8h\n"
          "ld1 {v17.8b}, [x1], %[input_depth]\n"
          "smlal v23.4s, v6.4h, v18.4h\n"
          "smlal2 v24.4s, v6.8h, v18.8h\n"
          "smlal v21.4s, v7.4h, v18.4h\n"
          "smlal2 v22.4s, v7.8h, v18.8h\n"
          "ld1 {v18.8b}, [x1], %[input_depth]\n"
          "smlal v23.4s, v7.4h, v19.4h\n"
          "smlal2 v24.4s, v7.8h, v19.8h\n"
          "smlal v21.4s, v8.4h, v19.4h\n"
          "smlal2 v22.4s, v8.8h, v19.8h\n"
          "ld1 {v19.8b}, [x1], %[input_depth]\n"
          "smlal v23.4s, v8.4h, v20.4h\n"
          "smlal2 v24.4s, v8.8h, v20.8h\n"
          "ld1 {v20.8b}, [x1], %[input_depth]\n"

          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v25.16b, v21.16b, v28.16b\n"
          "and v29.16b, v22.16b, v28.16b\n"
          "and v30.16b, v23.16b, v28.16b\n"
          "and v31.16b, v24.16b, v28.16b\n"
          "sshr v25.4s, v25.4s, #31\n"
          "sshr v29.4s, v29.4s, #31\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v21.4s, v21.4s, v25.4s\n"
          "sqadd v22.4s, v22.4s, v29.4s\n"
          "dup v29.4s, %w[output_offset]\n"
          "sqadd v23.4s, v23.4s, v30.4s\n"
          "dup v30.4s, %w[output_activation_min]\n"
          "sqadd v24.4s, v24.4s, v31.4s\n"
          "dup v31.4s, %w[output_activation_max]\n"
          "srshl v21.4s, v21.4s, v28.4s\n"
          "srshl v22.4s, v22.4s, v28.4s\n"
          "srshl v23.4s, v23.4s, v28.4s\n"
          "srshl v24.4s, v24.4s, v28.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "ld1 {v22.4s}, [x5]\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "ld1 {v24.4s}, [x5]\n"
          "sqxtun v21.8b, v21.8h\n"
          "sqxtun v23.8b, v23.8h\n"
          "uaddw v9.8h, v26.8h, v9.8b\n"
          "st1 {v21.8b}, [%[output_ptr]], %[output_depth]\n"
          "uaddw v10.8h, v26.8h, v10.8b\n"
          "st1 {v23.8b}, [%[output_ptr]], %[output_depth]\n"
          "uaddw v11.8h, v26.8h, v11.8b\n"
          "uaddw v12.8h, v26.8h, v12.8b\n"
          "uaddw v13.8h, v26.8h, v13.8b\n"
          "uaddw v14.8h, v26.8h, v14.8b\n"
          "uaddw v15.8h, v26.8h, v15.8b\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "uaddw v16.8h, v26.8h, v16.8b\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "uaddw v17.8h, v26.8h, v17.8b\n"
          "uaddw v18.8h, v26.8h, v18.8b\n"
          "uaddw v19.8h, v26.8h, v19.8b\n"
          "uaddw v20.8h, v26.8h, v20.8b\n"

          "bge " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "b\n"

        "cmp w8, #1\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        // Do bottom right output if exists.
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1 ":\n"
        "smlal v21.4s, v0.4h, v9.4h\n"
        "smlal2 v22.4s, v0.8h, v9.8h\n"
        "smlal v21.4s, v1.4h, v10.4h\n"
        "smlal2 v22.4s, v1.8h, v10.8h\n"
        "smlal v21.4s, v2.4h, v11.4h\n"
        "smlal2 v22.4s, v2.8h, v11.8h\n"
        "smlal v21.4s, v3.4h, v13.4h\n"
        "smlal2 v22.4s, v3.8h, v13.8h\n"
        "smlal v21.4s, v4.4h, v14.4h\n"
        "smlal2 v22.4s, v4.8h, v14.8h\n"
        "smlal v21.4s, v5.4h, v15.4h\n"
        "smlal2 v22.4s, v5.8h, v15.8h\n"
        "smlal v21.4s, v6.4h, v17.4h\n"
        "smlal2 v22.4s, v6.8h, v17.8h\n"
        "smlal v21.4s, v7.4h, v18.4h\n"
        "smlal2 v22.4s, v7.8h, v18.8h\n"
        "smlal v21.4s, v8.4h, v19.4h\n"
        "smlal2 v22.4s, v8.8h, v19.8h\n"

        "sqrdmulh v21.4s, v21.4s, v27.4s\n"
        "sqrdmulh v22.4s, v22.4s, v27.4s\n"
        "and v9.16b, v21.16b, v28.16b\n"
        "and v12.16b, v22.16b, v28.16b\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v12.4s, v12.4s, #31\n"
        "sqadd v21.4s, v21.4s, v9.4s\n"
        "sqadd v22.4s, v22.4s, v12.4s\n"
        "srshl v21.4s, v21.4s, v28.4s\n"
        "srshl v22.4s, v22.4s, v28.4s\n"
        "add v21.4s, v21.4s, v29.4s\n"
        "add v22.4s, v22.4s, v29.4s\n"
        "smax v21.4s, v21.4s, v30.4s\n"
        "smax v22.4s, v22.4s, v30.4s\n"
        "smin v21.4s, v21.4s, v31.4s\n"
        "smin v22.4s, v22.4s, v31.4s\n"
        "sqxtn v21.4h, v21.4s\n"
        "sqxtn2 v21.8h, v22.4s\n"
        "sqxtun v21.8b, v21.8h\n"
        "st1 {v21.8b}, [%[output_ptr]]\n"

        DEPTHWISECONV_LABEL_HEIGHT_1_END ":\n"

    :
    // Outputs.
    [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
    [output_ptr] "+r"(output_ptr),
    [output_window_height] "+r"(output_window_height)
    :
    // Inputs.
    [bias_ptr] "r"(bias_ptr), [output_depth] "r"(output_depth),
    [filter_offset] "r"(filter_offset), [input_row_size] "r"(input_row_size),
    [input_depth] "r"(input_depth), [input_offset] "r"(input_offset),
    [output_multiplier] "r"(output_multiplier),
    [output_shift] "r"(output_shift), [output_offset] "r"(output_offset),
    [output_activation_min] "r"(output_activation_min),
    [output_activation_max] "r"(output_activation_max),
    [output_row_size] "r"(output_row_size),
    [output_window_width] "r"(output_window_width),
    [input_width_increment] "r"(input_width_increment),
    [input_height_increment] "r"(input_height_increment),
    [output_height_increment] "r"(output_height_increment)
    :
    // Clobbers.
    // We use these NEON registers.
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
    "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
    "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
    // We use these general-purpose registers.
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "w8");

#undef DEPTHWISECONV_LABEL_HEIGHT_1_END
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1
#undef DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_LOOP
  }
};

template <>
struct DepthwiseConvWindow<8, 2, 2> {
  static inline void Run(const uint8* input_ptr, int64_t input_depth,
                         int32 input_offset, int64_t input_row_size,
                         const uint8* filter_ptr, int32 filter_offset,
                         const int32* bias_ptr, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_ptr,
                         int64_t output_depth, int output_width,
                         int output_window_height, int output_window_width) {
    const int64_t output_row_size = output_depth * output_width;
    const int64_t input_width_increment = 4 * input_depth;
    const int64_t input_height_increment = 4 * input_row_size;
    const int64_t output_height_increment = 2 * output_row_size;

#define DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "1"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "2"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1 "3"
#define DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "4"
#define DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "5"
#define DEPTHWISECONV_LABEL_HEIGHT_1 "6"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "7"
#define DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1 "8"
#define DEPTHWISECONV_LABEL_HEIGHT_1_END "9"

    asm volatile(
        // Performs depthwise convolutions for a window specified by
        // |output_window_height| and |output_window_width|. The inner-most loop
        // processes 2x2 outputs, and any leftovers at the end.
        //
        // Algorithm works as follows:
        //
        //   1. Load filters of 8 depth (8x3x3). Registers v0--v8 hold filter
        //      values.
        //   2. For 2 output heights at a time:
        //        i.  For 2 output widths at a time at stride 2, a 5x5 input
        //            window is required. To avoid register exhaustion, we load
        //            the first 2 rows of the 5x5 input window into registers
        //            v9--v18, and use the same registers to load the next 2
        //            rows, and finally v9--v13 to load the last row.
        //            Accumulators for all 2x2 outputs are reserved by registers
        //            v21-v22 (top left output), v23-v24 (top right output),
        //            v19-v20 (bottom left output), v25-v26 (bottom right
        //            output).
        //        ii. Handle single leftover width if exists.
        //   3. Handle single leftover height if exists.
        //        i.  For 2 output widths at a time at stride 2, load inputs for
        //            a 1x2 (1 height, 2 width) output window (3x5 input
        //            window). Registers v9--v24 hold input values. Mul-add with
        //            accumulators v24--v27.
        //        ii. Handle single leftover width if exists.
        //
        // Loads are placed as soon as the register is no longer needed and
        // interleaved with arithmetic operations to take advantage of
        // dual-issue pipelines. We also add input offsets as far from the loads
        // as possible to give loads enough cycles to fetch data from memory.

        // Set "constant" registers. These registers may be replaced with temp
        // values from time to time when there are not enough NEON registers.
        "neg w7, %w[output_shift]\n"
        "dup v26.4s, w7\n"
        "cmp %w[output_window_height], #2\n"
        "dup v27.4s, %w[output_multiplier]\n"
        "dup v28.8h, %w[input_offset]\n"
        "dup v29.4s, %w[output_offset]\n"
        "dup v30.4s, %w[output_activation_min]\n"
        "dup v31.4s, %w[output_activation_max]\n"

        // Load filters and add offsets.
        "add x5, %[bias_ptr], #16\n"
        "ld1 {v0.8b}, [%[filter_ptr]], %[output_depth]\n"
        "dup v9.8h, %w[filter_offset]\n"
        "ld1 {v1.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v0.8h, v9.8h, v0.8b\n"
        "ld1 {v2.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v1.8h, v9.8h, v1.8b\n"
        "ld1 {v3.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v2.8h, v9.8h, v2.8b\n"
        "ld1 {v4.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v3.8h, v9.8h, v3.8b\n"
        "ld1 {v5.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v4.8h, v9.8h, v4.8b\n"
        "ld1 {v6.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v5.8h, v9.8h, v5.8b\n"
        "ld1 {v7.8b}, [%[filter_ptr]], %[output_depth]\n"
        "uaddw v6.8h, v9.8h, v6.8b\n"
        "ld1 {v8.8b}, [%[filter_ptr]]\n"
        "uaddw v7.8h, v9.8h, v7.8b\n"
        "uaddw v8.8h, v9.8h, v8.8b\n"

        "blt " DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_2_LOOP ":\n"
          // Load the first two rows of the 5x5 input window, then reuse the
          // same registers to load subsequent rows as they become available.
          "mov x6, %[input_ptr]\n"
          "mov x0, x6\n"
          "add x1, x0, %[input_row_size]\n"
          "ld1 {v9.8b}, [x0], %[input_depth]\n"
          "mov w4, %w[output_window_width]\n"
          "ld1 {v10.8b}, [x0], %[input_depth]\n"
          "cmp w4, #2\n"
          "ld1 {v11.8b}, [x0], %[input_depth]\n"
          "add x2, x1, %[input_row_size]\n"
          "ld1 {v12.8b}, [x0], %[input_depth]\n"
          "ld1 {v13.8b}, [x0]\n"
          "add x0, x2, %[input_row_size]\n"
          "ld1 {v14.8b}, [x1], %[input_depth]\n"
          "mov x3, %[output_ptr]\n"
          "ld1 {v15.8b}, [x1], %[input_depth]\n"
          "add x10, %[output_ptr], %[output_row_size]\n"
          "ld1 {v16.8b}, [x1], %[input_depth]\n"
          "ld1 {v17.8b}, [x1], %[input_depth]\n"
          "ld1 {v18.8b}, [x1]\n"
          "add x1, x0, %[input_row_size]\n"

          "uaddw v9.8h, v28.8h, v9.8b\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "ld1 {v21.4s}, [%[bias_ptr]]\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "ld1 {v22.4s}, [x5]\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "ld1 {v23.4s}, [%[bias_ptr]]\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "ld1 {v24.4s}, [x5]\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "ld1 {v19.4s}, [%[bias_ptr]]\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "ld1 {v20.4s}, [x5]\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "ld1 {v25.4s}, [%[bias_ptr]]\n"
          "uaddw v18.8h, v28.8h, v18.8b\n"
          "ld1 {v26.4s}, [x5]\n"

          "blt " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1 "f\n"

          //"loop_%=:\n"
          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP ":\n"
            "smlal v21.4s, v0.4h, v9.4h\n"
            "subs w4, w4, #2\n"
            "smlal2 v22.4s, v0.8h, v9.8h\n"
            "ld1 {v9.8b}, [x2], %[input_depth]\n"
            "smlal v23.4s, v0.4h, v11.4h\n"
            "cmp w4, #2\n"
            "smlal2 v24.4s, v0.8h, v11.8h\n"
            "smlal v21.4s, v1.4h, v10.4h\n"
            "smlal2 v22.4s, v1.8h, v10.8h\n"
            "ld1 {v10.8b}, [x2], %[input_depth]\n"
            "smlal v23.4s, v1.4h, v12.4h\n"
            "smlal2 v24.4s, v1.8h, v12.8h\n"
            "smlal v21.4s, v2.4h, v11.4h\n"
            "smlal2 v22.4s, v2.8h, v11.8h\n"
            "ld1 {v11.8b}, [x2], %[input_depth]\n"
            "smlal v23.4s, v2.4h, v13.4h\n"
            "ld1 {v12.8b}, [x2], %[input_depth]\n"
            "smlal2 v24.4s, v2.8h, v13.8h\n"
            "ld1 {v13.8b}, [x2]\n"

            "smlal v21.4s, v3.4h, v14.4h\n"
            "smlal2 v22.4s, v3.8h, v14.8h\n"
            "ld1 {v14.8b}, [x0], %[input_depth]\n"
            "smlal v23.4s, v3.4h, v16.4h\n"
            "smlal2 v24.4s, v3.8h, v16.8h\n"
            "smlal v21.4s, v4.4h, v15.4h\n"
            "smlal2 v22.4s, v4.8h, v15.8h\n"
            "ld1 {v15.8b}, [x0], %[input_depth]\n"
            "smlal v23.4s, v4.4h, v17.4h\n"
            "smlal2 v24.4s, v4.8h, v17.8h\n"
            "smlal v21.4s, v5.4h, v16.4h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "smlal2 v22.4s, v5.8h, v16.8h\n"
            "ld1 {v16.8b}, [x0], %[input_depth]\n"
            "smlal v23.4s, v5.4h, v18.4h\n"
            "ld1 {v17.8b}, [x0], %[input_depth]\n"
            "smlal2 v24.4s, v5.8h, v18.8h\n"
            "ld1 {v18.8b}, [x0]\n"

            "smlal v21.4s, v6.4h, v9.4h\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "smlal2 v22.4s, v6.8h, v9.8h\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"
            "smlal v19.4s, v0.4h, v9.4h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal2 v20.4s, v0.8h, v9.8h\n"
            "ld1 {v9.8b}, [x1], %[input_depth]\n"
            "smlal v23.4s, v6.4h, v11.4h\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"
            "smlal2 v24.4s, v6.8h, v11.8h\n"
            "smlal v21.4s, v7.4h, v10.4h\n"
            "smlal2 v22.4s, v7.8h, v10.8h\n"
            "smlal v19.4s, v1.4h, v10.4h\n"
            "smlal2 v20.4s, v1.8h, v10.8h\n"
            "ld1 {v10.8b}, [x1], %[input_depth]\n"
            "smlal v23.4s, v7.4h, v12.4h\n"
            "smlal2 v24.4s, v7.8h, v12.8h\n"
            "smlal v25.4s, v1.4h, v12.4h\n"
            "smlal2 v26.4s, v1.8h, v12.8h\n"
            "smlal v21.4s, v8.4h, v11.4h\n"
            "smlal2 v22.4s, v8.8h, v11.8h\n"
            "smlal v19.4s, v2.4h, v11.4h\n"
            "add x6, x6, %[input_width_increment]\n"
            "smlal2 v20.4s, v2.8h, v11.8h\n"
            "mov x0, x6\n"

            "smlal v25.4s, v0.4h, v11.4h\n"
            "smlal2 v26.4s, v0.8h, v11.8h\n"
            "ld1 {v11.8b}, [x1], %[input_depth]\n"
            "smlal v23.4s, v8.4h, v13.4h\n"
            "ld1 {v12.8b}, [x1], %[input_depth]\n"
            "smlal2 v24.4s, v8.8h, v13.8h\n"
            "smlal v25.4s, v2.4h, v13.4h\n"
            "smlal2 v26.4s, v2.8h, v13.8h\n"
            "ld1 {v13.8b}, [x1]\n"
            "add x1, x0, %[input_row_size]\n"

            "dup v28.4s, w7\n"
            "add x2, x1, %[input_row_size]\n"
            "sqrdmulh v21.4s, v21.4s, v27.4s\n"
            "sqrdmulh v22.4s, v22.4s, v27.4s\n"
            "sqrdmulh v23.4s, v23.4s, v27.4s\n"
            "sqrdmulh v24.4s, v24.4s, v27.4s\n"
            "and v27.16b, v21.16b, v28.16b\n"
            "and v29.16b, v22.16b, v28.16b\n"
            "and v30.16b, v23.16b, v28.16b\n"
            "and v31.16b, v24.16b, v28.16b\n"
            "sshr v27.4s, v27.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v21.4s, v21.4s, v27.4s\n"
            "dup v27.4s, %w[output_multiplier]\n"
            "sqadd v22.4s, v22.4s, v29.4s\n"
            "dup v29.4s, %w[output_offset]\n"
            "sqadd v23.4s, v23.4s, v30.4s\n"
            "dup v30.4s, %w[output_activation_min]\n"
            "sqadd v24.4s, v24.4s, v31.4s\n"
            "dup v31.4s, %w[output_activation_max]\n"
            "srshl v21.4s, v21.4s, v28.4s\n"
            "srshl v22.4s, v22.4s, v28.4s\n"
            "srshl v23.4s, v23.4s, v28.4s\n"
            "srshl v24.4s, v24.4s, v28.4s\n"
            "dup v28.8h, %w[input_offset]\n"
            "add v21.4s, v21.4s, v29.4s\n"
            "add v22.4s, v22.4s, v29.4s\n"
            "add v23.4s, v23.4s, v29.4s\n"
            "add v24.4s, v24.4s, v29.4s\n"
            "smax v21.4s, v21.4s, v30.4s\n"
            "smax v22.4s, v22.4s, v30.4s\n"
            "smax v23.4s, v23.4s, v30.4s\n"
            "smax v24.4s, v24.4s, v30.4s\n"
            "smin v21.4s, v21.4s, v31.4s\n"
            "smin v22.4s, v22.4s, v31.4s\n"
            "smin v23.4s, v23.4s, v31.4s\n"
            "smin v24.4s, v24.4s, v31.4s\n"
            "sqxtn v21.4h, v21.4s\n"
            "sqxtn v23.4h, v23.4s\n"
            "sqxtn2 v21.8h, v22.4s\n"
            "ld1 {v22.4s}, [x5]\n"
            "sqxtn2 v23.8h, v24.4s\n"
            "ld1 {v24.4s}, [x5]\n"
            "sqxtun v21.8b, v21.8h\n"
            "sqxtun v23.8b, v23.8h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "st1 {v21.8b}, [x3], %[output_depth]\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "st1 {v23.8b}, [x3], %[output_depth]\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"

            "smlal v19.4s, v6.4h, v9.4h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal2 v20.4s, v6.8h, v9.8h\n"
            "ld1 {v9.8b}, [x0], %[input_depth]\n"
            "smlal v25.4s, v6.4h, v11.4h\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"
            "smlal2 v26.4s, v6.8h, v11.8h\n"
            "uaddw v14.8h, v28.8h, v14.8b\n"
            "smlal v19.4s, v7.4h, v10.4h\n"
            "uaddw v15.8h, v28.8h, v15.8b\n"
            "smlal2 v20.4s, v7.8h, v10.8h\n"
            "ld1 {v10.8b}, [x0], %[input_depth]\n"
            "smlal v25.4s, v7.4h, v12.4h\n"
            "uaddw v16.8h, v28.8h, v16.8b\n"
            "smlal2 v26.4s, v7.8h, v12.8h\n"
            "uaddw v17.8h, v28.8h, v17.8b\n"
            "smlal v19.4s, v8.4h, v11.4h\n"
            "uaddw v18.8h, v28.8h, v18.8b\n"
            "smlal2 v20.4s, v8.8h, v11.8h\n"
            "ld1 {v11.8b}, [x0], %[input_depth]\n"
            "smlal v25.4s, v8.4h, v13.4h\n"
            "ld1 {v12.8b}, [x0], %[input_depth]\n"
            "smlal2 v26.4s, v8.8h, v13.8h\n"
            "ld1 {v13.8b}, [x0]\n"
            "add x0, x2, %[input_row_size]\n"

            "smlal v19.4s, v3.4h, v14.4h\n"
            "smlal2 v20.4s, v3.8h, v14.8h\n"
            "ld1 {v14.8b}, [x1], %[input_depth]\n"
            "smlal v25.4s, v3.4h, v16.4h\n"
            "ld1 {v21.4s}, [%[bias_ptr]]\n"
            "smlal2 v26.4s, v3.8h, v16.8h\n"
            "ld1 {v23.4s}, [%[bias_ptr]]\n"
            "smlal v19.4s, v4.4h, v15.4h\n"
            "uaddw v9.8h, v28.8h, v9.8b\n"
            "smlal2 v20.4s, v4.8h, v15.8h\n"
            "ld1 {v15.8b}, [x1], %[input_depth]\n"
            "smlal v25.4s, v4.4h, v17.4h\n"
            "uaddw v10.8h, v28.8h, v10.8b\n"
            "smlal2 v26.4s, v4.8h, v17.8h\n"
            "uaddw v11.8h, v28.8h, v11.8b\n"
            "smlal v19.4s, v5.4h, v16.4h\n"
            "uaddw v12.8h, v28.8h, v12.8b\n"
            "smlal2 v20.4s, v5.8h, v16.8h\n"
            "ld1 {v16.8b}, [x1], %[input_depth]\n"
            "smlal v25.4s, v5.4h, v18.4h\n"
            "ld1 {v17.8b}, [x1], %[input_depth]\n"
            "smlal2 v26.4s, v5.8h, v18.8h\n"
            "ld1 {v18.8b}, [x1]\n"
            "add x1, x0, %[input_row_size]\n"
            "uaddw v13.8h, v28.8h, v13.8b\n"

            "dup v28.4s, w7\n"
            "sqrdmulh v19.4s, v19.4s, v27.4s\n"
            "sqrdmulh v20.4s, v20.4s, v27.4s\n"
            "sqrdmulh v25.4s, v25.4s, v27.4s\n"
            "sqrdmulh v26.4s, v26.4s, v27.4s\n"
            "and v27.16b, v19.16b, v28.16b\n"
            "and v29.16b, v20.16b, v28.16b\n"
            "and v30.16b, v25.16b, v28.16b\n"
            "and v31.16b, v26.16b, v28.16b\n"
            "sshr v27.4s, v27.4s, #31\n"
            "sshr v29.4s, v29.4s, #31\n"
            "sshr v30.4s, v30.4s, #31\n"
            "sshr v31.4s, v31.4s, #31\n"
            "sqadd v19.4s, v19.4s, v27.4s\n"
            "dup v27.4s, %w[output_multiplier]\n"
            "sqadd v20.4s, v20.4s, v29.4s\n"
            "dup v29.4s, %w[output_offset]\n"
            "sqadd v25.4s, v25.4s, v30.4s\n"
            "dup v30.4s, %w[output_activation_min]\n"
            "sqadd v26.4s, v26.4s, v31.4s\n"
            "dup v31.4s, %w[output_activation_max]\n"
            "srshl v19.4s, v19.4s, v28.4s\n"
            "srshl v20.4s, v20.4s, v28.4s\n"
            "srshl v25.4s, v25.4s, v28.4s\n"
            "srshl v26.4s, v26.4s, v28.4s\n"
            "dup v28.8h, %w[input_offset]\n"
            "add v19.4s, v19.4s, v29.4s\n"
            "add v20.4s, v20.4s, v29.4s\n"
            "add v25.4s, v25.4s, v29.4s\n"
            "add v26.4s, v26.4s, v29.4s\n"
            "smax v19.4s, v19.4s, v30.4s\n"
            "smax v20.4s, v20.4s, v30.4s\n"
            "smax v25.4s, v25.4s, v30.4s\n"
            "smax v26.4s, v26.4s, v30.4s\n"
            "smin v19.4s, v19.4s, v31.4s\n"
            "smin v20.4s, v20.4s, v31.4s\n"
            "smin v25.4s, v25.4s, v31.4s\n"
            "smin v26.4s, v26.4s, v31.4s\n"
            "sqxtn v19.4h, v19.4s\n"
            "sqxtn v25.4h, v25.4s\n"
            "sqxtn2 v19.8h, v20.4s\n"
            "ld1 {v20.4s}, [x5]\n"
            "sqxtn2 v25.8h, v26.4s\n"
            "ld1 {v26.4s}, [x5]\n"
            "sqxtun v19.8b, v19.8h\n"
            "sqxtun v25.8b, v25.8h\n"
            "uaddw v14.8h, v28.8h, v14.8b\n"
            "st1 {v19.8b}, [x10], %[output_depth]\n"
            "uaddw v15.8h, v28.8h, v15.8b\n"
            "st1 {v25.8b}, [x10], %[output_depth]\n"
            "uaddw v16.8h, v28.8h, v16.8b\n"
            "uaddw v17.8h, v28.8h, v17.8b\n"
            "ld1 {v19.4s}, [%[bias_ptr]]\n"
            "uaddw v18.8h, v28.8h, v18.8b\n"
            "ld1 {v25.4s}, [%[bias_ptr]]\n"

            "bge " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP "b\n"

          "cmp w4, #1\n"
          "blt " DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP "f\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1 ":\n"
          // Registers v9, v10, v11, v14, v15, and v16 have already been loaded
          // with the correct values at this point. This corresponds to the
          // first two input rows of the top left output. Now load the last
          // input row for this output. Once these inputs are no longer needed,
          // load the input rows for the bottom left output.
          "ld1 {v12.8b}, [x2], %[input_depth]\n"
          "smlal v21.4s, v0.4h, v9.4h\n"
          "ld1 {v13.8b}, [x2], %[input_depth]\n"
          "smlal2 v22.4s, v0.8h, v9.8h\n"
          "ld1 {v17.8b}, [x2]\n"
          "smlal v21.4s, v1.4h, v10.4h\n"
          "ld1 {v9.8b}, [x0], %[input_depth]\n"
          "smlal2 v22.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x0], %[input_depth]\n"
          "smlal v21.4s, v2.4h, v11.4h\n"
          "smlal2 v22.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x0]\n"
          "smlal v21.4s, v3.4h, v14.4h\n"
          "smlal2 v22.4s, v3.8h, v14.8h\n"
          "ld1 {v14.8b}, [x1], %[input_depth]\n"
          "smlal v21.4s, v4.4h, v15.4h\n"
          "smlal2 v22.4s, v4.8h, v15.8h\n"
          "ld1 {v15.8b}, [x1], %[input_depth]\n"
          "smlal v21.4s, v5.4h, v16.4h\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "smlal2 v22.4s, v5.8h, v16.8h\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "ld1 {v16.8b}, [x1]\n"

          "smlal v21.4s, v6.4h, v12.4h\n"
          "smlal2 v22.4s, v6.8h, v12.8h\n"
          "smlal v23.4s, v0.4h, v12.4h\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "smlal2 v24.4s, v0.8h, v12.8h\n"
          "smlal v21.4s, v7.4h, v13.4h\n"
          "smlal2 v22.4s, v7.8h, v13.8h\n"
          "smlal v23.4s, v1.4h, v13.4h\n"
          "smlal2 v24.4s, v1.8h, v13.8h\n"
          "smlal v21.4s, v8.4h, v17.4h\n"
          "smlal2 v22.4s, v8.8h, v17.8h\n"
          "smlal v23.4s, v2.4h, v17.4h\n"
          "smlal2 v24.4s, v2.8h, v17.8h\n"

          "dup v26.4s, w7\n"
          "sqrdmulh v21.4s, v21.4s, v27.4s\n"
          "sqrdmulh v22.4s, v22.4s, v27.4s\n"
          "and v18.16b, v21.16b, v26.16b\n"
          "and v19.16b, v22.16b, v26.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v21.4s, v21.4s, v18.4s\n"
          "sqadd v22.4s, v22.4s, v19.4s\n"
          "srshl v21.4s, v21.4s, v26.4s\n"
          "srshl v22.4s, v22.4s, v26.4s\n"
          "add v21.4s, v21.4s, v29.4s\n"
          "add v22.4s, v22.4s, v29.4s\n"
          "smax v21.4s, v21.4s, v30.4s\n"
          "smax v22.4s, v22.4s, v30.4s\n"
          "smin v21.4s, v21.4s, v31.4s\n"
          "smin v22.4s, v22.4s, v31.4s\n"
          "sqxtn v21.4h, v21.4s\n"
          "sqxtn2 v21.8h, v22.4s\n"
          "sqxtun v21.8b, v21.8h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "st1 {v21.8b}, [x3]\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"

          "smlal v23.4s, v3.4h, v9.4h\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "smlal2 v24.4s, v3.8h, v9.8h\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "smlal v23.4s, v4.4h, v10.4h\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "smlal2 v24.4s, v4.8h, v10.8h\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "smlal v23.4s, v5.4h, v11.4h\n"
          "smlal2 v24.4s, v5.8h, v11.8h\n"

          "smlal v23.4s, v6.4h, v14.4h\n"
          "smlal2 v24.4s, v6.8h, v14.8h\n"
          "smlal v23.4s, v7.4h, v15.4h\n"
          "smlal2 v24.4s, v7.8h, v15.8h\n"
          "smlal v23.4s, v8.4h, v16.4h\n"
          "smlal2 v24.4s, v8.8h, v16.8h\n"

          "sqrdmulh v23.4s, v23.4s, v27.4s\n"
          "sqrdmulh v24.4s, v24.4s, v27.4s\n"
          "and v18.16b, v23.16b, v26.16b\n"
          "and v19.16b, v24.16b, v26.16b\n"
          "sshr v18.4s, v18.4s, #31\n"
          "sshr v19.4s, v19.4s, #31\n"
          "sqadd v23.4s, v23.4s, v18.4s\n"
          "sqadd v24.4s, v24.4s, v19.4s\n"
          "srshl v23.4s, v23.4s, v26.4s\n"
          "srshl v24.4s, v24.4s, v26.4s\n"
          "add v23.4s, v23.4s, v29.4s\n"
          "add v24.4s, v24.4s, v29.4s\n"
          "smax v23.4s, v23.4s, v30.4s\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smin v23.4s, v23.4s, v31.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "sqxtn v23.4h, v23.4s\n"
          "sqxtn2 v23.8h, v24.4s\n"
          "sqxtun v23.8b, v23.8h\n"
          "st1 {v23.8b}, [x10]\n"

          DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP ":\n"
          "subs %w[output_window_height], %w[output_window_height], #2\n"
          "add %[input_ptr], %[input_ptr], %[input_height_increment]\n"
          "cmp %w[output_window_height], #2\n"
          "add %[output_ptr], %[output_ptr], %[output_height_increment]\n"
          "bge " DEPTHWISECONV_LABEL_HEIGHT_2_LOOP "b\n"

        DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP ":\n"
        "cmp %w[output_window_height], #1\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        DEPTHWISECONV_LABEL_HEIGHT_1 ":\n"
        "mov x6, %[input_ptr]\n"
        "mov x0, x6\n"
        "add x1, x0, %[input_row_size]\n"
        "ld1 {v9.8b}, [x0], %[input_depth]\n"
        "add x2, x1, %[input_row_size]\n"
        "ld1 {v10.8b}, [x0], %[input_depth]\n"
        "mov x3, %[output_ptr]\n"
        "ld1 {v11.8b}, [x0], %[input_depth]\n"
        "mov w4, %w[output_window_width]\n"
        "ld1 {v18.8b}, [x0], %[input_depth]\n"
        "cmp w4, #2\n"
        "ld1 {v19.8b}, [x0]\n"
        "ld1 {v12.8b}, [x1], %[input_depth]\n"
        "ld1 {v13.8b}, [x1], %[input_depth]\n"
        "ld1 {v14.8b}, [x1], %[input_depth]\n"
        "ld1 {v20.8b}, [x1], %[input_depth]\n"
        "ld1 {v21.8b}, [x1]\n"
        "ld1 {v15.8b}, [x2], %[input_depth]\n"
        "ld1 {v16.8b}, [x2], %[input_depth]\n"
        "ld1 {v17.8b}, [x2], %[input_depth]\n"
        "ld1 {v22.8b}, [x2], %[input_depth]\n"
        "ld1 {v23.8b}, [x2]\n"

        "uaddw v9.8h, v28.8h, v9.8b\n"
        "ld1 {v24.4s}, [%[bias_ptr]]\n"
        "uaddw v10.8h, v28.8h, v10.8b\n"
        "ld1 {v25.4s}, [x5]\n"
        "uaddw v11.8h, v28.8h, v11.8b\n"
        "ld1 {v26.4s}, [%[bias_ptr]]\n"
        "uaddw v18.8h, v28.8h, v18.8b\n"
        "ld1 {v27.4s}, [x5]\n"
        "uaddw v19.8h, v28.8h, v19.8b\n"
        "uaddw v12.8h, v28.8h, v12.8b\n"
        "uaddw v13.8h, v28.8h, v13.8b\n"
        "uaddw v14.8h, v28.8h, v14.8b\n"
        "uaddw v20.8h, v28.8h, v20.8b\n"
        "uaddw v21.8h, v28.8h, v21.8b\n"
        "uaddw v15.8h, v28.8h, v15.8b\n"
        "uaddw v16.8h, v28.8h, v16.8b\n"
        "uaddw v17.8h, v28.8h, v17.8b\n"
        "uaddw v22.8h, v28.8h, v22.8b\n"
        "uaddw v23.8h, v28.8h, v23.8b\n"

        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1 "f\n"

        //"loop_%=:\n"
        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP ":\n"
          "add x6, x6, %[input_width_increment]\n"
          "smlal v24.4s, v0.4h, v9.4h\n"
          "mov x0, x6\n"
          "add x1, x0, %[input_row_size]\n"
          "smlal2 v25.4s, v0.8h, v9.8h\n"
          "ld1 {v9.8b}, [x0], %[input_depth]\n"
          "smlal v26.4s, v0.4h, v11.4h\n"
          "add x2, x1, %[input_row_size]\n"
          "smlal2 v27.4s, v0.8h, v11.8h\n"
          "subs w4, w4, #2\n"
          "smlal v24.4s, v1.4h, v10.4h\n"
          "cmp w4, #2\n"
          "smlal2 v25.4s, v1.8h, v10.8h\n"
          "ld1 {v10.8b}, [x0], %[input_depth]\n"
          "smlal v26.4s, v1.4h, v18.4h\n"
          "smlal2 v27.4s, v1.8h, v18.8h\n"
          "smlal v24.4s, v2.4h, v11.4h\n"
          "smlal2 v25.4s, v2.8h, v11.8h\n"
          "ld1 {v11.8b}, [x0], %[input_depth]\n"
          "smlal v26.4s, v2.4h, v19.4h\n"
          "ld1 {v18.8b}, [x0], %[input_depth]\n"
          "smlal2 v27.4s, v2.8h, v19.8h\n"
          "ld1 {v19.8b}, [x0], %[input_depth]\n"
          "smlal v24.4s, v3.4h, v12.4h\n"
          "smlal2 v25.4s, v3.8h, v12.8h\n"
          "ld1 {v12.8b}, [x1], %[input_depth]\n"
          "smlal v26.4s, v3.4h, v14.4h\n"
          "smlal2 v27.4s, v3.8h, v14.8h\n"
          "smlal v24.4s, v4.4h, v13.4h\n"
          "smlal2 v25.4s, v4.8h, v13.8h\n"
          "ld1 {v13.8b}, [x1], %[input_depth]\n"
          "smlal v26.4s, v4.4h, v20.4h\n"
          "smlal2 v27.4s, v4.8h, v20.8h\n"
          "smlal v24.4s, v5.4h, v14.4h\n"
          "smlal2 v25.4s, v5.8h, v14.8h\n"
          "ld1 {v14.8b}, [x1], %[input_depth]\n"
          "smlal v26.4s, v5.4h, v21.4h\n"
          "ld1 {v20.8b}, [x1], %[input_depth]\n"
          "smlal2 v27.4s, v5.8h, v21.8h\n"
          "ld1 {v21.8b}, [x1], %[input_depth]\n"
          "smlal v24.4s, v6.4h, v15.4h\n"
          "smlal2 v25.4s, v6.8h, v15.8h\n"
          "ld1 {v15.8b}, [x2], %[input_depth]\n"
          "smlal v26.4s, v6.4h, v17.4h\n"
          "smlal2 v27.4s, v6.8h, v17.8h\n"
          "smlal v24.4s, v7.4h, v16.4h\n"
          "smlal2 v25.4s, v7.8h, v16.8h\n"
          "ld1 {v16.8b}, [x2], %[input_depth]\n"
          "smlal v26.4s, v7.4h, v22.4h\n"
          "smlal2 v27.4s, v7.8h, v22.8h\n"
          "smlal v24.4s, v8.4h, v17.4h\n"
          "smlal2 v25.4s, v8.8h, v17.8h\n"
          "ld1 {v17.8b}, [x2], %[input_depth]\n"
          "smlal v26.4s, v8.4h, v23.4h\n"
          "ld1 {v22.8b}, [x2], %[input_depth]\n"
          "smlal2 v27.4s, v8.8h, v23.8h\n"
          "ld1 {v23.8b}, [x2], %[input_depth]\n"

          "dup v28.4s, %w[output_multiplier]\n"
          "dup v29.4s, w7\n"
          "sqrdmulh v24.4s, v24.4s, v28.4s\n"
          "sqrdmulh v25.4s, v25.4s, v28.4s\n"
          "sqrdmulh v26.4s, v26.4s, v28.4s\n"
          "sqrdmulh v27.4s, v27.4s, v28.4s\n"
          "dup v28.4s, %w[output_offset]\n"
          "and v30.16b, v24.16b, v29.16b\n"
          "and v31.16b, v25.16b, v29.16b\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v24.4s, v24.4s, v30.4s\n"
          "sqadd v25.4s, v25.4s, v31.4s\n"
          "and v30.16b, v26.16b, v29.16b\n"
          "and v31.16b, v27.16b, v29.16b\n"
          "sshr v30.4s, v30.4s, #31\n"
          "sshr v31.4s, v31.4s, #31\n"
          "sqadd v26.4s, v26.4s, v30.4s\n"
          "dup v30.4s, %w[output_activation_min]\n"
          "sqadd v27.4s, v27.4s, v31.4s\n"
          "dup v31.4s, %w[output_activation_max]\n"
          "srshl v24.4s, v24.4s, v29.4s\n"
          "srshl v25.4s, v25.4s, v29.4s\n"
          "srshl v26.4s, v26.4s, v29.4s\n"
          "srshl v27.4s, v27.4s, v29.4s\n"
          "add v24.4s, v24.4s, v28.4s\n"
          "add v25.4s, v25.4s, v28.4s\n"
          "add v26.4s, v26.4s, v28.4s\n"
          "add v27.4s, v27.4s, v28.4s\n"
          "dup v28.8h, %w[input_offset]\n"
          "smax v24.4s, v24.4s, v30.4s\n"
          "smax v25.4s, v25.4s, v30.4s\n"
          "smax v26.4s, v26.4s, v30.4s\n"
          "smax v27.4s, v27.4s, v30.4s\n"
          "smin v24.4s, v24.4s, v31.4s\n"
          "smin v25.4s, v25.4s, v31.4s\n"
          "smin v26.4s, v26.4s, v31.4s\n"
          "smin v27.4s, v27.4s, v31.4s\n"
          "sqxtn v24.4h, v24.4s\n"
          "sqxtn v26.4h, v26.4s\n"
          "sqxtn2 v24.8h, v25.4s\n"
          "ld1 {v25.4s}, [x5]\n"
          "sqxtn2 v26.8h, v27.4s\n"
          "ld1 {v27.4s}, [x5]\n"
          "sqxtun v24.8b, v24.8h\n"
          "sqxtun v26.8b, v26.8h\n"
          "uaddw v9.8h, v28.8h, v9.8b\n"
          "st1 {v24.8b}, [x3], %[output_depth]\n"
          "uaddw v10.8h, v28.8h, v10.8b\n"
          "st1 {v26.8b}, [x3], %[output_depth]\n"
          "uaddw v11.8h, v28.8h, v11.8b\n"
          "uaddw v18.8h, v28.8h, v18.8b\n"
          "uaddw v19.8h, v28.8h, v19.8b\n"
          "uaddw v12.8h, v28.8h, v12.8b\n"
          "uaddw v13.8h, v28.8h, v13.8b\n"
          "uaddw v14.8h, v28.8h, v14.8b\n"
          "uaddw v20.8h, v28.8h, v20.8b\n"
          "uaddw v21.8h, v28.8h, v21.8b\n"
          "ld1 {v24.4s}, [%[bias_ptr]]\n"
          "uaddw v15.8h, v28.8h, v15.8b\n"
          "ld1 {v26.4s}, [%[bias_ptr]]\n"
          "uaddw v16.8h, v28.8h, v16.8b\n"
          "uaddw v17.8h, v28.8h, v17.8b\n"
          "uaddw v22.8h, v28.8h, v22.8b\n"
          "uaddw v23.8h, v28.8h, v23.8b\n"

          "bge " DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP "b\n"

        "cmp w4, #1\n"
        "blt " DEPTHWISECONV_LABEL_HEIGHT_1_END "f\n"

        DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1 ":\n"
        "dup v26.4s, w7\n"
        "dup v27.4s, %w[output_multiplier]\n"
        "dup v29.4s, %w[output_offset]\n"

        "smlal v24.4s, v0.4h, v9.4h\n"
        "smlal2 v25.4s, v0.8h, v9.8h\n"
        "smlal v24.4s, v1.4h, v10.4h\n"
        "smlal2 v25.4s, v1.8h, v10.8h\n"
        "smlal v24.4s, v2.4h, v11.4h\n"
        "smlal2 v25.4s, v2.8h, v11.8h\n"
        "smlal v24.4s, v3.4h, v12.4h\n"
        "smlal2 v25.4s, v3.8h, v12.8h\n"
        "smlal v24.4s, v4.4h, v13.4h\n"
        "smlal2 v25.4s, v4.8h, v13.8h\n"
        "smlal v24.4s, v5.4h, v14.4h\n"
        "smlal2 v25.4s, v5.8h, v14.8h\n"
        "smlal v24.4s, v6.4h, v15.4h\n"
        "smlal2 v25.4s, v6.8h, v15.8h\n"
        "smlal v24.4s, v7.4h, v16.4h\n"
        "smlal2 v25.4s, v7.8h, v16.8h\n"
        "smlal v24.4s, v8.4h, v17.4h\n"
        "smlal2 v25.4s, v8.8h, v17.8h\n"

        "sqrdmulh v24.4s, v24.4s, v27.4s\n"
        "sqrdmulh v25.4s, v25.4s, v27.4s\n"
        "and v18.16b, v24.16b, v26.16b\n"
        "and v19.16b, v25.16b, v26.16b\n"
        "sshr v18.4s, v18.4s, #31\n"
        "sshr v19.4s, v19.4s, #31\n"
        "sqadd v24.4s, v24.4s, v18.4s\n"
        "sqadd v25.4s, v25.4s, v19.4s\n"
        "srshl v24.4s, v24.4s, v26.4s\n"
        "srshl v25.4s, v25.4s, v26.4s\n"
        "add v24.4s, v24.4s, v29.4s\n"
        "add v25.4s, v25.4s, v29.4s\n"
        "smax v24.4s, v24.4s, v30.4s\n"
        "smax v25.4s, v25.4s, v30.4s\n"
        "smin v24.4s, v24.4s, v31.4s\n"
        "smin v25.4s, v25.4s, v31.4s\n"
        "sqxtn v24.4h, v24.4s\n"
        "sqxtn2 v24.8h, v25.4s\n"
        "sqxtun v24.8b, v24.8h\n"
        "st1 {v24.8b}, [x3]\n"

        DEPTHWISECONV_LABEL_HEIGHT_1_END ":\n"
    :
    // Outputs.
    [filter_ptr] "+r"(filter_ptr), [input_ptr] "+r"(input_ptr),
    [output_ptr] "+r"(output_ptr),
    [output_window_height] "+r"(output_window_height)
    :
    // Inputs.
    [bias_ptr] "r"(bias_ptr), [output_depth] "r"(output_depth),
    [filter_offset] "r"(filter_offset), [input_row_size] "r"(input_row_size),
    [input_depth] "r"(input_depth), [input_offset] "r"(input_offset),
    [output_multiplier] "r"(output_multiplier),
    [output_shift] "r"(output_shift), [output_offset] "r"(output_offset),
    [output_activation_min] "r"(output_activation_min),
    [output_activation_max] "r"(output_activation_max),
    [output_window_width] "r"(output_window_width),
    [input_width_increment] "r"(input_width_increment),
    [input_height_increment] "r"(input_height_increment),
    [output_height_increment] "r"(output_height_increment),
    [output_row_size] "r"(output_row_size)
    :
    // Clobbers.
    // We use these NEON registers.
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
    "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
    "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
    // We use these general-purpose registers.
    "x0", "x1", "x2", "x3", "w4", "x5", "x6", "w7", "x10");
#undef DEPTHWISECONV_LABEL_HEIGHT_1_END
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_1
#undef DEPTHWISECONV_LABEL_HEIGHT_1_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_1
#undef DEPTHWISECONV_LABEL_HEIGHT_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_AFTER_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_1
#undef DEPTHWISECONV_LABEL_HEIGHT_2_WIDTH_2_LOOP
#undef DEPTHWISECONV_LABEL_HEIGHT_2_LOOP
  }
};

// Copies a subset of the input designated by |input_ptr| into |output_ptr|
// with the specified output dimensions. Supports output depths of 64 only as
// this is the cache line size.
inline void ShuffleInput(const uint8* input_ptr, int64_t input_depth,
                         int input_width, int input_height,
                         int64_t output_depth, int output_width,
                         int output_height, uint8* output_ptr) {
  const int64_t input_row_size = input_depth * input_width;
  for (int y = 0; y < output_height; y++) {
    const uint8* ptr = input_ptr;
    for (int x = 0; x < output_width; x++) {
      memcpy(output_ptr, ptr, output_depth);
      output_ptr += output_depth;
      ptr += input_depth;
    }
    input_ptr += input_row_size;
  }
}

template <int kOutputRows, int kShuffleOutputHeight, int kShuffleOutputWidth,
    int kStrideWidth, int kStrideHeight>
struct DepthwiseConvMultiRow {
 public:
  constexpr static int kShuffleInputHeight =
      kStrideHeight * (kShuffleOutputHeight - 1) + 3;
  constexpr static int kShuffleInputWidth =
      kStrideWidth * (kShuffleOutputWidth - 1) + 3;

  static inline void Run(const uint8* input_data, int start_x, int start_y,
                         int64_t input_depth, int input_width, int input_height,
                         int64_t input_row_size, int32 input_offset,
                         const uint8* filter_data, int32 filter_offset,
                         const int32* bias_data, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         int64_t output_depth, int output_width,
                         uint8* shuffle_workspace) {
    // Make sure shuffle parameters fall within the allowed workspace size.
    static_assert(64 * kShuffleInputWidth * kShuffleInputHeight <=
                  DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE,
                  "Shuffle workspace size is too large.");

    // Although it is possible to have kOutputRows != kShuffleOutputHeight, the
    // below code assumes that they are the same.
    static_assert(kOutputRows == kShuffleOutputHeight,
                  "Output heights that are not equal to the shuffle output "
                  "height are not supported.");

    int out_x = start_x;
    // Run shuffling on inputs with sufficiently large depth and width. When
    // these parameters are large enough, more time is taken to load inputs from
    // memory. At this point, it becomes useful to prefetch and preshuffle the
    // input data to maximize locality.
    if (output_depth > 64 || (output_depth <= 64 && input_width > 150)) {
      for (; out_x <= output_width - kShuffleOutputWidth;
             out_x += kShuffleOutputWidth) {
        const uint8* input_ptr = input_data;
        const int32* bias_ptr = bias_data;
        const uint8* filter_ptr = filter_data;
        uint8* output_ptr = output_data;
        int64_t depth = 0;
        for (; depth <= output_depth - 64; depth += 64) {
          // Preload.
          const uint8* h_ptr = input_ptr;
          for (int i = 0; i < kShuffleInputHeight; i++) {
            const uint8* ptr = h_ptr;
            for (int j = 0; j < kShuffleInputWidth; j++) {
              asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
              ptr += input_depth;
            }
            h_ptr += input_row_size;
          }

          // For a large enough input, shuffle into 64 x kShuffleInputWidth x
          // kShuffleInputHeight buckets.
          ShuffleInput(input_ptr, input_depth, input_width, input_height, 64,
                       kShuffleInputWidth, kShuffleInputHeight,
                       shuffle_workspace);
          const uint8* shuffled_ptr = shuffle_workspace;

          for (int micro_depth = 0; micro_depth <= 64 - 8; micro_depth += 8) {
            DepthwiseConvWindow<8, kStrideWidth, kStrideHeight>::Run(
                shuffled_ptr, 64, input_offset, 64 * kShuffleInputWidth,
                filter_ptr, filter_offset, bias_ptr, output_offset,
                output_multiplier, output_shift, output_activation_min,
                output_activation_max, output_ptr, output_depth, output_width,
               kShuffleOutputHeight, kShuffleOutputWidth);

            shuffled_ptr += 8;
            output_ptr += 8;
            filter_ptr += 8;
            bias_ptr += 8;
          }
          input_ptr += 64;
        }

        // Preload.
        const uint8* h_ptr = input_ptr;
        for (int i = 0; i < kShuffleInputHeight; i++) {
          const uint8* ptr = h_ptr;
          for (int j = 0; j < kShuffleInputWidth; j++) {
            asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
            ptr += input_depth;
          }
          h_ptr += input_row_size;
        }

        // Handle leftover depth.
        for (; depth <= output_depth - 8; depth += 8) {
          DepthwiseConvWindow<8, kStrideWidth, kStrideHeight>::Run(input_ptr,
              input_depth, input_offset, input_row_size, filter_ptr,
              filter_offset, bias_ptr, output_offset, output_multiplier,
              output_shift, output_activation_min, output_activation_max,
              output_ptr, output_depth, output_width, kShuffleOutputHeight,
              kShuffleOutputWidth);

          input_ptr += 8;
          output_ptr += 8;
          filter_ptr += 8;
          bias_ptr += 8;
        }

        input_data += kShuffleOutputWidth * kStrideWidth * input_depth;
        output_data += kShuffleOutputWidth * output_depth;
      }
    }

    const int output_leftover_width = output_width - out_x;
    if (output_leftover_width > 0) {
      const int32* bias_ptr = bias_data;
      const uint8* filter_ptr = filter_data;
      const uint8* input_ptr = input_data;
      uint8* output_ptr = output_data;

      for (int64_t depth = 0; depth <= output_depth - 8; depth += 8) {
        DepthwiseConvWindow<8, kStrideWidth, kStrideHeight>::Run(input_ptr,
            input_depth, input_offset, input_row_size, filter_ptr,
            filter_offset, bias_ptr, output_offset, output_multiplier,
            output_shift, output_activation_min, output_activation_max,
            output_ptr, output_depth, output_width, kShuffleOutputHeight,
            output_leftover_width);

        input_ptr += 8;
        output_ptr += 8;
        filter_ptr += 8;
        bias_ptr += 8;
      }
    }
  }
};

inline bool Fast3x3FilterKernelSupported(const Dims<4>& input_dims,
                                         const Dims<4>& filter_dims,
                                         int stride_width, int stride_height,
                                         int pad_width, int pad_height,
                                         int depth_multiplier,
                                         const Dims<4>& output_dims) {
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  bool supported = filter_width == 3 && filter_height == 3 &&
                   depth_multiplier == 1 &&
                   (stride_width == 1 || stride_width == 2) &&
                   (stride_height == 1 || stride_height == 2) &&
                   (stride_width == stride_height) && pad_width == 0 &&
                   pad_height == 0 && (input_depth % 8) == 0;

  if (!supported) {
    return false;
  }

  // Handle case where padding is zero but padding type is not kValid.
  // This would require special boundary case handling that is not supported.

  const int out_x = output_width - 1;
  const int out_y = output_height - 1;

  const int in_x_origin = (out_x * stride_width) - pad_width;
  const int in_y_origin = (out_y * stride_height) - pad_height;

  const int in_x_end = in_x_origin + filter_width;
  const int in_y_end = in_y_origin + filter_height;

  // Supported only if filter on the right and bottom boundary lies completely
  // within the input.
  return in_x_end <= input_width && in_y_end <= input_height;
}

inline void DepthwiseConv3x3Filter(
    const uint8* input_data, const Dims<4>& input_dims, int32 input_offset,
    const uint8* filter_data, const Dims<4>& filter_dims, int32 filter_offset,
    const int32* bias_data, const Dims<4>& bias_dims, int stride_width,
    int stride_height, int pad_width, int pad_height, int depth_multiplier,
    int32 output_offset, int32 output_multiplier, int output_shift,
    int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  // 64-bit is used for types that will be added to 64-bit addresses in asm.
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int64_t output_depth =
      MatchingArraySize(filter_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int64_t input_depth = ArraySize(input_dims, 0);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  // Algorithm assumes below constraints. It is optimized for depth multiplier
  // of 1, 3x3 filter, no padding and strides 1 and 2.
  TFLITE_DCHECK(output_depth == input_depth * depth_multiplier);
  TFLITE_DCHECK(depth_multiplier == 1);
  TFLITE_DCHECK(filter_height == 3);
  TFLITE_DCHECK(filter_width == 3);
  TFLITE_DCHECK(pad_height == 0);
  TFLITE_DCHECK(pad_width == 0);
  TFLITE_DCHECK(stride_height == 1 || stride_height == 2);
  TFLITE_DCHECK(stride_width == 1 || stride_width == 2);
  TFLITE_DCHECK(stride_width == stride_height);

  const int64_t input_row_size = input_depth * (input_width + 2 * pad_width);
  const int64_t output_row_size = output_depth * output_width;
  const int64_t input_batch_size =
      input_row_size * (input_height + 2 * pad_height);
  const int64_t output_batch_size = output_depth * output_width * output_height;

  using conv_row_func_t = decltype(&DepthwiseConvMultiRow<1, 1, 1, 1, 1>::Run);
  conv_row_func_t conv_1_output_row, conv_2_output_rows, conv_4_output_rows,
      conv_8_output_rows;

  int conv_2_shuffle_input_width = 0;
  int conv_4_shuffle_input_width = 0;

  if (stride_width == 1) {
    conv_1_output_row = DepthwiseConvMultiRow<1, 1, 30, 1, 1>::Run;
    conv_2_output_rows = DepthwiseConvMultiRow<2, 2, 22, 1, 1>::Run;
    conv_4_output_rows = DepthwiseConvMultiRow<4, 4, 14, 1, 1>::Run;
    conv_8_output_rows = DepthwiseConvMultiRow<8, 8, 8, 1, 1>::Run;

    conv_2_shuffle_input_width =
        DepthwiseConvMultiRow<2, 2, 22, 1, 1>::kShuffleInputWidth;
    conv_4_shuffle_input_width =
        DepthwiseConvMultiRow<4, 4, 14, 1, 1>::kShuffleInputWidth;

  } else {
    conv_1_output_row = DepthwiseConvMultiRow<1, 1, 14, 2, 2>::Run;
    conv_2_output_rows = DepthwiseConvMultiRow<2, 2, 8, 2, 2>::Run;
    conv_4_output_rows = DepthwiseConvMultiRow<4, 4, 4, 2, 2>::Run;
    conv_8_output_rows = DepthwiseConvMultiRow<8, 8, 2, 2, 2>::Run;

    conv_2_shuffle_input_width =
        DepthwiseConvMultiRow<2, 2, 8, 2, 2>::kShuffleInputWidth;
    conv_4_shuffle_input_width =
        DepthwiseConvMultiRow<4, 4, 4, 2, 2>::kShuffleInputWidth;
  }

  // Allocate maximum memory needed for shuffled input.
  // TODO(mariewhite): The size of this workspace is small enough to be
  // allocated on the stack. Eventually we will want to move it to the heap
  // and have it allocated outside of this function, like the im2col_array used
  // in gemmlowp.
  uint8 shuffle_workspace[DEPTHWISECONV_SHUFFLE_WORKSPACE_SIZE];

  for (int b = 0; b < batches; ++b) {
    const uint8* input_ptr = input_data + b * input_batch_size;
    uint8* output_ptr = output_data + b * output_batch_size;

    int out_y = 0;

    // Shuffling shapes that maximize width over the shuffle workspace size
    // perform better since the inputs are closer together, minimizing shuffling
    // time.
    //
    // If the input shape has width large enough for the 2 height kernels
    // |conv_2_output_rows|, we prefer to use this. The innermost loop of the
    // kernels handle 2 height x 2 width so this is the fastest path.
    //
    // If the input shape has smaller width but larger height, shuffling is
    // still useful and can benefit from kernels |conv_4_output_rows| and
    // |conv_8_output_rows|.

    // Handle 8 rows at a time.
    if (input_width < conv_4_shuffle_input_width) {
      for (; out_y <= output_height - 8; out_y += 8) {
        conv_8_output_rows(input_ptr, 0, out_y, input_depth, input_width,
                           input_height, input_row_size, input_offset,
                           filter_data, filter_offset, bias_data,
                           output_offset, output_multiplier, output_shift,
                           output_activation_min, output_activation_max,
                           output_ptr, output_depth, output_width,
                           shuffle_workspace);

        input_ptr += 8 * stride_height * input_row_size;
        output_ptr += 8 * output_row_size;
      }
    }

    // Handle 4 rows at a time.
    if (input_width < conv_2_shuffle_input_width) {
      for (; out_y <= output_height - 4; out_y += 4) {
        conv_4_output_rows(input_ptr, 0, out_y, input_depth, input_width,
                           input_height, input_row_size, input_offset,
                           filter_data, filter_offset, bias_data,
                           output_offset, output_multiplier, output_shift,
                           output_activation_min, output_activation_max,
                           output_ptr, output_depth, output_width,
                           shuffle_workspace);

        input_ptr += 4 * stride_height * input_row_size;
        output_ptr += 4 * output_row_size;
      }
    }

    // Handle 2 rows at a time.
    for (; out_y <= output_height - 2; out_y += 2) {
      conv_2_output_rows(input_ptr, 0, out_y, input_depth, input_width,
                         input_height, input_row_size, input_offset,
                         filter_data, filter_offset, bias_data, output_offset,
                         output_multiplier, output_shift, output_activation_min,
                         output_activation_max, output_ptr, output_depth,
                         output_width, shuffle_workspace);

      input_ptr += 2 * stride_height * input_row_size;
      output_ptr += 2 * output_row_size;
    }

    // Handle one row at a time.
    for (; out_y < output_height; out_y++) {
      conv_1_output_row(input_ptr, 0, out_y, input_depth, input_width,
                        input_height, input_row_size, input_offset, filter_data,
                        filter_offset, bias_data, output_offset,
                        output_multiplier, output_shift, output_activation_min,
                        output_activation_max, output_ptr, output_depth,
                        output_width, shuffle_workspace);

      input_ptr += stride_height * input_row_size;
      output_ptr += output_row_size;
    }
  }
}
// clang-format on

#endif  // __aarch64__

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_3X3_FILTER_H_
