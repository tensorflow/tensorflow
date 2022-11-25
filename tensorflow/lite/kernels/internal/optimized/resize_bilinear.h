/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RESIZE_BILINEAR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RESIZE_BILINEAR_H_

#include <stdint.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {
namespace resize_bilinear {

#ifdef USE_NEON
// These utility functions are split off not just for convenience. Most
// incoporate packing or unpacking of data.
//
// (a) Optimizations can be tried experimentally.
// (b) Optimizations can be specialized for architectures, eg Intel vs ARM.

inline int16x8_t Load8IntoLowerS16(const uint8* data_ptr) {
  return vreinterpretq_s16_u16(vmovl_u8(vld1_u8(data_ptr)));
}

inline uint16x8_t Move8IntoUpperU16(const uint8x8_t vec_val) {
  // Alternatively one could zip with a zero vector.
  return vshlq_n_u16(vmovl_u8(vec_val), 8);
}

inline uint16x8_t Load8IntoUpperU16(const uint8* data_ptr) {
  return Move8IntoUpperU16(vld1_u8(data_ptr));
}

// Extract upper 8 bits from each 16-bit integer in vector registers. This is
// performed for a pair, because instructions often work on pairs.
inline void PairExtractUpper(const uint16x8_t accum_0, const uint16x8_t accum_1,
                             uint8x8_t* res_0, uint8x8_t* res_1) {
  uint8x16x2_t unzipped =
      vuzpq_u8(vreinterpretq_u8_u16(accum_0), vreinterpretq_u8_u16(accum_1));
  *res_0 = vget_low_u8(unzipped.val[1]);
  *res_1 = vget_high_u8(unzipped.val[1]);
}

// This is an exceptional definition.
//
// Modify int16x8_t, adding operators.
//
// There are exceptional circumstances that make it reasonable to write code
// on vector types for quantized resize bilinear in *some cases*.
//
// (a) In exact quant resize bilinear, it should be possible to guarantee that
//     arithmetic never overflows.
// (b) When the resize scaling is 2 or 4 or 8 it is possible to guarantee
//     exact accumulation and exact incrementation.
// (c) In quant resize bilinear the choice of unsigned vs signed accumulation
//     and saturated vs unsaturated arithmetic is often unimportant.
//
// This pattern simplifies the code considerably. This pattern should not be
// used more widely in code since it can hide important numerical detail.
//
// DO NOT add to this any "class-like" methods: only those that do no more than
// redirecting operators to specific intrinsics functions.
struct op_int16x8_t {
  inline op_int16x8_t() = default;
  inline explicit op_int16x8_t(const int16x8_t& initial_val) {
    val = initial_val;
  }
  inline op_int16x8_t& operator=(const int16x8_t& new_val) {
    val = new_val;
    return *this;
  }
  inline op_int16x8_t operator+=(const op_int16x8_t& add_val) {
    val = vaddq_s16(val, add_val.val);
    return *this;
  }
  inline op_int16x8_t operator-=(const op_int16x8_t& sub_val) {
    val = vsubq_s16(val, sub_val.val);
    return *this;
  }
  // This really selects vshlq_n_s16, but requires a longer implementation to
  // convert the shift argument back to a constant. In some compiles are macros
  // requiring constant args.
  inline op_int16x8_t operator<<=(int32 left_shift) {
    switch (left_shift) {
      case 1:
        val = vshlq_n_s16(val, 1);
        break;
      case 4:
        val = vshlq_n_s16(val, 4);
        break;
      case 8:
        val = vshlq_n_s16(val, 8);
        break;
      default:
        TFLITE_CHECK(false);
        break;
    }
    return *this;
  }
  // This really selects vshrq_n_u16, but requires a longer implementation to
  // convert the shift argument back to a constant. In some compiles are macros
  // requiring constant args.
  inline op_int16x8_t operator>>=(int32 right_shift) {
    switch (right_shift) {
      case 1:
        val = vshrq_n_s16(val, 1);
        break;
      case 4:
        val = vshrq_n_s16(val, 4);
        break;
      case 8:
        val = vshrq_n_s16(val, 8);
        break;
      default:
        TFLITE_CHECK(false);
        break;
    }
    return *this;
  }
  friend inline op_int16x8_t operator+(op_int16x8_t lhs,
                                       const op_int16x8_t& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend inline op_int16x8_t operator-(op_int16x8_t lhs,
                                       const op_int16x8_t& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend inline op_int16x8_t operator<<(op_int16x8_t lhs, int32 left_shift) {
    lhs <<= left_shift;
    return lhs;
  }
  friend inline op_int16x8_t operator>>(op_int16x8_t lhs, int32 right_shift) {
    lhs >>= right_shift;
    return lhs;
  }

  int16x8_t val;
};

// This is an exceptional definition.
//
// Modify uint16x8_t, adding operators.
//
// Important: See above notes on op_int16x8_t.
struct op_uint16x8_t {
  inline op_uint16x8_t() = default;
  inline explicit op_uint16x8_t(const uint16x8_t initial_val) {
    val = initial_val;
  }
  inline op_uint16x8_t& operator=(const uint16x8_t& new_val) {
    val = new_val;
    return *this;
  }
  inline op_uint16x8_t operator+=(const op_int16x8_t& add_val) {
    val = vaddq_u16(val, vreinterpretq_u16_s16(add_val.val));
    return *this;
  }
  inline op_uint16x8_t operator-=(const op_int16x8_t& sub_val) {
    val = vsubq_u16(val, vreinterpretq_u16_s16(sub_val.val));
    return *this;
  }
  // This really selects vshlq_n_s16, but requires a longer implementation to
  // convert the shift argument back to a constant. In some compiles are macros
  // requiring constant args.
  inline op_uint16x8_t operator<<=(int32 left_shift) {
    switch (left_shift) {
      case 1:
        val = vshlq_n_u16(val, 1);
        break;
      case 4:
        val = vshlq_n_u16(val, 4);
        break;
      case 8:
        val = vshlq_n_u16(val, 8);
        break;
      default:
        TFLITE_CHECK(false);
        break;
    }
    return *this;
  }
  // This really selects vshrq_n_u16, but requires a longer implementation to
  // convert the shift argument back to a constant. In some compiles are macros
  // requiring constant args.
  inline op_uint16x8_t operator>>=(int32 right_shift) {
    switch (right_shift) {
      case 1:
        val = vshrq_n_u16(val, 1);
        break;
      case 4:
        val = vshrq_n_u16(val, 4);
        break;
      case 8:
        val = vshrq_n_u16(val, 8);
        break;
      default:
        TFLITE_CHECK(false);
        break;
    }
    return *this;
  }
  friend inline op_uint16x8_t operator+(op_uint16x8_t lhs,
                                        const op_int16x8_t& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend inline op_uint16x8_t operator-(op_uint16x8_t lhs,
                                        const op_int16x8_t& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend inline op_uint16x8_t operator<<(op_uint16x8_t lhs, int32 left_shift) {
    lhs <<= left_shift;
    return lhs;
  }
  friend inline op_uint16x8_t operator>>(op_uint16x8_t lhs, int32 right_shift) {
    lhs >>= right_shift;
    return lhs;
  }

  uint16x8_t val;
};

inline op_uint16x8_t VReinterpretQU16S16(const op_int16x8_t& other) {
  op_uint16x8_t ret_val(vreinterpretq_u16_s16(other.val));
  return ret_val;
}
#endif  // USE_NEON

// Optimized resize-bilinear for the special case where the scaling is x8 in
// width and height, and where we can operate on depth-8 blocks at a time. So
// the output blocks are 8x8x8 in width-height-depth.
//
// This optimization is for the half_pixel_centers == true version, for uint8.
// There are versions for NEON and non-NEON compilation.
inline void ResizeBilinear888Uint8(int32 batches, int32 input_height,
                                   int32 input_width, int32 depth,
                                   const uint8* input_data,
                                   uint8* output_data) {
  TFLITE_DCHECK_GE(input_height, 1);
  TFLITE_DCHECK_GE(input_width, 1);
  TFLITE_DCHECK_EQ(depth % 8, 0);

  const int32 input_row_stride = input_width * depth;
  const int32 output_row_stride = input_row_stride * 8;
  for (int b = 0; b < batches; ++b) {
    const uint8* input_base_ptr =
        input_data + b * input_row_stride * input_height;
    uint8* output_base_ptr =
        output_data + b * output_row_stride * input_height * 8;

#ifdef USE_NEON
    for (int c_block = 0; c_block < depth; c_block += 8) {
      op_uint16x8_t accum_c_v;
      // Top-left margin corner.
      {
        uint8x8_t output_data = vld1_u8(&input_base_ptr[c_block]);
        vst1_u8(&output_base_ptr[c_block], output_data);
        vst1_u8(&output_base_ptr[c_block + depth], output_data);
        vst1_u8(&output_base_ptr[c_block + depth * 2], output_data);
        vst1_u8(&output_base_ptr[c_block + depth * 3], output_data);

        // Accumulate in 8.8 representation, pre-adding 0.5 for later rounding.
        accum_c_v = vaddq_u16(Move8IntoUpperU16(output_data), vdupq_n_u16(128));
      }

      // Top-centre margin.
      op_int16x8_t wdelta_c_v;
      op_int16x8_t wdelta_twice_c_v;
      for (int j = 0; j < (input_width - 1); ++j) {
        {
          uint8x8_t output_data_alt;
          uint8x8_t output_data;

          const op_int16x8_t tl_val(
              Load8IntoLowerS16(&input_base_ptr[c_block + depth * j]));
          const op_int16x8_t tr_val(
              Load8IntoLowerS16(&input_base_ptr[c_block + depth * (j + 1)]));
          wdelta_c_v = (tr_val - tl_val) << 4;
          wdelta_twice_c_v = wdelta_c_v << 1;

          op_uint16x8_t accum_c_v_alt = accum_c_v + wdelta_c_v;
          accum_c_v = accum_c_v_alt + wdelta_twice_c_v;
          PairExtractUpper(accum_c_v_alt.val, accum_c_v.val, &output_data_alt,
                           &output_data);

          vst1_u8(&output_base_ptr[c_block + depth * j * 8 + depth * 4],
                  output_data_alt);
          vst1_u8(&output_base_ptr[c_block + depth * j * 8 + depth + depth * 4],
                  output_data);

          for (int p = 2; p < 8; p += 2) {
            accum_c_v_alt = accum_c_v + wdelta_twice_c_v;
            accum_c_v = accum_c_v_alt + wdelta_twice_c_v;
            PairExtractUpper(accum_c_v_alt.val, accum_c_v.val, &output_data_alt,
                             &output_data);

            vst1_u8(&output_base_ptr[c_block + depth * j * 8 + depth * p +
                                     depth * 4],
                    output_data_alt);
            vst1_u8(&output_base_ptr[c_block + depth * j * 8 + depth * (p + 1) +
                                     depth * 4],
                    output_data);
          }
          accum_c_v += wdelta_c_v;
        }
      }

      // Top-right margin corner.
      {
        uint8x8_t output_data_discard;
        uint8x8_t output_data;

        // Accumulations have pre-added 0.5 for rounding, but that is just
        // discarded and this just avoids re-loading.
        PairExtractUpper(accum_c_v.val, accum_c_v.val, &output_data,
                         &output_data_discard);

        vst1_u8(&output_base_ptr[c_block + depth * (input_width - 1) * 8 +
                                 depth * 4],
                output_data);
        vst1_u8(&output_base_ptr[c_block + depth * (input_width - 1) * 8 +
                                 depth * 4 + depth],
                output_data);
        vst1_u8(&output_base_ptr[c_block + depth * (input_width - 1) * 8 +
                                 depth * 4 + depth * 2],
                output_data);
        vst1_u8(&output_base_ptr[c_block + depth * (input_width - 1) * 8 +
                                 depth * 4 + depth * 3],
                output_data);
      }
    }
    // Fill out remainder of top margin.
    std::memcpy(output_base_ptr + output_row_stride, output_base_ptr,
                output_row_stride * sizeof(uint8));
    std::memcpy(output_base_ptr + output_row_stride * 2, output_base_ptr,
                output_row_stride * sizeof(uint8));
    std::memcpy(output_base_ptr + output_row_stride * 3, output_base_ptr,
                output_row_stride * sizeof(uint8));
    output_base_ptr += output_row_stride * 4;

    // Main rows.
    for (int k = 0; k < (input_height - 1); ++k) {
      for (int c_block = 0; c_block < depth; c_block += 8) {
        uint8* output_base_ptr_0 = output_base_ptr;
        uint8* output_base_ptr_1;
        uint8* output_base_ptr_2;
        uint8* output_base_ptr_3;
        uint8* output_base_ptr_4;
        uint8* output_base_ptr_5;
        uint8* output_base_ptr_6;
        uint8* output_base_ptr_7;

        op_uint16x8_t accum_0_c_v;
        op_uint16x8_t accum_1_c_v;
        op_uint16x8_t accum_2_c_v;
        op_uint16x8_t accum_3_c_v;
        op_uint16x8_t accum_4_c_v;
        op_uint16x8_t accum_5_c_v;
        op_uint16x8_t accum_6_c_v;
        op_uint16x8_t accum_7_c_v;

        op_int16x8_t hdelta_c_v;
        op_int16x8_t hdelta_twice_c_v;

        // Left margin for 8 rows.
        {
          uint8x8_t output_data_0_c;
          uint8x8_t output_data_1_c;
          uint8x8_t output_data_2_c;
          uint8x8_t output_data_3_c;
          uint8x8_t output_data_4_c;
          uint8x8_t output_data_5_c;
          uint8x8_t output_data_6_c;
          uint8x8_t output_data_7_c;

          const op_int16x8_t tl_val(
              Load8IntoLowerS16(&input_base_ptr[c_block]));
          const op_int16x8_t bl_val(
              Load8IntoLowerS16(&input_base_ptr[c_block + input_row_stride]));
          hdelta_c_v = (bl_val - tl_val) << 4;

          // Accumulate in 8.8 representation, pre-adding 0.5 for later
          // rounding.
          accum_0_c_v = VReinterpretQU16S16(tl_val << 8);
          accum_0_c_v = vaddq_u16(accum_0_c_v.val, vdupq_n_u16(128));

          hdelta_twice_c_v = hdelta_c_v << 1;

          accum_0_c_v += hdelta_c_v;
          accum_1_c_v = accum_0_c_v + hdelta_twice_c_v;
          PairExtractUpper(accum_0_c_v.val, accum_1_c_v.val, &output_data_0_c,
                           &output_data_1_c);

          vst1_u8(&output_base_ptr_0[c_block], output_data_0_c);
          vst1_u8(&output_base_ptr_0[c_block + depth], output_data_0_c);
          vst1_u8(&output_base_ptr_0[c_block + depth * 2], output_data_0_c);
          vst1_u8(&output_base_ptr_0[c_block + depth * 3], output_data_0_c);

          output_base_ptr_1 = output_base_ptr_0 + output_row_stride;
          vst1_u8(&output_base_ptr_1[c_block], output_data_1_c);
          vst1_u8(&output_base_ptr_1[c_block + depth], output_data_1_c);
          vst1_u8(&output_base_ptr_1[c_block + depth * 2], output_data_1_c);
          vst1_u8(&output_base_ptr_1[c_block + depth * 3], output_data_1_c);

          //

          output_base_ptr_2 = output_base_ptr_1 + output_row_stride;
          accum_2_c_v = accum_1_c_v + hdelta_twice_c_v;
          accum_3_c_v = accum_2_c_v + hdelta_twice_c_v;
          PairExtractUpper(accum_2_c_v.val, accum_3_c_v.val, &output_data_2_c,
                           &output_data_3_c);

          vst1_u8(&output_base_ptr_2[c_block], output_data_2_c);
          vst1_u8(&output_base_ptr_2[c_block + depth], output_data_2_c);
          vst1_u8(&output_base_ptr_2[c_block + depth * 2], output_data_2_c);
          vst1_u8(&output_base_ptr_2[c_block + depth * 3], output_data_2_c);

          output_base_ptr_3 = output_base_ptr_2 + output_row_stride;
          vst1_u8(&output_base_ptr_3[c_block], output_data_3_c);
          vst1_u8(&output_base_ptr_3[c_block + depth], output_data_3_c);
          vst1_u8(&output_base_ptr_3[c_block + depth * 2], output_data_3_c);
          vst1_u8(&output_base_ptr_3[c_block + depth * 3], output_data_3_c);

          //

          output_base_ptr_4 = output_base_ptr_3 + output_row_stride;
          accum_4_c_v = accum_3_c_v + hdelta_twice_c_v;
          accum_5_c_v = accum_4_c_v + hdelta_twice_c_v;
          PairExtractUpper(accum_4_c_v.val, accum_5_c_v.val, &output_data_4_c,
                           &output_data_5_c);

          vst1_u8(&output_base_ptr_4[c_block], output_data_4_c);
          vst1_u8(&output_base_ptr_4[c_block + depth], output_data_4_c);
          vst1_u8(&output_base_ptr_4[c_block + depth * 2], output_data_4_c);
          vst1_u8(&output_base_ptr_4[c_block + depth * 3], output_data_4_c);

          output_base_ptr_5 = output_base_ptr_4 + output_row_stride;
          vst1_u8(&output_base_ptr_5[c_block], output_data_5_c);
          vst1_u8(&output_base_ptr_5[c_block + depth], output_data_5_c);
          vst1_u8(&output_base_ptr_5[c_block + depth * 2], output_data_5_c);
          vst1_u8(&output_base_ptr_5[c_block + depth * 3], output_data_5_c);

          //

          output_base_ptr_6 = output_base_ptr_5 + output_row_stride;
          accum_6_c_v = accum_5_c_v + hdelta_twice_c_v;
          accum_7_c_v = accum_6_c_v + hdelta_twice_c_v;
          PairExtractUpper(accum_6_c_v.val, accum_7_c_v.val, &output_data_6_c,
                           &output_data_7_c);

          vst1_u8(&output_base_ptr_6[c_block], output_data_6_c);
          vst1_u8(&output_base_ptr_6[c_block + depth], output_data_6_c);
          vst1_u8(&output_base_ptr_6[c_block + depth * 2], output_data_6_c);
          vst1_u8(&output_base_ptr_6[c_block + depth * 3], output_data_6_c);

          output_base_ptr_7 = output_base_ptr_6 + output_row_stride;
          vst1_u8(&output_base_ptr_7[c_block], output_data_7_c);
          vst1_u8(&output_base_ptr_7[c_block + depth], output_data_7_c);
          vst1_u8(&output_base_ptr_7[c_block + depth * 2], output_data_7_c);
          vst1_u8(&output_base_ptr_7[c_block + depth * 3], output_data_7_c);
        }

        // Main central body.
        op_int16x8_t wdelta_c;
        op_int16x8_t wdelta_twice_c;
        op_int16x8_t hwdelta_c;
        op_int16x8_t hwdelta_twice_c;

        op_int16x8_t incr_0_c;
        op_int16x8_t incr_1_c;
        op_int16x8_t incr_2_c;
        op_int16x8_t incr_3_c;
        op_int16x8_t incr_4_c;
        op_int16x8_t incr_5_c;
        op_int16x8_t incr_6_c;
        op_int16x8_t incr_7_c;

        uint8x8_t output_data_0_c;
        uint8x8_t output_data_1_c;
        uint8x8_t output_data_2_c;
        uint8x8_t output_data_3_c;
        uint8x8_t output_data_4_c;
        uint8x8_t output_data_5_c;
        uint8x8_t output_data_6_c;
        uint8x8_t output_data_7_c;
        for (int j = 0; j < (input_width - 1); ++j) {
          // output_base_ptr_0 = output_base_ptr;
          // output_base_ptr_1 = output_base_ptr_0 + output_row_stride; ETC
          {
            const op_int16x8_t tl_val(
                Load8IntoLowerS16(&input_base_ptr[c_block + depth * j]));
            const op_int16x8_t bl_val(Load8IntoLowerS16(
                &input_base_ptr[c_block + depth * j + input_row_stride]));
            const op_int16x8_t tr_val(
                Load8IntoLowerS16(&input_base_ptr[c_block + depth * (j + 1)]));
            const op_int16x8_t br_val(Load8IntoLowerS16(
                &input_base_ptr[c_block + depth * (j + 1) + input_row_stride]));

            const op_int16x8_t tmp_diff = tr_val - tl_val;
            wdelta_c = tmp_diff << 4;
            wdelta_twice_c = wdelta_c << 1;
            hwdelta_c = (br_val - bl_val) - tmp_diff;
            hwdelta_twice_c = hwdelta_c << 1;

            op_int16x8_t incr_base = wdelta_c + hwdelta_c;
            accum_0_c_v += incr_base;
            incr_0_c = incr_base << 1;
            incr_base += hwdelta_twice_c;
            accum_1_c_v += incr_base;
            incr_1_c = incr_base << 1;

            PairExtractUpper(accum_0_c_v.val, accum_1_c_v.val, &output_data_0_c,
                             &output_data_1_c);
            vst1_u8(&output_base_ptr_0[c_block + depth * j * 8 + depth * 4],
                    output_data_0_c);
            vst1_u8(&output_base_ptr_1[c_block + depth * j * 8 + depth * 4],
                    output_data_1_c);

            incr_base += hwdelta_twice_c;
            accum_2_c_v += incr_base;
            incr_2_c = incr_base << 1;
            incr_base += hwdelta_twice_c;
            accum_3_c_v += incr_base;
            incr_3_c = incr_base << 1;

            PairExtractUpper(accum_2_c_v.val, accum_3_c_v.val, &output_data_2_c,
                             &output_data_3_c);
            vst1_u8(&output_base_ptr_2[c_block + depth * j * 8 + depth * 4],
                    output_data_2_c);
            vst1_u8(&output_base_ptr_3[c_block + depth * j * 8 + depth * 4],
                    output_data_3_c);

            incr_base += hwdelta_twice_c;
            accum_4_c_v += incr_base;
            incr_4_c = incr_base << 1;
            incr_base += hwdelta_twice_c;
            accum_5_c_v += incr_base;
            incr_5_c = incr_base << 1;

            PairExtractUpper(accum_4_c_v.val, accum_5_c_v.val, &output_data_4_c,
                             &output_data_5_c);
            vst1_u8(&output_base_ptr_4[c_block + depth * j * 8 + depth * 4],
                    output_data_4_c);
            vst1_u8(&output_base_ptr_5[c_block + depth * j * 8 + depth * 4],
                    output_data_5_c);

            incr_base += hwdelta_twice_c;
            accum_6_c_v += incr_base;
            incr_6_c = incr_base << 1;
            incr_base += hwdelta_twice_c;
            accum_7_c_v += incr_base;
            incr_7_c = incr_base << 1;

            PairExtractUpper(accum_6_c_v.val, accum_7_c_v.val, &output_data_6_c,
                             &output_data_7_c);
            vst1_u8(&output_base_ptr_6[c_block + depth * j * 8 + depth * 4],
                    output_data_6_c);
            vst1_u8(&output_base_ptr_7[c_block + depth * j * 8 + depth * 4],
                    output_data_7_c);

            for (int p = 1; p < 8; ++p) {
              accum_0_c_v += incr_0_c;
              accum_1_c_v += incr_1_c;
              PairExtractUpper(accum_0_c_v.val, accum_1_c_v.val,
                               &output_data_0_c, &output_data_1_c);
              vst1_u8(&output_base_ptr_0[c_block + depth * j * 8 + depth * p +
                                         depth * 4],
                      output_data_0_c);
              vst1_u8(&output_base_ptr_1[c_block + depth * j * 8 + depth * p +
                                         depth * 4],
                      output_data_1_c);

              accum_2_c_v += incr_2_c;
              accum_3_c_v += incr_3_c;
              PairExtractUpper(accum_2_c_v.val, accum_3_c_v.val,
                               &output_data_2_c, &output_data_3_c);
              vst1_u8(&output_base_ptr_2[c_block + depth * j * 8 + depth * p +
                                         depth * 4],
                      output_data_2_c);
              vst1_u8(&output_base_ptr_3[c_block + depth * j * 8 + depth * p +
                                         depth * 4],
                      output_data_3_c);

              accum_4_c_v += incr_4_c;
              accum_5_c_v += incr_5_c;
              PairExtractUpper(accum_4_c_v.val, accum_5_c_v.val,
                               &output_data_4_c, &output_data_5_c);
              vst1_u8(&output_base_ptr_4[c_block + depth * j * 8 + depth * p +
                                         depth * 4],
                      output_data_4_c);
              vst1_u8(&output_base_ptr_5[c_block + depth * j * 8 + depth * p +
                                         depth * 4],
                      output_data_5_c);

              accum_6_c_v += incr_6_c;
              accum_7_c_v += incr_7_c;
              PairExtractUpper(accum_6_c_v.val, accum_7_c_v.val,
                               &output_data_6_c, &output_data_7_c);
              vst1_u8(&output_base_ptr_6[c_block + depth * j * 8 + depth * p +
                                         depth * 4],
                      output_data_6_c);
              vst1_u8(&output_base_ptr_7[c_block + depth * j * 8 + depth * p +
                                         depth * 4],
                      output_data_7_c);
            }

            accum_0_c_v += (incr_0_c >> 1);
            accum_1_c_v += (incr_1_c >> 1);
            accum_2_c_v += (incr_2_c >> 1);
            accum_3_c_v += (incr_3_c >> 1);
            accum_4_c_v += (incr_4_c >> 1);
            accum_5_c_v += (incr_5_c >> 1);
            accum_6_c_v += (incr_6_c >> 1);
            accum_7_c_v += (incr_7_c >> 1);
          }
        }

        // Right margin.
        {
          // Accumulations have pre-added 0.5 for rounding, but that is just
          // discarded and this just avoids re-loading.
          PairExtractUpper(accum_0_c_v.val, accum_1_c_v.val, &output_data_0_c,
                           &output_data_1_c);
          PairExtractUpper(accum_2_c_v.val, accum_3_c_v.val, &output_data_2_c,
                           &output_data_3_c);
          PairExtractUpper(accum_4_c_v.val, accum_5_c_v.val, &output_data_4_c,
                           &output_data_5_c);
          PairExtractUpper(accum_6_c_v.val, accum_7_c_v.val, &output_data_6_c,
                           &output_data_7_c);
          for (int p = 0; p < 4; ++p) {
            vst1_u8(&output_base_ptr_0[c_block + depth * (input_width - 1) * 8 +
                                       depth * 4 + depth * p],
                    output_data_0_c);
            vst1_u8(&output_base_ptr_1[c_block + depth * (input_width - 1) * 8 +
                                       depth * 4 + depth * p],
                    output_data_1_c);
            vst1_u8(&output_base_ptr_2[c_block + depth * (input_width - 1) * 8 +
                                       depth * 4 + depth * p],
                    output_data_2_c);
            vst1_u8(&output_base_ptr_3[c_block + depth * (input_width - 1) * 8 +
                                       depth * 4 + depth * p],
                    output_data_3_c);
            vst1_u8(&output_base_ptr_4[c_block + depth * (input_width - 1) * 8 +
                                       depth * 4 + depth * p],
                    output_data_4_c);
            vst1_u8(&output_base_ptr_5[c_block + depth * (input_width - 1) * 8 +
                                       depth * 4 + depth * p],
                    output_data_5_c);
            vst1_u8(&output_base_ptr_6[c_block + depth * (input_width - 1) * 8 +
                                       depth * 4 + depth * p],
                    output_data_6_c);
            vst1_u8(&output_base_ptr_7[c_block + depth * (input_width - 1) * 8 +
                                       depth * 4 + depth * p],
                    output_data_7_c);
          }
        }
      }

      output_base_ptr += output_row_stride * 8;
      input_base_ptr += input_row_stride;
    }

    //

    for (int c_block = 0; c_block < depth; c_block += 8) {
      op_uint16x8_t accum_c_v;
      // Bottom-left margin corner.
      {
        uint8x8_t output_data = vld1_u8(&input_base_ptr[c_block]);
        vst1_u8(&output_base_ptr[c_block], output_data);
        vst1_u8(&output_base_ptr[c_block + depth], output_data);
        vst1_u8(&output_base_ptr[c_block + depth * 2], output_data);
        vst1_u8(&output_base_ptr[c_block + depth * 3], output_data);

        // Accumulate in 8.8 representation, pre-adding 0.5 for later rounding.
        accum_c_v = vaddq_u16(Move8IntoUpperU16(output_data), vdupq_n_u16(128));
      }

      // Bottom-centre margin.
      op_int16x8_t wdelta_c_v;
      op_int16x8_t wdelta_twice_c_v;
      for (int j = 0; j < (input_width - 1); ++j) {
        {
          uint8x8_t output_data_alt;
          uint8x8_t output_data;

          const op_int16x8_t tl_val(
              Load8IntoLowerS16(&input_base_ptr[c_block + depth * j]));
          const op_int16x8_t tr_val(
              Load8IntoLowerS16(&input_base_ptr[c_block + depth * (j + 1)]));
          wdelta_c_v = (tr_val - tl_val) << 4;
          wdelta_twice_c_v = wdelta_c_v << 1;

          op_uint16x8_t accum_c_v_alt = accum_c_v + wdelta_c_v;
          accum_c_v = accum_c_v_alt + wdelta_twice_c_v;
          PairExtractUpper(accum_c_v_alt.val, accum_c_v.val, &output_data_alt,
                           &output_data);

          vst1_u8(&output_base_ptr[c_block + depth * j * 8 + depth * 4],
                  output_data_alt);
          vst1_u8(&output_base_ptr[c_block + depth * j * 8 + depth + depth * 4],
                  output_data);

          for (int p = 2; p < 8; p += 2) {
            accum_c_v_alt = accum_c_v + wdelta_twice_c_v;
            accum_c_v = accum_c_v_alt + wdelta_twice_c_v;
            PairExtractUpper(accum_c_v_alt.val, accum_c_v.val, &output_data_alt,
                             &output_data);

            vst1_u8(&output_base_ptr[c_block + depth * j * 8 + depth * p +
                                     depth * 4],
                    output_data_alt);
            vst1_u8(&output_base_ptr[c_block + depth * j * 8 + depth * (p + 1) +
                                     depth * 4],
                    output_data);
          }
          accum_c_v += wdelta_c_v;
        }
      }

      // Bottom-right margin corner.
      {
        uint8x8_t output_data_discard;
        uint8x8_t output_data;

        // Accumulations have pre-added 0.5 for rounding, but that is just
        // discarded and this just avoids re-loading.
        PairExtractUpper(accum_c_v.val, accum_c_v.val, &output_data,
                         &output_data_discard);

        vst1_u8(&output_base_ptr[c_block + depth * (input_width - 1) * 8 +
                                 depth * 4],
                output_data);
        vst1_u8(&output_base_ptr[c_block + depth * (input_width - 1) * 8 +
                                 depth * 4 + depth],
                output_data);
        vst1_u8(&output_base_ptr[c_block + depth * (input_width - 1) * 8 +
                                 depth * 4 + depth * 2],
                output_data);
        vst1_u8(&output_base_ptr[c_block + depth * (input_width - 1) * 8 +
                                 depth * 4 + depth * 3],
                output_data);
      }
    }
    // Fill out remainder of bottom margin.
    std::memcpy(output_base_ptr + output_row_stride, output_base_ptr,
                output_row_stride * sizeof(uint8));
    std::memcpy(output_base_ptr + output_row_stride * 2, output_base_ptr,
                output_row_stride * sizeof(uint8));
    std::memcpy(output_base_ptr + output_row_stride * 3, output_base_ptr,
                output_row_stride * sizeof(uint8));

#else  // USE_NEON
    for (int c_block = 0; c_block < depth; c_block += 8) {
      uint8 output_data[8];
      uint16 accum[8];
      // Top-left margin corner.
      for (int c = 0; c < 8; ++c) {
        output_data[c] = input_base_ptr[c_block + c];
        output_base_ptr[c_block + c] = output_data[c];
        output_base_ptr[c_block + c + depth] = output_data[c];
        output_base_ptr[c_block + c + depth * 2] = output_data[c];
        output_base_ptr[c_block + c + depth * 3] = output_data[c];

        // Accumulate in 8.8 representation, pre-adding 0.5 for later rounding.
        accum[c] =
            (output_data[c] << 8) + 128;  // 128 = 0.5 in 8.8 representation.
      }

      // Top-centre margin.
      uint16 wdelta[8];
      uint16 wdelta_twice[8];
      for (int j = 0; j < (input_width - 1); ++j) {
        for (int c = 0; c < 8; ++c) {
          wdelta[c] = static_cast<uint16>(
                          input_base_ptr[c_block + c + depth * (j + 1)] -
                          input_base_ptr[c_block + c + depth * j])
                      << 4;
          wdelta_twice[c] = wdelta[c] << 1;

          accum[c] += wdelta[c];
          output_base_ptr[c_block + c + depth * j * 8 + depth * 4] =
              accum[c] >> 8;
          for (int p = 1; p < 8; ++p) {
            accum[c] += wdelta_twice[c];
            output_base_ptr[c_block + c + depth * j * 8 + depth * p +
                            depth * 4] = accum[c] >> 8;
          }
          accum[c] += wdelta[c];
        }
      }

      // Top-right margin corner.
      for (int c = 0; c < 8; ++c) {
        // Accumulations have pre-added 0.5 for rounding, but that is just
        // discarded and this just avoids re-loading.
        output_data[c] = accum[c] >> 8;
        TFLITE_DCHECK_EQ(
            output_data[c],
            input_base_ptr[c_block + c + depth * (input_width - 1)]);
        output_base_ptr[c_block + c + depth * (input_width - 1) * 8 +
                        depth * 4] = output_data[c];
        output_base_ptr[c_block + c + depth * (input_width - 1) * 8 +
                        depth * 4 + depth] = output_data[c];
        output_base_ptr[c_block + c + depth * (input_width - 1) * 8 +
                        depth * 4 + depth * 2] = output_data[c];
        output_base_ptr[c_block + c + depth * (input_width - 1) * 8 +
                        depth * 4 + depth * 3] = output_data[c];
      }
    }
    // Fill out remainder of top margin.
    std::memcpy(output_base_ptr + output_row_stride, output_base_ptr,
                output_row_stride * sizeof(uint8));
    std::memcpy(output_base_ptr + output_row_stride * 2, output_base_ptr,
                output_row_stride * sizeof(uint8));
    std::memcpy(output_base_ptr + output_row_stride * 3, output_base_ptr,
                output_row_stride * sizeof(uint8));
    output_base_ptr += output_row_stride * 4;

    // Main rows.
    for (int k = 0; k < (input_height - 1); ++k) {
      for (int c_block = 0; c_block < depth; c_block += 8) {
        uint8* output_base_ptr_0 = output_base_ptr;
        uint8* output_base_ptr_1;
        uint8* output_base_ptr_2;
        uint8* output_base_ptr_3;
        uint8* output_base_ptr_4;
        uint8* output_base_ptr_5;
        uint8* output_base_ptr_6;
        uint8* output_base_ptr_7;
        uint16 accum_0[8];
        uint16 accum_1[8];
        uint16 accum_2[8];
        uint16 accum_3[8];
        uint16 accum_4[8];
        uint16 accum_5[8];
        uint16 accum_6[8];
        uint16 accum_7[8];

        // We prefer accum_0[c], etc, in sense of packed-data array for
        // register. However the compiler will not reliably optimize for an
        // array, and so we do most of the work in pure scalar variables.
        uint16 accum_0_c;
        uint16 accum_1_c;
        uint16 accum_2_c;
        uint16 accum_3_c;
        uint16 accum_4_c;
        uint16 accum_5_c;
        uint16 accum_6_c;
        uint16 accum_7_c;

        int16 hdelta_c;
        int16 hdelta_twice_c;

        // Left margin for 8 rows.
        for (int c = 0; c < 8; ++c) {
          hdelta_c = static_cast<uint16>(
                         input_base_ptr[c_block + c + input_row_stride] -
                         input_base_ptr[c_block + c])
                     << 4;

          // Accumulate in 8.8 representation, pre-adding 0.5 for later
          // rounding.
          accum_0_c = (input_base_ptr[c_block + c] << 8) + 128;

          accum_0_c += hdelta_c;
          output_base_ptr_0[c_block + c] = accum_0_c >> 8;
          output_base_ptr_0[c_block + c + depth] = accum_0_c >> 8;
          output_base_ptr_0[c_block + c + depth * 2] = accum_0_c >> 8;
          output_base_ptr_0[c_block + c + depth * 3] = accum_0_c >> 8;

          hdelta_twice_c = hdelta_c << 1;

          output_base_ptr_1 = output_base_ptr_0 + output_row_stride;
          accum_1_c = accum_0_c + hdelta_twice_c;
          output_base_ptr_1[c_block + c] = accum_1_c >> 8;
          output_base_ptr_1[c_block + c + depth] = accum_1_c >> 8;
          output_base_ptr_1[c_block + c + depth * 2] = accum_1_c >> 8;
          output_base_ptr_1[c_block + c + depth * 3] = accum_1_c >> 8;

          output_base_ptr_2 = output_base_ptr_1 + output_row_stride;
          accum_2_c = accum_1_c + hdelta_twice_c;
          output_base_ptr_2[c_block + c] = accum_2_c >> 8;
          output_base_ptr_2[c_block + c + depth] = accum_2_c >> 8;
          output_base_ptr_2[c_block + c + depth * 2] = accum_2_c >> 8;
          output_base_ptr_2[c_block + c + depth * 3] = accum_2_c >> 8;

          output_base_ptr_3 = output_base_ptr_2 + output_row_stride;
          accum_3_c = accum_2_c + hdelta_twice_c;
          output_base_ptr_3[c_block + c] = accum_3_c >> 8;
          output_base_ptr_3[c_block + c + depth] = accum_3_c >> 8;
          output_base_ptr_3[c_block + c + depth * 2] = accum_3_c >> 8;
          output_base_ptr_3[c_block + c + depth * 3] = accum_3_c >> 8;

          output_base_ptr_4 = output_base_ptr_3 + output_row_stride;
          accum_4_c = accum_3_c + hdelta_twice_c;
          output_base_ptr_4[c_block + c] = accum_4_c >> 8;
          output_base_ptr_4[c_block + c + depth] = accum_4_c >> 8;
          output_base_ptr_4[c_block + c + depth * 2] = accum_4_c >> 8;
          output_base_ptr_4[c_block + c + depth * 3] = accum_4_c >> 8;

          output_base_ptr_5 = output_base_ptr_4 + output_row_stride;
          accum_5_c = accum_4_c + hdelta_twice_c;
          output_base_ptr_5[c_block + c] = accum_5_c >> 8;
          output_base_ptr_5[c_block + c + depth] = accum_5_c >> 8;
          output_base_ptr_5[c_block + c + depth * 2] = accum_5_c >> 8;
          output_base_ptr_5[c_block + c + depth * 3] = accum_5_c >> 8;

          output_base_ptr_6 = output_base_ptr_5 + output_row_stride;
          accum_6_c = accum_5_c + hdelta_twice_c;
          output_base_ptr_6[c_block + c] = accum_6_c >> 8;
          output_base_ptr_6[c_block + c + depth] = accum_6_c >> 8;
          output_base_ptr_6[c_block + c + depth * 2] = accum_6_c >> 8;
          output_base_ptr_6[c_block + c + depth * 3] = accum_6_c >> 8;

          output_base_ptr_7 = output_base_ptr_6 + output_row_stride;
          accum_7_c = accum_6_c + hdelta_twice_c;
          output_base_ptr_7[c_block + c] = accum_7_c >> 8;
          output_base_ptr_7[c_block + c + depth] = accum_7_c >> 8;
          output_base_ptr_7[c_block + c + depth * 2] = accum_7_c >> 8;
          output_base_ptr_7[c_block + c + depth * 3] = accum_7_c >> 8;

          accum_0[c] = accum_0_c;
          accum_1[c] = accum_1_c;
          accum_2[c] = accum_2_c;
          accum_3[c] = accum_3_c;
          accum_4[c] = accum_4_c;
          accum_5[c] = accum_5_c;
          accum_6[c] = accum_6_c;
          accum_7[c] = accum_7_c;
        }

        // Main central body.
        int16 wdelta_c;
        int16 wdelta_twice_c;
        int16 hwdelta_c;
        int16 hwdelta_twice_c;

        int16 incr_0_c;
        int16 incr_1_c;
        int16 incr_2_c;
        int16 incr_3_c;
        int16 incr_4_c;
        int16 incr_5_c;
        int16 incr_6_c;
        int16 incr_7_c;
        for (int j = 0; j < (input_width - 1); ++j) {
          for (int c = 0; c < 8; ++c) {
            accum_0_c = accum_0[c];
            accum_1_c = accum_1[c];
            accum_2_c = accum_2[c];
            accum_3_c = accum_3[c];
            accum_4_c = accum_4[c];
            accum_5_c = accum_5[c];
            accum_6_c = accum_6[c];
            accum_7_c = accum_7[c];

            wdelta_c = static_cast<uint16>(
                           input_base_ptr[c_block + c + depth * (j + 1)] -
                           input_base_ptr[c_block + c + depth * j])
                       << 4;
            wdelta_twice_c = wdelta_c << 1;
            hwdelta_c = static_cast<uint16>(
                input_base_ptr[c_block + c + depth * (j + 1) +
                               input_row_stride] -
                input_base_ptr[c_block + c + depth * (j + 1)] -
                input_base_ptr[c_block + c + depth * j + input_row_stride] +
                input_base_ptr[c_block + c + depth * j]);
            hwdelta_twice_c = hwdelta_c << 1;

            uint16 incr_base = wdelta_c + hwdelta_c;
            accum_0_c += incr_base;
            output_base_ptr_0[c_block + c + depth * j * 8 + depth * 4] =
                accum_0_c >> 8;
            incr_0_c = incr_base << 1;

            incr_base += hwdelta_twice_c;
            accum_1_c += incr_base;
            output_base_ptr_1[c_block + c + depth * j * 8 + depth * 4] =
                accum_1_c >> 8;
            incr_1_c = incr_base << 1;

            incr_base += hwdelta_twice_c;
            accum_2_c += incr_base;
            output_base_ptr_2[c_block + c + depth * j * 8 + depth * 4] =
                accum_2_c >> 8;
            incr_2_c = incr_base << 1;

            incr_base += hwdelta_twice_c;
            accum_3_c += incr_base;
            output_base_ptr_3[c_block + c + depth * j * 8 + depth * 4] =
                accum_3_c >> 8;
            incr_3_c = incr_base << 1;

            incr_base += hwdelta_twice_c;
            accum_4_c += incr_base;
            output_base_ptr_4[c_block + c + depth * j * 8 + depth * 4] =
                accum_4_c >> 8;
            incr_4_c = incr_base << 1;

            incr_base += hwdelta_twice_c;
            accum_5_c += incr_base;
            output_base_ptr_5[c_block + c + depth * j * 8 + depth * 4] =
                accum_5_c >> 8;
            incr_5_c = incr_base << 1;

            incr_base += hwdelta_twice_c;
            accum_6_c += incr_base;
            output_base_ptr_6[c_block + c + depth * j * 8 + depth * 4] =
                accum_6_c >> 8;
            incr_6_c = incr_base << 1;

            incr_base += hwdelta_twice_c;
            accum_7_c += incr_base;
            output_base_ptr_7[c_block + c + depth * j * 8 + depth * 4] =
                accum_7_c >> 8;
            incr_7_c = incr_base << 1;

            for (int p = 1; p < 8; ++p) {
              accum_0_c += incr_0_c;
              output_base_ptr_0[c_block + c + depth * j * 8 + depth * p +
                                depth * 4] = accum_0_c >> 8;
              accum_1_c += incr_1_c;
              output_base_ptr_1[c_block + c + depth * j * 8 + depth * p +
                                depth * 4] = accum_1_c >> 8;
              accum_2_c += incr_2_c;
              output_base_ptr_2[c_block + c + depth * j * 8 + depth * p +
                                depth * 4] = accum_2_c >> 8;
              accum_3_c += incr_3_c;
              output_base_ptr_3[c_block + c + depth * j * 8 + depth * p +
                                depth * 4] = accum_3_c >> 8;
              accum_4_c += incr_4_c;
              output_base_ptr_4[c_block + c + depth * j * 8 + depth * p +
                                depth * 4] = accum_4_c >> 8;
              accum_5_c += incr_5_c;
              output_base_ptr_5[c_block + c + depth * j * 8 + depth * p +
                                depth * 4] = accum_5_c >> 8;
              accum_6_c += incr_6_c;
              output_base_ptr_6[c_block + c + depth * j * 8 + depth * p +
                                depth * 4] = accum_6_c >> 8;
              accum_7_c += incr_7_c;
              output_base_ptr_7[c_block + c + depth * j * 8 + depth * p +
                                depth * 4] = accum_7_c >> 8;
            }
            accum_0_c += incr_0_c / 2;
            accum_1_c += incr_1_c / 2;
            accum_2_c += incr_2_c / 2;
            accum_3_c += incr_3_c / 2;
            accum_4_c += incr_4_c / 2;
            accum_5_c += incr_5_c / 2;
            accum_6_c += incr_6_c / 2;
            accum_7_c += incr_7_c / 2;

            accum_0[c] = accum_0_c;
            accum_1[c] = accum_1_c;
            accum_2[c] = accum_2_c;
            accum_3[c] = accum_3_c;
            accum_4[c] = accum_4_c;
            accum_5[c] = accum_5_c;
            accum_6[c] = accum_6_c;
            accum_7[c] = accum_7_c;
          }
        }

        // Right margin.
        uint8 output_data_0_c;
        uint8 output_data_1_c;
        uint8 output_data_2_c;
        uint8 output_data_3_c;
        uint8 output_data_4_c;
        uint8 output_data_5_c;
        uint8 output_data_6_c;
        uint8 output_data_7_c;
        for (int c = 0; c < 8; ++c) {
          accum_0_c = accum_0[c];
          accum_1_c = accum_1[c];
          accum_2_c = accum_2[c];
          accum_3_c = accum_3[c];
          accum_4_c = accum_4[c];
          accum_5_c = accum_5[c];
          accum_6_c = accum_6[c];
          accum_7_c = accum_7[c];

          // Accumulations have pre-added 0.5 for rounding, but that is just
          // discarded and this just avoids re-loading.
          output_data_0_c = accum_0_c >> 8;
          output_data_1_c = accum_1_c >> 8;
          output_data_2_c = accum_2_c >> 8;
          output_data_3_c = accum_3_c >> 8;
          output_data_4_c = accum_4_c >> 8;
          output_data_5_c = accum_5_c >> 8;
          output_data_6_c = accum_6_c >> 8;
          output_data_7_c = accum_7_c >> 8;
          for (int p = 0; p < 4; ++p) {
            output_base_ptr_0[c_block + c + depth * (input_width - 1) * 8 +
                              depth * 4 + depth * p] = output_data_0_c;
            output_base_ptr_1[c_block + c + depth * (input_width - 1) * 8 +
                              depth * 4 + depth * p] = output_data_1_c;
            output_base_ptr_2[c_block + c + depth * (input_width - 1) * 8 +
                              depth * 4 + depth * p] = output_data_2_c;
            output_base_ptr_3[c_block + c + depth * (input_width - 1) * 8 +
                              depth * 4 + depth * p] = output_data_3_c;
            output_base_ptr_4[c_block + c + depth * (input_width - 1) * 8 +
                              depth * 4 + depth * p] = output_data_4_c;
            output_base_ptr_5[c_block + c + depth * (input_width - 1) * 8 +
                              depth * 4 + depth * p] = output_data_5_c;
            output_base_ptr_6[c_block + c + depth * (input_width - 1) * 8 +
                              depth * 4 + depth * p] = output_data_6_c;
            output_base_ptr_7[c_block + c + depth * (input_width - 1) * 8 +
                              depth * 4 + depth * p] = output_data_7_c;
          }

          accum_0[c] = accum_0_c;
          accum_1[c] = accum_1_c;
          accum_2[c] = accum_2_c;
          accum_3[c] = accum_3_c;
          accum_4[c] = accum_4_c;
          accum_5[c] = accum_5_c;
          accum_6[c] = accum_6_c;
          accum_7[c] = accum_7_c;
        }
      }

      output_base_ptr += output_row_stride * 8;
      input_base_ptr += input_row_stride;
    }

    for (int c_block = 0; c_block < depth; c_block += 8) {
      uint8 output_data[8];
      uint16 accum[8];
      // Bottom-left margin corner.
      for (int c = 0; c < 8; ++c) {
        output_data[c] = input_base_ptr[c_block + c];
        output_base_ptr[c_block + c] = output_data[c];
        output_base_ptr[c_block + c + depth] = output_data[c];
        output_base_ptr[c_block + c + depth * 2] = output_data[c];
        output_base_ptr[c_block + c + depth * 3] = output_data[c];

        // Accumulate in 8.8 representation, pre-adding 0.5 for later rounding.
        accum[c] =
            (output_data[c] << 8) + 128;  // 128 = 0.5 in 8.8 representation.
      }

      // Bottom-centre margin.
      uint16 wdelta[8];
      uint16 wdelta_twice[8];
      for (int j = 0; j < (input_width - 1); ++j) {
        for (int c = 0; c < 8; ++c) {
          wdelta[c] = static_cast<uint16>(
                          input_base_ptr[c_block + c + depth * (j + 1)] -
                          input_base_ptr[c_block + c + depth * j])
                      << 4;
          wdelta_twice[c] = wdelta[c] << 1;

          accum[c] += wdelta[c];
          output_base_ptr[c_block + c + depth * j * 8 + depth * 4] =
              accum[c] >> 8;
          for (int p = 1; p < 8; ++p) {
            accum[c] += wdelta_twice[c];
            output_base_ptr[c_block + c + depth * j * 8 + depth * p +
                            depth * 4] = accum[c] >> 8;
          }
          accum[c] += wdelta[c];
        }
      }

      // Bottom-right margin corner.
      for (int c = 0; c < 8; ++c) {
        // Accumulations have pre-added 0.5 for rounding, but that is just
        // discarded and this just avoids re-loading.
        output_data[c] = accum[c] >> 8;
        TFLITE_DCHECK_EQ(
            output_data[c],
            input_base_ptr[c_block + c + depth * (input_width - 1)]);
        output_base_ptr[c_block + c + depth * (input_width - 1) * 8 +
                        depth * 4] = output_data[c];
        output_base_ptr[c_block + c + depth * (input_width - 1) * 8 +
                        depth * 4 + depth] = output_data[c];
        output_base_ptr[c_block + c + depth * (input_width - 1) * 8 +
                        depth * 4 + depth * 2] = output_data[c];
        output_base_ptr[c_block + c + depth * (input_width - 1) * 8 +
                        depth * 4 + depth * 3] = output_data[c];
      }
    }
    // Fill out remainder of bottom margin.
    std::memcpy(output_base_ptr + output_row_stride, output_base_ptr,
                output_row_stride * sizeof(uint8));
    std::memcpy(output_base_ptr + output_row_stride * 2, output_base_ptr,
                output_row_stride * sizeof(uint8));
    std::memcpy(output_base_ptr + output_row_stride * 3, output_base_ptr,
                output_row_stride * sizeof(uint8));

#endif  // USE_NEON
  }
}  // NOLINT(readability/fn_size)

}  // namespace resize_bilinear

#ifdef USE_NEON
inline void ResizeBilinearKernel(const float* input_ptr, int32 depth,
                                 float scale, float* output_ptr) {
  int ic = 0;
  // Handle 32 input channels at a time.
  for (; ic <= depth - 32; ic += 32) {
    float32x4x2_t input[4];
    for (int i = 0; i < 4; i++) {
      input[i].val[0] = vld1q_f32(input_ptr + 8 * i);
      input[i].val[1] = vld1q_f32(input_ptr + 8 * i + 4);
    }
    float32x4x2_t acc[4];
    for (int i = 0; i < 4; i++) {
      acc[i].val[0] = vld1q_f32(output_ptr + 8 * i);
      acc[i].val[1] = vld1q_f32(output_ptr + 8 * i + 4);
    }
    for (int i = 0; i < 4; i++) {
      acc[i].val[0] = vmlaq_n_f32(acc[i].val[0], input[i].val[0], scale);
      acc[i].val[1] = vmlaq_n_f32(acc[i].val[1], input[i].val[1], scale);
    }
    for (int i = 0; i < 4; i++) {
      vst1q_f32(output_ptr, acc[i].val[0]);
      vst1q_f32(output_ptr + 4, acc[i].val[1]);
      output_ptr += 8;
    }
    input_ptr += 32;
  }
  // Handle 16 input channels at a time.
  for (; ic <= depth - 16; ic += 16) {
    float32x4x2_t input[2];
    for (int i = 0; i < 2; i++) {
      input[i].val[0] = vld1q_f32(input_ptr + 8 * i);
      input[i].val[1] = vld1q_f32(input_ptr + 8 * i + 4);
    }
    float32x4x2_t acc[2];
    for (int i = 0; i < 2; i++) {
      acc[i].val[0] = vld1q_f32(output_ptr + 8 * i);
      acc[i].val[1] = vld1q_f32(output_ptr + 8 * i + 4);
    }
    for (int i = 0; i < 2; i++) {
      acc[i].val[0] = vmlaq_n_f32(acc[i].val[0], input[i].val[0], scale);
      acc[i].val[1] = vmlaq_n_f32(acc[i].val[1], input[i].val[1], scale);
    }
    for (int i = 0; i < 2; i++) {
      vst1q_f32(output_ptr, acc[i].val[0]);
      vst1q_f32(output_ptr + 4, acc[i].val[1]);
      output_ptr += 8;
    }
    input_ptr += 16;
  }
  // Handle 8 input channels at a time.
  for (; ic <= depth - 8; ic += 8) {
    float32x4x2_t input;
    input.val[0] = vld1q_f32(input_ptr);
    input.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t acc;
    acc.val[0] = vld1q_f32(output_ptr);
    acc.val[1] = vld1q_f32(output_ptr + 4);
    acc.val[0] = vmlaq_n_f32(acc.val[0], input.val[0], scale);
    acc.val[1] = vmlaq_n_f32(acc.val[1], input.val[1], scale);

    vst1q_f32(output_ptr, acc.val[0]);
    vst1q_f32(output_ptr + 4, acc.val[1]);

    input_ptr += 8;
    output_ptr += 8;
  }
  // Handle 4 input channels at a time.
  for (; ic <= depth - 4; ic += 4) {
    float32x4_t input = vld1q_f32(input_ptr);
    float32x4_t acc = vld1q_f32(output_ptr);

    acc = vmlaq_n_f32(acc, input, scale);
    vst1q_f32(output_ptr, acc);

    input_ptr += 4;
    output_ptr += 4;
  }
  // Handle 1 input channel at a time.
  for (; ic < depth; ic++) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}
#else
inline void ResizeBilinearKernel(const float* input_ptr, int32 depth,
                                 float scale, float* output_ptr) {
  for (int32 i = 0; i < depth; i++) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}
#endif

inline void ResizeBilinearKernel2x2(int32 x0, int32 x1, int32 y0, int32 y1,
                                    int32 x, int32 y, int32 depth, int32 batch,
                                    const RuntimeShape& input_shape,
                                    const float* input_data,
                                    const RuntimeShape& output_shape,
                                    float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int32 input_width = input_shape.Dims(2);
  const int32 output_width = output_shape.Dims(2);

  const int32 input_x_offset = (x1 - x0) * depth;
  const int32 input_y_offset = (y1 - y0) * depth * input_width;
  const int32 output_x_offset = depth;
  const int32 output_y_offset = depth * output_width;

#ifdef USE_NEON
  TFLITE_DCHECK(x1 >= x0);
  TFLITE_DCHECK(y1 >= y0);

  int ic = 0;
  // Handle 8 input channels at a time.
  for (; ic <= depth - 8; ic += 8) {
    const float* input_ptr = nullptr;

    float32x4x2_t x0y0;
    input_ptr = &input_data[Offset(input_shape, batch, y0, x0, ic)];
    x0y0.val[0] = vld1q_f32(input_ptr);
    x0y0.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x1y0;
    input_ptr += input_x_offset;
    x1y0.val[0] = vld1q_f32(input_ptr);
    x1y0.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x0y1;
    input_ptr += -input_x_offset + input_y_offset;
    x0y1.val[0] = vld1q_f32(input_ptr);
    x0y1.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x1y1;
    input_ptr += input_x_offset;
    x1y1.val[0] = vld1q_f32(input_ptr);
    x1y1.val[1] = vld1q_f32(input_ptr + 4);

    // Top left corner.
    float* output_ptr = &output_data[Offset(output_shape, batch, y, x, ic)];
    vst1q_f32(output_ptr, x0y0.val[0]);
    vst1q_f32(output_ptr + 4, x0y0.val[1]);

    // Top right corner.
    output_ptr += output_x_offset;
    float32x4x2_t tr;
    tr.val[0] = vaddq_f32(x0y0.val[0], x1y0.val[0]);
    tr.val[1] = vaddq_f32(x0y0.val[1], x1y0.val[1]);
    tr.val[0] = vmulq_n_f32(tr.val[0], 0.5f);
    tr.val[1] = vmulq_n_f32(tr.val[1], 0.5f);

    vst1q_f32(output_ptr, tr.val[0]);
    vst1q_f32(output_ptr + 4, tr.val[1]);

    // Bottom left corner.
    output_ptr += -output_x_offset + output_y_offset;
    float32x4x2_t bl;
    bl.val[0] = vaddq_f32(x0y0.val[0], x0y1.val[0]);
    bl.val[1] = vaddq_f32(x0y0.val[1], x0y1.val[1]);
    bl.val[0] = vmulq_n_f32(bl.val[0], 0.5f);
    bl.val[1] = vmulq_n_f32(bl.val[1], 0.5f);
    vst1q_f32(output_ptr, bl.val[0]);
    vst1q_f32(output_ptr + 4, bl.val[1]);

    // Bottom right corner.
    output_ptr += output_x_offset;
    float32x4x2_t br;
    br.val[0] = vaddq_f32(x1y0.val[0], x1y1.val[0]);
    br.val[1] = vaddq_f32(x1y0.val[1], x1y1.val[1]);
    br.val[0] = vmlaq_n_f32(bl.val[0], br.val[0], 0.5f);
    br.val[1] = vmlaq_n_f32(bl.val[1], br.val[1], 0.5f);
    br.val[0] = vmulq_n_f32(br.val[0], 0.5f);
    br.val[1] = vmulq_n_f32(br.val[1], 0.5f);
    vst1q_f32(output_ptr, br.val[0]);
    vst1q_f32(output_ptr + 4, br.val[1]);
  }
  // Handle 4 input channels at a time.
  for (; ic <= depth - 4; ic += 4) {
    const float* input_ptr =
        &input_data[Offset(input_shape, batch, y0, x0, ic)];
    float32x4_t x0y0 = vld1q_f32(input_ptr);
    float32x4_t x1y0 = vld1q_f32(input_ptr + input_x_offset);
    float32x4_t x0y1 = vld1q_f32(input_ptr + input_y_offset);
    float32x4_t x1y1 = vld1q_f32(input_ptr + input_x_offset + input_y_offset);

    // Top left corner.
    float* output_ptr = &output_data[Offset(output_shape, batch, y, x, ic)];
    vst1q_f32(output_ptr, x0y0);

    // Top right corner.
    output_ptr += output_x_offset;
    float32x4_t tr = vaddq_f32(x0y0, x1y0);
    tr = vmulq_n_f32(tr, 0.5f);
    vst1q_f32(output_ptr, tr);

    // Bottom left corner.
    output_ptr += -output_x_offset + output_y_offset;
    float32x4_t bl = vaddq_f32(x0y0, x0y1);
    bl = vmulq_n_f32(bl, 0.5f);
    vst1q_f32(output_ptr, bl);

    // Bottom right corner.
    output_ptr += output_x_offset;
    float32x4_t br = vaddq_f32(x1y0, x1y1);
    br = vmlaq_n_f32(bl, br, 0.5f);
    br = vmulq_n_f32(br, 0.5f);
    vst1q_f32(output_ptr, br);
  }
  // Handle one input channel at a time.
  for (; ic < depth; ic++) {
    const int32 input_offset = Offset(input_shape, batch, y0, x0, ic);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32 output_offset = Offset(output_shape, batch, y, x, ic);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
        (output + ((x1y0 + x1y1) / 2)) / 2;
  }
#else
  for (int ch = 0; ch < depth; ch++) {
    const int32 input_offset = Offset(input_shape, batch, y0, x0, ch);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32 output_offset = Offset(output_shape, batch, y, x, ch);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
        (output + ((x1y0 + x1y1) / 2)) / 2;
  }
#endif
}

inline void ResizeBilinear2x2(int32 batches, int32 input_height,
                              int32 input_width, int32 depth,
                              int32 output_height, int32 output_width,
                              const RuntimeShape& input_shape,
                              const float* input_data,
                              const RuntimeShape& output_shape,
                              float* output_data) {
  for (int b = 0; b < batches; b++) {
    for (int y0 = 0, y = 0; y <= output_height - 2; y += 2, y0++) {
      for (int x0 = 0, x = 0; x <= output_width - 2; x += 2, x0++) {
        int32 x1 = std::min(x0 + 1, input_width - 1);
        int32 y1 = std::min(y0 + 1, input_height - 1);
        ResizeBilinearKernel2x2(x0, x1, y0, y1, x, y, depth, b, input_shape,
                                input_data, output_shape, output_data);
      }
    }
  }
}

inline void ResizeBilinearGeneric(
    int32 batches, int32 input_height, int32 input_width, int32 depth,
    int32 output_height, int32 output_width, float height_scale,
    float width_scale, const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& output_shape, float* output_data,
    const bool half_pixel_centers) {
  memset(output_data, 0,
         batches * output_height * output_width * depth * sizeof(float));

  int32 output_offset = 0;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y;
      int32 y0, y1;
      reference_ops::ComputeInterpolationValues(
          y, height_scale, half_pixel_centers, input_height, &input_y, &y0,
          &y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x;
        int32 x0, x1;
        reference_ops::ComputeInterpolationValues(
            x, width_scale, half_pixel_centers, input_width, &input_x, &x0,
            &x1);
        float* output_ptr = &output_data[output_offset];

        // Run kernel on the 4 corners of the bilinear resize algorithm.
        int32 input_offset = Offset(input_shape, b, y0, x0, 0);
        float scale = (1 - (input_y - y0)) * (1 - (input_x - x0));
        const float* input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y0, x1, 0);
        scale = (1 - (input_y - y0)) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y1, x0, 0);
        scale = (input_y - y0) * (1 - (input_x - x0));
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y1, x1, 0);
        scale = (input_y - y0) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        output_offset += depth;
      }
    }
  }
}

template <typename T>
inline void ResizeBilinearGenericSmallChannel(
    int32 batches, int32 input_height, int32 input_width, int32 depth,
    int32 output_height, int32 output_width, float height_scale,
    float width_scale, const RuntimeShape& input_shape, const T* input_data,
    const RuntimeShape& output_shape, T* output_data,
    const bool half_pixel_centers) {
  T* output_ptr = &output_data[0];
  const float rounding_offset = std::numeric_limits<T>::is_integer ? .5f : .0f;

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y;
      int32 y0, y1;
      reference_ops::ComputeInterpolationValues(
          y, height_scale, half_pixel_centers, input_height, &input_y, &y0,
          &y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x;
        int32 x0, x1;
        reference_ops::ComputeInterpolationValues(
            x, width_scale, half_pixel_centers, input_width, &input_x, &x0,
            &x1);

        int32 input_offset[4] = {Offset(input_shape, b, y0, x0, 0),
                                 Offset(input_shape, b, y0, x1, 0),
                                 Offset(input_shape, b, y1, x0, 0),
                                 Offset(input_shape, b, y1, x1, 0)};
        float scale[4] = {(1 - (input_y - y0)) * (1 - (input_x - x0)),
                          (1 - (input_y - y0)) * (input_x - x0),
                          (input_y - y0) * (1 - (input_x - x0)),
                          (input_y - y0) * (input_x - x0)};

        for (int d = 0; d < depth; d++) {
          const T* input_ptr = &input_data[d];
          *output_ptr++ = static_cast<T>(input_ptr[input_offset[0]] * scale[0] +
                                         input_ptr[input_offset[1]] * scale[1] +
                                         input_ptr[input_offset[2]] * scale[2] +
                                         input_ptr[input_offset[3]] * scale[3] +
                                         rounding_offset);
        }
      }
    }
  }
}

inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const float* input_data,
                           const RuntimeShape& output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           float* output_data) {
  ruy::profiler::ScopeLabel label("ResizeBilinear");
  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  // Specialize for 2x2 upsample.
  if (!op_params.align_corners && !op_params.half_pixel_centers &&
      output_height == 2 * input_height && output_width == 2 * input_width) {
    ResizeBilinear2x2(batches, input_height, input_width, depth, output_height,
                      output_width, input_shape, input_data, output_shape,
                      output_data);
  } else {
    float height_scale = static_cast<float>(input_height) / output_height;
    float width_scale = static_cast<float>(input_width) / output_width;
    if (op_params.align_corners && output_height > 1) {
      height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
    }
    if (op_params.align_corners && output_width > 1) {
      width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
    }

    ResizeBilinearGeneric(batches, input_height, input_width, depth,
                          output_height, output_width, height_scale,
                          width_scale, input_shape, input_data, output_shape,
                          output_data, op_params.half_pixel_centers);
  }
}

// Note: This is not a universal quantized bilinear. It does not use int8
// or int16 arithmetic.
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const uint8* input_data,
                           const RuntimeShape& output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           uint8* output_data) {
  ruy::profiler::ScopeLabel label("ResizeBilinearUint8");
  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  if (!op_params.align_corners && op_params.half_pixel_centers &&
      ((depth % 8) == 0)) {
    const int32 scale = output_height / input_height;
    // Restricting the minimum output dimensions may not be necessary, but
    // ensures that kernels can use unrolling with minimal code size.
    if ((output_height >= 8) && (output_width >= 8) &&
        ((input_height * scale) == output_height) &&
        ((input_width * scale) == output_width)) {
      if (scale == 8) {
        resize_bilinear::ResizeBilinear888Uint8(
            batches, input_height, input_width, depth, input_data, output_data);
        return;
      }
    }
  }

  float height_scale =
      (op_params.align_corners && output_height > 1)
          ? (static_cast<float>(input_height - 1) / (output_height - 1))
          : (static_cast<float>(input_height) / output_height);

  float width_scale =
      (op_params.align_corners && output_width > 1)
          ? (static_cast<float>(input_width - 1) / (output_width - 1))
          : (static_cast<float>(input_width) / output_width);

  ResizeBilinearGenericSmallChannel<uint8>(
      batches, input_height, input_width, depth, output_height, output_width,
      height_scale, width_scale, input_shape, input_data, output_shape,
      output_data, op_params.half_pixel_centers);
}

// TODO(b/180609127) Create optimized int8 version from uint8. Call from here.
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const int8* input_data,
                           const RuntimeShape& unextended_output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           int8* output_data) {
  reference_ops::ResizeBilinearInteger(op_params, unextended_input_shape,
                                       input_data, unextended_output_size_shape,
                                       output_size_data,
                                       unextended_output_shape, output_data);
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RESIZE_BILINEAR_H_
