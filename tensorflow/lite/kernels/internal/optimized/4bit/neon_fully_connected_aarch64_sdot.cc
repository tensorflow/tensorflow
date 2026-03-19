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
#include <stdint.h>

#include <algorithm>
#include <vector>

#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/neon_fully_connected_impl.h"

namespace tflite {
namespace optimized_4bit {

#define SDOT_16_12_0 ".word 0x4e809590\n"
#define SDOT_16_8_1 ".word 0x4e819510\n"
#define SDOT_17_13_0 ".word 0x4e8095b1\n"
#define SDOT_17_9_1 ".word 0x4e819531\n"
#define SDOT_18_10_1 ".word 0x4e819552\n"
#define SDOT_18_14_0 ".word 0x4e8095d2\n"
#define SDOT_19_11_1 ".word 0x4e819573\n"
#define SDOT_19_15_0 ".word 0x4e8095f3\n"
#define SDOT_20_12_2 ".word 0x4e829594\n"
#define SDOT_20_8_0 ".word 0x4e809514\n"
#define SDOT_20_8_3 ".word 0x4e839514\n"
#define SDOT_21_13_2 ".word 0x4e8295b5\n"
#define SDOT_21_9_0 ".word 0x4e809535\n"
#define SDOT_21_9_3 ".word 0x4e839535\n"
#define SDOT_22_10_0 ".word 0x4e809556\n"
#define SDOT_22_10_3 ".word 0x4e839556\n"
#define SDOT_22_14_2 ".word 0x4e8295d6\n"
#define SDOT_23_11_0 ".word 0x4e809577\n"
#define SDOT_23_11_3 ".word 0x4e839577\n"
#define SDOT_23_15_2 ".word 0x4e8295f7\n"
#define SDOT_24_12_1 ".word 0x4e819598\n"
#define SDOT_24_8_2 ".word 0x4e829518\n"
#define SDOT_25_13_1 ".word 0x4e8195b9\n"
#define SDOT_25_9_2 ".word 0x4e829539\n"
#define SDOT_26_10_2 ".word 0x4e82955a\n"
#define SDOT_26_14_1 ".word 0x4e8195da\n"
#define SDOT_27_11_2 ".word 0x4e82957b\n"
#define SDOT_27_15_1 ".word 0x4e8195fb\n"
#define SDOT_28_12_0 ".word 0x4e80959c\n"
#define SDOT_28_8_1 ".word 0x4e81951c\n"
#define SDOT_29_13_0 ".word 0x4e8095bd\n"
#define SDOT_29_9_1 ".word 0x4e81953d\n"
#define SDOT_30_10_1 ".word 0x4e81955e\n"
#define SDOT_30_14_0 ".word 0x4e8095de\n"
#define SDOT_31_11_1 ".word 0x4e81957f\n"
#define SDOT_31_15_0 ".word 0x4e8095ff\n"

#define INNER_LOOP_PREAMBLE "1"
#define OUTER_LOOP_BEGIN "2"
#define OUTER_LOOP_END "3"
#define INNER_LOOP_BEGIN "4"
#define INNER_LOOP "5"
#define INNER_LOOP_END "6"
#define INNER_LOOP_POSTAMBLE "7"
#define END "8"

#define KERNEL_4x1                                                   \
  "dup v24.16b, %w[bit_shift]\n"                                     \
  "mov x0, %[element_ptr]\n"                                         \
  "mov x6, %[lhs_val]\n"                                             \
  "mov x1, %[rhs_val]\n"                                             \
                                                                     \
      INNER_LOOP_BEGIN                                               \
  ":\n"                                                              \
  "mov x4, x6\n"                                                     \
  "ld1 {v4.16b}, [x4], #16\n"                                        \
  "dup v16.4s, wzr\n"                                                \
  "dup v17.4s, wzr\n"                                                \
  "ld1 {v5.16b}, [x4], #16\n"                                        \
  "dup v18.4s, wzr\n"                                                \
  "dup v19.4s, wzr\n"                                                \
  "ld1 {v6.16b, v7.16b}, [x4], #32\n"                                \
  "and v8.16b, v4.16b, v24.16b\n"                                    \
  "and v9.16b, v5.16b, v24.16b\n"                                    \
  "ushr v12.16b, v4.16b, #4\n"                                       \
  "ushr v13.16b, v5.16b, #4\n"                                       \
  "ld1 {v0.16b, v1.16b}, [x1], #32\n"                                \
  "and v10.16b, v6.16b, v24.16b\n"                                   \
  "and v11.16b, v7.16b, v24.16b\n"                                   \
  "ushr v14.16b, v6.16b, #4\n"                                       \
  "ushr v15.16b, v7.16b, #4\n"                                       \
  "mov w3, %w[run_depth]\n"                                          \
  "subs w3, w3, #1\n"                                                \
  "b.ls " INNER_LOOP_END "f\n"                                       \
                                                                     \
      INNER_LOOP                                                     \
  ":\n"                                                              \
  "ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x4], #64\n"                \
                                                                     \
      SDOT_16_8_1 SDOT_17_9_1 SDOT_18_10_1 SDOT_19_11_1 SDOT_16_12_0 \
          SDOT_17_13_0 SDOT_18_14_0 SDOT_19_15_0                     \
                                                                     \
  "and v8.16b, v4.16b, v24.16b\n"                                    \
  "and v9.16b, v5.16b, v24.16b\n"                                    \
  "ushr v12.16b, v4.16b, #4\n"                                       \
  "ushr v13.16b, v5.16b, #4\n"                                       \
  "ld1 {v0.16b, v1.16b}, [x1], #32\n"                                \
  "and v10.16b, v6.16b, v24.16b\n"                                   \
  "and v11.16b, v7.16b, v24.16b\n"                                   \
  "ushr v14.16b, v6.16b, #4\n"                                       \
  "ushr v15.16b, v7.16b, #4\n"                                       \
  "subs w3, w3, #1\n"                                                \
  "b.hi " INNER_LOOP "b\n"                                           \
                                                                     \
      INNER_LOOP_END ":\n"                                           \
                                                                     \
      SDOT_16_8_1 SDOT_17_9_1 SDOT_18_10_1 SDOT_19_11_1 SDOT_16_12_0 \
          SDOT_17_13_0 SDOT_18_14_0 SDOT_19_15_0                     \
                                                                     \
  "addp v4.4s, v16.4s, v17.4s\n"                                     \
  "addp v5.4s, v18.4s, v19.4s\n"                                     \
  "addp v6.4s, v4.4s, v5.4s\n"                                       \
  "st1 {v6.4s}, [x0], #16\n"

#define KERNEL_4x2                                                           \
  "dup v24.16b, %w[bit_shift]\n"                                             \
  "mov x0, %[element_ptr]\n"                                                 \
  "mov x6, %[lhs_val]\n"                                                     \
  "mov x1, %[rhs_val]\n"                                                     \
                                                                             \
      INNER_LOOP_BEGIN                                                       \
  ":\n"                                                                      \
  "mov x4, x6\n"                                                             \
  "ld1 {v4.16b}, [x4], #16\n"                                                \
  "dup v16.4s, wzr\n"                                                        \
  "dup v17.4s, wzr\n"                                                        \
  "ld1 {v5.16b}, [x4], #16\n"                                                \
  "dup v18.4s, wzr\n"                                                        \
  "dup v19.4s, wzr\n"                                                        \
  "ld1 {v6.16b, v7.16b}, [x4], #32\n"                                        \
  "dup v20.4s, wzr\n"                                                        \
  "dup v21.4s, wzr\n"                                                        \
  "and v8.16b, v4.16b, v24.16b\n"                                            \
  "and v9.16b, v5.16b, v24.16b\n"                                            \
  "dup v22.4s, wzr\n"                                                        \
  "dup v23.4s, wzr\n"                                                        \
  "ushr v12.16b, v4.16b, #4\n"                                               \
  "ushr v13.16b, v5.16b, #4\n"                                               \
  "ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64\n"                        \
  "and v10.16b, v6.16b, v24.16b\n"                                           \
  "and v11.16b, v7.16b, v24.16b\n"                                           \
  "ushr v14.16b, v6.16b, #4\n"                                               \
  "ushr v15.16b, v7.16b, #4\n"                                               \
  "mov w3, %w[run_depth]\n"                                                  \
  "subs w3, w3, #1\n"                                                        \
  "b.ls " INNER_LOOP_END "f\n"                                               \
                                                                             \
      INNER_LOOP                                                             \
  ":\n"                                                                      \
  "ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x4], #64\n"                        \
                                                                             \
      SDOT_16_12_0 SDOT_17_13_0 SDOT_18_14_0 SDOT_19_15_0 SDOT_16_8_1        \
          SDOT_17_9_1 SDOT_18_10_1 SDOT_19_11_1 SDOT_20_12_2 SDOT_21_13_2    \
              SDOT_22_14_2 SDOT_23_15_2 SDOT_20_8_3 SDOT_21_9_3 SDOT_22_10_3 \
                  SDOT_23_11_3                                               \
                                                                             \
  "and v8.16b, v4.16b, v24.16b\n"                                            \
  "and v9.16b, v5.16b, v24.16b\n"                                            \
  "ushr v12.16b, v4.16b, #4\n"                                               \
  "ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1], #64\n"                        \
  "ushr v13.16b, v5.16b, #4\n"                                               \
  "and v10.16b, v6.16b, v24.16b\n"                                           \
  "and v11.16b, v7.16b, v24.16b\n"                                           \
  "ushr v14.16b, v6.16b, #4\n"                                               \
  "ushr v15.16b, v7.16b, #4\n"                                               \
  "subs w3, w3, #1\n"                                                        \
  "b.hi " INNER_LOOP "b\n"                                                   \
                                                                             \
      INNER_LOOP_END ":\n"                                                   \
                                                                             \
      SDOT_16_12_0 SDOT_17_13_0 SDOT_18_14_0 SDOT_19_15_0 SDOT_16_8_1        \
          SDOT_17_9_1 SDOT_18_10_1 SDOT_19_11_1 SDOT_20_12_2 SDOT_21_13_2    \
              SDOT_22_14_2 SDOT_23_15_2 SDOT_20_8_3 SDOT_21_9_3 SDOT_22_10_3 \
                  SDOT_23_11_3                                               \
                                                                             \
  "addp v4.4s, v16.4s, v17.4s\n"                                             \
  "addp v5.4s, v18.4s, v19.4s\n"                                             \
  "addp v8.4s, v20.4s, v21.4s\n"                                             \
  "addp v9.4s, v22.4s, v23.4s\n"                                             \
  "addp v6.4s, v4.4s, v5.4s\n"                                               \
  "addp v7.4s, v8.4s, v9.4s\n"                                               \
  "st1 {v6.4s, v7.4s}, [x0], #32\n"

#define KERNEL_4x4                                                    \
  "dup v3.16b, %w[bit_shift]\n"                                       \
  "mov x0, %[element_ptr]\n"                                          \
  "mov x6, %[lhs_val]\n"                                              \
  "mov x1, %[rhs_val]\n"                                              \
                                                                      \
      INNER_LOOP_BEGIN                                                \
  ":\n"                                                               \
  "mov x4, x6\n"                                                      \
  "ld1 {v4.16b}, [x4], #16\n"                                         \
  "dup v16.4s, wzr\n"                                                 \
  "dup v17.4s, wzr\n"                                                 \
  "ld1 {v5.16b}, [x4], #16\n"                                         \
  "dup v18.4s, wzr\n"                                                 \
  "dup v19.4s, wzr\n"                                                 \
  "ld1 {v6.16b, v7.16b}, [x4], #32\n"                                 \
  "and v8.16b, v4.16b, v3.16b\n"                                      \
  "and v9.16b, v5.16b, v3.16b\n"                                      \
  "dup v20.4s, wzr\n"                                                 \
  "dup v21.4s, wzr\n"                                                 \
  "dup v22.4s, wzr\n"                                                 \
  "dup v23.4s, wzr\n"                                                 \
  "dup v24.4s, wzr\n"                                                 \
  "dup v25.4s, wzr\n"                                                 \
  "dup v26.4s, wzr\n"                                                 \
  "dup v27.4s, wzr\n"                                                 \
  "dup v28.4s, wzr\n"                                                 \
  "dup v29.4s, wzr\n"                                                 \
  "dup v30.4s, wzr\n"                                                 \
  "dup v31.4s, wzr\n"                                                 \
  "ushr v12.16b, v4.16b, #4\n"                                        \
  "ushr v13.16b, v5.16b, #4\n"                                        \
  "ld1 {v0.16b, v1.16b}, [x1], #32\n"                                 \
  "and v10.16b, v6.16b, v3.16b\n"                                     \
  "and v11.16b, v7.16b, v3.16b\n"                                     \
  "ushr v14.16b, v6.16b, #4\n"                                        \
  "ushr v15.16b, v7.16b, #4\n"                                        \
  "mov w3, %w[run_depth]\n"                                           \
  "subs w3, w3, #1\n"                                                 \
  "b.ls " INNER_LOOP_END "f\n"                                        \
                                                                      \
      INNER_LOOP                                                      \
  ":\n"                                                               \
  "ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x4], #64\n"                 \
                                                                      \
      SDOT_16_12_0 SDOT_17_13_0 SDOT_18_14_0 SDOT_19_15_0             \
                                                                      \
  "ld1 {v2.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_16_8_1 SDOT_17_9_1 SDOT_18_10_1 SDOT_19_11_1               \
                                                                      \
  "ld1 {v0.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_20_12_2 SDOT_21_13_2 SDOT_22_14_2 SDOT_23_15_2             \
                                                                      \
  "ld1 {v1.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_20_8_0 SDOT_21_9_0 SDOT_22_10_0 SDOT_23_11_0               \
                                                                      \
  "ld1 {v2.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_24_12_1 SDOT_25_13_1 SDOT_26_14_1 SDOT_27_15_1             \
                                                                      \
  "ld1 {v0.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_24_8_2 SDOT_25_9_2 SDOT_26_10_2 SDOT_27_11_2               \
                                                                      \
  "ld1 {v1.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_28_12_0 SDOT_29_13_0 SDOT_30_14_0 SDOT_31_15_0 SDOT_28_8_1 \
          SDOT_29_9_1 SDOT_30_10_1 SDOT_31_11_1                       \
                                                                      \
  "ld1 {v0.16b}, [x1], #16\n"                                         \
  "and v8.16b, v4.16b, v3.16b\n"                                      \
  "and v9.16b, v5.16b, v3.16b\n"                                      \
  "ushr v12.16b, v4.16b, #4\n"                                        \
  "ushr v13.16b, v5.16b, #4\n"                                        \
  "ld1 {v1.16b}, [x1], #16\n"                                         \
  "and v10.16b, v6.16b, v3.16b\n"                                     \
  "and v11.16b, v7.16b, v3.16b\n"                                     \
  "ushr v14.16b, v6.16b, #4\n"                                        \
  "ushr v15.16b, v7.16b, #4\n"                                        \
  "subs w3, w3, #1\n"                                                 \
  "b.hi " INNER_LOOP "b\n"                                            \
                                                                      \
      INNER_LOOP_END ":\n"                                            \
                                                                      \
      SDOT_16_12_0 SDOT_17_13_0 SDOT_18_14_0 SDOT_19_15_0             \
                                                                      \
  "ld1 {v2.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_16_8_1 SDOT_17_9_1 SDOT_18_10_1 SDOT_19_11_1               \
                                                                      \
  "ld1 {v0.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_20_12_2 SDOT_21_13_2 SDOT_22_14_2 SDOT_23_15_2             \
                                                                      \
  "ld1 {v1.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_20_8_0 SDOT_21_9_0 SDOT_22_10_0 SDOT_23_11_0               \
                                                                      \
  "ld1 {v2.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_24_12_1 SDOT_25_13_1 SDOT_26_14_1 SDOT_27_15_1             \
                                                                      \
  "ld1 {v0.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_24_8_2 SDOT_25_9_2 SDOT_26_10_2 SDOT_27_11_2               \
                                                                      \
  "ld1 {v1.16b}, [x1], #16\n"                                         \
                                                                      \
      SDOT_28_12_0 SDOT_29_13_0 SDOT_30_14_0 SDOT_31_15_0 SDOT_28_8_1 \
          SDOT_29_9_1 SDOT_30_10_1 SDOT_31_11_1                       \
                                                                      \
  "addp v14.4s, v16.4s, v17.4s\n"                                     \
  "addp v15.4s, v18.4s, v19.4s\n"                                     \
  "addp v12.4s, v20.4s, v21.4s\n"                                     \
  "addp v13.4s, v22.4s, v23.4s\n"                                     \
  "addp v10.4s, v24.4s, v25.4s\n"                                     \
  "addp v11.4s, v26.4s, v27.4s\n"                                     \
  "addp v8.4s, v28.4s, v29.4s\n"                                      \
  "addp v9.4s, v30.4s, v31.4s\n"                                      \
  "addp v4.4s, v14.4s, v15.4s\n"                                      \
  "addp v5.4s, v12.4s, v13.4s\n"                                      \
  "addp v6.4s, v10.4s, v11.4s\n"                                      \
  "addp v7.4s, v8.4s, v9.4s\n"                                        \
  "st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64\n"

template <int RowsLeft, int RowsRight, int Cols>
void NeonRunKernelSDot(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
                       int lhs_layout_rows, int lhs_layout_cols,
                       int rhs_layout_rows, int rhs_layout_cols,
                       int dst_layout_rows, int dst_layout_cols) {}

template <>
void NeonRunKernelSDot<4, 1, 32>(const uint8_t* lhs, const int8_t* rhs,
                                 int32_t* dst, int lhs_layout_rows,
                                 int lhs_layout_cols, int rhs_layout_rows,
                                 int rhs_layout_cols, int dst_layout_rows,
                                 int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 1;
  const int cols = 32;
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  int32_t* element_ptr = dst;
  const int outer_rows = (clamped_end_row + rows_left - 1) / rows_left;
  const int outer_cols = (clamped_end_col + rows_right - 1) / rows_right;
  const int depth = std::min(lhs_layout_cols / cols, rhs_layout_cols / cols);
  const uint8_t bit_shift = 15;
  const int run_depth = depth;
  for (int i = start_row; i < outer_rows; ++i) {
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_val_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_val = lhs_val_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_val = rhs + right_index;
      asm volatile(KERNEL_4x1
                   : [lhs_val] "+r"(lhs_val), [rhs_val] "+r"(rhs_val),
                     [element_ptr] "+r"(element_ptr)
                   : [bit_shift] "r"(bit_shift), [run_depth] "r"(run_depth)
                   : "cc", "memory", "x0", "x1", "w2", "w3", "x4", "w5", "x6",
                     "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                     "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                     "v19", "v20", "v21", "v22", "v23", "v24");
      element_ptr += 4;
    }
  }
}

template <>
void NeonRunKernelSDot<4, 2, 32>(const uint8_t* lhs, const int8_t* rhs,
                                 int32_t* dst, int lhs_layout_rows,
                                 int lhs_layout_cols, int rhs_layout_rows,
                                 int rhs_layout_cols, int dst_layout_rows,
                                 int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 2;
  const int cols = 32;
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  int32_t* element_ptr = dst;
  const int outer_rows = (clamped_end_row + rows_left - 1) / rows_left;
  const int outer_cols = (clamped_end_col + rows_right - 1) / rows_right;
  const int depth = std::min(lhs_layout_cols / cols, rhs_layout_cols / cols);
  const uint8_t bit_shift = 15;
  const int run_depth = depth;
  for (int i = start_row; i < outer_rows; ++i) {
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_val_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_val = lhs_val_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_val = rhs + right_index;
      asm volatile(KERNEL_4x2
                   : [lhs_val] "+r"(lhs_val), [rhs_val] "+r"(rhs_val),
                     [element_ptr] "+r"(element_ptr)
                   : [bit_shift] "r"(bit_shift), [run_depth] "r"(run_depth)
                   : "cc", "memory", "x0", "x1", "w2", "w3", "x4", "w5", "x6",
                     "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                     "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                     "v19", "v20", "v21", "v22", "v23", "v24");
      element_ptr += 8;
    }
  }
}

template <>
void NeonRunKernelSDot<4, 4, 32>(const uint8_t* lhs, const int8_t* rhs,
                                 int32_t* dst, int lhs_layout_rows,
                                 int lhs_layout_cols, int rhs_layout_rows,
                                 int rhs_layout_cols, int dst_layout_rows,
                                 int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 4;
  const int cols = 32;
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  int32_t* element_ptr = dst;
  const int outer_rows = (clamped_end_row + rows_left - 1) / rows_left;
  const int outer_cols = (clamped_end_col + rows_right - 1) / rows_right;
  const int depth = std::min(lhs_layout_cols / cols, rhs_layout_cols / cols);
  const uint8_t bit_shift = 15;
  const int run_depth = depth;
  for (int i = start_row; i < outer_rows; ++i) {
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_val_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_val = lhs_val_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_val = rhs + right_index;
      asm volatile(KERNEL_4x4
                   : [lhs_val] "+r"(lhs_val), [rhs_val] "+r"(rhs_val),
                     [element_ptr] "+r"(element_ptr)
                   : [bit_shift] "r"(bit_shift), [run_depth] "r"(run_depth)
                   : "cc", "memory", "x0", "x1", "w2", "w3", "x4", "w5", "x6",
                     "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                     "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                     "v19", "v20", "v21", "v22", "v23", "v24");
      element_ptr += 16;
    }
  }
}

#undef INNER_LOOP_PREAMBLE
#undef OUTER_LOOP_BEGIN
#undef OUTER_LOOP_END
#undef INNER_LOOP_BEGIN
#undef INNER_LOOP
#undef INNER_LOOP_END
#undef INNER_LOOP_POSTAMBLE
#undef END

#undef SDOT_16_12_0
#undef SDOT_16_8_1
#undef SDOT_17_13_0
#undef SDOT_17_9_1
#undef SDOT_18_10_1
#undef SDOT_18_14_0
#undef SDOT_19_11_1
#undef SDOT_19_15_0
#undef SDOT_20_12_2
#undef SDOT_20_8_0
#undef SDOT_20_8_3
#undef SDOT_21_13_2
#undef SDOT_21_9_0
#undef SDOT_21_9_3
#undef SDOT_22_10_0
#undef SDOT_22_10_3
#undef SDOT_22_14_2
#undef SDOT_23_11_0
#undef SDOT_23_11_3
#undef SDOT_23_15_2
#undef SDOT_24_12_1
#undef SDOT_24_8_2
#undef SDOT_25_13_1
#undef SDOT_25_9_2
#undef SDOT_26_10_2
#undef SDOT_26_14_1
#undef SDOT_27_11_2
#undef SDOT_27_15_1
#undef SDOT_28_12_0
#undef SDOT_28_8_1
#undef SDOT_29_13_0
#undef SDOT_29_9_1
#undef SDOT_30_10_1
#undef SDOT_30_14_0
#undef SDOT_31_11_1
#undef SDOT_31_15_0

}  // namespace optimized_4bit
}  // namespace tflite
#endif  // defined(FC_4BIT_NEON)...
