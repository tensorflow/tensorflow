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

#if defined(FC_4BIT_NEON) && (defined(__ARM_NEON__) || defined(__ARM_NEON))

#include <stdint.h>

#include <algorithm>

#include "tensorflow/lite/kernels/internal/optimized/4bit/neon_fully_connected_impl.h"

#define DOTPROD_ATTRIBUTE __attribute__((target("dotprod")))

namespace tflite {
namespace optimized_4bit {

template <int RowsLeft, int RowsRight, int Cols>
void NeonRunKernelSDot(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
                       int lhs_layout_rows, int lhs_layout_cols,
                       int rhs_layout_rows, int rhs_layout_cols,
                       int dst_layout_rows, int dst_layout_cols);

template <>
DOTPROD_ATTRIBUTE void NeonRunKernelSDot<4, 1, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 1;
  const int cols = 32;
  const int start_row = 0;
  const int start_col = 0;
  const int end_row = lhs_layout_rows;
  const int end_col = rhs_layout_rows;
  const int clamped_end_row = std::min(end_row, dst_layout_cols);
  const int clamped_end_col = std::min(end_col, dst_layout_rows);
  const int outer_rows = (clamped_end_row + rows_left - 1) / rows_left;
  const int outer_cols = (clamped_end_col + rows_right - 1) / rows_right;
  const int depth = std::min(lhs_layout_cols / cols, rhs_layout_cols / cols);
  for (int i = start_row; i < outer_rows; ++i) {
    const int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_ptr_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_ptr = lhs_ptr_data;
      const int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_ptr = rhs + right_index;
      int run_depth = depth;
      asm(R"asm(
          vmov.i8 q14, #15
          vld1.8 {q4}, [%[lhs_ptr]]!
          vmov.i32 q0, #0
          vmov.i32 q1, #0
          vld1.8 {q5}, [%[lhs_ptr]]!
          vmov.i32 q2, #0
          vmov.i32 q3, #0
          vld1.8 {q6}, [%[lhs_ptr]]!
          vand q8, q4, q14
          vand q9, q5, q14
          vld1.8 {q7}, [%[lhs_ptr]]!
          vshr.u8 q4, q4, #4
          vshr.u8 q5, q5, #4
          vld1.8 {q12}, [%[rhs_ptr]]!
          vand q10, q6, q14
          vand q11, q7, q14
          vld1.8 {q13}, [%[rhs_ptr]]!
          vshr.u8 q6, q6, #4
          vshr.u8 q7, q7, #4
          subs %[run_depth], %[run_depth], #1
          bls 1f /* skip loop */
            0: /* loop start */
            vsdot.s8 q0, q4, q12
            vsdot.s8 q0, q8, q13
            vsdot.s8 q1, q5, q12
            vsdot.s8 q1, q9, q13
            vld1.8 {q4, q5}, [%[lhs_ptr]]!
            vsdot.s8 q2, q6, q12
            vsdot.s8 q2, q10, q13
            vsdot.s8 q3, q7, q12
            vsdot.s8 q3, q11, q13
            vld1.8 {q6, q7}, [%[lhs_ptr]]!
            vld1.8 {q12, q13}, [%[rhs_ptr]]!
            vand q8, q4, q14
            vand q9, q5, q14
            vand q10, q6, q14
            vand q11, q7, q14
            vshr.u8 q4, q4, #4
            vshr.u8 q5, q5, #4
            vshr.u8 q6, q6, #4
            vshr.u8 q7, q7, #4
            subs %[run_depth], %[run_depth], #1
            bhi 0b /* loop branch */
          1: /* loop end */
          vsdot.s8 q0, q4, q12
          vsdot.s8 q0, q8, q13
          vsdot.s8 q1, q5, q12
          vsdot.s8 q1, q9, q13
          vsdot.s8 q2, q6, q12
          vsdot.s8 q2, q10, q13
          vsdot.s8 q3, q7, q12
          vsdot.s8 q3, q11, q13
          vpadd.i32 d0, d0, d1
          vpadd.i32 d1, d2, d3
          vpadd.i32 d2, d4, d5
          vpadd.i32 d3, d6, d7
          vpadd.i32 d4, d0, d1
          vpadd.i32 d5, d2, d3
          vst1.32 {d4, d5}, [%[dst]]!
          )asm"
          : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [dst] "+r"(dst),
            [run_depth] "+r"(run_depth)
          :
          : "cc", "memory", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
            "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
            "q11", "q12", "q13", "q14");
    }
  }
}

}  // namespace optimized_4bit
}  // namespace tflite
#endif  // defined(FC_4BIT_NEON) && (defined(__ARM_NEON__) ||
        // defined(__ARM_NEON))
