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
#include <vector>

#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/fully_connected_common.h"
#include "tensorflow/lite/kernels/internal/optimized/4bit/neon_fully_connected_impl.h"

namespace tflite {
namespace optimized_4bit {

#define INNER_LOOP_PREAMBLE "1"
#define OUTER_LOOP_BEGIN "2"
#define OUTER_LOOP_END "3"
#define INNER_LOOP_BEGIN "4"
#define INNER_LOOP "5"
#define INNER_LOOP_END "6"
#define INNER_LOOP_POSTAMBLE "7"
#define END "8"

#define KERNEL_4x1                       \
  "mov r8, #0xf\n"                       \
  "vdup.8 d28, r8\n"                     \
  "mov r0, %[element_ptr]\n"             \
  "mov r6, %[lhs_val]\n"                 \
  "mov r8, #0\n"                         \
  "mov r1, %[rhs_val]\n"                 \
                                         \
      INNER_LOOP_BEGIN                   \
  ":\n"                                  \
  "mov r4, r6\n"                         \
  "vld1.8 {d8, d9}, [r4]!\n"             \
  "vdup.8 q0, r8\n"                      \
  "vdup.8 q1, r8\n"                      \
  "vld1.8 {d10, d11}, [r4]!\n"           \
  "vdup.8 q2, r8\n"                      \
  "vdup.8 q3, r8\n"                      \
  "vld1.8 {d12, d13}, [r4]!\n"           \
  "vand d16, d8, d28\n"                  \
  "vand d17, d9, d28\n"                  \
  "vand d18, d10, d28\n"                 \
  "vand d19, d11, d28\n"                 \
  "vld1.8 {d14, d15}, [r4]!\n"           \
  "vshr.u8 q4, q4, #4\n"                 \
  "vshr.u8 q5, q5, #4\n"                 \
  "vld1.8 {d24, d25}, [r1]!\n"           \
  "vand d20, d12, d28\n"                 \
  "vand d21, d13, d28\n"                 \
  "vand d22, d14, d28\n"                 \
  "vand d23, d15, d28\n"                 \
  "vld1.8 {d26, d27}, [r1]!\n"           \
  "vshr.u8 q6, q6, #4\n"                 \
  "vshr.u8 q7, q7, #4\n"                 \
  "mov r3, %[run_depth]\n"               \
  "subs r3, r3, #1\n"                    \
  "bls " INNER_LOOP_END "f\n"            \
                                         \
      INNER_LOOP                         \
  ":\n"                                  \
                                         \
  "vmlal.s8 q0, d8, d24\n"               \
  "vmlal.s8 q1, d10, d24\n"              \
  "vmlal.s8 q0, d9, d25\n"               \
  "vmlal.s8 q1, d11, d25\n"              \
  "vld1.8 {d8, d9, d10, d11}, [r4]!\n"   \
                                         \
  "vmlal.s8 q2, d12, d24\n"              \
  "vmlal.s8 q3, d14, d24\n"              \
  "vmlal.s8 q2, d13, d25\n"              \
  "vmlal.s8 q3, d15, d25\n"              \
                                         \
  "vld1.8 {d12, d13, d14, d15}, [r4]!\n" \
                                         \
  "vmlal.s8 q0, d16, d26\n"              \
  "vmlal.s8 q1, d18, d26\n"              \
  "vmlal.s8 q2, d20, d26\n"              \
  "vmlal.s8 q3, d22, d26\n"              \
                                         \
  "vmlal.s8 q0, d17, d27\n"              \
  "vmlal.s8 q1, d19, d27\n"              \
  "vmlal.s8 q2, d21, d27\n"              \
  "vmlal.s8 q3, d23, d27\n"              \
                                         \
  "vld1.8 {d24, d25, d26, d27}, [r1]!\n" \
                                         \
  "vand d16, d8, d28\n"                  \
  "vand d17, d9, d28\n"                  \
  "vand d18, d10, d28\n"                 \
  "vand d19, d11, d28\n"                 \
  "vand d20, d12, d28\n"                 \
  "vand d21, d13, d28\n"                 \
  "vand d22, d14, d28\n"                 \
  "vand d23, d15, d28\n"                 \
  "vshr.u8 q4, q4, #4\n"                 \
  "vshr.u8 q5, q5, #4\n"                 \
  "vshr.u8 q6, q6, #4\n"                 \
  "vshr.u8 q7, q7, #4\n"                 \
                                         \
  "subs r3, r3, #1\n"                    \
  "bhi " INNER_LOOP "b\n"                \
                                         \
      INNER_LOOP_END                     \
  ":\n"                                  \
                                         \
  "vmlal.s8 q0, d8, d24\n"               \
  "vmlal.s8 q1, d10, d24\n"              \
  "vmlal.s8 q2, d12, d24\n"              \
  "vmlal.s8 q3, d14, d24\n"              \
                                         \
  "vmlal.s8 q0, d9, d25\n"               \
  "vmlal.s8 q1, d11, d25\n"              \
  "vmlal.s8 q2, d13, d25\n"              \
  "vmlal.s8 q3, d15, d25\n"              \
                                         \
  "vmlal.s8 q0, d16, d26\n"              \
  "vmlal.s8 q1, d18, d26\n"              \
  "vmlal.s8 q2, d20, d26\n"              \
  "vmlal.s8 q3, d22, d26\n"              \
                                         \
  "vmlal.s8 q0, d17, d27\n"              \
  "vmlal.s8 q1, d19, d27\n"              \
  "vmlal.s8 q2, d21, d27\n"              \
  "vmlal.s8 q3, d23, d27\n"              \
                                         \
  "vpaddl.s16 q4, q0\n"                  \
  "vpaddl.s16 q5, q1\n"                  \
  "vpaddl.s16 q6, q2\n"                  \
  "vpaddl.s16 q7, q3\n"                  \
                                         \
  "vpadd.i32 d0, d8, d9\n"               \
  "vpadd.i32 d1, d10, d11\n"             \
  "vpadd.i32 d2, d12, d13\n"             \
  "vpadd.i32 d3, d14, d15\n"             \
                                         \
  "vpadd.i32 d4, d0, d1\n"               \
  "vpadd.i32 d5, d2, d3\n"               \
  "vst1.32 {d4, d5}, [r0]!\n"

template <int RowsLeft, int RowsRight, int Cols>
void NeonRunKernelNoSDot(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
                         int lhs_layout_rows, int lhs_layout_cols,
                         int rhs_layout_rows, int rhs_layout_cols,
                         int dst_layout_rows, int dst_layout_cols) {}

template <>
void NeonRunKernelNoSDot<4, 1, 32>(const uint8_t* lhs, const int8_t* rhs,
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
    const int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_val_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_val = lhs_val_data;
      const int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_val = rhs + right_index;
      asm volatile(KERNEL_4x1
                   :
                   : [lhs_val] "r"(lhs_val), [rhs_val] "r"(rhs_val),
                     [element_ptr] "r"(element_ptr), [bit_shift] "r"(bit_shift),
                     [run_depth] "r"(run_depth)
                   : "cc", "memory", "r0", "r1", "r2", "r3", "r4", "r5", "r6",
                     "r8", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
                     "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
                     "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
                     "d25", "d26", "d27", "d28", "d29", "d30", "d31", "q0",
                     "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                     "q10", "q11", "q12", "q13", "q14");
      element_ptr += 4;
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

}  // namespace optimized_4bit
}  // namespace tflite
#endif  // defined(FC_4BIT_NEON) ...
