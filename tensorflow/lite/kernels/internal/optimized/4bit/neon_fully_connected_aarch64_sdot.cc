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
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_ptr_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_ptr = lhs_ptr_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_ptr = rhs + right_index;
      int run_depth = depth;
      asm(R"asm(
          movi v24.16b, #15
          ld1 {v4.16b}, [%[lhs_ptr]], #16
          movi v16.4s, #0
          movi v17.4s, #0
          ld1 {v5.16b}, [%[lhs_ptr]], #16
          movi v18.4s, #0
          movi v19.4s, #0
          ld1 {v6.16b, v7.16b}, [%[lhs_ptr]], #32
          and v8.16b, v4.16b, v24.16b
          and v9.16b, v5.16b, v24.16b
          ushr v12.16b, v4.16b, #4
          ushr v13.16b, v5.16b, #4
          ld1 {v0.16b, v1.16b}, [%[rhs_ptr]], #32
          and v10.16b, v6.16b, v24.16b
          and v11.16b, v7.16b, v24.16b
          ushr v14.16b, v6.16b, #4
          ushr v15.16b, v7.16b, #4
          subs %w[run_depth], %w[run_depth], #1
          b.ls 1f /* skip loop */
            0: /* loop start */
            ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [%[lhs_ptr]], #64
            sdot v16.4s, v8.16b, v1.16b
            sdot v17.4s, v9.16b, v1.16b
            sdot v18.4s, v10.16b, v1.16b
            sdot v19.4s, v11.16b, v1.16b
            sdot v16.4s, v12.16b, v0.16b
            sdot v17.4s, v13.16b, v0.16b
            sdot v18.4s, v14.16b, v0.16b
            sdot v19.4s, v15.16b, v0.16b
            and v8.16b, v4.16b, v24.16b
            and v9.16b, v5.16b, v24.16b
            ushr v12.16b, v4.16b, #4
            ushr v13.16b, v5.16b, #4
            ld1 {v0.16b, v1.16b}, [%[rhs_ptr]], #32
            and v10.16b, v6.16b, v24.16b
            and v11.16b, v7.16b, v24.16b
            ushr v14.16b, v6.16b, #4
            ushr v15.16b, v7.16b, #4
            subs %w[run_depth], %w[run_depth], #1
            b.hi 0b /* loop branch */
          1: /* loop end */
          sdot v16.4s, v8.16b, v1.16b
          sdot v17.4s, v9.16b, v1.16b
          sdot v18.4s, v10.16b, v1.16b
          sdot v19.4s, v11.16b, v1.16b
          sdot v16.4s, v12.16b, v0.16b
          sdot v17.4s, v13.16b, v0.16b
          sdot v18.4s, v14.16b, v0.16b
          sdot v19.4s, v15.16b, v0.16b
          addp v4.4s, v16.4s, v17.4s
          addp v5.4s, v18.4s, v19.4s
          addp v6.4s, v4.4s, v5.4s
          st1 {v6.4s}, [%[dst]], #16
          )asm"
          : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [dst] "+r"(dst),
            [run_depth] "+r"(run_depth)
          :
          : "cc", "memory", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
            "v19", "v24");
    }
  }
}

// Note: NeonRunKernelSDot<4, 2, 32> does not mutate registers v25-v31 in its
// inline assembly block, so they are intentionally omitted from the clobber
// list to avoid redundant register preservation.
template <>
DOTPROD_ATTRIBUTE void NeonRunKernelSDot<4, 2, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 2;
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
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_ptr_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_ptr = lhs_ptr_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_ptr = rhs + right_index;
      int run_depth = depth;
      asm(R"asm(
          movi v24.16b, #15
          ld1 {v4.16b}, [%[lhs_ptr]], #16
          movi v16.4s, #0
          movi v17.4s, #0
          ld1 {v5.16b}, [%[lhs_ptr]], #16
          movi v18.4s, #0
          movi v19.4s, #0
          ld1 {v6.16b, v7.16b}, [%[lhs_ptr]], #32
          movi v20.4s, #0
          movi v21.4s, #0
          and v8.16b, v4.16b, v24.16b
          and v9.16b, v5.16b, v24.16b
          movi v22.4s, #0
          movi v23.4s, #0
          ushr v12.16b, v4.16b, #4
          ushr v13.16b, v5.16b, #4
          ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%[rhs_ptr]], #64
          and v10.16b, v6.16b, v24.16b
          and v11.16b, v7.16b, v24.16b
          ushr v14.16b, v6.16b, #4
          ushr v15.16b, v7.16b, #4
          subs %w[run_depth], %w[run_depth], #1
          b.ls 1f /* skip loop */
            0: /* loop start */
            ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [%[lhs_ptr]], #64
            sdot v16.4s, v12.16b, v0.16b
            sdot v17.4s, v13.16b, v0.16b
            sdot v18.4s, v14.16b, v0.16b
            sdot v19.4s, v15.16b, v0.16b
            sdot v16.4s, v8.16b, v1.16b
            sdot v17.4s, v9.16b, v1.16b
            sdot v18.4s, v10.16b, v1.16b
            sdot v19.4s, v11.16b, v1.16b
            sdot v20.4s, v12.16b, v2.16b
            sdot v21.4s, v13.16b, v2.16b
            sdot v22.4s, v14.16b, v2.16b
            sdot v23.4s, v15.16b, v2.16b
            sdot v20.4s, v8.16b, v3.16b
            sdot v21.4s, v9.16b, v3.16b
            sdot v22.4s, v10.16b, v3.16b
            sdot v23.4s, v11.16b, v3.16b
            and v8.16b, v4.16b, v24.16b
            and v9.16b, v5.16b, v24.16b
            ushr v12.16b, v4.16b, #4
            ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%[rhs_ptr]], #64
            ushr v13.16b, v5.16b, #4
            and v10.16b, v6.16b, v24.16b
            and v11.16b, v7.16b, v24.16b
            ushr v14.16b, v6.16b, #4
            ushr v15.16b, v7.16b, #4
            subs %w[run_depth], %w[run_depth], #1
            b.hi 0b /* loop branch */
          1: /* loop end */
          sdot v16.4s, v12.16b, v0.16b
          sdot v17.4s, v13.16b, v0.16b
          sdot v18.4s, v14.16b, v0.16b
          sdot v19.4s, v15.16b, v0.16b
          sdot v16.4s, v8.16b, v1.16b
          sdot v17.4s, v9.16b, v1.16b
          sdot v18.4s, v10.16b, v1.16b
          sdot v19.4s, v11.16b, v1.16b
          sdot v20.4s, v12.16b, v2.16b
          sdot v21.4s, v13.16b, v2.16b
          sdot v22.4s, v14.16b, v2.16b
          sdot v23.4s, v15.16b, v2.16b
          sdot v20.4s, v8.16b, v3.16b
          sdot v21.4s, v9.16b, v3.16b
          sdot v22.4s, v10.16b, v3.16b
          sdot v23.4s, v11.16b, v3.16b
          addp v4.4s, v16.4s, v17.4s
          addp v5.4s, v18.4s, v19.4s
          addp v8.4s, v20.4s, v21.4s
          addp v9.4s, v22.4s, v23.4s
          addp v6.4s, v4.4s, v5.4s
          addp v7.4s, v8.4s, v9.4s
          st1 {v6.4s, v7.4s}, [%[dst]], #32
          )asm"
          : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [dst] "+r"(dst),
            [run_depth] "+r"(run_depth)
          :
          : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
            "v18", "v19", "v20", "v21", "v22", "v23", "v24");
    }
  }
}

template <>
DOTPROD_ATTRIBUTE void NeonRunKernelSDot<4, 4, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst, int lhs_layout_rows,
    int lhs_layout_cols, int rhs_layout_rows, int rhs_layout_cols,
    int dst_layout_rows, int dst_layout_cols) {
  const int rows_left = 4;
  const int rows_right = 4;
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
    int left_index = i * rows_left * lhs_layout_cols / 2;
    const uint8_t* lhs_ptr_data = lhs + left_index;
    for (int j = start_col; j < outer_cols; ++j) {
      const uint8_t* lhs_ptr = lhs_ptr_data;
      int right_index = j * rows_right * rhs_layout_cols;
      const int8_t* rhs_ptr = rhs + right_index;
      int run_depth = depth;
      asm(R"asm(
          movi v3.16b, #15
          ld1 {v4.16b}, [%[lhs_ptr]], #16
          movi v16.4s, #0
          movi v17.4s, #0
          ld1 {v5.16b}, [%[lhs_ptr]], #16
          movi v18.4s, #0
          movi v19.4s, #0
          ld1 {v6.16b, v7.16b}, [%[lhs_ptr]], #32
          and v8.16b, v4.16b, v3.16b
          and v9.16b, v5.16b, v3.16b
          movi v20.4s, #0
          movi v21.4s, #0
          movi v22.4s, #0
          movi v23.4s, #0
          movi v24.4s, #0
          movi v25.4s, #0
          movi v26.4s, #0
          movi v27.4s, #0
          movi v28.4s, #0
          movi v29.4s, #0
          movi v30.4s, #0
          movi v31.4s, #0
          ushr v12.16b, v4.16b, #4
          ushr v13.16b, v5.16b, #4
          ld1 {v0.16b, v1.16b}, [%[rhs_ptr]], #32
          and v10.16b, v6.16b, v3.16b
          and v11.16b, v7.16b, v3.16b
          ushr v14.16b, v6.16b, #4
          ushr v15.16b, v7.16b, #4
          subs %w[run_depth], %w[run_depth], #1
          b.ls 1f /* skip loop */
            0: /* loop start */
            ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [%[lhs_ptr]], #64
            sdot v16.4s, v12.16b, v0.16b
            sdot v17.4s, v13.16b, v0.16b
            sdot v18.4s, v14.16b, v0.16b
            sdot v19.4s, v15.16b, v0.16b
            ld1 {v2.16b}, [%[rhs_ptr]], #16
            sdot v16.4s, v8.16b, v1.16b
            sdot v17.4s, v9.16b, v1.16b
            sdot v18.4s, v10.16b, v1.16b
            sdot v19.4s, v11.16b, v1.16b
            ld1 {v0.16b}, [%[rhs_ptr]], #16
            sdot v20.4s, v12.16b, v2.16b
            sdot v21.4s, v13.16b, v2.16b
            sdot v22.4s, v14.16b, v2.16b
            sdot v23.4s, v15.16b, v2.16b
            ld1 {v1.16b}, [%[rhs_ptr]], #16
            sdot v20.4s, v8.16b, v0.16b
            sdot v21.4s, v9.16b, v0.16b
            sdot v22.4s, v10.16b, v0.16b
            sdot v23.4s, v11.16b, v0.16b
            ld1 {v2.16b}, [%[rhs_ptr]], #16
            sdot v24.4s, v12.16b, v1.16b
            sdot v25.4s, v13.16b, v1.16b
            sdot v26.4s, v14.16b, v1.16b
            sdot v27.4s, v15.16b, v1.16b
            ld1 {v0.16b}, [%[rhs_ptr]], #16
            sdot v24.4s, v8.16b, v2.16b
            sdot v25.4s, v9.16b, v2.16b
            sdot v26.4s, v10.16b, v2.16b
            sdot v27.4s, v11.16b, v2.16b
            ld1 {v1.16b}, [%[rhs_ptr]], #16
            sdot v28.4s, v12.16b, v0.16b
            sdot v29.4s, v13.16b, v0.16b
            sdot v30.4s, v14.16b, v0.16b
            sdot v31.4s, v15.16b, v0.16b
            sdot v28.4s, v8.16b, v1.16b
            sdot v29.4s, v9.16b, v1.16b
            sdot v30.4s, v10.16b, v1.16b
            sdot v31.4s, v11.16b, v1.16b
            ld1 {v0.16b}, [%[rhs_ptr]], #16
            and v8.16b, v4.16b, v3.16b
            and v9.16b, v5.16b, v3.16b
            ushr v12.16b, v4.16b, #4
            ushr v13.16b, v5.16b, #4
            ld1 {v1.16b}, [%[rhs_ptr]], #16
            and v10.16b, v6.16b, v3.16b
            and v11.16b, v7.16b, v3.16b
            ushr v14.16b, v6.16b, #4
            ushr v15.16b, v7.16b, #4
            subs %w[run_depth], %w[run_depth], #1
            b.hi 0b /* loop branch */
          1: /* loop end */
          sdot v16.4s, v12.16b, v0.16b
          sdot v17.4s, v13.16b, v0.16b
          sdot v18.4s, v14.16b, v0.16b
          sdot v19.4s, v15.16b, v0.16b
          ld1 {v2.16b}, [%[rhs_ptr]], #16
          sdot v16.4s, v8.16b, v1.16b
          sdot v17.4s, v9.16b, v1.16b
          sdot v18.4s, v10.16b, v1.16b
          sdot v19.4s, v11.16b, v1.16b
          ld1 {v0.16b}, [%[rhs_ptr]], #16
          sdot v20.4s, v12.16b, v2.16b
          sdot v21.4s, v13.16b, v2.16b
          sdot v22.4s, v14.16b, v2.16b
          sdot v23.4s, v15.16b, v2.16b
          ld1 {v1.16b}, [%[rhs_ptr]], #16
          sdot v20.4s, v8.16b, v0.16b
          sdot v21.4s, v9.16b, v0.16b
          sdot v22.4s, v10.16b, v0.16b
          sdot v23.4s, v11.16b, v0.16b
          ld1 {v2.16b}, [%[rhs_ptr]], #16
          sdot v24.4s, v12.16b, v1.16b
          sdot v25.4s, v13.16b, v1.16b
          sdot v26.4s, v14.16b, v1.16b
          sdot v27.4s, v15.16b, v1.16b
          ld1 {v0.16b}, [%[rhs_ptr]], #16
          sdot v24.4s, v8.16b, v2.16b
          sdot v25.4s, v9.16b, v2.16b
          sdot v26.4s, v10.16b, v2.16b
          sdot v27.4s, v11.16b, v2.16b
          ld1 {v1.16b}, [%[rhs_ptr]], #16
          sdot v28.4s, v12.16b, v0.16b
          sdot v29.4s, v13.16b, v0.16b
          sdot v30.4s, v14.16b, v0.16b
          sdot v31.4s, v15.16b, v0.16b
          sdot v28.4s, v8.16b, v1.16b
          sdot v29.4s, v9.16b, v1.16b
          sdot v30.4s, v10.16b, v1.16b
          sdot v31.4s, v11.16b, v1.16b
          addp v14.4s, v16.4s, v17.4s
          addp v15.4s, v18.4s, v19.4s
          addp v12.4s, v20.4s, v21.4s
          addp v13.4s, v22.4s, v23.4s
          addp v10.4s, v24.4s, v25.4s
          addp v11.4s, v26.4s, v27.4s
          addp v8.4s, v28.4s, v29.4s
          addp v9.4s, v30.4s, v31.4s
          addp v4.4s, v14.4s, v15.4s
          addp v5.4s, v12.4s, v13.4s
          addp v6.4s, v10.4s, v11.4s
          addp v7.4s, v8.4s, v9.4s
          st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[dst]], #64
          )asm"
          : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [dst] "+r"(dst),
            [run_depth] "+r"(run_depth)
          :
          : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
            "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
            "v27", "v28", "v29", "v30", "v31");
    }
  }
}

}  // namespace optimized_4bit
}  // namespace tflite
#endif  // defined(FC_4BIT_NEON) && (defined(__ARM_NEON__) ||
        // defined(__ARM_NEON))
