/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/kernel.h"

namespace ruy {

#if (defined RUY_NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#define RUY_ASM_LABEL_STORE_UINT8 91
#define RUY_ASM_LABEL_STORE_INT8 92
#define RUY_ASM_LABEL_STORE_INT16 93
#define RUY_ASM_LABEL_STORE_INT32 94
#define RUY_ASM_LABEL_AFTER_STORE 99

#define RUY_OFFSET_LHS_BASE_PTR 0
#define RUY_OFFSET_RHS_BASE_PTR 4
#define RUY_OFFSET_DST_BASE_PTR 8
#define RUY_OFFSET_BIAS 12
#define RUY_OFFSET_START_ROW 16
#define RUY_OFFSET_START_COL 20
#define RUY_OFFSET_LAST_ROW 24
#define RUY_OFFSET_LAST_COL 28
#define RUY_OFFSET_DST_ROWS 32
#define RUY_OFFSET_DST_COLS 36
#define RUY_OFFSET_LHS_STRIDE 40
#define RUY_OFFSET_RHS_STRIDE 44
#define RUY_OFFSET_DST_STRIDE 48
#define RUY_OFFSET_DEPTH 52
#define RUY_OFFSET_CLAMP_MIN 56
#define RUY_OFFSET_CLAMP_MAX 60
#define RUY_OFFSET_FLAGS 64

#define RUY_STACK_OFFSET_SIZE 96
#define RUY_STACK_OFFSET_DST_COL_PTR 0
#define RUY_STACK_OFFSET_DST_PTR 16
#define RUY_STACK_OFFSET_ROW 32
#define RUY_STACK_OFFSET_COL 48
#define RUY_STACK_OFFSET_LHS_COL_PTR 64
#define RUY_STACK_OFFSET_RHS_COL_PTR 80

template <typename Params>
void CheckOffsetsInKernelParamsFloat32(const Params&) {
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, rhs_base_ptr) == RUY_OFFSET_RHS_BASE_PTR, "");
  static_assert(offsetof(Params, dst_base_ptr) == RUY_OFFSET_DST_BASE_PTR, "");
  static_assert(offsetof(Params, bias) == RUY_OFFSET_BIAS, "");
  static_assert(offsetof(Params, start_row) == RUY_OFFSET_START_ROW, "");
  static_assert(offsetof(Params, start_col) == RUY_OFFSET_START_COL, "");
  static_assert(offsetof(Params, last_row) == RUY_OFFSET_LAST_ROW, "");
  static_assert(offsetof(Params, last_col) == RUY_OFFSET_LAST_COL, "");
  static_assert(offsetof(Params, dst_rows) == RUY_OFFSET_DST_ROWS, "");
  static_assert(offsetof(Params, lhs_stride) == RUY_OFFSET_LHS_STRIDE, "");
  static_assert(offsetof(Params, rhs_stride) == RUY_OFFSET_RHS_STRIDE, "");
  static_assert(offsetof(Params, dst_stride) == RUY_OFFSET_DST_STRIDE, "");
  static_assert(offsetof(Params, depth) == RUY_OFFSET_DEPTH, "");
  static_assert(offsetof(Params, clamp_min) == RUY_OFFSET_CLAMP_MIN, "");
  static_assert(offsetof(Params, clamp_max) == RUY_OFFSET_CLAMP_MAX, "");
  static_assert(offsetof(Params, flags) == RUY_OFFSET_FLAGS, "");
}

// Float kernel for ARM32 out-of-order cores.
// Just like Float 64 version, except accumulate in to 8x4 block to only
// use 16 128-bit NEON registers. This is a "first pass" kernel and not
// tuned. It is meant to run on out-of-order CPUs like the Krait 400 or A9.
void KernelFloat32NeonOutOfOrder(const KernelParamsFloat<8, 4>& params) {
  CheckOffsetsInKernelParamsFloat32(params);
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeon, optimized for out-of-order cores)");

  const float* lhs_ptr = params.lhs_base_ptr;
  const float* rhs_ptr = params.rhs_base_ptr;
  // In ARM32 NEON, there are 16 128-bit "q" registers. These registers are
  // each composed of two 64-bit "d" registers. The asm kernel below has the
  // following NEON register allocation:
  // Registers q3 -- q10 are accumulators. During accumulation,
  // q0 -- q2 (d0 -- d5) are used to load data from LHS and RHS. q0 and q1
  // are used to load a 8x1 block of LHS, and q2 is used to load a 1x4 block
  // of RHS, like this:

  //  Register layout in "q" registers:
  //                                    RHS 1x4 block
  //                           /--------------------------\
  //                           |q2.s[0] ...      q2.s[3]  |
  //                           \--------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /---------------------     \
  //  |        q0.s[0]      |  | q3.s[0]   ...    q9.s[0] |
  //  |         ...         |  |  ...               ...   |
  //  |        q0.s[3]      |  | q3.s[3]          q9.s[3] |
  //  |        q1.s[0]      |  | q4.s[0]         q10.s[0] |
  //  |         ...         |  |  ...      ...      ...   |
  //  |        q1.s[3]      |  | q4.s[3]   ..    q10.s[3] |
  //  \---------------------/  \--------------------------/
  //                             accumulators 8x4 block
  // q11, q14, q15 currently unused. q12 and q13 are used to load
  // parameters used for the post-accumulation part of the kernel.
  // For completeness, here is the register layout in "d" registers:
  //                                    RHS 1x4 block
  //                           /--------------------------\
  //                           |d4[0]     ...       d5[1] |
  //                           \--------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /--------------------------\
  //  |        d0[0]        |  | d6[0]    ...      d18[0] |
  //  |         ...         |  |  ...               ...   |
  //  |        d1[1]        |  | d7[1]             d19[1] |
  //  |        d2[0]        |  | d8[0]             d20[0] |
  //  |         ...         |  |  ...      ...      ...   |
  //  |        d3[1]        |  | d9[1]     ...     d21[1] |
  //  \---------------------/  \--------------------------/
  //                             accumulators 8x4 block
  asm volatile(
#define RUY_MAKE_ZERO(reg) "mov r0, 0\n vdup.32 " #reg ", r0\n"

        // clang-format off

        // Load the first 32 bytes of LHS and RHS data.
        // Load q0
        "vld1.32 {d0}, [%[lhs_ptr]]!\n"
        "vld1.32 {d1}, [%[lhs_ptr]]!\n"
        // Load q1
        "vld1.32 {d2}, [%[lhs_ptr]]!\n"
        "vld1.32 {d3}, [%[lhs_ptr]]!\n"
        // Load q2
        "vld1.32 {d4}, [%[rhs_ptr]]!\n"
        "vld1.32 {d5}, [%[rhs_ptr]]!\n"

        "sub sp, sp, #" RUY_STR(RUY_STACK_OFFSET_SIZE) "\n"

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_BASE_PTR) "]\n"
        "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_BASE_PTR) "]\n"
        "str r2, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"

        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_START_COL) "]\n"
        "str r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_RHS_BASE_PTR) "]\n"
        "str r2, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        // Clear accumulators.
        RUY_MAKE_ZERO(q3)
        RUY_MAKE_ZERO(q4)
        RUY_MAKE_ZERO(q5)
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)

        // r1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov r1, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Accumulation loop
        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"
        "cmp r1, r2\n"
        "beq 79f\n"

        "2:\n"

        "vmla.f32 q3, q0, d4[0]\n"
        "vmla.f32 q5, q0, d4[1]\n"
        "vmla.f32 q7, q0, d5[0]\n"
        "vmla.f32 q9, q0, d5[1]\n"
        "vld1.32 {d0}, [%[lhs_ptr]]!\n" // Reload LHS 1 into r0
        "vld1.32 {d1}, [%[lhs_ptr]]!\n" // Reload LHS 1 into r0

        "vmla.f32 q4, q1, d4[0]\n"
        "vmla.f32 q6, q1, d4[1]\n"
        "vmla.f32 q8, q1, d5[0]\n"
        "vmla.f32 q10, q1, d5[1]\n"
        "vld1.32 {d2}, [%[lhs_ptr]]!\n" // Reload LHS 2 into r1
        "vld1.32 {d3}, [%[lhs_ptr]]!\n" // Reload LHS 2 into r1
        "vld1.32 {d4}, [%[rhs_ptr]]!\n" // Reload RHS into r2
        "vld1.32 {d5}, [%[rhs_ptr]]!\n" // Reload RHS into r2

        "add r1, r1, #1\n"
        "cmp r1, r2\n"

        "blt 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        "vmla.f32 q3, q0, d4[0]\n"
        "vmla.f32 q5, q0, d4[1]\n"
        "vmla.f32 q7, q0, d5[0]\n"
        "vmla.f32 q9, q0, d5[1]\n"

        "vmla.f32 q4, q1, d4[0]\n"
        "vmla.f32 q6, q1, d4[1]\n"
        "vmla.f32 q8, q1, d5[0]\n"
        "vmla.f32 q10, q1, d5[1]\n"

        // End of accumulation. The registers q3 -- q10 contain the final
        // float32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final values from these accumulators
        // and advance to the next 8x8 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "cmp r1, r3\n"  // Have we finished the last row?

        "bge 4f\n"      // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
        "add r4, r4, r1, lsl #3\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        // Go back to first row
        "str r5, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "cmp r8, r4\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr r7, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "add r7, r7, r1, lsl #2\n"
        "str r7, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
        "mov %[lhs_ptr], r4\n"
        "ldr r5, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "mov %[rhs_ptr], r5\n"

        // Load some parameters needed for the end work on current block.
        "ldrb r4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Offset these base pointers as needed given the current row, col.
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "add r5, r1, r8, lsl #2\n"

        "tst r4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "it ne\n"
        "movne r1, r5\n"

        // Load 8 bias values.
        "vld1.32 {d24}, [r1]!\n"
        "vld1.32 {d25}, [r1]!\n"
        "vld1.32 {d26}, [r1]!\n"
        "vld1.32 {d27}, [r1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into q0 -- q2, as we don't need q0 -- q2 anymore
        // in the rest of the work on the current block.
        // Load q0
        "vld1.32 {d0}, [%[lhs_ptr]]!\n"
        "vld1.32 {d1}, [%[lhs_ptr]]!\n"
        // Load q1
        "vld1.32 {d2}, [%[lhs_ptr]]!\n"
        "vld1.32 {d3}, [%[lhs_ptr]]!\n"
        // Load q2
        "vld1.32 {d4}, [%[rhs_ptr]]!\n"
        "vld1.32 {d5}, [%[rhs_ptr]]!\n"


        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "vadd.f32 q3, q3, q12\n"
        "vadd.f32 q4, q4, q13\n"
        "vadd.f32 q5, q5, q12\n"
        "vadd.f32 q6, q6, q13\n"
        "vadd.f32 q7, q7, q12\n"
        "vadd.f32 q8, q8, q13\n"
        "vadd.f32 q9, q9, q12\n"
        "vadd.f32 q10, q10, q13\n"

        // Load the clamp_min, clamp_max bounds
        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "vdup.32 q12, r2\n"  // clamp_min
        "vdup.32 q13, r3\n"  // clamp_max

        // Apply the clamp_min bound
        "vmax.f32 q3, q3, q12\n"
        "vmax.f32 q4, q4, q12\n"
        "vmax.f32 q5, q5, q12\n"
        "vmax.f32 q6, q6, q12\n"
        "vmax.f32 q7, q7, q12\n"
        "vmax.f32 q8, q8, q12\n"
        "vmax.f32 q9, q9, q12\n"
        "vmax.f32 q10, q10, q12\n"

        // Apply the clamp_max bound
        "vmin.f32 q3, q3, q13\n"
        "vmin.f32 q4, q4, q13\n"
        "vmin.f32 q5, q5, q13\n"
        "vmin.f32 q6, q6, q13\n"
        "vmin.f32 q7, q7, q13\n"
        "vmin.f32 q8, q8, q13\n"
        "vmin.f32 q9, q9, q13\n"
        "vmin.f32 q10, q10, q13\n"

        // Compute how much of the 8x4 block of destination values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x4, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #8\n"
        "mov r5, #4\n"
        "cmp r1, #8\n"
        // Compute r1 = how many rows of the 8x4 block fit
        "it gt\n"
        "movgt r1, r3\n"
        "cmp r2, #4\n"
        // Compute r2 = how many cols of the 8x4 block fit
        "it gt\n"
        "movgt r2, r5\n"

        // Test if r1==8 && r2 == 4, i.e. if all of the 8x4 block fits.
        "cmp r1, r3\n"
        "it eq\n"
        "cmpeq r2, r5\n"
        // Yes, all of the 8x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x4 block fits.
        // Set (r3 address, r4 stride) to write to dst_tmp_buf
        "mov r3, %[dst_tmp_buf]\n"
        "mov r4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x4 block fits.
        // Set (r3 address, r4 stride) to write directly to destination matrix.
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r5\n"
        "31:\n"

        // Write our float values to the destination described by
        // (r3 address, r4 stride).
        // q3 = d6, d7
        "vstr d6, [r3, #0]\n"
        "vstr d7, [r3, #8]\n"
        // q4 = d8, d9
        "vstr d8, [r3, #16]\n"
        "vstr d9, [r3, #24]\n"
        "add r3, r3, r4\n"
        RUY_MAKE_ZERO(q3)
        RUY_MAKE_ZERO(q4)
        // q5 = d10, d11
        "vstr d10, [r3, #0]\n"
        "vstr d11, [r3, #8]\n"
        // q6 = d12, d13
        "vstr d12, [r3, #16]\n"
        "vstr d13, [r3, #24]\n"
        "add r3, r3, r4\n"
        RUY_MAKE_ZERO(q5)
        RUY_MAKE_ZERO(q6)
        // q7 = d14, d15
        "vstr d14, [r3, #0]\n"
        "vstr d15, [r3, #8]\n"
        // q8 = d16, d17
        "vstr d16, [r3, #16]\n"
        "vstr d17, [r3, #24]\n"
        "add r3, r3, r4\n"
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        // q9 = d18, d19
        "vstr d18, [r3, #0]\n"
        "vstr d19, [r3, #8]\n"
        // q10 = d20, d21
        "vstr d20, [r3, #16]\n"
        "vstr d21, [r3, #24]\n"
        "add r3, r3, r4\n"
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)

        // If all of the 8x4 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "ldr r8, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "mov r3, %[dst_tmp_buf]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r6, #0\n"
        "50:\n"
        "mov r5, #0\n"
        "51:\n"
        "ldr r7, [r3, r5, lsl #2]\n"
        "str r7, [r4, r5, lsl #2]\n"
        "add r5, r5, #1\n"
        "cmp r5, r1\n"
        "blt 51b\n"
        "add r6, r6, #1\n"
        "add r3, r3, #32\n"
        "add r4, r4, r8\n"
        // r2 = how many cols of the 8x4 block fit
        "cmp r6, r2\n"
        "blt 50b\n"
        "41:\n"
        // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #32\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        // Reload some params --- we had used r3, r5, r7 for a few other things
        // since the last time we had loaded them.
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr r6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "cmp r8, r3\n"

        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add r8, r8, #8\n"
        // Store new value of row
        "str r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"

        "b 21f\n"
        "20:\n"
        // Was already at end row.
        // Move back to first row.
        "str r6, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        // Move to the next column.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "add r4, r4, #4\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"

        "ldr r8, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"
        // Increment dst_col_ptr by 4 * dst_stride (i.e. 4 columns)
        "add r1, r1, r8, lsl #2\n"
        // Store dst_col_ptr
        "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"
        // Store dst_ptr
        "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "cmp r8, r4\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov r1, #1\n"

        "ble 1b\n"

        // Restore stack pointer.
        "add sp, sp, #" RUY_STR(RUY_STACK_OFFSET_SIZE) "\n"

        // clang-format on
        : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
        : [ params ] "r"(&params), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "cc",
          "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q12", "q13");
}

#undef RUY_OFFSET_BIAS
#undef RUY_OFFSET_FLAGS
#undef RUY_OFFSET_LHS_BASE_PTR
#undef RUY_OFFSET_CLAMP_MIN
#undef RUY_OFFSET_CLAMP_MAX
#undef RUY_OFFSET_START_ROW
#undef RUY_OFFSET_LAST_ROW
#undef RUY_OFFSET_LAST_COL
#undef RUY_OFFSET_LHS_STRIDE
#undef RUY_OFFSET_RHS_STRIDE
#undef RUY_OFFSET_DST_STRIDE
#undef RUY_OFFSET_DEPTH
#undef RUY_OFFSET_START_COL
#undef RUY_OFFSET_RHS_BASE_PTR
#undef RUY_OFFSET_DST_BASE_PTR

#endif  // (defined RUY_NEON_32) && (RUY_OPT_ENABLED(RUY_OPT_ASM)
}  // namespace ruy
