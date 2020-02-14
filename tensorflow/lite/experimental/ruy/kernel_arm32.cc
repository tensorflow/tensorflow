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

#include "tensorflow/lite/experimental/ruy/kernel.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"

namespace ruy {

#if RUY_PLATFORM(NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)

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
  profiler::ScopeLabel label(
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
#define RUY_MAKE_ZERO(reg) "vmov.f32 " #reg ", #0.0\n"

        // clang-format off

        // Load the first 32 bytes of LHS and RHS data.
        // Load q0, q1
        "vld1.32 {d0, d1}, [%[lhs_ptr]]!\n"
        "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
        RUY_PREFETCH_LOAD("pld [%[lhs_ptr]]\n")
        // Load q2
        "vld1.32 {d4, d5}, [%[rhs_ptr]]!\n"
        RUY_PREFETCH_LOAD("pld [%[rhs_ptr]]\n")

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
        "vld1.32 {d0, d1}, [%[lhs_ptr]]!\n" // Reload LHS

        "vmla.f32 q4, q1, d4[0]\n"
        "vmla.f32 q6, q1, d4[1]\n"
        "vmla.f32 q8, q1, d5[0]\n"
        "vmla.f32 q10, q1, d5[1]\n"
        "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n" // Reload LHS
        RUY_PREFETCH_LOAD("pld [%[lhs_ptr]]\n")
        "vld1.32 {d4, d5}, [%[rhs_ptr]]!\n" // Reload RHS
        RUY_PREFETCH_LOAD("pld [%[rhs_ptr]]\n")

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
        "ldr r10, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "add r10, r10, r1, lsl #2\n"
        "str r10, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
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
        "vld1.32 {d24, d25, d26, d27}, [r1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into q0 -- q2, as we don't need q0 -- q2 anymore
        // in the rest of the work on the current block.
        // Load q0, q1
        "vld1.32 {d0, d1, d2, d3}, [%[lhs_ptr]]!\n"
        RUY_PREFETCH_LOAD("pld [%[lhs_ptr]]\n")
        // Load q2
        "vld1.32 {d4, d5}, [%[rhs_ptr]]!\n"
        RUY_PREFETCH_LOAD("pld [%[rhs_ptr]]\n")

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
        // (r3 address, r4 stride)
        "vst1.32 {d6, d7, d8, d9}, [r3]\n"
        "add r3, r3, r4\n"
        RUY_MAKE_ZERO(q3)
        RUY_MAKE_ZERO(q4)
        "vst1.32 {d10, d11, d12, d13}, [r3]\n"
        "add r3, r3, r4\n"
        RUY_MAKE_ZERO(q5)
        RUY_MAKE_ZERO(q6)
        "vst1.32 {d14, d15, d16, d17}, [r3]\n"
        "add r3, r3, r4\n"
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        "vst1.32 {d18, d19, d20, d21}, [r3]\n"
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
        "ldr r10, [r3, r5, lsl #2]\n"
        "str r10, [r4, r5, lsl #2]\n"
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

        // Reload some params --- we had used r3, r5, r10 for a few other things
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

        // r1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov r1, #1\n"

        "ble 1b\n"

        // Restore stack pointer.
        "add sp, sp, #" RUY_STR(RUY_STACK_OFFSET_SIZE) "\n"

        // clang-format on
        : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
        : [ params ] "r"(&params), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        // Clobber list must specify q registers (and not their constituent
        // d registers). There is a (currently unexplained) slowdown if
        // d registers are listed in the clobbers list.
        : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r8", "r10", "cc",
          "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
          "q9", "q10", "q12", "q13");
}

#undef RUY_MAKE_ZERO
#undef RUY_STACK_OFFSET_SIZE
#undef RUY_STACK_OFFSET_DST_COL_PTR
#undef RUY_STACK_OFFSET_DST_PTR
#undef RUY_STACK_OFFSET_ROW
#undef RUY_STACK_OFFSET_COL
#undef RUY_STACK_OFFSET_LHS_COL_PTR
#undef RUY_STACK_OFFSET_RHS_COL_PTR

#undef RUY_OFFSET_LHS_BASE_PTR
#undef RUY_OFFSET_RHS_BASE_PTR
#undef RUY_OFFSET_DST_BASE_PTR
#undef RUY_OFFSET_BIAS
#undef RUY_OFFSET_START_ROW
#undef RUY_OFFSET_START_COL
#undef RUY_OFFSET_LAST_ROW
#undef RUY_OFFSET_LAST_COL
#undef RUY_OFFSET_DST_ROWS
#undef RUY_OFFSET_DST_COLS
#undef RUY_OFFSET_LHS_STRIDE
#undef RUY_OFFSET_RHS_STRIDE
#undef RUY_OFFSET_DST_STRIDE
#undef RUY_OFFSET_DEPTH
#undef RUY_OFFSET_CLAMP_MIN
#undef RUY_OFFSET_CLAMP_MAX
#undef RUY_OFFSET_FLAGS

#define RUY_OFFSET_BIAS 0
#define RUY_OFFSET_LHS_SUMS 4
#define RUY_OFFSET_RHS_SUMS 8
#define RUY_OFFSET_LHS_BASE_PTR 12
#define RUY_OFFSET_MULTIPLIER_FIXEDPOINT 16
#define RUY_OFFSET_MULTIPLIER_EXPONENT 20
#define RUY_OFFSET_RHS_BASE_PTR 24
#define RUY_OFFSET_DST_BASE_PTR 28
#define RUY_OFFSET_LHS_ZERO_POINT 32
#define RUY_OFFSET_RHS_ZERO_POINT 36
#define RUY_OFFSET_DST_ZERO_POINT 40
#define RUY_OFFSET_PROD_ZP_DEPTH 44
#define RUY_OFFSET_START_ROW 48
#define RUY_OFFSET_START_COL 52
#define RUY_OFFSET_LAST_ROW 56
#define RUY_OFFSET_LAST_COL 60
#define RUY_OFFSET_DST_ROWS 64
#define RUY_OFFSET_DST_COLS 68
#define RUY_OFFSET_LHS_STRIDE 72
#define RUY_OFFSET_RHS_STRIDE 76
#define RUY_OFFSET_DST_STRIDE 80
#define RUY_OFFSET_DEPTH 84
#define RUY_OFFSET_CLAMP_MIN 88
#define RUY_OFFSET_CLAMP_MAX 92
#define RUY_OFFSET_FLAGS 96
#define RUY_OFFSET_DST_TYPE_ID 97

#define RUY_STACK_OFFSET_SIZE 96
#define RUY_STACK_OFFSET_DST_COL_PTR 0
#define RUY_STACK_OFFSET_DST_PTR 16
#define RUY_STACK_OFFSET_ROW 32
#define RUY_STACK_OFFSET_COL 48
#define RUY_STACK_OFFSET_LHS_COL_PTR 64
#define RUY_STACK_OFFSET_RHS_COL_PTR 80

template <typename Params>
void CheckOffsetsInKernelParams8bit(const Params&) {
  static_assert(offsetof(Params, lhs_zero_point) == RUY_OFFSET_LHS_ZERO_POINT,
                "");
  static_assert(offsetof(Params, rhs_zero_point) == RUY_OFFSET_RHS_ZERO_POINT,
                "");
  static_assert(offsetof(Params, dst_zero_point) == RUY_OFFSET_DST_ZERO_POINT,
                "");
  static_assert(offsetof(Params, prod_zp_depth) == RUY_OFFSET_PROD_ZP_DEPTH,
                "");
  static_assert(offsetof(Params, multiplier_fixedpoint) ==
                    RUY_OFFSET_MULTIPLIER_FIXEDPOINT,
                "");
  static_assert(
      offsetof(Params, multiplier_exponent) == RUY_OFFSET_MULTIPLIER_EXPONENT,
      "");
  static_assert(offsetof(Params, clamp_min) == RUY_OFFSET_CLAMP_MIN, "");
  static_assert(offsetof(Params, clamp_max) == RUY_OFFSET_CLAMP_MAX, "");
  static_assert(offsetof(Params, bias) == RUY_OFFSET_BIAS, "");
  static_assert(offsetof(Params, lhs_sums) == RUY_OFFSET_LHS_SUMS, "");
  static_assert(offsetof(Params, rhs_sums) == RUY_OFFSET_RHS_SUMS, "");
  static_assert(offsetof(Params, flags) == RUY_OFFSET_FLAGS, "");
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, start_row) == RUY_OFFSET_START_ROW, "");
  static_assert(offsetof(Params, last_row) == RUY_OFFSET_LAST_ROW, "");
  static_assert(offsetof(Params, last_col) == RUY_OFFSET_LAST_COL, "");
  static_assert(offsetof(Params, lhs_stride) == RUY_OFFSET_LHS_STRIDE, "");
  static_assert(offsetof(Params, rhs_stride) == RUY_OFFSET_RHS_STRIDE, "");
  static_assert(offsetof(Params, dst_stride) == RUY_OFFSET_DST_STRIDE, "");
  static_assert(offsetof(Params, depth) == RUY_OFFSET_DEPTH, "");
}

// Fast-int8 kernel, ported from ARM 64 version.
// Relevant target CPUs for this kernel include Krait 400 and A9,
// since these are 32-bit, out-of-order CPUs.
void Kernel8bitNeonOutOfOrder(const KernelParams8bit<4, 2>& params) {
  profiler::ScopeLabel label(
      "Kernel (kNeon, optimized for out-of-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;

  // The asm kernel below has the following NEON register allocation:
  //
  // q6 - q13 are 128-bit (4x32b) accumulators.
  // During accumulation, d0 -- d7 are used to load int8 data from LHS and
  // d8 -- d11 from RHS:
  //                                      int8 RHS 16x2 block
  //                              /-----------------------------\
  //                              |d8.b[0-7]   .....  d10.b[0-7]|
  //                              |  ...                  ...   |
  //                              |d9.b[0-7]   .....  d11.b[0-7]|
  //                              \-----------------------------/
  //    int8 LHS 4x16 block
  //  /------------------------\  /-----------------------------\
  //  |d0.b[0-7] ... d1.b[0-7] |  | q6         .....      q10   |
  //  |d2.b[0-7] ... d3.b[0-7] |  | q7         .....      q11   |
  //  (Reload d0, d1, d2, d3)
  //  |d0.b[0-7] ... d1.b[0-7] |  | q8         .....      q12   |
  //  |d2.b[0-7] ... d3.b[0-7] |  | q9         .....      q13   |
  //  \------------------------/  \-----------------------------/
  //                                128-bit accumulators 4x2 block
  //
  // No attempt had been made so far at implementing the RUY_OPT_MAX_STREAMING
  // optimization for this kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "vmov.i32 " #reg ", #0x00000000\n"

        // clang-format off

        // Load the first 64 bytes of LHS and RHS data.
        "vld1.8 {d0, d1}, [%[lhs_ptr]]!\n"
        // Clear accumulators.
        RUY_MAKE_ZERO(q6)
        "vld1.8 {d2, d3}, [%[lhs_ptr]]!\n"
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        "vld1.8 {d8, d9}, [%[rhs_ptr]]!\n"
        RUY_MAKE_ZERO(q11)
        "vld1.8 {d10, d11}, [%[rhs_ptr]]!\n"

        "sub sp, sp, #" RUY_STR(RUY_STACK_OFFSET_SIZE) "\n"

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_BASE_PTR) "]\n"
        RUY_MAKE_ZERO(q12)
        "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_BASE_PTR) "]\n"
        RUY_MAKE_ZERO(q13)
        "str r2, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        RUY_MAKE_ZERO(q14)
        "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"

        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_START_COL) "]\n"
        RUY_MAKE_ZERO(q15)
        "str r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "str r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_RHS_BASE_PTR) "]\n"
        "str r2, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"


        // r1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov r1, #16\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // r1 is how many levels of depth we have already loaded
        // data for, r10 is the total depth.
        "ldr r10, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"
        "cmp r1, r10\n"
        "beq 79f\n"

        "2:\n"

        // Mult, mult-acc in to q14, q15, q2, q3
        "vmull.s8 q14, d0, d8\n"
        "vmull.s8 q2, d0, d10\n"

        "vmull.s8 q15, d2, d8\n"
        "vmull.s8 q3, d2, d10\n"

        "vmlal.s8 q14, d1, d9\n"
        "vmlal.s8 q2, d1, d11\n"
        "vmlal.s8 q15, d3, d9\n"
        "vmlal.s8 q3, d3, d11\n"
        "vld1.8 {d0, d1, d2, d3}, [%[lhs_ptr]]!\n" // Reload LHS

        // Then pairwise accumulate in to q6, q7, q10, q11
        "vpadal.s16 q6, q14\n"
        "vpadal.s16 q7, q15\n"
        "vpadal.s16 q10, q2\n"
        "vpadal.s16 q11, q3\n"

        // Mult, mult-acc in to q14, q15, q2, q3
        "vmull.s8 q14, d0, d8\n"
        "vmull.s8 q2, d0, d10\n"

        "vmull.s8 q15, d2, d8\n"
        "vmull.s8 q3, d2, d10\n"

        "vmlal.s8 q14, d1, d9\n"
        "vmlal.s8 q2, d1, d11\n"
        "vmlal.s8 q15, d3, d9\n"
        "vmlal.s8 q3, d3, d11\n"
        "vld1.8 {d0, d1, d2, d3}, [%[lhs_ptr]]!\n" // Reload LHS

        // Then pairwise accumulate in to q8, q9, q12, q13
        "vpadal.s16 q8, q14\n"
        "vld1.8 {d8, d9, d10, d11}, [%[rhs_ptr]]!\n"
        "vpadal.s16 q9, q15\n"
        "vpadal.s16 q12, q2\n"
        "vpadal.s16 q13, q3\n"

        // Prefetch the next 64 bytes of LHS and RHS data.
        RUY_PREFETCH_LOAD("pld [%[lhs_ptr]]\n")
        RUY_PREFETCH_LOAD("pld [%[rhs_ptr]]\n")

        // Each iteration of this loop advances by 16 levels of depth.
        "add r1, r1, #16\n"

        // Loop termination condition
        "cmp r1, r10\n"

        "blt 2b\n"

        "79:\n"

        // Mult, mult-acc in to q14, q15, q2, q3
        "vmull.s8 q14, d0, d8\n"
        "vmull.s8 q2, d0, d10\n"

        "vmull.s8 q15, d2, d8\n"
        "vmull.s8 q3, d2, d10\n"

        "vmlal.s8 q14, d1, d9\n"
        "vmlal.s8 q2, d1, d11\n"
        "vmlal.s8 q15, d3, d9\n"
        "vmlal.s8 q3, d3, d11\n"
        "vld1.8 {d0, d1, d2, d3}, [%[lhs_ptr]]!\n" // Reload LHS

        // Then pairwise accumulate in to q6, q7, q10, q11
        "vpadal.s16 q6, q14\n"
        "vpadal.s16 q7, q15\n"
        "vpadal.s16 q10, q2\n"
        "vpadal.s16 q11, q3\n"

        // Mult, mult-acc in to q14, q15, q2, q3
        "vmull.s8 q14, d0, d8\n"
        "vmull.s8 q2, d0, d10\n"

        "vmull.s8 q15, d2, d8\n"
        "vmull.s8 q3, d2, d10\n"

        "vmlal.s8 q14, d1, d9\n"
        "vmlal.s8 q2, d1, d11\n"
        "vmlal.s8 q15, d3, d9\n"
        "vmlal.s8 q3, d3, d11\n"

        // Then pairwise accumulate in to q8, q9, q12, q13
        "vpadal.s16 q8, q14\n"
        "vpadal.s16 q9, q15\n"
        "vpadal.s16 q12, q2\n"
        "vpadal.s16 q13, q3\n"


        // All accumulation over depth done. q6 - q13 contain the 4x32b
        // accumulators for the 4x2 final matrix.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x2 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // q6-q13 now contain 4 x 32b
        "vpadd.i32 d0, d12, d13\n"
        "vpadd.i32 d1, d14, d15\n"
        "vpadd.i32 d2, d16, d17\n"
        "vpadd.i32 d3, d18, d19\n"
        "vpadd.i32 d4, d20, d21\n"
        "vpadd.i32 d5, d22, d23\n"
        "vpadd.i32 d6, d24, d25\n"
        "vpadd.i32 d7, d26, d27\n"

        // d0-d7 each contain 2 x 32b accumulators.
        // Need to add pairwise to get 1 x 32b for each of the 4x2 entries
        // of destination, (Four 'd' registers total)
        "vpadd.i32 d28, d0, d1\n"
        "vpadd.i32 d29, d2, d3\n"
        "vpadd.i32 d30, d4, d5\n"
        "vpadd.i32 d31, d6, d7\n"

        //Now d28 - d31 have the 1 x 32b accumulators for the 4x2 entries

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "cmp r1, r3\n"  // Have we finished the last row?

        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
        "add r4, r4, r1, lsl #2\n"
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
        "ldr r10, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "add r10, r10, r1, lsl #1\n"
        "str r10, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
        "mov %[lhs_ptr], r4\n"
        "ldr r5, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "mov %[rhs_ptr], r5\n"

        // Now we load: bias data, LHS sums data, RHS sums data.

        // First, load the base pointers from the params.
        "ldrb r4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Offset these base pointers as needed given the current row, col.
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "add r5, r1, r8, lsl #2\n"

        "tst r4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "it ne\n"
        "movne r1, r5\n"

        // Load 4 bias values.
        "vld1.32 {d24, d25}, [r1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "vld1.8 {d0, d1, d2, d3}, [%[lhs_ptr]]!\n"
        RUY_PREFETCH_LOAD("pld [%[lhs_ptr]]\n")
        "vld1.8 {d8, d9, d10, d11}, [%[rhs_ptr]]!\n"
        RUY_PREFETCH_LOAD("pld [%[rhs_ptr]]\n")

        // Add to the bias values the product
        // (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in
        // https://arxiv.org/pdf/1712.05877.pdf
        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "vdup.32 q9, r3\n"
        "vadd.i32 q12, q12, q9\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "vadd.i32 q14, q14, q12\n"
        "vadd.i32 q15, q15, q12\n"

        // LHS/RHS zero points
        // Has RHS sums
        "ldrb r6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "tst r6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        // Offset by current col * number of bytes per value
        "add r3, r3, r4, lsl #2\n"
        "vld1.32 { d12 }, [r3]\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "vdup.32 q10, r5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "vmls.i32 q14, q10, d12[0]\n"
        "vmls.i32 q15, q10, d12[1]\n"
        "401:\n"

        // Has LHS sums
        "ldrb r6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "tst r6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        // Offset by current row * number of bytes per value
        "add r2, r2, r4, lsl #2\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"

        // Load 4 lhs_sums values.
        "vld1.32 {d22, d23}, [r2]\n"
        "vdup.32 d13, r5\n" // rhs_zero_point

        // Compute lhs_sums * rhs_zero_point.
        "vmul.i32 q11, q11, d13[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "vsub.s32 q14, q14, q11\n"
        "vsub.s32 q15, q15, q11\n"

        // If the destination is int32, it means the user asks for the raw
        // accumulators, no need for us to downquantize the value.
        "ldrb r10, [%[params], #" RUY_STR(RUY_OFFSET_DST_TYPE_ID) "]\n"
        "cmp r10, #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst r6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "add r5, r1, r4, lsl #2\n"
        "it ne\n"
        "movne r1, r5\n"

        "vld1.32 {q10}, [r1]\n"

        RUY_MAKE_ZERO(q8)
        "vmax.s32 q12, q10, q8\n"

        "vshl.s32 q14, q14, q12\n"
        "vshl.s32 q15, q15, q12\n"

        "vmin.s32 q12, q10, q8\n"

        // Load fixed point part of the multiplier
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        // r6 has flags, r4 has row
        "add r5, r1, r4, lsl #2\n"
        "tst r6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "it ne\n"
        "movne r1, r5\n"
        "vld1.32 {q10}, [r1]\n" // multiplier_fixedpoint

        // Apply the fixed-point part of the multiplier.
        "vqrdmulh.s32 q14, q14, q10\n"
        "vqrdmulh.s32 q15, q15, q10\n"

        // We have some rounding division-by-power-of-two to do. This should
        // always use "round to nearest". We allow for some
        // freedom in how ties are broken, to strike a good compromise of
        // performance on given hardware vs. perfect agreement of results
        // across hardware.
        //
        // When RUY_OPT_NATIVE_ROUNDING is enabled, we allow for implementation
        // defined tie-breaks to help performance. On NEON, this means that we
        // can just use the NEON rounding instructions, such as srshl. They
        // happen to be breaking ties upward.
        //
        // When RUY_OPT_NATIVE_ROUNDING is disabled, we implement strict
        // break-ties-away-from zero, as described in Appendix B of
        // https://arxiv.org/pdf/1712.05877.pdf
        // When we wrote that, we thought that that would be better unbiased
        // than the NEON upwards tie-breaks, and we had observed some
        // improvement on some model. However, that is only more unbiased for
        // data centered at zero, which was likely the case in that model,
        // but is not always the case. If we wanted something more consistently
        // unbiased then we should try breaking ties toward-nearest-even.
#if !RUY_OPT_ENABLED(RUY_OPT_NATIVE_ROUNDING)
        // Fix up values to be right-shifted, so that the (round to nearest,
        // break ties upward) behavior of srshl applied to these fixed-up
        // values, produces the same result as the desired (round to nearest,
        // break ties away from zero) behavior on the original values.
        "vand q8, q14, q12\n"
        "vand q9, q15, q12\n"
        "vshr.s32 q8, q8, #31\n"
        "vshr.s32 q9, q9, #31\n"
        "vqadd.s32 q14, q14, q8\n"
        "vqadd.s34 q15, q15, q9\n"

#endif
        // At this point we have reduced the problem of correctly implementing
        // rounding divide-by-power-of-two, to what the SRSHL instruction can
        // do.
        "vrshl.s32 q14, q14, q12\n"
        "vrshl.s32 q15, q15, q12\n"

        "ldrb r10, [%[params], #" RUY_STR(RUY_OFFSET_DST_TYPE_ID) "]\n"
        "cmp r10, #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp r10, #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        // Store uint8 values:
        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        // After this, all values for output are in q14.
        "vqmovn.s32 d28, q14\n"
        "vqmovn.s32 d29, q15\n"

        // At this point, d12 -- d26, d30, d31 aren't used anymore for the
        // current block, so we can start clearing these accumulators for the
        // next block (next iteration of the main loop).
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q15)

        // Load the destination zero point into each of the 8 16-bit slots
        // in a q register.
        "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "vdup.16 q13, r4\n" // dst_zero_point

        // Add the destination zero point
        "vadd.i16 q14, q14, q13\n"

        // Cast-and-saturate from int16 to uint8
        // Now all 8 1-byte values are in d30.
        "vqmovun.s16 d30, q14\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb r2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb r3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "vdup.8 d28, r2\n"  // clamp_min
        "vdup.8 d29, r3\n"  // clamp_max

        // Apply the clamp_min bound
        "vmax.u8 d30, d30, d28\n"
        // Apply the clamp_max bound
        "vmin.u8 d30, d30, d29\n"

        // Compute how much of the 4x2 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x2, there are some 4x2 blocks along the boundaries that do
        // not fit entirely.

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #4\n"
        "mov r5, #2\n"
        "cmp r1, #4\n"
        // Compute r1 = how many rows of the 4x2 block fit
        "it gt\n"
        "movgt r1, r3\n"

        "cmp r2, #2\n"
        // Compute r2 = how many cols of the 4x2 block fit
        "it gt\n"
        "movgt r2, r5\n"

        // Test if r1==4 && r2 == 2, i.e. if all of the 4x2 block fits.
        "cmp r1, r3\n"
        "it eq\n"
        "cmpeq r2, r5\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        // Yes, all of the 4x2 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x2 block fits.
        // Store to dst_tmp_buf
        // Set r3 address to write to dst_tmp_buf.
        "mov r3, %[dst_tmp_buf]\n"
        "vst1.8 {d30}, [r3]\n"

        // Slow loop copying from dst_tmp_buf to dst.
        "mov r6, #0\n"
        "50:\n"
        "mov r8, #0\n"
        "51:\n"
        "ldrb r10, [r3, r8]\n"
        "strb r10, [r4, r8]\n"
        "add r8, r8, #1\n"
        "cmp r8, r1\n"
        "blt 51b\n"
        "add r6, r6, #1\n"
        "add r3, r3, #4\n"
        "add r4, r4, r5\n"
        "cmp r6, r2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x2 block fits.
        // r3 address, r5 stride
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r3\n"
        "mov r6, #1\n"

        "vst1.32 {d30[0]}, [r3]\n"
        "add r4, r4, r5\n"
        "mov r3, r4\n"
        "vst1.32 {d30[1]}, [r3]\n"

        "31:\n"

        // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #4\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q14)
        RUY_MAKE_ZERO(q15)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        // Store int8 values:
        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        // After this, all values for output are in q14.
        "vqmovn.s32 d28, q14\n"
        "vqmovn.s32 d29, q15\n"

        // At this point, d12 -- d26, d30, d31 aren't used anymore for the
        // current block, so we can start clearing these accumulators for the
        // next block (next iteration of the main loop).
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q15)

        // Load the destination zero point into each of the 8 16-bit slots
        // in a q register.
        "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "vdup.16 q13, r4\n" // dst_zero_point

        // Add the destination zero point
        "vadd.i16 q14, q14, q13\n"

        // Cast-and-saturate from int16 to int8
        // Now all 8 1-byte values are in d30.
        "vqmovn.s16 d30, q14\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb r2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb r3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "vdup.8 d28, r2\n"  // clamp_min
        "vdup.8 d29, r3\n"  // clamp_max

        // Apply the clamp_min bound
        "vmax.s8 d30, d30, d28\n"
        // Apply the clamp_max bound
        "vmin.s8 d30, d30, d29\n"

        // Compute how much of the 4x2 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x2, there are some 4x2 blocks along the boundaries that do
        // not fit entirely.

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #4\n"
        "mov r5, #2\n"
        "cmp r1, #4\n"
        // Compute r1 = how many rows of the 4x2 block fit
        "it gt\n"
        "movgt r1, r3\n"

        "cmp r2, #2\n"
        // Compute r2 = how many cols of the 4x2 block fit
        "it gt\n"
        "movgt r2, r5\n"

        // Test if r1==4 && r2 == 2, i.e. if all of the 4x2 block fits.
        "cmp r1, r3\n"
        "it eq\n"
        "cmpeq r2, r5\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        // Yes, all of the 4x2 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x2 block fits.
        // Store to dst_tmp_buf
        // Set r3 address to write to dst_tmp_buf.
        "mov r3, %[dst_tmp_buf]\n"
        "vst1.8 {d30}, [r3]\n"

        // Slow loop copying from dst_tmp_buf to dst.
        "mov r6, #0\n"
        "50:\n"
        "mov r8, #0\n"
        "51:\n"
        "ldrb r10, [r3, r8]\n"
        "strb r10, [r4, r8]\n"
        "add r8, r8, #1\n"
        "cmp r8, r1\n"
        "blt 51b\n"
        "add r6, r6, #1\n"
        "add r3, r3, #4\n"
        "add r4, r4, r5\n"
        "cmp r6, r2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x2 block fits.
        // r3 address, r5 stride
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r3\n"
        "mov r6, #1\n"

        "vst1.32 {d30[0]}, [r3]\n"
        "add r4, r4, r5\n"
        "mov r3, r4\n"
        "vst1.32 {d30[1]}, [r3]\n"

        "31:\n"

        // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #4\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q14)
        RUY_MAKE_ZERO(q15)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Load the destination zero point into each of the 4 32-bit slots
        // in a q register.
        "ldrsh r4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "vdup.32 q13, r4\n" // dst_zero_point
        // Add the destination zero point
        "vadd.s32 q14, q14, q13\n"
        "vadd.s32 q15, q15, q13\n"

        // Cast-and-saturate from int32 to int16
        // After this, all values for output are in q14.
        "vqmovn.s32 d28, q14\n"
        "vqmovn.s32 d29, q15\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q15)

         // Load the clamp_min, clamp_max bounds
        "ldrh r2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrh r3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "vdup.16 q12, r2\n"  // clamp_min
        "vdup.16 q13, r3\n"  // clamp_max

        // Apply the clamp_min bound
        "vmax.s16 q14, q14, q12\n"
        // Apply the clamp_max bound
        "vmin.s16 q14, q14, q13\n"

        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)

        // Compute how much of the 4x2 block of destination 16-bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x2, there are some 4x2 blocks along the boundaries that do
        // not fit entirely.

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #4\n"
        "mov r5, #2\n"
        "cmp r1, #4\n"
        // Compute r1 = how many rows of the 4x2 block fit
        "it gt\n"
        "movgt r1, r3\n"

        "cmp r2, #2\n"
        // Compute r2 = how many cols of the 4x2 block fit
        "it gt\n"
        "movgt r2, r5\n"

        // Test if r1==4 && r2 == 2, i.e. if all of the 4x2 block fits.
        "cmp r1, r3\n"
        "it eq\n"
        "cmpeq r2, r5\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        // Yes, all of the 4x2 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x2 block fits.
        // Store to dst_tmp_buf
        // Set r3 address to write to dst_tmp_buf.
        "mov r3, %[dst_tmp_buf]\n"
        "vst1.16 {q14}, [r3]\n"

        // Slow loop copying from dst_tmp_buf to dst.
        "mov r6, #0\n"
        "50:\n"
        "mov r8, #0\n"
        "51:\n"
        // Shift of offset register for half-word loads not allowed in A32,
        // so we shift, load/store, then shift back r8.
        "lsl r8, r8, #1\n"
        "ldrh r10, [r3, r8]\n"
        "strh r10, [r4, r8]\n"
        "lsr r8, r8, #1\n"
        "add r8, r8, #1\n"
        "cmp r8, r1\n"
        "blt 51b\n"
        "add r6, r6, #1\n"
        "add r3, r3, #8\n"
        "add r4, r4, r5\n"
        "cmp r6, r2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x2 block fits.
        // r3 address, r5 stride
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r3\n"
        "mov r6, #2\n"

        "vst1.16 {d28[0]}, [r3], r6\n"
        "add r4, r4, r5\n"
        "vst1.16 {d28[1]}, [r3], r6\n"
        "vst1.16 {d28[2]}, [r3], r6\n"
        "vst1.16 {d28[3]}, [r3], r6\n"
        "mov r3, r4\n"
        "vst1.16 {d29[0]}, [r3], r6\n"
        "vst1.16 {d29[1]}, [r3], r6\n"
        "vst1.16 {d29[2]}, [r3], r6\n"
        "vst1.16 {d29[3]}, [r3], r6\n"
        "31:\n"

         // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #8\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        RUY_MAKE_ZERO(q14)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // At this point, v20 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        // Clear accumulators.
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)

        // Compute how much of the 4x2 block of destination 32 bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x2, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #4\n"
        "mov r5, #2\n"
        "cmp r1, #4\n"
        // Compute r1 = how many rows of the 4x2 block fit
        "it gt\n"
        "movgt r1, r3\n"

        "cmp r2, #2\n"
        // Compute r2 = how many cols of the 4x2 block fit
        "it gt\n"
        "movgt r2, r5\n"

        // Test if r1==4 && r2 == 2, i.e. if all of the 4x2 block fits.
        "cmp r1, r3\n"
        "it eq\n"
        "cmpeq r2, r5\n"
        // Yes, all of the 4x2 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x2 block fits.
        // Set (r3 address, r4 stride) to write to dst_tmp_buf
        "mov r3, %[dst_tmp_buf]\n"
        "mov r4, #16\n"
        "b 31f\n"

        "30:\n"
        // Yes, all of the 4x2 block fits.
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        // r3 address, r4 stride
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r5\n"

        "31:\n"

        "vst1.32 {d28, d29}, [r3]\n"
        "add r3, r3, r4\n"
        "vst1.32 {d30, d31}, [r3]\n"

        // If all of the 4x2 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 4x2 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "ldr r8, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "mov r3, %[dst_tmp_buf]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r6, #0\n"
        "50:\n"
        "mov r5, #0\n"
        "51:\n"
        "ldr r10, [r3, r5, lsl #2]\n"
        "str r10, [r4, r5, lsl #2]\n"
        "add r5, r5, #1\n"
        "cmp r5, r1\n"
        "blt 51b\n"
        "add r6, r6, #1\n"
        "add r3, r3, #16\n"
        "add r4, r4, r8\n"
        // r2 = how many cols of the 8x4 block fit
        "cmp r6, r2\n"
        "blt 50b\n"

        "41:\n"
        // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #16\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
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
        "add r8, r8, #4\n"
        // Store new value of row
        "str r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"

        "b 21f\n"
        "20:\n"
        // Was already at end row.
        // Move back to first row.
        "str r6, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        // Move to the next column.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "add r4, r4, #2\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"

        "ldr r8, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"
        // Increment dst_col_ptr by 2 * dst_stride (i.e. 2 columns)
        "add r1, r1, r8, lsl #1\n"
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
        // above, this is currently 16.
        "mov r1, #16\n"

        "ble 1b\n"

        // Restore stack pointer.
        "add sp, sp, #" RUY_STR(RUY_STACK_OFFSET_SIZE) "\n"

        // clang-format on

        : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
        : [ params ] "r"(&params), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r8", "r10", "cc",
           // Clobber list must specify q registers (and not their constituent
           // d registers). There is a (currently unexplained) slowdown if
           // d registers are listed in the clobbers list.
          "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
          "q9", "q10", "q12", "q13", "q14", "q15");
}

// Fast-int8 true "GEMV" kernel (RHS has 1 column). We assume the RHS
// is still packed as if it has two columns
void Kernel8bitNeonOutOfOrder1Col(const KernelParams8bit<4, 2>& params) {
  profiler::ScopeLabel label(
      "Kernel (kNeon, optimized for out-of-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;

  // The asm kernel below has the following NEON register allocation:
  //
  // q6 - q13 are 128-bit (4x32b) accumulators.
  // During accumulation, d0 -- d7 are used to load int8 data from LHS and
  // d8 -- d11 from RHS:
  //                                            int8 RHS 16x1 block
  //                                               /------------\
  //                                               | d8.b[0]    |
  //                                               | ...        |
  //                                               | d8.b[7]    |
  //                                               | d9.b[0]    |
  //                                               | ...        |
  //                                               | d9.b[7]    |
  //                                               \------------/
  //    int8 LHS 4x16 block
  //  /-----------------------------------------\  /------------\
  //  |d0.b[0] ... d0.b[7] d1.b[0] ... d1.b[7]  |  | q6         |
  //  |d2.b[0] ... d2.b[7] d3.b[0] ... d3.b[7]  |  | q7         |
  //  |d4.b[0] ... d4.b[7] d5.b[0] ... d5.b[7]  |  | q8         |
  //  |d6.b[0] ... d6.b[7] d7.b[0] ... d7.b[7]  |  | q9         |
  //  \-----------------------------------------/  \------------/
  //                              128-bit accumulators 4x1 block
  //
  // No attempt had been made so far at implementing the RUY_OPT_MAX_STREAMING
  // optimization for this kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "vmov.i32 " #reg ", #0x00000000\n"

        // clang-format off

        // Load the first 64 bytes of LHS and RHS data.
        "vld1.8 {d0, d1}, [%[lhs_ptr]]!\n"
        "vld1.8 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.8 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.8 {d6, d7}, [%[lhs_ptr]]!\n"
        "vld1.8 {d8, d9}, [%[rhs_ptr]]!\n"
        // Skip the other column and advance the pointer.
        "add %[rhs_ptr], %[rhs_ptr], #16\n"

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
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q14)
        RUY_MAKE_ZERO(q15)

        // r1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov r1, #16\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // r1 is how many levels of depth we have already loaded
        // data for, r10 is the total depth.
        "ldr r10, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"
        "cmp r1, r10\n"
        "beq 79f\n"

        "2:\n"

        // Mult, mult-acc in to q14, q15
        "vmull.s8 q14, d0, d8\n"
        "vmull.s8 q15, d2, d8\n"
        "vmlal.s8 q14, d1, d9\n"
        "vmlal.s8 q15, d3, d9\n"

        // Then pairwise accumulate in to q6, q7
        "vpadal.s16 q6, q14\n"
        "vpadal.s16 q7, q15\n"

        // Mult, mult-acc in to q14, q15
        "vmull.s8 q14, d4, d8\n"
        "vmull.s8 q15, d6, d8\n"
        "vmlal.s8 q14, d5, d9\n"
        "vmlal.s8 q15, d7, d9\n"

        // Then pairwise accumulate in to q8, q9
        "vpadal.s16 q8, q14\n"
        "vpadal.s16 q9, q15\n"


        // Load the next 64 bytes of LHS and RHS data.
        "vld1.8 {d0, d1}, [%[lhs_ptr]]!\n"
        "vld1.8 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.8 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.8 {d6, d7}, [%[lhs_ptr]]!\n"
        RUY_PREFETCH_LOAD("pld [%[lhs_ptr]]\n")
        "vld1.8 {d8, d9}, [%[rhs_ptr]]!\n"
        // Skip the other column and advance the pointer.
        "add %[rhs_ptr], %[rhs_ptr], #16\n"
        RUY_PREFETCH_LOAD("pld [%[rhs_ptr]]\n")

        // Each iteration of this loop advances by 16 levels of depth.
        "add r1, r1, #16\n"

        // Loop termination condition
        "cmp r1, r10\n"

        "blt 2b\n"

        "79:\n"

        // Mult, mult-acc in to q14, q15
        "vmull.s8 q14, d0, d8\n"
        "vmull.s8 q15, d2, d8\n"
        "vmlal.s8 q14, d1, d9\n"
        "vmlal.s8 q15, d3, d9\n"

        // Then pairwise accumulate in to q6, q7
        "vpadal.s16 q6, q14\n"
        "vpadal.s16 q7, q15\n"

        // Mult, mult-acc in to q14, q15
        "vmull.s8 q14, d4, d8\n"
        "vmull.s8 q15, d6, d8\n"
        "vmlal.s8 q14, d5, d9\n"
        "vmlal.s8 q15, d7, d9\n"

        // Then pairwise accumulate in to q8, q9
        "vpadal.s16 q8, q14\n"
        "vpadal.s16 q9, q15\n"

        // All accumulation over depth done. q6 - q9 contain the 4x32b
        // accumulators for the 4x1 final matrix. 
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x2 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // q6-q9 now contain 4 x 32b
        "vpadd.i32 d0, d12, d13\n"
        "vpadd.i32 d1, d14, d15\n"
        "vpadd.i32 d2, d16, d17\n"
        "vpadd.i32 d3, d18, d19\n"

        // d0-d4 each contain 2 x 32b accumulators.
        // Need to add pairwise to get 1 x 32b for each of the 4x1 entries
        // of destination, (Four 'd' registers total)
        "vpadd.i32 d28, d0, d1\n"
        "vpadd.i32 d29, d2, d3\n"

        // Now d28,d29 have the 1 x 32b accumulators for the 4x1 entries.

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "cmp r1, r3\n"  // Have we finished the last row?

        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
        "add r4, r4, r1, lsl #2\n"
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
        "ldr r10, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "add r10, r10, r1, lsl #1\n"
        "str r10, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_LHS_COL_PTR) "]\n"
        "mov %[lhs_ptr], r4\n"
        "ldr r5, [sp, #" RUY_STR(RUY_STACK_OFFSET_RHS_COL_PTR) "]\n"
        "mov %[rhs_ptr], r5\n"

        // Now we load: bias data, LHS sums data, RHS sums data.

        // First, load the base pointers from the params.
        "ldrb r4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Offset these base pointers as needed given the current row, col.
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "add r5, r1, r8, lsl #2\n"

        "tst r4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "it ne\n"
        "movne r1, r5\n"

        // Load 4 bias values.
        "vld1.32 {d24, d25}, [r1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "vld1.8 {d0, d1}, [%[lhs_ptr]]!\n"
        "vld1.8 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.8 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.8 {d6, d7}, [%[lhs_ptr]]!\n"
        RUY_PREFETCH_LOAD("pld [%[lhs_ptr]]\n")
        "vld1.8 {d8, d9}, [%[rhs_ptr]]!\n"
        // Skip the other column and advance the pointer.
        "add %[rhs_ptr], %[rhs_ptr], #16\n"
        RUY_PREFETCH_LOAD("pld [%[rhs_ptr]]\n")

        // Add to the bias values the product
        // (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in
        // https://arxiv.org/pdf/1712.05877.pdf
        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "vdup.32 q9, r3\n"
        "vadd.i32 q12, q12, q9\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "vadd.i32 q14, q14, q12\n"

        // LHS/RHS zero points
        // Has RHS sums
        "ldrb r6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "tst r6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        // Offset by current col * number of bytes per value
        "add r3, r3, r4, lsl #2\n"
        "vld1.32 { d12 }, [r3]\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "vdup.32 q10, r5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "vmls.i32 q14, q10, d12[0]\n"
        "401:\n"

        // Has LHS sums
        "ldrb r6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "tst r6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        // Offset by current row * number of bytes per value
        "add r2, r2, r4, lsl #2\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"

        // Load 4 lhs_sums values.
        "vld1.32 {d22, d23}, [r2]\n"
        "vdup.32 d13, r5\n" // rhs_zero_point

        // Compute lhs_sums * rhs_zero_point.
        "vmul.i32 q11, q11, d13[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "vsub.s32 q14, q14, q11\n"

        // If the destination is int32, it means the user asks for the raw
        // accumulators, no need for us to downquantize the value.
        "ldrb r10, [%[params], #" RUY_STR(RUY_OFFSET_DST_TYPE_ID) "]\n"
        "cmp r10, #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst r6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "add r5, r1, r4, lsl #2\n"
        "it ne\n"
        "movne r1, r5\n"

        "vld1.32 {q10}, [r1]\n"

        RUY_MAKE_ZERO(q8)
        "vmax.s32 q12, q10, q8\n"

        "vshl.s32 q14, q14, q12\n"

        "vmin.s32 q12, q10, q8\n"

        // Load fixed point part of the multiplier
        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        // r6 has flags, r4 has row
        "add r5, r1, r4, lsl #2\n"
        "tst r6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "it ne\n"
        "movne r1, r5\n"
        "vld1.32 {q10}, [r1]\n" // multiplier_fixedpoint

        // Apply the fixed-point part of the multiplier.
        "vqrdmulh.s32 q14, q14, q10\n"

        // We have some rounding division-by-power-of-two to do. This should
        // always use "round to nearest". We allow for some
        // freedom in how ties are broken, to strike a good compromise of
        // performance on given hardware vs. perfect agreement of results
        // across hardware.
        //
        // When RUY_OPT_NATIVE_ROUNDING is enabled, we allow for implementation
        // defined tie-breaks to help performance. On NEON, this means that we
        // can just use the NEON rounding instructions, such as srshl. They
        // happen to be breaking ties upward.
        //
        // When RUY_OPT_NATIVE_ROUNDING is disabled, we implement strict
        // break-ties-away-from zero, as described in Appendix B of
        // https://arxiv.org/pdf/1712.05877.pdf
        // When we wrote that, we thought that that would be better unbiased
        // than the NEON upwards tie-breaks, and we had observed some
        // improvement on some model. However, that is only more unbiased for
        // data centered at zero, which was likely the case in that model,
        // but is not always the case. If we wanted something more consistently
        // unbiased then we should try breaking ties toward-nearest-even.
#if !RUY_OPT_ENABLED(RUY_OPT_NATIVE_ROUNDING)
        // Fix up values to be right-shifted, so that the (round to nearest,
        // break ties upward) behavior of srshl applied to these fixed-up
        // values, produces the same result as the desired (round to nearest,
        // break ties away from zero) behavior on the original values.
        "vand q8, q14, q12\n"
        "vshr.s32 q8, q8, #31\n"
        "vqadd.s32 q14, q14, q8\n"

#endif
        // At this point we have reduced the problem of correctly implementing
        // rounding divide-by-power-of-two, to what the SRSHL instruction can
        // do.
        "vrshl.s32 q14, q14, q12\n"

        "ldrb r10, [%[params], #" RUY_STR(RUY_OFFSET_DST_TYPE_ID) "]\n"
        "cmp r10, #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp r10, #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        // Store uint8 values:
        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        // After this, all values for output are in d28.
        "vqmovn.s32 d28, q14\n"

        // At this point, d12 -- d26, d29, d30, d31 aren't used anymore for the
        // current block, so we can start clearing these accumulators for the
        // next block (next iteration of the main loop).
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q15)

        // Load the destination zero point into each of the 8 16-bit slots
        // in a q register.
        "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "vdup.16 q13, r4\n" // dst_zero_point

        // Add the destination zero point
        "vadd.i16 q14, q14, q13\n"

        // Cast-and-saturate from int16 to uint8
        "vqmovun.s16 d30, q14\n"
        // At this point, we only need 4 8-bit values in the lower half
        // of d30.


        // Load the clamp_min, clamp_max bounds
        "ldrb r2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb r3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "vdup.8 d28, r2\n"  // clamp_min
        "vdup.8 d29, r3\n"  // clamp_max

        // Apply the clamp_min bound
        "vmax.u8 d30, d30, d28\n"
        // Apply the clamp_max bound
        "vmin.u8 d30, d30, d29\n"

        // Compute how much of the 4x1 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x1, there are some 4x1 blocks along the boundaries that do
        // not fit entirely.

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #4\n"
        "mov r5, #2\n"
        "cmp r1, #4\n"
        // Compute r1 = how many rows of the 4x1 block fit
        "it gt\n"
        "movgt r1, r3\n"

        // Test if r1==4, i.e. if all of the 4x1 block fits.
        "cmp r1, r3\n"

        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        // Yes, all of the 4x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x1 block fits.
        // Store to dst_tmp_buf
        // Set r3 address to write to dst_tmp_buf.
        "mov r3, %[dst_tmp_buf]\n"
        "vst1.8 {d30}, [r3]\n"

        // Slow loop copying from dst_tmp_buf to dst.
        "50:\n"
        "mov r8, #0\n"
        "51:\n"
        "ldrb r10, [r3, r8]\n"
        "strb r10, [r4, r8]\n"
        "add r8, r8, #1\n"
        "cmp r8, r1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x1 block fits.
        // r3 address, r5 stride
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r3\n"
        "mov r6, #1\n"

        "vst1.8 {d30[0]}, [r3], r6\n"
        "vst1.8 {d30[1]}, [r3], r6\n"
        "vst1.8 {d30[2]}, [r3], r6\n"
        "vst1.8 {d30[3]}, [r3], r6\n"
        "31:\n"

        // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #4\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q14)
        RUY_MAKE_ZERO(q15)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        // Store int8 values:
        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        // After this, all values for output are in d28.
        "vqmovn.s32 d28, q14\n"

        // At this point, d12 -- d26, d29, d30, d31 aren't used anymore for the
        // current block, so we can start clearing these accumulators for the
        // next block (next iteration of the main loop).
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q15)

        // Load the destination zero point into each of the 8 16-bit slots
        // in a q register.
        "ldr r4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "vdup.16 q13, r4\n" // dst_zero_point

        // Add the destination zero point
        "vadd.i16 q14, q14, q13\n"

        // Cast-and-saturate from int16 to int8
        "vqmovn.s16 d30, q14\n"
        // At this point, we only need 4 8-bit values in the lower half
        // of d30.

        // Load the clamp_min, clamp_max bounds
        "ldrb r2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb r3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "vdup.8 d28, r2\n"  // clamp_min
        "vdup.8 d29, r3\n"  // clamp_max

        // Apply the clamp_min bound
        "vmax.s8 d30, d30, d28\n"
        // Apply the clamp_max bound
        "vmin.s8 d30, d30, d29\n"

        // Compute how much of the 4x1 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x2, there are some 4x2 blocks along the boundaries that do
        // not fit entirely.

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #4\n"
        "mov r5, #2\n"
        "cmp r1, #4\n"
        // Compute r1 = how many rows of the 4x2 block fit
        "it gt\n"
        "movgt r1, r3\n"

        // Test if r1==4 i.e. if all of the 4x1 block fits.
        "cmp r1, r3\n"

        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        // Yes, all of the 4x2 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x2 block fits.
        // Store to dst_tmp_buf
        // Set r3 address to write to dst_tmp_buf.
        "mov r3, %[dst_tmp_buf]\n"
        "vst1.8 {d30}, [r3]\n"

        // Slow loop copying from dst_tmp_buf to dst.
        "50:\n"
        "mov r8, #0\n"
        "51:\n"
        "ldrb r10, [r3, r8]\n"
        "strb r10, [r4, r8]\n"
        "add r8, r8, #1\n"
        "cmp r8, r1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x1 block fits.
        // r3 address, r5 stride
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r3\n"
        "mov r6, #1\n"

        "vst1.8 {d30[0]}, [r3], r6\n"
        "vst1.8 {d30[1]}, [r3], r6\n"
        "vst1.8 {d30[2]}, [r3], r6\n"
        "vst1.8 {d30[3]}, [r3], r6\n"
        "31:\n"

        // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #4\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        RUY_MAKE_ZERO(q13)
        RUY_MAKE_ZERO(q14)
        RUY_MAKE_ZERO(q15)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Load the destination zero point into each of the 4 32-bit slots
        // in a q register.
        "ldrsh r4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "vdup.32 q13, r4\n" // dst_zero_point
        // Add the destination zero point
        "vadd.s32 q14, q14, q13\n"
        //"vadd.s32 q15, q15, q13\n"

        // Cast-and-saturate from int32 to int16
        // After this, all values for output are in d28.
        "vqmovn.s32 d28, q14\n"

        // At this point, d12 -- d26, d29, d30, d31 aren't used anymore for the
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q15)

         // Load the clamp_min, clamp_max bounds
        "ldrh r2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrh r3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "vdup.16 d24, r2\n"  // clamp_min
        "vdup.16 d26, r3\n"  // clamp_max

        // Apply the clamp_min bound
        "vmax.s16 d28, d28, d24\n"
        // Apply the clamp_max bound
        "vmin.s16 d28, d28, d26\n"

        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)

        // Compute how much of the 4x1 block of destination 16-bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x1, there are some 4x1 blocks along the boundaries that do
        // not fit entirely.

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #4\n"
        "mov r5, #2\n"
        "cmp r1, #4\n"
        // Compute r1 = how many rows of the 4x1 block fit
        "it gt\n"
        "movgt r1, r3\n"

        // Test if r1==4, i.e. if all of the 4x1 block fits.
        "cmp r1, r3\n"

        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        // Yes, all of the 4x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x1 block fits.
        // Store to dst_tmp_buf
        // Set r3 address to write to dst_tmp_buf.
        "mov r3, %[dst_tmp_buf]\n"
        "vst1.16 {d28}, [r3]\n"

        // Slow loop copying from dst_tmp_buf to dst.
        "50:\n"
        "mov r8, #0\n"
        "51:\n"
        // Shift of offset register for half-word loads not allowed in A32,
        // so we shift, load/store, then shift back r8.
        "lsl r8, r8, #1\n"
        "ldrh r10, [r3, r8]\n"
        "strh r10, [r4, r8]\n"
        "lsr r8, r8, #1\n"
        "add r8, r8, #1\n"
        "cmp r8, r1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x1 block fits.
        // r3 address, r5 stride
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r3\n"
        "mov r6, #2\n"

        "vst1.16 {d28[0]}, [r3], r6\n"
        "vst1.16 {d28[1]}, [r3], r6\n"
        "vst1.16 {d28[2]}, [r3], r6\n"
        "vst1.16 {d28[3]}, [r3], r6\n"
        "31:\n"

         // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #8\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        RUY_MAKE_ZERO(q14)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // At this point, v20 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        // Clear accumulators.
        RUY_MAKE_ZERO(q6)
        RUY_MAKE_ZERO(q7)
        RUY_MAKE_ZERO(q8)
        RUY_MAKE_ZERO(q9)
        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)
        RUY_MAKE_ZERO(q12)
        RUY_MAKE_ZERO(q13)

        // Compute how much of the 4x1 block of destination 32 bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x2, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.

        "ldr r1, [%[params], #" RUY_STR(RUY_OFFSET_DST_ROWS) "]\n"
        "ldr r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        "sub r1, r1, r8\n"

        "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_DST_COLS) "]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "sub r2, r2, r4\n"
        "mov r3, #4\n"
        "mov r5, #2\n"
        "cmp r1, #4\n"
        // Compute r1 = how many rows of the 4x2 block fit
        "it gt\n"
        "movgt r1, r3\n"

        // Test if r1==4, i.e. if all of the 4x1 block fits.
        "cmp r1, r3\n"

        // Yes, all of the 4x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x1 block fits.
        // Set (r3 address, r4 stride) to write to dst_tmp_buf
        "mov r3, %[dst_tmp_buf]\n"
        "mov r4, #16\n"
        "b 31f\n"

        "30:\n"
        // Yes, all of the 4x1 block fits.
        "ldr r5, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        // r3 address, r4 stride
        "ldr r3, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "mov r4, r5\n"

        "31:\n"

        "vst1.32 {d28, d29}, [r3]\n"

        // If all of the 4x1 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 4x1 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "ldr r8, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "mov r3, %[dst_tmp_buf]\n"
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "50:\n"
        "mov r5, #0\n"
        "51:\n"
        "ldr r10, [r3, r5, lsl #2]\n"
        "str r10, [r4, r5, lsl #2]\n"
        "add r5, r5, #1\n"
        "cmp r5, r1\n"
        "blt 51b\n"

        "41:\n"
        // Load dst_ptr, increment, and write back.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"
        "add r4, r4, #16\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_PTR) "]\n"

        RUY_MAKE_ZERO(q10)
        RUY_MAKE_ZERO(q11)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
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
        "add r8, r8, #4\n"
        // Store new value of row
        "str r8, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"

        "b 21f\n"
        "20:\n"
        // Was already at end row.
        // Move back to first row.
        "str r6, [sp, #" RUY_STR(RUY_STACK_OFFSET_ROW) "]\n"
        // Move to the next column.
        "ldr r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"
        "add r4, r4, #2\n"
        "str r4, [sp, #" RUY_STR(RUY_STACK_OFFSET_COL) "]\n"

        "ldr r8, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr r1, [sp, #" RUY_STR(RUY_STACK_OFFSET_DST_COL_PTR) "]\n"
        // Increment dst_col_ptr by dst_stride (i.e. 1 column)
        "add r1, r1, r8\n"
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
        // above, this is currently 16.
        "mov r1, #16\n"

        "ble 1b\n"

        // Restore stack pointer.
        "add sp, sp, #" RUY_STR(RUY_STACK_OFFSET_SIZE) "\n"

        // clang-format on

        : [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr)
        : [ params ] "r"(&params), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r8", "r10", "cc",
           // Clobber list must specify q registers (and not their constituent
           // d registers). There is a (currently unexplained) slowdown if
           // d registers are listed in the clobbers list.
          "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
          "q9", "q10", "q12", "q13", "q14", "q15");
}

#undef RUY_OFFSET_BIAS
#undef RUY_OFFSET_LHS_SUMS
#undef RUY_OFFSET_RHS_SUMS
#undef RUY_OFFSET_LHS_BASE_PTR
#undef RUY_OFFSET_MULTIPLIER_FIXEDPOINT
#undef RUY_OFFSET_MULTIPLIER_EXPONENT
#undef RUY_OFFSET_RHS_BASE_PTR
#undef RUY_OFFSET_DST_BASE_PTR
#undef RUY_OFFSET_LHS_ZERO_POINT
#undef RUY_OFFSET_RHS_ZERO_POINT
#undef RUY_OFFSET_DST_ZERO_POINT
#undef RUY_OFFSET_PROD_ZP_DEPTH
#undef RUY_OFFSET_START_ROW
#undef RUY_OFFSET_START_COL
#undef RUY_OFFSET_LAST_ROW
#undef RUY_OFFSET_LAST_COL
#undef RUY_OFFSET_DST_ROWS
#undef RUY_OFFSET_DST_COLS
#undef RUY_OFFSET_LHS_STRIDE
#undef RUY_OFFSET_RHS_STRIDE
#undef RUY_OFFSET_DST_STRIDE
#undef RUY_OFFSET_DEPTH
#undef RUY_OFFSET_CLAMP_MIN
#undef RUY_OFFSET_CLAMP_MAX
#undef RUY_OFFSET_FLAGS
#undef RUY_OFFSET_DST_TYPE_ID

#undef RUY_STACK_OFFSET_SIZE
#undef RUY_STACK_OFFSET_DST_COL_PTR
#undef RUY_STACK_OFFSET_DST_PTR
#undef RUY_STACK_OFFSET_ROW
#undef RUY_STACK_OFFSET_COL
#undef RUY_STACK_OFFSET_LHS_COL_PTR
#undef RUY_STACK_OFFSET_RHS_COL_PTR

#endif  // RUY_PLATFORM(NEON_32) && (RUY_OPT_ENABLED(RUY_OPT_ASM)
}  // namespace ruy
