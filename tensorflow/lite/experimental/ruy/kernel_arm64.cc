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

#include <cstdint>

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/kernel.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/platform.h"

namespace ruy {

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#define RUY_ASM_LABEL_STORE_UINT8 91
#define RUY_ASM_LABEL_STORE_INT8 92
#define RUY_ASM_LABEL_STORE_INT16 93
#define RUY_ASM_LABEL_STORE_INT32 94
#define RUY_ASM_LABEL_AFTER_STORE 99

#define RUY_OFFSET_BIAS 0
#define RUY_OFFSET_LHS_SUMS 8
#define RUY_OFFSET_RHS_SUMS 16
#define RUY_OFFSET_LHS_BASE_PTR 24
#define RUY_OFFSET_MULTIPLIER_FIXEDPOINT 32
#define RUY_OFFSET_MULTIPLIER_EXPONENT 40
#define RUY_OFFSET_RHS_BASE_PTR 48
#define RUY_OFFSET_DST_BASE_PTR 56
#define RUY_OFFSET_LHS_ZERO_POINT 64
#define RUY_OFFSET_RHS_ZERO_POINT 68
#define RUY_OFFSET_DST_ZERO_POINT 72
#define RUY_OFFSET_PROD_ZP_DEPTH 76
#define RUY_OFFSET_START_ROW 80
#define RUY_OFFSET_START_COL 84
#define RUY_OFFSET_LAST_ROW 88
#define RUY_OFFSET_LAST_COL 92
#define RUY_OFFSET_DST_ROWS 96
#define RUY_OFFSET_DST_COLS 100
#define RUY_OFFSET_LHS_STRIDE 104
#define RUY_OFFSET_RHS_STRIDE 108
#define RUY_OFFSET_DST_STRIDE 112
#define RUY_OFFSET_DEPTH 116
#define RUY_OFFSET_CLAMP_MIN 120
#define RUY_OFFSET_CLAMP_MAX 124
#define RUY_OFFSET_FLAGS 128

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

// Fast-int8-trick kernel, similar to this production gemmlowp kernel:
// NEON_64bit_GEMM_Int8Operands_AccumTwoWithin16Bits
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L2296
//
// Relevant target CPUs for this kernel include ARM Cortex-A73 and Cortex-A75,
// since these are 64-bit, out-of-order and without dotprod support.
void Kernel8bitNeonOutOfOrder(const KernelParams8bit<4, 4>& params) {
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeon, optimized for out-of-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v3 are used to load int8 data from LHS and
  // v4 -- v7 from RHS:
  //
  //                                      int8 RHS 16x4 block
  //                           /-----------------------------------------\
  //                           |v4.b[0]          ...           v7.b[0]   |
  //                           |  ...                            ...     |
  //                           |v4.b[15]         ...           v7.b[15]  |
  //                           \-----------------------------------------/
  //    int8 LHS 4x16 block
  //  /---------------------\  /-----------------------------------------\
  //  |v0.b[0] ... v0.b[15] |  |v16.4s           ...           v28.4s    |
  //  |v1.b[0] ... v1.b[15] |  |v17.4s           ...           v29.4s    |
  //  |v2.b[0] ... v2.b[15] |  |v18.4s           ...           v30.4s    |
  //  |v3.b[0] ... v3.b[15] |  |v19.4s           ...           v31.4s    |
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 4x4 block
  //
  // No attempt had been made so far at implementing the RUY_OPT_MAX_STREAMING
  // optimization for this kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 64 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov w1, #16\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"


        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Some multiplications and 16-bit accumulation were already done above,
        // so we start right away in the middle.
        "sadalp  v16.4s, v8.8h\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "sadalp  v24.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "sadalp  v25.4s, v9.8h\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "sadalp  v26.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "sadalp  v27.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "sadalp  v28.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "sadalp  v29.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "sadalp  v30.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "sadalp  v31.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"



        // Each iteration of this loop advances by 16 levels of depth.
        "add w1, w1, #16\n"

        // Loop termination condition
        "cmp w1, w12\n"

        "blt 2b\n"

        "79:\n"

        "sadalp  v16.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"

        "sadalp  v24.4s, v8.8h\n"
        "sadalp  v25.4s, v9.8h\n"
        "sadalp  v26.4s, v10.8h\n"
        "sadalp  v27.4s, v11.8h\n"
        "sadalp  v28.4s, v12.8h\n"
        "sadalp  v29.4s, v13.8h\n"
        "sadalp  v30.4s, v14.8h\n"
        "sadalp  v31.4s, v15.8h\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 4x4 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x4 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Reduce 32bit accumulators horizontally.
        "addp v16.4s, v16.4s, v17.4s\n"
        "addp v18.4s, v18.4s, v19.4s\n"
        "addp v20.4s, v20.4s, v21.4s\n"
        "addp v22.4s, v22.4s, v23.4s\n"
        "addp v24.4s, v24.4s, v25.4s\n"
        "addp v26.4s, v26.4s, v27.4s\n"
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "addp v16.4s, v16.4s, v18.4s\n"
        "addp v17.4s, v20.4s, v22.4s\n"
        "addp v18.4s, v24.4s, v26.4s\n"
        "addp v19.4s, v28.4s, v30.4s\n"

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #2\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #2\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        RUY_MAKE_ZERO(v8)
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec
        "add x5, x4, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"

        "ld1 {v15.4s}, [x4]\n" // multiplier_fixedpoint

        // Now we load: bias data, LHS sums data, RHS sums data.

        // First, load the base pointers from the params.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        "add x5, x1, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 4 bias values.
        "ld1 {v14.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v14.4s\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "add v19.4s, v19.4s, v14.4s\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[1]\n"
        "mls v18.4s, v10.4s, v14.s[2]\n"
        "mls v19.4s, v10.4s, v14.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v11.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v11.4s\n"

        // If the destination is int32, it means the user asks for the raw
        // accumulators, no need for us to downquantize the value.
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, %x[row], lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ld1 {v14.4s}, [x1]\n"

        "smax v12.4s, v14.4s, v8.4s\n"

        "sshl v16.4s, v16.4s, v12.4s\n"
        "sshl v17.4s, v17.4s, v12.4s\n"
        "sshl v18.4s, v18.4s, v12.4s\n"
        "sshl v19.4s, v19.4s, v12.4s\n"

        "smin v12.4s, v14.4s, v8.4s\n"

        // Apply the fixed-point part of the multiplier.
        "sqrdmulh v16.4s, v16.4s, v15.4s\n"
        "sqrdmulh v17.4s, v17.4s, v15.4s\n"
        "sqrdmulh v18.4s, v18.4s, v15.4s\n"
        "sqrdmulh v19.4s, v19.4s, v15.4s\n"

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
        "and v8.16b, v16.16b, v12.16b\n"
        "and v9.16b, v17.16b, v12.16b\n"
        "and v14.16b, v18.16b, v12.16b\n"
        "and v15.16b, v19.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v16.4s, v16.4s, v8.4s\n"
        "sqadd v17.4s, v17.4s, v9.4s\n"
        "sqadd v18.4s, v18.4s, v14.4s\n"
        "sqadd v19.4s, v19.4s, v15.4s\n"
#endif
        // At this point we have reduced the problem of correctly implementing
        // rounding divide-by-power-of-two, to what the SRSHL instruction can
        // do.
        "srshl v16.4s, v16.4s, v12.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"
        "srshl v18.4s, v18.4s, v12.4s\n"
        "srshl v19.4s, v19.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "add v16.8h, v16.8h, v14.8h\n"
        "add v17.8h, v17.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        "sqxtun2 v16.16b, v17.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4 && w2 == 4, i.e. if all of the 4x4 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #4\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[4], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[5], [x3], #1\n"
        "st1 {v16.b}[6], [x3], #1\n"
        "st1 {v16.b}[7], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[8], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[9], [x3], #1\n"
        "st1 {v16.b}[10], [x3], #1\n"
        "st1 {v16.b}[11], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[12], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[13], [x3], #1\n"
        "st1 {v16.b}[14], [x3], #1\n"
        "st1 {v16.b}[15], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "add v16.8h, v16.8h, v14.8h\n"
        "add v17.8h, v17.8h, v14.8h\n"

        // Cast-and-saturate from int16 to int8
        "sqxtn v16.8b, v16.8h\n"
        "sqxtn2 v16.16b, v17.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4 && w2 == 4, i.e. if all of the 4x4 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #4\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[4], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[5], [x3], #1\n"
        "st1 {v16.b}[6], [x3], #1\n"
        "st1 {v16.b}[7], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[8], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[9], [x3], #1\n"
        "st1 {v16.b}[10], [x3], #1\n"
        "st1 {v16.b}[11], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[12], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[13], [x3], #1\n"
        "st1 {v16.b}[14], [x3], #1\n"
        "st1 {v16.b}[15], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.4h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Load the clamp_min, clamp_max bounds
        "ldrh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        "str q17, [%[dst_tmp_buf], #16]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.h}[0], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v16.h}[1], [x3], #2\n"
        "st1 {v16.h}[2], [x3], #2\n"
        "st1 {v16.h}[3], [x3], #2\n"
        "mov x3, x4\n"
        "st1 {v16.h}[4], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v16.h}[5], [x3], #2\n"
        "st1 {v16.h}[6], [x3], #2\n"
        "st1 {v16.h}[7], [x3], #2\n"
        "mov x3, x4\n"
        "st1 {v17.h}[0], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v17.h}[1], [x3], #2\n"
        "st1 {v17.h}[2], [x3], #2\n"
        "st1 {v17.h}[3], [x3], #2\n"
        "mov x3, x4\n"
        "st1 {v17.h}[4], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v17.h}[5], [x3], #2\n"
        "st1 {v17.h}[6], [x3], #2\n"
        "st1 {v17.h}[7], [x3], #2\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #8\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // At this point, v20 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        "str q17, [%[dst_tmp_buf], #16]\n"
        "str q18, [%[dst_tmp_buf], #32]\n"
        "str q19, [%[dst_tmp_buf], #48]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v16.s}[1], [x3], #4\n"
        "st1 {v16.s}[2], [x3], #4\n"
        "st1 {v16.s}[3], [x3], #4\n"
        "mov x3, x4\n"
        "st1 {v17.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v17.s}[1], [x3], #4\n"
        "st1 {v17.s}[2], [x3], #4\n"
        "st1 {v17.s}[3], [x3], #4\n"
        "mov x3, x4\n"
        "st1 {v18.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v18.s}[1], [x3], #4\n"
        "st1 {v18.s}[2], [x3], #4\n"
        "st1 {v18.s}[3], [x3], #4\n"
        "mov x3, x4\n"
        "st1 {v19.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v19.s}[1], [x3], #4\n"
        "st1 {v19.s}[2], [x3], #4\n"
        "st1 {v19.s}[3], [x3], #4\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #16\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #4\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #4\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #2\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #16\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Similar to existing Kernel8bitNeonOutOfOrder but specialized for the case of
// RHS cols == 1.
// Relevant target CPUs for this kernel include ARM Cortex-A73 and Cortex-A75,
// since these are 64-bit, out-of-order and without dotprod support.
void Kernel8bitNeonOutOfOrder1Col(const KernelParams8bit<4, 4>& params) {
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeon, optimized for out-of-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v19 are int32 accumulators.
  // During accumulation, v0 -- v3 are used to load int8 data from LHS and
  // v4 from RHS:
  //
  //                         int8 RHS 16x1 block
  //                           /-----------\
  //                           |v4.b[0]    |
  //                           |  ...      |
  //                           |v4.b[15]   |
  //                           \-----------/
  //    int8 LHS 4x16 block
  //  /---------------------\  /-----------\
  //  |v0.b[0] ... v0.b[15] |  |v16.4s     |
  //  |v1.b[0] ... v1.b[15] |  |v17.4s     |
  //  |v2.b[0] ... v2.b[15] |  |v18.4s     |
  //  |v3.b[0] ... v3.b[15] |  |v19.4s     |
  //  \---------------------/  \-----------/
  //                         int32 accumulators 4x1 block
  //
  // No attempt had been made so far at implementing the RUY_OPT_MAX_STREAMING
  // optimization for this kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 64 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "add %[rhs_ptr], %[rhs_ptr], #48\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov w1, #16\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Some multiplications and 16-bit accumulation were already done above,
        // so we start right away in the middle.
        "sadalp  v16.4s, v8.8h\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "add %[rhs_ptr], %[rhs_ptr], #48\n"
        "sadalp  v17.4s, v9.8h\n"
        "sadalp  v18.4s, v10.8h\n"
        "sadalp  v19.4s, v11.8h\n"

        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        // Each iteration of this loop advances by 16 levels of depth.
        "add w1, w1, #16\n"

        // Loop termination condition
        "cmp w1, w12\n"

        "blt 2b\n"

        "79:\n"

        "sadalp  v16.4s, v8.8h\n"
        "sadalp  v17.4s, v9.8h\n"
        "sadalp  v18.4s, v10.8h\n"
        "sadalp  v19.4s, v11.8h\n"

        // End of accumulation. The registers v16 -- v19 contain the final
        // int32 accumulator values of the current 4x1 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x1 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Reduce 32bit accumulators horizontally.
        "addp v16.4s, v16.4s, v17.4s\n"
        "addp v18.4s, v18.4s, v19.4s\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "addp v16.4s, v16.4s, v18.4s\n"

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #2\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        // (still multiply column stride by 4 due to packing)
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #2\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        RUY_MAKE_ZERO(v8)
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec
        "add x5, x4, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"

        "ld1 {v15.4s}, [x4]\n" // multiplier_fixedpoint

        // Now we load: bias data, LHS sums data, RHS sums data.

        // First, load the base pointers from the params.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        "add x5, x1, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 4 bias values.
        "ld1 {v14.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        "add %[rhs_ptr], %[rhs_ptr], #48\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        // (all four 32-bit accumulators are in v16 at this point)
        "add v16.4s, v16.4s, v14.4s\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"

        // If the destination is int32, it means the user asks for the raw
        // accumulators, no need for us to downquantize the value.
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, %x[row], lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ld1 {v14.4s}, [x1]\n"

        "smax v12.4s, v14.4s, v8.4s\n"

        "sshl v16.4s, v16.4s, v12.4s\n"

        "smin v12.4s, v14.4s, v8.4s\n"

        // Apply the fixed-point part of the multiplier.
        "sqrdmulh v16.4s, v16.4s, v15.4s\n"

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
        "and v8.16b, v16.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sqadd v16.4s, v16.4s, v8.4s\n"
#endif
        // At this point we have reduced the problem of correctly implementing
        // rounding divide-by-power-of-two, to what the SRSHL instruction can
        // do.
        "srshl v16.4s, v16.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        // After this instruction, all data is in lower half (64-bits) of v16
        "sqxtn v16.4h, v16.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "add v16.8h, v16.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        // Now all data is in the first 32-bits of v16
        "sqxtun v16.8b, v16.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"

        // Compute how much of the 4x1 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4, i.e. if all of the 4x4 block fits.
        "cmp w1, w3\n"

        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        // After this, all values for output are in the lower half (64 bits) of v16.
        "sqxtn v16.4h, v16.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "add v16.8h, v16.8h, v14.8h\n"

        // Cast-and-saturate from int16 to int8
        "sqxtn v16.8b, v16.8h\n"
        // At this point, we only need 4 lowest 8-bit values in v16.

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"

        // Test if w1==4, i.e. if all of the 4x1 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.4h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        // After this instruction, all data is in lower half of v16.
        "sqxtn v16.4h, v16.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        // Load the clamp_min, clamp_max bounds
        "ldrh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.h}[0], [x3], #2\n"
        "st1 {v16.h}[1], [x3], #2\n"
        "st1 {v16.h}[2], [x3], #2\n"
        "st1 {v16.h}[3], [x3], #2\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #8\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"

        // Test if w1==4 i.e. if all of the 4x1 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.s}[0], [x3], #4\n"
        "st1 {v16.s}[1], [x3], #4\n"
        "st1 {v16.s}[2], [x3], #4\n"
        "st1 {v16.s}[3], [x3], #4\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #16\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #4\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #4\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #2\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov w1, #16\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19");
}

// Variant of the above Kernel8bitNeonOutOfOrder, tuned for in-order CPUs.
// Specifically here, the relevant in-order CPUs are ARM Cortex-A53 and
// the original Cortex-A55, since these are 64-bit and do not support dotprod.
//
// While this kernel does not have a direct equivalent in gemmlowp, it was
// developed based on insights that David Mansell at ARM shared with their
// contribution of gemmlowp kernels tuned for Cortex-A53, with very helpful
// comments. Specifically, see this comment about tuning for Cortex-A53:
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L4215
void Kernel8bitNeonInOrder(const KernelParams8bit<4, 4>& params) {
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeon, optimized for in-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v3 are used to load int8 data from LHS and
  // v4 -- v7 from RHS:
  //
  //                                      int8 RHS 16x4 block
  //                           /-----------------------------------------\
  //                           |v4.b[0]          ...           v7.b[0]   |
  //                           |  ...                            ...     |
  //                           |v4.b[15]         ...           v7.b[15]  |
  //                           \-----------------------------------------/
  //    int8 LHS 4x16 block
  //  /---------------------\  /-----------------------------------------\
  //  |v0.b[0] ... v0.b[15] |  |v16.4s           ...           v28.4s    |
  //  |v1.b[0] ... v1.b[15] |  |v17.4s           ...           v29.4s    |
  //  |v2.b[0] ... v2.b[15] |  |v18.4s           ...           v30.4s    |
  //  |v3.b[0] ... v3.b[15] |  |v19.4s           ...           v31.4s    |
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 4x4 block
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        RUY_MAKE_ZERO(v16)
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        RUY_MAKE_ZERO(v17)
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        RUY_MAKE_ZERO(v18)
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        RUY_MAKE_ZERO(v19)
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        RUY_MAKE_ZERO(v20)
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        RUY_MAKE_ZERO(v21)
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        RUY_MAKE_ZERO(v22)
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"
        RUY_MAKE_ZERO(v23)

        // Load the first 64 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v24)
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v25)
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v26)
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v27)
        "ld1 {v4.16b}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v28)
        "ld1 {v5.16b}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v29)
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v30)
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v31)


        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 16.
        "mov w1, #16\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"


        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Some multiplications and 16-bit accumulation were already done above,
        // so we start right away in the middle.
        "sadalp  v16.4s, v8.8h\n"
        "ldr d4, [%[rhs_ptr], #0]\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "ldr x7, [%[rhs_ptr], #8]\n"
        "sadalp  v17.4s, v9.8h\n"
        "ldr d5, [%[rhs_ptr], #16]\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "ldr x8, [%[rhs_ptr], #24]\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        "sadalp  v20.4s, v12.8h\n"
        // Each iteration of this loop advances by 16 levels of depth.
        "add w1, w1, #16\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        // Loop termination condition
        "cmp w1, w12\n"
        "sadalp  v21.4s, v13.8h\n"
        "ldr x3, [%[lhs_ptr], #-56]\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "ldr x4, [%[lhs_ptr], #-40]\n"
        "sadalp  v22.4s, v14.8h\n"
        "ldr x5, [%[lhs_ptr], #-24]\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "ldr x6, [%[lhs_ptr], #-8]\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "ldr x9, [%[rhs_ptr], #-24]\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"
        "ldr d6, [%[rhs_ptr], #-32]\n"
        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "ldr d0, [%[lhs_ptr], #-64]\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "ldr d1, [%[lhs_ptr], #-48]\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "ins v4.d[1], x7\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"
        "ins v5.d[1], x8\n"

        "ldr d2, [%[lhs_ptr], #-32]\n"
        "ins v0.d[1], x3\n"
        "sadalp  v24.4s, v8.8h\n"
        "ldr d3, [%[lhs_ptr], #-16]\n"
        "ins v1.d[1], x4\n"
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "ins v2.d[1], x5\n"
        "sadalp  v25.4s, v9.8h\n"
        "ins v3.d[1], x6\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "ldr d7, [%[rhs_ptr], #-16]\n"
        "sadalp  v26.4s, v10.8h\n"
        "ldr x10, [%[rhs_ptr], #-8]\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        "sadalp  v27.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "sadalp  v28.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "sadalp  v29.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "sadalp  v30.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "sadalp  v31.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"

        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "ins v6.d[1], x9\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "ins v7.d[1], x10\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"

        "blt 2b\n"

        "79:\n"

        "sadalp  v16.4s, v8.8h\n"
        "smull    v8.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v9.8h\n"
        "smull    v9.8h,  v1.8b,  v6.8b\n"
        "sadalp  v18.4s, v10.8h\n"
        "smull    v10.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v11.8h\n"
        "smull    v11.8h,  v3.8b,  v6.8b\n"
        "sadalp  v20.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v7.8b\n"
        "sadalp  v21.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v7.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "smlal2   v8.8h,  v0.16b,  v6.16b\n"
        "smlal2   v9.8h,  v1.16b,  v6.16b\n"
        "smlal2   v10.8h,  v2.16b,  v6.16b\n"
        "smlal2   v11.8h,  v3.16b,  v6.16b\n"

        "smlal2   v12.8h,  v0.16b,  v7.16b\n"
        "smlal2   v13.8h,  v1.16b,  v7.16b\n"
        "smlal2   v14.8h,  v2.16b,  v7.16b\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"

        "sadalp  v24.4s, v8.8h\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "sadalp  v25.4s, v9.8h\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "sadalp  v26.4s, v10.8h\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "sadalp  v27.4s, v11.8h\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "sadalp  v28.4s, v12.8h\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "sadalp  v29.4s, v13.8h\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "sadalp  v30.4s, v14.8h\n"
        "sadalp  v31.4s, v15.8h\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 4x4 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 4x4 block. We intertwine
        // these two aspects whenever possible for optimal pipelining, both
        // at the data flow level (prefetch data for next block as early as
        // possible) and instruction pipelining level (some of the next-block
        // work can dual-issue with some of the final work on the current
        // block).

        // Reduce 32bit accumulators horizontally.
        "addp v16.4s, v16.4s, v17.4s\n"
        "addp v18.4s, v18.4s, v19.4s\n"
        "addp v20.4s, v20.4s, v21.4s\n"
        "addp v22.4s, v22.4s, v23.4s\n"
        "addp v24.4s, v24.4s, v25.4s\n"
        "addp v26.4s, v26.4s, v27.4s\n"
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "addp v16.4s, v16.4s, v18.4s\n"
        "addp v17.4s, v20.4s, v22.4s\n"
        "addp v18.4s, v24.4s, v26.4s\n"
        "addp v19.4s, v28.4s, v30.4s\n"

        // Logic to advance to the next block in preparation for the next
        // iteration of the main loop. For now, we only want to compute
        // the LHS and RHS data pointers, lhs_col_ptr and rhs_col_ptr. We are
        // not yet ready to update the values of row and col, as we still need
        // the current values for the rest of the work on the current block.

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #2\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #2\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        RUY_MAKE_ZERO(v8)
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec
        "add x5, x4, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"

        "ld1 {v15.4s}, [x4]\n" // multiplier_fixedpoint

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"
        "add x5, x1, %x[row], lsl #2\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 4 bias values.
        "ld1 {v14.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "ldr d0, [%[lhs_ptr], #0]\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "add v16.4s, v16.4s, v14.4s\n"
        "ldr d1, [%[lhs_ptr], #16]\n"
        "add v17.4s, v17.4s, v14.4s\n"
        "ldr d2, [%[lhs_ptr], #32]\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "ldr d3, [%[lhs_ptr], #48]\n"
        "add v19.4s, v19.4s, v14.4s\n"
        "ldr d4, [%[rhs_ptr], #0]\n"
        "ldr d5, [%[rhs_ptr], #16]\n"
        "ldr d6, [%[rhs_ptr], #32]\n"
        "ldr d7, [%[rhs_ptr], #48]\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[1]\n"
        "mls v18.4s, v10.4s, v14.s[2]\n"
        "mls v19.4s, v10.4s, v14.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v11.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v11.4s\n"

        // If the destination is int32, it means the user asks for the raw
        // accumulators, no need for us to downquantize the value.
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.


        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, %x[row], lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ld1 {v14.4s}, [x1]\n"

        "smax v12.4s, v14.4s, v8.4s\n"
        "ldr x1, [%[lhs_ptr], #8]\n"

        "sshl v16.4s, v16.4s, v12.4s\n"
        "ldr x2, [%[lhs_ptr], #24]\n"
        "sshl v17.4s, v17.4s, v12.4s\n"
        "ldr x3, [%[lhs_ptr], #40]\n"
        "sshl v18.4s, v18.4s, v12.4s\n"
        "ldr x4, [%[lhs_ptr], #56]\n"
        "sshl v19.4s, v19.4s, v12.4s\n"

        "smin v12.4s, v14.4s, v8.4s\n"

        // Apply the fixed-point part of the multiplier.
        "ins v0.d[1], x1\n"
        "ldr x1, [%[rhs_ptr], #8]\n"
        "sqrdmulh v16.4s, v16.4s, v15.4s\n"
        "ins v1.d[1], x2\n"
        "ldr x2, [%[rhs_ptr], #24]\n"
        "sqrdmulh v17.4s, v17.4s, v15.4s\n"
        "ins v2.d[1], x3\n"
        "ldr x3, [%[rhs_ptr], #40]\n"
        "sqrdmulh v18.4s, v18.4s, v15.4s\n"
        "ins v3.d[1], x4\n"
        "ldr x4, [%[rhs_ptr], #56]\n"
        "sqrdmulh v19.4s, v19.4s, v15.4s\n"

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
        "and v8.16b, v16.16b, v12.16b\n"
        "and v9.16b, v17.16b, v12.16b\n"
        "and v14.16b, v18.16b, v12.16b\n"
        "and v15.16b, v19.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v16.4s, v16.4s, v8.4s\n"
        "sqadd v17.4s, v17.4s, v9.4s\n"
        "sqadd v18.4s, v18.4s, v14.4s\n"
        "sqadd v19.4s, v19.4s, v15.4s\n"
#endif
        // At this point we have reduced the problem of correctly implementing
        // rounding divide-by-power-of-two, to what the SRSHL instruction can
        // do.
        "srshl v16.4s, v16.4s, v12.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"
        "srshl v18.4s, v18.4s, v12.4s\n"
        "srshl v19.4s, v19.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        "ins v4.d[1], x1\n"
        "sqxtn v16.4h, v16.4s\n"
        "ins v5.d[1], x2\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "ins v6.d[1], x3\n"
        "sqxtn v17.4h, v18.4s\n"
        "ins v7.d[1], x4\n"
        RUY_MAKE_ZERO(v18)
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v19)

        // Add the destination zero point
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "dup v14.8h, v13.h[4]\n"
        RUY_MAKE_ZERO(v20)
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        "add v16.8h, v16.8h, v14.8h\n"
        RUY_MAKE_ZERO(v21)
        "add v17.8h, v17.8h, v14.8h\n"
        RUY_MAKE_ZERO(v22)

        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        RUY_MAKE_ZERO(v23)
        "sqxtun2 v16.16b, v17.8h\n"
        RUY_MAKE_ZERO(v24)

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        RUY_MAKE_ZERO(v25)
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        RUY_MAKE_ZERO(v26)
        "dup v14.16b, w2\n"  // clamp_min
        RUY_MAKE_ZERO(v27)
        "dup v15.16b, w3\n"  // clamp_max
        RUY_MAKE_ZERO(v28)

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        RUY_MAKE_ZERO(v29)
        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"
        RUY_MAKE_ZERO(v30)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        RUY_MAKE_ZERO(v31)
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #4\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[4], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[5], [x3], #1\n"
        "st1 {v16.b}[6], [x3], #1\n"
        "st1 {v16.b}[7], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[8], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[9], [x3], #1\n"
        "st1 {v16.b}[10], [x3], #1\n"
        "st1 {v16.b}[11], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[12], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[13], [x3], #1\n"
        "st1 {v16.b}[14], [x3], #1\n"
        "st1 {v16.b}[15], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        "ins v4.d[1], x1\n"
        "sqxtn v16.4h, v16.4s\n"
        "ins v5.d[1], x2\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "ins v6.d[1], x3\n"
        "sqxtn v17.4h, v18.4s\n"
        "ins v7.d[1], x4\n"
        RUY_MAKE_ZERO(v18)
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v19)

        // Add the destination zero point
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "dup v14.8h, v13.h[4]\n"
        RUY_MAKE_ZERO(v20)
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        "add v16.8h, v16.8h, v14.8h\n"
        RUY_MAKE_ZERO(v21)
        "add v17.8h, v17.8h, v14.8h\n"
        RUY_MAKE_ZERO(v22)

        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"
        RUY_MAKE_ZERO(v23)
        "sqxtn2 v16.16b, v17.8h\n"
        RUY_MAKE_ZERO(v24)

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        RUY_MAKE_ZERO(v25)
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        RUY_MAKE_ZERO(v26)
        "dup v14.16b, w2\n"  // clamp_min
        RUY_MAKE_ZERO(v27)
        "dup v15.16b, w3\n"  // clamp_max
        RUY_MAKE_ZERO(v28)

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        RUY_MAKE_ZERO(v29)
        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"
        RUY_MAKE_ZERO(v30)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        RUY_MAKE_ZERO(v31)
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "st1 {v16.16b}, [%[dst_tmp_buf]]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #4\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.b}[0], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[1], [x3], #1\n"
        "st1 {v16.b}[2], [x3], #1\n"
        "st1 {v16.b}[3], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[4], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[5], [x3], #1\n"
        "st1 {v16.b}[6], [x3], #1\n"
        "st1 {v16.b}[7], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[8], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[9], [x3], #1\n"
        "st1 {v16.b}[10], [x3], #1\n"
        "st1 {v16.b}[11], [x3], #1\n"
        "mov x3, x4\n"
        "st1 {v16.b}[12], [x3], #1\n"
        "add x4, x4, x11\n"
        "st1 {v16.b}[13], [x3], #1\n"
        "st1 {v16.b}[14], [x3], #1\n"
        "st1 {v16.b}[15], [x3], #1\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #4\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.4h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "ins v4.d[1], x1\n"
        "sqxtn v16.4h, v16.4s\n"
        "ins v5.d[1], x2\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "ins v6.d[1], x3\n"
        "sqxtn v17.4h, v18.4s\n"
        "ins v7.d[1], x4\n"
        RUY_MAKE_ZERO(v18)
        "sqxtn2 v17.8h, v19.4s\n"

        // At this point, v18 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v19)

        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        RUY_MAKE_ZERO(v20)
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)

        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)

        // Load the clamp_min, clamp_max bounds
        "ldrh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        RUY_MAKE_ZERO(v25)
        "ldrh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        RUY_MAKE_ZERO(v26)
        "dup v14.8h, w2\n"  // clamp_min
        RUY_MAKE_ZERO(v27)
        "dup v15.8h, w3\n"  // clamp_max
        RUY_MAKE_ZERO(v28)

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        RUY_MAKE_ZERO(v29)
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"
        RUY_MAKE_ZERO(v30)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        RUY_MAKE_ZERO(v31)
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

       // Test if w1==4 && w2 == 4, i.e. if all of the 4x4 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        "str q17, [%[dst_tmp_buf], #16]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.h}[0], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v16.h}[1], [x3], #2\n"
        "st1 {v16.h}[2], [x3], #2\n"
        "st1 {v16.h}[3], [x3], #2\n"
        "mov x3, x4\n"
        "st1 {v16.h}[4], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v16.h}[5], [x3], #2\n"
        "st1 {v16.h}[6], [x3], #2\n"
        "st1 {v16.h}[7], [x3], #2\n"
        "mov x3, x4\n"
        "st1 {v17.h}[0], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v17.h}[1], [x3], #2\n"
        "st1 {v17.h}[2], [x3], #2\n"
        "st1 {v17.h}[3], [x3], #2\n"
        "mov x3, x4\n"
        "st1 {v17.h}[4], [x3], #2\n"
        "add x4, x4, x11\n"
        "st1 {v17.h}[5], [x3], #2\n"
        "st1 {v17.h}[6], [x3], #2\n"
        "st1 {v17.h}[7], [x3], #2\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #8\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        "ldr x1, [%[lhs_ptr], #8]\n"
        "ldr x2, [%[lhs_ptr], #24]\n"
        "ldr x3, [%[lhs_ptr], #40]\n"
        "ldr x4, [%[lhs_ptr], #56]\n"

        "ins v0.d[1], x1\n"
        "ldr x1, [%[rhs_ptr], #8]\n"
        "ins v1.d[1], x2\n"
        "ldr x2, [%[rhs_ptr], #24]\n"
        "ins v2.d[1], x3\n"
        "ldr x3, [%[rhs_ptr], #40]\n"
        "ins v3.d[1], x4\n"
        "ldr x4, [%[rhs_ptr], #56]\n"
        "ins v4.d[1], x1\n"
        "ins v5.d[1], x2\n"
        "ins v6.d[1], x3\n"
        "ins v7.d[1], x4\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // At this point, v20 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).

        RUY_MAKE_ZERO(v20)
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        RUY_MAKE_ZERO(v21)
        "add %[rhs_ptr], %[rhs_ptr], #64\n"
        RUY_MAKE_ZERO(v22)

        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)

        // Compute how much of the 4x4 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 4x4, there are some 4x4 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        RUY_MAKE_ZERO(v31)
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #4\n"
        "cmp w1, #4\n"
        // Compute w1 = how many rows of the 4x4 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #4\n"
        // Compute w2 = how many cols of the 4x4 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==4 && w2 == 4, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        "mov x4, %[dst_ptr]\n"
        // Yes, all of the 4x4 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 4x4 block fits.
        // Store to dst_tmp_buf
        "str q16, [%[dst_tmp_buf], #0]\n"
        "str q17, [%[dst_tmp_buf], #16]\n"
        "str q18, [%[dst_tmp_buf], #32]\n"
        "str q19, [%[dst_tmp_buf], #48]\n"
        // Slow loop copying from dst_tmp_buf to dst.
        "mov x3, %[dst_tmp_buf]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 4x4 block fits.
        "mov x3, x4\n"
        "st1 {v16.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v16.s}[1], [x3], #4\n"
        "st1 {v16.s}[2], [x3], #4\n"
        "st1 {v16.s}[3], [x3], #4\n"
        "mov x3, x4\n"
        "st1 {v17.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v17.s}[1], [x3], #4\n"
        "st1 {v17.s}[2], [x3], #4\n"
        "st1 {v17.s}[3], [x3], #4\n"
        "mov x3, x4\n"
        "st1 {v18.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v18.s}[1], [x3], #4\n"
        "st1 {v18.s}[2], [x3], #4\n"
        "st1 {v18.s}[3], [x3], #4\n"
        "mov x3, x4\n"
        "st1 {v19.s}[0], [x3], #4\n"
        "add x4, x4, x11\n"
        "st1 {v19.s}[1], [x3], #4\n"
        "st1 {v19.s}[2], [x3], #4\n"
        "st1 {v19.s}[3], [x3], #4\n"
        "31:\n"

        "add %[dst_ptr], %[dst_ptr], #16\n"

        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        "smull    v8.8h,  v0.8b,  v4.8b\n"
        "smull    v9.8h,  v1.8b,  v4.8b\n"
        "smull    v10.8h,  v2.8b,  v4.8b\n"
        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "smull    v11.8h,  v3.8b,  v4.8b\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "smull    v12.8h,  v0.8b,  v5.8b\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "smull    v13.8h,  v1.8b,  v5.8b\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"
        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "smlal2   v8.8h,  v0.16b,  v4.16b\n"
        "smlal2   v9.8h,  v1.16b,  v4.16b\n"
        "smlal2   v10.8h,  v2.16b,  v4.16b\n"
        "smlal2   v11.8h,  v3.16b,  v4.16b\n"
        "smlal2   v12.8h,  v0.16b,  v5.16b\n"
        "smlal2   v13.8h,  v1.16b,  v5.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"


        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #4\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #4\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #2\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #16\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params),[dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Kernel taking advantage of the optional dotprod instruction.
// This is very similar to (and directly inspired by) this gemmlowp kernel
// which was contributed by David Mansell at ARM:
// NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_dotproduct
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L3391
//
// Besides the ruy-ification, the main difference here is that we use a 8x8
// instead of 12x8 width, so as to stick to power-of-two widths. This slightly
// narrower kernel layout is still wide enough to achieve high performance
// although we haven't actually performed a real comparison to know exactly
// how this compares to ARM's aforementioned kernel.
//
// Relevant target CPUs for this kernel include ARM Cortex-A76,
// since these are 64-bit, out-of-order and with dotprod support.
void Kernel8bitNeonDotprodOutOfOrder(const KernelParams8bit<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeonDotprod, optimized for out-of-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v15 are used to load int8 data from LHS and
  // RHS. At least v0 and v1 are used to load a 8x4 block of LHS, and v2 and
  // v3 are used to load a 4x8 block of RHS, like this:
  //
  //                                      int8 RHS 4x8 block
  //                           /-----------------------------------------\
  //                           |v2.b[0] ... v2.b[12] v3.b[0] ... v3.b[12]|
  //                           |  ...                              ...   |
  //                           |v2.b[3] ... v2.b[15] v3.b[3] ... v3.b[15]|
  //                           \-----------------------------------------/
  //    int8 LHS 8x4 block
  //  /---------------------\  /-----------------------------------------\
  //  |v0.b[0]  ... v0.b[3] |  |v16.s[0]           ...           v30.s[0]|
  //  |  ...          ...   |  |  ...                              ...   |
  //  |v0.b[12] ... v0.b[15]|  |v16.s[3]           ...           v30.s[3]|
  //  |v1.b[0]  ... v1.b[3] |  |v17.s[0]           ...           v31.s[0]|
  //  |  ...         ...    |  |  ...                              ...   |
  //  |v1.b[12] ... v1.b[15]|  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 8x8 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for loading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Optional, maximally-streaming, partial-unrolling (4x unrolled)
        // optimization of the kernel inner loop (over depth). For more
        // comments, see the non-unrolled loop below after the #endif.
#if RUY_OPT_ENABLED(RUY_OPT_MAX_STREAMING)
        "cmp w12, #32\n"
        "blt 78f\n"

        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v8.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v9.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v10.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v11.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v12.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v13.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v14.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v15.16b}, [%[rhs_ptr]], #16\n"
        "mov w1, #16\n"

        "and w3, w12, #-16\n"
        "81:\n"
        "add w1, w1, #16\n"

        ".word 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        ".word 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        ".word 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".word 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        "ldr q0, [%[lhs_ptr], #0]\n"
        ".word 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".word 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        ".word 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".word 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        "ldr q2, [%[rhs_ptr], #0]\n"
        ".word 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".word 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".word 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".word 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"
        "ldr q1, [%[lhs_ptr], #16]\n"

        ".word 0x4f87e098  // sdot v24.4s, v4.16b, v7.4b[0]\n"
        ".word 0x4fa7e09a  // sdot v26.4s, v4.16b, v7.4b[1]\n"
        "ldr q3, [%[rhs_ptr], #16]\n"
        ".word 0x4f87e89c  // sdot v28.4s, v4.16b, v7.4b[2]\n"
        ".word 0x4fa7e89e  // sdot v30.4s, v4.16b, v7.4b[3]\n"
        ".word 0x4f86e0b1  // sdot v17.4s, v5.16b, v6.4b[0]\n"
        ".word 0x4fa6e0b3  // sdot v19.4s, v5.16b, v6.4b[1]\n"
        ".word 0x4f86e8b5  // sdot v21.4s, v5.16b, v6.4b[2]\n"
        ".word 0x4fa6e8b7  // sdot v23.4s, v5.16b, v6.4b[3]\n"
        ".word 0x4f87e0b9  // sdot v25.4s, v5.16b, v7.4b[0]\n"
        ".word 0x4fa7e0bb  // sdot v27.4s, v5.16b, v7.4b[1]\n"
        ".word 0x4f87e8bd  // sdot v29.4s, v5.16b, v7.4b[2]\n"
        ".word 0x4fa7e8bf  // sdot v31.4s, v5.16b, v7.4b[3]\n"
        "ldr q5, [%[lhs_ptr], #48]\n"
        ".word 0x4f86e090  // sdot v16.4s, v4.16b, v6.4b[0]\n"
        ".word 0x4fa6e092  // sdot v18.4s, v4.16b, v6.4b[1]\n"
        "ldr q7, [%[rhs_ptr], #48]\n"
        ".word 0x4f86e894  // sdot v20.4s, v4.16b, v6.4b[2]\n"
        ".word 0x4fa6e896  // sdot v22.4s, v4.16b, v6.4b[3]\n"
        "ldr q4, [%[lhs_ptr], #32]\n"

        ".word 0x4f8be118  // sdot v24.4s, v8.16b, v11.4b[0]\n"
        ".word 0x4fabe11a  // sdot v26.4s, v8.16b, v11.4b[1]\n"
        "ldr q6, [%[rhs_ptr], #32]\n"
        ".word 0x4f8be91c  // sdot v28.4s, v8.16b, v11.4b[2]\n"
        ".word 0x4fabe91e  // sdot v30.4s, v8.16b, v11.4b[3]\n"
        ".word 0x4f8ae131  // sdot v17.4s, v9.16b, v10.4b[0]\n"
        ".word 0x4faae133  // sdot v19.4s, v9.16b, v10.4b[1]\n"
        ".word 0x4f8ae935  // sdot v21.4s, v9.16b, v10.4b[2]\n"
        ".word 0x4faae937  // sdot v23.4s, v9.16b, v10.4b[3]\n"
        ".word 0x4f8be139  // sdot v25.4s, v9.16b, v11.4b[0]\n"
        ".word 0x4fabe13b  // sdot v27.4s, v9.16b, v11.4b[1]\n"
        ".word 0x4f8be93d  // sdot v29.4s, v9.16b, v11.4b[2]\n"
        ".word 0x4fabe93f  // sdot v31.4s, v9.16b, v11.4b[3]\n"
        "ldr q9, [%[lhs_ptr], #80]\n"
        ".word 0x4f8ae110  // sdot v16.4s, v8.16b, v10.4b[0]\n"
        ".word 0x4faae112  // sdot v18.4s, v8.16b, v10.4b[1]\n"
        "ldr q11, [%[rhs_ptr], #80]\n"
        ".word 0x4f8ae914  // sdot v20.4s, v8.16b, v10.4b[2]\n"
        ".word 0x4faae916  // sdot v22.4s, v8.16b, v10.4b[3]\n"
        "ldr q8, [%[lhs_ptr], #64]\n"

        ".word 0x4f8fe198  // sdot v24.4s, v12.16b, v15.4b[0]\n"
        ".word 0x4fafe19a  // sdot v26.4s, v12.16b, v15.4b[1]\n"
        "ldr q10, [%[rhs_ptr], #64]\n"
        ".word 0x4f8fe99c  // sdot v28.4s, v12.16b, v15.4b[2]\n"
        ".word 0x4fafe99e  // sdot v30.4s, v12.16b, v15.4b[3]\n"
        "add %[lhs_ptr], %[lhs_ptr], #128\n"
        ".word 0x4f8ee1b1  // sdot v17.4s, v13.16b, v14.4b[0]\n"
        ".word 0x4faee1b3  // sdot v19.4s, v13.16b, v14.4b[1]\n"
        "add %[rhs_ptr], %[rhs_ptr], #128\n"
        ".word 0x4f8ee9b5  // sdot v21.4s, v13.16b, v14.4b[2]\n"
        ".word 0x4faee9b7  // sdot v23.4s, v13.16b, v14.4b[3]\n"
        ".word 0x4f8fe1b9  // sdot v25.4s, v13.16b, v15.4b[0]\n"
        ".word 0x4fafe1bb  // sdot v27.4s, v13.16b, v15.4b[1]\n"
        "cmp w1, w3\n"
        ".word 0x4f8fe9bd  // sdot v29.4s, v13.16b, v15.4b[2]\n"
        ".word 0x4fafe9bf  // sdot v31.4s, v13.16b, v15.4b[3]\n"
        "ldr q13, [%[lhs_ptr], #-16]\n"
        ".word 0x4f8ee190  // sdot v16.4s, v12.16b, v14.4b[0]\n"
        ".word 0x4faee192  // sdot v18.4s, v12.16b, v14.4b[1]\n"
        "ldr q15, [%[rhs_ptr], #-16]\n"
        ".word 0x4f8ee994  // sdot v20.4s, v12.16b, v14.4b[2]\n"
        ".word 0x4faee996  // sdot v22.4s, v12.16b, v14.4b[3]\n"
        "ldr q12, [%[lhs_ptr], #-32]\n"

        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        "ldr q14, [%[rhs_ptr], #-32]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        "blt 81b\n"

        ".word 0x4f87e098  // sdot v24.4s, v4.16b, v7.4b[0]\n"
        ".word 0x4fa7e09a  // sdot v26.4s, v4.16b, v7.4b[1]\n"
        ".word 0x4f87e89c  // sdot v28.4s, v4.16b, v7.4b[2]\n"
        ".word 0x4fa7e89e  // sdot v30.4s, v4.16b, v7.4b[3]\n"
        ".word 0x4f86e0b1  // sdot v17.4s, v5.16b, v6.4b[0]\n"
        ".word 0x4fa6e0b3  // sdot v19.4s, v5.16b, v6.4b[1]\n"
        ".word 0x4f86e8b5  // sdot v21.4s, v5.16b, v6.4b[2]\n"
        ".word 0x4fa6e8b7  // sdot v23.4s, v5.16b, v6.4b[3]\n"
        ".word 0x4f87e0b9  // sdot v25.4s, v5.16b, v7.4b[0]\n"
        ".word 0x4fa7e0bb  // sdot v27.4s, v5.16b, v7.4b[1]\n"
        ".word 0x4f87e8bd  // sdot v29.4s, v5.16b, v7.4b[2]\n"
        ".word 0x4fa7e8bf  // sdot v31.4s, v5.16b, v7.4b[3]\n"
        ".word 0x4f86e090  // sdot v16.4s, v4.16b, v6.4b[0]\n"
        ".word 0x4fa6e092  // sdot v18.4s, v4.16b, v6.4b[1]\n"
        ".word 0x4f86e894  // sdot v20.4s, v4.16b, v6.4b[2]\n"
        ".word 0x4fa6e896  // sdot v22.4s, v4.16b, v6.4b[3]\n"

        ".word 0x4f8be118  // sdot v24.4s, v8.16b, v11.4b[0]\n"
        ".word 0x4fabe11a  // sdot v26.4s, v8.16b, v11.4b[1]\n"
        ".word 0x4f8be91c  // sdot v28.4s, v8.16b, v11.4b[2]\n"
        ".word 0x4fabe91e  // sdot v30.4s, v8.16b, v11.4b[3]\n"
        ".word 0x4f8ae131  // sdot v17.4s, v9.16b, v10.4b[0]\n"
        ".word 0x4faae133  // sdot v19.4s, v9.16b, v10.4b[1]\n"
        ".word 0x4f8ae935  // sdot v21.4s, v9.16b, v10.4b[2]\n"
        ".word 0x4faae937  // sdot v23.4s, v9.16b, v10.4b[3]\n"
        ".word 0x4f8be139  // sdot v25.4s, v9.16b, v11.4b[0]\n"
        ".word 0x4fabe13b  // sdot v27.4s, v9.16b, v11.4b[1]\n"
        ".word 0x4f8be93d  // sdot v29.4s, v9.16b, v11.4b[2]\n"
        ".word 0x4fabe93f  // sdot v31.4s, v9.16b, v11.4b[3]\n"
        ".word 0x4f8ae110  // sdot v16.4s, v8.16b, v10.4b[0]\n"
        ".word 0x4faae112  // sdot v18.4s, v8.16b, v10.4b[1]\n"
        ".word 0x4f8ae914  // sdot v20.4s, v8.16b, v10.4b[2]\n"
        ".word 0x4faae916  // sdot v22.4s, v8.16b, v10.4b[3]\n"

        ".word 0x4f8fe198  // sdot v24.4s, v12.16b, v15.4b[0]\n"
        ".word 0x4fafe19a  // sdot v26.4s, v12.16b, v15.4b[1]\n"
        ".word 0x4f8fe99c  // sdot v28.4s, v12.16b, v15.4b[2]\n"
        ".word 0x4fafe99e  // sdot v30.4s, v12.16b, v15.4b[3]\n"
        ".word 0x4f8ee1b1  // sdot v17.4s, v13.16b, v14.4b[0]\n"
        ".word 0x4faee1b3  // sdot v19.4s, v13.16b, v14.4b[1]\n"
        ".word 0x4f8ee9b5  // sdot v21.4s, v13.16b, v14.4b[2]\n"
        ".word 0x4faee9b7  // sdot v23.4s, v13.16b, v14.4b[3]\n"
        ".word 0x4f8fe1b9  // sdot v25.4s, v13.16b, v15.4b[0]\n"
        ".word 0x4fafe1bb  // sdot v27.4s, v13.16b, v15.4b[1]\n"
        ".word 0x4f8fe9bd  // sdot v29.4s, v13.16b, v15.4b[2]\n"
        ".word 0x4fafe9bf  // sdot v31.4s, v13.16b, v15.4b[3]\n"
        ".word 0x4f8ee190  // sdot v16.4s, v12.16b, v14.4b[0]\n"
        ".word 0x4faee192  // sdot v18.4s, v12.16b, v14.4b[1]\n"
        ".word 0x4f8ee994  // sdot v20.4s, v12.16b, v14.4b[2]\n"
        ".word 0x4faee996  // sdot v22.4s, v12.16b, v14.4b[3]\n"

        "78:\n"

#endif  // #if RUY_OPT_ENABLED(RUY_OPT_MAX_STREAMING)

        // Ordinary kernel inner loop (over depth), the simpler loop that the
        // above was an equivalent 4x-partially-unrolled version of.

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Because of the data that we have already loaded, we can start the
        // loop body right away with some multiply-adds.
        ".word 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        ".word 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        // Each iteration of this loop advances by 4 levels of depth.
        "add w1, w1, #4\n"
        ".word 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".word 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        ".word 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".word 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        // Loop termination condition.
        "cmp w1, w12\n"
        ".word 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".word 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        ".word 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".word 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".word 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".word 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"

        "blt 2b\n"

        "79:\n"
        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last 4 levels of depth, for which the LHS
        // and RHS data is already loaded.

        ".word 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        ".word 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        ".word 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".word 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        ".word 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".word 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        ".word 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".word 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        ".word 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".word 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".word 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".word 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
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

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        RUY_MAKE_ZERO(v8)
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec
        "add x5, x4, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"
        "add x5, x1, %x[row], lsl #2\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "add v15.4s, v15.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v15.4s\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "add v19.4s, v19.4s, v15.4s\n"
        "add v20.4s, v20.4s, v14.4s\n"
        "add v21.4s, v21.4s, v15.4s\n"
        "add v22.4s, v22.4s, v14.4s\n"
        "add v23.4s, v23.4s, v15.4s\n"
        "add v24.4s, v24.4s, v14.4s\n"
        "add v25.4s, v25.4s, v15.4s\n"
        "add v26.4s, v26.4s, v14.4s\n"
        "add v27.4s, v27.4s, v15.4s\n"
        "add v28.4s, v28.4s, v14.4s\n"
        "add v29.4s, v29.4s, v15.4s\n"
        "add v30.4s, v30.4s, v14.4s\n"
        "add v31.4s, v31.4s, v15.4s\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3], #16\n"
        "ld1 {v15.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[0]\n"
        "mls v18.4s, v10.4s, v14.s[1]\n"
        "mls v19.4s, v10.4s, v14.s[1]\n"
        "mls v20.4s, v10.4s, v14.s[2]\n"
        "mls v21.4s, v10.4s, v14.s[2]\n"
        "mls v22.4s, v10.4s, v14.s[3]\n"
        "mls v23.4s, v10.4s, v14.s[3]\n"
        "mls v24.4s, v10.4s, v15.s[0]\n"
        "mls v25.4s, v10.4s, v15.s[0]\n"
        "mls v26.4s, v10.4s, v15.s[1]\n"
        "mls v27.4s, v10.4s, v15.s[1]\n"
        "mls v28.4s, v10.4s, v15.s[2]\n"
        "mls v29.4s, v10.4s, v15.s[2]\n"
        "mls v30.4s, v10.4s, v15.s[3]\n"
        "mls v31.4s, v10.4s, v15.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2], #16\n"
        "ld1 {v12.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        "mul v12.4s, v12.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v12.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v12.4s\n"
        "sub v20.4s, v20.4s, v11.4s\n"
        "sub v21.4s, v21.4s, v12.4s\n"
        "sub v22.4s, v22.4s, v11.4s\n"
        "sub v23.4s, v23.4s, v12.4s\n"
        "sub v24.4s, v24.4s, v11.4s\n"
        "sub v25.4s, v25.4s, v12.4s\n"
        "sub v26.4s, v26.4s, v11.4s\n"
        "sub v27.4s, v27.4s, v12.4s\n"
        "sub v28.4s, v28.4s, v11.4s\n"
        "sub v29.4s, v29.4s, v12.4s\n"
        "sub v30.4s, v30.4s, v11.4s\n"
        "sub v31.4s, v31.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, %x[row], lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ldr q9, [x1]\n"
        "ldr q10, [x1, #16]\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_NEEDS_LEFT_SHIFT) "\n"
        "beq 403f\n"
        "smax v11.4s, v9.4s, v8.4s\n"
        "smax v12.4s, v10.4s, v8.4s\n"
        "sshl v16.4s, v16.4s, v11.4s\n"
        "sshl v17.4s, v17.4s, v12.4s\n"
        "sshl v18.4s, v18.4s, v11.4s\n"
        "sshl v19.4s, v19.4s, v12.4s\n"
        "sshl v20.4s, v20.4s, v11.4s\n"
        "sshl v21.4s, v21.4s, v12.4s\n"
        "sshl v22.4s, v22.4s, v11.4s\n"
        "sshl v23.4s, v23.4s, v12.4s\n"
        "sshl v24.4s, v24.4s, v11.4s\n"
        "sshl v25.4s, v25.4s, v12.4s\n"
        "sshl v26.4s, v26.4s, v11.4s\n"
        "sshl v27.4s, v27.4s, v12.4s\n"
        "sshl v28.4s, v28.4s, v11.4s\n"
        "sshl v29.4s, v29.4s, v12.4s\n"
        "sshl v30.4s, v30.4s, v11.4s\n"
        "sshl v31.4s, v31.4s, v12.4s\n"
        "403:\n"

        "ldr q14, [x4]\n" // multiplier_fixedpoint
        "ldr q15, [x4, #16]\n" // multiplier_fixedpoint

        "smin v11.4s, v9.4s, v8.4s\n"
        "smin v12.4s, v10.4s, v8.4s\n"

        // Apply the fixed-point part of the multiplier.
        "sqrdmulh v16.4s, v16.4s, v14.4s\n"
        "sqrdmulh v17.4s, v17.4s, v15.4s\n"
        "sqrdmulh v18.4s, v18.4s, v14.4s\n"
        "sqrdmulh v19.4s, v19.4s, v15.4s\n"
        "sqrdmulh v20.4s, v20.4s, v14.4s\n"
        "sqrdmulh v21.4s, v21.4s, v15.4s\n"
        "sqrdmulh v22.4s, v22.4s, v14.4s\n"
        "sqrdmulh v23.4s, v23.4s, v15.4s\n"
        "sqrdmulh v24.4s, v24.4s, v14.4s\n"
        "sqrdmulh v25.4s, v25.4s, v15.4s\n"
        "sqrdmulh v26.4s, v26.4s, v14.4s\n"
        "sqrdmulh v27.4s, v27.4s, v15.4s\n"
        "sqrdmulh v28.4s, v28.4s, v14.4s\n"
        "sqrdmulh v29.4s, v29.4s, v15.4s\n"
        "sqrdmulh v30.4s, v30.4s, v14.4s\n"
        "sqrdmulh v31.4s, v31.4s, v15.4s\n"

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
        "and v8.16b, v16.16b, v11.16b\n"
        "and v9.16b, v17.16b, v12.16b\n"
        "and v14.16b, v18.16b, v11.16b\n"
        "and v15.16b, v19.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v16.4s, v16.4s, v8.4s\n"
        "sqadd v17.4s, v17.4s, v9.4s\n"
        "sqadd v18.4s, v18.4s, v14.4s\n"
        "sqadd v19.4s, v19.4s, v15.4s\n"
        "and v8.16b, v20.16b, v11.16b\n"
        "and v9.16b, v21.16b, v12.16b\n"
        "and v14.16b, v22.16b, v11.16b\n"
        "and v15.16b, v23.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v20.4s, v20.4s, v8.4s\n"
        "sqadd v21.4s, v21.4s, v9.4s\n"
        "sqadd v22.4s, v22.4s, v14.4s\n"
        "sqadd v23.4s, v23.4s, v15.4s\n"
        "and v8.16b, v24.16b, v11.16b\n"
        "and v9.16b, v25.16b, v12.16b\n"
        "and v14.16b, v26.16b, v11.16b\n"
        "and v15.16b, v27.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v24.4s, v24.4s, v8.4s\n"
        "sqadd v25.4s, v25.4s, v9.4s\n"
        "sqadd v26.4s, v26.4s, v14.4s\n"
        "sqadd v27.4s, v27.4s, v15.4s\n"
        "and v8.16b, v28.16b, v11.16b\n"
        "and v9.16b, v29.16b, v12.16b\n"
        "and v14.16b, v30.16b, v11.16b\n"
        "and v15.16b, v31.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v28.4s, v28.4s, v8.4s\n"
        "sqadd v29.4s, v29.4s, v9.4s\n"
        "sqadd v30.4s, v30.4s, v14.4s\n"
        "sqadd v31.4s, v31.4s, v15.4s\n"
#endif
        // At this point we have reduced the problem of correctly implementing
        // rounding divide-by-power-of-two, to what the SRSHL instruction can
        // do.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"
        "srshl v18.4s, v18.4s, v11.4s\n"
        "srshl v19.4s, v19.4s, v12.4s\n"
        "srshl v20.4s, v20.4s, v11.4s\n"
        "srshl v21.4s, v21.4s, v12.4s\n"
        "srshl v22.4s, v22.4s, v11.4s\n"
        "srshl v23.4s, v23.4s, v12.4s\n"
        "srshl v24.4s, v24.4s, v11.4s\n"
        "srshl v25.4s, v25.4s, v12.4s\n"
        "srshl v26.4s, v26.4s, v11.4s\n"
        "srshl v27.4s, v27.4s, v12.4s\n"
        "srshl v28.4s, v28.4s, v11.4s\n"
        "srshl v29.4s, v29.4s, v12.4s\n"
        "srshl v30.4s, v30.4s, v11.4s\n"
        "srshl v31.4s, v31.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "add v16.8h, v16.8h, v14.8h\n"
        "add v17.8h, v17.8h, v14.8h\n"
        "add v18.8h, v18.8h, v14.8h\n"
        "add v19.8h, v19.8h, v14.8h\n"
        "add v20.8h, v20.8h, v14.8h\n"
        "add v21.8h, v21.8h, v14.8h\n"
        "add v22.8h, v22.8h, v14.8h\n"
        "add v23.8h, v23.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        "sqxtun2 v16.16b, v17.8h\n"
        "sqxtun v17.8b, v18.8h\n"
        "sqxtun2 v17.16b, v19.8h\n"
        "sqxtun v18.8b, v20.8h\n"
        "sqxtun2 v18.16b, v21.8h\n"
        "sqxtun v19.8b, v22.8h\n"
        "sqxtun2 v19.16b, v23.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        "umax v17.16b, v17.16b, v14.16b\n"
        "umax v18.16b, v18.16b, v14.16b\n"
        "umax v19.16b, v19.16b, v14.16b\n"

        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"
        "umin v17.16b, v17.16b, v15.16b\n"
        "umin v18.16b, v18.16b, v15.16b\n"
        "umin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "add v16.8h, v16.8h, v14.8h\n"
        "add v17.8h, v17.8h, v14.8h\n"
        "add v18.8h, v18.8h, v14.8h\n"
        "add v19.8h, v19.8h, v14.8h\n"
        "add v20.8h, v20.8h, v14.8h\n"
        "add v21.8h, v21.8h, v14.8h\n"
        "add v22.8h, v22.8h, v14.8h\n"
        "add v23.8h, v23.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"
        "sqxtn2 v16.16b, v17.8h\n"
        "sqxtn v17.8b, v18.8h\n"
        "sqxtn2 v17.16b, v19.8h\n"
        "sqxtn v18.8b, v20.8h\n"
        "sqxtn2 v18.16b, v21.8h\n"
        "sqxtn v19.8b, v22.8h\n"
        "sqxtn2 v19.16b, v23.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        "smax v17.16b, v17.16b, v14.16b\n"
        "smax v18.16b, v18.16b, v14.16b\n"
        "smax v19.16b, v19.16b, v14.16b\n"

        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"
        "smin v17.16b, v17.16b, v15.16b\n"
        "smin v18.16b, v18.16b, v15.16b\n"
        "smin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 130f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 131f\n"
        "130:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "131:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 141f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "150:\n"
        "mov w5, #0\n"
        "151:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 151b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 150b\n"
        "141:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"
        "saddw v20.4s, v20.4s, v14.4h\n"
        "saddw v21.4s, v21.4s, v14.4h\n"
        "saddw v22.4s, v22.4s, v14.4h\n"
        "saddw v23.4s, v23.4s, v14.4h\n"
        "saddw v24.4s, v24.4s, v14.4h\n"
        "saddw v25.4s, v25.4s, v14.4h\n"
        "saddw v26.4s, v26.4s, v14.4h\n"
        "saddw v27.4s, v27.4s, v14.4h\n"
        "saddw v28.4s, v28.4s, v14.4h\n"
        "saddw v29.4s, v29.4s, v14.4h\n"
        "saddw v30.4s, v30.4s, v14.4h\n"
        "saddw v31.4s, v31.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Load the clamp_min, clamp_max bounds
        "ldrsh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrsh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        "smax v18.8h, v18.8h, v14.8h\n"
        "smax v19.8h, v19.8h, v14.8h\n"
        "smax v20.8h, v20.8h, v14.8h\n"
        "smax v21.8h, v21.8h, v14.8h\n"
        "smax v22.8h, v22.8h, v14.8h\n"
        "smax v23.8h, v23.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"
        "smin v18.8h, v18.8h, v15.8h\n"
        "smin v19.8h, v19.8h, v15.8h\n"
        "smin v20.8h, v20.8h, v15.8h\n"
        "smin v21.8h, v21.8h, v15.8h\n"
        "smin v22.8h, v22.8h, v15.8h\n"
        "smin v23.8h, v23.8h, v15.8h\n"

        // Compute how much of the 8x8 block of destination 16bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 230f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"
        "b 231f\n"
        "230:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "231:\n"

        // Write our 16bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v17.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v18.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v19.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v20.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v21.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v22.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v23.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 241f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "250:\n"
        "mov w5, #0\n"
        "251:\n"
        "ldrsh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 251b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 250b\n"
        "241:\n"
        "add %[dst_ptr], %[dst_ptr], #16\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 8x8 block of destination 32it values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 330f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"

        // Write our 32bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v17.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v18.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v19.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v20.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v21.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v22.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v23.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)
        "st1 {v24.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v24)
        "st1 {v25.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v25)
        "st1 {v26.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v26)
        "st1 {v27.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v27)
        "st1 {v28.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v28)
        "st1 {v29.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v29)
        "st1 {v30.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v30)
        "st1 {v31.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v31)

        "b 331f\n"

        "330:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x4, %[dst_ptr]\n"
        "mov x3, x4\n"

        // Write our 32bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.4s, v17.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v18.4s, v19.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v20.4s, v21.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v22.4s, v23.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v24.4s, v25.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v26.4s, v27.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v28.4s, v29.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v30.4s, v31.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        "331:\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 341f\n"

        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "350:\n"
        "mov w5, #0\n"
        "351:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 351b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 350b\n"
        "341:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Similar to the above 8-bit dotprod kernel, but specialized for the case of
// RHS cols == 1.
// Relevant target CPUs for this kernel include ARM Cortex-A76,
// since these are 64-bit, out-of-order and with dotprod support.
void Kernel8bitNeonDotprodOutOfOrder1Col(const KernelParams8bit<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeonDotprod, optimized for out-of-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v15 are used to load int8 data from LHS and
  // RHS. At least v0 and v1 are used to load a 8x4 block of LHS, and v2 and
  // v3 are used to load a 4x8 block of RHS, like this:
  //
  //                            int8 RHS 4x1 block
  //                           /-------\
  //                           |v2.b[0]|
  //                           |  ...  |
  //                           |v2.b[3]|
  //                           \-------/
  //    int8 LHS 8x4 block
  //  /---------------------\  /--------\
  //  |v0.b[0]  ... v0.b[3] |  |v16.s[0]|
  //  |  ...          ...   |  |  ...   |
  //  |v0.b[12] ... v0.b[15]|  |v16.s[3]|
  //  |v1.b[0]  ... v1.b[3] |  |v17.s[0]|
  //  |  ...         ...    |  |  ...   |
  //  |v1.b[12] ... v1.b[15]|  |v17.s[3]|
  //  \---------------------/  \--------/
  //                           int32 accumulators 8x1 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for loading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.8b}, [%[rhs_ptr]]\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        // Ordinary kernel inner loop (over depth), the simpler loop that the
        // above was an equivalent 4x-partially-unrolled version of.

        // Reminder - w1 is how many levels of depth we have already loaded
        // data for, w12 is the total depth.
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"

        // Because of the data that we have already loaded, we can start the
        // loop body right away with some multiply-adds.
        // Each iteration of this loop advances by 4 levels of depth.
        "add w1, w1, #4\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        ".word 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        // Loop termination condition.
        "cmp w1, w12\n"
        "ld1 {v2.8b}, [%[rhs_ptr]]\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"

        "blt 2b\n"

        "79:\n"
        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last 4 levels of depth, for which the LHS
        // and RHS data is already loaded.

        ".word 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
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

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        RUY_MAKE_ZERO(v8)
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec
        "add x5, x4, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"
        "add x5, x1, %x[row], lsl #2\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.8b}, [%[rhs_ptr]]\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "add v15.4s, v15.4s, v9.4s\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v15.4s\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ld1 {v14.4s}, [x3], #16\n"
        "ld1 {v15.4s}, [x3]\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[0]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        // Load 4 lhs_sums values.
        "ld1 {v11.4s}, [x2], #16\n"
        "ld1 {v12.4s}, [x2]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        "mul v12.4s, v12.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, %x[row], lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ldr q9, [x1]\n"
        "ldr q10, [x1, #16]\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_NEEDS_LEFT_SHIFT) "\n"
        "beq 403f\n"
        "smax v11.4s, v9.4s, v8.4s\n"
        "smax v12.4s, v10.4s, v8.4s\n"
        "sshl v16.4s, v16.4s, v11.4s\n"
        "sshl v17.4s, v17.4s, v12.4s\n"
        "403:\n"

        "ldr q14, [x4]\n" // multiplier_fixedpoint
        "ldr q15, [x4, #16]\n" // multiplier_fixedpoint

        "smin v11.4s, v9.4s, v8.4s\n"
        "smin v12.4s, v10.4s, v8.4s\n"

        // Apply the fixed-point part of the multiplier.
        "sqrdmulh v16.4s, v16.4s, v14.4s\n"
        "sqrdmulh v17.4s, v17.4s, v15.4s\n"

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
        "and v8.16b, v16.16b, v11.16b\n"
        "and v9.16b, v17.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sqadd v16.4s, v16.4s, v8.4s\n"
        "sqadd v17.4s, v17.4s, v9.4s\n"

#endif
        // At this point we have reduced the problem of correctly implementing
        // rounding divide-by-power-of-two, to what the SRSHL instruction can
        // do.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        // All data in v16 at this point.

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "add v16.8h, v16.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8, leaving all data in the
        // lower half of v16.
        "sqxtun v16.8b, v16.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"

        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"

        // Compute how much of the 8x1 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x1, there are some 8x1 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"

        // Test if w1==8, i.e. if all of the 8x1 block fits.
        "cmp w1, w3\n"
        // Yes, all of the 8x1 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"


        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "add v16.8h, v16.8h, v14.8h\n"

        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"

        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"

        // Compute how much of the 8x1 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"

        // Test if w1==8, i.e. if all of the 8x1 block fits.
        "cmp w1, w3\n"
        // Yes, all of the 8x1 block fits, go to fast path.
        "beq 130f\n"
        // Not all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 131f\n"
        "130:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "131:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 141f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "150:\n"
        "mov w5, #0\n"
        "151:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 151b\n"
        "141:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"

        // Load the clamp_min, clamp_max bounds
        "ldrsh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrsh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"

        // Compute how much of the 8x1 block of destination 16bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x1 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"

        // Test if w1==8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        // Yes, all of the 8x1 block fits, go to fast path.
        "beq 230f\n"
        // Not all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"
        "b 231f\n"
        "230:\n"
        // Yes, all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "231:\n"

        // Write our 16bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // If all of the 8x1 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 241f\n"
        // Not all of the 8x1 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "250:\n"
        "mov w5, #0\n"
        "251:\n"
        "ldrsh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 251b\n"
        "241:\n"
        "add %[dst_ptr], %[dst_ptr], #16\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 8x1 block of destination 32 bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x1, there are some 8x1 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x1 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        // Yes, all of the 8x1 block fits, go to fast path.
        "beq 330f\n"
        // Not all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"

        // Write our 32bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v17.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)

        "b 331f\n"

        "330:\n"
        // Yes, all of the 8x1 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x4, %[dst_ptr]\n"
        "mov x3, x4\n"

        // Write our 32bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.4s, v17.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)

        "331:\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 341f\n"

        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "350:\n"
        "mov w5, #0\n"
        "351:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 351b\n"
        "341:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 4.
        "mov w1, #4\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17");
}

// Variant of the above Kernel8bitNeonDotprodOutOfOrder, tuned for in-order
// CPUs. Specifically here, the relevant in-order CPUs are ARM Cortex-A55r1,
// since these are 64-bit and support dotprod.
//
// While this kernel does not have a direct equivalent in gemmlowp, it was
// developed based on insights that David Mansell at ARM shared with their
// contribution of gemmlowp kernels tuned for Cortex-A55r1, with very helpful
// comments. Specifically, see this comment about tuning for Cortex-A55r1:
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L4412
void Kernel8bitNeonDotprodInOrder(const KernelParams8bit<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeonDotprod, optimized for in-order cores)");

  CheckOffsetsInKernelParams8bit(params);

  const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  const std::int8_t* lhs_ptr = lhs_col_ptr;
  const std::int8_t* rhs_ptr = rhs_col_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  void* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are int32 accumulators.
  // During accumulation, v0 -- v3 are used to load int8 data from LHS and
  // RHS.
  //
  //                                      int8 RHS 4x8 block
  //                           /-----------------------------------------\
  //                           |v2.b[0] ... v2.b[12] v3.b[0] ... v3.b[12]|
  //                           |  ...                              ...   |
  //                           |v2.b[3] ... v2.b[15] v3.b[3] ... v3.b[15]|
  //                           \-----------------------------------------/
  //    int8 LHS 8x4 block
  //  /---------------------\  /-----------------------------------------\
  //  |v0.b[0]  ... v0.b[3] |  |v16.s[0]           ...           v30.s[0]|
  //  |  ...          ...   |  |  ...                              ...   |
  //  |v0.b[12] ... v0.b[15]|  |v16.s[3]           ...           v30.s[3]|
  //  |v1.b[0]  ... v1.b[3] |  |v17.s[0]           ...           v31.s[0]|
  //  |  ...         ...    |  |  ...                              ...   |
  //  |v1.b[12] ... v1.b[15]|  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                  int32 accumulators 8x8 block
  //
  // There is no RUY_OPT_MAX_STREAMING 4x-unrolled part in this kernel because
  // we did not observe a benefit of such partial unrolling on in-order CPUs.
  //
  // v4 -- v7 are unused, and v8 -- v15 are used for loading parameters used for
  // the post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        RUY_MAKE_ZERO(v16)
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        RUY_MAKE_ZERO(v17)
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        RUY_MAKE_ZERO(v18)
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        RUY_MAKE_ZERO(v19)
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        RUY_MAKE_ZERO(v20)
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        RUY_MAKE_ZERO(v21)
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        RUY_MAKE_ZERO(v22)
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        // Perform the first few multiply-adds on the data that we have already
        // loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        RUY_MAKE_ZERO(v28)
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        RUY_MAKE_ZERO(v29)
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        RUY_MAKE_ZERO(v30)
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        RUY_MAKE_ZERO(v31)


        "1:\n"

        "add x5, %[lhs_ptr], x12, lsl #3\n"
        "sub x5, x5, #32\n"
        "cmp %[lhs_ptr], x5\n"

        "beq 79f\n"

        // Main accumulation loop
        "2:\n"
        ".word 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        "ldr x1, [%[lhs_ptr], #8]\n"
        ".word 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        "ldr x3, [%[rhs_ptr], #8]\n"
        ".word 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        "ldr x4, [%[rhs_ptr], #24]\n"
        ".word 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        "ldr d0, [%[lhs_ptr], #0]\n"
        ".word 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        "ins v0.d[1], x1\n"
        ".word 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        "ldr x2, [%[lhs_ptr], #24]\n"
        ".word 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        "add %[lhs_ptr], %[lhs_ptr], #32\n"
        ".word 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        "ldr d2, [%[rhs_ptr], #0]\n"
        ".word 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        "ins v2.d[1], x3\n"
        ".word 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        "cmp %[lhs_ptr], x5\n"
        ".word 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"
        ".word 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"
        "ldr d3, [%[rhs_ptr], #-16]\n"
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        "ldr d1, [%[lhs_ptr], #-16]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        "ins v3.d[1], x4\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        "ins v1.d[1], x2\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        "blt 2b\n"

        // Last accumulation steps, nothing left to load.
        "79:\n"
        ".word 0x4f83e018  // sdot v24.4s, v0.16b, v3.4b[0]\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        ".word 0x4fa3e01a  // sdot v26.4s, v0.16b, v3.4b[1]\n"
        "cmp %w[row], w7\n"  // Have we finished the last row?
        ".word 0x4f83e81c  // sdot v28.4s, v0.16b, v3.4b[2]\n"
        ".word 0x4fa3e81e  // sdot v30.4s, v0.16b, v3.4b[3]\n"
        ".word 0x4f82e031  // sdot v17.4s, v1.16b, v2.4b[0]\n"
        ".word 0x4fa2e033  // sdot v19.4s, v1.16b, v2.4b[1]\n"
        ".word 0x4f82e835  // sdot v21.4s, v1.16b, v2.4b[2]\n"
        ".word 0x4fa2e837  // sdot v23.4s, v1.16b, v2.4b[3]\n"
        ".word 0x4f83e039  // sdot v25.4s, v1.16b, v3.4b[0]\n"
        ".word 0x4fa3e03b  // sdot v27.4s, v1.16b, v3.4b[1]\n"
        ".word 0x4f83e83d  // sdot v29.4s, v1.16b, v3.4b[2]\n"
        ".word 0x4fa3e83f  // sdot v31.4s, v1.16b, v3.4b[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
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

        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        // Load some parameters needed for the end work on current block.
        RUY_MAKE_ZERO(v8)
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"
        "ldr w4, [%[params], #" RUY_STR(RUY_OFFSET_DST_ZERO_POINT) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_PROD_ZP_DEPTH) "]\n"
        "ins v13.h[4], w4\n" // dst_zero_point
        "ldr x4, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_FIXEDPOINT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "dup v9.4s, w3\n"   // create prod_zp_depth_vec
        "add x5, x4, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "csel x4, x4, x5, eq\n"

        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"
        "add x5, x1, %x[row], lsl #2\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.2s}, [x1], #8\n"
        "ldr x5, [x1], #8\n"
        "ins v14.d[1], x5\n"
        "ld1 {v15.2s}, [x1], #8\n"
        "ldr x5, [x1], #8\n"
        "ins v15.d[1], x5\n"

        // Add to the bias values the product (depth * lhs_zero_point * rhs_zero_point),
        // See the term NZ1Z2 in equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "add v14.4s, v14.4s, v9.4s\n"
        "add v15.4s, v15.4s, v9.4s\n"
        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "add v16.4s, v16.4s, v14.4s\n"
        "add v17.4s, v17.4s, v15.4s\n"
        "add v18.4s, v18.4s, v14.4s\n"
        "add v19.4s, v19.4s, v15.4s\n"
        "add v20.4s, v20.4s, v14.4s\n"
        "add v21.4s, v21.4s, v15.4s\n"
        "add v22.4s, v22.4s, v14.4s\n"
        "add v23.4s, v23.4s, v15.4s\n"
        "add v24.4s, v24.4s, v14.4s\n"
        "add v25.4s, v25.4s, v15.4s\n"
        "add v26.4s, v26.4s, v14.4s\n"
        "add v27.4s, v27.4s, v15.4s\n"
        "add v28.4s, v28.4s, v14.4s\n"
        "add v29.4s, v29.4s, v15.4s\n"
        "add v30.4s, v30.4s, v14.4s\n"
        "add v31.4s, v31.4s, v15.4s\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_RHS_SUMS) "\n"
        "beq 401f\n"
        "ldr x3, [%[params], #" RUY_STR(RUY_OFFSET_RHS_SUMS) "]\n"
        "add x3, x3, %x[col], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_ZERO_POINT) "]\n"
        "dup v10.4s, w5\n"  // create lhs_zero_point_vec
        // Load 8 rhs_sums values.
        "ld1 {v14.2s}, [x3], #8\n"
        "ldr x7, [x3], #8\n"
        "ld1 {v15.2s}, [x3], #8\n"
        "ins v14.d[1], x7\n"
        "ldr x7, [x3], #8\n"
        "ins v15.d[1], x7\n"
        // Subtract rhs_sums * lhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "mls v16.4s, v10.4s, v14.s[0]\n"
        "mls v17.4s, v10.4s, v14.s[0]\n"
        "mls v18.4s, v10.4s, v14.s[1]\n"
        "mls v19.4s, v10.4s, v14.s[1]\n"
        "mls v20.4s, v10.4s, v14.s[2]\n"
        "mls v21.4s, v10.4s, v14.s[2]\n"
        "mls v22.4s, v10.4s, v14.s[3]\n"
        "mls v23.4s, v10.4s, v14.s[3]\n"
        "mls v24.4s, v10.4s, v15.s[0]\n"
        "mls v25.4s, v10.4s, v15.s[0]\n"
        "mls v26.4s, v10.4s, v15.s[1]\n"
        "mls v27.4s, v10.4s, v15.s[1]\n"
        "mls v28.4s, v10.4s, v15.s[2]\n"
        "mls v29.4s, v10.4s, v15.s[2]\n"
        "mls v30.4s, v10.4s, v15.s[3]\n"
        "mls v31.4s, v10.4s, v15.s[3]\n"
        "401:\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_LHS_SUMS) "\n"
        "beq 402f\n"
        "ldr x2, [%[params], #" RUY_STR(RUY_OFFSET_LHS_SUMS) "]\n"
        "add x2, x2, %x[row], lsl #2\n"
        "ldr w5, [%[params], #" RUY_STR(RUY_OFFSET_RHS_ZERO_POINT) "]\n"
        "ins v13.s[1], w5\n" // rhs_zero_point
        // Load 8 lhs_sums values.
        "ld1 {v11.2s}, [x2], #8\n"
        "ldr x6, [x2], #8\n"
        "ins v11.d[1], x6\n"
        "ld1 {v12.2s}, [x2], #8\n"
        "ldr x6, [x2], #8\n"
        "ins v12.d[1], x6\n"
        // Compute lhs_sums * rhs_zero_point.
        "mul v11.4s, v11.4s, v13.s[1]\n"
        "mul v12.4s, v12.4s, v13.s[1]\n"
        // Subtract lhs_sums * rhs_zero_point, per
        // equation (7) in https://arxiv.org/pdf/1712.05877.pdf
        "sub v16.4s, v16.4s, v11.4s\n"
        "sub v17.4s, v17.4s, v12.4s\n"
        "sub v18.4s, v18.4s, v11.4s\n"
        "sub v19.4s, v19.4s, v12.4s\n"
        "sub v20.4s, v20.4s, v11.4s\n"
        "sub v21.4s, v21.4s, v12.4s\n"
        "sub v22.4s, v22.4s, v11.4s\n"
        "sub v23.4s, v23.4s, v12.4s\n"
        "sub v24.4s, v24.4s, v11.4s\n"
        "sub v25.4s, v25.4s, v12.4s\n"
        "sub v26.4s, v26.4s, v11.4s\n"
        "sub v27.4s, v27.4s, v12.4s\n"
        "sub v28.4s, v28.4s, v11.4s\n"
        "sub v29.4s, v29.4s, v12.4s\n"
        "sub v30.4s, v30.4s, v11.4s\n"
        "sub v31.4s, v31.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT32) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT32) "f\n"

        "402:\n"

        // At this point we have computed the final int32 values. Now we
        // start down-quantizing them to obtain the final 8bit values from them.

        // As part of this down-quantization, our int32 values will be
        // multiplied by a multiplier that has a fixed-point component and an
        // exponent component.

        //Load the exponent part of the multiplier.
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_MULTIPLIER_EXPONENT) "]\n"
        "ldrb w6, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "tst w6, #" RUY_STR(RUY_ASM_FLAG_HAS_PERCHANNEL) "\n"
        "add x5, x1, %x[row], lsl #2\n"
        "csel x1, x1, x5, eq\n"

        "ldr q9, [x1]\n"
        "ldr q10, [x1, #16]\n"

        "tst w6, #" RUY_STR(RUY_ASM_FLAG_NEEDS_LEFT_SHIFT) "\n"
        "beq 403f\n"
        "smax v11.4s, v9.4s, v8.4s\n"
        "smax v12.4s, v10.4s, v8.4s\n"
        "sshl v16.4s, v16.4s, v11.4s\n"
        "sshl v17.4s, v17.4s, v12.4s\n"
        "sshl v18.4s, v18.4s, v11.4s\n"
        "sshl v19.4s, v19.4s, v12.4s\n"
        "sshl v20.4s, v20.4s, v11.4s\n"
        "sshl v21.4s, v21.4s, v12.4s\n"
        "sshl v22.4s, v22.4s, v11.4s\n"
        "sshl v23.4s, v23.4s, v12.4s\n"
        "sshl v24.4s, v24.4s, v11.4s\n"
        "sshl v25.4s, v25.4s, v12.4s\n"
        "sshl v26.4s, v26.4s, v11.4s\n"
        "sshl v27.4s, v27.4s, v12.4s\n"
        "sshl v28.4s, v28.4s, v11.4s\n"
        "sshl v29.4s, v29.4s, v12.4s\n"
        "sshl v30.4s, v30.4s, v11.4s\n"
        "sshl v31.4s, v31.4s, v12.4s\n"
        "403:\n"

        "ldr q14, [x4]\n" // multiplier_fixedpoint
        "ldr q15, [x4, #16]\n" // multiplier_fixedpoint

        "smin v11.4s, v9.4s, v8.4s\n"
        "smin v12.4s, v10.4s, v8.4s\n"

        // Apply the fixed-point part of the multiplier.
        //
        // ... and, interleaved into that:
        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.8b}, [%[lhs_ptr]], #8\n"
        "sqrdmulh v16.4s, v16.4s, v14.4s\n"
        "ldr x1, [%[lhs_ptr]], #8\n"
        "sqrdmulh v17.4s, v17.4s, v15.4s\n"
        "ld1 {v1.8b}, [%[lhs_ptr]], #8\n"
        "sqrdmulh v18.4s, v18.4s, v14.4s\n"
        "ldr x2, [%[lhs_ptr]], #8\n"
        "sqrdmulh v19.4s, v19.4s, v15.4s\n"
        "ld1 {v2.8b}, [%[rhs_ptr]], #8\n"
        "sqrdmulh v20.4s, v20.4s, v14.4s\n"
        "ldr x5, [%[rhs_ptr]], #8\n"
        "sqrdmulh v21.4s, v21.4s, v15.4s\n"
        "ld1 {v3.8b}, [%[rhs_ptr]], #8\n"
        "sqrdmulh v22.4s, v22.4s, v14.4s\n"
        "ldr x6, [%[rhs_ptr]], #8\n"
        "sqrdmulh v23.4s, v23.4s, v15.4s\n"
        "sqrdmulh v24.4s, v24.4s, v14.4s\n"
        "sqrdmulh v25.4s, v25.4s, v15.4s\n"
        "sqrdmulh v26.4s, v26.4s, v14.4s\n"
        "sqrdmulh v27.4s, v27.4s, v15.4s\n"
        "sqrdmulh v28.4s, v28.4s, v14.4s\n"
        "sqrdmulh v29.4s, v29.4s, v15.4s\n"
        "sqrdmulh v30.4s, v30.4s, v14.4s\n"
        "sqrdmulh v31.4s, v31.4s, v15.4s\n"

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
        "and v8.16b, v16.16b, v11.16b\n"
        "and v9.16b, v17.16b, v12.16b\n"
        "and v14.16b, v18.16b, v11.16b\n"
        "and v15.16b, v19.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v16.4s, v16.4s, v8.4s\n"
        "sqadd v17.4s, v17.4s, v9.4s\n"
        "sqadd v18.4s, v18.4s, v14.4s\n"
        "sqadd v19.4s, v19.4s, v15.4s\n"
        "and v8.16b, v20.16b, v11.16b\n"
        "and v9.16b, v21.16b, v12.16b\n"
        "and v14.16b, v22.16b, v11.16b\n"
        "and v15.16b, v23.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v20.4s, v20.4s, v8.4s\n"
        "sqadd v21.4s, v21.4s, v9.4s\n"
        "sqadd v22.4s, v22.4s, v14.4s\n"
        "sqadd v23.4s, v23.4s, v15.4s\n"
        "and v8.16b, v24.16b, v11.16b\n"
        "and v9.16b, v25.16b, v12.16b\n"
        "and v14.16b, v26.16b, v11.16b\n"
        "and v15.16b, v27.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v24.4s, v24.4s, v8.4s\n"
        "sqadd v25.4s, v25.4s, v9.4s\n"
        "sqadd v26.4s, v26.4s, v14.4s\n"
        "sqadd v27.4s, v27.4s, v15.4s\n"
        "and v8.16b, v28.16b, v11.16b\n"
        "and v9.16b, v29.16b, v12.16b\n"
        "and v14.16b, v30.16b, v11.16b\n"
        "and v15.16b, v31.16b, v12.16b\n"
        "sshr v8.4s, v8.4s, #31\n"
        "sshr v9.4s, v9.4s, #31\n"
        "sshr v14.4s, v14.4s, #31\n"
        "sshr v15.4s, v15.4s, #31\n"
        "sqadd v28.4s, v28.4s, v8.4s\n"
        "sqadd v29.4s, v29.4s, v9.4s\n"
        "sqadd v30.4s, v30.4s, v14.4s\n"
        "sqadd v31.4s, v31.4s, v15.4s\n"
#endif
        // At this point we have reduced the problem of correctly implementing
        // rounding divide-by-power-of-two, to what the SRSHL instruction can
        // do.
        "srshl v16.4s, v16.4s, v11.4s\n"
        "srshl v17.4s, v17.4s, v12.4s\n"
        "srshl v18.4s, v18.4s, v11.4s\n"
        "srshl v19.4s, v19.4s, v12.4s\n"
        "srshl v20.4s, v20.4s, v11.4s\n"
        "srshl v21.4s, v21.4s, v12.4s\n"
        "srshl v22.4s, v22.4s, v11.4s\n"
        "srshl v23.4s, v23.4s, v12.4s\n"
        "srshl v24.4s, v24.4s, v11.4s\n"
        "srshl v25.4s, v25.4s, v12.4s\n"
        "srshl v26.4s, v26.4s, v11.4s\n"
        "srshl v27.4s, v27.4s, v12.4s\n"
        "ins v0.d[1], x1\n"
        "srshl v28.4s, v28.4s, v11.4s\n"
        "ins v1.d[1], x2\n"
        "srshl v29.4s, v29.4s, v12.4s\n"
        "ins v2.d[1], x5\n"
        "srshl v30.4s, v30.4s, v11.4s\n"
        "ins v3.d[1], x6\n"
        "srshl v31.4s, v31.4s, v12.4s\n"

        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT16) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT16) "f\n"
        "cmp %w[dst_type_id], #" RUY_STR(RUY_ASM_TYPE_ID_INT8) "\n"
        "beq " RUY_STR(RUY_ASM_LABEL_STORE_INT8) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_UINT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // Destination zero_point
        "dup v14.8h, v13.h[4]\n"
        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "add v16.8h, v16.8h, v14.8h\n"
        "add v17.8h, v17.8h, v14.8h\n"
        "add v18.8h, v18.8h, v14.8h\n"
        "add v19.8h, v19.8h, v14.8h\n"
        "add v20.8h, v20.8h, v14.8h\n"
        "add v21.8h, v21.8h, v14.8h\n"
        "add v22.8h, v22.8h, v14.8h\n"
        "add v23.8h, v23.8h, v14.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        // Cast-and-saturate from int16 to uint8
        "sqxtun v16.8b, v16.8h\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "sqxtun2 v16.16b, v17.8h\n"
        "sqxtun v17.8b, v18.8h\n"
        "sqxtun2 v17.16b, v19.8h\n"
        "sqxtun v18.8b, v20.8h\n"
        "sqxtun2 v18.16b, v21.8h\n"
        "sqxtun v19.8b, v22.8h\n"
        "sqxtun2 v19.16b, v23.8h\n"

        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        // Apply the clamp_min bound
        "umax v16.16b, v16.16b, v14.16b\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "umax v17.16b, v17.16b, v14.16b\n"
        "mov w3, #8\n"
        "umax v18.16b, v18.16b, v14.16b\n"
        "cmp w1, #8\n"
        "umax v19.16b, v19.16b, v14.16b\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        // Apply the clamp_max bound
        "umin v16.16b, v16.16b, v15.16b\n"
        "cmp w2, #8\n"
        "umin v17.16b, v17.16b, v15.16b\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"
        "umin v18.16b, v18.16b, v15.16b\n"
        "umin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"

        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT8) ":\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // Destination zero_point
        "dup v14.8h, v13.h[4]\n"
        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Add the destination zero point
        "add v16.8h, v16.8h, v14.8h\n"
        "add v17.8h, v17.8h, v14.8h\n"
        "add v18.8h, v18.8h, v14.8h\n"
        "add v19.8h, v19.8h, v14.8h\n"
        "add v20.8h, v20.8h, v14.8h\n"
        "add v21.8h, v21.8h, v14.8h\n"
        "add v22.8h, v22.8h, v14.8h\n"
        "add v23.8h, v23.8h, v14.8h\n"

        // Load the clamp_min, clamp_max bounds
        "ldrb w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        // Cast-and-saturate from int16 to uint8
        "sqxtn v16.8b, v16.8h\n"
        "ldrb w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "sqxtn2 v16.16b, v17.8h\n"
        "sqxtn v17.8b, v18.8h\n"
        "sqxtn2 v17.16b, v19.8h\n"
        "sqxtn v18.8b, v20.8h\n"
        "sqxtn2 v18.16b, v21.8h\n"
        "sqxtn v19.8b, v22.8h\n"
        "sqxtn2 v19.16b, v23.8h\n"

        "dup v14.16b, w2\n"  // clamp_min
        "dup v15.16b, w3\n"  // clamp_max

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        // Apply the clamp_min bound
        "smax v16.16b, v16.16b, v14.16b\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "smax v17.16b, v17.16b, v14.16b\n"
        "mov w3, #8\n"
        "smax v18.16b, v18.16b, v14.16b\n"
        "cmp w1, #8\n"
        "smax v19.16b, v19.16b, v14.16b\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        // Apply the clamp_max bound
        "smin v16.16b, v16.16b, v15.16b\n"
        "cmp w2, #8\n"
        "smin v17.16b, v17.16b, v15.16b\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"
        "smin v18.16b, v18.16b, v15.16b\n"
        "smin v19.16b, v19.16b, v15.16b\n"

        // Make it so that all of the final 8bit values are stored in the
        // first 64bits of 128bit NEON registers, so they can be stored
        // by 64bit st1 store instructions with byte alignment.
        "dup d20, v16.d[1]\n"
        "dup d21, v17.d[1]\n"
        "dup d22, v18.d[1]\n"
        "dup d23, v19.d[1]\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 130f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #8\n"
        "b 131f\n"
        "130:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "131:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v20.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        "st1 {v17.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        "st1 {v21.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v18.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v22.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        "st1 {v19.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"
        "st1 {v23.8b}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 141f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "150:\n"
        "mov w5, #0\n"
        "151:\n"
        "ldrb w7, [x3, w5, uxtw]\n"
        "strb w7, [x4, w5, uxtw]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 151b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #8\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 150b\n"
        "141:\n"
        "add %[dst_ptr], %[dst_ptr], #8\n"

        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT16) ":\n"

        // Add the destination zero point
        "dup v14.8h, v13.h[4]\n"
        "saddw v16.4s, v16.4s, v14.4h\n"
        "saddw v17.4s, v17.4s, v14.4h\n"
        "saddw v18.4s, v18.4s, v14.4h\n"
        "saddw v19.4s, v19.4s, v14.4h\n"
        "saddw v20.4s, v20.4s, v14.4h\n"
        "saddw v21.4s, v21.4s, v14.4h\n"
        "saddw v22.4s, v22.4s, v14.4h\n"
        "saddw v23.4s, v23.4s, v14.4h\n"
        "saddw v24.4s, v24.4s, v14.4h\n"
        "saddw v25.4s, v25.4s, v14.4h\n"
        "saddw v26.4s, v26.4s, v14.4h\n"
        "saddw v27.4s, v27.4s, v14.4h\n"
        "saddw v28.4s, v28.4s, v14.4h\n"
        "saddw v29.4s, v29.4s, v14.4h\n"
        "saddw v30.4s, v30.4s, v14.4h\n"
        "saddw v31.4s, v31.4s, v14.4h\n"

        // Cast-and-saturate from int32 to int16
        "sqxtn v16.4h, v16.4s\n"
        "sqxtn2 v16.8h, v17.4s\n"
        "sqxtn v17.4h, v18.4s\n"
        "sqxtn2 v17.8h, v19.4s\n"
        "sqxtn v18.4h, v20.4s\n"
        "sqxtn2 v18.8h, v21.4s\n"
        "sqxtn v19.4h, v22.4s\n"
        "sqxtn2 v19.8h, v23.4s\n"
        "sqxtn v20.4h, v24.4s\n"
        "sqxtn2 v20.8h, v25.4s\n"
        "sqxtn v21.4h, v26.4s\n"
        "sqxtn2 v21.8h, v27.4s\n"
        "sqxtn v22.4h, v28.4s\n"
        "sqxtn2 v22.8h, v29.4s\n"
        "sqxtn v23.4h, v30.4s\n"
        "sqxtn2 v23.8h, v31.4s\n"

        // At this point, v24 -- v31 aren't used anymore for the current block,
        // so we can start clearing these accumulators for the next block
        // (next iteration of the main loop).
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // Load the clamp_min, clamp_max bounds
        "ldrsh w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldrsh w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.8h, w2\n"  // clamp_min
        "dup v15.8h, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "smax v16.8h, v16.8h, v14.8h\n"
        "smax v17.8h, v17.8h, v14.8h\n"
        "smax v18.8h, v18.8h, v14.8h\n"
        "smax v19.8h, v19.8h, v14.8h\n"
        "smax v20.8h, v20.8h, v14.8h\n"
        "smax v21.8h, v21.8h, v14.8h\n"
        "smax v22.8h, v22.8h, v14.8h\n"
        "smax v23.8h, v23.8h, v14.8h\n"
        // Apply the clamp_max bound
        "smin v16.8h, v16.8h, v15.8h\n"
        "smin v17.8h, v17.8h, v15.8h\n"
        "smin v18.8h, v18.8h, v15.8h\n"
        "smin v19.8h, v19.8h, v15.8h\n"
        "smin v20.8h, v20.8h, v15.8h\n"
        "smin v21.8h, v21.8h, v15.8h\n"
        "smin v22.8h, v22.8h, v15.8h\n"
        "smin v23.8h, v23.8h, v15.8h\n"

        // Compute how much of the 8x8 block of destination 16bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 230f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"
        "b 231f\n"
        "230:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "231:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v17.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v18.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v19.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v20.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v21.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v22.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v23.8h}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 241f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "250:\n"
        "mov w5, #0\n"
        "251:\n"
        "ldrsh w7, [x3, x5, lsl #1]\n"
        "strh w7, [x4, x5, lsl #1]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 251b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #16\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 250b\n"
        "241:\n"
        "add %[dst_ptr], %[dst_ptr], #16\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        "b " RUY_STR(RUY_ASM_LABEL_AFTER_STORE) "f\n"

        RUY_STR(RUY_ASM_LABEL_STORE_INT32) ":\n"

        "ld1 {v0.8b}, [%[lhs_ptr]], #8\n"
        "ldr x1, [%[lhs_ptr]], #8\n"
        "ld1 {v1.8b}, [%[lhs_ptr]], #8\n"
        "ldr x2, [%[lhs_ptr]], #8\n"
        "ld1 {v2.8b}, [%[rhs_ptr]], #8\n"
        "ldr x5, [%[rhs_ptr]], #8\n"
        "ld1 {v3.8b}, [%[rhs_ptr]], #8\n"
        "ldr x6, [%[rhs_ptr]], #8\n"
        "ins v0.d[1], x1\n"
        "ins v1.d[1], x2\n"
        "ins v2.d[1], x5\n"
        "ins v3.d[1], x6\n"

        // Since the store type is the same as the accum type, no need for
        // downcast. There's also no need for clamp by min/max.

        // Compute how much of the 8x8 block of destination 32it values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 330f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #16\n"

        // Write our 32bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v16)
        "st1 {v17.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v17)
        "st1 {v18.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v18)
        "st1 {v19.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v19)
        "st1 {v20.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v20)
        "st1 {v21.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v21)
        "st1 {v22.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v22)
        "st1 {v23.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v23)
        "st1 {v24.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v24)
        "st1 {v25.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v25)
        "st1 {v26.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v26)
        "st1 {v27.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v27)
        "st1 {v28.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v28)
        "st1 {v29.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v29)
        "st1 {v30.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v30)
        "st1 {v31.4s}, [x3], x4\n"
        RUY_MAKE_ZERO(v31)

        "b 331f\n"

        "330:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x4, %[dst_ptr]\n"
        "mov x3, x4\n"

        // Write our 32bit values to the destination described by
        // (x3 address, x4 stride).
        "st1 {v16.4s, v17.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v18.4s, v19.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v20.4s, v21.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v22.4s, v23.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v24.4s, v25.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v26.4s, v27.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v28.4s, v29.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "add x4, x4, x11\n"
        "mov x3, x4\n"
        "st1 {v30.4s, v31.4s}, [x3], #32\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        "331:\n"

        // For the next block: perform the first few multiply-adds on the data
        // that we have already loaded.
        ".word 0x4f82e010  // sdot v16.4s, v0.16b, v2.4b[0]\n"
        ".word 0x4fa2e012  // sdot v18.4s, v0.16b, v2.4b[1]\n"
        ".word 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".word 0x4fa2e816  // sdot v22.4s, v0.16b, v2.4b[3]\n"

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 341f\n"

        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "350:\n"
        "mov w5, #0\n"
        "351:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 351b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 350b\n"
        "341:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        RUY_STR(RUY_ASM_LABEL_AFTER_STORE) ":\n"

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf),
          [dst_type_id] "r"(params.dst_type_id)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
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

#define RUY_OFFSET_LHS_BASE_PTR 0
#define RUY_OFFSET_RHS_BASE_PTR 8
#define RUY_OFFSET_DST_BASE_PTR 16
#define RUY_OFFSET_BIAS 24
#define RUY_OFFSET_START_ROW 32
#define RUY_OFFSET_START_COL 36
#define RUY_OFFSET_LAST_ROW 40
#define RUY_OFFSET_LAST_COL 44
#define RUY_OFFSET_LHS_STRIDE 56
#define RUY_OFFSET_RHS_STRIDE 60
#define RUY_OFFSET_DST_STRIDE 64
#define RUY_OFFSET_DEPTH 68
#define RUY_OFFSET_CLAMP_MIN 72
#define RUY_OFFSET_CLAMP_MAX 76
#define RUY_OFFSET_FLAGS 80

template <typename Params>
void CheckOffsetsInKernelParamsFloat(const Params&) {
  static_assert(offsetof(Params, lhs_base_ptr) == RUY_OFFSET_LHS_BASE_PTR, "");
  static_assert(offsetof(Params, rhs_base_ptr) == RUY_OFFSET_RHS_BASE_PTR, "");
  static_assert(offsetof(Params, dst_base_ptr) == RUY_OFFSET_DST_BASE_PTR, "");
  static_assert(offsetof(Params, bias) == RUY_OFFSET_BIAS, "");
  static_assert(offsetof(Params, start_row) == RUY_OFFSET_START_ROW, "");
  static_assert(offsetof(Params, start_col) == RUY_OFFSET_START_COL, "");
  static_assert(offsetof(Params, last_row) == RUY_OFFSET_LAST_ROW, "");
  static_assert(offsetof(Params, last_col) == RUY_OFFSET_LAST_COL, "");
  static_assert(offsetof(Params, lhs_stride) == RUY_OFFSET_LHS_STRIDE, "");
  static_assert(offsetof(Params, rhs_stride) == RUY_OFFSET_RHS_STRIDE, "");
  static_assert(offsetof(Params, dst_stride) == RUY_OFFSET_DST_STRIDE, "");
  static_assert(offsetof(Params, depth) == RUY_OFFSET_DEPTH, "");
  static_assert(offsetof(Params, clamp_min) == RUY_OFFSET_CLAMP_MIN, "");
  static_assert(offsetof(Params, clamp_max) == RUY_OFFSET_CLAMP_MAX, "");
  static_assert(offsetof(Params, flags) == RUY_OFFSET_FLAGS, "");
}

// Just a plain float kernel; good enough for out-of-order cores.
// The closest to it in the gemmlowp collection would be
// NEON_64bit_GEMM_Float32_WithScalar,
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L3925
//
// Besides ruy-ification, the main nuance here is that we stick to a 8x8
// width instead of the wider 12x8 that the register space permits and that
// the aforementioned gemmlowp kernel uses.  Ruy likes powers of two for now
// and we don't have evidence that going beyond 8x8 is needed.
void KernelFloatNeonOutOfOrder(const KernelParamsFloat<8, 8>& params) {
  CheckOffsetsInKernelParamsFloat(params);
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeon, optimized for out-of-order cores)");

  const float* lhs_col_ptr = params.lhs_base_ptr;
  const float* rhs_col_ptr = params.rhs_base_ptr;
  const float* lhs_ptr = lhs_col_ptr;
  const float* rhs_ptr = rhs_col_ptr;
  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are accumulators.
  // During accumulation, v0 -- v15 are used to load data from LHS and RHS.
  // At least v0 and v1 are used to load a 8x1 block of LHS, and v2 and
  // v3 are used to load a 1x8 block of RHS, like this:
  //
  //                                          RHS 1x8 block
  //                           /-----------------------------------------\
  //                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
  //                           \-----------------------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /-----------------------------------------\
  //  |        v0.s[0]      |  |v16.s[0]           ...           v30.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v0.s[3]      |  |v16.s[3]           ...           v30.s[3]|
  //  |        v1.s[0]      |  |v17.s[0]           ...           v31.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v1.s[3]      |  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                      accumulators 8x8 block
  //
  // In the RUY_OPT_MAX_STREAMING part of the kernel, this elementary step
  // is repeated 4 times, using 4x more registers for LHS and RHS, so that
  // is where instead of using v0 -- v3 for LHS and RHS, we use v0 -- v15.
  //
  // Outside of the RUY_OPT_MAX_STREAMING part of the kernel, v4 -- v7 are
  // unused, and v8 -- v15 are used for floading parameters used for the
  // post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"

        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov w1, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

#if RUY_OPT_ENABLED(RUY_OPT_MAX_STREAMING)
        "cmp w12, #8\n"
        "blt 78f\n"
        "and w2, w12, #-4\n"

        "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v5.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v6.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v7.4s}, [%[rhs_ptr]], #16\n"

        "ld1 {v8.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v9.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v10.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v11.4s}, [%[rhs_ptr]], #16\n"

        "ld1 {v12.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v13.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v14.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v15.4s}, [%[rhs_ptr]], #16\n"
        "mov w1, #4\n"

        "80:\n"

        "add %[lhs_ptr], %[lhs_ptr], #128\n"
        "add %[rhs_ptr], %[rhs_ptr], #128\n"

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ldr q0, [%[lhs_ptr], #-128]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ldr q3, [%[rhs_ptr], #-112]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        "ldr q1, [%[lhs_ptr], #-112]\n"
        "fmla v16.4s, v4.4s, v6.s[0]\n"
        "fmla v18.4s, v4.4s, v6.s[1]\n"
        "ldr q2, [%[rhs_ptr], #-128]\n"
        "fmla v20.4s, v4.4s, v6.s[2]\n"
        "fmla v22.4s, v4.4s, v6.s[3]\n"

        "fmla v24.4s, v4.4s, v7.s[0]\n"
        "fmla v26.4s, v4.4s, v7.s[1]\n"
        "fmla v28.4s, v4.4s, v7.s[2]\n"
        "fmla v30.4s, v4.4s, v7.s[3]\n"
        "ldr q4, [%[lhs_ptr], #-96]\n"
        "fmla v25.4s, v5.4s, v7.s[0]\n"
        "fmla v27.4s, v5.4s, v7.s[1]\n"
        "fmla v29.4s, v5.4s, v7.s[2]\n"
        "fmla v31.4s, v5.4s, v7.s[3]\n"
        "ldr q7, [%[rhs_ptr], #-80]\n"
        "fmla v17.4s, v5.4s, v6.s[0]\n"
        "fmla v19.4s, v5.4s, v6.s[1]\n"
        "fmla v21.4s, v5.4s, v6.s[2]\n"
        "fmla v23.4s, v5.4s, v6.s[3]\n"
        "ldr q5, [%[lhs_ptr], #-80]\n"
        "fmla v16.4s, v8.4s, v10.s[0]\n"
        "fmla v18.4s, v8.4s, v10.s[1]\n"
        "ldr q6, [%[rhs_ptr], #-96]\n"
        "fmla v20.4s, v8.4s, v10.s[2]\n"
        "fmla v22.4s, v8.4s, v10.s[3]\n"

        "fmla v24.4s, v8.4s, v11.s[0]\n"
        "fmla v26.4s, v8.4s, v11.s[1]\n"
        "fmla v28.4s, v8.4s, v11.s[2]\n"
        "fmla v30.4s, v8.4s, v11.s[3]\n"
        "ldr q8, [%[lhs_ptr], #-64]\n"
        "fmla v25.4s, v9.4s, v11.s[0]\n"
        "fmla v27.4s, v9.4s, v11.s[1]\n"
        "fmla v29.4s, v9.4s, v11.s[2]\n"
        "fmla v31.4s, v9.4s, v11.s[3]\n"
        "ldr q11, [%[rhs_ptr], #-48]\n"
        "fmla v17.4s, v9.4s, v10.s[0]\n"
        "fmla v19.4s, v9.4s, v10.s[1]\n"
        "fmla v21.4s, v9.4s, v10.s[2]\n"
        "fmla v23.4s, v9.4s, v10.s[3]\n"
        "ldr q9, [%[lhs_ptr], #-48]\n"
        "fmla v16.4s, v12.4s, v14.s[0]\n"
        "fmla v18.4s, v12.4s, v14.s[1]\n"
        "ldr q10, [%[rhs_ptr], #-64]\n"
        "fmla v20.4s, v12.4s, v14.s[2]\n"
        "fmla v22.4s, v12.4s, v14.s[3]\n"

        "fmla v24.4s, v12.4s, v15.s[0]\n"
        "fmla v26.4s, v12.4s, v15.s[1]\n"
        "fmla v28.4s, v12.4s, v15.s[2]\n"
        "fmla v30.4s, v12.4s, v15.s[3]\n"
        "ldr q12, [%[lhs_ptr], #-32]\n"
        "fmla v25.4s, v13.4s, v15.s[0]\n"
        "fmla v27.4s, v13.4s, v15.s[1]\n"
        "fmla v29.4s, v13.4s, v15.s[2]\n"
        "fmla v31.4s, v13.4s, v15.s[3]\n"
        "ldr q15, [%[rhs_ptr], #-16]\n"
        "fmla v17.4s, v13.4s, v14.s[0]\n"
        "fmla v19.4s, v13.4s, v14.s[1]\n"
        "fmla v21.4s, v13.4s, v14.s[2]\n"
        "fmla v23.4s, v13.4s, v14.s[3]\n"
        "ldr q13, [%[lhs_ptr], #-16]\n"
        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "ldr q14, [%[rhs_ptr], #-32]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

        "add w1, w1, #4\n"
        "cmp w1, w2\n"
        "blt 80b\n"

        "fmla v16.4s, v4.4s, v6.s[0]\n"
        "fmla v18.4s, v4.4s, v6.s[1]\n"
        "fmla v20.4s, v4.4s, v6.s[2]\n"
        "fmla v22.4s, v4.4s, v6.s[3]\n"
        "fmla v24.4s, v4.4s, v7.s[0]\n"
        "fmla v26.4s, v4.4s, v7.s[1]\n"
        "fmla v28.4s, v4.4s, v7.s[2]\n"
        "fmla v30.4s, v4.4s, v7.s[3]\n"
        "fmla v25.4s, v5.4s, v7.s[0]\n"
        "fmla v27.4s, v5.4s, v7.s[1]\n"
        "fmla v29.4s, v5.4s, v7.s[2]\n"
        "fmla v31.4s, v5.4s, v7.s[3]\n"
        "fmla v17.4s, v5.4s, v6.s[0]\n"
        "fmla v19.4s, v5.4s, v6.s[1]\n"
        "fmla v21.4s, v5.4s, v6.s[2]\n"
        "fmla v23.4s, v5.4s, v6.s[3]\n"

        "fmla v16.4s, v8.4s, v10.s[0]\n"
        "fmla v18.4s, v8.4s, v10.s[1]\n"
        "fmla v20.4s, v8.4s, v10.s[2]\n"
        "fmla v22.4s, v8.4s, v10.s[3]\n"
        "fmla v24.4s, v8.4s, v11.s[0]\n"
        "fmla v26.4s, v8.4s, v11.s[1]\n"
        "fmla v28.4s, v8.4s, v11.s[2]\n"
        "fmla v30.4s, v8.4s, v11.s[3]\n"
        "fmla v25.4s, v9.4s, v11.s[0]\n"
        "fmla v27.4s, v9.4s, v11.s[1]\n"
        "fmla v29.4s, v9.4s, v11.s[2]\n"
        "fmla v31.4s, v9.4s, v11.s[3]\n"
        "fmla v17.4s, v9.4s, v10.s[0]\n"
        "fmla v19.4s, v9.4s, v10.s[1]\n"
        "fmla v21.4s, v9.4s, v10.s[2]\n"
        "fmla v23.4s, v9.4s, v10.s[3]\n"

        "fmla v16.4s, v12.4s, v14.s[0]\n"
        "fmla v18.4s, v12.4s, v14.s[1]\n"
        "fmla v20.4s, v12.4s, v14.s[2]\n"
        "fmla v22.4s, v12.4s, v14.s[3]\n"
        "fmla v24.4s, v12.4s, v15.s[0]\n"
        "fmla v26.4s, v12.4s, v15.s[1]\n"
        "fmla v28.4s, v12.4s, v15.s[2]\n"
        "fmla v30.4s, v12.4s, v15.s[3]\n"
        "fmla v25.4s, v13.4s, v15.s[0]\n"
        "fmla v27.4s, v13.4s, v15.s[1]\n"
        "fmla v29.4s, v13.4s, v15.s[2]\n"
        "fmla v31.4s, v13.4s, v15.s[3]\n"
        "fmla v17.4s, v13.4s, v14.s[0]\n"
        "fmla v19.4s, v13.4s, v14.s[1]\n"
        "fmla v21.4s, v13.4s, v14.s[2]\n"
        "fmla v23.4s, v13.4s, v14.s[3]\n"

        "78:\n"
#endif

        // Accumulation loop
        "cmp w1, w12\n"
        "beq 79f\n"

        "2:\n"
        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "ld1 {v4.4s}, [%[rhs_ptr]], #16\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "add w1, w1, #1\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "cmp w1, w12\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "fmla v16.4s, v0.4s, v4.s[0]\n"
        "fmla v18.4s, v0.4s, v4.s[1]\n"
        "mov v2.16b, v4.16b\n"
        "fmla v20.4s, v0.4s, v4.s[2]\n"
        "fmla v22.4s, v0.4s, v4.s[3]\n"
        "blt 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
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

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Offset these base pointers as needed given the current row, col.
        "add x5, x1, %x[row], lsl #2\n"

        "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fadd v17.4s, v17.4s, v15.4s\n"
        "fadd v18.4s, v18.4s, v14.4s\n"
        "fadd v19.4s, v19.4s, v15.4s\n"
        "fadd v20.4s, v20.4s, v14.4s\n"
        "fadd v21.4s, v21.4s, v15.4s\n"
        "fadd v22.4s, v22.4s, v14.4s\n"
        "fadd v23.4s, v23.4s, v15.4s\n"
        "fadd v24.4s, v24.4s, v14.4s\n"
        "fadd v25.4s, v25.4s, v15.4s\n"
        "fadd v26.4s, v26.4s, v14.4s\n"
        "fadd v27.4s, v27.4s, v15.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v15.4s\n"
        "fadd v30.4s, v30.4s, v14.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"

        // Load the clamp_min, clamp_max bounds
        "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.4s, w2\n"  // clamp_min
        "dup v15.4s, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "fmax v16.4s, v16.4s, v14.4s\n"
        "fmax v17.4s, v17.4s, v14.4s\n"
        "fmax v18.4s, v18.4s, v14.4s\n"
        "fmax v19.4s, v19.4s, v14.4s\n"
        "fmax v20.4s, v20.4s, v14.4s\n"
        "fmax v21.4s, v21.4s, v14.4s\n"
        "fmax v22.4s, v22.4s, v14.4s\n"
        "fmax v23.4s, v23.4s, v14.4s\n"
        "fmax v24.4s, v24.4s, v14.4s\n"
        "fmax v25.4s, v25.4s, v14.4s\n"
        "fmax v26.4s, v26.4s, v14.4s\n"
        "fmax v27.4s, v27.4s, v14.4s\n"
        "fmax v28.4s, v28.4s, v14.4s\n"
        "fmax v29.4s, v29.4s, v14.4s\n"
        "fmax v30.4s, v30.4s, v14.4s\n"
        "fmax v31.4s, v31.4s, v14.4s\n"

        // Apply the clamp_max bound
        "fmin v16.4s, v16.4s, v15.4s\n"
        "fmin v17.4s, v17.4s, v15.4s\n"
        "fmin v18.4s, v18.4s, v15.4s\n"
        "fmin v19.4s, v19.4s, v15.4s\n"
        "fmin v20.4s, v20.4s, v15.4s\n"
        "fmin v21.4s, v21.4s, v15.4s\n"
        "fmin v22.4s, v22.4s, v15.4s\n"
        "fmin v23.4s, v23.4s, v15.4s\n"
        "fmin v24.4s, v24.4s, v15.4s\n"
        "fmin v25.4s, v25.4s, v15.4s\n"
        "fmin v26.4s, v26.4s, v15.4s\n"
        "fmin v27.4s, v27.4s, v15.4s\n"
        "fmin v28.4s, v28.4s, v15.4s\n"
        "fmin v29.4s, v29.4s, v15.4s\n"
        "fmin v30.4s, v30.4s, v15.4s\n"
        "fmin v31.4s, v31.4s, v15.4s\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "str q16, [x3, #0]\n"
        "str q17, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "str q18, [x3, #0]\n"
        "str q19, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "str q20, [x3, #0]\n"
        "str q21, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "str q22, [x3, #0]\n"
        "str q23, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "str q24, [x3, #0]\n"
        "str q25, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "str q26, [x3, #0]\n"
        "str q27, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "str q28, [x3, #0]\n"
        "str q29, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "str q30, [x3, #0]\n"
        "str q31, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that we have already loaded
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently 1.
        "mov w1, #1\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Variant of KernelFloatNeonOutOfOrder tuned for in-order CPUs that do not
// support dotprod (while dotprod by itself is not relevant to floating-point,
// this additional bit of information that we have about the target happens to
// be useful here).
//
// So a typical target CPU here would be ARM Cortex-A53 or the original
// Cortex-A55.
//
// This kernel is similar to and inspired by gemmlowp's
// NEON_64bit_GEMM_Float32_WithScalar_A53.
// which was contributed by David Mansell with very helpful
// comments. Specifically, see this comment about tuning for Cortex-A53:
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L4215
void KernelFloatNeonInOrder(const KernelParamsFloat<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeon, optimized for in-order cores)");

  CheckOffsetsInKernelParamsFloat(params);

  const float* lhs_col_ptr = params.lhs_base_ptr;
  const float* rhs_col_ptr = params.rhs_base_ptr;
  const float* lhs_ptr = lhs_col_ptr;
  const float* rhs_ptr = rhs_col_ptr;
  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are accumulators.
  // During accumulation, v0 -- v3 are used to load data from LHS and RHS.
  //
  //                                          RHS 1x8 block
  //                           /-----------------------------------------\
  //                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
  //                           \-----------------------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /-----------------------------------------\
  //  |        v0.s[0]      |  |v16.s[0]           ...           v30.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v0.s[3]      |  |v16.s[3]           ...           v30.s[3]|
  //  |        v1.s[0]      |  |v17.s[0]           ...           v31.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v1.s[3]      |  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                      accumulators 8x8 block
  //
  // There is no RUY_OPT_MAX_STREAMING 4x-unrolled part in this kernel because
  // we did not observe a benefit of such partial unrolling on in-order CPUs.
  //
  // v4 -- v7 are unused, and v8 -- v15 are used for floading parameters used
  // for the post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"


        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v17)
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v18)
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v19)
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #64]\n")
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #64]\n")
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #128]\n")
        RUY_MAKE_ZERO(v23)
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #128]\n")
        RUY_MAKE_ZERO(v24)
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #192]\n")
        RUY_MAKE_ZERO(v25)
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #192]\n")
        RUY_MAKE_ZERO(v26)
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #256]\n")
        RUY_MAKE_ZERO(v27)
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #256]\n")
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that remain to load
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently depth - 1.
        "sub w1, w12, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        "cmp w1, #0\n"
        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

        // Accumulation loop
        "beq 79f\n"

        "2:\n"

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "ldr x2, [%[lhs_ptr], #8]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "ldr x3, [%[lhs_ptr], #24]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "ldr x5, [%[rhs_ptr], #24]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ldr x4, [%[rhs_ptr], #8]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "subs w1, w1, #1\n"
        "ldr d0, [%[lhs_ptr]], #32\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ins v0.d[1], x2\n"
        "ldr d3, [%[rhs_ptr], #16]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "ins v3.d[1], x5\n"
        "ldr d4, [%[rhs_ptr]], #32\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        "fmla v16.4s, v0.4s, v4.s[0]\n"
        "ins v4.d[1], x4\n"
        "ldr d1, [%[lhs_ptr], #-16]\n"
        "fmla v18.4s, v0.4s, v4.s[1]\n"
        "fmla v20.4s, v0.4s, v4.s[2]\n"
        "ins v1.d[1], x3\n"
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #256]\n")
        "mov v2.16b, v4.16b\n"
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #256]\n")
        "fmla v22.4s, v0.4s, v4.s[3]\n"
        "bne 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
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

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Offset these base pointers as needed given the current row, col.
        "add x5, x1, %x[row], lsl #2\n"

        "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fadd v17.4s, v17.4s, v15.4s\n"
        "fadd v18.4s, v18.4s, v14.4s\n"
        "fadd v19.4s, v19.4s, v15.4s\n"
        "fadd v20.4s, v20.4s, v14.4s\n"
        "fadd v21.4s, v21.4s, v15.4s\n"
        "fadd v22.4s, v22.4s, v14.4s\n"
        "fadd v23.4s, v23.4s, v15.4s\n"
        "fadd v24.4s, v24.4s, v14.4s\n"
        "fadd v25.4s, v25.4s, v15.4s\n"
        "fadd v26.4s, v26.4s, v14.4s\n"
        "fadd v27.4s, v27.4s, v15.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v15.4s\n"
        "fadd v30.4s, v30.4s, v14.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"

        // Load the clamp_min, clamp_max bounds
        "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.4s, w2\n"  // clamp_min
        "dup v15.4s, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "fmax v16.4s, v16.4s, v14.4s\n"
        "fmax v17.4s, v17.4s, v14.4s\n"
        "fmax v18.4s, v18.4s, v14.4s\n"
        "fmax v19.4s, v19.4s, v14.4s\n"
        "fmax v20.4s, v20.4s, v14.4s\n"
        "fmax v21.4s, v21.4s, v14.4s\n"
        "fmax v22.4s, v22.4s, v14.4s\n"
        "fmax v23.4s, v23.4s, v14.4s\n"
        "fmax v24.4s, v24.4s, v14.4s\n"
        "fmax v25.4s, v25.4s, v14.4s\n"
        "fmax v26.4s, v26.4s, v14.4s\n"
        "fmax v27.4s, v27.4s, v14.4s\n"
        "fmax v28.4s, v28.4s, v14.4s\n"
        "fmax v29.4s, v29.4s, v14.4s\n"
        "fmax v30.4s, v30.4s, v14.4s\n"
        "fmax v31.4s, v31.4s, v14.4s\n"

        // Apply the clamp_max bound
        "fmin v16.4s, v16.4s, v15.4s\n"
        "fmin v17.4s, v17.4s, v15.4s\n"
        "fmin v18.4s, v18.4s, v15.4s\n"
        "fmin v19.4s, v19.4s, v15.4s\n"
        "fmin v20.4s, v20.4s, v15.4s\n"
        "fmin v21.4s, v21.4s, v15.4s\n"
        "fmin v22.4s, v22.4s, v15.4s\n"
        "fmin v23.4s, v23.4s, v15.4s\n"
        "fmin v24.4s, v24.4s, v15.4s\n"
        "fmin v25.4s, v25.4s, v15.4s\n"
        "fmin v26.4s, v26.4s, v15.4s\n"
        "fmin v27.4s, v27.4s, v15.4s\n"
        "fmin v28.4s, v28.4s, v15.4s\n"
        "fmin v29.4s, v29.4s, v15.4s\n"
        "fmin v30.4s, v30.4s, v15.4s\n"
        "fmin v31.4s, v31.4s, v15.4s\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "str q16, [x3, #0]\n"
        "str q17, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "str q18, [x3, #0]\n"
        "str q19, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "str q20, [x3, #0]\n"
        "str q21, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "str q22, [x3, #0]\n"
        "str q23, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "str q24, [x3, #0]\n"
        "str q25, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "str q26, [x3, #0]\n"
        "str q27, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "str q28, [x3, #0]\n"
        "str q29, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "str q30, [x3, #0]\n"
        "str q31, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that remain to load
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently depth - 1.
        "sub w1, w12, #1\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
}

// Variant of KernelFloatNeonInOrder tuned for in-order CPUs that do
// support dotprod (while dotprod by itself is not relevant to floating-point,
// this additional bit of information that we have about the target happens to
// be useful here).
//
// So a typical target CPU here would be ARM Cortex-A55r1.
//
// This kernel is similar to and inspired by gemmlowp's
// NEON_64bit_GEMM_Float32_WithScalar_A55r1.
// which was contributed by David Mansell with very helpful
// comments. Specifically, see this comment about tuning for Cortex-A55r1:
// https://github.com/google/gemmlowp/blob/36212ad3651871bc3e9a599f1a6d5324778aea25/standalone/neon-gemm-kernel-benchmark.cc#L4412
void KernelFloatNeonDotprodInOrder(const KernelParamsFloat<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label(
      "Kernel (kNeonDotprod, optimized for in-order cores)");

  CheckOffsetsInKernelParamsFloat(params);

  const float* lhs_col_ptr = params.lhs_base_ptr;
  const float* rhs_col_ptr = params.rhs_base_ptr;
  const float* lhs_ptr = lhs_col_ptr;
  const float* rhs_ptr = rhs_col_ptr;
  float* dst_col_ptr = params.dst_base_ptr;
  float* dst_ptr = dst_col_ptr;
  int row = params.start_row;
  int col = params.start_col;

  // The asm kernel below has the following NEON register allocation:
  //
  // v16 -- v31 are accumulators.
  // During accumulation, v0 -- v3 are used to load data from LHS and RHS.
  //
  //                                          RHS 1x8 block
  //                           /-----------------------------------------\
  //                           |v2.s[0] ... v2.s[3]   v3.s[0] ... v3.s[3]|
  //                           \-----------------------------------------/
  //        LHS 8x1 block
  //  /---------------------\  /-----------------------------------------\
  //  |        v0.s[0]      |  |v16.s[0]           ...           v30.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v0.s[3]      |  |v16.s[3]           ...           v30.s[3]|
  //  |        v1.s[0]      |  |v17.s[0]           ...           v31.s[0]|
  //  |         ...         |  |  ...                              ...   |
  //  |        v1.s[3]      |  |v17.s[3]           ...           v31.s[3]|
  //  \---------------------/  \-----------------------------------------/
  //                                      accumulators 8x8 block
  //
  // There is no RUY_OPT_MAX_STREAMING 4x-unrolled part in this kernel because
  // we did not observe a benefit of such partial unrolling on in-order CPUs.
  //
  // v4 -- v7 are unused, and v8 -- v15 are used for floading parameters used
  // for the post-accumulation part of the kernel.
  asm volatile(
#define RUY_MAKE_ZERO(reg) "dup " #reg ".4s, wzr\n"

        // clang-format off

        // Load some parameters into registers.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"
        "ldr w8, [%[params], #" RUY_STR(RUY_OFFSET_LAST_COL) "]\n"
        "ldr w9, [%[params], #" RUY_STR(RUY_OFFSET_LHS_STRIDE) "]\n"
        "ldr w10, [%[params], #" RUY_STR(RUY_OFFSET_RHS_STRIDE) "]\n"
        "ldr w11, [%[params], #" RUY_STR(RUY_OFFSET_DST_STRIDE) "]\n"
        "ldr w12, [%[params], #" RUY_STR(RUY_OFFSET_DEPTH) "]\n"


        // Clear accumulators.
        RUY_MAKE_ZERO(v16)
        // Load the first 32 bytes of LHS and RHS data.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v17)
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v18)
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v19)
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"
        RUY_MAKE_ZERO(v20)
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #64]\n")
        RUY_MAKE_ZERO(v21)
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #64]\n")
        RUY_MAKE_ZERO(v22)
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #128]\n")
        RUY_MAKE_ZERO(v23)
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #128]\n")
        RUY_MAKE_ZERO(v24)
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #192]\n")
        RUY_MAKE_ZERO(v25)
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #192]\n")
        RUY_MAKE_ZERO(v26)
        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #256]\n")
        RUY_MAKE_ZERO(v27)
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #256]\n")
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // w1 is the number of levels of depth that remain to load
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently depth - 1.
        "sub w1, w12, #1\n"

        // Main loop of the whole GEMM, over rows and columns of the
        // destination matrix.
        "1:\n"

        "cmp w1, #0\n"
        "fmla v16.4s, v0.4s, v2.s[0]\n"
        "fmla v18.4s, v0.4s, v2.s[1]\n"
        "fmla v20.4s, v0.4s, v2.s[2]\n"
        "fmla v22.4s, v0.4s, v2.s[3]\n"

        // Accumulation loop
        "beq 79f\n"

        "2:\n"

        RUY_PREFETCH("prfm pldl1keep, [%[lhs_ptr], #256]\n")
        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "ldr x2, [%[lhs_ptr], #8]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "ldr x3, [%[lhs_ptr], #24]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "ldr x5, [%[rhs_ptr], #24]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "ldr d0, [%[lhs_ptr]], #32\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "ldr x4, [%[rhs_ptr], #8]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "subs w1, w1, #1\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "ins v0.d[1], x2\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ldr d3, [%[rhs_ptr], #16]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "ins v3.d[1], x5\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "ldr d4, [%[rhs_ptr]], #32\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "ins v4.d[1], x4\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"
        RUY_PREFETCH("prfm pldl1keep, [%[rhs_ptr], #256]\n")
        "fmla v16.4s, v0.4s, v4.s[0]\n"
        "ldr d1, [%[lhs_ptr], #-16]\n"
        "fmla v18.4s, v0.4s, v4.s[1]\n"
        "ins v1.d[1], x3\n"
        "fmla v20.4s, v0.4s, v4.s[2]\n"
        "mov v2.16b, v4.16b\n"
        "fmla v22.4s, v0.4s, v4.s[3]\n"
        "bne 2b\n"

        "79:\n"

        // End of the inner loop on depth. Now perform the remaining
        // multiply-adds of the last level of depth, for which the LHS
        // and RHS data is already loaded.

        "fmla v24.4s, v0.4s, v3.s[0]\n"
        "fmla v26.4s, v0.4s, v3.s[1]\n"
        "fmla v28.4s, v0.4s, v3.s[2]\n"
        "fmla v30.4s, v0.4s, v3.s[3]\n"
        "fmla v25.4s, v1.4s, v3.s[0]\n"
        "fmla v27.4s, v1.4s, v3.s[1]\n"
        "fmla v29.4s, v1.4s, v3.s[2]\n"
        "fmla v31.4s, v1.4s, v3.s[3]\n"
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "fmla v17.4s, v1.4s, v2.s[0]\n"
        "fmla v19.4s, v1.4s, v2.s[1]\n"
        "fmla v21.4s, v1.4s, v2.s[2]\n"
        "fmla v23.4s, v1.4s, v2.s[3]\n"

        // End of accumulation. The registers v16 -- v31 contain the final
        // int32 accumulator values of the current 8x8 destination block.
        // We now have to compute the final 8-bit values from these int32
        // accumulators, and advance to the next 8x8 block. We intertwine
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

        "cmp %w[row], w7\n"  // Have we finished the last row?
        "bge 4f\n"           // If finished last row, go to 4
        // Not finished last row: then advance to next row.
        "add %[lhs_col_ptr], %[lhs_col_ptr], x9, lsl #3\n"
        "b 5f\n"
        "4:\n"  // Finished last row...
        "mov %[lhs_col_ptr], x5\n"  // Go back to first row
        // Now we need to advance to the next column. If we already
        // finished the last column, then in principle we are done, however
        // we can't just return here, as we need to allow the end work of the
        // current block to complete. The good news is that at this point it
        // doesn't matter what data we load for the next column, since
        // we will exit from the main loop below before actually storing
        // anything computed from that data.
        "cmp %w[col], w8\n"  // Have we finished the last column?
        "bge 5f\n" // If yes, just carry on without updating the column pointer.
        // Not finished last column: then advance to next column.
        "add %[rhs_col_ptr], %[rhs_col_ptr], x10, lsl #3\n"
        "5:\n"

        // Set the LHS and RHS data pointers to the start of the columns just
        // computed.
        "mov %[lhs_ptr], %[lhs_col_ptr]\n"
        "mov %[rhs_ptr], %[rhs_col_ptr]\n"

        // Load some parameters needed for the end work on current block.
        "ldrb w4, [%[params], #" RUY_STR(RUY_OFFSET_FLAGS) "]\n"
        "ldr x1, [%[params], #" RUY_STR(RUY_OFFSET_BIAS) "]\n"

        // Offset these base pointers as needed given the current row, col.
        "add x5, x1, %x[row], lsl #2\n"

        "tst w4, #" RUY_STR(RUY_ASM_FLAG_HAS_BIAS) "\n"
        "csel x1, x1, x5, eq\n"

        // Load 8 bias values.
        "ld1 {v14.4s}, [x1], #16\n"
        "ld1 {v15.4s}, [x1]\n"

        // Now that we know what LHS and RHS data the next iteration of the
        // main loop will need to load, we start loading the first 32 bytes of
        // each of LHS and RHS, into v0 -- v3, as we don't need v0 -- v3 anymore
        // in the rest of the work on the current block.
        "ld1 {v0.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v2.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[rhs_ptr]], #16\n"

        // Perform the bias-addition (per the above, we have just folded into
        // the bias the (depth * lhs_zero_point * rhs_zero_point) term.)
        "fadd v16.4s, v16.4s, v14.4s\n"
        "fadd v17.4s, v17.4s, v15.4s\n"
        "fadd v18.4s, v18.4s, v14.4s\n"
        "fadd v19.4s, v19.4s, v15.4s\n"
        "fadd v20.4s, v20.4s, v14.4s\n"
        "fadd v21.4s, v21.4s, v15.4s\n"
        "fadd v22.4s, v22.4s, v14.4s\n"
        "fadd v23.4s, v23.4s, v15.4s\n"
        "fadd v24.4s, v24.4s, v14.4s\n"
        "fadd v25.4s, v25.4s, v15.4s\n"
        "fadd v26.4s, v26.4s, v14.4s\n"
        "fadd v27.4s, v27.4s, v15.4s\n"
        "fadd v28.4s, v28.4s, v14.4s\n"
        "fadd v29.4s, v29.4s, v15.4s\n"
        "fadd v30.4s, v30.4s, v14.4s\n"
        "fadd v31.4s, v31.4s, v15.4s\n"

        // Load the clamp_min, clamp_max bounds
        "ldr w2, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MIN) "]\n"
        "ldr w3, [%[params], #" RUY_STR(RUY_OFFSET_CLAMP_MAX) "]\n"
        "dup v14.4s, w2\n"  // clamp_min
        "dup v15.4s, w3\n"  // clamp_max

        // Apply the clamp_min bound
        "fmax v16.4s, v16.4s, v14.4s\n"
        "fmax v17.4s, v17.4s, v14.4s\n"
        "fmax v18.4s, v18.4s, v14.4s\n"
        "fmax v19.4s, v19.4s, v14.4s\n"
        "fmax v20.4s, v20.4s, v14.4s\n"
        "fmax v21.4s, v21.4s, v14.4s\n"
        "fmax v22.4s, v22.4s, v14.4s\n"
        "fmax v23.4s, v23.4s, v14.4s\n"
        "fmax v24.4s, v24.4s, v14.4s\n"
        "fmax v25.4s, v25.4s, v14.4s\n"
        "fmax v26.4s, v26.4s, v14.4s\n"
        "fmax v27.4s, v27.4s, v14.4s\n"
        "fmax v28.4s, v28.4s, v14.4s\n"
        "fmax v29.4s, v29.4s, v14.4s\n"
        "fmax v30.4s, v30.4s, v14.4s\n"
        "fmax v31.4s, v31.4s, v14.4s\n"

        // Apply the clamp_max bound
        "fmin v16.4s, v16.4s, v15.4s\n"
        "fmin v17.4s, v17.4s, v15.4s\n"
        "fmin v18.4s, v18.4s, v15.4s\n"
        "fmin v19.4s, v19.4s, v15.4s\n"
        "fmin v20.4s, v20.4s, v15.4s\n"
        "fmin v21.4s, v21.4s, v15.4s\n"
        "fmin v22.4s, v22.4s, v15.4s\n"
        "fmin v23.4s, v23.4s, v15.4s\n"
        "fmin v24.4s, v24.4s, v15.4s\n"
        "fmin v25.4s, v25.4s, v15.4s\n"
        "fmin v26.4s, v26.4s, v15.4s\n"
        "fmin v27.4s, v27.4s, v15.4s\n"
        "fmin v28.4s, v28.4s, v15.4s\n"
        "fmin v29.4s, v29.4s, v15.4s\n"
        "fmin v30.4s, v30.4s, v15.4s\n"
        "fmin v31.4s, v31.4s, v15.4s\n"

        // Compute how much of the 8x8 block of destination 8bit values that
        // we have computed, fit in the destination matrix. Typically, all of
        // it fits, but when the destination matrix shape is not a multiple
        // of 8x8, there are some 8x8 blocks along the boundaries that do
        // not fit entirely.
        "sub w1, %w[dst_rows], %w[row]\n"
        "sub w2, %w[dst_cols], %w[col]\n"
        "mov w3, #8\n"
        "cmp w1, #8\n"
        // Compute w1 = how many rows of the 8x8 block fit
        "csel w1, w1, w3, le\n"
        "cmp w2, #8\n"
        // Compute w2 = how many cols of the 8x8 block fit
        "csel w2, w2, w3, le\n"

        // Test if w1==8 && w2 == 8, i.e. if all of the 8x8 block fits.
        "cmp w1, w3\n"
        "ccmp w2, w3, 0, eq\n"
        // Yes, all of the 8x8 block fits, go to fast path.
        "beq 30f\n"
        // Not all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write to dst_tmp_buf
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, #32\n"
        "b 31f\n"
        "30:\n"
        // Yes, all of the 8x8 block fits.
        // Set (x3 address, x4 stride) to write directly to destination matrix.
        "mov x3, %[dst_ptr]\n"
        "mov x4, x11\n"
        "31:\n"

        // Write our 8bit values to the destination described by
        // (x3 address, x4 stride).
        "str q16, [x3, #0]\n"
        "str q17, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v16)
        RUY_MAKE_ZERO(v17)
        "str q18, [x3, #0]\n"
        "str q19, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v18)
        RUY_MAKE_ZERO(v19)
        "str q20, [x3, #0]\n"
        "str q21, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v20)
        RUY_MAKE_ZERO(v21)
        "str q22, [x3, #0]\n"
        "str q23, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v22)
        RUY_MAKE_ZERO(v23)
        "str q24, [x3, #0]\n"
        "str q25, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v24)
        RUY_MAKE_ZERO(v25)
        "str q26, [x3, #0]\n"
        "str q27, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v26)
        RUY_MAKE_ZERO(v27)
        "str q28, [x3, #0]\n"
        "str q29, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v28)
        RUY_MAKE_ZERO(v29)
        "str q30, [x3, #0]\n"
        "str q31, [x3, #16]\n"
        "add x3, x3, x4\n"
        RUY_MAKE_ZERO(v30)
        RUY_MAKE_ZERO(v31)

        // If all of the 8x8 block fits, we just finished writing it to the
        // destination, so we skip the next part.
        "beq 41f\n"
        // Not all of the 8x8 block fits in the destination matrix.  We just
        // wrote it to dst_tmp_buf. Now we perform the slow scalar loop over
        // it to copy into the destination matrix the part that fits.
        "mov x3, %[dst_tmp_buf]\n"
        "mov x4, %[dst_ptr]\n"
        "mov w6, #0\n"
        "50:\n"
        "mov w5, #0\n"
        "51:\n"
        "ldr w7, [x3, x5, lsl #2]\n"
        "str w7, [x4, x5, lsl #2]\n"
        "add w5, w5, #1\n"
        "cmp w5, w1\n"
        "blt 51b\n"
        "add w6, w6, #1\n"
        "add x3, x3, #32\n"
        "add x4, x4, x11\n"
        "cmp w6, w2\n"
        "blt 50b\n"
        "41:\n"
        "add %[dst_ptr], %[dst_ptr], #32\n"
        // At this point we have completely finished writing values to the
        // destination matrix for the current block.

        // Reload some params --- we had used x5 -- x7 for a few other things
        // since the last time we had loaded them.
        "ldr x5, [%[params], #" RUY_STR(RUY_OFFSET_LHS_BASE_PTR) "]\n"
        "ldr w6, [%[params], #" RUY_STR(RUY_OFFSET_START_ROW) "]\n"
        "ldr w7, [%[params], #" RUY_STR(RUY_OFFSET_LAST_ROW) "]\n"

        // Move to the next block of the destination matrix, for the next iter
        // of the main loop.  Notice that lhs_col_ptr, rhs_col_ptr have already
        // been updated earlier.
        // Have we reached the end row?
        "cmp %w[row], w7\n"
        "beq 20f\n"  // yes, end row.
        // Not end row. Move to the next row.
        "add %w[row], %w[row], #8\n"
        "b 21f\n"
        "20:\n"
        // Was already at end row.
        "mov %w[row], w6\n"  // Move back to first row.
        "add %w[col], %w[col], #8\n"  // Move to the next column.
        "add %[dst_col_ptr], %[dst_col_ptr], x11, lsl #3\n"
        "mov %[dst_ptr], %[dst_col_ptr]\n"
        "21:\n"

        // Main loop exit condition: have we hit the end column?
        "cmp %w[col], w8\n"

        // w1 is the number of levels of depth that remain to load
        // LHS and RHS data for. Corresponding to the initial ld1 instructions
        // above, this is currently depth - 1.
        "sub w1, w12, #1\n"

        "ble 1b\n"

        // clang-format on

        : [ lhs_col_ptr ] "+r"(lhs_col_ptr), [rhs_col_ptr] "+r"(rhs_col_ptr),
          [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
          [dst_col_ptr] "+r"(dst_col_ptr), [dst_ptr] "+r"(dst_ptr), [row] "+r"(row), [col] "+r"(col)
        : [ params ] "r"(&params), [dst_rows] "r"(params.dst_rows),
          [dst_cols] "r"(params.dst_cols), [dst_tmp_buf] "r"(params.dst_tmp_buf)
        : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
          "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
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

#endif  // RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy
