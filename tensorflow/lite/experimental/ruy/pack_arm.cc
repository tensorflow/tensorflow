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

#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/pack.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"

namespace ruy {

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

void Pack8bitNeonOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                            const void* src_ptr2, const void* src_ptr3,
                            int src_inc0, int src_inc1, int src_inc2,
                            int src_inc3, int src_rows, int src_zero_point,
                            std::int8_t* packed_ptr, int start_col, int end_col,
                            std::int32_t* sums_ptr, int input_xor) {
  profiler::ScopeLabel label("Pack (kNeon, optimized for out-of-order cores)");
  asm volatile(
      // clang-format off
          "dup v26.16b, %w[input_xor]\n"
          "mov w1, #0\n"
          "dup v28.4s, wzr\n"
          "dup v29.4s, wzr\n"
          "dup v30.4s, wzr\n"
          "dup v31.4s, wzr\n"

          "and w2, %w[rows], #-16\n"
          "cmp w1, w2\n"
          "beq 3f\n"

          "add w1, w1, #16\n"
          "ld1 {v0.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v1.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "cmp w1, w2\n"
          "ld1 {v2.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v3.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "beq 2f\n"

          "1:\n"

          "add w1, w1, #16\n"
          "eor v4.16b, v0.16b, v26.16b\n"
          "ld1 {v0.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "eor v5.16b, v1.16b, v26.16b\n"
          "ld1 {v1.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "eor v6.16b, v2.16b, v26.16b\n"
          "ld1 {v2.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "eor v7.16b, v3.16b, v26.16b\n"
          "ld1 {v3.16b}, [%[src_ptr3]], %[src_inc3]\n"

          "saddlp v16.8h, v4.16b\n"
          "str q4, [%[packed_ptr], #0]\n"
          "saddlp v17.8h, v5.16b\n"
          "str q5, [%[packed_ptr], #16]\n"
          "saddlp v18.8h, v6.16b\n"
          "str q6, [%[packed_ptr], #32]\n"
          "saddlp v19.8h, v7.16b\n"
          "str q7, [%[packed_ptr], #48]\n"
          "sadalp v28.4s, v16.8h\n"
          "cmp w1, w2\n"
          "sadalp v29.4s, v17.8h\n"
          "add %[packed_ptr], %[packed_ptr], #64\n"
          "sadalp v30.4s, v18.8h\n"
          "sadalp v31.4s, v19.8h\n"

          "bne 1b\n"

          "2:\n"

          "eor v4.16b, v0.16b, v26.16b\n"
          "eor v5.16b, v1.16b, v26.16b\n"
          "eor v6.16b, v2.16b, v26.16b\n"
          "eor v7.16b, v3.16b, v26.16b\n"

          "saddlp v16.8h, v4.16b\n"
          "str q4, [%[packed_ptr], #0]\n"
          "saddlp v17.8h, v5.16b\n"
          "str q5, [%[packed_ptr], #16]\n"
          "saddlp v18.8h, v6.16b\n"
          "str q6, [%[packed_ptr], #32]\n"
          "saddlp v19.8h, v7.16b\n"
          "str q7, [%[packed_ptr], #48]\n"
          "sadalp v28.4s, v16.8h\n"
          "sadalp v29.4s, v17.8h\n"
          "sadalp v30.4s, v18.8h\n"
          "sadalp v31.4s, v19.8h\n"

          "add %[packed_ptr], %[packed_ptr], #64\n"

          "3:\n"

          "ands w2, %w[rows], #15\n"
          "beq 4f\n"
          "dup v0.16b, %w[src_zero_point]\n"
          "dup v1.16b, %w[src_zero_point]\n"
          "dup v2.16b, %w[src_zero_point]\n"
          "dup v3.16b, %w[src_zero_point]\n"
#define RUY_LOAD_ONE_ROW(R)                   \
  "cmp w2, #" #R "\n"                         \
  "beq 5f\n"                                  \
  "ld1 { v0.b }[" #R "], [%[src_ptr0]], #1\n" \
  "ld1 { v1.b }[" #R "], [%[src_ptr1]], #1\n" \
  "ld1 { v2.b }[" #R "], [%[src_ptr2]], #1\n" \
  "ld1 { v3.b }[" #R "], [%[src_ptr3]], #1\n"

          RUY_LOAD_ONE_ROW(0)
          RUY_LOAD_ONE_ROW(1)
          RUY_LOAD_ONE_ROW(2)
          RUY_LOAD_ONE_ROW(3)
          RUY_LOAD_ONE_ROW(4)
          RUY_LOAD_ONE_ROW(5)
          RUY_LOAD_ONE_ROW(6)
          RUY_LOAD_ONE_ROW(7)
          RUY_LOAD_ONE_ROW(8)
          RUY_LOAD_ONE_ROW(9)
          RUY_LOAD_ONE_ROW(10)
          RUY_LOAD_ONE_ROW(11)
          RUY_LOAD_ONE_ROW(12)
          RUY_LOAD_ONE_ROW(13)
          RUY_LOAD_ONE_ROW(14)
          RUY_LOAD_ONE_ROW(15)
#undef RUY_LOAD_ONE_ROW
          "5:\n"

          "eor v4.16b, v0.16b, v26.16b\n"
          "eor v5.16b, v1.16b, v26.16b\n"
          "eor v6.16b, v2.16b, v26.16b\n"
          "eor v7.16b, v3.16b, v26.16b\n"

          "saddlp v16.8h, v4.16b\n"
          "saddlp v17.8h, v5.16b\n"
          "saddlp v18.8h, v6.16b\n"
          "saddlp v19.8h, v7.16b\n"
          "sadalp v28.4s, v16.8h\n"
          "sadalp v29.4s, v17.8h\n"
          "sadalp v30.4s, v18.8h\n"
          "sadalp v31.4s, v19.8h\n"

          "str q4, [%[packed_ptr], #0]\n"
          "str q5, [%[packed_ptr], #16]\n"
          "str q6, [%[packed_ptr], #32]\n"
          "str q7, [%[packed_ptr], #48]\n"
          "add %[packed_ptr], %[packed_ptr], #64\n"

          "4:\n"

          "addp v28.4s, v28.4s, v29.4s\n"
          "addp v30.4s, v30.4s, v31.4s\n"
          "addp v28.4s, v28.4s, v30.4s\n"

          "cmp %[sums_ptr], #0\n"
          "beq 6f\n"
          "st1 {v28.4s}, [%[sums_ptr]], #16\n"
          "6:\n"
      // clang-format on

      : [ src_ptr0 ] "+r"(src_ptr0), [ src_ptr1 ] "+r"(src_ptr1),
        [ src_ptr2 ] "+r"(src_ptr2), [ src_ptr3 ] "+r"(src_ptr3),
        [ packed_ptr ] "+r"(packed_ptr), [ sums_ptr ] "+r"(sums_ptr)
      : [ src_inc0 ] "r"(static_cast<std::int64_t>(src_inc0)),
        [ src_inc1 ] "r"(static_cast<std::int64_t>(src_inc1)),
        [ src_inc2 ] "r"(static_cast<std::int64_t>(src_inc2)),
        [ src_inc3 ] "r"(static_cast<std::int64_t>(src_inc3)),
        [ rows ] "r"(src_rows), [ src_zero_point ] "r"(src_zero_point),
        [ input_xor ] "r"(input_xor)
      : "cc", "memory", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
}
#endif

#if RUY_PLATFORM(NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#define RUY_OFFSET_SRC_PTR0 0
#define RUY_OFFSET_SRC_PTR1 4
#define RUY_OFFSET_SRC_PTR2 8
#define RUY_OFFSET_SRC_PTR3 12
#define RUY_OFFSET_SUMS_PTR 16
#define RUY_OFFSET_PACKED_PTR 20
#define RUY_OFFSET_SRC_INC0 24
#define RUY_OFFSET_SRC_INC1 28
#define RUY_OFFSET_SRC_INC2 32
#define RUY_OFFSET_SRC_INC3 36
#define RUY_OFFSET_SRC_ROWS 40
#define RUY_OFFSET_SRC_ZERO_POINT 44
#define RUY_OFFSET_INPUT_XOR 48

template <typename Params>
void CheckOffsetsInPackParams8bit(const Params&) {
  static_assert(offsetof(Params, src_ptr0) == RUY_OFFSET_SRC_PTR0, "");
  static_assert(offsetof(Params, src_ptr1) == RUY_OFFSET_SRC_PTR1, "");
  static_assert(offsetof(Params, src_ptr2) == RUY_OFFSET_SRC_PTR2, "");
  static_assert(offsetof(Params, src_ptr3) == RUY_OFFSET_SRC_PTR3, "");
  static_assert(offsetof(Params, sums_ptr) == RUY_OFFSET_SUMS_PTR, "");
  static_assert(offsetof(Params, packed_ptr) == RUY_OFFSET_PACKED_PTR, "");
  static_assert(offsetof(Params, src_inc0) == RUY_OFFSET_SRC_INC0, "");
  static_assert(offsetof(Params, src_inc1) == RUY_OFFSET_SRC_INC1, "");
  static_assert(offsetof(Params, src_inc2) == RUY_OFFSET_SRC_INC2, "");
  static_assert(offsetof(Params, src_inc3) == RUY_OFFSET_SRC_INC3, "");
  static_assert(offsetof(Params, src_rows) == RUY_OFFSET_SRC_ROWS, "");
  static_assert(offsetof(Params, src_zero_point) == RUY_OFFSET_SRC_ZERO_POINT,
                "");
  static_assert(offsetof(Params, input_xor) == RUY_OFFSET_INPUT_XOR, "");
}

// Packing code for out-of-order ARMv7 CPUs like the Krait 400 or A9.
// No attempt made at making this code efficient on in-order cores yet.
void Pack8bitNeonOutOfOrder4Cols(const PackParams8bit& params) {
  CheckOffsetsInPackParams8bit(params);
  profiler::ScopeLabel label("Pack (kNeon, optimized for out-of-order cores)");
  const void* src_ptr0 = params.src_ptr0;
  const void* src_ptr1 = params.src_ptr1;
  const void* src_ptr2 = params.src_ptr2;
  const void* src_ptr3 = params.src_ptr3;
  const int src_inc0 = params.src_inc0;
  const int src_inc1 = params.src_inc1;
  const int src_inc2 = params.src_inc2;
  const int src_inc3 = params.src_inc3;
  const std::int8_t* packed_ptr = params.packed_ptr;

  asm volatile(
      // clang-format off

          "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_INPUT_XOR) "]\n"
          "vdup.8 q11, r2\n"
          "mov r1, #0\n"
          // Zero-out the accumulators
          "vmov.i32 q12, #0\n"
          "vmov.i32 q13, #0\n"
          "vmov.i32 q14, #0\n"
          "vmov.i32 q15, #0\n"

          // Round down src_rows to nearest multiple of 16.
          "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_SRC_ROWS) "]\n"
          "and r2, r3, #-16\n"
          "cmp r1, r2\n"
          "beq 3f\n"

          "1:\n"
          "add r1, r1, #16\n"
          /* Load q0 */
          "vld1.8 {d0, d1}, [%[src_ptr0]]\n"
          "add %[src_ptr0], %[src_ptr0], %[src_inc0]\n"
          RUY_PREFETCH("pld [%[src_ptr0]]\n")

          /* Load q1 */
          "vld1.8 {d2, d3}, [%[src_ptr1]]\n"
          "add %[src_ptr1], %[src_ptr1], %[src_inc1]\n"
          RUY_PREFETCH("pld [%[src_ptr1]]\n")

          "veor.8 q4, q0, q11\n"
          "veor.8 q5, q1, q11\n"

          // Pairwise add in to 16b accumulators.
          "vpaddl.s8 q8, q4\n"
          "vpaddl.s8 q9, q5\n"

          "vst1.32 {q4}, [%[packed_ptr]]!\n"
          "vst1.32 {q5}, [%[packed_ptr]]!\n"

          // Pairwise add accumulate into 32b accumulators.
          // q12 and q13 contain 4x32b accumulators
          "vpadal.s16 q12, q8\n"
          "vpadal.s16 q13, q9\n"

          // Now do the same for src_ptr2 and src_ptr3.
          "vld1.8 {d0, d1}, [%[src_ptr2]]\n"
          "add %[src_ptr2], %[src_ptr2], %[src_inc2]\n"
          RUY_PREFETCH("pld [%[src_ptr2]]\n")

          "vld1.8 {d2, d3}, [%[src_ptr3]]\n"
          "add %[src_ptr3], %[src_ptr3], %[src_inc3]\n"
          RUY_PREFETCH("pld [%[src_ptr3]]\n")

          "veor.8 q4, q0, q11\n"
          "veor.8 q5, q1, q11\n"

          "vpaddl.s8 q8, q4\n"
          "vpaddl.s8 q9, q5\n"

          "vst1.32 {q4}, [%[packed_ptr]]!\n"
          "vst1.32 {q5}, [%[packed_ptr]]!\n"

          // Pairwise add accumulate into 32b accumulators.
          // q14 and q15 contain 4x32b accumulators
          "vpadal.s16 q14, q8\n"
          "vpadal.s16 q15, q9\n"

          "cmp r1, r2\n"
          "bne 1b\n"

          "3:\n"

          // Now pack the last (num_rows % 16) rows.
          "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_SRC_ROWS) "]\n"
          "ands r2, r3, #15\n"
          "beq 4f\n"
          "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_SRC_ZERO_POINT) "]\n"
          "vdup.8 q0, r3\n"
          "vdup.8 q1, r3\n"

// First, read/accumulate/write for src_ptr0 and src_ptr1.
#define RUY_LOAD_ONE_ROW1(I, R)            \
  "cmp r2, #" #I "\n"                      \
  "beq 5f\n"                               \
  "vld1.8 { d0[" #R "]}, [%[src_ptr0]]!\n" \
  "vld1.8 { d2[" #R "]}, [%[src_ptr1]]!\n" \

          RUY_LOAD_ONE_ROW1(0, 0)
          RUY_LOAD_ONE_ROW1(1, 1)
          RUY_LOAD_ONE_ROW1(2, 2)
          RUY_LOAD_ONE_ROW1(3, 3)
          RUY_LOAD_ONE_ROW1(4, 4)
          RUY_LOAD_ONE_ROW1(5, 5)
          RUY_LOAD_ONE_ROW1(6, 6)
          RUY_LOAD_ONE_ROW1(7, 7)
#undef RUY_LOAD_ONE_ROW1

#define RUY_LOAD_ONE_ROW2(I, R)            \
  "cmp r2, #" #I "\n"                      \
  "beq 5f\n"                               \
  "vld1.8 { d1[" #R "]}, [%[src_ptr0]]!\n" \
  "vld1.8 { d3[" #R "]}, [%[src_ptr1]]!\n" \

          RUY_LOAD_ONE_ROW2(8, 0)
          RUY_LOAD_ONE_ROW2(9, 1)
          RUY_LOAD_ONE_ROW2(10, 2)
          RUY_LOAD_ONE_ROW2(11, 3)
          RUY_LOAD_ONE_ROW2(12, 4)
          RUY_LOAD_ONE_ROW2(13, 5)
          RUY_LOAD_ONE_ROW2(14, 6)
          RUY_LOAD_ONE_ROW2(15, 7)
#undef RUY_LOAD_ONE_ROW2

          "5:\n"

          "veor.16 q4, q0, q11\n"
          "veor.16 q5, q1, q11\n"

          "vpaddl.s8 q8, q4\n"
          "vpaddl.s8 q9, q5\n"

          // Pairwise add accumulate to 4x32b accumulators.
          "vpadal.s16 q12, q8\n"
          "vpadal.s16 q13, q9\n"

          "vst1.32 {q4}, [%[packed_ptr]]!\n"
          "vst1.32 {q5}, [%[packed_ptr]]!\n"

          // Reset to src_zero for src_ptr2 and src_ptr3.
          "vdup.8 q0, r3\n"
          "vdup.8 q1, r3\n"

// Next, read/accumulate/write for src_ptr2 and src_ptr3.
#define RUY_LOAD_ONE_ROW1(I, R)            \
  "cmp r2, #" #I "\n"                      \
  "beq 5f\n"                               \
  "vld1.8 { d0[" #R "]}, [%[src_ptr2]]!\n" \
  "vld1.8 { d2[" #R "]}, [%[src_ptr3]]!\n" \

          RUY_LOAD_ONE_ROW1(0, 0)
          RUY_LOAD_ONE_ROW1(1, 1)
          RUY_LOAD_ONE_ROW1(2, 2)
          RUY_LOAD_ONE_ROW1(3, 3)
          RUY_LOAD_ONE_ROW1(4, 4)
          RUY_LOAD_ONE_ROW1(5, 5)
          RUY_LOAD_ONE_ROW1(6, 6)
          RUY_LOAD_ONE_ROW1(7, 7)
#undef RUY_LOAD_ONE_ROW1

#define RUY_LOAD_ONE_ROW2(I, R)            \
  "cmp r2, #" #I "\n"                      \
  "beq 5f\n"                               \
  "vld1.8 { d1[" #R "]}, [%[src_ptr2]]!\n" \
  "vld1.8 { d3[" #R "]}, [%[src_ptr3]]!\n" \

          RUY_LOAD_ONE_ROW2(8, 0)
          RUY_LOAD_ONE_ROW2(9, 1)
          RUY_LOAD_ONE_ROW2(10, 2)
          RUY_LOAD_ONE_ROW2(11, 3)
          RUY_LOAD_ONE_ROW2(12, 4)
          RUY_LOAD_ONE_ROW2(13, 5)
          RUY_LOAD_ONE_ROW2(14, 6)
          RUY_LOAD_ONE_ROW2(15, 7)
#undef RUY_LOAD_ONE_ROW2

          "5:\n"

          "veor.16 q4, q0, q11\n"
          "veor.16 q5, q1, q11\n"

          "vpaddl.s8 q8, q4\n"
          "vpaddl.s8 q9, q5\n"

          // Pairwise add accumulate to 4x32b accumulators.
          "vpadal.s16 q14, q8\n"
          "vpadal.s16 q15, q9\n"

          "vst1.32 {q4}, [%[packed_ptr]]!\n"
          "vst1.32 {q5}, [%[packed_ptr]]!\n"

          "4:\n"
          // Pairwise add 32-bit accumulators
          "vpadd.i32 d24, d24, d25\n"
          "vpadd.i32 d26, d26, d27\n"
          "vpadd.i32 d28, d28, d29\n"
          "vpadd.i32 d30, d30, d31\n"
          // Final 32-bit values per row
          "vpadd.i32 d25, d24, d26\n"
          "vpadd.i32 d27, d28, d30\n"

          "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_SUMS_PTR) "]\n"
          "cmp r3, #0\n"
          "beq 6f\n"
          "vst1.32 {d25}, [r3]!\n"
          "vst1.32 {d27}, [r3]!\n"
          "6:\n"
      // clang-format on

      : [ src_ptr0 ] "+r"(src_ptr0), [ src_ptr1 ] "+r"(src_ptr1),
        [ src_ptr2 ] "+r"(src_ptr2), [ src_ptr3 ] "+r"(src_ptr3)
      : [ src_inc0 ] "r"(src_inc0), [ src_inc1 ] "r"(src_inc1),
        [ src_inc2 ] "r"(src_inc2), [ src_inc3 ] "r"(src_inc3),
        [ packed_ptr ] "r"(packed_ptr), [ params ] "r"(&params)
      : "cc", "memory", "r1", "r2", "r3", "q0", "q1", "q2", "q3",
        "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13");
}

// Packing code for out-of-order ARMv7 CPUs like the Krait 400 or A9.
// No attempt made at making this code efficient on in-order cores yet.
// This version differs from the above in that we only handle two columns
// at a time.
void Pack8bitNeonOutOfOrder2Cols(const PackParams8bit& params) {
  CheckOffsetsInPackParams8bit(params);
  profiler::ScopeLabel label("Pack (kNeon, optimized for out-of-order cores)");
  const void* src_ptr0 = params.src_ptr0;
  const void* src_ptr1 = params.src_ptr1;
  const int src_inc0 = params.src_inc0;
  const int src_inc1 = params.src_inc1;
  const std::int8_t* packed_ptr = params.packed_ptr;

  asm volatile(
      // clang-format off

          "ldr r2, [%[params], #" RUY_STR(RUY_OFFSET_INPUT_XOR) "]\n"
          "vdup.8 q11, r2\n"
          "mov r1, #0\n"
          // Zero-out the accumulators
          "vmov.i32 q12, #0\n"
          "vmov.i32 q13, #0\n"

          // Round down src_rows to nearest multiple of 16.
          "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_SRC_ROWS) "]\n"
          "and r2, r3, #-16\n"
          "cmp r1, r2\n"
          "beq 3f\n"

          "1:\n"
          "add r1, r1, #16\n"
          /* Load q0 */
          "vld1.8 {d0, d1}, [%[src_ptr0]]\n"
          "add %[src_ptr0], %[src_ptr0], %[src_inc0]\n"

          /* Load q1 */
          "vld1.8 {d2, d3}, [%[src_ptr1]]\n"
          "add %[src_ptr1], %[src_ptr1], %[src_inc1]\n"

          "veor.8 q4, q0, q11\n"
          "veor.8 q5, q1, q11\n"

          // Pairwise add in to 16b accumulators.
          "vpaddl.s8 q8, q4\n"
          "vpaddl.s8 q9, q5\n"

          "vst1.32 {q4}, [%[packed_ptr]]!\n"
          "vst1.32 {q5}, [%[packed_ptr]]!\n"

          // Pairwise add accumulate into 32b accumulators.
          // q12 and q13 contain 4x32b accumulators
          "vpadal.s16 q12, q8\n"
          "vpadal.s16 q13, q9\n"

          "cmp r1, r2\n"

          "bne 1b\n"

          "3:\n"

          // Now pack the last (num_rows % 16) rows.
          "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_SRC_ROWS) "]\n"
          "ands r2, r3, #15\n"
          "beq 4f\n"
          "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_SRC_ZERO_POINT) "]\n"
          "vdup.8 q0, r3\n"
          "vdup.8 q1, r3\n"

// Read/accumulate/write for src_ptr0 and src_ptr1.
#define RUY_LOAD_ONE_ROW1(I, R)            \
  "cmp r2, #" #I "\n"                      \
  "beq 5f\n"                               \
  "vld1.8 { d0[" #R "]}, [%[src_ptr0]]!\n" \
  "vld1.8 { d2[" #R "]}, [%[src_ptr1]]!\n" \

          RUY_LOAD_ONE_ROW1(0, 0)
          RUY_LOAD_ONE_ROW1(1, 1)
          RUY_LOAD_ONE_ROW1(2, 2)
          RUY_LOAD_ONE_ROW1(3, 3)
          RUY_LOAD_ONE_ROW1(4, 4)
          RUY_LOAD_ONE_ROW1(5, 5)
          RUY_LOAD_ONE_ROW1(6, 6)
          RUY_LOAD_ONE_ROW1(7, 7)
#undef RUY_LOAD_ONE_ROW1

#define RUY_LOAD_ONE_ROW2(I, R)            \
  "cmp r2, #" #I "\n"                      \
  "beq 5f\n"                               \
  "vld1.8 { d1[" #R "]}, [%[src_ptr0]]!\n" \
  "vld1.8 { d3[" #R "]}, [%[src_ptr1]]!\n" \

          RUY_LOAD_ONE_ROW2(8, 0)
          RUY_LOAD_ONE_ROW2(9, 1)
          RUY_LOAD_ONE_ROW2(10, 2)
          RUY_LOAD_ONE_ROW2(11, 3)
          RUY_LOAD_ONE_ROW2(12, 4)
          RUY_LOAD_ONE_ROW2(13, 5)
          RUY_LOAD_ONE_ROW2(14, 6)
          RUY_LOAD_ONE_ROW2(15, 7)
#undef RUY_LOAD_ONE_ROW2

          "5:\n"

          "veor.16 q4, q0, q11\n"
          "veor.16 q5, q1, q11\n"

          "vpaddl.s8 q8, q4\n"
          "vpaddl.s8 q9, q5\n"


          // Pairwise add accumulate to 4x32b accumulators.
          "vpadal.s16 q12, q8\n"
          "vpadal.s16 q13, q9\n"

          "vst1.32 {q4}, [%[packed_ptr]]!\n"
          "vst1.32 {q5}, [%[packed_ptr]]!\n"

          "4:\n"

          // Pairwise add 32-bit accumulators
          "vpadd.i32 d24, d24, d25\n"
          "vpadd.i32 d26, d26, d27\n"
          // Final 32-bit values per row
          "vpadd.i32 d25, d24, d26\n"

          "ldr r3, [%[params], #" RUY_STR(RUY_OFFSET_SUMS_PTR) "]\n"
          "cmp r3, #0\n"
          "beq 6f\n"
          "vst1.32 {d25}, [r3]!\n"
          "6:\n"
      // clang-format on

      : [ src_ptr0 ] "+r"(src_ptr0), [ src_ptr1 ] "+r"(src_ptr1)
      : [ src_inc0 ] "r"(src_inc0), [ src_inc1 ] "r"(src_inc1),
        [ packed_ptr ] "r"(packed_ptr), [ params ] "r"(&params)
      : "cc", "memory", "r1", "r2", "r3", "q0", "q1", "q2", "q3",
        "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13");
}

#undef RUY_OFFSET_SRC_PTR0
#undef RUY_OFFSET_SRC_PTR1
#undef RUY_OFFSET_SRC_PTR2
#undef RUY_OFFSET_SRC_PTR32
#undef RUY_OFFSET_SUMS_PTR
#undef RUY_OFFSET_PACKED_PTR0
#undef RUY_OFFSET_SRC_INC0
#undef RUY_OFFSET_SRC_INC1
#undef RUY_OFFSET_SRC_INC2
#undef RUY_OFFSET_SRC_INC3
#undef RUY_OFFSET_SRC_ROWS
#undef RUY_OFFSET_SRC_ZERO_POINT
#undef RUY_OFFSET_INPUT_XOR

#endif  //  RUY_PLATFORM(NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

void Pack8bitNeonInOrder(const void* src_ptr0, const void* src_ptr1,
                         const void* src_ptr2, const void* src_ptr3,
                         int src_inc0, int src_inc1, int src_inc2, int src_inc3,
                         int src_rows, int src_zero_point,
                         std::int8_t* packed_ptr, int start_col, int end_col,
                         std::int32_t* sums_ptr, int input_xor) {
  profiler::ScopeLabel label("Pack (kNeon, optimized for in-order cores)");
  asm volatile(
          // clang-format off
          "dup v26.16b, %w[input_xor]\n"
          "mov w1, #0\n"
          "dup v28.4s, wzr\n"
          "dup v29.4s, wzr\n"
          "dup v30.4s, wzr\n"
          "dup v31.4s, wzr\n"

          "and w2, %w[rows], #-16\n"
          "cmp w1, w2\n"
          "beq 3f\n"
          "ldr x10, [%[src_ptr0], #8]\n"
          "ld1 {v0.8b}, [%[src_ptr0]], %[src_inc0]\n"
          "ldr x11, [%[src_ptr1], #8]\n"
          "ld1 {v1.8b}, [%[src_ptr1]], %[src_inc1]\n"
          "ldr x12, [%[src_ptr2], #8]\n"
          "ld1 {v2.8b}, [%[src_ptr2]], %[src_inc2]\n"
          "ldr x13, [%[src_ptr3], #8]\n"
          "ld1 {v3.8b}, [%[src_ptr3]], %[src_inc3]\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #192]\n")
          "add w1, w1, #16\n"
          "cmp w1, w2\n"

          "beq 2f\n"

          "1:\n"
          "add w1, w1, #16\n"
          "ins v0.d[1], x10\n"
          "ldr x10, [%[src_ptr0], #8]\n"
          "ins v1.d[1], x11\n"
          "ldr x11, [%[src_ptr1], #8]\n"
          "ins v2.d[1], x12\n"
          "ldr x12, [%[src_ptr2], #8]\n"
          "ins v3.d[1], x13\n"
          "ldr x13, [%[src_ptr3], #8]\n"
          "eor v4.16b, v0.16b, v26.16b\n"
          "ld1 {v0.8b}, [%[src_ptr0]], %[src_inc0]\n"
          "eor v5.16b, v1.16b, v26.16b\n"
          "ld1 {v1.8b}, [%[src_ptr1]], %[src_inc1]\n"
          "eor v6.16b, v2.16b, v26.16b\n"
          "ld1 {v2.8b}, [%[src_ptr2]], %[src_inc2]\n"
          "eor v7.16b, v3.16b, v26.16b\n"
          "ld1 {v3.8b}, [%[src_ptr3]], %[src_inc3]\n"
          "saddlp v16.8h, v4.16b\n"
          "str q4, [%[packed_ptr], #0]\n"
          "saddlp v17.8h, v5.16b\n"
          "str q5, [%[packed_ptr], #16]\n"
          "saddlp v18.8h, v6.16b\n"
          "str q6, [%[packed_ptr], #32]\n"
          "saddlp v19.8h, v7.16b\n"
          "str q7, [%[packed_ptr], #48]\n"
          "sadalp v28.4s, v16.8h\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #240]\n")
          "cmp w1, w2\n"
          "sadalp v29.4s, v17.8h\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #240]\n")
          "add %[packed_ptr], %[packed_ptr], #64\n"
          "sadalp v30.4s, v18.8h\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #240]\n")
          "sadalp v31.4s, v19.8h\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #240]\n")

          "bne 1b\n"

          "2:\n"
          "ins v0.d[1], x10\n"
          "ins v1.d[1], x11\n"
          "ins v2.d[1], x12\n"
          "ins v3.d[1], x13\n"
          "eor v4.16b, v0.16b, v26.16b\n"
          "eor v5.16b, v1.16b, v26.16b\n"
          "eor v6.16b, v2.16b, v26.16b\n"
          "eor v7.16b, v3.16b, v26.16b\n"

          "saddlp v16.8h, v4.16b\n"
          "str q4, [%[packed_ptr], #0]\n"
          "saddlp v17.8h, v5.16b\n"
          "str q5, [%[packed_ptr], #16]\n"
          "saddlp v18.8h, v6.16b\n"
          "str q6, [%[packed_ptr], #32]\n"
          "saddlp v19.8h, v7.16b\n"
          "str q7, [%[packed_ptr], #48]\n"
          "sadalp v28.4s, v16.8h\n"
          "sadalp v29.4s, v17.8h\n"
          "sadalp v30.4s, v18.8h\n"
          "sadalp v31.4s, v19.8h\n"

          "add %[packed_ptr], %[packed_ptr], #64\n"

          "3:\n"

          "ands w2, %w[rows], #15\n"
          "beq 4f\n"
          "dup v0.16b, %w[src_zero_point]\n"
          "dup v1.16b, %w[src_zero_point]\n"
          "dup v2.16b, %w[src_zero_point]\n"
          "dup v3.16b, %w[src_zero_point]\n"
#define RUY_LOAD_ONE_ROW(R)                   \
  "cmp w2, #" #R "\n"                         \
  "beq 5f\n"                                  \
  "ld1 { v0.b }[" #R "], [%[src_ptr0]], #1\n" \
  "ld1 { v1.b }[" #R "], [%[src_ptr1]], #1\n" \
  "ld1 { v2.b }[" #R "], [%[src_ptr2]], #1\n" \
  "ld1 { v3.b }[" #R "], [%[src_ptr3]], #1\n"

          RUY_LOAD_ONE_ROW(0)
          RUY_LOAD_ONE_ROW(1)
          RUY_LOAD_ONE_ROW(2)
          RUY_LOAD_ONE_ROW(3)
          RUY_LOAD_ONE_ROW(4)
          RUY_LOAD_ONE_ROW(5)
          RUY_LOAD_ONE_ROW(6)
          RUY_LOAD_ONE_ROW(7)
          RUY_LOAD_ONE_ROW(8)
          RUY_LOAD_ONE_ROW(9)
          RUY_LOAD_ONE_ROW(10)
          RUY_LOAD_ONE_ROW(11)
          RUY_LOAD_ONE_ROW(12)
          RUY_LOAD_ONE_ROW(13)
          RUY_LOAD_ONE_ROW(14)
          RUY_LOAD_ONE_ROW(15)
#undef RUY_LOAD_ONE_ROW
          "5:\n"

          "eor v4.16b, v0.16b, v26.16b\n"
          "eor v5.16b, v1.16b, v26.16b\n"
          "eor v6.16b, v2.16b, v26.16b\n"
          "eor v7.16b, v3.16b, v26.16b\n"

          "saddlp v16.8h, v4.16b\n"
          "saddlp v17.8h, v5.16b\n"
          "saddlp v18.8h, v6.16b\n"
          "saddlp v19.8h, v7.16b\n"
          "sadalp v28.4s, v16.8h\n"
          "sadalp v29.4s, v17.8h\n"
          "sadalp v30.4s, v18.8h\n"
          "sadalp v31.4s, v19.8h\n"

          "str q4, [%[packed_ptr], #0]\n"
          "str q5, [%[packed_ptr], #16]\n"
          "str q6, [%[packed_ptr], #32]\n"
          "str q7, [%[packed_ptr], #48]\n"
          "add %[packed_ptr], %[packed_ptr], #64\n"

          "4:\n"

          "addp v28.4s, v28.4s, v29.4s\n"
          "addp v30.4s, v30.4s, v31.4s\n"
          "addp v28.4s, v28.4s, v30.4s\n"

          "cmp %[sums_ptr], #0\n"
          "beq 6f\n"
          "st1 {v28.4s}, [%[sums_ptr]], #16\n"
          "6:\n"
          // clang-format on

          : [ src_ptr0 ] "+r"(src_ptr0), [ src_ptr1 ] "+r"(src_ptr1),
            [ src_ptr2 ] "+r"(src_ptr2), [ src_ptr3 ] "+r"(src_ptr3),
            [ packed_ptr ] "+r"(packed_ptr), [ sums_ptr ] "+r"(sums_ptr)
          : [ src_inc0 ] "r"(static_cast<std::int64_t>(src_inc0)), [ src_inc1 ] "r"(static_cast<std::int64_t>(src_inc1)),
            [ src_inc2 ] "r"(static_cast<std::int64_t>(src_inc2)), [ src_inc3 ] "r"(static_cast<std::int64_t>(src_inc3)),
            [ rows ] "r"(src_rows),
            [ src_zero_point ] "r"(src_zero_point),
            [input_xor] "r"(input_xor)
          : "cc", "memory", "x1", "x2", "x10", "x11", "x12", "x13", "v0", "v1", "v2", "v3", "v4", "v5",
            "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
            "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
            "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

void Pack8bitNeonDotprodInOrder(const void* src_ptr0, const void* src_ptr1,
                                const void* src_ptr2, const void* src_ptr3,
                                int src_inc0, int src_inc1, int src_inc2,
                                int src_inc3, int src_rows, int src_zero_point,
                                std::int8_t* packed_ptr, int start_col,
                                int end_col, std::int32_t* sums_ptr,
                                int input_xor) {
  profiler::ScopeLabel label(
      "Pack (kNeonDotprod, optimized for in-order cores)");
  asm volatile(
          // clang-format off
          "dup v26.16b, %w[input_xor]\n"
          "mov w1, #1\n"
          "dup v27.16b, w1\n"
          "mov w1, #0\n"
          "dup v28.4s, wzr\n"
          "dup v29.4s, wzr\n"
          "dup v30.4s, wzr\n"
          "dup v31.4s, wzr\n"

          "and w2, %w[rows], #-16\n"
          "cmp w1, w2\n"
          "beq 3f\n"
          "ldr x10, [%[src_ptr0], #8]\n"
          "ld1 {v0.8b}, [%[src_ptr0]], %[src_inc0]\n"
          "ldr x11, [%[src_ptr1], #8]\n"
          "ld1 {v1.8b}, [%[src_ptr1]], %[src_inc1]\n"
          "ldr x12, [%[src_ptr2], #8]\n"
          "ld1 {v2.8b}, [%[src_ptr2]], %[src_inc2]\n"
          "ldr x13, [%[src_ptr3], #8]\n"
          "ld1 {v3.8b}, [%[src_ptr3]], %[src_inc3]\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #192]\n")
          "add w1, w1, #16\n"
          "cmp w1, w2\n"

          "beq 2f\n"

          "1:\n"
          "add w1, w1, #16\n"
          "ins v0.d[1], x10\n"
          "ldr x10, [%[src_ptr0], #8]\n"
          "ins v1.d[1], x11\n"
          "ldr x11, [%[src_ptr1], #8]\n"
          "ins v2.d[1], x12\n"
          "ldr x12, [%[src_ptr2], #8]\n"
          "ins v3.d[1], x13\n"
          "ldr x13, [%[src_ptr3], #8]\n"

          "eor v4.16b, v0.16b, v26.16b\n"
          "ld1 {v0.8b}, [%[src_ptr0]], %[src_inc0]\n"
          "eor v5.16b, v1.16b, v26.16b\n"
          "ld1 {v1.8b}, [%[src_ptr1]], %[src_inc1]\n"
          "eor v6.16b, v2.16b, v26.16b\n"
          "ld1 {v2.8b}, [%[src_ptr2]], %[src_inc2]\n"
          "eor v7.16b, v3.16b, v26.16b\n"
          "ld1 {v3.8b}, [%[src_ptr3]], %[src_inc3]\n"

          "trn1 v16.4s, v4.4s, v5.4s\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #240]\n")
          "trn2 v17.4s, v4.4s, v5.4s\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #240]\n")
          "trn1 v18.4s, v6.4s, v7.4s\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #240]\n")
          "trn2 v19.4s, v6.4s, v7.4s\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #240]\n")

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"
          "cmp w1, w2\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          "str q20, [%[packed_ptr], #0]\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          "str q21, [%[packed_ptr], #32]\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          "str q22, [%[packed_ptr], #64]\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"
          "str q23, [%[packed_ptr], #96]\n"

          "add %[packed_ptr], %[packed_ptr], #128\n"

          "bne 1b\n"

          "2:\n"
          "ins v0.d[1], x10\n"
          "ins v1.d[1], x11\n"
          "ins v2.d[1], x12\n"
          "ins v3.d[1], x13\n"
          "eor v0.16b, v0.16b, v26.16b\n"
          "eor v1.16b, v1.16b, v26.16b\n"
          "eor v2.16b, v2.16b, v26.16b\n"
          "eor v3.16b, v3.16b, v26.16b\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          "str q20, [%[packed_ptr], #0]\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          "str q21, [%[packed_ptr], #32]\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          "str q22, [%[packed_ptr], #64]\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "3:\n"

          "ands w2, %w[rows], #15\n"
          "beq 4f\n"
          "dup v0.16b, %w[src_zero_point]\n"
          "dup v1.16b, %w[src_zero_point]\n"
          "dup v2.16b, %w[src_zero_point]\n"
          "dup v3.16b, %w[src_zero_point]\n"
#define RUY_LOAD_ONE_ROW(R)                   \
  "cmp w2, #" #R "\n"                         \
  "beq 5f\n"                                  \
  "ld1 { v0.b }[" #R "], [%[src_ptr0]], #1\n" \
  "ld1 { v1.b }[" #R "], [%[src_ptr1]], #1\n" \
  "ld1 { v2.b }[" #R "], [%[src_ptr2]], #1\n" \
  "ld1 { v3.b }[" #R "], [%[src_ptr3]], #1\n"

          RUY_LOAD_ONE_ROW(0)
          RUY_LOAD_ONE_ROW(1)
          RUY_LOAD_ONE_ROW(2)
          RUY_LOAD_ONE_ROW(3)
          RUY_LOAD_ONE_ROW(4)
          RUY_LOAD_ONE_ROW(5)
          RUY_LOAD_ONE_ROW(6)
          RUY_LOAD_ONE_ROW(7)
          RUY_LOAD_ONE_ROW(8)
          RUY_LOAD_ONE_ROW(9)
          RUY_LOAD_ONE_ROW(10)
          RUY_LOAD_ONE_ROW(11)
          RUY_LOAD_ONE_ROW(12)
          RUY_LOAD_ONE_ROW(13)
          RUY_LOAD_ONE_ROW(14)
          RUY_LOAD_ONE_ROW(15)
#undef RUY_LOAD_ONE_ROW
          "5:\n"

          "eor v0.16b, v0.16b, v26.16b\n"
          "eor v1.16b, v1.16b, v26.16b\n"
          "eor v2.16b, v2.16b, v26.16b\n"
          "eor v3.16b, v3.16b, v26.16b\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          "str q20, [%[packed_ptr], #0]\n"
          "cmp w2, #4\n"
          "ble 4f\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          "str q21, [%[packed_ptr], #32]\n"
          "cmp w2, #8\n"
          "ble 4f\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          "str q22, [%[packed_ptr], #64]\n"
          "cmp w2, #12\n"
          "ble 4f\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "4:\n"

          "add v28.4s, v28.4s, v29.4s\n"
          "add v30.4s, v30.4s, v31.4s\n"
          "add v28.4s, v28.4s, v30.4s\n"

          "cmp %[sums_ptr], #0\n"
          "beq 6f\n"
          "st1 {v28.4s}, [%[sums_ptr]], #16\n"
          "6:\n"
          // clang-format on

          : [ src_ptr0 ] "+r"(src_ptr0), [src_ptr1] "+r"(src_ptr1), [src_ptr2] "+r"(src_ptr2),
            [src_ptr3] "+r"(src_ptr3), [packed_ptr] "+r"(packed_ptr), [sums_ptr] "+r"(sums_ptr)
          : [ src_inc0 ] "r"(static_cast<std::int64_t>(src_inc0)), [ src_inc1 ] "r"(static_cast<std::int64_t>(src_inc1)),
            [ src_inc2 ] "r"(static_cast<std::int64_t>(src_inc2)), [ src_inc3 ] "r"(static_cast<std::int64_t>(src_inc3)),
                [rows] "r"(src_rows),
            [src_zero_point] "r"(static_cast<int>(src_zero_point)),
            [input_xor] "r"(input_xor)
          : "cc", "memory", "x1", "x2", "x10", "x11", "x12", "x13", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
            "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

void Pack8bitNeonDotprodOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                                   const void* src_ptr2, const void* src_ptr3,
                                   int src_inc0, int src_inc1, int src_inc2,
                                   int src_inc3, int src_rows,
                                   int src_zero_point, std::int8_t* packed_ptr,
                                   int start_col, int end_col,
                                   std::int32_t* sums_ptr, int input_xor) {
  profiler::ScopeLabel label(
      "Pack (kNeonDotprod, optimized for out-of-order cores)");
  asm volatile(
      // clang-format off
          "dup v26.16b, %w[input_xor]\n"
          "mov w1, #1\n"
          "dup v27.16b, w1\n"
          "mov w1, #0\n"
          "dup v28.4s, wzr\n"
          "dup v29.4s, wzr\n"
          "dup v30.4s, wzr\n"
          "dup v31.4s, wzr\n"

#if RUY_OPT_ENABLED(RUY_OPT_MAX_STREAMING)
          "and w2, %w[rows], #-64\n"
          "cmp w1, w2\n"
          "beq 9f\n"

          "ld1 {v0.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v1.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v2.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v3.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "ld1 {v4.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v5.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v6.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v7.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "ld1 {v8.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v9.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v10.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v11.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "ld1 {v12.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v13.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v14.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v15.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "add w1, w1, #64\n"
          "cmp w1, w2\n"
          "beq 8f\n"

          "7:\n"
          "eor v0.16b, v0.16b, v26.16b\n"
          "eor v1.16b, v1.16b, v26.16b\n"
          "eor v2.16b, v2.16b, v26.16b\n"
          "eor v3.16b, v3.16b, v26.16b\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "ld1 {v0.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v1.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v2.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v3.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "add w1, w1, #16\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "eor v4.16b, v4.16b, v26.16b\n"
          "eor v5.16b, v5.16b, v26.16b\n"
          "eor v6.16b, v6.16b, v26.16b\n"
          "eor v7.16b, v7.16b, v26.16b\n"

          "trn1 v16.4s, v4.4s, v5.4s\n"
          "trn2 v17.4s, v4.4s, v5.4s\n"
          "trn1 v18.4s, v6.4s, v7.4s\n"
          "trn2 v19.4s, v6.4s, v7.4s\n"

          "ld1 {v4.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v5.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v6.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v7.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "add w1, w1, #16\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "eor v8.16b, v8.16b, v26.16b\n"
          "eor v9.16b, v9.16b, v26.16b\n"
          "eor v10.16b, v10.16b, v26.16b\n"
          "eor v11.16b, v11.16b, v26.16b\n"

          "trn1 v16.4s, v8.4s, v9.4s\n"
          "trn2 v17.4s, v8.4s, v9.4s\n"
          "trn1 v18.4s, v10.4s, v11.4s\n"
          "trn2 v19.4s, v10.4s, v11.4s\n"

          "ld1 {v8.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v9.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v10.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v11.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "add w1, w1, #16\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "eor v12.16b, v12.16b, v26.16b\n"
          "eor v13.16b, v13.16b, v26.16b\n"
          "eor v14.16b, v14.16b, v26.16b\n"
          "eor v15.16b, v15.16b, v26.16b\n"

          "trn1 v16.4s, v12.4s, v13.4s\n"
          "trn2 v17.4s, v12.4s, v13.4s\n"
          "trn1 v18.4s, v14.4s, v15.4s\n"
          "trn2 v19.4s, v14.4s, v15.4s\n"

          "ld1 {v12.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v13.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v14.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v15.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "add w1, w1, #16\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "cmp w1, w2\n"
          "bne 7b\n"

          "8:\n"

          "eor v0.16b, v0.16b, v26.16b\n"
          "eor v1.16b, v1.16b, v26.16b\n"
          "eor v2.16b, v2.16b, v26.16b\n"
          "eor v3.16b, v3.16b, v26.16b\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "eor v4.16b, v4.16b, v26.16b\n"
          "eor v5.16b, v5.16b, v26.16b\n"
          "eor v6.16b, v6.16b, v26.16b\n"
          "eor v7.16b, v7.16b, v26.16b\n"

          "trn1 v16.4s, v4.4s, v5.4s\n"
          "trn2 v17.4s, v4.4s, v5.4s\n"
          "trn1 v18.4s, v6.4s, v7.4s\n"
          "trn2 v19.4s, v6.4s, v7.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "eor v8.16b, v8.16b, v26.16b\n"
          "eor v9.16b, v9.16b, v26.16b\n"
          "eor v10.16b, v10.16b, v26.16b\n"
          "eor v11.16b, v11.16b, v26.16b\n"

          "trn1 v16.4s, v8.4s, v9.4s\n"
          "trn2 v17.4s, v8.4s, v9.4s\n"
          "trn1 v18.4s, v10.4s, v11.4s\n"
          "trn2 v19.4s, v10.4s, v11.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "eor v12.16b, v12.16b, v26.16b\n"
          "eor v13.16b, v13.16b, v26.16b\n"
          "eor v14.16b, v14.16b, v26.16b\n"
          "eor v15.16b, v15.16b, v26.16b\n"

          "trn1 v16.4s, v12.4s, v13.4s\n"
          "trn2 v17.4s, v12.4s, v13.4s\n"
          "trn1 v18.4s, v14.4s, v15.4s\n"
          "trn2 v19.4s, v14.4s, v15.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "9:\n"
#endif  // #if RUY_OPT_ENABLED(RUY_OPT_MAX_STREAMING)
          "and w2, %w[rows], #-16\n"
          "cmp w1, w2\n"
          "beq 3f\n"

          "ld1 {v0.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v1.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v2.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v3.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "add w1, w1, #16\n"
          "cmp w1, w2\n"
          "beq 2f\n"

          "1:\n"

          "eor v0.16b, v0.16b, v26.16b\n"
          "eor v1.16b, v1.16b, v26.16b\n"
          "eor v2.16b, v2.16b, v26.16b\n"
          "eor v3.16b, v3.16b, v26.16b\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "ld1 {v0.16b}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v1.16b}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v2.16b}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v3.16b}, [%[src_ptr3]], %[src_inc3]\n"
          "add w1, w1, #16\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "cmp w1, w2\n"
          "bne 1b\n"

          "2:\n"

          "eor v0.16b, v0.16b, v26.16b\n"
          "eor v1.16b, v1.16b, v26.16b\n"
          "eor v2.16b, v2.16b, v26.16b\n"
          "eor v3.16b, v3.16b, v26.16b\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "3:\n"

          "ands w2, %w[rows], #15\n"
          "beq 4f\n"
          "dup v0.16b, %w[src_zero_point]\n"
          "dup v1.16b, %w[src_zero_point]\n"
          "dup v2.16b, %w[src_zero_point]\n"
          "dup v3.16b, %w[src_zero_point]\n"
#define RUY_LOAD_ONE_ROW(R)                   \
  "cmp w2, #" #R "\n"                         \
  "beq 5f\n"                                  \
  "ld1 { v0.b }[" #R "], [%[src_ptr0]], #1\n" \
  "ld1 { v1.b }[" #R "], [%[src_ptr1]], #1\n" \
  "ld1 { v2.b }[" #R "], [%[src_ptr2]], #1\n" \
  "ld1 { v3.b }[" #R "], [%[src_ptr3]], #1\n"

          RUY_LOAD_ONE_ROW(0)
          RUY_LOAD_ONE_ROW(1)
          RUY_LOAD_ONE_ROW(2)
          RUY_LOAD_ONE_ROW(3)
          RUY_LOAD_ONE_ROW(4)
          RUY_LOAD_ONE_ROW(5)
          RUY_LOAD_ONE_ROW(6)
          RUY_LOAD_ONE_ROW(7)
          RUY_LOAD_ONE_ROW(8)
          RUY_LOAD_ONE_ROW(9)
          RUY_LOAD_ONE_ROW(10)
          RUY_LOAD_ONE_ROW(11)
          RUY_LOAD_ONE_ROW(12)
          RUY_LOAD_ONE_ROW(13)
          RUY_LOAD_ONE_ROW(14)
          RUY_LOAD_ONE_ROW(15)
#undef RUY_LOAD_ONE_ROW
          "5:\n"

          "eor v0.16b, v0.16b, v26.16b\n"
          "eor v1.16b, v1.16b, v26.16b\n"
          "eor v2.16b, v2.16b, v26.16b\n"
          "eor v3.16b, v3.16b, v26.16b\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          ".word 0x4e9b969c  // sdot v28.4s, v20.16b, v27.16b\n"
          "str q20, [%[packed_ptr], #0]\n"
          "cmp w2, #4\n"
          "ble 4f\n"
          ".word 0x4e9b96be  // sdot v30.4s, v21.16b, v27.16b\n"
          "str q21, [%[packed_ptr], #32]\n"
          "cmp w2, #8\n"
          "ble 4f\n"
          ".word 0x4e9b96dd  // sdot v29.4s, v22.16b, v27.16b\n"
          "str q22, [%[packed_ptr], #64]\n"
          "cmp w2, #12\n"
          "ble 4f\n"
          ".word 0x4e9b96ff  // sdot v31.4s, v23.16b, v27.16b\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "4:\n"

          "add v28.4s, v28.4s, v29.4s\n"
          "add v30.4s, v30.4s, v31.4s\n"
          "add v28.4s, v28.4s, v30.4s\n"

          "cmp %[sums_ptr], #0\n"
          "beq 6f\n"
          "st1 {v28.4s}, [%[sums_ptr]], #16\n"
          "6:\n"
      // clang-format on

      : [ src_ptr0 ] "+r"(src_ptr0), [ src_ptr1 ] "+r"(src_ptr1),
        [ src_ptr2 ] "+r"(src_ptr2), [ src_ptr3 ] "+r"(src_ptr3),
        [ packed_ptr ] "+r"(packed_ptr), [ sums_ptr ] "+r"(sums_ptr)
      : [ src_inc0 ] "r"(static_cast<std::int64_t>(src_inc0)),
        [ src_inc1 ] "r"(static_cast<std::int64_t>(src_inc1)),
        [ src_inc2 ] "r"(static_cast<std::int64_t>(src_inc2)),
        [ src_inc3 ] "r"(static_cast<std::int64_t>(src_inc3)),
        [ rows ] "r"(src_rows),
        [ src_zero_point ] "r"(static_cast<int>(src_zero_point)),
        [ input_xor ] "r"(input_xor)
      : "cc", "memory", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
}

#endif  // RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void PackFloatNeonOutOfOrder(const float* src_ptr0, const float* src_ptr1,
                             const float* src_ptr2, const float* src_ptr3,
                             int src_inc0, int src_inc1, int src_inc2,
                             int src_inc3, int src_rows, int src_zero_point,
                             float* packed_ptr, int start_col, int end_col) {
  profiler::ScopeLabel label("Pack (kNeon, optimized for out-of-order cores)");
  asm volatile(
      // clang-format off
          "mov w1, #0\n"

          "and w2, %w[rows], #-4\n"
          "cmp w1, w2\n"
          "beq 3f\n"
          "ld1 {v0.4s}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v1.4s}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v2.4s}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v3.4s}, [%[src_ptr3]], %[src_inc3]\n"
          "add w1, w1, #4\n"
          "cmp w1, w2\n"

          "beq 2f\n"

          "1:\n"
          "add w1, w1, #4\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "ld1 {v0.4s}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v1.4s}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v2.4s}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v3.4s}, [%[src_ptr3]], %[src_inc3]\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"
          "cmp w1, w2\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"

          "add %[packed_ptr], %[packed_ptr], #128\n"

          "bne 1b\n"

          "2:\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "3:\n"

          "ands w2, %w[rows], #3\n"
          "beq 4f\n"
          "dup v0.16b, wzr\n"
          "dup v1.16b, wzr\n"
          "dup v2.16b, wzr\n"
          "dup v3.16b, wzr\n"
#define RUY_LOAD_ONE_ROW(R)                   \
  "cmp w2, #" #R "\n"                         \
  "beq 5f\n"                                  \
  "ld1 { v0.s }[" #R "], [%[src_ptr0]], #4\n" \
  "ld1 { v1.s }[" #R "], [%[src_ptr1]], #4\n" \
  "ld1 { v2.s }[" #R "], [%[src_ptr2]], #4\n" \
  "ld1 { v3.s }[" #R "], [%[src_ptr3]], #4\n"

          RUY_LOAD_ONE_ROW(0)
          RUY_LOAD_ONE_ROW(1)
          RUY_LOAD_ONE_ROW(2)
          RUY_LOAD_ONE_ROW(3)
#undef RUY_LOAD_ONE_ROW
          "5:\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          "mov x1, #32\n"

#define RUY_STORE_ONE_ROW(ROW, REGISTER)                  \
          "cmp w2, #" #ROW "\n"                           \
          "beq 4f\n"                                      \
          "st1 {" #REGISTER ".4s}, [%[packed_ptr]], x1\n"

          RUY_STORE_ONE_ROW(0, v20)
          RUY_STORE_ONE_ROW(1, v21)
          RUY_STORE_ONE_ROW(2, v22)
          RUY_STORE_ONE_ROW(3, v23)

#undef RUY_STORE_ONE_ROW

          "4:\n"

      // clang-format on

      : [ src_ptr0 ] "+r"(src_ptr0), [ src_ptr1 ] "+r"(src_ptr1),
        [ src_ptr2 ] "+r"(src_ptr2), [ src_ptr3 ] "+r"(src_ptr3),
        [ packed_ptr ] "+r"(packed_ptr)
      : [ src_inc0 ] "r"(static_cast<std::int64_t>(src_inc0)),
        [ src_inc1 ] "r"(static_cast<std::int64_t>(src_inc1)),
        [ src_inc2 ] "r"(static_cast<std::int64_t>(src_inc2)),
        [ src_inc3 ] "r"(static_cast<std::int64_t>(src_inc3)),
        [ rows ] "r"(src_rows)
      : "cc", "memory", "x1", "x2", "x10", "x11", "x12", "x13", "v0", "v1",
        "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
        "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
#endif

#if RUY_PLATFORM(NEON_32) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void PackFloatNeonOutOfOrder(const float* src_ptr0, const float* src_ptr1,
                             const float* src_ptr2, const float* src_ptr3,
                             int src_inc, int src_rows, int src_zero_point,
                             float* packed_ptr, int start_col, int end_col,
                             int output_stride) {
  profiler::ScopeLabel label("Pack (kNeon, optimized for out-of-order cores)");
  asm volatile(
      // clang-format off
          "mov r1, #0\n"
          "and r2, %[rows], #-4\n"
          "cmp r1, r2\n"
          "beq 3f\n"
#define RUY_LOAD_FOUR_BY_FOUR()               \
  /* Load q0 */                               \
  "vld1.32 {d0, d1}, [%[src_ptr0]]\n"         \
  /* if src_inc0 != 0, add 16 to src_ptr0 */  \
  "and r3, %[src_inc], #1\n"                  \
  "add %[src_ptr0], %[src_ptr0], r3, lsl #4\n"\
  /* Load q1 */                               \
  "vld1.32 {d2, d3}, [%[src_ptr1]]\n"         \
  /* if src_inc1 != 0, add 16 to src_ptr0 */  \
  "and r3, %[src_inc], #2\n"                  \
  "add %[src_ptr1], %[src_ptr1], r3, lsl #3\n"\
  /* Load q2 */                               \
  "vld1.32 {d4, d5}, [%[src_ptr2]]\n"         \
  /* if src_inc2 != 0, add 16 to src_ptr0 */  \
  "and r3, %[src_inc], #4\n"                  \
  "add %[src_ptr2], %[src_ptr2], r3, lsl #2\n"\
  /* Load q3 */                               \
  "vld1.32 {d6, d7}, [%[src_ptr3]]\n"         \
  /* if src_inc3 != 0, add 16 to src_ptr0 */  \
  "and r3, %[src_inc], #8\n"                  \
  "add %[src_ptr3], %[src_ptr3], r3, lsl #1\n"\

          RUY_LOAD_FOUR_BY_FOUR()
          "add r1, r1, #4\n"
          "cmp r1, r2\n"

          "beq 2f\n"

          "1:\n"
          "add r1, r1, #4\n"

          // Transpose 4x4 matrix.
          "vzip.32 q0, q1\n"
          "vzip.32 q2, q3\n"

          "vtrn.32 q0, q2\n"
          "vtrn.32 q1, q3\n"

          "vzip.32 q0, q2\n"
          "vzip.32 q1, q3\n"

          "vmov q8, q0\n"
          "vmov q9, q1\n"
          "vmov q10, q2\n"
          "vmov q11, q3\n"

          RUY_LOAD_FOUR_BY_FOUR()
#undef RUY_LOAD_FOUR_BY_FOUR

#define RUY_STORE_FOUR_BY_FOUR()                  \
  /* Store q8, q10, q9, q11 */                    \
  /* q8 = d16, d17 */                             \
  "vst1.32 {d16, d17}, [%[packed_ptr]]\n"         \
  /* q10 = d20, d21 */                            \
  "add %[packed_ptr], %[packed_ptr], %[stride]\n" \
  "vst1.32 {d20, d21}, [%[packed_ptr]]\n"         \
  /* q9 = d18, d19 */                             \
  "add %[packed_ptr], %[packed_ptr], %[stride]\n" \
  "vst1.32 {d18, d19}, [%[packed_ptr]]\n"         \
  /* q11 = d22, d23 */                            \
  "add %[packed_ptr], %[packed_ptr], %[stride]\n" \
  "vst1.32 {d22, d23}, [%[packed_ptr]]\n"         \
  "add %[packed_ptr], %[packed_ptr], %[stride]\n" \

          RUY_STORE_FOUR_BY_FOUR()
          "cmp r1, r2\n"

          "bne 1b\n"

          "2:\n"

          // Transpose 4x4 matrix.
          "vzip.32 q0, q1\n"
          "vzip.32 q2, q3\n"

          "vtrn.32 q0, q2\n"
          "vtrn.32 q1, q3\n"

          "vzip.32 q0, q2\n"
          "vzip.32 q1, q3\n"

          "vmov q8, q0\n"
          "vmov q9, q1\n"
          "vmov q10, q2\n"
          "vmov q11, q3\n"

          RUY_STORE_FOUR_BY_FOUR()
#undef RUY_STORE_FOUR_BY_FOUR
          "3:\n"

          "ands r2, %[rows], #3\n"
          "beq 4f\n"
          "mov r0, #0\n"
          // Zero out q0 - q3
          "vdup.32 q0, r0\n"
          "vdup.32 q1, r0\n"
          "vdup.32 q2, r0\n"
          "vdup.32 q3, r0\n"
#define RUY_LOAD_ONE_ROW_FIRST_HALF(R, I)    \
  "cmp r2, #" #R "\n"                        \
  "beq 5f\n"                                 \
  "vld1.32 { d0[" #I "] }, [%[src_ptr0]]!\n" \
  "vld1.32 { d2[" #I "] }, [%[src_ptr1]]!\n" \
  "vld1.32 { d4[" #I "] }, [%[src_ptr2]]!\n" \
  "vld1.32 { d6[" #I "] }, [%[src_ptr3]]!\n"

#define RUY_LOAD_ONE_ROW_SECOND_HALF(R, I)      \
  "vld1.32 { d1[" #I "] }, [%[src_ptr0]]!\n" \
  "vld1.32 { d3[" #I "] }, [%[src_ptr1]]!\n" \
  "vld1.32 { d5[" #I "] }, [%[src_ptr2]]!\n" \
  "vld1.32 { d7[" #I "] }, [%[src_ptr3]]!\n"

          RUY_LOAD_ONE_ROW_FIRST_HALF(0, 0)
          RUY_LOAD_ONE_ROW_FIRST_HALF(1, 1)
          RUY_LOAD_ONE_ROW_SECOND_HALF(2, 0)
          RUY_LOAD_ONE_ROW_SECOND_HALF(3, 1)
#undef RUY_LOAD_ONE_ROW_SECOND_HALF
#undef RUY_LOAD_ONE_ROW_FIRST_HALF
          "5:\n"

          // Transpose 4x4 matrix.
          "vzip.32 q0, q1\n"
          "vzip.32 q2, q3\n"

          "vtrn.32 q0, q2\n"
          "vtrn.32 q1, q3\n"

          "vzip.32 q0, q2\n"
          "vzip.32 q1, q3\n"

          "vmov q8, q0\n"
          "vmov q9, q1\n"
          "vmov q10, q2\n"
          "vmov q11, q3\n"

          "mov r1, #32\n"

#define RUY_STORE_ONE_ROW(ROW, REGISTER)      \
          "cmp r2, #" #ROW "\n"                           \
          "beq 4f\n"                                      \
          "vst1.32 {" #REGISTER "}, [%[packed_ptr]]\n"    \
          "add %[packed_ptr], %[packed_ptr], %[stride]\n"

          // Store q8
          RUY_STORE_ONE_ROW(0, q8)
          // Store q10
          RUY_STORE_ONE_ROW(1, q10)
          // Store q9
          RUY_STORE_ONE_ROW(2, q9)
          // Store q11
          RUY_STORE_ONE_ROW(3, q11)

#undef RUY_STORE_ONE_ROW

          "4:\n"

      // clang-format on
      : [ src_ptr0 ] "+r"(src_ptr0), [ src_ptr1 ] "+r"(src_ptr1),
        [ src_ptr2 ] "+r"(src_ptr2), [ src_ptr3 ] "+r"(src_ptr3),
        [ packed_ptr ] "+r"(packed_ptr)
      : [ src_inc ] "r"(static_cast<std::int64_t>(src_inc)),
        [ rows ] "r"(src_rows), [ stride ] "r"(output_stride)
      : "cc", "memory", "r0", "r1", "r2", "r3", "q0", "q1", "q2", "q3",
        "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
}

#endif  // (RUY_PLATFORM(NEON_32)

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void PackFloatNeonInOrder(const float* src_ptr0, const float* src_ptr1,
                          const float* src_ptr2, const float* src_ptr3,
                          int src_inc0, int src_inc1, int src_inc2,
                          int src_inc3, int src_rows, int src_zero_point,
                          float* packed_ptr, int start_col, int end_col) {
  profiler::ScopeLabel label("Pack (kNeon, optimized for in-order cores)");

  asm volatile(
          // clang-format off
          "mov w1, #0\n"

          "and w2, %w[rows], #-4\n"
          "cmp w1, w2\n"
          "beq 3f\n"
          "ld1 {v0.4s}, [%[src_ptr0]], %[src_inc0]\n"
          "ld1 {v1.4s}, [%[src_ptr1]], %[src_inc1]\n"
          "ld1 {v2.4s}, [%[src_ptr2]], %[src_inc2]\n"
          "ld1 {v3.4s}, [%[src_ptr3]], %[src_inc3]\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #64]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #128]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #192]\n")
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #192]\n")
          "add w1, w1, #4\n"
          "cmp w1, w2\n"

          "beq 2f\n"

          "1:\n"
          "add w1, w1, #4\n"

          "ldr x10, [%[src_ptr0], #8]\n"
          "trn1 v16.4s, v0.4s, v1.4s\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr0], #240]\n")
          "ldr x11, [%[src_ptr1], #8]\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr1], #240]\n")
          "ldr x12, [%[src_ptr2], #8]\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr2], #240]\n")
          "ldr x13, [%[src_ptr3], #8]\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"
          RUY_PREFETCH("prfm pldl1strm, [%[src_ptr3], #240]\n")

          "ld1 {v0.2s}, [%[src_ptr0]], %[src_inc0]\n"
          "trn1 v20.2d, v16.2d, v18.2d\n"
          "ld1 {v1.2s}, [%[src_ptr1]], %[src_inc1]\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "ld1 {v2.2s}, [%[src_ptr2]], %[src_inc2]\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "ld1 {v3.2s}, [%[src_ptr3]], %[src_inc3]\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"
          "cmp w1, w2\n"

          "ins v0.d[1], x10\n"
          "str q20, [%[packed_ptr], #0]\n"
          "ins v1.d[1], x11\n"
          "str q21, [%[packed_ptr], #32]\n"
          "ins v2.d[1], x12\n"
          "str q22, [%[packed_ptr], #64]\n"
          "ins v3.d[1], x13\n"
          "str q23, [%[packed_ptr], #96]\n"

          "add %[packed_ptr], %[packed_ptr], #128\n"

          "bne 1b\n"

          "2:\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          "str q20, [%[packed_ptr], #0]\n"
          "str q21, [%[packed_ptr], #32]\n"
          "str q22, [%[packed_ptr], #64]\n"
          "str q23, [%[packed_ptr], #96]\n"
          "add %[packed_ptr], %[packed_ptr], #128\n"

          "3:\n"

          "ands w2, %w[rows], #3\n"
          "beq 4f\n"
          "dup v0.16b, wzr\n"
          "dup v1.16b, wzr\n"
          "dup v2.16b, wzr\n"
          "dup v3.16b, wzr\n"
#define RUY_LOAD_ONE_ROW(R)                   \
  "cmp w2, #" #R "\n"                         \
  "beq 5f\n"                                  \
  "ld1 { v0.s }[" #R "], [%[src_ptr0]], #4\n" \
  "ld1 { v1.s }[" #R "], [%[src_ptr1]], #4\n" \
  "ld1 { v2.s }[" #R "], [%[src_ptr2]], #4\n" \
  "ld1 { v3.s }[" #R "], [%[src_ptr3]], #4\n"

          RUY_LOAD_ONE_ROW(0)
          RUY_LOAD_ONE_ROW(1)
          RUY_LOAD_ONE_ROW(2)
          RUY_LOAD_ONE_ROW(3)
#undef RUY_LOAD_ONE_ROW
          "5:\n"

          "trn1 v16.4s, v0.4s, v1.4s\n"
          "trn2 v17.4s, v0.4s, v1.4s\n"
          "trn1 v18.4s, v2.4s, v3.4s\n"
          "trn2 v19.4s, v2.4s, v3.4s\n"

          "trn1 v20.2d, v16.2d, v18.2d\n"
          "trn2 v22.2d, v16.2d, v18.2d\n"
          "trn1 v21.2d, v17.2d, v19.2d\n"
          "trn2 v23.2d, v17.2d, v19.2d\n"

          "mov x1, #32\n"

#define RUY_STORE_ONE_ROW(ROW, REGISTER)                  \
          "cmp w2, #" #ROW "\n"                           \
          "beq 4f\n"                                      \
          "st1 {" #REGISTER ".4s}, [%[packed_ptr]], x1\n"

          RUY_STORE_ONE_ROW(0, v20)
          RUY_STORE_ONE_ROW(1, v21)
          RUY_STORE_ONE_ROW(2, v22)
          RUY_STORE_ONE_ROW(3, v23)

#undef RUY_STORE_ONE_ROW

          "4:\n"

          // clang-format on

          : [ src_ptr0 ] "+r"(src_ptr0), [src_ptr1] "+r"(src_ptr1), [src_ptr2] "+r"(src_ptr2),
            [src_ptr3] "+r"(src_ptr3), [packed_ptr] "+r"(packed_ptr)
          : [ src_inc0 ] "r"(static_cast<std::int64_t>(src_inc0)), [src_inc1] "r"(static_cast<std::int64_t>(src_inc1)), [src_inc2] "r"(static_cast<std::int64_t>(src_inc2)),
            [src_inc3] "r"(static_cast<std::int64_t>(src_inc3)), [rows] "r"(src_rows)
          : "cc", "memory", "x1", "x2", "x10", "x11", "x12", "x13", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
            "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}
#endif  // RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy
