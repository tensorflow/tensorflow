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

#include "tensorflow/lite/experimental/ruy/pack.h"

#include "tensorflow/lite/experimental/ruy/platform.h"

namespace ruy {

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)

void Pack8bitNeonOutOfOrder(const void* src_ptr0, const void* src_ptr1,
                            const void* src_ptr2, const void* src_ptr3,
                            int src_inc0, int src_inc1, int src_inc2,
                            int src_inc3, int src_rows, int src_zero_point,
                            std::int8_t* packed_ptr, int start_col, int end_col,
                            std::int32_t* sums_ptr, int input_xor) {
  gemmlowp::ScopedProfilingLabel label(
      "Pack (kNeon, optimized for out-of-order cores)");
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

void Pack8bitNeonInOrder(const void* src_ptr0, const void* src_ptr1,
                         const void* src_ptr2, const void* src_ptr3,
                         int src_inc0, int src_inc1, int src_inc2, int src_inc3,
                         int src_rows, int src_zero_point,
                         std::int8_t* packed_ptr, int start_col, int end_col,
                         std::int32_t* sums_ptr, int input_xor) {
  gemmlowp::ScopedProfilingLabel label(
      "Pack (kNeon, optimized for in-order cores)");
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
  gemmlowp::ScopedProfilingLabel label(
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
  gemmlowp::ScopedProfilingLabel label(
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
  gemmlowp::ScopedProfilingLabel label(
      "Pack (kNeon, optimized for out-of-order cores)");
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
  gemmlowp::ScopedProfilingLabel label(
      "Pack (kNeon, optimized for out-of-order cores)");
  asm volatile(
      // clang-format off
          "mov r1, #0\n"
          "and r2, %[rows], #-4\n"
          "cmp r1, r2\n"
          "beq 3f\n"
#define RUY_LOAD_FOUR_BY_FOUR()               \
  /* Load q0 */                               \
  "vldr d0, [%[src_ptr0], #0]\n"              \
  "vldr d1, [%[src_ptr0], #8]\n"              \
  /* if src_inc0 != 0, add 16 to src_ptr0 */  \
  "and r3, %[src_inc], #1\n"                  \
  "add %[src_ptr0], %[src_ptr0], r3, lsl #4\n"\
  /* Load q1 */                               \
  "vldr d2, [%[src_ptr1], #0]\n"              \
  "vldr d3, [%[src_ptr1], #8]\n"              \
  /* if src_inc1 != 0, add 16 to src_ptr0 */  \
  "and r3, %[src_inc], #2\n"                  \
  "add %[src_ptr1], %[src_ptr1], r3, lsl #3\n"\
  /* Load q2 */                               \
  "vldr d4, [%[src_ptr2], #0]\n"              \
  "vldr d5, [%[src_ptr2], #8]\n"              \
  /* if src_inc2 != 0, add 16 to src_ptr0 */  \
  "and r3, %[src_inc], #4\n"                  \
  "add %[src_ptr2], %[src_ptr2], r3, lsl #2\n"\
  /* Load q3 */                               \
  "vldr d6, [%[src_ptr3], #0]\n"              \
  "vldr d7, [%[src_ptr3], #8]\n"              \
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
  "vstr d16, [%[packed_ptr], #0]\n"               \
  "vstr d17, [%[packed_ptr], #8]\n"               \
  /* q10 = d20, d21 */                            \
  "add %[packed_ptr], %[packed_ptr], %[stride]\n" \
  "vstr d20, [%[packed_ptr], #0]\n"               \
  "vstr d21, [%[packed_ptr], #8]\n"               \
  /* q9 = d18, d19 */                             \
  "add %[packed_ptr], %[packed_ptr], %[stride]\n" \
  "vstr d18, [%[packed_ptr], #0]\n"               \
  "vstr d19, [%[packed_ptr], #8]\n"               \
  /* q11 = d22, d23 */                            \
  "add %[packed_ptr], %[packed_ptr], %[stride]\n" \
  "vstr d22, [%[packed_ptr], #0]\n"               \
  "vstr d23, [%[packed_ptr], #8]\n"               \
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
          "mov r0, 0\n"
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

#define RUY_STORE_ONE_ROW(ROW, REGISTER1, REGISTER2)      \
          "cmp r2, #" #ROW "\n"                           \
          "beq 4f\n"                                      \
          "vstr " #REGISTER1 ", [%[packed_ptr]]\n"    \
          "vstr " #REGISTER2 ", [%[packed_ptr], #8]\n"    \
          "add %[packed_ptr], %[packed_ptr], %[stride]\n"

          // Store q8
          RUY_STORE_ONE_ROW(0, d16, d17)
          // Store q10
          RUY_STORE_ONE_ROW(1, d20, d21)
          // Store q9
          RUY_STORE_ONE_ROW(2, d18, d19)
          // Store q11
          RUY_STORE_ONE_ROW(3, d22, d23)

#undef RUY_STORE_ONE_ROW

          "4:\n"

      // clang-format on
      : [ src_ptr0 ] "+r"(src_ptr0), [ src_ptr1 ] "+r"(src_ptr1),
        [ src_ptr2 ] "+r"(src_ptr2), [ src_ptr3 ] "+r"(src_ptr3),
        [ packed_ptr ] "+r"(packed_ptr)
      : [ src_inc ] "r"(static_cast<std::int64_t>(src_inc)),
        [ rows ] "r"(src_rows), [ stride ] "r"(output_stride)
      : "cc", "memory", "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3",
        "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13",
        "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23");
}

#endif  // (RUY_PLATFORM(NEON_32)

#if RUY_PLATFORM(NEON_64) && RUY_OPT_ENABLED(RUY_OPT_ASM)
void PackFloatNeonInOrder(const float* src_ptr0, const float* src_ptr1,
                          const float* src_ptr2, const float* src_ptr3,
                          int src_inc0, int src_inc1, int src_inc2,
                          int src_inc3, int src_rows, int src_zero_point,
                          float* packed_ptr, int start_col, int end_col) {
  gemmlowp::ScopedProfilingLabel label(
      "Pack (kNeon, optimized for in-order cores)");

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
