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

#define KERNEL_4x1                   \
  "dup v24.16b, %w[bit_shift]\n"     \
  "mov x0, %[element_ptr]\n"         \
  "mov x6, %[lhs_val]\n"             \
  "mov x1, %[rhs_val]\n"             \
                                     \
      INNER_LOOP_BEGIN               \
  ":\n"                              \
  "mov x4, x6\n"                     \
  "ld1 {v4.16b}, [x4], #16\n"        \
  "dup v16.4s, wzr\n"                \
  "dup v17.4s, wzr\n"                \
  "ld1 {v5.16b}, [x4], #16\n"        \
  "dup v18.4s, wzr\n"                \
  "dup v19.4s, wzr\n"                \
  "ld1 {v6.16b}, [x4], #16\n"        \
  "and v8.16b, v4.16b, v24.16b\n"    \
  "and v9.16b, v5.16b, v24.16b\n"    \
  "ld1 {v7.16b}, [x4], #16\n"        \
  "ushr v12.16b, v4.16b, #4\n"       \
  "ushr v13.16b, v5.16b, #4\n"       \
  "ld1 {v0.16b}, [x1], #16\n"        \
  "and v10.16b, v6.16b, v24.16b\n"   \
  "and v11.16b, v7.16b, v24.16b\n"   \
  "ld1 {v1.16b}, [x1], #16\n"        \
  "ushr v14.16b, v6.16b, #4\n"       \
  "ushr v15.16b, v7.16b, #4\n"       \
  "mov w3, %w[run_depth]\n"          \
  "subs w3, w3, #1\n"                \
  "b.ls " INNER_LOOP_END "f\n"       \
                                     \
      INNER_LOOP                     \
  ":\n"                              \
  "ld1 {v4.16b}, [x4], #16\n"        \
  "smull v20.8h, v12.8b, v0.8b\n"    \
  "smull v21.8h, v13.8b, v0.8b\n"    \
  "smull v22.8h, v14.8b, v0.8b\n"    \
  "ld1 {v5.16b}, [x4], #16\n"        \
  "smull v23.8h, v15.8b, v0.8b\n"    \
  "smlal v20.8h, v8.8b, v1.8b\n"     \
  "smlal v21.8h, v9.8b, v1.8b\n"     \
  "ld1 {v6.16b}, [x4], #16\n"        \
  "smlal v22.8h, v10.8b, v1.8b\n"    \
  "smlal v23.8h, v11.8b, v1.8b\n"    \
  "smlal2 v20.8h, v12.16b, v0.16b\n" \
  "ld1 {v7.16b}, [x4], #16\n"        \
  "smlal2 v21.8h, v13.16b, v0.16b\n" \
  "smlal2 v22.8h, v14.16b, v0.16b\n" \
  "smlal2 v23.8h, v15.16b, v0.16b\n" \
  "smlal2 v20.8h, v8.16b, v1.16b\n"  \
  "smlal2 v21.8h, v9.16b, v1.16b\n"  \
  "smlal2 v22.8h, v10.16b, v1.16b\n" \
  "ld1 {v0.16b}, [x1], #16\n"        \
  "smlal2 v23.8h, v11.16b, v1.16b\n" \
  "sadalp v16.4s, v20.8h\n"          \
  "sadalp v17.4s, v21.8h\n"          \
  "sadalp v18.4s, v22.8h\n"          \
  "sadalp v19.4s, v23.8h\n"          \
  "ld1 {v1.16b}, [x1], #16\n"        \
  "and v8.16b, v4.16b, v24.16b\n"    \
  "and v9.16b, v5.16b, v24.16b\n"    \
  "ushr v12.16b, v4.16b, #4\n"       \
  "ushr v13.16b, v5.16b, #4\n"       \
  "and v10.16b, v6.16b, v24.16b\n"   \
  "and v11.16b, v7.16b, v24.16b\n"   \
  "ushr v14.16b, v6.16b, #4\n"       \
  "ushr v15.16b, v7.16b, #4\n"       \
  "subs w3, w3, #1\n"                \
  "b.hi " INNER_LOOP "b\n"           \
                                     \
      INNER_LOOP_END                 \
  ":\n"                              \
  "smull v20.8h, v12.8b, v0.8b\n"    \
  "smull v21.8h, v13.8b, v0.8b\n"    \
  "smull v22.8h, v14.8b, v0.8b\n"    \
  "smull v23.8h, v15.8b, v0.8b\n"    \
  "smlal v20.8h, v8.8b, v1.8b\n"     \
  "smlal v21.8h, v9.8b, v1.8b\n"     \
  "smlal v22.8h, v10.8b, v1.8b\n"    \
  "smlal v23.8h, v11.8b, v1.8b\n"    \
  "smlal2 v20.8h, v12.16b, v0.16b\n" \
  "smlal2 v21.8h, v13.16b, v0.16b\n" \
  "smlal2 v22.8h, v14.16b, v0.16b\n" \
  "smlal2 v23.8h, v15.16b, v0.16b\n" \
  "smlal2 v20.8h, v8.16b, v1.16b\n"  \
  "smlal2 v21.8h, v9.16b, v1.16b\n"  \
  "smlal2 v22.8h, v10.16b, v1.16b\n" \
  "smlal2 v23.8h, v11.16b, v1.16b\n" \
  "sadalp v16.4s, v20.8h\n"          \
  "sadalp v17.4s, v21.8h\n"          \
  "sadalp v18.4s, v22.8h\n"          \
  "sadalp v19.4s, v23.8h\n"          \
  "addp v4.4s, v16.4s, v17.4s\n"     \
  "addp v5.4s, v18.4s, v19.4s\n"     \
  "addp v6.4s, v4.4s, v5.4s\n"       \
  "st1 {v6.4s}, [x0], #16\n"

#define KERNEL_4x2                        \
  "mov x0, %[element_ptr]\n"              \
  "mov x6, %[lhs_val]\n"                  \
  "mov x1, %[rhs_val]\n" INNER_LOOP_BEGIN \
  ":\n"                                   \
  "mov x4, x6\n"                          \
  "ld1 {v4.16b}, [x4], #16\n"             \
  "dup v31.16b, %w[bit_shift]\n"          \
  "dup v16.4s, wzr\n"                     \
  "dup v17.4s, wzr\n"                     \
  "ld1 {v5.16b}, [x4], #16\n"             \
  "dup v18.4s, wzr\n"                     \
  "dup v19.4s, wzr\n"                     \
  "ld1 {v6.16b}, [x4], #16\n"             \
  "and v8.16b, v4.16b, v31.16b\n"         \
  "and v9.16b, v5.16b, v31.16b\n"         \
  "ld1 {v7.16b}, [x4], #16\n"             \
  "ushr v12.16b, v4.16b, #4\n"            \
  "ushr v13.16b, v5.16b, #4\n"            \
  "dup v24.4s, wzr\n"                     \
  "ld1 {v0.16b}, [x1], #16\n"             \
  "dup v25.4s, wzr\n"                     \
  "dup v26.4s, wzr\n"                     \
  "ld1 {v1.16b}, [x1], #16\n"             \
  "dup v27.4s, wzr\n"                     \
  "and v10.16b, v6.16b, v31.16b\n"        \
  "ld1 {v2.16b}, [x1], #16\n"             \
  "and v11.16b, v7.16b, v31.16b\n"        \
  "ushr v14.16b, v6.16b, #4\n"            \
  "ld1 {v3.16b}, [x1], #16\n"             \
  "ushr v15.16b, v7.16b, #4\n"            \
  "mov w3, %w[run_depth]\n"               \
  "subs w3, w3, #1\n"                     \
  "b.ls " INNER_LOOP_END "f\n"            \
                                          \
      INNER_LOOP                          \
  ":\n"                                   \
  "smull v20.8h, v12.8b, v0.8b\n"         \
  "smull v21.8h, v13.8b, v0.8b\n"         \
  "smull v22.8h, v14.8b, v0.8b\n"         \
  "ld1 {v4.16b}, [x4], #16\n"             \
  "smull v23.8h, v15.8b, v0.8b\n"         \
  "smlal v20.8h, v8.8b, v1.8b\n"          \
  "smlal v21.8h, v9.8b, v1.8b\n"          \
  "ld1 {v5.16b}, [x4], #16\n"             \
  "smlal v22.8h, v10.8b, v1.8b\n"         \
  "smlal v23.8h, v11.8b, v1.8b\n"         \
  "smlal2 v20.8h, v12.16b, v0.16b\n"      \
  "ld1 {v6.16b}, [x4], #16\n"             \
  "smlal2 v21.8h, v13.16b, v0.16b\n"      \
  "smlal2 v22.8h, v14.16b, v0.16b\n"      \
  "smlal2 v23.8h, v15.16b, v0.16b\n"      \
  "ld1 {v7.16b}, [x4], #16\n"             \
  "smlal2 v20.8h, v8.16b, v1.16b\n"       \
  "smlal2 v21.8h, v9.16b, v1.16b\n"       \
  "smlal2 v22.8h, v10.16b, v1.16b\n"      \
  "smlal2 v23.8h, v11.16b, v1.16b\n"      \
                                          \
  "ld1 {v0.16b}, [x1], #16\n"             \
                                          \
  "sadalp v16.4s, v20.8h\n"               \
  "sadalp v17.4s, v21.8h\n"               \
  "sadalp v18.4s, v22.8h\n"               \
  "sadalp v19.4s, v23.8h\n"               \
                                          \
  "ld1 {v1.16b}, [x1], #16\n"             \
                                          \
  "smull v28.8h, v12.8b, v2.8b\n"         \
  "smull v29.8h, v13.8b, v2.8b\n"         \
  "smull v30.8h, v14.8b, v2.8b\n"         \
  "smull v20.8h, v15.8b, v2.8b\n"         \
                                          \
  "smlal v28.8h, v8.8b, v3.8b\n"          \
  "smlal v29.8h, v9.8b, v3.8b\n"          \
  "smlal v30.8h, v10.8b, v3.8b\n"         \
  "smlal v20.8h, v11.8b, v3.8b\n"         \
  "smlal2 v28.8h, v12.16b, v2.16b\n"      \
  "smlal2 v29.8h, v13.16b, v2.16b\n"      \
  "smlal2 v30.8h, v14.16b, v2.16b\n"      \
  "smlal2 v20.8h, v15.16b, v2.16b\n"      \
  "smlal2 v28.8h, v8.16b, v3.16b\n"       \
  "smlal2 v29.8h, v9.16b, v3.16b\n"       \
  "smlal2 v30.8h, v10.16b, v3.16b\n"      \
  "smlal2 v20.8h, v11.16b, v3.16b\n"      \
                                          \
  "ld1 {v2.16b}, [x1], #16\n"             \
                                          \
  "sadalp v24.4s, v28.8h\n"               \
  "sadalp v25.4s, v29.8h\n"               \
  "sadalp v26.4s, v30.8h\n"               \
  "sadalp v27.4s, v20.8h\n"               \
                                          \
  "ld1 {v3.16b}, [x1], #16\n"             \
                                          \
  "and v8.16b, v4.16b, v31.16b\n"         \
  "and v9.16b, v5.16b, v31.16b\n"         \
  "ushr v12.16b, v4.16b, #4\n"            \
  "ushr v13.16b, v5.16b, #4\n"            \
                                          \
  "subs w3, w3, #1\n"                     \
                                          \
  "and v10.16b, v6.16b, v31.16b\n"        \
  "and v11.16b, v7.16b, v31.16b\n"        \
  "ushr v14.16b, v6.16b, #4\n"            \
  "ushr v15.16b, v7.16b, #4\n"            \
                                          \
  "b.hi " INNER_LOOP "b\n"                \
                                          \
      INNER_LOOP_END                      \
  ":\n"                                   \
  "smull v20.8h, v12.8b, v0.8b\n"         \
  "smull v21.8h, v13.8b, v0.8b\n"         \
  "smull v22.8h, v14.8b, v0.8b\n"         \
  "smull v23.8h, v15.8b, v0.8b\n"         \
  "smlal v20.8h, v8.8b, v1.8b\n"          \
  "smlal v21.8h, v9.8b, v1.8b\n"          \
  "smlal v22.8h, v10.8b, v1.8b\n"         \
  "smlal v23.8h, v11.8b, v1.8b\n"         \
  "smlal2 v20.8h, v12.16b, v0.16b\n"      \
  "smlal2 v21.8h, v13.16b, v0.16b\n"      \
  "smlal2 v22.8h, v14.16b, v0.16b\n"      \
  "smlal2 v23.8h, v15.16b, v0.16b\n"      \
  "smlal2 v20.8h, v8.16b, v1.16b\n"       \
  "smlal2 v21.8h, v9.16b, v1.16b\n"       \
  "smlal2 v22.8h, v10.16b, v1.16b\n"      \
  "smlal2 v23.8h, v11.16b, v1.16b\n"      \
  "smull v28.8h, v12.8b, v2.8b\n"         \
  "smull v29.8h, v13.8b, v2.8b\n"         \
  "smull v30.8h, v14.8b, v2.8b\n"         \
  "smull v31.8h, v15.8b, v2.8b\n"         \
  "smlal v28.8h, v8.8b, v3.8b\n"          \
  "smlal v29.8h, v9.8b, v3.8b\n"          \
  "smlal v30.8h, v10.8b, v3.8b\n"         \
  "smlal v31.8h, v11.8b, v3.8b\n"         \
  "smlal2 v28.8h, v12.16b, v2.16b\n"      \
  "smlal2 v29.8h, v13.16b, v2.16b\n"      \
  "smlal2 v30.8h, v14.16b, v2.16b\n"      \
  "smlal2 v31.8h, v15.16b, v2.16b\n"      \
  "smlal2 v28.8h, v8.16b, v3.16b\n"       \
  "smlal2 v29.8h, v9.16b, v3.16b\n"       \
  "smlal2 v30.8h, v10.16b, v3.16b\n"      \
  "smlal2 v31.8h, v11.16b, v3.16b\n"      \
                                          \
  "sadalp v16.4s, v20.8h\n"               \
  "sadalp v17.4s, v21.8h\n"               \
  "sadalp v18.4s, v22.8h\n"               \
  "sadalp v19.4s, v23.8h\n"               \
  "sadalp v24.4s, v28.8h\n"               \
  "sadalp v25.4s, v29.8h\n"               \
  "sadalp v26.4s, v30.8h\n"               \
  "sadalp v27.4s, v31.8h\n"               \
                                          \
  "addp v4.4s, v16.4s, v17.4s\n"          \
  "addp v5.4s, v18.4s, v19.4s\n"          \
  "addp v8.4s, v24.4s, v25.4s\n"          \
  "addp v9.4s, v26.4s, v27.4s\n"          \
  "addp v6.4s, v4.4s, v5.4s\n"            \
  "addp v7.4s, v8.4s, v9.4s\n"            \
  "st1 {v6.4s, v7.4s}, [x0], #32\n"

#define KERNEL_4x4                  \
  "dup v3.16b, %w[bit_shift]\n"     \
  "mov x0, %[element_ptr]\n"        \
  "mov x6, %[lhs_val]\n"            \
  "mov x1, %[rhs_val]\n"            \
                                    \
      INNER_LOOP_BEGIN              \
  ":\n"                             \
  "mov x4, x6\n"                    \
  "ld1 {v4.16b}, [x4], #16\n"       \
  "dup v16.4s, wzr\n"               \
  "dup v17.4s, wzr\n"               \
  "dup v18.4s, wzr\n"               \
  "dup v19.4s, wzr\n"               \
  "ld1 {v5.16b}, [x4], #16\n"       \
  "dup v20.4s, wzr\n"               \
  "dup v21.4s, wzr\n"               \
  "dup v22.4s, wzr\n"               \
  "ld1 {v6.16b}, [x4], #16\n"       \
  "dup v23.4s, wzr\n"               \
  "dup v24.4s, wzr\n"               \
  "dup v25.4s, wzr\n"               \
  "ld1 {v7.16b}, [x4], #16\n"       \
  "dup v26.4s, wzr\n"               \
  "dup v27.4s, wzr\n"               \
  "dup v28.4s, wzr\n"               \
  "dup v29.4s, wzr\n"               \
  "ld1 {v0.16b}, [x1], #16\n"       \
  "dup v30.4s, wzr\n"               \
  "dup v31.4s, wzr\n"               \
  "mov w3, %w[run_depth]\n"         \
  "ld1 {v1.16b}, [x1], #16\n"       \
  "and v8.16b, v4.16b, v3.16b\n"    \
  "and v9.16b, v5.16b, v3.16b\n"    \
  "and v10.16b, v6.16b, v3.16b\n"   \
  "and v11.16b, v7.16b, v3.16b\n"   \
  "ushr v12.16b, v4.16b, #4\n"      \
  "ushr v13.16b, v5.16b, #4\n"      \
  "ushr v14.16b, v6.16b, #4\n"      \
  "ushr v15.16b, v7.16b, #4\n"      \
  "subs w3, w3, #1\n"               \
  "b.ls " INNER_LOOP_END "f\n"      \
                                    \
      INNER_LOOP                    \
  ":\n"                             \
  "smull v4.8h, v12.8b, v0.8b\n"    \
  "smull v5.8h, v13.8b, v0.8b\n"    \
  "smull v6.8h, v14.8b, v0.8b\n"    \
  "smull v7.8h, v15.8b, v0.8b\n"    \
  "smlal2 v4.8h, v12.16b, v0.16b\n" \
  "smlal2 v5.8h, v13.16b, v0.16b\n" \
  "smlal2 v6.8h, v14.16b, v0.16b\n" \
  "smlal2 v7.8h, v15.16b, v0.16b\n" \
                                    \
  "ld1 {v2.16b}, [x1], #16\n"       \
                                    \
  "smlal v4.8h, v8.8b, v1.8b\n"     \
  "smlal v5.8h, v9.8b, v1.8b\n"     \
  "smlal v6.8h, v10.8b, v1.8b\n"    \
  "smlal v7.8h, v11.8b, v1.8b\n"    \
  "smlal2 v4.8h, v8.16b, v1.16b\n"  \
  "smlal2 v5.8h, v9.16b, v1.16b\n"  \
  "smlal2 v6.8h, v10.16b, v1.16b\n" \
  "smlal2 v7.8h, v11.16b, v1.16b\n" \
                                    \
  "ld1 {v0.16b}, [x1], #16\n"       \
                                    \
  "sadalp v16.4s, v4.8h\n"          \
  "sadalp v17.4s, v5.8h\n"          \
  "sadalp v18.4s, v6.8h\n"          \
  "sadalp v19.4s, v7.8h\n"          \
                                    \
  "smull v4.8h, v12.8b, v2.8b\n"    \
  "smull v5.8h, v13.8b, v2.8b\n"    \
  "smull v6.8h, v14.8b, v2.8b\n"    \
  "smull v7.8h, v15.8b, v2.8b\n"    \
  "smlal2 v4.8h, v12.16b, v2.16b\n" \
  "smlal2 v5.8h, v13.16b, v2.16b\n" \
  "smlal2 v6.8h, v14.16b, v2.16b\n" \
  "smlal2 v7.8h, v15.16b, v2.16b\n" \
                                    \
  "ld1 {v1.16b}, [x1], #16\n"       \
                                    \
  "smlal v4.8h, v8.8b, v0.8b\n"     \
  "smlal v5.8h, v9.8b, v0.8b\n"     \
  "smlal v6.8h, v10.8b, v0.8b\n"    \
  "smlal v7.8h, v11.8b, v0.8b\n"    \
  "smlal2 v4.8h, v8.16b, v0.16b\n"  \
  "smlal2 v5.8h, v9.16b, v0.16b\n"  \
  "smlal2 v6.8h, v10.16b, v0.16b\n" \
  "smlal2 v7.8h, v11.16b, v0.16b\n" \
                                    \
  "ld1 {v2.16b}, [x1], #16\n"       \
                                    \
  "sadalp v20.4s, v4.8h\n"          \
  "sadalp v21.4s, v5.8h\n"          \
  "sadalp v22.4s, v6.8h\n"          \
  "sadalp v23.4s, v7.8h\n"          \
                                    \
  "smull v4.8h, v12.8b, v1.8b\n"    \
  "smull v5.8h, v13.8b, v1.8b\n"    \
  "smull v6.8h, v14.8b, v1.8b\n"    \
  "smull v7.8h, v15.8b, v1.8b\n"    \
  "smlal2 v4.8h, v12.16b, v1.16b\n" \
  "smlal2 v5.8h, v13.16b, v1.16b\n" \
  "smlal2 v6.8h, v14.16b, v1.16b\n" \
  "smlal2 v7.8h, v15.16b, v1.16b\n" \
                                    \
  "ld1 {v0.16b}, [x1], #16\n"       \
                                    \
  "smlal v4.8h, v8.8b, v2.8b\n"     \
  "smlal v5.8h, v9.8b, v2.8b\n"     \
  "smlal v6.8h, v10.8b, v2.8b\n"    \
  "smlal v7.8h, v11.8b, v2.8b\n"    \
  "smlal2 v4.8h, v8.16b, v2.16b\n"  \
  "smlal2 v5.8h, v9.16b, v2.16b\n"  \
  "smlal2 v6.8h, v10.16b, v2.16b\n" \
  "smlal2 v7.8h, v11.16b, v2.16b\n" \
                                    \
  "ld1 {v1.16b}, [x1], #16\n"       \
                                    \
  "sadalp v24.4s, v4.8h\n"          \
  "sadalp v25.4s, v5.8h\n"          \
  "sadalp v26.4s, v6.8h\n"          \
  "sadalp v27.4s, v7.8h\n"          \
                                    \
  "smull v4.8h, v12.8b, v0.8b\n"    \
  "smull v5.8h, v13.8b, v0.8b\n"    \
  "smull v6.8h, v14.8b, v0.8b\n"    \
  "smull v7.8h, v15.8b, v0.8b\n"    \
  "smlal2 v4.8h, v12.16b, v0.16b\n" \
  "smlal2 v5.8h, v13.16b, v0.16b\n" \
  "smlal2 v6.8h, v14.16b, v0.16b\n" \
  "smlal2 v7.8h, v15.16b, v0.16b\n" \
                                    \
  "ld1 {v12.16b}, [x4], #16\n"      \
                                    \
  "smlal v4.8h, v8.8b, v1.8b\n"     \
  "smlal v5.8h, v9.8b, v1.8b\n"     \
  "smlal v6.8h, v10.8b, v1.8b\n"    \
  "smlal v7.8h, v11.8b, v1.8b\n"    \
                                    \
  "ld1 {v13.16b}, [x4], #16\n"      \
                                    \
  "smlal2 v4.8h, v8.16b, v1.16b\n"  \
  "smlal2 v5.8h, v9.16b, v1.16b\n"  \
  "smlal2 v6.8h, v10.16b, v1.16b\n" \
  "smlal2 v7.8h, v11.16b, v1.16b\n" \
                                    \
  "ld1 {v14.16b}, [x4], #16\n"      \
                                    \
  "sadalp v28.4s, v4.8h\n"          \
  "sadalp v29.4s, v5.8h\n"          \
  "sadalp v30.4s, v6.8h\n"          \
  "sadalp v31.4s, v7.8h\n"          \
                                    \
  "ld1 {v15.16b}, [x4], #16\n"      \
                                    \
  "and v8.16b, v12.16b, v3.16b\n"   \
  "and v9.16b, v13.16b, v3.16b\n"   \
  "and v10.16b, v14.16b, v3.16b\n"  \
                                    \
  "ld1 {v0.16b}, [x1], #16\n"       \
                                    \
  "and v11.16b, v15.16b, v3.16b\n"  \
  "ushr v12.16b, v12.16b, #4\n"     \
  "ushr v13.16b, v13.16b, #4\n"     \
                                    \
  "ld1 {v1.16b}, [x1], #16\n"       \
                                    \
  "ushr v14.16b, v14.16b, #4\n"     \
  "ushr v15.16b, v15.16b, #4\n"     \
                                    \
  "subs w3, w3, #1\n"               \
  "b.hi " INNER_LOOP "b\n"          \
                                    \
      INNER_LOOP_END                \
  ":\n"                             \
  "smull v4.8h, v12.8b, v0.8b\n"    \
  "smull v5.8h, v13.8b, v0.8b\n"    \
  "smull v6.8h, v14.8b, v0.8b\n"    \
  "smull v7.8h, v15.8b, v0.8b\n"    \
  "smlal2 v4.8h, v12.16b, v0.16b\n" \
  "smlal2 v5.8h, v13.16b, v0.16b\n" \
  "smlal2 v6.8h, v14.16b, v0.16b\n" \
  "smlal2 v7.8h, v15.16b, v0.16b\n" \
                                    \
  "ld1 {v2.16b}, [x1], #16\n"       \
                                    \
  "smlal v4.8h, v8.8b, v1.8b\n"     \
  "smlal v5.8h, v9.8b, v1.8b\n"     \
  "smlal v6.8h, v10.8b, v1.8b\n"    \
  "smlal v7.8h, v11.8b, v1.8b\n"    \
  "smlal2 v4.8h, v8.16b, v1.16b\n"  \
  "smlal2 v5.8h, v9.16b, v1.16b\n"  \
  "smlal2 v6.8h, v10.16b, v1.16b\n" \
  "smlal2 v7.8h, v11.16b, v1.16b\n" \
                                    \
  "ld1 {v0.16b}, [x1], #16\n"       \
                                    \
  "sadalp v16.4s, v4.8h\n"          \
  "sadalp v17.4s, v5.8h\n"          \
  "sadalp v18.4s, v6.8h\n"          \
  "sadalp v19.4s, v7.8h\n"          \
                                    \
  "smull v4.8h, v12.8b, v2.8b\n"    \
  "smull v5.8h, v13.8b, v2.8b\n"    \
  "smull v6.8h, v14.8b, v2.8b\n"    \
  "smull v7.8h, v15.8b, v2.8b\n"    \
  "smlal2 v4.8h, v12.16b, v2.16b\n" \
  "smlal2 v5.8h, v13.16b, v2.16b\n" \
  "smlal2 v6.8h, v14.16b, v2.16b\n" \
  "smlal2 v7.8h, v15.16b, v2.16b\n" \
                                    \
  "ld1 {v1.16b}, [x1], #16\n"       \
                                    \
  "smlal v4.8h, v8.8b, v0.8b\n"     \
  "smlal v5.8h, v9.8b, v0.8b\n"     \
  "smlal v6.8h, v10.8b, v0.8b\n"    \
  "smlal v7.8h, v11.8b, v0.8b\n"    \
  "smlal2 v4.8h, v8.16b, v0.16b\n"  \
  "smlal2 v5.8h, v9.16b, v0.16b\n"  \
  "smlal2 v6.8h, v10.16b, v0.16b\n" \
  "smlal2 v7.8h, v11.16b, v0.16b\n" \
                                    \
  "ld1 {v2.16b}, [x1], #16\n"       \
                                    \
  "sadalp v20.4s, v4.8h\n"          \
  "sadalp v21.4s, v5.8h\n"          \
  "sadalp v22.4s, v6.8h\n"          \
  "sadalp v23.4s, v7.8h\n"          \
                                    \
  "smull v4.8h, v12.8b, v1.8b\n"    \
  "smull v5.8h, v13.8b, v1.8b\n"    \
  "smull v6.8h, v14.8b, v1.8b\n"    \
  "smull v7.8h, v15.8b, v1.8b\n"    \
  "smlal2 v4.8h, v12.16b, v1.16b\n" \
  "smlal2 v5.8h, v13.16b, v1.16b\n" \
  "smlal2 v6.8h, v14.16b, v1.16b\n" \
  "smlal2 v7.8h, v15.16b, v1.16b\n" \
                                    \
  "ld1 {v0.16b}, [x1], #16\n"       \
                                    \
  "smlal v4.8h, v8.8b, v2.8b\n"     \
  "smlal v5.8h, v9.8b, v2.8b\n"     \
  "smlal v6.8h, v10.8b, v2.8b\n"    \
  "smlal v7.8h, v11.8b, v2.8b\n"    \
  "smlal2 v4.8h, v8.16b, v2.16b\n"  \
  "smlal2 v5.8h, v9.16b, v2.16b\n"  \
  "smlal2 v6.8h, v10.16b, v2.16b\n" \
  "smlal2 v7.8h, v11.16b, v2.16b\n" \
                                    \
  "ld1 {v1.16b}, [x1], #16\n"       \
                                    \
  "sadalp v24.4s, v4.8h\n"          \
  "sadalp v25.4s, v5.8h\n"          \
  "sadalp v26.4s, v6.8h\n"          \
  "sadalp v27.4s, v7.8h\n"          \
                                    \
  "smull v4.8h, v12.8b, v0.8b\n"    \
  "smull v5.8h, v13.8b, v0.8b\n"    \
  "smull v6.8h, v14.8b, v0.8b\n"    \
  "smull v7.8h, v15.8b, v0.8b\n"    \
  "smlal2 v4.8h, v12.16b, v0.16b\n" \
  "smlal2 v5.8h, v13.16b, v0.16b\n" \
  "smlal2 v6.8h, v14.16b, v0.16b\n" \
  "smlal2 v7.8h, v15.16b, v0.16b\n" \
                                    \
  "smlal v4.8h, v8.8b, v1.8b\n"     \
  "smlal v5.8h, v9.8b, v1.8b\n"     \
  "smlal v6.8h, v10.8b, v1.8b\n"    \
  "smlal v7.8h, v11.8b, v1.8b\n"    \
  "smlal2 v4.8h, v8.16b, v1.16b\n"  \
  "smlal2 v5.8h, v9.16b, v1.16b\n"  \
  "smlal2 v6.8h, v10.16b, v1.16b\n" \
  "smlal2 v7.8h, v11.16b, v1.16b\n" \
                                    \
  "sadalp v28.4s, v4.8h\n"          \
  "sadalp v29.4s, v5.8h\n"          \
  "sadalp v30.4s, v6.8h\n"          \
  "sadalp v31.4s, v7.8h\n"          \
                                    \
  "addp v14.4s, v16.4s, v17.4s\n"   \
  "addp v15.4s, v18.4s, v19.4s\n"   \
  "addp v12.4s, v20.4s, v21.4s\n"   \
  "addp v13.4s, v22.4s, v23.4s\n"   \
  "addp v10.4s, v24.4s, v25.4s\n"   \
  "addp v11.4s, v26.4s, v27.4s\n"   \
  "addp v8.4s, v28.4s, v29.4s\n"    \
  "addp v9.4s, v30.4s, v31.4s\n"    \
  "addp v4.4s, v14.4s, v15.4s\n"    \
  "addp v5.4s, v12.4s, v13.4s\n"    \
  "addp v6.4s, v10.4s, v11.4s\n"    \
  "addp v7.4s, v8.4s, v9.4s\n"      \
  "st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64\n"

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
void NeonRunKernelNoSDot<4, 2, 32>(const uint8_t* lhs, const int8_t* rhs,
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
void NeonRunKernelNoSDot<4, 4, 32>(const uint8_t* lhs, const int8_t* rhs,
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

}  // namespace optimized_4bit
}  // namespace tflite
#endif  // defined(FC_4BIT_NEON)...
