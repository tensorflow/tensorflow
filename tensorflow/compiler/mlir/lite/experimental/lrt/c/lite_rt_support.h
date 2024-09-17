// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_C_LITE_RT_SUPPORT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_C_LITE_RT_SUPPORT_H_

#include <stdio.h>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_common.h"  // IWYU pragma: keep

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// #define LRT_ABORT abort()
// TODO: b/365295276 - Find a fatal error approach that will pass kokoro.
#define LRT_ABORT

#define LRT_FATAL(msg)              \
  {                                 \
    fprintf(stderr, "%s\n", (msg)); \
    LRT_ABORT;                      \
  }

#define LRT_RETURN_STATUS_IF_NOT_OK(expr)                 \
  {                                                       \
    LrtStatus stat = expr;                                \
    if (GetStatusCode(stat) != kLrtStatusOk) return stat; \
    StatusDestroy(stat);                                  \
  }

// TODO: b/365295276 - Add optional debug only print messages support
// to all macros.
#define LRT_RETURN_STATUS_IF_NOT_OK_MSG(expr, d_msg) \
  LRT_RETURN_STATUS_IF_NOT_OK(expr)

#define LRT_RETURN_VAL_IF_NOT_OK(expr, ret_val) \
  {                                             \
    LrtStatus stat = expr;                      \
    LrtStatusCode code_ = GetStatusCode(stat);  \
    StatusDestroy(stat);                        \
    if (code_ != kLrtStatusOk) return ret_val;  \
  }


#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_C_LITE_RT_SUPPORT_H_
