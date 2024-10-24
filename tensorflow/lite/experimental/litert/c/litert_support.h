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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_SUPPORT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_SUPPORT_H_

#include <alloca.h>
#include <stdio.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"  // IWYU pragma: keep

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// #define LITERT_ABORT abort()
// TODO: b/365295276 - Find a fatal error approach that will pass kokoro.
#define LITERT_ABORT

#define LITERT_FATAL(msg)           \
  {                                 \
    fprintf(stderr, "%s\n", (msg)); \
    LITERT_ABORT;                   \
  }

#define LITERT_RETURN_STATUS_IF_NOT_OK(expr) \
  if (LiteRtStatus status = expr; status != kLiteRtStatusOk) return status;

#define LITERT_RETURN_STATUS_IF_NOT_OK_OR_NOT_MATCHED(expr)               \
  if (LiteRtStatus status = expr;                                         \
      (status != kLiteRtStatusOk && status != kLrtStatusLegalizeNoMatch)) \
    return status;

// TODO: b/365295276 - Add optional debug only print messages support
// to all macros.
#define LITERT_RETURN_STATUS_IF_NOT_OK_MSG(expr, d_msg) \
  LITERT_RETURN_STATUS_IF_NOT_OK(expr)

#define LITERT_RETURN_VAL_IF_NOT_OK(expr, ret_val) \
  if (LiteRtStatus status = expr; status != kLiteRtStatusOk) return ret_val;

#define LITERT_STACK_ARRAY(ty, var, size, init) \
  ty* var = (ty*)alloca(sizeof(ty) * size);     \
  for (ty* e = var; e < var + size; ++e) {      \
    *e = init;                                  \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_SUPPORT_H_
