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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_LAYOUT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_LAYOUT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Max number of dimensions in any ranked tensor type.
#define LITERT_TENSOR_MAX_RANK 8

// The shape information for tensor types of fixed rank.
typedef struct {
  // The number of dimensions.
  uint32_t rank;

  // Dimension sizes, array of length `rank`. Dynamic dimensions are anything
  // less than 0. Everything from [rank, LITERT_MAX_RANK) is undefined.
  int32_t dimensions[LITERT_TENSOR_MAX_RANK];

  // Strides for a nomimal NWHC layout. NULL if unused.
  const uint32_t* strides;
} LiteRtLayout;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_LAYOUT_H_
