/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_WINDOWS_INTRINSICS_PORT_H_
#define TENSORFLOW_TSL_PLATFORM_WINDOWS_INTRINSICS_PORT_H_

#ifdef _MSC_VER
// the following avx intrinsics are not defined on windows
// in immintrin.h so we define them here.
//
#include "tensorflow/tsl/platform/types.h"

#define _mm_load_pd1 _mm_load1_pd

// only define these intrinsics if immintrin.h doesn't have them (VS2015 and
// earlier)
#if _MSC_VER < 1910
static inline int _mm256_extract_epi32(__m256i a, const int i) {
  return a.m256i_i32[i & 7];
}

static inline __m256i _mm256_insert_epi32(__m256i a, int b, const int i) {
  __m256i c = a;
  c.m256i_i32[i & 7] = b;
  return c;
}
#endif
#endif
#endif
