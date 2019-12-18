/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_CHECK_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_CHECK_H_

#if defined(__SSE4_1__)
// SSE 4.1 available: Use the SSE code.
#define SSE_OR_PORTABLE(funcname, ...) Sse##funcname(__VA_ARGS__)

#else

#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"

// No SSE 4.1 available: Fall back to NEON_OR_PORTABLE, potentially used with
// NEON_2_SSE translator library. As the library requires SSSE3, the fallback is
// generally using Portable code, only a narrow subset of processors supporting
// SSSE3 but no SSE4.1 is affected - but that includes the android_x86 ABI (not
// android_x86_64).
#define SSE_OR_PORTABLE(...) NEON_OR_PORTABLE(__VA_ARGS__)
#endif

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_CHECK_H_
