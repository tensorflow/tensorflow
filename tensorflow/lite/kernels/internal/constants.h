/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_CONSTANTS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_CONSTANTS_H_

// Maths constants.
// The following macros are not always available on all platforms.
// E.g. MSVC requires additional compile flag to export those.
#ifndef M_E
#define M_E 2.7182818284590452354 /* e */
#endif
#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074 /* log_2 e */
#endif
#ifndef M_LOG10E
#define M_LOG10E 0.43429448190325182765 /* log_10 e */
#endif
#ifndef M_LN2
#define M_LN2 0.69314718055994530942 /* log_e 2 */
#endif
#ifndef M_LN10
#define M_LN10 2.30258509299404568402 /* log_e 10 */
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923 /* pi/2 */
#endif
#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962 /* pi/4 */
#endif
#ifndef M_1_PI
#define M_1_PI 0.31830988618379067154 /* 1/pi */
#endif
#ifndef M_2_PI
#define M_2_PI 0.63661977236758134308 /* 2/pi */
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390 /* 2/sqrt(pi) */
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880 /* sqrt(2) */
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440 /* 1/sqrt(2) */
#endif

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_CONSTANTS_H_
