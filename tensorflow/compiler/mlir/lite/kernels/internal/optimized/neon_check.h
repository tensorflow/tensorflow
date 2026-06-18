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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_CHECK_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_CHECK_H_

// LINT.IfChange

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#include <arm_neon.h>  // IWYU pragma: export
#endif

#if defined __GNUC__ && defined __SSE4_1__ && !defined TF_LITE_DISABLE_X86_NEON
#define USE_NEON
#include "NEON_2_SSE.h"  // IWYU pragma: export
#endif

// LINT.ThenChange(//tensorflow/lite/kernels/internal/optimized/neon_check.h)

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_CHECK_H_
