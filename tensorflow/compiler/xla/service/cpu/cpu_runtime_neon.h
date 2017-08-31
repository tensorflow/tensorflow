/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_RUNTIME_NEON_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_RUNTIME_NEON_H_

// This header declares functions which may be called by the generated code on
// the CPU. Calls to these functions must be resolved explicitly in the JIT in
// xla::cpu::SimpleResolver.

#include "tensorflow/core/platform/macros.h"

#ifdef __ARM_NEON__
// For the other runtimes (AVX, SSE4.1) we define the vector type directly using
// __attribute__((__vector_size__(*))).  Unfortunately, the typedef for the ARM
// NEON SIMD types is not portable, so the type has to come from <arm_neon.h>
#include <arm_neon.h>
#endif  // __ARM_NEON__

namespace xla {
namespace cpu {
namespace runtime {

extern const char *const kExpV4F32NEONSymbolName;
extern const char *const kLogV4F32NEONSymbolName;

#ifdef __ARM_NEON__
typedef float32x4_t V4F32NEON;
#else
// On non-ARM platforms ensure the declaration is present
struct V4F32NEON;
#endif  // __ARM_NEON__

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

extern "C" {

// The following functions are vectorized versions of a selection of libm
// library functions.
// References to these functions are created by the LLVM vectorizer.
xla::cpu::runtime::V4F32NEON __xla_cpu_runtime_ExpV4F32NEON(
    xla::cpu::runtime::V4F32NEON x) TF_ATTRIBUTE_WEAK;

xla::cpu::runtime::V4F32NEON __xla_cpu_runtime_LogV4F32NEON(
    xla::cpu::runtime::V4F32NEON x) TF_ATTRIBUTE_WEAK;
}

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_RUNTIME_NEON_H_
