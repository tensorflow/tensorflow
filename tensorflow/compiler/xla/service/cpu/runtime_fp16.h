/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FP16_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FP16_H_

#include <stdint.h>

// _Float16 always gets us the correct ABI type, so use that if available.
// AArch64 GCC defines __FLT16_MANT_DIG__ even when _Float16 is not available.
#if defined(__FLT16_MANT_DIG__) && \
    (defined(__clang__) || !(defined(__GNUC__) && defined(__aarch64__)))
using XlaF16ABIType = _Float16;
#elif defined(__x86_64__)
// Older versions of Clang don't have _Float16. Since both float and _Float16
// are passed in the same register we can use the wider type and careful casting
// to conform to x86_64 psABI. This only works with the assumption that we're
// dealing with little-endian values passed in wider registers.
using XlaF16ABIType = float;
#else
// Default to uint16_t if we have nothing else.
using XlaF16ABIType = uint16_t;
#endif

// Converts an F32 value to a F16.
extern "C" XlaF16ABIType __gnu_f2h_ieee(float);

// Converts an F16 value to a F32.
extern "C" float __gnu_h2f_ieee(XlaF16ABIType);

// Converts an F64 value to a F16.
extern "C" XlaF16ABIType __truncdfhf2(double);

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FP16_H_
