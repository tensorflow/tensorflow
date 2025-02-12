/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_COMPILER_MACROS_H_
#define XLA_COMPILER_MACROS_H_

#if (defined(__GNUC__) || defined(__clang__)) && defined(__SSE2__)
#define XLA_HAS_SSE2
#elif defined(_MSC_VER) && !defined(_M_ARM64EC) && defined(_M_X64)
#define XLA_HAS_SSE2
#elif defined(_MSC_VER) && !defined(_M_ARM64EC) && \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define XLA_HAS_SSE2
#elif defined(__AVX__)
#define XLA_HAS_SSE2
#endif

#if defined(_M_ARM64) || defined(_M_ARM64EC)
#define XLA_HAS_ARM64
#define XLA_HAS_ARM_NEON
#elif defined(__ARM_NEON) && !defined(__ARM_BIG_ENDIAN)
#define XLA_HAS_ARM_NEON

#if defined(__aarch64__)
#define XLA_HAS_ARM64
#endif  // defined(__aarch64__)

#endif  // defined(_M_ARM64) || defined(_M_ARM64EC)

#if defined(__clang__)
#define XLA_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define XLA_UNROLL _Pragma("GCC unroll 128")
#else
#define XLA_UNROLL
#endif

#if defined(__GNUC__) || defined(__clang__)
#define XLA_FLATTEN __attribute__((flatten))
#elif defined(_MSC_VER)
#define XLA_FLATTEN [[msvc::flatten]]
#else
#define XLA_FLATTEN
#endif

#endif  // XLA_COMPILER_MACROS_H_
