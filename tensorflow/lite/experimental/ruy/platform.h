/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PLATFORM_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PLATFORM_H_

#ifdef __ANDROID_NDK__
#include <android/ndk-version.h>
#endif

#define RUY_PLATFORM(X) ((RUY_DONOTUSEDIRECTLY_##X) != 0)

// Architecture-level platform detection.
//
// Ruy requires these to be mutually exclusive.

// Detect x86.
#if defined(__x86_64__) || defined(__i386__) || defined(__i386) || \
    defined(__x86__) || defined(__X86__) || defined(_X86_) ||      \
    defined(_M_IX86) || defined(_M_X64)
#define RUY_DONOTUSEDIRECTLY_X86 1
#else
#define RUY_DONOTUSEDIRECTLY_X86 0
#endif

// Detect ARM 32-bit.
#ifdef __arm__
#define RUY_DONOTUSEDIRECTLY_ARM_32 1
#else
#define RUY_DONOTUSEDIRECTLY_ARM_32 0
#endif

// Detect ARM 64-bit.
#ifdef __aarch64__
#define RUY_DONOTUSEDIRECTLY_ARM_64 1
#else
#define RUY_DONOTUSEDIRECTLY_ARM_64 0
#endif

// Combined ARM.
#define RUY_DONOTUSEDIRECTLY_ARM \
  (RUY_DONOTUSEDIRECTLY_ARM_64 || RUY_DONOTUSEDIRECTLY_ARM_32)

// Feature and capability platform detection.
//
// These are mostly sub-selections of architectures.

// Detect NEON. Explicitly avoid emulation, or anything like it, on x86.
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !RUY_PLATFORM(X86)
#define RUY_DONOTUSEDIRECTLY_NEON 1
#else
#define RUY_DONOTUSEDIRECTLY_NEON 0
#endif

// Define ARM 32-bit NEON.
#define RUY_DONOTUSEDIRECTLY_NEON_32 \
  (RUY_DONOTUSEDIRECTLY_NEON && RUY_DONOTUSEDIRECTLY_ARM_32)

// Define ARM 64-bit NEON.
// Note: NEON is implied by ARM64, so this define is redundant.
// It still allows some conveyance of intent.
#define RUY_DONOTUSEDIRECTLY_NEON_64 \
  (RUY_DONOTUSEDIRECTLY_NEON && RUY_DONOTUSEDIRECTLY_ARM_64)

// Disable X86 enhancements on __APPLE__ because b/138922878, see comment #8, we
// may only need to disable this on XCode <= 10.2.
//
// Disable when not using Clang-Linux, because too many user issues arise from
// compilation variations.
//
// NOTE: Consider guarding by !defined(__APPLE__) when removing Linux-only
// restriction.
//
// __EMSCRIPTEN__ is checked because the runtime Path resolution can use asm.
//
// The Android NDK logic excludes earlier and very broken versions of intrinsics
// headers.
#if defined(RUY_FORCE_ENABLE_X86_ENHANCEMENTS) ||                          \
    (defined(__clang__) && (__clang_major__ >= 8) && defined(__linux__) && \
     !defined(__EMSCRIPTEN__) &&                                           \
     (!defined(__ANDROID_NDK__) ||                                         \
      (defined(__NDK_MAJOR__) && (__NDK_MAJOR__ >= 20))))
#define RUY_DONOTUSEDIRECTLY_X86_ENHANCEMENTS 1
#else
#define RUY_DONOTUSEDIRECTLY_X86_ENHANCEMENTS 0
#endif

// These CPU capabilities will all be true when Skylake, etc, are enabled during
// compilation.
#if RUY_PLATFORM(X86_ENHANCEMENTS) && RUY_PLATFORM(X86) &&                    \
    defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512CD__) && \
    defined(__AVX512BW__) && defined(__AVX512VL__)
#define RUY_DONOTUSEDIRECTLY_AVX512 1
#else
#define RUY_DONOTUSEDIRECTLY_AVX512 0
#endif

#if RUY_PLATFORM(X86_ENHANCEMENTS) && RUY_PLATFORM(X86) && defined(__AVX2__)
#define RUY_DONOTUSEDIRECTLY_AVX2 1
#else
#define RUY_DONOTUSEDIRECTLY_AVX2 0
#endif

// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// Note does not check for LZCNT or POPCNT.
#if defined(RUY_ENABLE_SSE_ENHANCEMENTS) && RUY_PLATFORM(X86_ENHANCEMENTS) && \
    RUY_PLATFORM(X86) && defined(__SSE4_2__) && defined(__FMA__)
#define RUY_DONOTUSEDIRECTLY_SSE42 1
#else
#define RUY_DONOTUSEDIRECTLY_SSE42 0
#endif

// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// Note that defined(__AVX512VBMI2__) can be false for compilation with
// -march=cascadelake.
// TODO(b/146646451) Check if we should also gate on defined(__AVX512VBMI2__).
#if defined(RUY_ENABLE_VNNI_ENHANCEMENTS) && RUY_PLATFORM(AVX512) && \
    defined(__AVX512VNNI__)
#define RUY_DONOTUSEDIRECTLY_AVX_VNNI 1
#else
#define RUY_DONOTUSEDIRECTLY_AVX_VNNI 0
#endif

// Detect APPLE.
#ifdef __APPLE__
#define RUY_DONOTUSEDIRECTLY_APPLE 1
#else
#define RUY_DONOTUSEDIRECTLY_APPLE 0
#endif

// Detect Emscripten, typically Wasm.
#ifdef __EMSCRIPTEN__
#define RUY_DONOTUSEDIRECTLY_EMSCRIPTEN 1
#else
#define RUY_DONOTUSEDIRECTLY_EMSCRIPTEN 0
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PLATFORM_H_
