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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_CPU_CHECK_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_CPU_CHECK_H_

#include "tensorflow/lite/kernels/cpu_backend_context.h"

#if !defined(NEON_OR_PORTABLE_USE_PORTABLE) && \
    !defined(NEON_OR_PORTABLE_USE_NEON)
// If neither is defined, figure out if we can use NEON_OR_PORTABLE_USE_PORTABLE
// or NEON_OR_PORTABLE_USE_NEON

#if defined(__ARM_ARCH_5TE__)
// NEON isn't available at all on ARMv5.
#define NEON_OR_PORTABLE_USE_PORTABLE

#elif defined(__aarch64__)
// A64 always has NEON support.
#define NEON_OR_PORTABLE_USE_NEON

#elif defined(__ANDROID__)
// Runtime check for NEON support on Android.

#include "ndk/sources/android/cpufeatures/cpu-features.h"

namespace tflite {

inline bool TestCPUFeatureNeon() {
  static const bool kUseAndroidNeon =
      (android_getCpuFamily() == ANDROID_CPU_FAMILY_ARM &&
       android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_ARMv7 &&
       android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_NEON);
  return kUseAndroidNeon;
}

}  // namespace tflite

#elif defined USE_NEON || defined __ARM_NEON
// Non-Android build using NEON
#define NEON_OR_PORTABLE_USE_NEON

#else
// All else: use Portable
#define NEON_OR_PORTABLE_USE_PORTABLE

#endif  // 'switch' among architectures

#endif  // !defined(NEON_OR_PORTABLE_USE_PORTABLE) &&
        // !defined(NEON_OR_PORTABLE_USE_NEON)

namespace tflite {

struct CpuFlags {
  bool neon_dotprod = false;
};

inline void GetCpuFlags(CpuBackendContext* cpu_backend_context,
                        CpuFlags* cpu_flags) {
  ruy::Context* ruy_context = cpu_backend_context->ruy_context();
  cpu_flags->neon_dotprod =
      ruy_context != nullptr && (ruy_context->GetRuntimeEnabledPaths() &
                                 ruy::Path::kNeonDotprod) != ruy::Path::kNone;
}

}  // namespace tflite

// NEON_OR_PORTABLE(SomeFunc, args) calls:
//  NeonSomeFunc(args) if NEON_OR_PORTABLE_USE_NEON is defined, or
//  PortableSomeFunc(args) if NEON_OR_PORTABLE_USE_PORTABLE is defined, or
// detects Neon at runtime and calls the appropriate version.

#if defined(NEON_OR_PORTABLE_USE_NEON)
// Always use Neon code
#define NEON_OR_PORTABLE(funcname, ...) Neon##funcname(__VA_ARGS__)

#elif defined(NEON_OR_PORTABLE_USE_PORTABLE)
// Always use Portable code
#define NEON_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)

#else
// Detect availability of Neon at runtime
#define NEON_OR_PORTABLE(funcname, ...)              \
  TestCPUFeatureNeon() ? Neon##funcname(__VA_ARGS__) \
                       : Portable##funcname(__VA_ARGS__)
#endif

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_CPU_CHECK_H_
