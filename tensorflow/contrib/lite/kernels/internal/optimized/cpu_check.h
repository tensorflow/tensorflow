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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CPU_CHECK_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CPU_CHECK_

namespace tflite {

#ifdef __ANDROID__
#include "ndk/sources/android/cpufeatures/cpu-features.h"

// Runtime check for Neon support on Android.
inline bool TestCPUFeatureNeon() {
#ifdef __aarch64__
  // ARM-64 always has NEON support.
  return true;
#else
  static bool kUseAndroidNeon =
      (android_getCpuFamily() == ANDROID_CPU_FAMILY_ARM &&
       android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_ARMv7 &&
       android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_NEON);
  return kUseAndroidNeon;
#endif  // __aarch64__
}

#elif defined USE_NEON || defined __ARM_NEON 

inline bool TestCPUFeatureNeon() { return true; }

#else

inline bool TestCPUFeatureNeon() { return false; }

#endif

}  // namespace tflite

// NEON_OR_PORTABLE(SomeFunc, arcs) calls NeonSomeFunc(args) if Neon is both
// enabled at build time and detected at runtime, or PortableSomeFunc(args)
// otherwise.
#ifdef __ARM_ARCH_5TE__
// Neon isn't available at all on ARMv5.
#define NEON_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)
#else
#define NEON_OR_PORTABLE(funcname, ...)              \
  TestCPUFeatureNeon() ? Neon##funcname(__VA_ARGS__) \
                       : Portable##funcname(__VA_ARGS__)
#endif

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CPU_CHECK_
