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

#ifndef TENSORFLOW_PLATFORM_PLATFORM_DEFINE_H_
#define TENSORFLOW_PLATFORM_PLATFORM_DEFINE_H_

// Set one PLATFORM_* macro and set IS_MOBILE_PLATFORM if the platform is for
// mobile.

#if !defined(PLATFORM_POSIX) && !defined(PLATFORM_GOOGLE) &&                 \
    !defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID) && \
    !defined(PLATFORM_WINDOWS)

// Choose which platform we are on.
#if defined(ANDROID) || defined(__ANDROID__)
#define PLATFORM_POSIX_ANDROID
#define IS_MOBILE_PLATFORM

#elif defined(__APPLE__)
#define PLATFORM_POSIX
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
#define IS_MOBILE_PLATFORM
#elif TARGET_OS_IPHONE
#define IS_MOBILE_PLATFORM
#endif

#elif defined(_WIN32)
#define PLATFORM_WINDOWS

#elif defined(__arm__)
#define PLATFORM_POSIX

// Since there's no macro for the Raspberry Pi, assume we're on a mobile
// platform if we're compiling for the ARM CPU.
#define IS_MOBILE_PLATFORM

#else
// If no platform specified, use:
#define PLATFORM_POSIX

#endif
#endif

// Look for both gcc/clang and Visual Studio macros indicating we're compiling
// for an x86 device.
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_IX86) || \
    defined(_M_X64)
#define PLATFORM_IS_X86
#endif

#endif  // TENSORFLOW_PLATFORM_PLATFORM_DEFINE_H_
