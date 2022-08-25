/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_STACKTRACE_H_
#define TENSORFLOW_CORE_PLATFORM_STACKTRACE_H_

#include "tensorflow/core/platform/platform.h"  // IWYU pragma: export

// Include appropriate platform-dependent implementation.
#if defined(PLATFORM_GOOGLE)
#include "tensorflow/tsl/platform/google/stacktrace.h"  // IWYU pragma: export
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS)
#include "tensorflow/tsl/platform/default/stacktrace.h"  // IWYU pragma: export
#elif defined(PLATFORM_WINDOWS)
#include "tensorflow/tsl/platform/windows/stacktrace.h"  // IWYU pragma: export
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_STACKTRACE_H_
