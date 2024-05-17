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

#ifndef TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_ANDROID_ARMV7A_CPU_UTILS_HELPER_H_
#define TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_ANDROID_ARMV7A_CPU_UTILS_HELPER_H_

#include <sys/types.h>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/profile_utils/i_cpu_utils_helper.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/profile_utils/android_armv7a_cpu_utils_helper.h"

#if defined(__ANDROID__) && (__ANDROID_API__ >= 21) && \
    (defined(__ARM_ARCH_7A__) || defined(__aarch64__))

struct perf_event_attr;

namespace tensorflow {
namespace profile_utils {
using tsl::profile_utils::AndroidArmV7ACpuUtilsHelper;  // NOLINT
}  // namespace profile_utils
}  // namespace tensorflow

#endif  // defined(__ANDROID__) && (__ANDROID_API__ >= 21) &&
        // (defined(__ARM_ARCH_7A__) || defined(__aarch64__))

#endif  // TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_ANDROID_ARMV7A_CPU_UTILS_HELPER_H_
