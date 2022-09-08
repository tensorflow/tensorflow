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

#ifndef TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_
#define TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_

#include <algorithm>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/tsl/platform/profile_utils/clock_cycle_profiler.h"

namespace tensorflow {
using tsl::ClockCycleProfiler;  // NOLINT
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CLOCK_CYCLE_PROFILER_H_
