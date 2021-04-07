/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/profiling/platform_profiler.h"

#include <memory>

#include "tensorflow/lite/core/api/profiler.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/profiling/atrace_profiler.h"
#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IOS
#define SIGNPOST_PLATFORM_PROFILER
#include "tensorflow/lite/profiling/signpost_profiler.h"
#endif
#endif

namespace tflite {
namespace profiling {

std::unique_ptr<tflite::Profiler> MaybeCreatePlatformProfiler() {
#if defined(__ANDROID__)
  return MaybeCreateATraceProfiler();
#elif defined(SIGNPOST_PLATFORM_PROFILER)
  return MaybeCreateSignpostProfiler();
#else
  return nullptr;
#endif
}

}  // namespace profiling
}  // namespace tflite
