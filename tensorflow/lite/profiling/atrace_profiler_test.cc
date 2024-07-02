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
#include "tensorflow/lite/profiling/atrace_profiler.h"

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include <gtest/gtest.h>

namespace tflite {
namespace profiling {

namespace {

TEST(ATraceProfilerTest, MaybeCreateATraceProfiler) {
  auto initial_state_profiler = MaybeCreateATraceProfiler();
#if !defined(TFLITE_ENABLE_DEFAULT_PROFILER)
  EXPECT_EQ(nullptr, initial_state_profiler.get());
#else
  EXPECT_NE(nullptr, initial_state_profiler.get());
#endif

#if defined(__ANDROID__)
  if (__system_property_set("debug.tflite.trace", "1") == 0) {
    auto on_state_profiler = MaybeCreateATraceProfiler();
    EXPECT_NE(nullptr, on_state_profiler.get());
  }

  if (__system_property_set("debug.tflite.trace", "0") == 0) {
    auto off_state_profiler = MaybeCreateATraceProfiler();
#if !defined(TFLITE_ENABLE_DEFAULT_PROFILER)
    EXPECT_EQ(nullptr, off_state_profiler.get());
#else
    EXPECT_NE(nullptr, off_state_profiler.get());
#endif
  }
#endif  // __ANDROID__
}

}  // namespace
}  // namespace profiling
}  // namespace tflite
