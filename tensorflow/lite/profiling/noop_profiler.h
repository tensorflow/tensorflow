/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_PROFILING_NOOP_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_NOOP_PROFILER_H_

#include <vector>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/profiling/profile_buffer.h"

namespace tflite {
namespace profiling {

// A noop version of profiler when profiling is disabled.
class NoopProfiler : public tflite::Profiler {
 public:
  NoopProfiler() {}
  explicit NoopProfiler(int max_profiling_buffer_entries) {}

  uint32_t BeginEvent(const char*, EventType, uint32_t, uint32_t) override {
    return 0;
  }
  void EndEvent(uint32_t) override {}

  void StartProfiling() {}
  void StopProfiling() {}
  void Reset() {}
  std::vector<const ProfileEvent*> GetProfileEvents() { return {}; }
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_NOOP_PROFILER_H_
