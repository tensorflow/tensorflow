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
#ifndef TENSORFLOW_LITE_PROFILING_ATRACE_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_ATRACE_PROFILER_H_

#include <type_traits>

#include "tensorflow/lite/core/api/profiler.h"

namespace tflite {
namespace profiling {

// Profiler reporting to ATrace.
class ATraceProfiler : public tflite::Profiler {
 public:
  ATraceProfiler();

  ~ATraceProfiler() override;

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      uint32_t event_metadata,
                      uint32_t event_subgraph_index) override;

  void EndEvent(uint32_t event_handle) override;

 private:
  using FpIsEnabled = std::add_pointer<bool()>::type;
  using FpBeginSection = std::add_pointer<void(const char*)>::type;
  using FpEndSection = std::add_pointer<void()>::type;

  // Handle to libandroid.so library. Null if not supported.
  void* handle_;
  FpIsEnabled atrace_is_enabled_;
  FpBeginSection atrace_begin_section_;
  FpEndSection atrace_end_section_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_ATRACE_PROFILER_H_
