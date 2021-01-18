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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_PROFILER_H_
#define TENSORFLOW_LITE_MICRO_MICRO_PROFILER_H_

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/micro/compatibility.h"

namespace tflite {

// MicroProfiler creates a common way to gain fine-grained insight into runtime
// performance. Bottleck operators can be identified along with slow code
// sections. This can be used in conjunction with running the relevant micro
// benchmark to evaluate end-to-end performance.
//
// Usage example:
// MicroProfiler profiler(error_reporter);
// {
//   ScopedProfile scoped_profile(profiler, tag);
//   work_to_profile();
// }
//
// This will call the following methods in order:
// int event_handle = profiler->BeginEvent(op_name, EventType::DEFAULT, 0)
// work_to_profile();
// profiler->EndEvent(event_handle)
class MicroProfiler : public tflite::Profiler {
 public:
  explicit MicroProfiler(tflite::ErrorReporter* reporter);
  ~MicroProfiler() override = default;

  // AddEvent is unused for Tf Micro.
  void AddEvent(const char* tag, EventType event_type, uint64_t start,
                uint64_t end, int64_t event_metadata1,
                int64_t event_metadata2) override{};

  // BeginEvent followed by code followed by EndEvent will profile the code
  // enclosed. Multiple concurrent events are unsupported, so the return value
  // is always 0. Event_metadata1 and event_metadata2 are unused. The tag
  // pointer must be valid until EndEvent is called.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  // Event_handle is ignored since TF Micro does not support concurrent events.
  void EndEvent(uint32_t event_handle) override;

 private:
  tflite::ErrorReporter* reporter_;
  int32_t start_time_;
  const char* event_tag_;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_PROFILER_H_
