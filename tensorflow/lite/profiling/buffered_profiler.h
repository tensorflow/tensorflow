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
#ifndef TENSORFLOW_LITE_PROFILING_BUFFERED_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_BUFFERED_PROFILER_H_

#include <cstdint>
#include <vector>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/profiling/profile_buffer.h"

namespace tflite {
namespace profiling {

// Controls whether profiling is enabled or disabled and collects profiles.
// TFLite is used on platforms that don't have posix threads, so the profiler is
// kept as simple as possible. It is designed to be used only on a single
// thread.
//
// Profiles are collected using Scoped*Profile objects that begin and end a
// profile event.
// An example usage is shown in the example below:
//
// Say Worker class has a DoWork method and we are interested in profiling
// the overall execution time for DoWork and time spent in Task1 and Task2
// functions.
//
// class Worker {
//  public:
//   void DoWork() {
//    ScopedProfile(&controller, "DoWork");
//    Task1();
//    Task2();
//    .....
//   }
//
//   void Task1() {
//    ScopedProfile(&controller, "Task1");
//    ....
//   }
//
//   void Task2() {
//    ScopedProfile(&controller, "Task2");
//   }
//
//    Profiler profiler;
// }
//
// We instrument the functions that need to be profiled.
//
// Profile can be collected by enable profiling and then getting profile
// events.
//
//  void ProfileWorker() {
//    Worker worker;
//    worker.profiler.EnableProfiling();
//    worker.DoWork();
//    worker.profiler.DisableProfiling();
//    // Profiling is complete, extract profiles.
//    auto profile_events = worker.profiler.GetProfiles();
//  }
//
//
class BufferedProfiler : public tflite::Profiler {
 public:
  BufferedProfiler(uint32_t max_num_initial_entries,
                   bool allow_dynamic_buffer_increase)
      : buffer_(max_num_initial_entries, false /*enabled*/,
                allow_dynamic_buffer_increase),
        supported_event_types_(~static_cast<uint64_t>(
            EventType::GENERAL_RUNTIME_INSTRUMENTATION_EVENT)) {}

  explicit BufferedProfiler(uint32_t max_num_entries)
      : BufferedProfiler(max_num_entries,
                         false /*allow_dynamic_buffer_increase*/) {}

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override {
    if (!ShouldAddEvent(event_type)) return kInvalidEventHandle;
    return buffer_.BeginEvent(tag, event_type, event_metadata1,
                              event_metadata2);
  }

  void EndEvent(uint32_t event_handle) override {
    buffer_.EndEvent(event_handle);
  }

  void EndEvent(uint32_t event_handle, int64_t event_metadata1,
                int64_t event_metadata2) override {
    buffer_.EndEvent(event_handle, &event_metadata1, &event_metadata2);
  }

  void AddEvent(const char* tag, EventType event_type, uint64_t start,
                uint64_t end, int64_t event_metadata1,
                int64_t event_metadata2) override {
    if (!ShouldAddEvent(event_type)) return;
    buffer_.AddEvent(tag, event_type, start, end, event_metadata1,
                     event_metadata2);
  }

  void StartProfiling() { buffer_.SetEnabled(true); }
  void StopProfiling() { buffer_.SetEnabled(false); }
  void Reset() { buffer_.Reset(); }
  std::vector<const ProfileEvent*> GetProfileEvents() {
    std::vector<const ProfileEvent*> profile_events;
    profile_events.reserve(buffer_.Size());
    for (size_t i = 0; i < buffer_.Size(); i++) {
      profile_events.push_back(buffer_.At(i));
    }
    return profile_events;
  }

 protected:
  bool ShouldAddEvent(EventType event_type) {
    return (static_cast<uint64_t>(event_type) & supported_event_types_) != 0;
  }

 private:
  ProfileBuffer* GetProfileBuffer() { return &buffer_; }
  ProfileBuffer buffer_;
  const uint64_t supported_event_types_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_BUFFERED_PROFILER_H_
