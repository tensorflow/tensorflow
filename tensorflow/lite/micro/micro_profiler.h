/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"

namespace tflite {

// MicroProfiler creates a common way to gain fine-grained insight into runtime
// performance. Bottleck operators can be identified along with slow code
// sections. This can be used in conjunction with running the relevant micro
// benchmark to evaluate end-to-end performance.
class MicroProfiler {
 public:
  MicroProfiler() = default;
  virtual ~MicroProfiler() = default;

  // Marks the start of a new event and returns an event handle that can be used
  // to mark the end of the event via EndEvent. The lifetime of the tag
  // parameter must exceed that of the MicroProfiler.
  virtual uint32_t BeginEvent(const char* tag);

  // Marks the end of an event associated with event_handle. It is the
  // responsibility of the caller to ensure than EndEvent is called once and
  // only once per event_handle.
  //
  // If EndEvent is called more than once for the same event_handle, the last
  // call will be used as the end of event marker.If EndEvent is called 0 times
  // for a particular event_handle, the duration of that event will be 0 ticks.
  virtual void EndEvent(uint32_t event_handle);

  // Clears all the events that have been currently profiled.
  void ClearEvents() { num_events_ = 0; }

  // Returns the sum of the ticks taken across all the events. This number
  // is only meaningful if all of the events are disjoint (the end time of
  // event[i] <= start time of event[i+1]).
  int32_t GetTotalTicks() const;

  // Prints the profiling information of each of the events.
  void Log() const;

 private:
  // Maximum number of events that this class can keep track of. If we call
  // AddEvent more than kMaxEvents number of times, then the oldest event's
  // profiling information will be overwritten.
  static constexpr int kMaxEvents = 50;

  const char* tags_[kMaxEvents];
  int32_t start_ticks_[kMaxEvents];
  int32_t end_ticks_[kMaxEvents];
  int num_events_ = 0;

  TF_LITE_REMOVE_VIRTUAL_DELETE;
};

#if defined(TF_LITE_STRIP_ERROR_STRINGS)
// For release builds, the ScopedMicroProfiler is a noop.
//
// This is done because the ScipedProfiler is used as part of the
// MicroInterpreter and we want to ensure zero overhead for the release builds.
class ScopedMicroProfiler {
 public:
  explicit ScopedMicroProfiler(const char* tag, MicroProfiler* profiler) {}
};

#else

// This class can be used to add events to a MicroProfiler object that span the
// lifetime of the ScopedMicroProfiler object.
// Usage example:
//
// MicroProfiler profiler();
// ...
// {
//   ScopedMicroProfiler scoped_profiler("custom_tag", profiler);
//   work_to_profile();
// }
class ScopedMicroProfiler {
 public:
  explicit ScopedMicroProfiler(const char* tag, MicroProfiler* profiler)
      : profiler_(profiler) {
    if (profiler_ != nullptr) {
      event_handle_ = profiler_->BeginEvent(tag);
    }
  }

  ~ScopedMicroProfiler() {
    if (profiler_ != nullptr) {
      profiler_->EndEvent(event_handle_);
    }
  }

 private:
  uint32_t event_handle_ = 0;
  MicroProfiler* profiler_ = nullptr;
};
#endif  // !defined(NDEBUG)

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_PROFILER_H_
