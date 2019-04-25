/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_PROFILING_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_PROFILER_H_

#include <vector>

#include "tensorflow/lite/profiling/profile_buffer.h"

#ifdef TFLITE_PROFILING_ENABLED

namespace tflite {
namespace profiling {
class ScopedProfile;
class ScopedOperatorProfile;

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
class Profiler {
 public:
  Profiler() : buffer_(1024, false) {}

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

 private:
  friend class ScopedProfile;
  friend class ScopedOperatorProfile;
  ProfileBuffer* GetProfileBuffer() { return &buffer_; }
  ProfileBuffer buffer_;
};

class ScopedProfile {
 public:
  // Adds a profile event to profile that begins with the construction
  // of object and ends when the object goes out of scope.
  // The lifetime of tag should be at least the lifetime of profiler.

  ScopedProfile(Profiler* profiler, const char* tag)
      : buffer_(nullptr), event_handle_(0) {
    if (profiler) {
      buffer_ = profiler->GetProfileBuffer();
      event_handle_ =
          buffer_->BeginEvent(tag, ProfileEvent::EventType::DEFAULT, 0);
    }
  }
  ~ScopedProfile() {
    if (buffer_) {
      buffer_->EndEvent(event_handle_);
    }
  }

 private:
  ProfileBuffer* buffer_;
  int32_t event_handle_;
};

class ScopedOperatorProfile {
 public:
  // Adds a profile event to profile that begins with the construction
  // of object and ends when the object goes out of scope.
  // The lifetime of tag should be at least the lifetime of profiler.
  ScopedOperatorProfile(Profiler* profiler, const char* tag, int node_index)
      : buffer_(nullptr), event_handle_(0) {
    if (profiler) {
      buffer_ = profiler->GetProfileBuffer();
      event_handle_ = buffer_->BeginEvent(
          tag, ProfileEvent::EventType::OPERATOR_INVOKE_EVENT, node_index);
    }
  }

  ~ScopedOperatorProfile() {
    if (buffer_) {
      buffer_->EndEvent(event_handle_);
    }
  }

 private:
  ProfileBuffer* buffer_;
  int32_t event_handle_;
};

}  // namespace profiling
}  // namespace tflite

#define VARNAME_UNIQ(name, ctr) name##ctr

#define SCOPED_TAGGED_OPERATOR_PROFILE(profiler, tag, node_index) \
  tflite::profiling::ScopedOperatorProfile VARNAME_UNIQ(          \
      _profile_, __COUNTER__)((profiler), (tag), (node_index))
#define SCOPED_OPERATOR_PROFILE(profiler, node_index) \
  SCOPED_TAGGED_OPERATOR_PROFILE((profiler), "OpInvoke", (node_index))
#else

namespace tflite {
namespace profiling {
// A noop version of profiler when profiling is disabled.
class Profiler {
 public:
  Profiler() {}
  void StartProfiling() {}
  void StopProfiling() {}
  void Reset() {}
  std::vector<const ProfileEvent*> GetProfileEvents() { return {}; }
};
}  // namespace profiling
}  // namespace tflite

#define SCOPED_TAGGED_OPERATOR_PROFILE(profiler, tag, node_index)
#define SCOPED_OPERATOR_PROFILE(profiler, node_index)

#endif  // TFLITE_PROFILING_ENABLED

#endif  // TENSORFLOW_LITE_PROFILING_PROFILER_H_
