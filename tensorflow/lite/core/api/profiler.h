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
#ifndef TENSORFLOW_LITE_CORE_API_PROFILER_H_
#define TENSORFLOW_LITE_CORE_API_PROFILER_H_

#include <cstdint>

namespace tflite {

// A simple utility for enabling profiled event tracing in TensorFlow Lite.
class Profiler {
 public:
  enum class EventType {
    // Default event type, the metadata field has no special significance.
    DEFAULT = 0,

    // The event is an operator invocation and the event_metadata field is the
    // index of operator node.
    OPERATOR_INVOKE_EVENT = 1,

    // The event is an invocation for an internal operator of a TFLite delegate.
    // The event_metadata field is the index of operator node that's specific to
    // the delegate.
    DELEGATE_OPERATOR_INVOKE_EVENT = 2
  };

  virtual ~Profiler() {}

  // Signals the beginning of an event from a subgraph indexed at
  // 'event_subgraph_index', returning a handle to the profile event.
  virtual uint32_t BeginEvent(const char* tag, EventType event_type,
                              uint32_t event_metadata,
                              uint32_t event_subgraph_index) = 0;
  // Similar w/ the above, but the event comes from the primary subgraph that's
  // indexed at 0.
  virtual uint32_t BeginEvent(const char* tag, EventType event_type,
                              uint32_t event_metadata) {
    return BeginEvent(tag, event_type, event_metadata, /*primary subgraph*/ 0);
  }

  // Signals an end to the specified profile event.
  virtual void EndEvent(uint32_t event_handle) = 0;
};

// Adds a profile event to `profiler` that begins with the construction
// of the object and ends when the object goes out of scope.
// The lifetime of tag should be at least the lifetime of `profiler`.
// `profiler` may be null, in which case nothing is profiled.
class ScopedProfile {
 public:
  ScopedProfile(Profiler* profiler, const char* tag,
                Profiler::EventType event_type = Profiler::EventType::DEFAULT,
                uint32_t event_metadata = 0)
      : profiler_(profiler), event_handle_(0) {
    if (profiler) {
      event_handle_ = profiler_->BeginEvent(tag, event_type, event_metadata);
    }
  }

  ~ScopedProfile() {
    if (profiler_) {
      profiler_->EndEvent(event_handle_);
    }
  }

 private:
  Profiler* const profiler_;
  uint32_t event_handle_;
};

class ScopedOperatorProfile : public ScopedProfile {
 public:
  ScopedOperatorProfile(Profiler* profiler, const char* tag, int node_index)
      : ScopedProfile(profiler, tag, Profiler::EventType::OPERATOR_INVOKE_EVENT,
                      static_cast<uint32_t>(node_index)) {}
};

class ScopedDelegateOperatorProfile : public ScopedProfile {
 public:
  ScopedDelegateOperatorProfile(Profiler* profiler, const char* tag,
                                int node_index)
      : ScopedProfile(profiler, tag,
                      Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT,
                      static_cast<uint32_t>(node_index)) {}
};

}  // namespace tflite

#define TFLITE_VARNAME_UNIQ(name, ctr) name##ctr

#define TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler, tag)          \
  tflite::ScopedProfile TFLITE_VARNAME_UNIQ(_profile_, __COUNTER__)( \
      (profiler), (tag))

#define TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE(profiler, tag, node_index)     \
  tflite::ScopedOperatorProfile TFLITE_VARNAME_UNIQ(_profile_, __COUNTER__)( \
      (profiler), (tag), (node_index))

#define TFLITE_SCOPED_DELEGATE_OPERATOR_PROFILE(profiler, tag, node_index) \
  tflite::ScopedDelegateOperatorProfile TFLITE_VARNAME_UNIQ(               \
      _profile_, __COUNTER__)((profiler), (tag), (node_index))

#endif  // TENSORFLOW_LITE_CORE_API_PROFILER_H_
