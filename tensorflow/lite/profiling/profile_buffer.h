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
#ifndef TENSORFLOW_LITE_PROFILING_PROFILE_BUFFER_H_
#define TENSORFLOW_LITE_PROFILING_PROFILE_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace profiling {

constexpr uint32_t kInvalidEventHandle = static_cast<uint32_t>(~0) - 1;

// A profiling event.
struct ProfileEvent {
  // Describes the type of event.
  // The event_metadata field may contain additional data for interpreting
  // the event.
  using EventType = tflite::Profiler::EventType;

  // Label of the event. This usually describes the event.
  std::string tag;
  // Timestamp in microseconds when the event began.
  uint64_t begin_timestamp_us;
  // Event processing time in microseconds.
  uint64_t elapsed_time;

  // The memory usage when the event begins.
  memory::MemoryUsage begin_mem_usage;
  // The memory usage when the event ends.
  memory::MemoryUsage end_mem_usage;

  // The field containing the type of event. This must be one of the event types
  // in EventType.
  EventType event_type;
  // Meta data associated w/ the event.
  int64_t event_metadata;
  // Note: if this is an OPERATOR_INVOKE_EVENT, 'extra_event_metadata' will
  // represent the index of the subgraph that this event comes from.
  int64_t extra_event_metadata;
};

// A buffer of profile events. In general, the buffer works like a ring buffer.
// However, when 'allow_dynamic_expansion' is set, a unlimitted number of buffer
// entries is allowed and more profiling overhead could occur.
// This class is *not thread safe*.
class ProfileBuffer {
 public:
  ProfileBuffer(uint32_t max_num_entries, bool enabled,
                bool allow_dynamic_expansion = false)
      : enabled_(enabled),
        current_index_(0),
        event_buffer_(max_num_entries),
        allow_dynamic_expansion_(allow_dynamic_expansion) {}

  // Adds an event to the buffer with begin timestamp set to the current
  // timestamp. Returns a handle to event that can be used to call EndEvent. If
  // buffer is disabled this has no affect.
  // The tag of the event should remain valid till the buffer is valid.
  uint32_t BeginEvent(const char* tag, ProfileEvent::EventType event_type,
                      int64_t event_metadata1, int64_t event_metadata2);

  // Sets the enabled state of buffer to |enabled|
  void SetEnabled(bool enabled) { enabled_ = enabled; }

  // Sets the end timestamp for event for the handle to current time.
  // If the buffer is disabled or previous event has been overwritten this
  // operation has not effect.
  void EndEvent(uint32_t event_handle, const int64_t* event_metadata1 = nullptr,
                const int64_t* event_metadata2 = nullptr);

  void AddEvent(const char* tag, ProfileEvent::EventType event_type,
                uint64_t elapsed_time, int64_t event_metadata1,
                int64_t event_metadata2);

  // Returns the size of the buffer.
  size_t Size() const {
    return (current_index_ >= event_buffer_.size()) ? event_buffer_.size()
                                                    : current_index_;
  }

  // Resets the buffer.
  void Reset() {
    enabled_ = false;
    current_index_ = 0;
  }

  // Returns the profile event at the given index. If the index is invalid a
  // nullptr is returned. The return event may get overwritten if more events
  // are added to buffer.
  const struct ProfileEvent* At(size_t index) const;

 private:
  // Returns a pair of values. The 1st element refers to the next buffer id,
  // the 2nd element refers to whether the buffer reaches its allowed capacity.
  std::pair<int, bool> GetNextEntryIndex();

  bool enabled_;
  uint32_t current_index_;
  std::vector<ProfileEvent> event_buffer_;
  const bool allow_dynamic_expansion_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_PROFILE_BUFFER_H_
