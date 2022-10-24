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
                      int64_t event_metadata1, int64_t event_metadata2) {
    if (!enabled_) {
      return kInvalidEventHandle;
    }
    uint64_t timestamp = time::NowMicros();
    const auto next_index = GetNextEntryIndex();
    if (next_index.second) {
      return next_index.first;
    }
    const int index = next_index.first;
    event_buffer_[index].tag = tag;
    event_buffer_[index].event_type = event_type;
    event_buffer_[index].event_metadata = event_metadata1;
    event_buffer_[index].extra_event_metadata = event_metadata2;
    event_buffer_[index].begin_timestamp_us = timestamp;
    event_buffer_[index].elapsed_time = 0;
    if (event_type != Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      event_buffer_[index].begin_mem_usage = memory::GetMemoryUsage();
    }
    current_index_++;
    return index;
  }

  // Sets the enabled state of buffer to |enabled|
  void SetEnabled(bool enabled) { enabled_ = enabled; }

  // Sets the end timestamp for event for the handle to current time.
  // If the buffer is disabled or previous event has been overwritten this
  // operation has not effect.
  void EndEvent(uint32_t event_handle, const int64_t* event_metadata1 = nullptr,
                const int64_t* event_metadata2 = nullptr) {
    if (!enabled_ || event_handle == kInvalidEventHandle ||
        event_handle > current_index_) {
      return;
    }
    const uint32_t max_size = event_buffer_.size();
    if (current_index_ > (max_size + event_handle)) {
      // Ignore, buffer has already overflowed.
      fprintf(stderr, "Warning: Dropping ProfileBuffer event.\n");
      return;
    }

    int event_index = event_handle % max_size;
    event_buffer_[event_index].elapsed_time =
        time::NowMicros() - event_buffer_[event_index].begin_timestamp_us;
    if (event_buffer_[event_index].event_type !=
        Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      event_buffer_[event_index].end_mem_usage = memory::GetMemoryUsage();
    }
    if (event_metadata1) {
      event_buffer_[event_index].event_metadata = *event_metadata1;
    }
    if (event_metadata2) {
      event_buffer_[event_index].extra_event_metadata = *event_metadata2;
    }
  }

  void AddEvent(const char* tag, ProfileEvent::EventType event_type,
                uint64_t elapsed_time, int64_t event_metadata1,
                int64_t event_metadata2) {
    if (!enabled_) {
      return;
    }
    const auto next_index = GetNextEntryIndex();
    if (next_index.second) {
      return;
    }
    const int index = next_index.first;
    event_buffer_[index].tag = tag;
    event_buffer_[index].event_type = event_type;
    event_buffer_[index].event_metadata = event_metadata1;
    event_buffer_[index].extra_event_metadata = event_metadata2;
    event_buffer_[index].begin_timestamp_us = 0;
    event_buffer_[index].elapsed_time = elapsed_time;
    current_index_++;
  }

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
  const struct ProfileEvent* At(size_t index) const {
    size_t size = Size();
    if (index >= size) {
      return nullptr;
    }
    const uint32_t max_size = event_buffer_.size();
    uint32_t start =
        (current_index_ > max_size) ? current_index_ % max_size : max_size;
    index = (index + start) % max_size;
    return &event_buffer_[index];
  }

 private:
  // Returns a pair of values. The 1st element refers to the next buffer id,
  // the 2nd element refers to whether the buffer reaches its allowed capacity.
  std::pair<int, bool> GetNextEntryIndex() {
    int index = current_index_ % event_buffer_.size();
    if (current_index_ == 0 || index != 0) {
      return std::make_pair(index, false);
    }

    // Current buffer is full
    if (!allow_dynamic_expansion_) {
      fprintf(stderr, "Warning: Dropping ProfileBuffer event.\n");
      return std::make_pair(current_index_, true);
    } else {
      fprintf(stderr, "Warning: Doubling internal profiling buffer.\n");
      event_buffer_.resize(current_index_ * 2);
      return std::make_pair(current_index_, false);
    }
  }

  bool enabled_;
  uint32_t current_index_;
  std::vector<ProfileEvent> event_buffer_;
  const bool allow_dynamic_expansion_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_PROFILE_BUFFER_H_
