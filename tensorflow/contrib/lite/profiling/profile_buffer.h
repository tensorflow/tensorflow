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
#ifndef TENSORFLOW_CONTRIB_LITE_PROFILING_PROFILE_BUFFER_H_
#define TENSORFLOW_CONTRIB_LITE_PROFILING_PROFILE_BUFFER_H_

#include <cstddef>
#include <cstdint>

namespace tflite {
namespace profiling {

// A profiling event.
struct ProfileEvent {
  // Describes the type of event.
  // The event_metadata field may contain additional data for interpreting
  // the event.
  enum class EventType {
    // Default event type, the metadata field has no special significance.
    DEFAULT = 0,
    // The event is an operator invocation and the event_metadata field is the
    // index of operator node.
    OPERATOR_INVOKE_EVENT = 1
  };

  // Label of the event. This usually describes the event.
  const char* tag;
  // Timestamp in microseconds when the event began.
  int64_t begin_timestamp_us;
  // Timestamp in microseconds when the event ended.
  int64_t end_timestamp_us;
  // The field containing the type of event. This must be one of the event types
  // in EventType.
  EventType event_type;
  // Extra data describing the details of the event.
  uint32_t event_metadata;
};
}  // namespace profiling
}  // namespace tflite

#ifdef TFLITE_PROFILING_ENABLED

#include <sys/time.h>
#include <vector>

namespace tflite {
namespace profiling {
constexpr uint32_t kInvalidEventHandle = static_cast<uint32_t>(~0) - 1;

// A ring buffer of profile events.
// This class is not thread safe.
class ProfileBuffer {
 public:
  ProfileBuffer(uint32_t max_num_entries, bool enabled)
      : enabled_(enabled), current_index_(0), event_buffer_(max_num_entries) {}

  // Adds an event to the buffer with begin timestamp set to the current
  // timestamp. Returns a handle to event that can be used to call EndEvent. If
  // buffer is disabled this has no affect.
  // The tag of the event should remain valid till the buffer is valid.
  uint32_t BeginEvent(const char* tag, ProfileEvent::EventType event_type,
                      uint32_t event_metadata) {
    if (!enabled_) {
      return kInvalidEventHandle;
    }
    int64_t timestamp = NowMicros();
    int index = current_index_ % event_buffer_.size();
    event_buffer_[index].tag = tag;
    event_buffer_[index].event_type = event_type;
    event_buffer_[index].event_metadata = event_metadata;
    event_buffer_[index].begin_timestamp_us = timestamp;
    event_buffer_[index].end_timestamp_us = 0;
    current_index_++;
    return index;
  }

  // Sets the enabled state of buffer to |enabled|
  void SetEnabled(bool enabled) { enabled_ = enabled; }

  // Sets the end timestamp for event for the handle to current time.
  // If the buffer is disabled or previous event has been overwritten this
  // operation has not effect.
  void EndEvent(uint32_t event_handle) {
    if (!enabled_ || event_handle == kInvalidEventHandle ||
        event_handle > current_index_) {
      return;
    }
    const uint32_t max_size = event_buffer_.size();
    if (current_index_ > (max_size + event_handle)) {
      // Ignore, buffer has already overflowed.
      return;
    }

    int event_index = event_handle % max_size;
    event_buffer_[event_index].end_timestamp_us = NowMicros();
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
  const struct ProfileEvent* const At(int index) const {
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
  static int64_t NowMicros() {
    // TODO(shashishekhar): Refactor this to a separate file.
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }
  bool enabled_;
  uint32_t current_index_;
  std::vector<ProfileEvent> event_buffer_;
};
}  // namespace profiling
}  // namespace tflite
#endif  // TFLITE_PROFILING_ENABLED
#endif  // TENSORFLOW_CONTRIB_LITE_PROFILING_PROFILE_BUFFER_H_
