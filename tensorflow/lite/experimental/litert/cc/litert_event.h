// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EVENT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EVENT_H_

#include <cstdint>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"

namespace litert {

class Event : public internal::Handle<LiteRtEvent, LiteRtDestroyEvent> {
 public:
  // Parameter `owned` indicates if the created TensorBufferRequirements object
  // should take ownership of the provided `requirements` handle.
  explicit Event(LiteRtEvent event, bool owned = true)
      : internal::Handle<LiteRtEvent, LiteRtDestroyEvent>(event, owned) {}

  static Expected<Event> CreateFromSyncFenceFd(int sync_fence_fd,
                                               bool owns_fd) {
    LiteRtEvent event;
    if (auto status =
            LiteRtCreateEventFromSyncFenceFd(sync_fence_fd, owns_fd, &event);
        status != kLiteRtStatusOk) {
      return Error(status, "Failed to create event from sync fence fd");
    }
    return Event(event);
  }

  Expected<int> GetSyncFenceFd(LiteRtEvent event) {
    int fd;
    if (auto status = LiteRtGetEventSyncFenceFd(Get(), &fd);
        status != kLiteRtStatusOk) {
      return Error(status, "Failed to get sync fence fd from event");
    }
    return fd;
  }

  // Pass -1 for timeout_in_ms for indefinite wait.
  Expected<void> Wait(int64_t timeout_in_ms) {
    if (auto status = LiteRtEventWait(Get(), timeout_in_ms);
        status != kLiteRtStatusOk) {
      return Error(status, "Failed to wait on event");
    }
    return {};
  }
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EVENT_H_
