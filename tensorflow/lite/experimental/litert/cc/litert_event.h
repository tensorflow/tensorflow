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

#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_event_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

extern "C" {
// Forward declaration of OpenCL event to avoid including OpenCL headers.
typedef struct _cl_event* cl_event;
}

namespace litert {

class Event : public internal::Handle<LiteRtEvent, LiteRtDestroyEvent> {
 public:
  // Parameter `owned` indicates if the created TensorBufferRequirements object
  // should take ownership of the provided `requirements` handle.
  explicit Event(LiteRtEvent event, bool owned = true)
      : internal::Handle<LiteRtEvent, LiteRtDestroyEvent>(event, owned) {}

  // Creates an Event object with the given `sync_fence_fd`.
  static Expected<Event> CreateFromSyncFenceFd(int sync_fence_fd,
                                               bool owns_fd) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(
        LiteRtCreateEventFromSyncFenceFd(sync_fence_fd, owns_fd, &event));
    return Event(event);
  }

  // Creates an Event object with the given `cl_event`.
  static Expected<Event> CreateFromOpenClEvent(cl_event cl_event) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(LiteRtCreateEventFromOpenClEvent(cl_event, &event));
    return Event(event);
  }

  // Creates a managed event of the given `type`. Currently only
  // LiteRtEventTypeOpenCl is supported.
  static Expected<Event> CreateManaged(LiteRtEventType type) {
    LiteRtEvent event;
    LITERT_RETURN_IF_ERROR(LiteRtCreateManagedEvent(type, &event));
    return Event(event);
  }

  Expected<int> GetSyncFenceFd() {
    int fd;
    LITERT_RETURN_IF_ERROR(LiteRtGetEventSyncFenceFd(Get(), &fd));
    return fd;
  }

  // Returns the underlying OpenCL event if the event type is OpenCL.
  Expected<cl_event> GetOpenClEvent() {
    cl_event cl_event;
    LITERT_RETURN_IF_ERROR(LiteRtGetEventOpenClEvent(Get(), &cl_event));
    return cl_event;
  }

  // Pass -1 for timeout_in_ms for indefinite wait.
  Expected<void> Wait(int64_t timeout_in_ms) {
    LITERT_RETURN_IF_ERROR(LiteRtEventWait(Get(), timeout_in_ms));
    return {};
  }

  // Singal the event.
  // Note: This is only supported for OpenCL events.
  Expected<void> Signal() {
    LITERT_RETURN_IF_ERROR(LiteRtEventSignal(Get()));
    return {};
  }

  // Returns the underlying event type.
  LiteRtEventType Type() const {
    LiteRtEventType type;
    LiteRtGetEventEventType(Get(), &type);
    return type;
  }
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EVENT_H_
