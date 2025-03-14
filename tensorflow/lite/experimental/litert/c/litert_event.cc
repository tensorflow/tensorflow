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

#include "tensorflow/lite/experimental/litert/c/litert_event.h"

#include <fcntl.h>

#include <cstdint>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event_type.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/runtime/event.h"

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtCreateEventFromSyncFenceFd(int sync_fence_fd, bool owns_fd,
                                              LiteRtEvent* event) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  *event = new LiteRtEventT{.type = LiteRtEventTypeSyncFenceFd,
                            .fd = sync_fence_fd,
                            .owns_fd = owns_fd};
  return kLiteRtStatusOk;
#else
  return kLiteRtStatusErrorUnsupported;
#endif
}

#if LITERT_HAS_OPENCL_SUPPORT
LiteRtStatus LiteRtCreateEventFromOpenClEvent(cl_event cl_event,
                                              LiteRtEvent* event) {
  *event = new LiteRtEventT{
      .type = LiteRtEventTypeOpenCl,
      .opencl_event = cl_event,
  };
  return kLiteRtStatusOk;
}
#endif

LiteRtStatus LiteRtGetEventEventType(LiteRtEvent event, LiteRtEventType* type) {
  *type = event->type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetEventSyncFenceFd(LiteRtEvent event, int* sync_fence_fd) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  if (event->type == LiteRtEventTypeSyncFenceFd) {
    *sync_fence_fd = event->fd;
    return kLiteRtStatusOk;
  }
#endif
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtGetEventOpenClEvent(LiteRtEvent event, cl_event* cl_event) {
#if LITERT_HAS_OPENCL_SUPPORT
  if (event->type == LiteRtEventTypeOpenCl) {
    *cl_event = event->opencl_event;
    return kLiteRtStatusOk;
  }
#endif
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtCreateManagedEvent(LiteRtEventType type,
                                      LiteRtEvent* event) {
  auto event_res = LiteRtEventT::CreateManaged(type);
  if (!event_res) {
    return kLiteRtStatusErrorUnsupported;
  }
  *event = *event_res;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtEventWait(LiteRtEvent event, int64_t timeout_in_ms) {
  LITERT_RETURN_IF_ERROR(event->Wait(timeout_in_ms));
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtEventSignal(LiteRtEvent event) {
  LITERT_RETURN_IF_ERROR(event->Signal());
  return kLiteRtStatusOk;
}

void LiteRtDestroyEvent(LiteRtEvent event) { delete event; }

#ifdef __cplusplus
}  // extern "C"
#endif
