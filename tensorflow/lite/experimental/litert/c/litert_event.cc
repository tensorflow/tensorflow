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
#include <poll.h>
#include <unistd.h>

#include <cstdint>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/runtime/event.h"

LiteRtStatus LiteRtCreateEventFromSyncFenceFd(int sync_fence_fd, bool owns_fd,
                                              LiteRtEvent* event) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  *event = new LiteRtEventT{.fd = sync_fence_fd, .owns_fd = owns_fd};
  return kLiteRtStatusOk;
#else
  return kLiteRtStatusErrorUnsupported;
#endif
}

LiteRtStatus LiteRtGetEventSyncFenceFd(LiteRtEvent event, int* sync_fence_fd) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  *sync_fence_fd = event->fd;
  return kLiteRtStatusOk;
#else
  return kLiteRtStatusErrorUnsupported;
#endif
}

LiteRtStatus LiteRtEventWait(LiteRtEvent event, int64_t timeout_in_ms) {
  if (auto status = event->Wait(timeout_in_ms); !status) {
    LITERT_LOG(LITERT_ERROR, "%s", status.Error().Message().data());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

void LiteRtDestroyEvent(LiteRtEvent event) { delete event; }
