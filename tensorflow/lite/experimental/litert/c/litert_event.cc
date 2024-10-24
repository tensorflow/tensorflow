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
#include "tensorflow/lite/experimental/litert/core/event.h"

#if LITERT_HAS_SYNC_FENCE_SUPPORT
LiteRtStatus LiteRtEventCreateFromSyncFenceFd(int sync_fence_fd, bool owns_fd,
                                              LiteRtEvent* event) {
  *event = new LiteRtEventT{.fd = sync_fence_fd, .owns_fd = owns_fd};
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtEventGetSyncFenceFd(LiteRtEvent event, int* sync_fence_fd) {
  *sync_fence_fd = event->fd;
  return kLiteRtStatusOk;
}
#endif

LiteRtStatus LiteRtEventWait(LiteRtEvent event, int64_t timeout_in_ms) {
  return event->Wait(timeout_in_ms);
}

void LiteRtEventDestroy(LiteRtEvent event) { delete event; }
