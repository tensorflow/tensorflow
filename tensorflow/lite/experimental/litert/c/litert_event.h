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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_EVENT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_EVENT_H_

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stdint.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtEvent);

LiteRtStatus LiteRtCreateEventFromSyncFenceFd(int sync_fence_fd, bool owns_fd,
                                              LiteRtEvent* event);

LiteRtStatus LiteRtGetEventSyncFenceFd(LiteRtEvent event, int* sync_fence_fd);

// Pass -1 for timeout_in_ms for indefinite wait.
LiteRtStatus LiteRtEventWait(LiteRtEvent event, int64_t timeout_in_ms);

void LiteRtDestroyEvent(LiteRtEvent event);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_EVENT_H_
