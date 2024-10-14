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

#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"

struct LrtEventT {
#if LRT_HAS_SYNC_FENCE_SUPPORT
  int fd;
  bool owns_fd;
#endif
};

#if LRT_HAS_SYNC_FENCE_SUPPORT
LrtStatus LrtEventCreateFromSyncFenceFd(int sync_fence_fd, bool owns_fd,
                                        LrtEvent* event) {
  *event = new LrtEventT{.fd = sync_fence_fd, .owns_fd = owns_fd};
  return kLrtStatusOk;
}

LrtStatus LrtEventGetSyncFenceFd(LrtEvent event, int* sync_fence_fd) {
  *sync_fence_fd = event->fd;
  return kLrtStatusOk;
}
#endif

LrtStatus LrtEventWait(LrtEvent event, int64_t timeout_in_ms) {
#if LRT_HAS_SYNC_FENCE_SUPPORT
  int fd = event->fd;

  struct pollfd fds = {
      .fd = fd,
      .events = POLLIN,
  };

  int ret;
  do {
    ret = ::poll(&fds, 1, timeout_in_ms);
    if (ret == 1) {
      break;
    } else if (ret == 0) {
      LITE_RT_LOG(LRT_WARNING, "Timeout expired: %d", timeout_in_ms);
      return kLrtStatusErrorTimeoutExpired;
    }
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

  if (ret < 0) {
    LITE_RT_LOG(LRT_ERROR, "Error waiting for fence: %s", ::strerror(errno));
    return kLrtStatusErrorRuntimeFailure;
  }

  return kLrtStatusOk;

#else
  LITE_RT_LOG(LRT_ERROR, "LrtEventWait not implemented for this platform");
  return kLrtStatusErrorUnsupported;
#endif
}

namespace {
inline bool IsFdValid(int fd) {
  return ::fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}
}  // namespace

LrtStatus LrtEventDestroy(LrtEvent event) {
#if LRT_HAS_SYNC_FENCE_SUPPORT
  if (event->owns_fd && IsFdValid(event->fd)) {
    ::close(event->fd);
  }
  delete event;
  return kLrtStatusOk;
#else
  LITE_RT_LOG(LRT_ERROR, "LrtEventDestroy not implemented for this platform");
  return kLrtStatusErrorUnsupported;
#endif
}
