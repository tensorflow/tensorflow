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

#include "tensorflow/lite/experimental/litert/runtime/event.h"

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

using litert::Error;
using litert::Expected;

Expected<void> LiteRtEventT::Wait(int64_t timeout_in_ms) {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
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
      return Error(kLiteRtStatusErrorTimeoutExpired, "Timeout expired");
    }
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

  if (ret < 0) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Error waiting for fence");
  }

  return {};

#else
  return Error(kLiteRtStatusErrorUnsupported,
               "LiteRtEventWait not implemented for this platform");
#endif
}

namespace {
inline bool IsFdValid(int fd) {
  return ::fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}
}  // namespace

LiteRtEventT::~LiteRtEventT() {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  if (owns_fd && IsFdValid(fd)) {
    ::close(fd);
  }
#endif
}
