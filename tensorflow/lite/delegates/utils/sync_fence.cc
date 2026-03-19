/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/utils/sync_fence.h"

#include <poll.h>

#include <cerrno>
#include <cstddef>
#include <optional>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/utils/ret_macros.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite::delegates::utils {
namespace {

// Returns how many file descriptors have been signalled, or an error.
// Note that the implementation is loosely based on Android's libsync.
std::optional<size_t> PollFds(absl::Span<const int> fds, bool block) {
  constexpr auto kError = std::optional<size_t>();

  const int timeout = block ? -1 : 0;

  if (fds.empty()) {
    return 0;
  }

  std::vector<struct pollfd> pfds;
  pfds.reserve(fds.size());
  for (int fd : fds) {
    const struct pollfd pfd = {
        fd,      // .fd
        POLLIN,  // .events
    };
    pfds.push_back(pfd);
  }

  while (true) {
    const int ret = poll(pfds.data(), pfds.size(), timeout);

    // Handle redo
    if (ret == -1 && (errno == EINTR || errno == EAGAIN)) {
      continue;
    }

    // Handle error
    TFLITE_RET_CHECK(ret >= 0u, "Poll failed", kError);

    // Handle none ready
    if (ret == 0) {
      TFLITE_RET_CHECK(!block, "", kError);
      return 0;
    }

    // Count how many fds have been signalled, setting them to -1 so they are
    // not queried on subsequent passes of the loop.
    size_t signalled_count = 0;
    for (auto& fd : pfds) {
      if (fd.revents & (POLLERR | POLLNVAL)) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "invalid fd to poll");
        return {};
      }
      if (fd.fd == -1 || (fd.revents & POLLIN)) {
        fd.fd = -1;
        signalled_count++;
      }
    }

    // If we are blocking and there are any fds that have not yet been
    // signalled, poll again on the next loop.
    if (block && signalled_count != fds.size()) {
      continue;
    }

    return signalled_count;
  }

  TFLITE_ABORT_CHECK(false,
                     "The code should never reach this point");  // Crash OK
  return {};
}

}  // namespace

std::optional<std::monostate> WaitForAllFds(absl::Span<const int> fds) {
  constexpr auto kError = std::optional<std::monostate>();
  TFLITE_ASSIGN_OR_RETURN(const size_t signalled_count,
                          PollFds(fds, /*block=*/true), kError);
  TFLITE_RET_CHECK(signalled_count == fds.size(), "", kError);
  return std::monostate{};
}

std::optional<bool> AreAllFdsSignalled(absl::Span<const int> fds) {
  TFLITE_ASSIGN_OR_RETURN(const size_t signalled_count,
                          PollFds(fds, /*block=*/false), std::optional<bool>());
  return signalled_count == fds.size();
}

}  // namespace tflite::delegates::utils
