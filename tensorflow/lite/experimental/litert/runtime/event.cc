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

#include <cerrno>
#include <cstdint>

#include "absl/strings/str_format.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/runtime/gpu_environment.h"

#if LITERT_HAS_SYNC_FENCE_SUPPORT
#include <poll.h>
#include <unistd.h>
#endif  // LITERT_HAS_SYNC_FENCE_SUPPORT
#if LITERT_HAS_OPENCL_SUPPORT
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_event.h"
#endif  // LITERT_HAS_OPENCL_SUPPORT

using litert::Error;
using litert::Expected;

Expected<void> LiteRtEventT::Wait(int64_t timeout_in_ms) {
  if (type == LiteRtEventTypeSyncFenceFd) {
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
  } else if (type == LiteRtEventTypeOpenCl) {
#if LITERT_HAS_OPENCL_SUPPORT
    return litert::cl::WaitForEvents(/*num_events=*/1,
                                     /*event_list=*/&opencl_event);
#else
  return Error(kLiteRtStatusErrorUnsupported,
               "LiteRtEventWait not implemented for this platform");
#endif
  }
  return Error(kLiteRtStatusErrorInvalidArgument, "Invalid event type");
}

#if LITERT_HAS_SYNC_FENCE_SUPPORT
namespace {
inline bool IsFdValid(int fd) {
  return ::fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}
}  // namespace
#endif

LiteRtEventT::~LiteRtEventT() {
#if LITERT_HAS_SYNC_FENCE_SUPPORT
  if (type == LiteRtEventTypeSyncFenceFd && owns_fd && IsFdValid(fd)) {
    ::close(fd);
  }
#endif
}

Expected<void> LiteRtEventT::Signal() {
#if LITERT_HAS_OPENCL_SUPPORT
  if (type == LiteRtEventTypeOpenCl) {
    return litert::cl::SetUserEventStatus(opencl_event);
  }
#endif
  return Error(kLiteRtStatusErrorInvalidArgument,
               "The event signal is not supported");
}

Expected<LiteRtEventT*> LiteRtEventT::CreateManaged(LiteRtEventType type) {
#if LITERT_HAS_OPENCL_SUPPORT
  if (type == LiteRtEventTypeOpenCl) {
    auto& env = litert::internal::GpuEnvironmentSingleton::GetInstance();
    LITERT_ASSIGN_OR_RETURN(
        cl_event user_event,
        litert::cl::CreateUserEvent(env.getContext()->context()));
    return new LiteRtEventT{
        .type = LiteRtEventTypeOpenCl,
        .opencl_event = user_event,
    };
  }
#endif
  return Error(kLiteRtStatusErrorInvalidArgument,
               absl::StrFormat("CreateManaged doesn't support type %d", type));
}
