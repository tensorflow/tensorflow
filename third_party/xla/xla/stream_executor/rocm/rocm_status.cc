/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_status.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/hip_runtime.h"

namespace stream_executor::gpu {

// Formats hipError_t to output prettified values into a log stream.
// Error summaries taken from:
std::string ToString(hipError_t result) {
#define OSTREAM_ROCM_ERROR(__name) \
  case hipError##__name:           \
    return "HIP_ERROR_" #__name;

  switch (result) {
    OSTREAM_ROCM_ERROR(InvalidValue)
    OSTREAM_ROCM_ERROR(OutOfMemory)
    OSTREAM_ROCM_ERROR(NotInitialized)
    OSTREAM_ROCM_ERROR(Deinitialized)
    OSTREAM_ROCM_ERROR(NoDevice)
    OSTREAM_ROCM_ERROR(InvalidDevice)
    OSTREAM_ROCM_ERROR(InvalidImage)
    OSTREAM_ROCM_ERROR(InvalidContext)
    OSTREAM_ROCM_ERROR(InvalidHandle)
    OSTREAM_ROCM_ERROR(NotFound)
    OSTREAM_ROCM_ERROR(NotReady)
    OSTREAM_ROCM_ERROR(NoBinaryForGpu)

    // Encountered an uncorrectable ECC error during execution.
    OSTREAM_ROCM_ERROR(ECCNotCorrectable)

    // Load/store on an invalid address. Must reboot all context.
    case 700:
      return "ROCM_ERROR_ILLEGAL_ADDRESS";
    // Passed too many / wrong arguments, too many threads for register count.
    case 701:
      return "ROCM_ERROR_LAUNCH_OUT_OF_RESOURCES";

      OSTREAM_ROCM_ERROR(ContextAlreadyInUse)
      OSTREAM_ROCM_ERROR(PeerAccessUnsupported)
      OSTREAM_ROCM_ERROR(Unknown)  // Unknown internal error to ROCM.
#if TF_ROCM_VERSION >= 60200
      OSTREAM_ROCM_ERROR(LaunchTimeOut)
      OSTREAM_ROCM_ERROR(PeerAccessAlreadyEnabled)
      OSTREAM_ROCM_ERROR(PeerAccessNotEnabled)
      OSTREAM_ROCM_ERROR(SetOnActiveProcess)
      OSTREAM_ROCM_ERROR(ContextIsDestroyed)
      OSTREAM_ROCM_ERROR(Assert)
      OSTREAM_ROCM_ERROR(HostMemoryAlreadyRegistered)
      OSTREAM_ROCM_ERROR(HostMemoryNotRegistered)
      OSTREAM_ROCM_ERROR(LaunchFailure)
      OSTREAM_ROCM_ERROR(CooperativeLaunchTooLarge)
      OSTREAM_ROCM_ERROR(NotSupported)
      OSTREAM_ROCM_ERROR(StreamCaptureUnsupported)
      OSTREAM_ROCM_ERROR(StreamCaptureInvalidated)
      OSTREAM_ROCM_ERROR(StreamCaptureMerge)
      OSTREAM_ROCM_ERROR(StreamCaptureUnmatched)
      OSTREAM_ROCM_ERROR(StreamCaptureUnjoined)
      OSTREAM_ROCM_ERROR(StreamCaptureIsolation)
      OSTREAM_ROCM_ERROR(StreamCaptureImplicit)
      OSTREAM_ROCM_ERROR(CapturedEvent)
      OSTREAM_ROCM_ERROR(StreamCaptureWrongThread)
      OSTREAM_ROCM_ERROR(GraphExecUpdateFailure)
      OSTREAM_ROCM_ERROR(RuntimeMemory)
      OSTREAM_ROCM_ERROR(RuntimeOther)
#endif  // TF_ROCM_VERSION >= 60200
    default:
      return absl::StrCat("hipError_t(", static_cast<int>(result), ")");
  }
#undef OSTREAM_ROCM_ERROR
}

namespace internal {
absl::Status ToStatusSlow(hipError_t result, absl::string_view detail) {
  std::string error_message = absl::StrCat(detail, ": ", ToString(result));
  if (result == hipErrorOutOfMemory) {
    return absl::ResourceExhaustedError(error_message);
  }
  return absl::InternalError(error_message);
}
}  // namespace internal

}  // namespace stream_executor::gpu
