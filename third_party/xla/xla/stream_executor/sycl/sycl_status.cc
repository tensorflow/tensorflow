/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_status.h"

#include <string>

namespace stream_executor::gpu {
std::string ToString(SyclError error) {
  switch (error) {
    case SyclError::kSyclSuccess:
      return "SYCL succeeded.";
    case SyclError::kSyclErrorNoDevice:
      return "SYCL did not find the device.";
    case SyclError::kSyclErrorInvalidDevice:
      return "SYCL got invalid device id.";
    case SyclError::kSyclErrorInvalidPointer:
      return "SYCL got invalid pointer.";
    case SyclError::kSyclErrorInvalidStream:
      return "SYCL got invalid stream.";
    case SyclError::kSyclErrorDestroyDefaultStream:
      return "SYCL cannot destroy default stream.";
    case SyclError::kSyclErrorNotReady:
      return "SYCL is not ready.";
    case SyclError::kSyclErrorZeError:
      return "SYCL got ZE error.";
    default:
      return "SYCL got invalid error code.";
  }
}
}  // namespace stream_executor::gpu
