/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_event.h"

#include <cstdint>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"

namespace stream_executor {
namespace gpu {
namespace {
absl::Status WaitStreamOnEvent(Context* context, CUstream stream,
                               CUevent event) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuStreamWaitEvent(stream, event, 0 /* = flags */));
}
}  // namespace

Event::Status CudaEvent::PollForStatus() {
  ScopedActivateContext activated(context());
  CUresult res = cuEventQuery(gpu_event());
  if (res == CUDA_SUCCESS) {
    return Event::Status::kComplete;
  } else if (res == CUDA_ERROR_NOT_READY) {
    return Event::Status::kPending;
  }
  return Event::Status::kError;
}

absl::Status CudaEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  return WaitStreamOnEvent(context(), absl::bit_cast<CUstream>(stream),
                           gpu_event());
}

}  // namespace gpu
}  // namespace stream_executor
