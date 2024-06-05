/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_event.h"

#include "xla/stream_executor/gpu/gpu_event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/rocm/rocm_driver.h"

namespace stream_executor {
namespace gpu {

Event::Status RocmEvent::PollForStatus() {
  absl::StatusOr<hipError_t> status =
      QueryEvent(parent()->gpu_context(), gpu_event());
  if (!status.ok()) {
    LOG(ERROR) << "Error polling for event status: "
               << status.status().message();
    return Event::Status::kError;
  }

  switch (status.value()) {
    case hipSuccess:
      return Event::Status::kComplete;
    case hipErrorNotReady:
      return Event::Status::kPending;
    default:
      LOG(INFO) << "Error condition returned for event status: "
                << status.value();
      return Event::Status::kError;
  }
}

}  // namespace gpu
}  // namespace stream_executor
