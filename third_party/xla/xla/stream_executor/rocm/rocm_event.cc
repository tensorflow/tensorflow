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
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/rocm/rocm_driver.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"

namespace stream_executor {
namespace gpu {

Event::Status RocmEvent::PollForStatus() {
  ScopedActivateContext activated(context());
  hipError_t res = wrap::hipEventQuery(gpu_event());

  if (res == hipSuccess) {
    return Event::Status::kComplete;
  } else if (res == hipErrorNotReady) {
    return Event::Status::kPending;
  }

  return Event::Status::kError;
}

}  // namespace gpu
}  // namespace stream_executor
