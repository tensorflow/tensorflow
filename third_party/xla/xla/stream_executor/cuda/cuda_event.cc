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
#include <memory>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/event.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {
namespace {
absl::Status WaitStreamOnEvent(StreamExecutor *executor, CUstream stream,
                               CUevent event) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  return cuda::ToStatus(cuStreamWaitEvent(stream, event, 0 /* = flags */));
}

void DestroyEvent(StreamExecutor *executor, CUevent event) {
  if (event == nullptr) {
    return;
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  auto result =
      cuda::ToStatus(cuEventDestroy(event), "Error destroying CUDA event");
  if (!result.ok()) {
    LOG(ERROR) << result.message();
  }
}

enum class EventFlags { kDefault, kDisableTiming };
absl::StatusOr<CUevent> InitEvent(StreamExecutor *executor, EventFlags flags) {
  int cuflags;
  switch (flags) {
    case EventFlags::kDefault:
      cuflags = CU_EVENT_DEFAULT;
      break;
    case EventFlags::kDisableTiming:
      cuflags = CU_EVENT_DISABLE_TIMING;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(flags);
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CUevent event_handle;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuEventCreate(&event_handle, cuflags)));
  return event_handle;
}

}  // namespace

Event::Status CudaEvent::PollForStatus() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  CUresult res = cuEventQuery(handle_);
  if (res == CUDA_SUCCESS) {
    return Event::Status::kComplete;
  } else if (res == CUDA_ERROR_NOT_READY) {
    return Event::Status::kPending;
  }
  return Event::Status::kError;
}

absl::Status CudaEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  return WaitStreamOnEvent(executor_, absl::bit_cast<CUstream>(stream),
                           handle_);
}

absl::StatusOr<CudaEvent> CudaEvent::Create(StreamExecutor *executor,
                                            bool allow_timing) {
  TF_ASSIGN_OR_RETURN(
      CUevent event_handle,
      InitEvent(executor, allow_timing ? EventFlags::kDefault
                                       : EventFlags::kDisableTiming));

  return CudaEvent(executor, event_handle);
}

CudaEvent::~CudaEvent() { DestroyEvent(executor_, handle_); }

CudaEvent& CudaEvent::operator=(CudaEvent&& other) {
  if (this == &other) {
    return *this;
  }

  DestroyEvent(executor_, handle_);

  executor_ = other.executor_;
  handle_ = other.handle_;
  other.executor_ = nullptr;
  other.handle_ = nullptr;

  return *this;
}

CudaEvent::CudaEvent(CudaEvent &&other)
    : executor_(other.executor_), handle_(other.handle_) {
  other.executor_ = nullptr;
  other.handle_ = nullptr;
}

}  // namespace gpu
}  // namespace stream_executor
