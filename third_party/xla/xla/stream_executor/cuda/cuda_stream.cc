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

#include "xla/stream_executor/cuda/cuda_stream.h"

#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_event.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

namespace {
absl::Status WaitStreamOnEvent(Context* context, CUstream stream,
                               CUevent event) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuStreamWaitEvent(stream, event, 0 /* = flags */));
}

absl::Status RecordEvent(Context* context, CUevent event, CUstream stream) {
  ScopedActivateContext activated{context};
  return cuda::ToStatus(cuEventRecord(event, stream),
                        "Error recording CUDA event");
}

int GetGpuStreamPriority(Context* context,
                         stream_executor::StreamPriority stream_priority) {
  ScopedActivateContext activation(context);
  if (stream_priority == stream_executor::StreamPriority::Default) {
    return 0;
  }
  int lowest, highest;
  auto status = cuda::ToStatus(cuCtxGetStreamPriorityRange(&lowest, &highest));
  if (!status.ok()) {
    LOG(ERROR)
        << "Could not query stream priority range. Returning default priority.";
    return 0;
  }
  return stream_priority == stream_executor::StreamPriority::Highest ? highest
                                                                     : lowest;
}

absl::StatusOr<CUstream> CreateStream(Context* context, int priority) {
  ScopedActivateContext activated(context);
  CUstream stream;
  // If the priority is 0, then use the previous api to create the stream with
  // the default priority for backward compatibility. Probably there is no
  // difference in using the new api call but leaving it as is for now.
  if (priority == 0) {
    TF_RETURN_IF_ERROR(
        cuda::ToStatus(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)));
  } else {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, priority)));
  }

  VLOG(2) << "successfully created stream " << stream << " for context "
          << context << " on thread";
  return stream;
}

}  // namespace

absl::StatusOr<std::unique_ptr<CudaStream>> CudaStream::Create(
    GpuExecutor* executor, CudaEvent completed_event,
    std::optional<std::variant<StreamPriority, int>> priority) {
  int stream_priority = [&]() {
    if (priority.has_value() && std::holds_alternative<int>(priority.value())) {
      return std::get<int>(priority.value());
    }
    return GetGpuStreamPriority(
        executor->gpu_context(),
        std::get<StreamPriority>(priority.value_or(StreamPriority::Default)));
  }();
  TF_ASSIGN_OR_RETURN(auto stream_handle,
                      CreateStream(executor->gpu_context(), stream_priority));

  return std::unique_ptr<CudaStream>(new CudaStream(
      executor, std::move(completed_event), priority, stream_handle));
}

absl::Status CudaStream::WaitFor(Stream* other) {
  CudaStream* other_stream = static_cast<CudaStream*>(other);

  TF_RETURN_IF_ERROR(other_stream->RecordCompletedEvent());
  return WaitStreamOnEvent(executor_->gpu_context(), gpu_stream(),
                           other_stream->completed_event_.GetHandle());
}

absl::Status CudaStream::RecordEvent(Event* event) {
  return stream_executor::gpu::RecordEvent(
      executor_->gpu_context(), static_cast<CudaEvent*>(event)->GetHandle(),
      gpu_stream());
}

absl::Status CudaStream::WaitFor(Event* event) {
  return WaitStreamOnEvent(executor_->gpu_context(), gpu_stream(),
                           static_cast<CudaEvent*>(event)->GetHandle());
}

absl::Status CudaStream::RecordCompletedEvent() {
  return RecordEvent(&completed_event_);
}

CudaStream::~CudaStream() {
  BlockHostUntilDone().IgnoreError();
  executor_->DeallocateStream(this);
}

}  // namespace gpu
}  // namespace stream_executor
