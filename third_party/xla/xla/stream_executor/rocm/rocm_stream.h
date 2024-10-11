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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace gpu {

class RocmStream : public GpuStream {
 public:
  absl::Status WaitFor(Stream* other) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status WaitFor(Event* event) override;

  absl::Status Memset32(DeviceMemoryBase* location, uint32_t pattern,
                        uint64_t size) override;

  static absl::StatusOr<std::unique_ptr<RocmStream>> Create(
      GpuExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority);

  ~RocmStream() override;

 private:
  RocmStream(GpuExecutor* executor, RocmEvent completed_event,
             std::optional<std::variant<StreamPriority, int>> priority,
             hipStream_t stream_handle)
      : GpuStream(executor, priority, stream_handle),
        executor_(executor),
        completed_event_(std::move(completed_event)) {}

  absl::Status RecordCompletedEvent();

  GpuExecutor* executor_;
  RocmEvent completed_event_;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_
