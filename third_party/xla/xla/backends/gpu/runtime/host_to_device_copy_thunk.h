/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_HOST_TO_DEVICE_COPY_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_HOST_TO_DEVICE_COPY_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"

namespace xla {
namespace gpu {

// A thunk that copies data from a host buffer to a device buffer.
class HostToDeviceCopyThunk : public CopyThunk {
 public:
  // Constructs a HostToDeviceCopyThunk that copies data from `source_buffer` to
  // the device buffer `destination_buffer`. `mem_size` is the size of the data
  // in bytes. `events` are the cuda record/wait events.
  // `instr` is the copy-start instruction.
  HostToDeviceCopyThunk(ThunkInfo thunk_info, const ShapedSlice& source_buffer,
                        const ShapedSlice& destination_buffer, int64_t mem_size,
                        std::shared_ptr<CopyThunk::AsyncEvents> events,
                        int64_t instr_id);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<HostToDeviceCopyThunk>> FromProto(
      ThunkInfo thunk_info, const HostToDeviceCopyThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      CopyThunk::AsyncEventsMap& async_events_map);

  std::optional<AsyncEventsUniqueId> GetAsyncEventsUniqueId() const override;

  bool IsAsyncStart() const override { return async_events_ != nullptr; }

 private:
  std::shared_ptr<CopyThunk::AsyncEvents> async_events_;
  int64_t instr_id_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_HOST_TO_DEVICE_COPY_THUNK_H_
