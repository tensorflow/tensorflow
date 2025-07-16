/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_HOST_OFFLOADING_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_HOST_OFFLOADING_THUNK_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/host_offloading/host_offloading_allocator.h"
#include "xla/core/host_offloading/host_offloading_executable.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"

namespace xla {
namespace gpu {

// HostOffloadingAsyncEvents is a shared state that is used to coordinate
// execution between the HostOffloadingStartThunk and the
// HostOffloadingDoneThunk. The HostOffloadingStartThunk adds an event to the
// map when it dispatches execution to the host, and the HostOffloadingDoneThunk
// retrieves the event when it is executed.
class HostOffloadingAsyncEvents {
 public:
  absl::Status AddEvent(stream_executor::StreamExecutor* executor,
                        size_t run_id, tsl::AsyncValueRef<tsl::Chain> event);

  absl::StatusOr<tsl::AsyncValueRef<tsl::Chain>> GetEvent(
      stream_executor::StreamExecutor* executor, size_t run_id);

  absl::Status RemoveEvent(stream_executor::StreamExecutor* executor,
                           size_t run_id);

 private:
  absl::flat_hash_map<std::pair<stream_executor::StreamExecutor*, size_t>,
                      tsl::AsyncValueRef<tsl::Chain>>
      events_;
};

struct SliceAndShape {
  BufferAllocation::Slice slice;
  Shape shape;
};

class HostOffloadingStartThunk : public Thunk {
 public:
  HostOffloadingStartThunk(Thunk::ThunkInfo thunk_info,
                           const HloModule& hlo_module,
                           absl::InlinedVector<SliceAndShape, 4> args,
                           absl::InlinedVector<SliceAndShape, 4> results);
  HostOffloadingStartThunk(const HostOffloadingStartThunk&) = delete;
  HostOffloadingStartThunk& operator=(const HostOffloadingStartThunk&) = delete;
  ~HostOffloadingStartThunk() override = default;

  std::string ToString(int indent) const override;

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<HostOffloadingStartThunk>> FromProto(
      ThunkInfo thunk_info, const HostOffloadingStartThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  // Returns the async events for the host offloading execution. This is
  // intended to be shared with the corresponding HostOffloadingDoneThunk.
  std::shared_ptr<HostOffloadingAsyncEvents> GetAsyncEvents() const {
    return async_events_;
  }

 private:
  std::unique_ptr<HostOffloadingExecutable> executable_;
  absl::InlinedVector<SliceAndShape, 4> args_;
  absl::InlinedVector<SliceAndShape, 4> results_;
  HostOffloadingExecutableProto executable_proto_;
  HostOffloadingAllocator* allocator_ = nullptr;
  std::shared_ptr<HostOffloadingAsyncEvents> async_events_;
};

class HostOffloadingDoneThunk : public Thunk {
 public:
  explicit HostOffloadingDoneThunk(
      Thunk::ThunkInfo thunk_info,
      std::shared_ptr<HostOffloadingAsyncEvents> async_events);
  HostOffloadingDoneThunk(const HostOffloadingDoneThunk&) = delete;
  HostOffloadingDoneThunk& operator=(const HostOffloadingDoneThunk&) = delete;
  ~HostOffloadingDoneThunk() override = default;

  std::string ToString(int indent) const override;

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<HostOffloadingDoneThunk>> FromProto(
      ThunkInfo thunk_info, const HostOffloadingDoneThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::shared_ptr<HostOffloadingAsyncEvents> async_events_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_HOST_OFFLOADING_THUNK_H_
