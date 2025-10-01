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

#ifndef XLA_BACKENDS_GPU_RUNTIME_HOST_EXECUTE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_HOST_EXECUTE_THUNK_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/host_offloading/host_offloading_allocator.h"
#include "xla/core/host_offloading/host_offloading_executable.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"

namespace xla::gpu {

// HostExecuteAsyncEvents is a shared state that is used to coordinate
// execution between the HostExecuteStartThunk and the
// HostExecuteDoneThunk. The HostExecuteStartThunk adds an event to the
// map when it dispatches execution to the host, and the HostExecuteDoneThunk
// retrieves the event when it is executed.
class HostExecuteAsyncEvents {
 public:
  // The async value will be awaited for by the HostExecuteDoneThunk, while the
  // given event will be awaited on by the compute stream which requires the
  // published results.
  using HostExecuteEvent = tsl::AsyncValueRef<std::unique_ptr<se::Event>>;

  // Creates an event for the given executor and run id and returns it to the
  // user if the event was created successfully.
  absl::StatusOr<HostExecuteEvent> CreateEvent(se::StreamExecutor* executor,
                                               RunId run_id);

  absl::StatusOr<HostExecuteEvent> ExtractEvent(se::StreamExecutor* executor,
                                                RunId run_id);

 private:
  absl::Mutex events_mu_;
  absl::flat_hash_map<std::pair<se::StreamExecutor*, RunId>, HostExecuteEvent>
      events_ ABSL_GUARDED_BY(events_mu_);
};

class HostExecuteStartThunk : public Thunk {
 public:
  struct SliceAndShape {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  HostExecuteStartThunk(Thunk::ThunkInfo thunk_info,
                        const HloModule& hlo_module,
                        absl::InlinedVector<SliceAndShape, 4> args,
                        absl::InlinedVector<SliceAndShape, 4> results);

  static absl::StatusOr<std::unique_ptr<HostExecuteStartThunk>> Create(
      Thunk::ThunkInfo thunk_info,
      const HostOffloadingExecutableProto& host_offloading_executable_proto,
      absl::InlinedVector<SliceAndShape, 4> args,
      absl::InlinedVector<SliceAndShape, 4> results);

  HostExecuteStartThunk(const HostExecuteStartThunk&) = delete;
  HostExecuteStartThunk& operator=(const HostExecuteStartThunk&) = delete;
  ~HostExecuteStartThunk() override = default;

  std::string ToString(int indent) const override;

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<HostExecuteStartThunk>> FromProto(
      ThunkInfo thunk_info, const HostExecuteStartThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  // Returns the async events for the host offloading execution. This is
  // intended to be shared with the corresponding HostExecuteDoneThunk.
  std::shared_ptr<HostExecuteAsyncEvents> async_events() const {
    return async_events_;
  }

  absl::Status LoadExecutable();

  const HostOffloadingExecutableProto& executable_proto() const {
    return executable_proto_;
  }

  HostOffloadingExecutableProto* mutable_executable_proto() {
    return &executable_proto_;
  }

 protected:
  HostExecuteStartThunk(
      Thunk::ThunkInfo thunk_info,
      const HostOffloadingExecutableProto& host_offloading_executable_proto,
      absl::InlinedVector<SliceAndShape, 4> args,
      absl::InlinedVector<SliceAndShape, 4> results);

 private:
  absl::once_flag executable_init_flag_;
  std::unique_ptr<HostOffloadingExecutable> executable_;
  absl::InlinedVector<SliceAndShape, 4> args_;
  absl::InlinedVector<SliceAndShape, 4> results_;
  HostOffloadingExecutableProto executable_proto_;
  HostOffloadingAllocator* allocator_ = nullptr;
  std::shared_ptr<HostExecuteAsyncEvents> async_events_;
};

class HostExecuteDoneThunk : public Thunk {
 public:
  explicit HostExecuteDoneThunk(
      Thunk::ThunkInfo thunk_info,
      std::shared_ptr<HostExecuteAsyncEvents> async_events);
  HostExecuteDoneThunk(const HostExecuteDoneThunk&) = delete;
  HostExecuteDoneThunk& operator=(const HostExecuteDoneThunk&) = delete;
  ~HostExecuteDoneThunk() override = default;

  std::string ToString(int indent) const override;

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<HostExecuteDoneThunk>> FromProto(
      ThunkInfo thunk_info, const HostExecuteDoneThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::shared_ptr<HostExecuteAsyncEvents> async_events_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_HOST_EXECUTE_THUNK_H_
