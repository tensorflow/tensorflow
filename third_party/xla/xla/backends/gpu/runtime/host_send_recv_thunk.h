/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_HOST_SEND_RECV_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_HOST_SEND_RECV_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/global_device_id.h"
#include "xla/shape.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// HostSendRecvAsyncEvents
//===----------------------------------------------------------------------===//

// Host Send/Recv operations have two levels of async behavior:
//
// (1) AsyncValueRef will become available only after send/recv handler
//     schedules all activities on the device.
//
// (2) se::Event will become available when device activity recorded by
//     send/recv handlers complete.
//
// We  keep track of Send/Recv commands in flight, and synchronize `send` and
// `recv` operations with corresponding `send-done` and `recv-done`.
//
// Each channel can have at most one event in flight for a given executor.
//
// We have a single instance of `HostSendRecvAsyncEvents` for each Gpu
// executable, and all thunks share it using a shared pointer.
//
// TODO(ezhulenev): Rename to `SendRecvEvents` once we remove deprecated XLA
// runtime, as it has name conflict.
class HostSendRecvAsyncEvents {
 public:
  // Emplace a new send/recv completion event.
  absl::Status Emplace(se::StreamExecutor* executor, int32_t channel_id,
                       tsl::AsyncValueRef<std::unique_ptr<se::Event>> event);

  // Extract a send/recv completion event.
  absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> Extract(
      se::StreamExecutor* executor, int32_t channel_id);

 private:
  using Key = std::pair<se::StreamExecutor*, /*channel_id=*/int64_t>;

  absl::Mutex mutex_;
  absl::flat_hash_map<Key, tsl::AsyncValueRef<std::unique_ptr<se::Event>>>
      events_ ABSL_GUARDED_BY(mutex_);
};

//===----------------------------------------------------------------------===//
// HostSendThunk
//===----------------------------------------------------------------------===//

class HostSendThunk : public Thunk {
 public:
  HostSendThunk(ThunkInfo thunk_info, Shape shape,
                BufferAllocation::Slice buffer, int64_t channel_id,
                std::shared_ptr<HostSendRecvAsyncEvents> events,
                absl::flat_hash_map<std::string, std::string> frontend_attrs,
                std::optional<GlobalDeviceId> device_constraint);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  Shape shape_;
  BufferAllocation::Slice buffer_;

  int64_t channel_id_;

  std::shared_ptr<HostSendRecvAsyncEvents> events_;
  absl::flat_hash_map<std::string, std::string> frontend_attrs_;
  std::optional<GlobalDeviceId> device_constraint_;
};

//===----------------------------------------------------------------------===//
// HostSendDoneThunk
//===----------------------------------------------------------------------===//

class HostSendDoneThunk : public Thunk {
 public:
  HostSendDoneThunk(ThunkInfo thunk_info, int64_t channel_id,
                    std::shared_ptr<HostSendRecvAsyncEvents> events,
                    std::optional<GlobalDeviceId> device_constraint);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  int64_t channel_id_;

  std::shared_ptr<HostSendRecvAsyncEvents> events_;
  std::optional<GlobalDeviceId> device_constraint_;
};

//===----------------------------------------------------------------------===//
// HostRecvThunk
//===----------------------------------------------------------------------===//

class HostRecvThunk : public Thunk {
 public:
  HostRecvThunk(ThunkInfo thunk_info, Shape shape,
                BufferAllocation::Slice buffer, int64_t channel_id,
                std::shared_ptr<HostSendRecvAsyncEvents> events,
                absl::flat_hash_map<std::string, std::string> frontend_attrs,
                std::optional<GlobalDeviceId> device_constraint);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  Shape shape_;
  BufferAllocation::Slice buffer_;

  int64_t channel_id_;

  std::shared_ptr<HostSendRecvAsyncEvents> events_;
  absl::flat_hash_map<std::string, std::string> frontend_attrs_;
  std::optional<GlobalDeviceId> device_constraint_;
};

//===----------------------------------------------------------------------===//
// HostRecvDoneThunk
//===----------------------------------------------------------------------===//

class HostRecvDoneThunk : public Thunk {
 public:
  HostRecvDoneThunk(ThunkInfo thunk_info, int64_t channel_id,
                    std::shared_ptr<HostSendRecvAsyncEvents> events,
                    std::optional<GlobalDeviceId> device_constraint);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  int64_t channel_id_;

  std::shared_ptr<HostSendRecvAsyncEvents> events_;
  std::optional<GlobalDeviceId> device_constraint_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_HOST_SEND_RECV_THUNK_H_
