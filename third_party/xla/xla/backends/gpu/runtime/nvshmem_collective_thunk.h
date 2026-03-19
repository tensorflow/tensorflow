/* Copyright 2025 The OpenXLA Authors.
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

#ifndef XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_THUNK_H_

#include <memory>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_kind.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// NvshmemCollectiveThunk
//===----------------------------------------------------------------------===//

// Thunk base class for NVSHMEM collective operations.
// TODO tixxx refactor Collective and NvshmemCollective thunks
// to have a single parent class for all gpu comm backends.
class NvshmemCollectiveThunk : public Thunk {
 public:
  absl::Status Prepare(const PrepareParams& params) override;

  absl::Status Initialize(const InitializeParams& params) override;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events() const {
    return async_events_;
  }
  void set_async_events(
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events) {
    async_events_ = async_events;
  }

  std::optional<AsyncEventsUniqueId> GetAsyncEventsUniqueId() const override;

  bool IsAsyncStart() const override { return async_events_ != nullptr; }

 protected:
  NvshmemCollectiveThunk(Kind kind, ThunkInfo thunk_info, bool is_sync);

  virtual absl::Status RunNvshmemCollective(const ExecuteParams& params,
                                            se::Stream& stream) = 0;
  virtual const CollectiveConfig& config() const = 0;
  virtual AsyncStreamKind GetAsyncStreamKind() const {
    return AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE;
  }

 private:
  bool IsAsync() const { return async_events_ != nullptr; }
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events_;
};

//===----------------------------------------------------------------------===//
// NvshmemCollectiveDoneThunk
//===----------------------------------------------------------------------===//

class NvshmemCollectiveDoneThunk : public Thunk {
 public:
  NvshmemCollectiveDoneThunk(
      Thunk::Kind kind, ThunkInfo thunk_info,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<NvshmemCollectiveDoneThunk>> FromProto(
      ThunkInfo thunk_info, const NvshmemCollectiveDoneThunkProto& thunk_proto,
      CollectiveThunk::AsyncEventsMap& async_events_map);

  std::optional<AsyncEventsUniqueId> GetAsyncEventsUniqueId() const override;

  bool IsAsyncDone() const override { return async_events_ != nullptr; }

 private:
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events_;
};

//===----------------------------------------------------------------------===//

absl::Status IsValidNvshmemOperand(Shape shape, Thunk::Kind reduction_op);

//===----------------------------------------------------------------------===//

absl::StatusOr<xla::gpu::GpuCollectives*> GetNvshmemCollectivesFromRegistry();

// NvshmemBufferAddresses provides a mechanism to store and retrieve buffer
// addresses for NVSHMEM put/get operations. This is necessary because NVSHMEM
// uses one-way put/get operations where both source and destination addresses
// are required.
class NvshmemBufferAddresses {
 public:
  // Get buffer address for a device
  absl::StatusOr<void*> GetNvshmemPtr(int device_ordinal);

  // Store buffer address for a device
  void StoreNvshmemPtr(int device_ordinal, void* buffer_addr);

 private:
  absl::Mutex mu_;
  // Map from device ordinal to buffer address
  absl::flat_hash_map<int, void*> buffer_addrs_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_THUNK_H_
