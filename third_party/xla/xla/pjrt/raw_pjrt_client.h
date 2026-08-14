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

#ifndef XLA_PJRT_RAW_PJRT_CLIENT_H_
#define XLA_PJRT_RAW_PJRT_CLIENT_H_

#include <cstddef>
#include <optional>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

// Represents the launch state for a loaded executable. This state must be
// reconstructed each time we want to launch the executable.
class PjRtRawLoadedExecutable {
 public:
  virtual ~PjRtRawLoadedExecutable() = default;

  struct RawExecuteResult {
    std::optional<tsl::Future<>> future;
    PjRtDeviceEventRef primary_execute_event;
    absl::Status inline_status;
  };
  virtual RawExecuteResult Execute(const ExecuteOptions& options,
                                   absl::Span<const PjRtRawBufferRef> inputs,
                                   absl::Span<const PjRtRawBufferRef> results,
                                   PjRtDeviceEventRefVector extra_deps,
                                   PjRtDeviceEventRefVector control_deps,
                                   bool is_predetermined_error,
                                   bool fill_future) && = 0;
};

// Represents state associated with a loaded executable that persists across
// raw executable launches.
class PjRtExecutableLoadState
    : public tsl::ReferenceCounted<PjRtExecutableLoadState> {
 public:
  virtual ~PjRtExecutableLoadState() = default;

  struct DeviceAndAssignment {
    PjRtDevice* device;
    std::shared_ptr<DeviceAssignment> device_assignment;
    std::optional<int32_t> slice_id;
    int replica;
    int partition;
  };

  virtual void Delete() = 0;
  virtual bool IsDeleted() const = 0;

  virtual absl::StatusOr<std::unique_ptr<PjRtRawLoadedExecutable>>
  LoadRawExecutable(tsl::AsyncValueRef<PjRtExecutable> executable,
                    const ExecuteOptions& options, size_t host_callback_idx,
                    xla::RunId run_id, DeviceAndAssignment device_and_assign,
                    int attempt) = 0;
};

// PjRtRawClient provides an interface for directly enqueuing fundamental
// operations (d2h, h2d, execute, allocation) in a platform agnostic way.
// These operations are all performed on raw buffers (PjRtRawBuffer) and
// chained through PjRtDeviceEvent dependencies.
class PjRtRawClient {
 public:
  virtual ~PjRtRawClient() = default;

  using PjRtFulfillAliasRawBufferCallback =
      absl::AnyInvocable<absl::Status(absl::StatusOr<PjRtRawBufferRef>) &&>;

  // Allocates a raw buffer of a particular size after an optional
  // allocate_after. Backends may support retrying allocation on oom which
  // can be controlled via retry_on_oom.
  virtual absl::StatusOr<PjRtRawBufferRef> AllocateRawBuffer(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom, tsl::AsyncValueRef<bool> allocate_after) = 0;

  // Allocates a raw buffer of a particular size. Backends may support retrying
  // allocation on oom which can be controlled via retry_on_oom.
  // This is separate from AllocateRawBuffer so that backends can specialize
  // allocating buffers used in the execute path.
  virtual absl::StatusOr<PjRtRawBufferRef> AllocateRawBufferForExecute(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom) {
    return AllocateRawBuffer(memory_space, on_device_bytes_count, retry_on_oom,
                             {});
  }

  // Creates a raw buffer channel. Returns a tuple containing:
  // 1.  A PjRtRawBufferRef which is an alias for a future
  //     raw buffer.
  // 2.  A PjRtFulfillAliasRawBufferCallback to fulfill the alias.
  virtual absl::StatusOr<
      std::pair<PjRtRawBufferRef, PjRtFulfillAliasRawBufferCallback>>
  CreateRawBufferChannel(PjRtMemorySpace* memory_space,
                         size_t on_device_bytes_count) {
    return absl::UnimplementedError("CreateRawBufferChannel is not supported");
  }

  // Create a linked device-event and device-event-promise such that
  // setting an event into the event promise populates the device-event.
  virtual absl::StatusOr<
      std::pair<PjRtDeviceEventPromiseRef, PjRtDeviceEventRef>>
  CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                           absl::string_view debug_info) {
    return absl::UnimplementedError(
        "CreateLinkedEventPromise is not supported");
  }

  // Creates a device event that signals completion of a dependency future.
  virtual absl::StatusOr<PjRtDeviceEventRef> CreateDeviceEvent(
      PjRtMemorySpace* memory_space, Future<> dependency) = 0;

  // Creates a device event that signals completion of work on an external
  // stream.
  virtual absl::StatusOr<PjRtDeviceEventRef> CreateDeviceEventForStream(
      PjRtMemorySpace* memory_space, std::intptr_t stream) {
    return absl::UnimplementedError(
        "CreateDeviceEventForStream is not supported");
  }

  // Returns the process-level key-value store, if supported.
  virtual std::optional<std::shared_ptr<KeyValueStoreInterface>>
  key_value_store() const {
    return std::nullopt;
  }

  // Maps host memory for DMA transfers.
  virtual absl::Status DmaMap(void* data, size_t size) {
    return absl::UnimplementedError("DmaMap is not supported.");
  }

  // Unmaps host memory previously mapped for DMA.
  virtual absl::Status DmaUnmap(void* data) {
    return absl::UnimplementedError("DmaUnmap is not supported.");
  }

  // Imports foreign memory as a raw buffer.
  virtual absl::StatusOr<PjRtRawBufferRef> ImportForeignMemory(
      PjRtMemorySpace* memory_space, void* device_ptr, size_t size,
      absl::AnyInvocable<void() &&> on_delete_callback) = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_RAW_PJRT_CLIENT_H_
