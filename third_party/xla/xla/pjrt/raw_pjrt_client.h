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
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
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
    // Returns the event that defines the result buffer at `result_index`. Raw
    // clients can provide an event for each result to make it available before
    // the executable finishes. Results without an individual event fall back to
    // `primary_execute_event`, which is sequenced after the whole executable.
    PjRtDeviceEventRef definition_event(size_t result_index) const {
      if (result_index < result_definition_events.size() &&
          result_definition_events[result_index]) {
        return result_definition_events[result_index];
      }
      return primary_execute_event;
    }

    std::optional<tsl::Future<>> future;
    PjRtDeviceEventRef primary_execute_event;
    std::vector<PjRtDeviceEventRef> result_definition_events;
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

  virtual absl::Status Preload(PjRtExecutable* executable) {
    return absl::OkStatus();
  }

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

  virtual AsyncWorkRunner* async_work_runner() const = 0;

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

  // Returns the required byte alignment for host memory when performing DMA.
  virtual size_t GetDmaHostAlignment() const { return 1; }

  virtual void UpdateGlobalProcessInfo(
      absl::Span<xla::coordination::TaskInfo> infos) {
    LOG(WARNING) << "UpdateGlobalProcessInfo is not supported.";
  }

  // Imports foreign memory as a raw buffer.
  virtual absl::StatusOr<PjRtRawBufferRef> ImportForeignMemory(
      PjRtMemorySpace* memory_space, void* device_ptr, size_t size,
      absl::AnyInvocable<void() &&> on_delete_callback, bool is_mutable) = 0;

  virtual absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>>
  RuntimeAbiVersion() const {
    return absl::UnimplementedError("RuntimeAbiVersion is not supported.");
  }

  virtual tsl::AsyncValueRef<PjRtExecutable> ToAsyncExecutable(
      std::shared_ptr<PjRtExecutable> executable) const = 0;

  virtual tsl::RCReference<PjRtExecutableLoadState> MakeLoadState() {
    LOG(FATAL) << "Implement MakeLoadState()";
  }

  virtual absl::StatusOr<bool> PoisonExecution(LocalDeviceId local_device_id,
                                               int32_t launch_id,
                                               absl::Status error) {
    return absl::UnimplementedError("PoisonExecution is not supported");
  }

  virtual absl::Status TransferToInfeed(LocalDeviceId local_device_id,
                                        const LiteralSlice& literal) {
    return absl::UnimplementedError("TransferToInfeed is not supported");
  }

  virtual absl::Status TransferFromOutfeed(LocalDeviceId local_device_id,
                                           MutableBorrowingLiteral literal) {
    return absl::UnimplementedError("TransferToOutfeed is not supported");
  }

  virtual void ScheduleRemoteSend(PjRtMemorySpace* memory_space,
                                  PjRtRawBufferRef raw_buffer,
                                  PjRtDeviceEventRefVector definition_events,
                                  PjRtDeviceEventPromiseRef usage_event_promise,
                                  Future<std::string> serialized_descriptor,
                                  PjRtBuffer::RemoteSendCallback on_done);

  // Similar to PjRtClient::MakeCrossHostReceiveBuffers, but uses PjRtRawBuffer
  // instead of PjRtBuffer.
  // Takes raw buffers, a notifier, and the transfer dependency AVs that must
  // be ready before the receive can complete. Returns a vector of definition
  // events that will be fulfilled once the receive operation completes.
  virtual absl::StatusOr<PjRtDeviceEventRefVector> CrossHostReceiveBuffersInto(
      absl::Span<const PjRtRawBufferRef> buffers,
      PjRtCrossHostRecvNotifier notifier,
      PjRtDeviceEventSpan transfer_dependency_avs) {
    return absl::UnimplementedError(
        "CrossHostReceiveBuffersInto is not implemented.");
  }

  // Similar to PjRtClient::CrossHost{Send/Receive}Buffers, but uses
  // PjRtRawBuffer instead of PjRtBuffer.
  // Takes in a vector of transfer dependencies and transfer specs, and launches
  // the data transfers specified by the transfer specs so that they occur after
  // all transfer dependencies are complete.
  struct CrossHostTransferSpec {
    GlobalDeviceId src_global_device_id;
    GlobalDeviceId dst_global_device_id;
    PjRtRawBufferRef raw_buffer;
  };

  virtual absl::StatusOr<PjRtDeviceEventRefVector> CrossHostTransferBuffers(
      PjRtDeviceEventRefVector transfer_dependencies,
      std::vector<CrossHostTransferSpec> transfer_specs) {
    return absl::UnimplementedError(
        "CrossHostTransferBuffers is not implemented.");
  }
};

}  // namespace xla

#endif  // XLA_PJRT_RAW_PJRT_CLIENT_H_
