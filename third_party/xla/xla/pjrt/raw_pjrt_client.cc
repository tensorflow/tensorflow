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

#include "xla/pjrt/raw_pjrt_client.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_client.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"

namespace xla {

void PjRtRawClientDeleter::operator()(PjRtRawClientInterface* client) const {
  if (client != nullptr) {
    client->vtable->destroy(client);
  }
}

const PJRT_RawClient_FunctionTable PjRtRawClient::kRawClientVtable = {
    /*struct_size=*/PJRT_RawClient_FunctionTable_STRUCT_SIZE,
    /*instance_size=*/PJRT_RawClient_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*destroy=*/
    +[](PJRT_RawClient* client) { delete static_cast<PjRtRawClient*>(client); },
    /*quiesce=*/
    +[](PJRT_RawClient* client) {
      static_cast<PjRtRawClient*>(client)->Quiesce();
    },
    /*should_create_async_allocation_event=*/
    +[](const PJRT_RawClient* client, int memory_kind_id) -> bool {
      return static_cast<const PjRtRawClient*>(client)
          ->ShouldCreateAsyncAllocationEvent(memory_kind_id);
    },
    /*dma_map=*/
    +[](PJRT_RawClient* client, void* data, size_t size) -> PJRT_Error* {
      return pjrt::StatusToPjRtError(
          static_cast<PjRtRawClient*>(client)->DmaMap(data, size));
    },
    /*dma_unmap=*/
    +[](PJRT_RawClient* client, void* data) -> PJRT_Error* {
      return pjrt::StatusToPjRtError(
          static_cast<PjRtRawClient*>(client)->DmaUnmap(data));
    },
    /*is_dma_mapped=*/
    +[](const PJRT_RawClient* client, const void* data,
        int64_t transfer_size) -> bool {
      return static_cast<const PjRtRawClient*>(client)->IsDmaMapped(
          data, transfer_size);
    },
    /*get_numa_node=*/
    +[](const PJRT_RawClient* client, int32_t local_device_id) -> int {
      return static_cast<const PjRtRawClient*>(client)->GetNumaNode(
          LocalDeviceId(local_device_id));
    },
    /*get_device_address_alignment=*/
    +[](const PJRT_RawClient* client, size_t* out_alignment) -> PJRT_Error* {
      auto result = static_cast<const PjRtRawClient*>(client)
                        ->GetDeviceAddressAlignment();
      if (!result.ok()) {
        return pjrt::StatusToPjRtError(result.status());
      }
      *out_alignment = *result;
      return nullptr;
    },
    /*get_dma_host_alignment=*/
    +[](const PJRT_RawClient* client) -> size_t {
      return static_cast<const PjRtRawClient*>(client)->GetDmaHostAlignment();
    },
    /*get_dma_size_alignment=*/
    +[](const PJRT_RawClient* client, size_t* out_alignment) -> bool {
      std::optional<size_t> alignment =
          static_cast<const PjRtRawClient*>(client)->GetDmaSizeAlignment();
      if (!alignment.has_value()) {
        return false;
      }
      *out_alignment = *alignment;
      return true;
    },
    /*call_oom_handlers=*/
    +[](const PJRT_RawClient* client) {
      static_cast<const PjRtRawClient*>(client)->CallOomHandlers();
    },
    /*record_memory_stats=*/
    +[](PJRT_RawClient* client) {
      static_cast<PjRtRawClient*>(client)->RecordMemoryStats();
    },
    /*record_linearize_bytes=*/
    +[](PJRT_RawClient* client, bool is_h2d, size_t bytes) {
      static_cast<PjRtRawClient*>(client)->RecordLinearizeBytes(is_h2d, bytes);
    },
    /*get_stream_for_external_ready_events=*/
    +[](const PJRT_RawClient* client, int32_t local_device_id,
        intptr_t* out_stream) -> PJRT_Error* {
      auto result =
          static_cast<const PjRtRawClient*>(client)
              ->GetStreamForExternalReadyEvents(LocalDeviceId(local_device_id));
      if (!result.ok()) {
        return pjrt::StatusToPjRtError(result.status());
      }
      *out_stream = *result;
      return nullptr;
    },
    /*clear_memory_stats=*/
    +[](PJRT_RawClient* client, int32_t local_device_id) -> PJRT_Error* {
      return pjrt::StatusToPjRtError(
          static_cast<PjRtRawClient*>(client)->ClearMemoryStats(
              LocalDeviceId(local_device_id)));
    },
};

PjRtRawClient::PjRtRawClient() { vtable = &kRawClientVtable; }

absl::StatusOr<absl::InlinedVector<PjRtRawBufferRef, 4>>
PjRtRawClient::AllocateRawBuffersForExecute(
    PJRT_Memory* memory_space, absl::Span<const size_t> on_device_bytes_counts,
    bool retry_on_oom) {
  const tsl::profiler::MemoryDebugAnnotation outer_anno =
      tsl::profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
  absl::InlinedVector<PjRtRawBufferRef, 4> buffers;
  buffers.reserve(on_device_bytes_counts.size());
  for (size_t i = 0; i < on_device_bytes_counts.size(); ++i) {
    std::optional<tsl::profiler::ScopedMemoryDebugAnnotation> anno;
    if (outer_anno.pending_shape_func != nullptr) {
      anno.emplace("dummy", outer_anno.pending_region_type,
                   outer_anno.pending_data_type, [&outer_anno, i](size_t) {
                     return outer_anno.pending_shape_func(i);
                   });
    }
    ABSL_ASSIGN_OR_RETURN(
        PjRtRawBufferRef buffer,
        AllocateRawBufferForExecute(memory_space, on_device_bytes_counts[i],
                                    retry_on_oom));
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

void PjRtRawClient::ScheduleRemoteSend(
    LocalDeviceId local_device_id, int memory_kind_id,
    PjRtRawBufferRef raw_buffer, PjRtDeviceEventRefVector definition_events,
    PjRtDeviceEventPromiseRef usage_event_promise,
    Future<std::string> serialized_descriptor,
    PjRtRawBufferInterface::RemoteSendCallback on_done) {
  auto error = absl::UnimplementedError(
      absl::StrCat("ScheduleRemoteSend is not implemented for device ",
                   local_device_id.value(), " kind ", memory_kind_id));
  on_done(error, /*sends_were_enqueued=*/false);
  usage_event_promise.SetError(error);
}

void PjRtRawClientInterface::Quiesce() { vtable->quiesce(this); }

AsyncWorkRunner* PjRtRawClientInterface::async_work_runner() const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->async_work_runner();
  }
  return nullptr;
}

PjRtEventTracker* PjRtRawClientInterface::event_tracker() const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->event_tracker();
  }
  return nullptr;
}

bool PjRtRawClientInterface::ShouldCreateAsyncAllocationEvent(
    int memory_kind_id) const {
  return vtable->should_create_async_allocation_event(this, memory_kind_id);
}

absl::StatusOr<PjRtRawBufferRef> PjRtRawClientInterface::AllocateRawBuffer(
    PJRT_Memory* memory_space, size_t on_device_bytes_count, bool retry_on_oom,
    tsl::AsyncValueRef<bool> allocate_after) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->AllocateRawBuffer(memory_space, on_device_bytes_count,
                                  retry_on_oom, std::move(allocate_after));
  }
  return absl::UnimplementedError("AllocateRawBuffer is not supported.");
}

absl::StatusOr<PjRtRawBufferRef>
PjRtRawClientInterface::AllocateRawBufferForExecute(
    PJRT_Memory* memory_space, size_t on_device_bytes_count,
    bool retry_on_oom) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->AllocateRawBufferForExecute(memory_space, on_device_bytes_count,
                                            retry_on_oom);
  }
  return absl::UnimplementedError(
      "AllocateRawBufferForExecute is not supported.");
}

absl::StatusOr<absl::InlinedVector<PjRtRawBufferRef, 4>>
PjRtRawClientInterface::AllocateRawBuffersForExecute(
    PJRT_Memory* memory_space, absl::Span<const size_t> on_device_bytes_counts,
    bool retry_on_oom) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->AllocateRawBuffersForExecute(
        memory_space, on_device_bytes_counts, retry_on_oom);
  }
  return absl::UnimplementedError(
      "AllocateRawBuffersForExecute is not supported.");
}

absl::StatusOr<
    std::pair<PjRtRawBufferRef,
              PjRtRawClientInterface::PjRtFulfillAliasRawBufferCallback>>
PjRtRawClientInterface::CreateRawBufferChannel(PJRT_Memory* memory_space,
                                               size_t on_device_bytes_count) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CreateRawBufferChannel(memory_space, on_device_bytes_count);
  }
  return absl::UnimplementedError("CreateRawBufferChannel is not supported");
}

absl::StatusOr<std::pair<PjRtDeviceEventPromiseRef, PjRtDeviceEventRef>>
PjRtRawClientInterface::CreateLinkedEventPromise(LocalDeviceId local_device_id,
                                                 int memory_kind_id,
                                                 absl::string_view debug_info) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CreateLinkedEventPromise(local_device_id, memory_kind_id,
                                         debug_info);
  }
  return absl::UnimplementedError("CreateLinkedEventPromise is not supported");
}

absl::StatusOr<PjRtDeviceEventRef> PjRtRawClientInterface::CreateDeviceEvent(
    LocalDeviceId local_device_id, int memory_kind_id, Future<> dependency) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CreateDeviceEvent(local_device_id, memory_kind_id,
                                  std::move(dependency));
  }
  return absl::UnimplementedError("CreateDeviceEvent is not supported");
}

absl::StatusOr<PjRtDeviceEventRef>
PjRtRawClientInterface::CreateDeviceEventForStream(
    LocalDeviceId local_device_id, std::intptr_t stream) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CreateDeviceEventForStream(local_device_id, stream);
  }
  return absl::UnimplementedError(
      "CreateDeviceEventForStream is not supported");
}

std::optional<std::shared_ptr<KeyValueStoreInterface>>
PjRtRawClientInterface::key_value_store() const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->key_value_store();
  }
  return std::nullopt;
}

absl::Status PjRtRawClientInterface::DmaMap(void* data, size_t size) {
  return pjrt::PjrtErrorToStatus(vtable->dma_map(this, data, size));
}

absl::Status PjRtRawClientInterface::DmaUnmap(void* data) {
  return pjrt::PjrtErrorToStatus(vtable->dma_unmap(this, data));
}

bool PjRtRawClientInterface::IsDmaMapped(const void* data,
                                         int64_t transfer_size) const {
  return vtable->is_dma_mapped(this, data, transfer_size);
}

HostMemoryAllocator* PjRtRawClientInterface::GetHostMemoryAllocator() const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->GetHostMemoryAllocator();
  }
  return nullptr;
}

int PjRtRawClientInterface::GetNumaNode(LocalDeviceId local_device_id) const {
  return vtable->get_numa_node(this, local_device_id.value());
}

absl::StatusOr<size_t> PjRtRawClientInterface::GetDeviceAddressAlignment()
    const {
  size_t alignment = 0;
  PJRT_Error* error = vtable->get_device_address_alignment(this, &alignment);
  if (error != nullptr) {
    return pjrt::PjrtErrorToStatus(error);
  }
  return alignment;
}

size_t PjRtRawClientInterface::GetDmaHostAlignment() const {
  return vtable->get_dma_host_alignment(this);
}

std::optional<size_t> PjRtRawClientInterface::GetDmaSizeAlignment() const {
  size_t alignment = 0;
  if (!vtable->get_dma_size_alignment(this, &alignment)) {
    return std::nullopt;
  }
  return alignment;
}

void PjRtRawClientInterface::LaunchOnDevice(
    LocalDeviceId device_id, absl::AnyInvocable<void()> execute_fn) const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    cpp->LaunchOnDevice(device_id, std::move(execute_fn));
    return;
  }
  if (AsyncWorkRunner* runner = async_work_runner()) {
    runner->Execute(std::move(execute_fn));
  }
}

void PjRtRawClientInterface::CallOomHandlers() const {
  vtable->call_oom_handlers(this);
}

bool PjRtRawClientInterface::ShouldRetryOnOom(
    int attempts, LocalDeviceId local_device_id,
    PjRtExecutableLoadState* load_state, absl::Status prepare_status) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->ShouldRetryOnOom(attempts, local_device_id, load_state,
                                 std::move(prepare_status));
  }
  return false;
}

void PjRtRawClientInterface::UpdateGlobalProcessInfo(
    absl::Span<xla::coordination::TaskInfo> infos) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    cpp->UpdateGlobalProcessInfo(infos);
    return;
  }
  LOG(WARNING) << "UpdateGlobalProcessInfo is not supported.";
}

void PjRtRawClientInterface::RecordMemoryStats() {
  vtable->record_memory_stats(this);
}

void PjRtRawClientInterface::RecordLinearizeBytes(bool is_h2d, size_t bytes) {
  vtable->record_linearize_bytes(this, is_h2d, bytes);
}

absl::StatusOr<PjRtRawBufferRef> PjRtRawClientInterface::ImportForeignMemory(
    PJRT_Memory* memory_space, void* device_ptr, size_t size,
    absl::AnyInvocable<void() &&> on_delete_callback, bool is_mutable) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->ImportForeignMemory(memory_space, device_ptr, size,
                                    std::move(on_delete_callback), is_mutable);
  }
  return absl::UnimplementedError("ImportForeignMemory is not supported.");
}

absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>>
PjRtRawClientInterface::RuntimeAbiVersion() const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->RuntimeAbiVersion();
  }
  return absl::UnimplementedError("RuntimeAbiVersion is not supported.");
}

tsl::AsyncValueRef<PjRtExecutable> PjRtRawClientInterface::ToAsyncExecutable(
    std::shared_ptr<PjRtExecutable> executable) const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->ToAsyncExecutable(std::move(executable));
  }
  return tsl::MakeErrorAsyncValueRef(
      absl::UnimplementedError("ToAsyncExecutable is not supported."));
}

std::pair<tsl::AsyncValueRef<PjRtExecutable>,
          PjRtRawClientInterface::PjRtFulfillAsyncExecutableCallback>
PjRtRawClientInterface::CreateLinkedExecutablePromise() const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CreateLinkedExecutablePromise();
  }
  return {tsl::MakeErrorAsyncValueRef(absl::UnimplementedError(
              "CreateLinkedExecutablePromise is not supported.")),
          [](absl::StatusOr<tsl::AsyncValueRef<PjRtExecutable>>) {}};
}

absl::StatusOr<tsl::RCReference<PjRtExecutableLoadState>>
PjRtRawClientInterface::MakeLoadState(const LoadOptions& load_options) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->MakeLoadState(load_options);
  }
  return absl::UnimplementedError("MakeLoadState is not supported.");
}

absl::StatusOr<GlobalDeviceId>
PjRtRawClientInterface::TranslateSubsliceDeviceId(
    int64_t subslice_device_id, const PjRtExecutable* executable,
    const PjRtDeviceDimensions& computation_origin) const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->TranslateSubsliceDeviceId(subslice_device_id, executable,
                                          computation_origin);
  }
  return absl::UnimplementedError(
      "Subslice computation_origin is not supported.");
}

std::unique_ptr<ScopedAsyncTrackingEvent>
PjRtRawClientInterface::CreateAsyncTrackingEvent(
    LocalDeviceId local_device_id, absl::string_view description) const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CreateAsyncTrackingEvent(local_device_id, description);
  }
  return nullptr;
}

absl::StatusOr<bool> PjRtRawClientInterface::PoisonExecution(
    LocalDeviceId local_device_id, int32_t launch_id, absl::Status error) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->PoisonExecution(local_device_id, launch_id, std::move(error));
  }
  return absl::UnimplementedError("PoisonExecution is not supported");
}

bool PjRtRawClientInterface::CanRecoverDonation(
    const absl::Status& status) const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CanRecoverDonation(status);
  }
  return false;
}

absl::Status PjRtRawClientInterface::TransferToInfeed(
    LocalDeviceId local_device_id, const LiteralSlice& literal) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->TransferToInfeed(local_device_id, literal);
  }
  return absl::UnimplementedError("TransferToInfeed is not supported");
}

absl::Status PjRtRawClientInterface::TransferFromOutfeed(
    LocalDeviceId local_device_id, MutableBorrowingLiteral literal) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->TransferFromOutfeed(local_device_id, std::move(literal));
  }
  return absl::UnimplementedError("TransferToOutfeed is not supported");
}

absl::Status PjRtRawClientInterface::WaitOnStream(LocalDeviceId local_device_id,
                                                  PjRtDeviceEventRef event,
                                                  std::intptr_t stream) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->WaitOnStream(local_device_id, std::move(event), stream);
  }
  return absl::UnimplementedError(
      "WaitUntilBufferReadyOnStream is only implemented for GPU.");
}

absl::StatusOr<std::intptr_t>
PjRtRawClientInterface::GetStreamForExternalReadyEvents(
    LocalDeviceId local_device_id) const {
  intptr_t stream = 0;
  PJRT_Error* error = vtable->get_stream_for_external_ready_events(
      this, local_device_id.value(), &stream);
  if (error != nullptr) {
    return pjrt::PjrtErrorToStatus(error);
  }
  return stream;
}

absl::StatusOr<tsl::AllocatorStats> PjRtRawClientInterface::GetAllocatorStats(
    LocalDeviceId local_device_id) const {
  if (const auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->GetAllocatorStats(local_device_id);
  }
  return absl::UnimplementedError("GetAllocatorStats is not supported.");
}

absl::Status PjRtRawClientInterface::ClearMemoryStats(
    LocalDeviceId local_device_id) {
  return pjrt::PjrtErrorToStatus(
      vtable->clear_memory_stats(this, local_device_id.value()));
}

void PjRtRawClientInterface::ScheduleRemoteSend(
    LocalDeviceId local_device_id, int memory_kind_id,
    PjRtRawBufferRef raw_buffer, PjRtDeviceEventRefVector definition_events,
    PjRtDeviceEventPromiseRef usage_event_promise,
    Future<std::string> serialized_descriptor,
    PjRtRawBufferInterface::RemoteSendCallback on_done) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    cpp->ScheduleRemoteSend(
        local_device_id, memory_kind_id, std::move(raw_buffer),
        std::move(definition_events), std::move(usage_event_promise),
        std::move(serialized_descriptor), std::move(on_done));
    return;
  }
  auto error = absl::UnimplementedError(
      absl::StrCat("ScheduleRemoteSend is not implemented for device ",
                   local_device_id.value(), " kind ", memory_kind_id));
  on_done(error, /*sends_were_enqueued=*/false);
  usage_event_promise.SetError(error);
}

absl::StatusOr<PjRtDeviceEventRefVector>
PjRtRawClientInterface::CrossHostReceiveBuffersInto(
    absl::Span<const PjRtRawBufferRef> buffers,
    PjRtCrossHostRecvNotifier notifier,
    PjRtDeviceEventSpan transfer_dependency_avs) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CrossHostReceiveBuffersInto(buffers, std::move(notifier),
                                            transfer_dependency_avs);
  }
  return absl::UnimplementedError(
      "CrossHostReceiveBuffersInto is not implemented.");
}

absl::StatusOr<PjRtDeviceEventRefVector>
PjRtRawClientInterface::CrossHostTransferBuffers(
    PjRtDeviceEventRefVector transfer_dependencies,
    std::vector<CrossHostTransferSpec> transfer_specs) {
  if (auto* cpp = down_cast<PjRtRawClient>()) {
    return cpp->CrossHostTransferBuffers(std::move(transfer_dependencies),
                                         std::move(transfer_specs));
  }
  return absl::UnimplementedError(
      "CrossHostTransferBuffers is not implemented.");
}

}  // namespace xla
