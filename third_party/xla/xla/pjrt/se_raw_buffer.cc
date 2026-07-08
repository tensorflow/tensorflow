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

#include "xla/pjrt/se_raw_buffer.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/device_event_utils.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace {
static absl::StatusOr<xla::BufferSequencingEventRef>
GetOrCreateAllocationReadyEvent(const xla::RawSEDeviceMemory& device_buffer,
                                xla::PjRtStreamExecutorClient* client,
                                xla::LocalDeviceState* local_device) {
  ASSIGN_OR_RETURN(auto result,
                   device_buffer.GetDefinitionEvent(client->async_work_runner(),
                                                    /*nullptr_if_past=*/false));
  if (!result) {
    result = xla::BufferSequencingEvent::Create(client->async_work_runner());
    auto stream = local_device->BorrowStreamFromPool();
    auto status = client->AllocateAndRecordEvent(
        result, local_device, stream.get(),
        "PjRtStreamExecutorRawBuffer::MakeAllocationReadyEvent");
    local_device->ReturnStreamToPool(std::move(stream));
    RETURN_IF_ERROR(status);
  }
  return result;
}
}  // namespace

namespace xla {

PjRtStreamExecutorDeviceEventPromise::PjRtStreamExecutorDeviceEventPromise(
    PjRtStreamExecutorClient* client, LocalDeviceState* local_device,
    AsyncWorkRunner* async_work_runner)
    : client_(client),
      local_device_(local_device),
      av_(tsl::MakeIndirectAsyncValue()),
      event_(tsl::MakeConstructedAsyncValueRef<BufferSequencingEvent>(
          async_work_runner,
          tsl::AsyncValueRef<BufferSequencingEvent::EventState>(av_))) {}

void PjRtStreamExecutorDeviceEventPromise::Set(PjRtDeviceEventRef event) {
  SetFromSEEvent(std::move(event).down_cast<BufferSequencingEvent>());
}

void PjRtStreamExecutorDeviceEventPromise::SetFromSEEvent(
    BufferSequencingEventRef event) {
  av_->ForwardTo(event->event().CopyRCRef());
  event.AndThen([event = event_, original_event = event]() {
    if (auto* error = original_event.GetErrorIfPresent()) {
      event.SetError(*error);
    } else {
      event.SetStateConcrete();
    }
  });
}

void PjRtStreamExecutorDeviceEventPromise::SetReady() {
  auto result = BufferSequencingEvent::Create(client_->async_work_runner());
  auto stream = local_device_->BorrowStreamFromPool();
  auto status = client_->AllocateAndRecordEvent(
      result, local_device_, stream.get(),
      "PjRtStreamExecutorDeviceEventPromise::SetReady");
  local_device_->ReturnStreamToPool(std::move(stream));
  if (!status.ok()) {
    SetError(status);
  } else {
    SetFromSEEvent(result);
  }
}

absl::StatusOr<PjRtDeviceEventRef>
PjRtStreamExecutorRawBuffer::CopyRawHostToDeviceAndReturnEvent(
    const void* src, int64_t offset, int64_t transfer_size,
    PjRtDeviceEventRefVector dependencies) {
  se::Stream* stream = local_device_->host_to_device_stream();
  auto device_event =
      BufferSequencingEvent::Create(client_->async_work_runner());
  device_event.AndThen([device_buffer = device_buffer_]() {});

  auto run_transfer = [client = client_, device_event,
                       local_device = local_device_, stream, src, offset,
                       transfer_size, buf = tsl::FormRef(this),
                       dependencies = dependencies]() mutable {
    absl::Status dep_status = GetErrors(dependencies);
    if (!dep_status.ok()) {
      client->SetEventAsError(device_event, dep_status);
      return;
    }

    se::DeviceAddressBase sub_buffer = buf->device_buffer_->mem();
    if (transfer_size < sub_buffer.size()) {
      sub_buffer = sub_buffer.GetByteSlice(offset, transfer_size);
    }
    std::shared_ptr<void> staging_buffer;
    auto status = [&]() -> absl::Status {
      RETURN_IF_ERROR(client->WaitForAllocation(stream, *buf));
      if (transfer_size > 0) {
        if (client->ShouldStageHostToDeviceTransfers(src, transfer_size)) {
          if (client->GetHostMemoryAllocator() == nullptr) {
            return absl::InvalidArgumentError(
                "host_memory_allocator should be initialized for "
                "staging buffer transfer.");
          }
          HostMemoryAllocator::AllocateOptions alloc_opts;
          alloc_opts.numa_node = stream->parent()->numa_node();
          alloc_opts.local_device_id = local_device->local_device_id();
          staging_buffer = client->GetHostMemoryAllocator()->Allocate(
              transfer_size, alloc_opts);
          auto copy_to_staging_buffer = [src, transfer_size,
                                         staging_buffer]() mutable {
            std::memcpy(staging_buffer.get(), src, transfer_size);
          };
          RETURN_IF_ERROR(stream->DoHostCallback(copy_to_staging_buffer));
          RETURN_IF_ERROR(
              stream->Memcpy(&sub_buffer, staging_buffer.get(), transfer_size));
        } else {
          RETURN_IF_ERROR(stream->Memcpy(&sub_buffer, src, transfer_size));
        }
      }
      return absl::OkStatus();
    }();
    if (status.ok()) {
      status = client->AllocateAndRecordEvent(
          device_event, local_device, stream,
          "PjRtStreamExecutorRawBuffer::CopyRawHostToDevice",
          [staging_buffer = std::move(staging_buffer)]() mutable {
            staging_buffer.reset();
          });
    }
    if (!status.ok()) {
      client->SetEventAsError(device_event, status);
    }
  };

  ExecuteWhenReady(dependencies, client_->async_work_runner(),
                   std::move(run_transfer));

  return PjRtDeviceEventRef(std::move(device_event));
}

absl::StatusOr<PjRtDeviceEventRef>
PjRtStreamExecutorRawBuffer::CopyRawDeviceToHostAndReturnEvent(
    void* dst, int64_t offset, int64_t transfer_size,
    PjRtDeviceEventRefVector dependencies) {
  se::Stream* stream = local_device_->GetDeviceToHostStream();
  auto device_event =
      BufferSequencingEvent::Create(client_->async_work_runner());
  device_event.AndThen([device_buffer = device_buffer_]() {});

  auto run_transfer = [client = client_, device_event,
                       local_device = local_device_, stream, dst, offset,
                       transfer_size, buf = tsl::FormRef(this),
                       dependencies = dependencies]() mutable {
    absl::Status dep_status = GetErrors(dependencies);
    if (!dep_status.ok()) {
      client->SetEventAsError(device_event, dep_status);
      return;
    }

    se::DeviceAddressBase sub_buffer = buf->device_buffer_->mem();
    if (transfer_size < sub_buffer.size()) {
      sub_buffer = sub_buffer.GetByteSlice(offset, transfer_size);
    }
    std::shared_ptr<void> staging_buffer;
    auto status = [&]() -> absl::Status {
      RETURN_IF_ERROR(client->WaitForAllocation(stream, *buf));
      if (transfer_size > 0) {
        if (client->ShouldStageHostToDeviceTransfers(dst, transfer_size)) {
          if (client->GetHostMemoryAllocator() == nullptr) {
            return absl::InvalidArgumentError(
                "host_memory_allocator should be initialized for "
                "staging buffer transfer.");
          }
          HostMemoryAllocator::AllocateOptions alloc_opts;
          alloc_opts.numa_node = stream->parent()->numa_node();
          alloc_opts.local_device_id = local_device->local_device_id();
          staging_buffer = client->GetHostMemoryAllocator()->Allocate(
              transfer_size, alloc_opts);
          RETURN_IF_ERROR(
              stream->Memcpy(staging_buffer.get(), sub_buffer, transfer_size));
          auto copy_from_staging_buffer = [dst, transfer_size,
                                           staging_buffer]() mutable {
            std::memcpy(dst, staging_buffer.get(), transfer_size);
          };
          // TODO(parkers): This failing maybe consitutes a race.
          RETURN_IF_ERROR(stream->DoHostCallback(copy_from_staging_buffer));
        } else {
          RETURN_IF_ERROR(stream->Memcpy(dst, sub_buffer, transfer_size));
        }
      }
      return absl::OkStatus();
    }();
    if (status.ok()) {
      status = client->AllocateAndRecordEvent(
          device_event, local_device, stream,
          "PjRtStreamExecutorRawBuffer::CopyRawDeviceToHost",
          [staging_buffer = std::move(staging_buffer)]() mutable {
            staging_buffer.reset();
          });
    }
    if (!status.ok()) {
      client->SetEventAsError(device_event, status);
    }
  };

  ExecuteWhenReady(dependencies, client_->async_work_runner(),
                   std::move(run_transfer));

  return PjRtDeviceEventRef(std::move(device_event));
}

absl::Status PjRtStreamExecutorRawBuffer::ValidateSlice(int64_t offset,
                                                        int64_t slice_size) {
  size_t buffer_size = GetOnDeviceSizeInBytes();
  if (offset < 0 || offset > buffer_size || buffer_size - offset < slice_size) {
    return InvalidArgument(
        "Invalid slicing of buffer size %lld with "
        "invalid offset %lld, slice size %lld",
        buffer_size, offset, slice_size);
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtRawBufferRef> PjRtStreamExecutorRawBuffer::Slice(
    int64_t offset, int64_t size) {
  RETURN_IF_ERROR(ValidateSlice(offset, size));
  return tsl::MakeRef<PjRtStreamExecutorRawBuffer>(
      client_, memory_space_, local_device_,
      RawSEDeviceMemory::CreateSlice(device_buffer_, offset, size), size);
}

absl::StatusOr<PjRtDeviceEventRef>
PjRtStreamExecutorRawBuffer::MakeAllocationReadyEvent() {
  auto* client =
      tensorflow::down_cast<PjRtStreamExecutorClient*>(memory_space_->client());
  if (device_buffer_.IsConcrete()) {
    ASSIGN_OR_RETURN(auto result, GetOrCreateAllocationReadyEvent(
                                      *device_buffer_, client, local_device_));
    return PjRtDeviceEventRef(std::move(result));
  }
  if (device_buffer_.IsError()) {
    return device_buffer_.GetError();
  }

  ASSIGN_OR_RETURN(
      auto promise_and_event,
      client->CreateLinkedEventPromise(
          memory_space_,
          "PjRtStreamExecutorRawBuffer::MakeAllocationReadyEvent"));
  auto [promise, event] = std::move(promise_and_event);

  device_buffer_.AndThen([client, promise = std::move(promise),
                          device_buffer = device_buffer_,
                          local_device = local_device_]() mutable {
    if (device_buffer.IsError()) {
      promise.SetError(device_buffer.GetError());
      return;
    }
    auto result_or =
        GetOrCreateAllocationReadyEvent(*device_buffer, client, local_device);
    if (!result_or.ok()) {
      promise.SetError(result_or.status());
      return;
    }
    promise.Set(PjRtDeviceEventRef(std::move(*result_or)));
  });

  return event;
}

void PjRtStreamExecutorRawBuffer::CopyTo(
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  bool is_intra_client =
      dst_raw_buffer->memory_space()->client() == memory_space()->client();
  if (!is_intra_client && allocation_event) {
    std::move(allocation_event)(absl::OkStatus());
  }
  if (is_intra_client) {
    IntraClientCopyToWithDependencies(
        /*dependencies=*/{}, std::move(dst_raw_buffer),
        std::move(definition_event_promise), std::move(src_usage_event_promise),
        std::move(allocation_event));
  } else if (auto* src_ptr = GetHostPointer()) {
    auto h2d_event = dst_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
        src_ptr, 0, GetOnDeviceSizeInBytes());
    if (!h2d_event.ok()) {
      definition_event_promise.SetError(h2d_event.status());
      src_usage_event_promise.SetError(h2d_event.status());
      return;
    }
    (*h2d_event)
        .AndThen([src_usage_event_promise = std::move(src_usage_event_promise),
                  src_buffer = tsl::FormRef(this)]() {
          src_usage_event_promise.SetReady();
        });
    definition_event_promise.Set(*std::move(h2d_event));
    return;
  } else if (auto* dst_ptr = dst_raw_buffer->GetHostPointer()) {
    auto d2h_event = CopyRawDeviceToHostAndReturnEvent(
        dst_ptr, 0, GetOnDeviceSizeInBytes(), {});
    if (!d2h_event.ok()) {
      definition_event_promise.SetError(d2h_event.status());
      src_usage_event_promise.SetError(d2h_event.status());
      return;
    }
    (*d2h_event)
        .AndThen(
            [definition_event_promise = std::move(definition_event_promise),
             d2h_event = *d2h_event, dst_buffer = dst_raw_buffer]() {
              if (const absl::Status* error =
                      d2h_event.async_value()->GetErrorIfPresent()) {
                definition_event_promise.SetError(*error);
              } else {
                definition_event_promise.SetReady();
              }
            });
    src_usage_event_promise.Set(*std::move(d2h_event));
    return;
  } else {
    HostMemoryAllocator::AllocateOptions alloc_opts;
    alloc_opts.numa_node = local_device_->executor()->numa_node();
    alloc_opts.local_device_id = local_device_->local_device_id();
    std::shared_ptr<void> staging_buffer =
        client_->GetHostMemoryAllocator()->Allocate(GetOnDeviceSizeInBytes(),
                                                    alloc_opts);
    auto d2h_event = CopyRawDeviceToHostAndReturnEvent(
        staging_buffer.get(), 0, GetOnDeviceSizeInBytes(), {});
    if (!d2h_event.ok()) {
      definition_event_promise.SetError(d2h_event.status());
      src_usage_event_promise.SetError(d2h_event.status());
      return;
    }
    (*d2h_event)
        .AndThen([staging_buffer, dst_raw_buffer,
                  definition_event_promise =
                      std::move(definition_event_promise),
                  d2h_event = *d2h_event]() {
          if (const absl::Status* error =
                  d2h_event.async_value()->GetErrorIfPresent()) {
            definition_event_promise.SetError(*error);
          } else {
            auto h2d_event = dst_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
                staging_buffer.get(), 0,
                dst_raw_buffer->GetOnDeviceSizeInBytes());
            if (!h2d_event.ok()) {
              definition_event_promise.SetError(*error);
            } else {
              (*h2d_event).AndThen([staging_buffer]() {});
              definition_event_promise.Set(*std::move(h2d_event));
            }
          }
        });
    src_usage_event_promise.Set(*std::move(d2h_event));
  }
}

// We override ScheduleCopyTo so that for intra-client D2D copies, we can issue
// the transfer after the definition events are enqueued (instead of until after
// they are complete).
void PjRtStreamExecutorRawBuffer::ScheduleCopyTo(
    PjRtDeviceEventRefVector transfer_dependency_events,
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  if (dst_raw_buffer->memory_space()->client() == memory_space()->client()) {
    client_->async_work_runner()->Execute(
        [this_ref = tsl::FormRef(this),
         transfer_dependency_events = std::move(transfer_dependency_events),
         dst_raw_buffer = std::move(dst_raw_buffer),
         definition_event_promise = std::move(definition_event_promise),
         src_usage_event_promise = std::move(src_usage_event_promise),
         allocation_event = std::move(allocation_event)]() mutable {
          this_ref->IntraClientCopyToWithDependencies(
              std::move(transfer_dependency_events), std::move(dst_raw_buffer),
              std::move(definition_event_promise),
              std::move(src_usage_event_promise), std::move(allocation_event));
        });
    return;
  }
  CommonPjRtRawBufferImpl::ScheduleCopyTo(
      std::move(transfer_dependency_events), std::move(dst_raw_buffer),
      std::move(definition_event_promise), std::move(src_usage_event_promise),
      std::move(allocation_event));
}

void PjRtStreamExecutorRawBuffer::IntraClientCopyToWithDependencies(
    PjRtDeviceEventRefVector dependencies, PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  auto usage_event =
      BufferSequencingEvent::Create(client_->async_work_runner());

  PjRtDeviceEventSpan deps_span(dependencies);

  auto task = [client = client_, local_device = local_device_,
               src_buffer = device_buffer_,
               dst_raw_buffer = std::move(dst_raw_buffer),
               src_raw_buffer = tsl::FormRef(this), usage_event,
               dependencies = std::move(dependencies),
               allocation_event = std::move(allocation_event)]() mutable {
    se::Stream* stream = local_device->GetDeviceToDeviceStream();
    bool allocation_set = false;
    auto status = [&]() -> absl::Status {
      // Handle errors in pre-scheduling async values.
      for (size_t i = 0; i < dependencies.size(); ++i) {
        const auto& dep = dependencies[i];
        if (auto event_ref = dep.down_cast<BufferSequencingEvent>()) {
          if (auto* error = event_ref->event().GetErrorIfPresent()) {
            return *error;
          }
        } else if (auto error = dep.GetErrorIfPresent()) {
          return *error;
        }
      }

      // Wait for BufferSequencingEvent based dependencies on the stream.
      for (size_t i = 0; i < dependencies.size(); ++i) {
        const auto& dep = dependencies[i];
        if (auto event_ref = dep.down_cast<BufferSequencingEvent>()) {
          event_ref->WaitForEventOnStream(stream);
        } else {
          xla::BlockUntilReady(dep);
        }
      }

      ASSIGN_OR_RETURN(EventPool::Handle event,
                       local_device->event_pool().AllocateEvent(
                           client->async_work_runner(), stream->parent()));

      if (allocation_event) {
        allocation_set = true;
        std::move(allocation_event)(absl::OkStatus());
      }

      auto* dst_cpp_buffer =
          dst_raw_buffer->down_cast<const PjRtStreamExecutorRawBuffer>();
      if (dst_cpp_buffer == nullptr) {
        return absl::InvalidArgumentError(
            "Destination buffer is not a StreamExecutor raw buffer");
      }
      auto dst_buffer = dst_cpp_buffer->device_buffer();
      auto dst_buffer_mem = dst_buffer->mem();
      RETURN_IF_ERROR(client->WaitForAllocation(stream, *src_raw_buffer));
      RETURN_IF_ERROR(client->WaitForAllocation(stream, *dst_raw_buffer));
      if (dst_buffer_mem.size() > 0) {
        RETURN_IF_ERROR(stream->MemcpyD2D(&dst_buffer_mem, src_buffer->mem(),
                                          dst_buffer_mem.size()));
      }
      client->ThenRecordEvent(usage_event, local_device, std::move(event),
                              stream);
      usage_event.AndThen([src_buffer, dst_raw_buffer]() {});
      return absl::OkStatus();
    }();
    if (!status.ok()) {
      client->SetEventAsError(usage_event, status);
      if (allocation_event && !allocation_set) {
        std::move(allocation_event)(status);
      }
      return;
    }
  };

  {
    ScopedLauncher launcher(std::move(task), client_->async_work_runner());
    for (size_t i = 0; i < deps_span.size(); ++i) {
      const auto& dep = deps_span[i];
      if (auto event_ref = dep.down_cast<BufferSequencingEvent>()) {
        launcher.AddDependency(event_ref->event().GetAsyncValue());
      } else {
        launcher.AddDependency(dep);
      }
    }
  }

  definition_event_promise.Set(PjRtDeviceEventRef(usage_event));
  src_usage_event_promise.Set(PjRtDeviceEventRef(std::move(usage_event)));
}

}  // namespace xla
