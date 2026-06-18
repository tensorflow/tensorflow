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

#include "xla/pjrt/raw_buffer.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/future.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/device_event_utils.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/staging_buffer.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

namespace {
class HostRawBufferPjRtStagingBuffer : public PjRtStagingBuffer {
 public:
  HostRawBufferPjRtStagingBuffer(absl::Span<uint8_t> data,
                                 PjRtDeviceEventPromiseRef usage_promise,
                                 PjRtRawBufferRef raw_buffer)
      : data_(data),
        usage_promise_(std::move(usage_promise)),
        raw_buffer_(std::move(raw_buffer)) {}

  ~HostRawBufferPjRtStagingBuffer() override { usage_promise_.SetReady(); }

  absl::Span<uint8_t> data() override { return data_; }
  absl::Span<const uint8_t> const_data() const override { return data_; }

 private:
  absl::Span<uint8_t> data_;
  PjRtDeviceEventPromiseRef usage_promise_;
  PjRtRawBufferRef raw_buffer_;
};

class WrappedPjRtStagingBuffer : public PjRtStagingBuffer {
 public:
  explicit WrappedPjRtStagingBuffer(
      tsl::AsyncValueRef<PjRtStagingBuffer> real_buffer)
      : real_buffer_(std::move(real_buffer)) {}

  absl::Span<uint8_t> data() override { return (*real_buffer_).data(); }
  absl::Span<const uint8_t> const_data() const override {
    return (*real_buffer_).const_data();
  }

 private:
  tsl::AsyncValueRef<PjRtStagingBuffer> real_buffer_;
};

Future<> ConvertEventToFuture(PjRtDeviceEventRef event) {
  auto [promise, future] = tsl::MakePromise<void>();
  event.AndThen([promise = std::move(promise), event]() mutable {
    auto error = event.GetErrorIfPresent();
    if (error.has_value() && !error->ok()) {
      promise.Set(*error);
    } else {
      promise.Set();
    }
  });
  return std::move(future);
}
}  // namespace

std::vector<RegisterRawBufferFactory::FactoryFuncT>& GetFactoryFuncs() {
  static auto* const funcs =
      new std::vector<RegisterRawBufferFactory::FactoryFuncT>;
  return *funcs;
}

absl::StatusOr<std::vector<PjRtRawBufferRef>> CommonPjRtRawBuffer::MultiSlice(
    absl::Span<const SliceInfo> slices) {
  std::vector<PjRtRawBufferRef> results;
  results.reserve(slices.size());
  for (const auto& slice : slices) {
    ASSIGN_OR_RETURN(auto sub_slice, Slice(slice.offset, slice.size));
    results.push_back(std::move(sub_slice));
  }
  return results;
}

void CommonPjRtRawBuffer::ScheduleCopyTo(
    AsyncWorkRunner* async_work_runner,
    PjRtDeviceEventRefVector transfer_dependency_events,
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  PjRtDeviceEventSpan events_span(transfer_dependency_events);
  xla::ExecuteWhenReady(
      events_span, async_work_runner,
      [src_raw_buffer = tsl::FormRef(this),
       dst_raw_buffer = std::move(dst_raw_buffer),
       definition_event_promise = std::move(definition_event_promise),
       src_usage_event_promise = std::move(src_usage_event_promise),
       allocation_event = std::move(allocation_event),
       transfer_dependency_events =
           std::move(transfer_dependency_events)]() mutable {
        absl::Status status = xla::GetErrors(transfer_dependency_events);
        if (!status.ok()) {
          if (allocation_event) {
            std::move(allocation_event)(status);
          }
          definition_event_promise.SetError(status);
          src_usage_event_promise.SetError(status);
          return;
        }

        src_raw_buffer->CopyTo(
            std::move(dst_raw_buffer), std::move(definition_event_promise),
            std::move(src_usage_event_promise), std::move(allocation_event));
      });
}

void CommonPjRtRawBuffer::DecrefAfter(PjRtDeviceEventRefVector events) {
  xla::RunWhenReady(events, [this]() { DropRef(); });
}

absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>
PjRtRawBuffer::CreateRawAliasOfBuffer(PjRtBuffer* buffer) {
  for (auto* func : GetFactoryFuncs()) {
    auto res = (*func)(buffer);
    if (res.has_value()) {
      return *res;
    }
  }
  if (buffer == nullptr) {
    return absl::InvalidArgumentError("Cannot create view of null buffer.");
  }
  return absl::UnimplementedError(
      absl::StrCat("CreateRawAliasOfBuffer not implemented for: ",
                   buffer->client()->platform_version()));
}

RegisterRawBufferFactory::RegisterRawBufferFactory(
    RegisterRawBufferFactory::FactoryFuncT func) {
  GetFactoryFuncs().push_back(func);
}

tsl::AsyncValueRef<PjRtStagingBuffer> ToStagingBuffer(
    PjRtRawBufferRef raw_buffer, PjRtDeviceEventPromiseRef usage_promise,
    absl::FunctionRef<tsl::AsyncValueRef<PjRtStagingBuffer>(size_t,
                                                            PjRtMemorySpace*)>
        allocate_staging_buffer) {
  void* host_ptr = raw_buffer->GetHostPointer();
  if (host_ptr != nullptr) {
    size_t size = raw_buffer->GetOnDeviceSizeInBytes();
    absl::Span<uint8_t> data_span(static_cast<uint8_t*>(host_ptr), size);

    auto staging_buffer =
        tsl::MakeAvailableAsyncValueRef<HostRawBufferPjRtStagingBuffer>(
            data_span, std::move(usage_promise), std::move(raw_buffer));

    return tsl::AsyncValueRef<PjRtStagingBuffer>(std::move(staging_buffer));
  } else {
    auto returned_staging_buffer_av =
        tsl::MakeUnconstructedAsyncValueRef<WrappedPjRtStagingBuffer>();

    size_t size = raw_buffer->GetOnDeviceSizeInBytes();
    auto real_staging_buffer_av =
        allocate_staging_buffer(size, raw_buffer->memory_space());

    real_staging_buffer_av.AndThen([raw_buffer, usage_promise,
                                    real_staging_buffer_av,
                                    returned_staging_buffer_av,
                                    size]() mutable {
      if (real_staging_buffer_av.IsError()) {
        returned_staging_buffer_av.SetError(real_staging_buffer_av.GetError());
        usage_promise.SetError(real_staging_buffer_av.GetError());
        return;
      }

      auto real_staging_buffer = real_staging_buffer_av.AsPtr();
      void* dst = real_staging_buffer->data().data();
      auto copy_status_or_event =
          raw_buffer->CopyRawDeviceToHostAndReturnEvent(dst, 0, size);
      if (!copy_status_or_event.ok()) {
        returned_staging_buffer_av.SetError(copy_status_or_event.status());
        usage_promise.SetError(copy_status_or_event.status());
        return;
      }

      auto copy_event = *copy_status_or_event;
      usage_promise.Set(copy_event);

      copy_event.AndThen([returned_staging_buffer_av, real_staging_buffer_av,
                          copy_event, raw_buffer]() mutable {
        if (auto error = copy_event.GetErrorIfPresent()) {
          returned_staging_buffer_av.SetError(*error);
        } else {
          returned_staging_buffer_av.emplace(std::move(real_staging_buffer_av));
        }
      });
    });

    return tsl::AsyncValueRef<PjRtStagingBuffer>(
        std::move(returned_staging_buffer_av));
  }
}

const PJRT_RawBuffer_FunctionTable PjRtRawBuffer::kRawBufferVtable = {
    /*struct_size=*/PJRT_RawBuffer_FunctionTable_STRUCT_SIZE,
    /*instance_size=*/PJRT_RawBuffer_STRUCT_SIZE,
    /*extension_start=*/nullptr,
    /*inc_ref=*/
    +[](PJRT_RawBuffer* raw_buffer) {
      static_cast<PjRtRawBuffer*>(raw_buffer)->AddRef();
    },
    /*dec_ref=*/
    +[](PJRT_RawBuffer* raw_buffer) {
      static_cast<PjRtRawBuffer*>(raw_buffer)->DropRef();
    },
    /*get_on_device_size_in_bytes=*/
    +[](const PJRT_RawBuffer* raw_buffer) -> size_t {
      return static_cast<const PjRtRawBuffer*>(raw_buffer)
          ->GetOnDeviceSizeInBytes();
    },
    /*get_memory_space=*/
    +[](const PJRT_RawBuffer* raw_buffer) -> PJRT_Memory* {
      return static_cast<const PjRtRawBuffer*>(raw_buffer)
          ->memory_space()
          ->ToCApiPtr();
    },
    /*get_host_pointer=*/
    +[](const PJRT_RawBuffer* raw_buffer) -> void* {
      return static_cast<const PjRtRawBuffer*>(raw_buffer)->GetHostPointer();
    },
    /*copy_raw_host_to_device_and_return_event=*/
    +[](PJRT_RawBuffer* raw_buffer, const void* src, int64_t offset,
        int64_t transfer_size, PJRT_DeviceEvent* event) -> PJRT_Error* {
      auto result =
          static_cast<PjRtRawBuffer*>(raw_buffer)
              ->CopyRawHostToDeviceAndReturnEvent(src, offset, transfer_size);
      if (!result.ok()) {
        return pjrt::StatusToPjRtError(result.status());
      }
      *event = std::move(*result).release().ToC();
      return nullptr;
    },
    /*copy_raw_device_to_host_and_return_event=*/
    +[](PJRT_RawBuffer* raw_buffer, void* dst, int64_t offset,
        int64_t transfer_size, PJRT_DeviceEvent* event) -> PJRT_Error* {
      auto result =
          static_cast<PjRtRawBuffer*>(raw_buffer)
              ->CopyRawDeviceToHostAndReturnEvent(dst, offset, transfer_size);
      if (!result.ok()) {
        return pjrt::StatusToPjRtError(result.status());
      }
      *event = std::move(*result).release().ToC();
      return nullptr;
    },
    /*opaque_device_memory_data_pointer=*/
    +[](const PJRT_RawBuffer* raw_buffer) -> void* {
      return static_cast<const PjRtRawBuffer*>(raw_buffer)
          ->OpaqueDeviceMemoryDataPointer();
    },
};

PjRtRawBuffer::PjRtRawBuffer() { vtable = &kRawBufferVtable; }

absl::StatusOr<PjRtDeviceEventRef>
PjRtRawBufferInterface::CopyRawHostToDeviceAndReturnEvent(
    const void* src, int64_t offset, int64_t transfer_size) {
  PJRT_DeviceEvent device_event;
  PJRT_Error* error = vtable->copy_raw_host_to_device_and_return_event(
      this, src, offset, transfer_size, &device_event);
  if (error != nullptr) {
    return pjrt::PjrtErrorToStatus(error);
  }
  return PjRtDeviceEventRef::TakeRef(PjRtDeviceEventPtr(device_event));
}

absl::StatusOr<PjRtDeviceEventRef>
PjRtRawBufferInterface::CopyRawDeviceToHostAndReturnEvent(
    void* dst, int64_t offset, int64_t transfer_size) {
  PJRT_DeviceEvent device_event;
  PJRT_Error* error = vtable->copy_raw_device_to_host_and_return_event(
      this, dst, offset, transfer_size, &device_event);
  if (error != nullptr) {
    return pjrt::PjrtErrorToStatus(error);
  }
  return PjRtDeviceEventRef::TakeRef(PjRtDeviceEventPtr(device_event));
}

void* PjRtRawBufferInterface::OpaqueDeviceMemoryDataPointer() const {
  return vtable->opaque_device_memory_data_pointer(this);
}

void PjRtRawBufferInterface::AddRef() { vtable->inc_ref(this); }

void PjRtRawBufferInterface::DropRef() { vtable->dec_ref(this); }

PjRtMemorySpace* PjRtRawBufferInterface::memory_space() const {
  return PjRtMemorySpace::FromC(vtable->get_memory_space(this));
}

void* PjRtRawBufferInterface::GetHostPointer() const {
  return vtable->get_host_pointer(this);
}

size_t PjRtRawBufferInterface::GetOnDeviceSizeInBytes() const {
  return vtable->get_on_device_size_in_bytes(this);
}

Future<> PjRtRawBufferInterface::CopyRawHostToDevice(const void* src,
                                                     int64_t offset,
                                                     int64_t transfer_size) {
  ASSIGN_OR_RETURN(auto event, CopyRawHostToDeviceAndReturnEvent(
                                   src, offset, transfer_size));
  return ConvertEventToFuture(std::move(event));
}

Future<> PjRtRawBufferInterface::CopyRawDeviceToHost(void* dst, int64_t offset,
                                                     int64_t transfer_size) {
  ASSIGN_OR_RETURN(auto event, CopyRawDeviceToHostAndReturnEvent(
                                   dst, offset, transfer_size));
  return ConvertEventToFuture(std::move(event));
}

}  // namespace xla
