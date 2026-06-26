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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/future.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
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

absl::StatusOr<std::vector<PjRtRawBufferRef>> PjRtRawBuffer::MultiSlice(
    absl::Span<const SliceInfo> slices) {
  std::vector<PjRtRawBufferRef> results;
  results.reserve(slices.size());
  for (const auto& slice : slices) {
    ASSIGN_OR_RETURN(auto sub_slice, Slice(slice.offset, slice.size));
    results.push_back(std::move(sub_slice));
  }
  return results;
}

void PjRtRawBuffer::DecrefAfter(PjRtDeviceEventRefVector events) {
  xla::RunWhenReady(events, [this]() { DropRef(); });
}

absl::StatusOr<PjRtRawBufferRef> PjRtRawBuffer::CreateRawAliasOfBuffer(
    PjRtBuffer* buffer) {
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
        int64_t transfer_size, PJRT_DeviceEventVector* dependencies,
        PJRT_DeviceEvent* event) -> PJRT_Error* {
      PjRtDeviceEventRefVector cpp_deps;
      if (dependencies != nullptr) {
        cpp_deps = PjRtDeviceEventRefVector::MoveFromC(dependencies);
      }
      auto result = static_cast<PjRtRawBuffer*>(raw_buffer)
                        ->CopyRawHostToDeviceAndReturnEvent(
                            src, offset, transfer_size, std::move(cpp_deps));
      if (!result.ok()) {
        return pjrt::StatusToPjRtError(result.status());
      }
      *event = std::move(*result).release().ToC();
      return nullptr;
    },
    /*copy_raw_device_to_host_and_return_event=*/
    +[](PJRT_RawBuffer* raw_buffer, void* dst, int64_t offset,
        int64_t transfer_size, PJRT_DeviceEventVector* dependencies,
        PJRT_DeviceEvent* event) -> PJRT_Error* {
      PjRtDeviceEventRefVector cpp_deps;
      if (dependencies != nullptr) {
        cpp_deps = PjRtDeviceEventRefVector::MoveFromC(dependencies);
      }
      auto result = static_cast<PjRtRawBuffer*>(raw_buffer)
                        ->CopyRawDeviceToHostAndReturnEvent(
                            dst, offset, transfer_size, std::move(cpp_deps));
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
    /*make_allocation_ready_event=*/
    +[](PJRT_RawBuffer* raw_buffer, PJRT_DeviceEvent* event) -> PJRT_Error* {
      auto result =
          static_cast<PjRtRawBuffer*>(raw_buffer)->MakeAllocationReadyEvent();
      if (!result.ok()) {
        return pjrt::StatusToPjRtError(result.status());
      }
      *event = std::move(*result).release().ToC();
      return nullptr;
    },
    /*get_raw_buffer_async_value=*/
    +[](PJRT_RawBuffer* raw_buffer, PJRT_DeviceEvent* event) -> PJRT_Error* {
      auto result =
          static_cast<PjRtRawBuffer*>(raw_buffer)->GetRawBufferAsyncValue();
      *event = result.ToC();
      return nullptr;
    },
    /*is_mutable=*/
    +[](const PJRT_RawBuffer* raw_buffer) -> bool {
      return static_cast<const PjRtRawBuffer*>(raw_buffer)->is_mutable();
    },
    /*slice=*/
    +[](PJRT_RawBuffer* raw_buffer, int64_t offset, int64_t slice_size,
        PJRT_RawBuffer** sliced_buffer) -> PJRT_Error* {
      auto result =
          static_cast<PjRtRawBuffer*>(raw_buffer)->Slice(offset, slice_size);
      if (!result.ok()) {
        return pjrt::StatusToPjRtError(result.status());
      }
      *sliced_buffer = std::move(*result).release();
      return nullptr;
    },
    /*schedule_copy_to=*/
    +[](PJRT_RawBuffer* src_buffer,
        PJRT_DeviceEventVector* transfer_dependency_events,
        PJRT_RawBuffer* dst_buffer,
        PJRT_DeviceEventPromise* definition_event_promise,
        PJRT_DeviceEventPromise* src_usage_event_promise,
        void (*allocation_event_callback)(PJRT_Error* status, void* user_data),
        void* allocation_event_user_data) -> void {
      PjRtRawBuffer* cpp_src = static_cast<PjRtRawBuffer*>(src_buffer);
      PjRtRawBuffer* cpp_dst = static_cast<PjRtRawBuffer*>(dst_buffer);

      PjRtDeviceEventPromiseRef cpp_def_promise =
          PjRtDeviceEventPromiseRef::TakeRef(definition_event_promise);
      PjRtDeviceEventPromiseRef cpp_usage_promise =
          PjRtDeviceEventPromiseRef::TakeRef(src_usage_event_promise);

      absl::AnyInvocable<void(absl::Status) &&> cpp_allocation_event =
          !allocation_event_callback
              ? absl::AnyInvocable<void(absl::Status) &&>()
              : [allocation_event_callback,
                 allocation_event_user_data](absl::Status status) {
                  allocation_event_callback(pjrt::StatusToPjRtError(status),
                                            allocation_event_user_data);
                };

      if (transfer_dependency_events != nullptr) {
        PjRtDeviceEventRefVector cpp_deps =
            PjRtDeviceEventRefVector::MoveFromC(transfer_dependency_events);
        cpp_src->ScheduleCopyTo(std::move(cpp_deps), tsl::FormRef(cpp_dst),
                                std::move(cpp_def_promise),
                                std::move(cpp_usage_promise),
                                std::move(cpp_allocation_event));
      } else {
        cpp_src->CopyTo(tsl::FormRef(cpp_dst), std::move(cpp_def_promise),
                        std::move(cpp_usage_promise),
                        std::move(cpp_allocation_event));
      }
    },
};

PjRtRawBuffer::PjRtRawBuffer() { vtable = &kRawBufferVtable; }

absl::StatusOr<PjRtDeviceEventRef>
PjRtRawBufferInterface::CopyRawHostToDeviceAndReturnEvent(
    const void* src, int64_t offset, int64_t transfer_size,
    PjRtDeviceEventRefVector dependencies) {
  PJRT_DeviceEvent device_event;
  PJRT_Error* error = vtable->copy_raw_host_to_device_and_return_event(
      this, src, offset, transfer_size, &dependencies.ToC(), &device_event);
  if (error != nullptr) {
    return pjrt::PjrtErrorToStatus(error);
  }
  return PjRtDeviceEventRef::TakeRef(PjRtDeviceEventPtr(device_event));
}

absl::StatusOr<PjRtDeviceEventRef>
PjRtRawBufferInterface::CopyRawDeviceToHostAndReturnEvent(
    void* dst, int64_t offset, int64_t transfer_size,
    PjRtDeviceEventRefVector dependencies) {
  PJRT_DeviceEvent device_event;
  PJRT_Error* error = vtable->copy_raw_device_to_host_and_return_event(
      this, dst, offset, transfer_size, &dependencies.ToC(), &device_event);
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

bool PjRtRawBufferInterface::is_mutable() const {
  return vtable->is_mutable(this);
}

absl::StatusOr<PjRtDeviceEventRef>
PjRtRawBufferInterface::MakeAllocationReadyEvent() {
  PJRT_DeviceEvent device_event;
  PJRT_Error* error = vtable->make_allocation_ready_event(this, &device_event);
  if (error != nullptr) {
    return pjrt::PjrtErrorToStatus(error);
  }
  return PjRtDeviceEventRef::TakeRef(PjRtDeviceEventPtr(device_event));
}

PjRtDeviceEventPtr PjRtRawBufferInterface::GetRawBufferAsyncValue() {
  PJRT_DeviceEvent device_event;
  PJRT_Error* error = vtable->get_raw_buffer_async_value(this, &device_event);
  if (error != nullptr) {
    LOG(FATAL) << "Failed to get raw buffer async value: "
               << pjrt::PjrtErrorToStatus(error);
  }
  return PjRtDeviceEventPtr(device_event);
}

absl::StatusOr<PjRtRawBufferRef> PjRtRawBufferInterface::Slice(int64_t offset,
                                                               int64_t size) {
  PJRT_RawBuffer* sliced_c_buffer;
  PJRT_Error* error = vtable->slice(this, offset, size, &sliced_c_buffer);
  if (error != nullptr) {
    return pjrt::PjrtErrorToStatus(error);
  }
  if (sliced_c_buffer->vtable != &PjRtRawBuffer::kRawBufferVtable) {
    tsl::TakeRef(static_cast<PjRtRawBufferInterface*>(sliced_c_buffer));
    return absl::UnimplementedError("Slice not supported");
  }
  return tsl::TakeRef(static_cast<PjRtRawBuffer*>(sliced_c_buffer));
}

absl::StatusOr<std::vector<PjRtRawBufferRef>>
PjRtRawBufferInterface::MultiSlice(absl::Span<const SliceInfo> slices) {
  if (auto* cpp_buf = down_cast<PjRtRawBuffer>()) {
    return cpp_buf->MultiSlice(slices);
  }
  std::vector<PjRtRawBufferRef> results;
  results.reserve(slices.size());
  for (const auto& slice : slices) {
    ASSIGN_OR_RETURN(auto sub_slice, Slice(slice.offset, slice.size));
    results.push_back(std::move(sub_slice));
  }
  return results;
}

static void RawBufferInterfaceScheduleCopyToHelper(
    PjRtRawBufferInterface* src, PJRT_DeviceEventVector* c_deps,
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  const PJRT_RawBuffer_FunctionTable* vtable = src->vtable;
  PJRT_RawBuffer* c_dst = dst_raw_buffer.get();

  PJRT_DeviceEventPromise* c_def_promise = definition_event_promise.get();
  c_def_promise->vtable->inc_ref(c_def_promise);

  PJRT_DeviceEventPromise* c_usage_promise = src_usage_event_promise.get();
  c_usage_promise->vtable->inc_ref(c_usage_promise);

  absl::AnyInvocable<void(absl::Status) &&>* cb = nullptr;
  void (*c_callback)(PJRT_Error*, void*) = nullptr;

  if (allocation_event) {
    cb = new absl::AnyInvocable<void(absl::Status) &&>(
        std::move(allocation_event));
    c_callback = +[](PJRT_Error* error, void* user_data) {
      auto* cb =
          static_cast<absl::AnyInvocable<void(absl::Status) &&>*>(user_data);
      absl::Status status = pjrt::PjrtErrorToStatus(error);
      if (*cb) {
        std::move (*cb)(status);
      }
      delete cb;
    };
  }

  vtable->schedule_copy_to(src, c_deps, c_dst, c_def_promise, c_usage_promise,
                           c_callback, cb);
}

void PjRtRawBufferInterface::CopyTo(
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  RawBufferInterfaceScheduleCopyToHelper(
      this, /*c_deps=*/nullptr, std::move(dst_raw_buffer),
      std::move(definition_event_promise), std::move(src_usage_event_promise),
      std::move(allocation_event));
}

void PjRtRawBufferInterface::ScheduleCopyTo(
    PjRtDeviceEventRefVector transfer_dependency_events,
    PjRtRawBufferRef dst_raw_buffer,
    PjRtDeviceEventPromiseRef definition_event_promise,
    PjRtDeviceEventPromiseRef src_usage_event_promise,
    absl::AnyInvocable<void(absl::Status) &&> allocation_event) {
  RawBufferInterfaceScheduleCopyToHelper(
      this, &transfer_dependency_events.ToC(), std::move(dst_raw_buffer),
      std::move(definition_event_promise), std::move(src_usage_event_promise),
      std::move(allocation_event));
}

void PjRtRawBufferInterface::DecrefAfter(PjRtDeviceEventRefVector events) {
  if (auto* cpp_buf = down_cast<PjRtRawBuffer>()) {
    cpp_buf->DecrefAfter(std::move(events));
  } else {
    xla::RunWhenReady(events, [this]() { DropRef(); });
  }
}

}  // namespace xla
