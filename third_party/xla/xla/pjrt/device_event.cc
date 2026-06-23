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
#include "xla/pjrt/device_event.h"

#include <algorithm>
#include <cstddef>
#include <optional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/tsl/concurrency/async_value.h"

namespace xla {
namespace internal {

const PJRT_DeviceEvent_FunctionTable* GetBuiltinAsyncValueCApiFunctionTable() {
  static const PJRT_DeviceEvent_FunctionTable vtable = {
      /*struct_size=*/sizeof(PJRT_DeviceEvent_FunctionTable),
      /*extension_start=*/nullptr,
      /*inc_ref=*/
      +[](void* device_event) {
        reinterpret_cast<tsl::AsyncValue*>(device_event)->AddRef();
      },
      /*dec_ref=*/
      +[](void* device_event) {
        reinterpret_cast<tsl::AsyncValue*>(device_event)->DropRef();
      },
      /*and_then=*/
      +[](void* device_event, PJRT_DeviceEvent_AndThen callback,
          void* user_arg) {
        reinterpret_cast<tsl::AsyncValue*>(device_event)
            ->AndThen([callback, user_arg]() { callback(user_arg); });
      },
      /*get_error_if_present=*/
      +[](void* device_event, PJRT_Error_Code* code, const char** message,
          size_t* message_size) -> int {
        if (const absl::Status* error =
                reinterpret_cast<tsl::AsyncValue*>(device_event)
                    ->GetErrorIfPresent()) {
          *code = pjrt::StatusCodeToPjrtErrorCode(error->code());
          absl::string_view error_message = error->message();
          *message = error_message.data();
          *message_size = error_message.size();
          return 1;
        }
        return 0;
      },
      /*parent=*/nullptr,  // Root of local AsyncValue hierarchy
      /*get_state=*/
      +[](void* device_event) -> PJRT_DeviceEvent_State {
        auto* async_value = reinterpret_cast<tsl::AsyncValue*>(device_event);
        return ToPjrtDeviceEventState(async_value->state());
      }};
  return &vtable;
}

PJRT_DeviceEvent_State ToPjrtDeviceEventState(tsl::AsyncValue::State state) {
  switch (state) {
    case tsl::AsyncValue::State::kConcrete:
      return PJRT_DeviceEvent_State_Ready;
    case tsl::AsyncValue::State::kError:
      return PJRT_DeviceEvent_State_Error;
    default:
      return PJRT_DeviceEvent_State_Unavailable;
  }
}

}  // namespace internal

void PjRtDeviceEventPtr::IncRef() const {
  if (event_.device_event != nullptr && event_.vtable != nullptr) {
    event_.vtable->inc_ref(event_.device_event);
  }
}

void PjRtDeviceEventPtr::DecRef() const {
  if (event_.device_event != nullptr && event_.vtable != nullptr) {
    event_.vtable->dec_ref(event_.device_event);
  }
}

PjRtDeviceEventRef PjRtDeviceEventPtr::CopyRef() const {
  IncRef();
  return PjRtDeviceEventRef::TakeRef(*this);
}

std::optional<absl::Status> PjRtDeviceEventPtr::GetErrorIfPresent() const {
  if (event_.device_event == nullptr || event_.vtable == nullptr) {
    return std::nullopt;
  }
  if (IsCompatibleWithLocalAsyncValue()) {
    if (const absl::Status* error =
            reinterpret_cast<tsl::AsyncValue*>(event_.device_event)
                ->GetErrorIfPresent()) {
      return *error;
    }
    return std::nullopt;
  }

  PJRT_Error_Code code;
  const char* message = nullptr;
  size_t message_size = 0;
  if (event_.vtable->get_error_if_present(event_.device_event, &code, &message,
                                          &message_size)) {
    return absl::Status(static_cast<absl::StatusCode>(code),
                        absl::string_view(message, message_size));
  }
  return std::nullopt;
}

PJRT_DeviceEvent_State PjRtDeviceEventPtr::state() const {
  if (event_.device_event == nullptr || event_.vtable == nullptr) {
    LOG(FATAL) << "state() called on invalid PjRtDeviceEventPtr.";
  }
  return event_.vtable->get_state(event_.device_event);
}

std::optional<PjRtDeviceEventPtr::DefinitionStreamInfo>
PjRtDeviceEventPtr::GetDefinitionStream() const {
  if (event_.device_event == nullptr || event_.vtable == nullptr) {
    return std::nullopt;
  }
  uint64_t sequence_number = 0;
  intptr_t stream = event_.vtable->get_definition_stream(event_.device_event,
                                                         &sequence_number);
  if (stream == 0) {
    return std::nullopt;
  }
  return DefinitionStreamInfo{stream, sequence_number};
}

tsl::AsyncValue* PjRtDeviceEventPtr::async_value() const {
  if (event_.device_event == nullptr || !IsCompatibleWithLocalAsyncValue()) {
    return nullptr;
  }
  return reinterpret_cast<tsl::AsyncValue*>(event_.device_event);
}

bool PjRtDeviceEventPtr::IsCompatibleWithLocalAsyncValue() const {
  const PJRT_DeviceEvent_FunctionTable* current = event_.vtable;
  const PJRT_DeviceEvent_FunctionTable* target =
      internal::GetBuiltinAsyncValueCApiFunctionTable();
  while (current != nullptr) {
    if (current == target) {
      return true;
    }
    if (current->parent == current) {
      break;
    }
    current = current->parent;
  }
  return false;
}

static const PJRT_DeviceEventPromise_FunctionTable kBuiltinPromiseVtable = {
    /*struct_size=*/sizeof(PJRT_DeviceEventPromise_FunctionTable),
    /*instance_size=*/sizeof(PJRT_DeviceEventPromise),
    /*extension_start=*/nullptr,
    /*inc_ref=*/
    +[](PJRT_DeviceEventPromise* promise) {
      static_cast<PjRtDeviceEventPromise*>(promise)->AddRef();
    },
    /*dec_ref=*/
    +[](PJRT_DeviceEventPromise* promise) {
      static_cast<PjRtDeviceEventPromise*>(promise)->DropRef();
    },
    /*event=*/
    +[](PJRT_DeviceEventPromise* promise) -> PJRT_DeviceEvent {
      return static_cast<PjRtDeviceEventPromise*>(promise)->event().ToC();
    },
    /*set=*/
    +[](PJRT_DeviceEventPromise* promise, PJRT_DeviceEvent event) {
      static_cast<PjRtDeviceEventPromise*>(promise)->Set(
          PjRtDeviceEventRef::TakeRef(PjRtDeviceEventPtr(event)));
    },
    /*set_error=*/
    +[](PJRT_DeviceEventPromise* promise, PJRT_Error* error) {
      static_cast<PjRtDeviceEventPromise*>(promise)->SetError(
          pjrt::PjrtErrorToStatus(error));
    },
    /*set_ready=*/
    +[](PJRT_DeviceEventPromise* promise) {
      static_cast<PjRtDeviceEventPromise*>(promise)->SetReady();
    }};

namespace internal {
const PJRT_DeviceEventPromise_FunctionTable*
GetBuiltinDeviceEventPromiseCApiFunctionTable() {
  return &kBuiltinPromiseVtable;
}
}  // namespace internal

PjRtDeviceEventPromise::PjRtDeviceEventPromise() {
  vtable = &kBuiltinPromiseVtable;
}

PjRtDeviceEventPromiseRef::PjRtDeviceEventPromiseRef(
    tsl::RCReference<PjRtDeviceEventPromise> promise)
    : promise_(promise.release()) {}

PjRtDeviceEventPromiseRef PjRtDeviceEventPromiseRef::TakeRef(
    PJRT_DeviceEventPromise* promise) {
  return PjRtDeviceEventPromiseRef(promise);
}

PjRtDeviceEventPromiseRef PjRtDeviceEventPromiseRef::FormRef(
    PJRT_DeviceEventPromise* promise) {
  if (promise != nullptr) {
    promise->vtable->inc_ref(promise);
  }
  return PjRtDeviceEventPromiseRef(promise);
}

PjRtDeviceEventPromiseRef::~PjRtDeviceEventPromiseRef() {
  if (promise_ != nullptr) {
    promise_->vtable->dec_ref(promise_);
  }
}

PjRtDeviceEventPromiseRef::PjRtDeviceEventPromiseRef(
    const PjRtDeviceEventPromiseRef& other)
    : promise_(other.promise_) {
  if (promise_ != nullptr) {
    promise_->vtable->inc_ref(promise_);
  }
}

PjRtDeviceEventPromiseRef& PjRtDeviceEventPromiseRef::operator=(
    const PjRtDeviceEventPromiseRef& other) {
  if (this != &other) {
    if (promise_ != nullptr) {
      promise_->vtable->dec_ref(promise_);
    }
    promise_ = other.promise_;
    if (promise_ != nullptr) {
      promise_->vtable->inc_ref(promise_);
    }
  }
  return *this;
}

PjRtDeviceEventPromiseRef::PjRtDeviceEventPromiseRef(
    PjRtDeviceEventPromiseRef&& other) noexcept
    : promise_(other.promise_) {
  other.promise_ = nullptr;
}

PjRtDeviceEventPromiseRef& PjRtDeviceEventPromiseRef::operator=(
    PjRtDeviceEventPromiseRef&& other) noexcept {
  if (this != &other) {
    if (promise_ != nullptr) {
      promise_->vtable->dec_ref(promise_);
    }
    promise_ = other.promise_;
    other.promise_ = nullptr;
  }
  return *this;
}

PjRtDeviceEventPtr PjRtDeviceEventPromiseRef::event() const {
  return PjRtDeviceEventPtr(promise_->vtable->event(promise_));
}

void PjRtDeviceEventPromiseRef::Set(PjRtDeviceEventRef event) const {
  promise_->vtable->set(promise_, std::move(event).release().ToC());
}

void PjRtDeviceEventPromiseRef::SetError(absl::Status s) const {
  PJRT_Error* error = pjrt::StatusToPjRtError(std::move(s));
  promise_->vtable->set_error(promise_, error);
}

void PjRtDeviceEventPromiseRef::SetReady() const {
  promise_->vtable->set_ready(promise_);
}

static void DestroyDeviceEventVectorData(PJRT_DeviceEvent* data) {
  delete[] data;
}

PjRtDeviceEventRefVector::~PjRtDeviceEventRefVector() { Clear(); }

PjRtDeviceEventRefVector::PjRtDeviceEventRefVector(
    const PjRtDeviceEventRefVector& other) {
  reserve(other.size());
  for (size_t i = 0; i < other.size(); ++i) {
    push_back(other[i].CopyRef());
  }
}

PjRtDeviceEventRefVector::PjRtDeviceEventRefVector(
    PjRtDeviceEventRefVector&& other) noexcept
    : vector_(other.vector_) {
  other.vector_ = {nullptr, 0, 0, nullptr};
}

PjRtDeviceEventRefVector& PjRtDeviceEventRefVector::operator=(
    const PjRtDeviceEventRefVector& other) {
  if (this != &other) {
    Clear();
    reserve(other.size());
    for (size_t i = 0; i < other.size(); ++i) {
      push_back(other[i].CopyRef());
    }
  }
  return *this;
}

PjRtDeviceEventRefVector& PjRtDeviceEventRefVector::operator=(
    PjRtDeviceEventRefVector&& other) noexcept {
  if (this != &other) {
    Clear();
    vector_ = other.vector_;
    other.vector_ = {nullptr, 0, 0, nullptr};
  }
  return *this;
}

PjRtDeviceEventRefVector::PjRtDeviceEventRefVector(
    std::initializer_list<PjRtDeviceEventRef> init) {
  reserve(init.size());
  for (const auto& ref : init) {
    push_back(ref.ptr().CopyRef());
  }
}

void PjRtDeviceEventRefVector::push_back(const PjRtDeviceEventRef& value) {
  reserve(vector_.size + 1);
  vector_.data[vector_.size] = value.ptr().CopyRef().release().ToC();
  ++vector_.size;
}

void PjRtDeviceEventRefVector::push_back(PjRtDeviceEventRef&& value) {
  reserve(vector_.size + 1);
  vector_.data[vector_.size] = std::move(value).release().ToC();
  ++vector_.size;
}

void PjRtDeviceEventRefVector::reserve(size_t new_cap) {
  if (new_cap <= vector_.capacity) {
    return;
  }
  new_cap = std::max(new_cap, vector_.capacity == 0 ? 4 : vector_.capacity * 2);
  auto* new_data = new PJRT_DeviceEvent[new_cap];
  for (size_t i = 0; i < vector_.size; ++i) {
    new_data[i] = vector_.data[i];
  }
  if (vector_.destroy != nullptr) {
    vector_.destroy(vector_.data);
  }
  vector_.data = new_data;
  vector_.capacity = new_cap;
  vector_.destroy = DestroyDeviceEventVectorData;
}

void PjRtDeviceEventRefVector::Clear() {
  for (size_t i = 0; i < vector_.size; ++i) {
    PjRtDeviceEventPtr(vector_.data[i]).DecRef();
  }
  if (vector_.destroy != nullptr) {
    vector_.destroy(vector_.data);
  }
  vector_ = {nullptr, 0, 0, nullptr};
}

PjRtDeviceEventRefVector PjRtDeviceEventRefVector::MoveFromC(
    PJRT_DeviceEventVector* data) {
  PjRtDeviceEventRefVector v;
  v.vector_ = *data;
  *data = {nullptr, 0, 0, nullptr};
  return v;
}

}  // namespace xla
