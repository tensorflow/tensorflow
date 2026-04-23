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

#ifndef XLA_PJRT_DEVICE_EVENT_H_
#define XLA_PJRT_DEVICE_EVENT_H_

#include <cstddef>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

namespace internal {

template <typename T>
const PJRT_DeviceEvent_FunctionTable* GetBuiltinDeviceEventCApiFunctionTable() {
  static const PJRT_DeviceEvent_FunctionTable device_event_vtable = {
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
      }};
  return &device_event_vtable;
}

}  // namespace internal

// RAII type for holding type-checked tsl::AsyncValue* for the
// underlying device event types.
class PjRtDeviceEventRef {
 public:
  PjRtDeviceEventRef() = default;
  ~PjRtDeviceEventRef() { reset(); }

  PjRtDeviceEventRef(const PjRtDeviceEventRef& other)
      : vtable_(other.vtable_), device_event_(other.CopyRawRef()) {}

  PjRtDeviceEventRef(PjRtDeviceEventRef&& other) noexcept
      : vtable_(other.vtable_), device_event_(other.ReleaseRawRef()) {}

  PjRtDeviceEventRef& operator=(const PjRtDeviceEventRef& other) {
    if (this != &other) {
      reset();
      vtable_ = other.vtable_;
      device_event_ = other.CopyRawRef();
    }
    return *this;
  }

  PjRtDeviceEventRef& operator=(PjRtDeviceEventRef&& other) noexcept {
    if (this != &other) {
      reset();
      vtable_ = other.vtable_;
      device_event_ = other.ReleaseRawRef();
    }
    return *this;
  }

  template <typename T>
  explicit PjRtDeviceEventRef(tsl::AsyncValueRef<T> value)
      : vtable_(internal::GetBuiltinDeviceEventCApiFunctionTable<T>()),
        device_event_(value.ReleaseRCRef().release()) {}

  // Runs a callback when an event becomes ready.
  template <typename Waiter>
  void AndThen(Waiter&& cb) const {
    async_value()->AndThen(std::forward<Waiter>(cb));
  }

  // TODO(parkers): Remove direct async_value usages.
  tsl::AsyncValue* async_value() const {
    return reinterpret_cast<tsl::AsyncValue*>(device_event_);
  }

  template <typename T>
  tsl::AsyncValueRef<T> down_cast() const& {
    if (device_event_ == nullptr ||
        vtable_ != internal::GetBuiltinDeviceEventCApiFunctionTable<T>()) {
      return nullptr;
    }
    return tsl::AsyncValueRef<T>(tsl::FormRef(async_value()));
  }

  template <typename T>
  tsl::AsyncValueRef<T> down_cast() && {
    if (device_event_ == nullptr ||
        vtable_ != internal::GetBuiltinDeviceEventCApiFunctionTable<T>()) {
      return nullptr;
    }
    return tsl::AsyncValueRef<T>(
        tsl::TakeRef(reinterpret_cast<tsl::AsyncValue*>(ReleaseRawRef())));
  }

  void reset() {
    if (device_event_ != nullptr) {
      auto* vtable = vtable_;
      vtable->dec_ref(ReleaseRawRef());
    }
  }

  explicit operator bool() const { return device_event_ != nullptr; }

 private:
  const PJRT_DeviceEvent_FunctionTable* vtable_ = nullptr;
  void* CopyRawRef() const {
    if (device_event_ != nullptr) {
      vtable_->inc_ref(device_event_);
    }
    return device_event_;
  }
  void* ReleaseRawRef() {
    vtable_ = nullptr;
    auto* device_event = device_event_;
    device_event_ = nullptr;
    return device_event;
  }
  void* device_event_ = nullptr;
};

// Instead of taking a device event as an argument, apis may instead decide to
// return a promise which is fulfilled later.
class PjRtDeviceEventPromise
    : public tsl::ReferenceCounted<PjRtDeviceEventPromise> {
 public:
  virtual ~PjRtDeviceEventPromise() = default;

  // The underlying AsyncValue.
  virtual tsl::AsyncValue* async_value() const = 0;

  // Fulfill the promise.
  virtual void Set(PjRtDeviceEventRef event) = 0;

  // Mark the promise as an error.
  virtual void SetError(absl::Status s) = 0;

  // Mark the event as ready.
  virtual void SetReady() = 0;
};

// A collection of events. This is not an event itself because we may want to
// add events in the future.
class PjRtDeviceEventSet {
 public:
  virtual ~PjRtDeviceEventSet() = default;
};

}  // namespace xla

#endif  // XLA_PJRT_DEVICE_EVENT_H_
