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
#include <type_traits>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

namespace internal {

const PJRT_DeviceEvent_FunctionTable* GetBuiltinAsyncValueCApiFunctionTable();

PJRT_DeviceEvent_State ToPjrtDeviceEventState(tsl::AsyncValue::State state);

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
      },
      /*parent=*/GetBuiltinAsyncValueCApiFunctionTable(),
      /*get_state=*/
      +[](void* device_event) -> PJRT_DeviceEvent_State {
        auto* async_value = reinterpret_cast<tsl::AsyncValue*>(device_event);
        return ToPjrtDeviceEventState(async_value->state());
      }};
  return &device_event_vtable;
}

}  // namespace internal

class PjRtDeviceEventRef;

// c++ convenience wrapper around PJRT_DeviceEvent.
class PjRtDeviceEventPtr {
 public:
  PjRtDeviceEventPtr() = default;
  explicit PjRtDeviceEventPtr(PJRT_DeviceEvent event) : event_(event) {}

  // Runs a callback when the event becomes ready.
  template <typename Waiter>
  void AndThen(Waiter&& cb) const {
    if (event_.device_event == nullptr || event_.vtable == nullptr) {
      return;
    }
    if (IsCompatibleWithLocalAsyncValue()) {
      reinterpret_cast<tsl::AsyncValue*>(event_.device_event)
          ->AndThen(std::forward<Waiter>(cb));
      return;
    }

    using WaiterType = std::decay_t<Waiter>;
    auto* waiter_ptr = new WaiterType(std::forward<Waiter>(cb));

    auto c_callback = +[](void* user_arg) {
      auto* waiter = static_cast<WaiterType*>(user_arg);
      std::move (*waiter)();
      delete waiter;
    };

    event_.vtable->and_then(event_.device_event, c_callback, waiter_ptr);
  }

  template <typename T>
  tsl::AsyncValueRef<T> down_cast() const {
    if (event_.device_event == nullptr ||
        event_.vtable !=
            internal::GetBuiltinDeviceEventCApiFunctionTable<T>()) {
      return nullptr;
    }
    return tsl::AsyncValueRef<T>(
        tsl::FormRef(reinterpret_cast<tsl::AsyncValue*>(event_.device_event)));
  }

  template <typename T>
  tsl::AsyncValueRef<T> steal_down_cast() && {
    if (event_.device_event == nullptr ||
        event_.vtable !=
            internal::GetBuiltinDeviceEventCApiFunctionTable<T>()) {
      return nullptr;
    }
    auto result = tsl::AsyncValueRef<T>(
        tsl::TakeRef(reinterpret_cast<tsl::AsyncValue*>(event_.device_event)));
    event_ = {nullptr, nullptr};
    return result;
  }

  std::optional<absl::Status> GetErrorIfPresent() const;

  PJRT_DeviceEvent ToC() const { return event_; }

  // TODO(parkers): Remove direct async_value usages.
  tsl::AsyncValue* async_value() const;

  explicit operator bool() const { return event_.device_event != nullptr; }

  PjRtDeviceEventRef CopyRef() const;
  void DecRef() const;

 private:
  void IncRef() const;
  bool IsCompatibleWithLocalAsyncValue() const;

  PJRT_DeviceEvent event_ = {nullptr, nullptr};
};

// RAII type for holding owning references to device events.
class PjRtDeviceEventRef {
 public:
  PjRtDeviceEventRef() = default;

  ~PjRtDeviceEventRef() { reset(); }

  PjRtDeviceEventRef(const PjRtDeviceEventRef& other)
      : ptr_(other.ptr_.CopyRef().release()) {}

  PjRtDeviceEventRef(PjRtDeviceEventRef&& other) noexcept
      : ptr_(std::exchange(other.ptr_, {})) {}

  PjRtDeviceEventRef& operator=(const PjRtDeviceEventRef& other) {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_.CopyRef().release();
    }
    return *this;
  }

  PjRtDeviceEventRef& operator=(PjRtDeviceEventRef&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = std::move(other).release();
    }
    return *this;
  }

  template <typename T>
  explicit PjRtDeviceEventRef(tsl::AsyncValueRef<T> value)
      : ptr_({internal::GetBuiltinDeviceEventCApiFunctionTable<T>(),
              value.ReleaseRCRef().release()}) {}

  PjRtDeviceEventPtr ptr() const { return ptr_; }

  template <typename Waiter>
  void AndThen(Waiter&& cb) const {
    ptr_.AndThen(std::forward<Waiter>(cb));
  }

  template <typename T>
  tsl::AsyncValueRef<T> down_cast() const& {
    return ptr_.down_cast<T>();
  }

  template <typename T>
  tsl::AsyncValueRef<T> down_cast() && {
    return std::move(ptr_).steal_down_cast<T>();
  }

  std::optional<absl::Status> GetErrorIfPresent() const {
    return ptr_.GetErrorIfPresent();
  }

  // TODO(parkers): Remove direct async_value usages.
  tsl::AsyncValue* async_value() const { return ptr_.async_value(); }

  void reset() {
    ptr_.DecRef();
    ptr_ = {};
  }

  PjRtDeviceEventPtr release() && { return std::exchange(ptr_, {}); }

  static PjRtDeviceEventRef TakeRef(PjRtDeviceEventPtr ptr) {
    PjRtDeviceEventRef ref;
    ref.ptr_ = ptr;
    return ref;
  }

  explicit operator bool() const { return static_cast<bool>(ptr_); }

 private:
  PjRtDeviceEventPtr ptr_;
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

  virtual void AddEvent(PjRtDeviceEventRef event) = 0;

  virtual void AppendTo(
      std::vector<tsl::RCReference<tsl::AsyncValue>>& events) = 0;
  virtual void AppendTo(PjRtDeviceEventSet& events) = 0;

  virtual std::unique_ptr<PjRtDeviceEventSet> Clone() const = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_DEVICE_EVENT_H_
