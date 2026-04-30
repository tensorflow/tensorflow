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

}  // namespace xla
