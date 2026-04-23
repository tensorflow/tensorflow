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

#ifndef XLA_PJRT_C_PJRT_C_API_DEVICE_EVENT_H_
#define XLA_PJRT_C_PJRT_C_API_DEVICE_EVENT_H_

#include <stddef.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Signature for user-provided AndThen callbacks.
typedef void (*PJRT_DeviceEvent_AndThen)(void* user_arg);

struct PJRT_DeviceEvent_FunctionTable {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  // tsl::AsyncValue::AddRef();
  void (*inc_ref)(void* device_event);
  // tsl::AsyncValue::DropRef();
  void (*dec_ref)(void* device_event);
  // tsl::AsyncValue::AndThen() (callback will be called exactly once);
  void (*and_then)(void* device_event, PJRT_DeviceEvent_AndThen callback,
                   void* user_arg);
  // The returned string only lives as long as the device_event.
  int (*get_error_if_present)(void* device_event, PJRT_Error_Code* code,
                              const char** message, size_t* message_size);
};

// A PJRT_DeviceEvent is a pair of pointers containing both type information
// and the actual opaque device event object. See: xla::PjRtDeviceEventRef.
struct PJRT_DeviceEvent {
  struct PJRT_DeviceEvent_FunctionTable* vtable;
  void* device_event;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_DEVICE_EVENT_H_
