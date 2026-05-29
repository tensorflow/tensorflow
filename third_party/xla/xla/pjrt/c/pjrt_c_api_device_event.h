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

typedef enum {
  PJRT_DeviceEvent_State_Unavailable = 0,
  PJRT_DeviceEvent_State_Ready = 1,
  PJRT_DeviceEvent_State_Error = 2,
} PJRT_DeviceEvent_State;

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
  // If not null, the event can be aliased as this event type.
  // For a single plugin, events should have the same parent type
  // which allows using them as that type.
  const struct PJRT_DeviceEvent_FunctionTable* parent;
  // Gets the current state of the event. Underlying events may have
  // additional states but they should be mapped to
  // unavailable, ready, or error.
  PJRT_DeviceEvent_State (*get_state)(void* device_event);
  // Opaque platform-specific stream-id. Can be 0 if not supported or unknown.
  intptr_t (*get_definition_stream)(void* device_event,
                                    uint64_t* sequence_number);
};

PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceEvent_FunctionTable,
                          get_definition_stream);

// A PJRT_DeviceEvent is a pair of pointers containing both type information
// and the actual opaque device event object. See: xla::PjRtDeviceEventRef.
struct PJRT_DeviceEvent {
  const struct PJRT_DeviceEvent_FunctionTable* vtable;
  void* device_event;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_DEVICE_EVENT_H_
