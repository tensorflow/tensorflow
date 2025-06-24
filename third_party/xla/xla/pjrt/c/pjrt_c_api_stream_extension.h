/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_PJRT_C_PJRT_C_API_STREAM_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_STREAM_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_STREAM_EXTENSION_VERSION 0

struct PJRT_Get_Stream_For_External_Ready_Events_Args {
  size_t struct_size;
  PJRT_Device* device;
  intptr_t stream;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Get_Stream_For_External_Ready_Events_Args,
                          stream);

// Returns a platform-specific stream handle that should be used to track when
// an externally-managed buffer is ready to use on this device.
typedef PJRT_Error* PJRT_Get_Stream_For_External_Ready_Events(
    PJRT_Get_Stream_For_External_Ready_Events_Args* args);

struct PJRT_Wait_Until_Buffer_Ready_On_Stream_Args {
  size_t struct_size;
  intptr_t stream;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Wait_Until_Buffer_Ready_On_Stream_Args, buffer);

// Waits until buffer is ready on stream.
typedef PJRT_Error* PJRT_Wait_Until_Buffer_Ready_On_Stream(
    PJRT_Wait_Until_Buffer_Ready_On_Stream_Args* args);

typedef struct PJRT_Stream_Extension {
  PJRT_Extension_Base base;
  PJRT_Get_Stream_For_External_Ready_Events* get_stream;
  PJRT_Wait_Until_Buffer_Ready_On_Stream* wait_stream;
} PJRT_Stream_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Stream_Extension, wait_stream);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_STREAM_EXTENSION_H_
