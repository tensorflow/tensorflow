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

#ifndef XLA_PJRT_C_PJRT_C_API_CALLBACK_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_CALLBACK_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality for registering callbacks.

#define PJRT_API_CALLBACK_EXTENSION_VERSION 1

// ------------------------------ Callback types ------------------------------

enum PJRT_Callback_Type {
  PJRT_Callback_Type_Unknown,
  PJRT_Callback_Type_Tpu_SliceBuilder,
  PJRT_Callback_Type_Prefatal,
};

// A TPU ICI Slice's failure as captured by SliceBuilder. The failure detected
// could be because of software, hardware, or firmware issues. It surfaces to
// SliceBuilder via software polling APIs.
#ifdef __cplusplus
enum class PJRT_Callback_Tpu_SliceFailureType : int32_t {
#else
enum PJRT_Callback_Tpu_SliceFailureType : int32_t {
#endif

  // An undefined slice failure.
  SLICE_FAILURE_UNKNOWN = 0,

  // Slice failure during slice initialization (ICI network config) phase.
  SLICE_FAILURE_INIT_ERROR = 1,

  // Slice failure when a worker is disconnected from the rest of the job
  // after
  // the heartbeat check threshold is reached.
  SLICE_FAILURE_WORKER_UNAVAILABLE = 2,

  // Slice failure due to a flapping task, potentially caused by fast task
  // restart, which leads to mis-configuring a previous built slice in a job.
  // This failure scenario is amenable to job quick_exit.
  SLICE_FAILURE_FLAPPING_TASK_ERROR = 3,

  // Slice failure injected by software, e.g., from the TPU runtime to mimic
  // slice hardware error for various control actions, or for testing purpose.
  SLICE_FAILURE_SW_INJECT_ERROR = 4,

  // Slice failure due to a chip driver reporting some error (could correspond
  // to bad TPU or ICI link going down) as perceived from SliceBuilder
  // software polling after the slice has been successfully built.
  SLICE_FAILURE_CHIP_DRIVER_ERROR = 5
};

struct PJRT_Callback_Tpu_SliceBuilderArgs {
  size_t struct_size;
  PJRT_Callback_Tpu_SliceFailureType failure_type;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Callback_Tpu_SliceBuilderArgs, failure_type);

struct PJRT_Callback_PrefatalArgs {
  size_t struct_size;

  PJRT_Error_Code error_code;

  // The error message is only valid for the duration of the callback.
  const char* error_message;
  size_t error_message_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Callback_PrefatalArgs, error_message_size);

// ---------------------------------- Methods ----------------------------------

// The type of a callback function. The first argument is the callback type
// specific arguments, and the second argument is the user provided argument.
typedef void PJRT_Callback_Function(void* args, void* user_arg);
typedef struct PJRT_Callback_RegisterCallback_Args {
  size_t struct_size;
  PJRT_Client* client;

  // The type of callback to be registered.
  PJRT_Callback_Type type;

  // The callback to be registered.
  PJRT_Callback_Function* callback;

  // The user argument to be passed to the callback.
  void* user_arg;
} PJRT_Callback_RegisterCallback_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Callback_RegisterCallback_Args, user_arg);

// Registers a callback.
typedef PJRT_Error* PJRT_Register_Callback(
    PJRT_Callback_RegisterCallback_Args* args);

typedef struct PJRT_Callback_InvokeCallback_Args {
  size_t struct_size;
  PJRT_Client* client;

  // The type of callback to be registered.
  PJRT_Callback_Type type;

  // The callback type specific arguments. The registered callbacks are invoked
  // synchronously, so `args` needs to be valid for the duration of the
  // `PJRT_Callback_InvokeCallback` call.
  void* args;
} PJRT_Callback_InvokeCallback_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Callback_InvokeCallback_Args, args);

typedef PJRT_Error* PJRT_Callback_InvokeCallback(
    PJRT_Callback_InvokeCallback_Args* args);

// --------------------------- Extension entrypoint ----------------------------

typedef struct PJRT_Callback_Extension {
  PJRT_Extension_Base base;
  PJRT_Register_Callback* register_callback;
  PJRT_Callback_InvokeCallback* invoke_callback;
} PJRT_Callback_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Callback_Extension, invoke_callback);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_CALLBACK_EXTENSION_H_
