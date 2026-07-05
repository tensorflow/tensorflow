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

#ifndef XLA_PJRT_C_PJRT_C_API_XLA_TRANSFORM_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_XLA_TRANSFORM_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION 1

typedef struct PJRT_XlaTransform_string {
  const char* data;
  size_t size;
} PJRT_XlaTransform_string;

// Generalized version/error struct for C API extension.
// The 'api_version' member should be set to
// PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION for the current
// implementation.
// The 'data' member pointer should be set to a pointer to the actual extension
// struct by the extension implementation.
// The 'cleanup_fn' member pointer should be set to a function that will clean
// up the extension data by the extension implementation. If no cleanup is
// needed, the 'cleanup_fn' member pointer should be set to NULL. If has_error
// is true, the 'code' and 'error_msg' members should be set to the error code
// and error message, respectively.
typedef struct PJRT_XlaTransform_version_and_error {
  int64_t api_version;
  void* data;                          // out
  void (*cleanup_fn)(void* data);      // out
  bool has_error;                      // out
  PJRT_Error_Code code;                // out
  PJRT_XlaTransform_string error_msg;  // out
} PJRT_XlaTransform_version_and_error;

// Struct representing the arguments and results of the
// transform_hlo_module callback.
// The member 'struct_size' must always be set to the size of the struct by
// the client. The member 'header' must always be set to the version and error
// struct by the client.
// The member 'changed' indicates whether the HloModuleProto has been modified.
typedef struct PJRT_XlaTransform_Args {
  size_t struct_size;
  PJRT_XlaTransform_version_and_error header;

  // In: Serialized HloModuleProto
  PJRT_XlaTransform_string hlo_module;

  // Out: Serialized HloModuleProto
  PJRT_XlaTransform_string transformed_hlo_module;
  bool changed;  // out
} PJRT_XlaTransform_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_XlaTransform_Args, changed);

// List of callbacks that the caller (callback implementation)
// should provide. All callbacks are optional, and may be set to NULL.
typedef struct PJRT_XlaTransform_Callbacks {
  int64_t version;
  void (*dtor)(struct PJRT_XlaTransform_Callbacks* data);
  void (*transform_hlo_module)(struct PJRT_XlaTransform_Callbacks* data,
                               PJRT_XlaTransform_Args* args);
} PJRT_XlaTransform_Callbacks;

// Matching xla_transform.h:
// enum class PipelineStage { kPreScheduler, kPostScheduler };
typedef enum {
  PJRT_XlaTransform_PipelineStage_kPreScheduler = 0,
  PJRT_XlaTransform_PipelineStage_kPostScheduler = 1,
} PJRT_XlaTransform_PipelineStage;

// Struct representing the arguments for the register_xla_transform callback.
// The member 'struct_size' must always be set to the size of the struct by
// the client. The member 'name' is the name of the XlaTransform. The member
// 'stage' is the pipeline stage at which the XlaTransform should be applied.
// The member 'callbacks' is a pointer to the callbacks struct.
struct PJRT_Register_Xla_Transform_Args {
  size_t struct_size;
  const char* name;
  size_t name_size;
  PJRT_XlaTransform_PipelineStage stage;
  PJRT_XlaTransform_Callbacks* callbacks;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Register_Xla_Transform_Args, callbacks);

typedef PJRT_Error* (*PJRT_Register_Xla_Transform_Fn)(
    PJRT_Register_Xla_Transform_Args* args);

// Struct representing the arguments for the clear_xla_transform callback.
// The member 'struct_size' must always be set to the size of the struct by
// the client. The member 'stage' is the pipeline stage of the transform to be
// cleared. The member 'name' is the name of the transform to be cleared.
// The member 'cleared' is an out parameter indicating whether the transform
// was cleared.
struct PJRT_Clear_Xla_Transform_Args {
  size_t struct_size;
  PJRT_XlaTransform_PipelineStage stage;
  const char* name;
  size_t name_size;
  PJRT_XlaTransform_Callbacks* callbacks;
  bool cleared;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Clear_Xla_Transform_Args, cleared);

typedef PJRT_Error* (*PJRT_Clear_Xla_Transform_Fn)(
    PJRT_Clear_Xla_Transform_Args* args);

typedef struct PJRT_Xla_Transform_Extension {
  PJRT_Extension_Base base;
  PJRT_Register_Xla_Transform_Fn register_xla_transform;
  PJRT_Clear_Xla_Transform_Fn clear_xla_transform;
} PJRT_Xla_Transform_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Xla_Transform_Extension, clear_xla_transform);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_XLA_TRANSFORM_EXTENSION_H_
