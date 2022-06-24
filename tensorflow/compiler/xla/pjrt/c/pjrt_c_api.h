/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_H_

#include <stddef.h>

#define PJRT_STRUCT_SIZE(struct_type, last_field) \
  offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field)

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------- Errors -----------------------------------

// PJRT C API methods generally return a PJRT_Error*, which is nullptr if there
// is no error and set if there is. The implementation allocates any returned
// PJRT_Errors, but the caller is always responsible for freeing them via
// PJRT_Error_Destroy.

typedef struct PJRT_Error PJRT_Error;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Error* error;
} PJRT_Error_Destroy_Args;
const size_t PJRT_Error_Destroy_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Error_Destroy_Args, error);

// Frees `error`. `error` can be nullptr.
typedef void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Error* error;
  // Has the lifetime of `error`.
  const char* message;  // out
  size_t message_size;  // out
} PJRT_Error_Message_Args;
const size_t PJRT_Error_Message_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Error_Message_Args, error);

// Gets the human-readable reason for `error`. `message` has the lifetime of
// `error`.
typedef void PJRT_Error_Message(PJRT_Error_Message_Args* args);

// ---------------------------------- Client -----------------------------------

typedef struct PJRT_Client PJRT_Client;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;  // out
} PJRT_Client_Create_Args;
const size_t PJRT_Client_Create_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_Create_Args, client);

// Creates and initializes a new PJRT_Client and returns in `client`.
typedef PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
} PJRT_Client_Destroy_Args;
const size_t PJRT_Client_Destroy_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_Destroy_Args, client);

// Shuts down and frees `client`. `client` can be nullptr.
typedef PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // `platform_name` has the same lifetime as `client`. It is owned by `client`.
  const char* platform_name;  // out
  size_t platform_name_size;  // out
} PJRT_Client_PlatformName_Args;

const size_t PJRT_Client_PlatformName_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_PlatformName_Args, platform_name_size);

// Returns a string that identifies the platform (e.g. "cpu", "gpu", "tpu").
typedef PJRT_Error* PJRT_Client_PlatformName(
    PJRT_Client_PlatformName_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  int process_index;  // out
} PJRT_Client_Process_Index_Args;
const size_t PJRT_Client_Process_Index_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_Process_Index_Args, process_index);

// Return the process index of this client. Always 0 in single-process
// settings.
typedef PJRT_Error* PJRT_Client_Process_Index(
    PJRT_Client_Process_Index_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // `platform_version` has the same lifetime as `client`. It's owned by
  // `client`.
  const char* platform_version;  // out
  size_t platform_version_size;  // out
} PJRT_Client_PlatformVersion_Args;

const size_t PJRT_Client_PlatformVersion_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_PlatformVersion_Args, platform_version_size);

// Returns a string containing human-readable, platform-specific version info
// (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
typedef PJRT_Error* PJRT_Client_PlatformVersion(
    PJRT_Client_PlatformVersion_Args* args);

// --------------------------------- Devices -----------------------------------

typedef struct PJRT_Device PJRT_Device;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  int id;  // out
} PJRT_Device_Id_Args;
const size_t PJRT_Device_Id_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_Id_Args, id);

// The ID of this device. IDs are unique among devices of this type
// (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
// hosts' devices.
typedef PJRT_Error* PJRT_Device_Id(PJRT_Device_Id_Args* args);

// ------------------------------- Executables ---------------------------------

typedef struct PJRT_Executable PJRT_Executable;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  // `executable_name` has the same lifetime as `executable`. It is owned by
  // `executable`.
  const char* executable_name;  // out
  size_t executable_name_size;  // out
} PJRT_Executable_Name_Args;

const size_t PJRT_Executable_Name_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Executable_Name_Args, executable_name_size);

// Returns a string that identifies the executable.
typedef PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args);

// -------------------------------- API access ---------------------------------

#define PJRT_API_STRUCT_FIELD(fn_type) fn_type* fn_type

typedef struct {
  size_t struct_size;
  void* priv;

  PJRT_API_STRUCT_FIELD(PJRT_Error_Destroy);
  PJRT_API_STRUCT_FIELD(PJRT_Error_Message);

  PJRT_API_STRUCT_FIELD(PJRT_Client_Create);
  PJRT_API_STRUCT_FIELD(PJRT_Client_Destroy);
  PJRT_API_STRUCT_FIELD(PJRT_Client_PlatformName);
  PJRT_API_STRUCT_FIELD(PJRT_Client_Process_Index);
  PJRT_API_STRUCT_FIELD(PJRT_Client_PlatformVersion);

  PJRT_API_STRUCT_FIELD(PJRT_Device_Id);

  PJRT_API_STRUCT_FIELD(PJRT_Executable_Name);
} PJRT_Api;

const size_t PJRT_Api_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Executable_Name);

#undef PJRT_API_STRUCT_FIELD

#ifdef __cplusplus
}
#endif

#undef PJRT_API_STRUCT_FIELD

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_H_
