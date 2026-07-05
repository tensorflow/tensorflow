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

#ifndef XLA_PJRT_C_PJRT_C_API_ABI_VERSION_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_ABI_VERSION_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_ABI_VERSION_EXTENSION_VERSION 1

typedef struct PJRT_RuntimeAbiVersion PJRT_RuntimeAbiVersion;
typedef struct PJRT_ExecutableAbiVersion PJRT_ExecutableAbiVersion;
typedef struct PJRT_SerializedProto PJRT_SerializedProto;

struct PJRT_Client_RuntimeAbiVersion_Args {
  size_t struct_size;
  PJRT_Client* client;
  PJRT_RuntimeAbiVersion* abi_version;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_RuntimeAbiVersion_Args, abi_version);

typedef PJRT_Error* PJRT_Client_RuntimeAbiVersion(
    PJRT_Client_RuntimeAbiVersion_Args* args);

struct PJRT_Executable_GetAbiVersion_Args {
  size_t struct_size;
  PJRT_Executable* executable;
  PJRT_ExecutableAbiVersion* abi_version;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_GetAbiVersion_Args, abi_version);

typedef PJRT_Error* PJRT_Executable_GetAbiVersion(
    PJRT_Executable_GetAbiVersion_Args* args);

struct PJRT_RuntimeAbiVersion_Destroy_Args {
  size_t struct_size;
  PJRT_RuntimeAbiVersion* abi_version;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RuntimeAbiVersion_Destroy_Args, abi_version);

typedef PJRT_Error* PJRT_RuntimeAbiVersion_Destroy(
    PJRT_RuntimeAbiVersion_Destroy_Args* args);

struct PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args {
  size_t struct_size;
  const PJRT_RuntimeAbiVersion* abi_version;
  const PJRT_RuntimeAbiVersion* other_abi_version;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args,
                          other_abi_version);

typedef PJRT_Error* PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime(
    PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime_Args* args);

struct PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args {
  size_t struct_size;
  const PJRT_RuntimeAbiVersion* abi_version;
  const PJRT_ExecutableAbiVersion* executable_abi_version;
};
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args,
    executable_abi_version);

typedef PJRT_Error* PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable(
    PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable_Args* args);

struct PJRT_RuntimeAbiVersion_ToProto_Args {
  size_t struct_size;
  const PJRT_RuntimeAbiVersion* abi_version;
  const char* serialized_proto;                                    // out
  size_t serialized_proto_size;                                    // out
  PJRT_SerializedProto* serialized_proto_holder;                   // out
  void (*serialized_proto_deleter)(PJRT_SerializedProto* holder);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RuntimeAbiVersion_ToProto_Args,
                          serialized_proto_deleter);

typedef PJRT_Error* PJRT_RuntimeAbiVersion_ToProto(
    PJRT_RuntimeAbiVersion_ToProto_Args* args);

struct PJRT_RuntimeAbiVersion_PlatformId_Args {
  size_t struct_size;
  const PJRT_RuntimeAbiVersion* abi_version;
  uint64_t platform_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RuntimeAbiVersion_PlatformId_Args, platform_id);

typedef PJRT_Error* PJRT_RuntimeAbiVersion_PlatformId(
    PJRT_RuntimeAbiVersion_PlatformId_Args* args);

struct PJRT_ExecutableAbiVersion_Destroy_Args {
  size_t struct_size;
  PJRT_ExecutableAbiVersion* abi_version;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecutableAbiVersion_Destroy_Args, abi_version);

typedef PJRT_Error* PJRT_ExecutableAbiVersion_Destroy(
    PJRT_ExecutableAbiVersion_Destroy_Args* args);

struct PJRT_ExecutableAbiVersion_ToProto_Args {
  size_t struct_size;
  const PJRT_ExecutableAbiVersion* abi_version;
  const char* serialized_proto;                                    // out
  size_t serialized_proto_size;                                    // out
  PJRT_SerializedProto* serialized_proto_holder;                   // out
  void (*serialized_proto_deleter)(PJRT_SerializedProto* holder);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecutableAbiVersion_ToProto_Args,
                          serialized_proto_deleter);

typedef PJRT_Error* PJRT_ExecutableAbiVersion_ToProto(
    PJRT_ExecutableAbiVersion_ToProto_Args* args);

struct PJRT_ExecutableAbiVersion_PlatformId_Args {
  size_t struct_size;
  const PJRT_ExecutableAbiVersion* abi_version;
  uint64_t platform_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecutableAbiVersion_PlatformId_Args,
                          platform_id);

typedef PJRT_Error* PJRT_ExecutableAbiVersion_PlatformId(
    PJRT_ExecutableAbiVersion_PlatformId_Args* args);

struct PJRT_RuntimeAbiVersion_FromProto_Args {
  size_t struct_size;
  const char* serialized_proto;
  size_t serialized_proto_size;
  PJRT_RuntimeAbiVersion* abi_version;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RuntimeAbiVersion_FromProto_Args, abi_version);

typedef PJRT_Error* PJRT_RuntimeAbiVersion_FromProto(
    PJRT_RuntimeAbiVersion_FromProto_Args* args);

struct PJRT_ExecutableAbiVersion_FromProto_Args {
  size_t struct_size;
  const char* serialized_proto;
  size_t serialized_proto_size;
  PJRT_ExecutableAbiVersion* abi_version;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecutableAbiVersion_FromProto_Args,
                          abi_version);

typedef PJRT_Error* PJRT_ExecutableAbiVersion_FromProto(
    PJRT_ExecutableAbiVersion_FromProto_Args* args);

typedef struct PJRT_AbiVersion_Extension {
  PJRT_Extension_Base base;
  PJRT_Client_RuntimeAbiVersion* client_runtime_abi_version;
  PJRT_Executable_GetAbiVersion* executable_get_abi_version;
  PJRT_RuntimeAbiVersion_Destroy* runtime_abi_version_destroy;
  PJRT_RuntimeAbiVersion_IsCompatibleWithRuntime*
      runtime_abi_version_is_compatible_with_runtime;
  PJRT_RuntimeAbiVersion_IsCompatibleWithExecutable*
      runtime_abi_version_is_compatible_with_executable;
  PJRT_RuntimeAbiVersion_ToProto* runtime_abi_version_to_proto;
  PJRT_RuntimeAbiVersion_PlatformId* runtime_abi_version_platform_id;
  PJRT_ExecutableAbiVersion_Destroy* executable_abi_version_destroy;
  PJRT_ExecutableAbiVersion_ToProto* executable_abi_version_to_proto;
  PJRT_ExecutableAbiVersion_PlatformId* executable_abi_version_platform_id;
  PJRT_RuntimeAbiVersion_FromProto* runtime_abi_version_from_proto;
  PJRT_ExecutableAbiVersion_FromProto* executable_abi_version_from_proto;
} PJRT_AbiVersion_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_AbiVersion_Extension,
                          executable_abi_version_from_proto);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_ABI_VERSION_EXTENSION_H_
