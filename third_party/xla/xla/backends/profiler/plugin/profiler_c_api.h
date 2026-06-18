/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_PLUGIN_PROFILER_C_API_H_
#define XLA_BACKENDS_PROFILER_PLUGIN_PROFILER_C_API_H_

#include <stddef.h>
#include <stdint.h>

#define PROFILER_STRUCT_SIZE(struct_type, last_field) \
  offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field)

#define PROFILER_DEFINE_STRUCT_TRAITS(sname, last_field) \
  typedef struct sname sname;                            \
  enum { sname##_STRUCT_SIZE = PROFILER_STRUCT_SIZE(sname, last_field) }

#ifdef __cplusplus
extern "C" {
#endif

#define PLUGIN_PROFILER_VERSION 1

typedef struct PLUGIN_Profiler PLUGIN_Profiler;
typedef struct PLUGIN_Profiler_Error PLUGIN_Profiler_Error;

struct PLUGIN_Profiler_Error_Destroy_Args {
  size_t struct_size;
  void* priv;
  PLUGIN_Profiler_Error* error;
};
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_Error_Destroy_Args, error);

// Frees `error`. `error` can be nullptr.
typedef void PLUGIN_Profiler_Error_Destroy(
    PLUGIN_Profiler_Error_Destroy_Args* args);

struct PLUGIN_Profiler_Error_Message_Args {
  size_t struct_size;
  void* priv;
  const PLUGIN_Profiler_Error* error;
  // Has the lifetime of `error`.
  const char* message;  // out
  size_t message_size;  // out
};
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_Error_Message_Args, message_size);

// Gets the human-readable reason for `error`. `message` has the lifetime of
// `error`.
typedef void PLUGIN_Profiler_Error_Message(
    PLUGIN_Profiler_Error_Message_Args* args);

struct PLUGIN_Profiler_Error_GetCode_Args {
  size_t struct_size;
  void* priv;
  const PLUGIN_Profiler_Error* error;
  int code;  // out
};
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_Error_GetCode_Args, code);

typedef PLUGIN_Profiler_Error* PLUGIN_Profiler_Error_GetCode(
    PLUGIN_Profiler_Error_GetCode_Args* args);

struct PLUGIN_Profiler_Create_Args {
  size_t struct_size;
  const char* options;
  size_t options_size;
  PLUGIN_Profiler* profiler;  // out
};
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_Create_Args, profiler);

typedef PLUGIN_Profiler_Error* PLUGIN_Profiler_Create(
    PLUGIN_Profiler_Create_Args* args);

struct PLUGIN_Profiler_Destroy_Args {
  size_t struct_size;
  PLUGIN_Profiler* profiler;
};
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_Destroy_Args, profiler);

typedef PLUGIN_Profiler_Error* PLUGIN_Profiler_Destroy(
    PLUGIN_Profiler_Destroy_Args* args);

struct PLUGIN_Profiler_Start_Args {
  size_t struct_size;
  PLUGIN_Profiler* profiler;
};
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_Start_Args, profiler);

typedef PLUGIN_Profiler_Error* PLUGIN_Profiler_Start(
    PLUGIN_Profiler_Start_Args* args);

struct PLUGIN_Profiler_Stop_Args {
  size_t struct_size;
  PLUGIN_Profiler* profiler;
};
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_Stop_Args, profiler);

typedef PLUGIN_Profiler_Error* PLUGIN_Profiler_Stop(
    PLUGIN_Profiler_Stop_Args* args);

struct PLUGIN_Profiler_CollectData_Args {
  size_t struct_size;
  PLUGIN_Profiler* profiler;
  uint8_t* buffer;              // in/out
  size_t buffer_size_in_bytes;  // out
};
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_CollectData_Args,
                              buffer_size_in_bytes);

// Callers should generally call this function twice with the same `args`.
// In the first call, `args->buffer` must be nullptr. This call will populate
// `args->buffer_size_in_bytes`. Clients should then allocate a buffer `buffer`
// of at least `buffer_size_in_bytes` bytes. Before the second call, callers
// should set `args->buffer = buffer`. The second call will then write the
// serialized data to `buffer`.
typedef PLUGIN_Profiler_Error* PLUGIN_Profiler_CollectData(
    PLUGIN_Profiler_CollectData_Args* args);

typedef struct PLUGIN_Profiler_Api {
  size_t struct_size;
  void* priv;
  PLUGIN_Profiler_Error_Destroy* error_destroy;
  PLUGIN_Profiler_Error_Message* error_message;
  PLUGIN_Profiler_Error_GetCode* error_get_code;
  PLUGIN_Profiler_Create* create;
  PLUGIN_Profiler_Destroy* destroy;
  PLUGIN_Profiler_Start* start;
  PLUGIN_Profiler_Stop* stop;
  PLUGIN_Profiler_CollectData* collect_data;
} PLUGIN_Profiler_Api;
PROFILER_DEFINE_STRUCT_TRAITS(PLUGIN_Profiler_Api, collect_data);

#ifdef __cplusplus
}
#endif

#endif  // XLA_BACKENDS_PROFILER_PLUGIN_PROFILER_C_API_H_
