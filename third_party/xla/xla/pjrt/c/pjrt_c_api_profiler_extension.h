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

#ifndef XLA_PJRT_C_PJRT_C_API_PROFILER_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_PROFILER_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_PROFILER_EXTENSION_VERSION 1

typedef struct PJRT_Profiler_Extension {
  size_t struct_size;
  PJRT_Extension_Type type;
  PJRT_Extension_Base* next;
  // can be nullptr if PJRT_Profiler_Extension is used as an args extension
  PLUGIN_Profiler_Api* profiler_api;
  // valid only when used as an args extension
  int64_t traceme_context_id;
} PJRT_Profiler_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Profiler_Extension, traceme_context_id);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_PROFILER_EXTENSION_H_
