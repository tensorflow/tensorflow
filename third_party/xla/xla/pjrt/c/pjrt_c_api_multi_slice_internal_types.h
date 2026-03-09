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

#ifndef XLA_PJRT_C_PJRT_C_API_MULTI_SLICE_INTERNAL_TYPES_H_
#define XLA_PJRT_C_PJRT_C_API_MULTI_SLICE_INTERNAL_TYPES_H_

#include <memory>

#include "xla/pjrt/pjrt_executable.h"

typedef struct PJRT_MultiSlice_Config {
  std::unique_ptr<const xla::MultiSliceConfig> config;
} PJRT_MultiSlice_Config;

#endif  // XLA_PJRT_C_PJRT_C_API_MULTI_SLICE_INTERNAL_TYPES_H_
