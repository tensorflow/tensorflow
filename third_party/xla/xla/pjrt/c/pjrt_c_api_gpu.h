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

#ifndef XLA_PJRT_C_PJRT_C_API_GPU_H_
#define XLA_PJRT_C_PJRT_C_API_GPU_H_

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

// Does not pass ownership of returned PJRT_Api* to caller.
PJRT_CAPI_EXPORT const PJRT_Api* GetPjrtApi();

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_GPU_H_
