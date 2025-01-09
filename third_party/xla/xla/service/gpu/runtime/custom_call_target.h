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

#ifndef XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_TARGET_H_
#define XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_TARGET_H_

#include <cstddef>

#include "xla/service/custom_call_status.h"

namespace xla::gpu {

// Custom call signature as used by API_VERSION_ORIGINAL.
using CustomCallWithOpaqueStreamHandle = void (*)(void* stream, void** buffers,
                                                  const char* opaque,
                                                  size_t opaque_len);

// Custom call signature as used by API_VERSION_STATUS_RETURNING.
using CustomCallWithStatusAndOpaqueStreamHandle =
    void (*)(void* stream, void** buffers, const char* opaque,
             size_t opaque_len, XlaCustomCallStatus* status);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_CUSTOM_CALL_TARGET_H_
