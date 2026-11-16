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

#ifndef XLA_BACKENDS_GPU_RUNTIME_HOST_ASYNC_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_HOST_ASYNC_THUNK_H_

#include <cstdint>

#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// Unique identifier for async events. The same identifier is expected to be
// shared between a pair of host async start and corresponding done thunks.
TSL_LIB_GTL_DEFINE_INT_TYPE(AsyncEventsUniqueId, uint64_t);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_HOST_ASYNC_THUNK_H_
