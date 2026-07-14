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
#include <optional>

#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// Unique identifier for async events. The same identifier is expected to be
// shared between a pair of host async start and corresponding done thunks.
TSL_LIB_GTL_DEFINE_INT_TYPE(AsyncEventsUniqueId, uint64_t);

// Base class for host async thunks that provides async event tracking APIs.
// Host send/recv and host execute thunks inherit from this class.
class HostAsyncThunk : public Thunk {
 public:
  using Thunk::Thunk;

  virtual std::optional<AsyncEventsUniqueId> GetAsyncEventsUniqueId() const {
    return std::nullopt;
  }

  virtual bool IsAsyncStart() const { return false; }
  virtual bool IsAsyncDone() const { return false; }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_HOST_ASYNC_THUNK_H_
