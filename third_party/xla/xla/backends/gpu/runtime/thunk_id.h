/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_ID_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_ID_H_

#include <cstdint>

#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// Unique identifier for a thunk. When creating a thunk graph it's up to the
// emitter to generate unique IDs.
TSL_LIB_GTL_DEFINE_INT_TYPE(ThunkId, uint64_t);

// Generates unique IDs for thunks. This class is thread-compatible.
class ThunkIdGenerator {
 public:
  ThunkId GetNextThunkId() { return ThunkId(next_thunk_id_++); }

 private:
  uint64_t next_thunk_id_ = 1;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_ID_H_
