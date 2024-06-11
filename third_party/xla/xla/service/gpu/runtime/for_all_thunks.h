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

#ifndef XLA_SERVICE_GPU_RUNTIME_FOR_ALL_THUNKS_H_
#define XLA_SERVICE_GPU_RUNTIME_FOR_ALL_THUNKS_H_

#include "absl/functional/function_ref.h"
#include "xla/service/gpu/runtime/thunk.h"

namespace xla::gpu {

// Recursively invokes `fn` for all `Thunks` in `root`, including those nested
// within other `Thunks` (e.g. the condition `Thunk` within a `WhileThunk`).
void ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn, const Thunk* thunk);

// Same as above but for a `ThunkSequence` root.
void ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn,
                  const ThunkSequence* thunks);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_FOR_ALL_THUNKS_H_
