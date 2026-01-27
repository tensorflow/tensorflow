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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_SAVER_INSERTER_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_SAVER_INSERTER_H_

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::gpu {

// Records outputs of thunks selected by ThunkFilter.
absl::Status RunDebugSaverInserter(SequentialThunk& root_thunk,
                                   const DebugOptions& debug_options,
                                   const HloModule& hlo_module);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_SAVER_INSERTER_H_
