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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_CHECKSUM_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_CHECKSUM_H_

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

absl::Status RunChecksumPassInternal(SequentialThunk* root_thunk,
                                     const DebugOptions& debug_options,
                                     const HloModule* absl_nonnull hlo_module,
                                     ThunkPassBufferAllocator& allocator);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_CHECKSUM_H_
