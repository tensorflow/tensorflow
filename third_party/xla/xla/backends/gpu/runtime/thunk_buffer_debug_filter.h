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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_FILTER_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_FILTER_H_

#include "absl/functional/any_invocable.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// A boolean-like value returned from thunk filters to indicate whether the
// thunk should be instrumented or left as is.
enum class InstrumentAction : bool {
  // Don't instrument the thunk, leave it as is.
  kSkip,
  // Instrument the thunk.
  kInstrument,
};

using ThunkFilter = absl::AnyInvocable<InstrumentAction(const Thunk&) const>;

ThunkFilter CreateThunkFilter(const DebugOptions& debug_options);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_FILTER_H_
