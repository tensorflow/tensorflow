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

#ifndef XLA_DEBUG_OPTIONS_RUNTIME_FIELDS_H_
#define XLA_DEBUG_OPTIONS_RUNTIME_FIELDS_H_

#include "absl/types/span.h"

namespace xla {

// Returns the field numbers of the runtime `DebugOptions` flags, sorted. Used
// by tools such as AOT golden verification.
//
// A runtime flag is read after compilation (i.e. when loading or running the
// executable) and can change execution: results, which kernels or thunks run,
// or whether loading or running fails. A flag whose value is copied into the
// HLO, a backend config or a thunk while compiling isn't a runtime flag, since
// the executable carries the copy.
//
// Flags that only affect diagnostics (dump formats and filters, NVTX
// annotations, profiler setup) are left out even when read after compilation.
// The exception is `xla_dump_to`, because whether it's empty changes
// execution: with the NaN/Inf checker in DUMP mode an empty value makes loading
// fail, and with the buffer saver enabled it skips the saver thunks. Tools
// should compare `empty()` rather than the path.
//
// If you're adding a new runtime flag, add it to `kRuntimeFields` in
// debug_options_runtime_fields.cc, since nothing enforces this list. If unsure,
// add it: a missing flag makes tools silently ignore a real difference, an
// extra one only makes them report a harmless one.
absl::Span<const int> RuntimeDebugOptionsFieldNumbers();

// Returns whether `field_number` is in `RuntimeDebugOptionsFieldNumbers()`.
bool IsRuntimeDebugOptionsField(int field_number);

}  // namespace xla

#endif  // XLA_DEBUG_OPTIONS_RUNTIME_FIELDS_H_
