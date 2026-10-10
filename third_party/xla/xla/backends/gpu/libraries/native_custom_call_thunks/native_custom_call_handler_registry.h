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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_REGISTRY_H_
#define XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_REGISTRY_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_scratch_context.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"

namespace xla::gpu {

// A compile-time handler that lowers a GPU custom call directly to a native
// ThunkSequence, bypassing CustomCallThunk (FFI/legacy).
//
// Handlers run in the XLA compiler (not the runtime), analogous to an FFI
// `Instantiate` call. They must be statically linked into the compiler and may
// only be registered from packages on the visibility allowlist (see the
// XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER macro below).
using NativeCustomCallHandler =
    absl::AnyInvocable<absl::StatusOr<ThunkSequence>(
        const HloCustomCallInstruction&, const NativeCustomCallEmitterContext&)
                           const>;

using NativeCustomCallHandlerRef =
    absl::FunctionRef<absl::StatusOr<ThunkSequence>(
        const HloCustomCallInstruction&,
        const NativeCustomCallEmitterContext&)>;

// An optional compile-time handler that declares the scratch buffers a custom
// call needs. It runs during HLO optimization (see
// `CustomCallScratchAssigner`), i.e. before buffer assignment.
//
// Each returned shape describes one scratch buffer: a static array whose
// element type and dimensions determine its byte size, and whose layout memory
// space selects where it lives. The buffer is guaranteed to be aligned to the
// platform's buffer alignment (at least 256 bytes on GPU):
//
//   * `Layout::kDefaultMemorySpace` (0): regular device memory.
//   * `Layout::kCollectiveMemorySpace` (7): collective (symmetric) memory that
//     buffer assignment may share with other collective buffers whose live
//     ranges don't overlap.
//
// A shape without a layout gets the default dense layout in memory space 0.
// Returning an empty vector means the custom call needs no scratch memory.
//
// The custom call's result is rewritten to a tuple with the original result at
// index `{0}` and the i-th scratch buffer at index `{1 + i}`. See
// `ScratchShapeIndex` in native_custom_call_handler_utils.h.
using NativeCustomCallScratchHandler =
    absl::AnyInvocable<absl::StatusOr<std::vector<Shape>>(
        const HloCustomCallInstruction&, const NativeCustomCallScratchContext&)
                           const>;

using NativeCustomCallScratchHandlerRef =
    absl::FunctionRef<absl::StatusOr<std::vector<Shape>>(
        const HloCustomCallInstruction&,
        const NativeCustomCallScratchContext&)>;

// Frontend attribute that `CustomCallScratchAssigner` sets on a custom call
// whose result it extended by scratch buffers. Its value is the number of
// appended scratch buffers; a custom call without the attribute has none.
inline constexpr absl::string_view kNativeCustomCallNumScratchBuffersAttr =
    "xla_gpu_native_custom_call_num_scratch_buffers";

// All handlers registered for one custom-call target.
struct NativeCustomCallHandlerBundle {
  NativeCustomCallHandler emit_thunks;                      // mandatory
  NativeCustomCallScratchHandler request_scratch_buffers =  // optional
      nullptr;
};

// Process-global registry mapping a custom-call target name to its handlers.
//
// Registration happens at static-initialization time via the
// XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER macro.
class NativeCustomCallHandlerRegistry {
 public:
  // Returns the process-global registry instance.
  static NativeCustomCallHandlerRegistry& GetGlobal();

  // Returns the thunk handler registered for `target`, if any.
  std::optional<NativeCustomCallHandlerRef> Lookup(
      absl::string_view target) const;

  // Returns the scratch handler registered for `target`, if any. A target can
  // have a thunk handler but no scratch handler.
  std::optional<NativeCustomCallScratchHandlerRef> LookupScratchHandler(
      absl::string_view target) const;

  // Registers `handler` as the thunk handler for `target`. Returns
  // AlreadyExistsError if handlers are already registered for `target`, or
  // InvalidArgumentError if `handler` is null. Prefer the registration macro
  // over calling this directly.
  absl::Status Register(absl::string_view target,
                        NativeCustomCallHandler handler);

  // Registers all handlers in `bundle` for `target`. Returns AlreadyExistsError
  // if handlers are already registered for `target`, or InvalidArgumentError if
  // `bundle.emit_thunks` is null.
  absl::Status Register(absl::string_view target,
                        NativeCustomCallHandlerBundle bundle);

 private:
  absl::node_hash_map<std::string, NativeCustomCallHandlerBundle> handlers_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_REGISTRY_H_
