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

#include "absl/container/node_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instructions.h"

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

// Process-global registry mapping a custom-call target name to its handler.
//
// Registration happens at static-initialization time via the
// XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER macro.
class NativeCustomCallHandlerRegistry {
 public:
  // Returns the process-global registry instance.
  static NativeCustomCallHandlerRegistry& GetGlobal();

  std::optional<NativeCustomCallHandlerRef> Lookup(
      absl::string_view target) const;

  // Registers `handler` for `target`. Returns AlreadyExistsError if a handler
  // is already registered for `target`, or InvalidArgumentError if `handler` is
  // null. Prefer the registration macro over calling this directly.
  absl::Status Register(absl::string_view target,
                        NativeCustomCallHandler handler);

 private:
  absl::node_hash_map<std::string, NativeCustomCallHandler> handlers_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_REGISTRY_H_
