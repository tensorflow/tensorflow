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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_REGISTRATION_H_
#define XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_REGISTRATION_H_

#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"

// This header provides the registration side of the custom-call thunk-folding
// API.
namespace xla::gpu {
namespace native_custom_call_internal {

// Helper whose constructor registers handlers. Used by the registration macro.
struct Registrar {
  Registrar(absl::string_view target, NativeCustomCallHandler handler);
  Registrar(absl::string_view target, NativeCustomCallHandlerBundle bundle);
};

}  // namespace native_custom_call_internal
}  // namespace xla::gpu

#define XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER_UNIQ_HELPER(ctr, target, \
                                                                ...)         \
  static ::xla::gpu::native_custom_call_internal::Registrar                  \
  xla_gpu_native_custom_call_registrar_##ctr(target, __VA_ARGS__)

#define XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER_UNIQ(ctr, target, ...) \
  XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER_UNIQ_HELPER(ctr, target,     \
                                                          __VA_ARGS__)

// Registers the handlers for a custom-call target name. Place at namespace
// scope in a `.cc` file. Either pass just the thunk handler:
//
//   XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER("my.custom.call", MyHandler);
//
// or a NativeCustomCallHandlerBundle to also register a scratch handler:
//
//   XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(
//       "my.custom.call", {.emit_thunks = MyHandler,
//                          .request_scratch_buffers = MyScratchHandler});
//
// The registering target must be on the visibility allowlist of the
// native_custom_call_thunks package.
#define XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(target, ...)        \
  XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER_UNIQ(__COUNTER__, target, \
                                                   __VA_ARGS__)

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_NATIVE_CUSTOM_CALL_HANDLER_REGISTRATION_H_
