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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_FFI_ATTRIBUTES_FROM_BACKEND_CONFIG_H_
#define XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_FFI_ATTRIBUTES_FROM_BACKEND_CONFIG_H_

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/ffi/attributes.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla::gpu {

// Decodes the backend config of `instr` into FFI attributes.
//
// This applies the same rules as the FFI lowering path: the backend config
// must be a string that parses into an MLIR dictionary attribute. An empty
// backend config yields an empty set of attributes.
//
// Shared between `ThunkEmitter`, which serves this to native custom-call
// handlers, and the handler test harness, so that both agree on how a backend
// config becomes attributes.
absl::StatusOr<xla::ffi::Attributes> FfiAttributesFromBackendConfig(
    const HloCustomCallInstruction& instr, mlir::MLIRContext& mlir_context);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_NATIVE_CUSTOM_CALL_THUNKS_FFI_ATTRIBUTES_FROM_BACKEND_CONFIG_H_
