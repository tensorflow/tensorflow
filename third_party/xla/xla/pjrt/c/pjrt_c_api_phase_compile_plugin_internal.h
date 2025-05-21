/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_
#define XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_

#include <string>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {
namespace phase_compile_cpu_plugin {

// This file provides internal utilities and the entry point for a sample
// CPU plugin that demonstrates PJRT's phased compilation capabilities.
// It includes a serialization helper for StableHLO and declares the function
// responsible for providing the PJRT API for this specific plugin.

// Helper class for serializing and deserializing StableHLO MLIR modules.
// This is crucial for converting `PjRtPartialProgramProto` bytes to
// actual MLIR modules and vice-versa, allowing programs to be transferred
// between compilation phases.
class StableHLOTypeSerialization {
 public:
  // Deserializes a StableHLO program from a string into an MLIR ModuleOp.
  // Returns an error if deserialization fails (e.g., invalid artifact).
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> deserialize(
      const std::string& program, mlir::MLIRContext& context);

  // Serializes an MLIR ModuleOp into a StableHLO bytecode string.
  // Returns an error if serialization fails.
  static absl::StatusOr<std::string> serialize(
      const mlir::OwningOpRef<mlir::ModuleOp>& module_op);

 private:
  StableHLOTypeSerialization() = delete;
};

// Returns the PJRT_Api struct specific to the phase compile plugin for CPU.
const PJRT_Api* GetPhaseCompileForCpuPjrtApi();

// Returns the PJRT_Api struct with an invalid extension start. For testing
// purposes only.
const PJRT_Api* GetPjrtApiWithNullPhaseExtension();

// Returns the PJRT_Api struct with an invalid platform. For testing purposes
// only.
const PJRT_Api* GetPjrtApiWithInvalidPlatform();

}  // namespace phase_compile_cpu_plugin
}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_PLUGIN_INTERNAL_H_
