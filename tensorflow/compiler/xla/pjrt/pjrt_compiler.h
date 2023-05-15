/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_COMPILER_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_device_description.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/tsl/platform/fingerprint.h"

namespace xla {

using PjRtPlatformId = uint64_t;

inline const char* CpuName() {
  static constexpr char kCpuName[] = "cpu";
  return kCpuName;
}
inline const char* GpuName() {
  static constexpr char kGpuName[] = "gpu";
  return kGpuName;
}
inline const char* TpuName() {
  static constexpr char kTpuName[] = "tpu";
  return kTpuName;
}
inline PjRtPlatformId CpuId() {
  static const PjRtPlatformId kCpuId = tsl::Fingerprint64(CpuName());
  return kCpuId;
}
inline PjRtPlatformId GpuId() {
  static const PjRtPlatformId kGpuId = tsl::Fingerprint64(GpuName());
  return kGpuId;
}
inline PjRtPlatformId TpuId() {
  static const PjRtPlatformId kTpuId = tsl::Fingerprint64(TpuName());
  return kTpuId;
}

class PjRtCompiler;
class PjRtClient;

// TODO(b/240299401): Move CompileOptions to this file.

// Abstract interface to represent device topology that is used by the compiler.
class PjRtTopologyDescription {
 public:
  virtual ~PjRtTopologyDescription() = default;

  // Return an ID that identifies the platform (CPU/GPU/TPU).
  virtual PjRtPlatformId platform_id() const = 0;

  // Returns a string that identifies the platform (CPU/GPU/TPU).
  virtual absl::string_view platform_name() const = 0;

  // Returns a string containing human-readable, platform-specific version info
  // (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
  virtual absl::string_view platform_version() const = 0;

  // If non-null, overrides the compiler for this topology.
  virtual std::optional<PjRtCompiler*> compiler() const { return std::nullopt; }

  // Returns an unordered list of descriptions for all devices in this topology.
  virtual std::vector<std::unique_ptr<const PjRtDeviceDescription>>
  DeviceDescriptions() const = 0;
};

// Abstract interface that all registered compilers must implement.
class PjRtCompiler {
 public:
  virtual ~PjRtCompiler() {}

  // Compiles the 'computation' and returns a 'PjRtExecutable'. The returned
  // PjRtExecutable must be loaded by a compatible client before execution.
  virtual StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) = 0;

  // Variant of `Compile` that accepts an MLIR module.
  virtual StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) = 0;
};

// Registers a compiler to compile programs for 'platform_name'.
// Takes ownership of 'compiler'.
//
// REQUIRES: No compiler has been registered for the platform yet.
void PjRtRegisterCompiler(absl::string_view platform_name,
                          std::unique_ptr<PjRtCompiler> compiler);

// Compiles a 'computation' and generates a 'PjRtExecutable' using the compiler
// registered for the platform using PjRtRegisterCompiler. The returned
// PjRtExecutable must be loaded by a compatible client before execution.
//
// The actual compiler used may be overridden by Topology::compiler().
//
// Returns error::NotFound if a compiler has not been registered for the
// platform. Forwards errors returned from the registered compiler in case of a
// compilation failure.
StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client = nullptr);

// Variant of `PjRtCompile` that accepts an MLIR module.
StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client = nullptr);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_COMPILER_H_
