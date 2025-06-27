/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_COMPILER_H_
#define XLA_PJRT_PJRT_COMPILER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

using PjRtPlatformId = uint64_t;

inline const char* CpuName() {
  static constexpr char kCpuName[] = "cpu";
  return kCpuName;
}
inline const char* CudaName() {
  static constexpr char kCudaName[] = "cuda";
  return kCudaName;
}
inline const char* RocmName() {
  static constexpr char kRocmName[] = "rocm";
  return kRocmName;
}
inline const char* SyclName() {
  static constexpr char kSyclName[] = "sycl";
  return kSyclName;
}
inline const char* TpuName() {
  static constexpr char kTpuName[] = "tpu";
  return kTpuName;
}
inline PjRtPlatformId CpuId() {
  static const PjRtPlatformId kCpuId = tsl::Fingerprint64(CpuName());
  return kCpuId;
}
inline PjRtPlatformId CudaId() {
  static const PjRtPlatformId kCudaId = tsl::Fingerprint64(CudaName());
  return kCudaId;
}
inline PjRtPlatformId RocmId() {
  static const PjRtPlatformId kRocmId = tsl::Fingerprint64(RocmName());
  return kRocmId;
}
inline PjRtPlatformId SyclId() {
  static const PjRtPlatformId kSyclId = tsl::Fingerprint64(SyclName());
  return kSyclId;
}
inline PjRtPlatformId TpuId() {
  static const PjRtPlatformId kTpuId = tsl::Fingerprint64(TpuName());
  return kTpuId;
}

class PjRtCompiler;
class PjRtClient;

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

  // Returns true if the topology represents subslice.
  virtual bool is_subslice_topology() const { return false; }

  // Returns the number of processes (usually the number of hosts, except in
  // topologies with multiple processes per host).
  virtual absl::StatusOr<int> ProcessCount() const {
    return absl::UnimplementedError("ProcessCount is unsupported.");
  }

  // Returns the total number of cores of the default type.
  virtual absl::StatusOr<int> CoreCountOfDefaultType() const {
    return absl::UnimplementedError("CoreCountOfDefaultType is unsupported.");
  }

  // Returns the total number of logical devices of the default type.
  virtual absl::StatusOr<int> LogicalDeviceCountOfDefaultType() const {
    return absl::UnimplementedError(
        "LogicalDeviceCountOfDefaultType is unsupported.");
  }

  // Returns the number of cores of the default type per process.
  virtual absl::StatusOr<int> CoreCountOfDefaultTypePerProcess() const {
    return absl::UnimplementedError(
        "CoreCountOfDefaultTypePerProcess is unsupported.");
  }

  // Returns the number of cores per chip for the default type.
  virtual absl::StatusOr<int> CoreCountOfDefaultTypePerChip() const {
    return absl::UnimplementedError(
        "CoreCountOfDefaultTypePerChip is unsupported.");
  }

  // Serializes the topology for use in cache keys. (No guarantees on
  // stability).
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  // Returns vendor specific attributes about the topology.
  virtual const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
  Attributes() const = 0;

  // Returns the default device layout for a buffer with `element_type` and
  // `dims`. The default layout is a platform-specific layout used when no other
  // layout is specified, e.g. for host-to-device transfers. When compiling, the
  // default layout is used for program arguments and outputs unless
  // user-specified or compiler-chosen layouts are requested via the
  // "mhlo.layout_mode" attribute.
  virtual absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) const = 0;
};

// Abstract interface that all registered compilers must implement.
class PjRtCompiler {
 public:
  virtual ~PjRtCompiler() = default;

  // Compiles the 'computation' and returns a 'PjRtExecutable'. The returned
  // PjRtExecutable must be loaded by a compatible client before execution.
  virtual absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) = 0;

  // Variant of `Compile` that accepts an MLIR module.
  virtual absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
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
absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, const XlaComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client = nullptr);

// Variant of `PjRtCompile` that accepts an MLIR module.
absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client = nullptr);

// Stores a compilation phase's compiler and validator functions.
// This struct bundles the essential functional components required to define
// a single phase within a multi-phase compilation pipeline. It is used by
// PJRT plugins to register their custom compilation stages with the
// `PjRtPhaseCompiler`.
struct CompilationPhaseFunctions {
  // `compiler`: A function that performs the core logic of a compilation phase.
  // It accepts the global compilation `options`, a vector of
  // `PjRtPartialProgramProto` representing input programs, and a
  // `PjRtTopologyDescription` describing the target hardware. It transforms the
  // input programs based on the phase's logic and returns a vector of
  // `PjRtPartialProgramProto` or an error status if compilation fails.
  std::function<absl::StatusOr<std::vector<PjRtPartialProgramProto>>(
      CompileOptions, const std::vector<PjRtPartialProgramProto>&,
      const PjRtTopologyDescription&)>
      compiler;

  // `validator`: A function that performs plugin-specific validation of the
  // input programs for a given compilation phase. It takes a `std::vector` of
  // `PjRtPartialProgramProto` as input and returns `absl::OkStatus()` if
  // validation is successful; otherwise, it returns an `absl::Status`
  // indicating the reason for failure (e.g., incompatible `program_format`).
  std::function<absl::Status(const std::vector<PjRtPartialProgramProto>&)>
      validator;
};

// PjRtPhaseCompiler is a specialized PjRtCompiler that supports multi-stage,
// "phased" compilation. It allows plugins to register individual compilation
// phases (PhaseCompiler and PhaseValidator functions) and manages their
// execution.
class PjRtPhaseCompiler : public PjRtCompiler {
 public:
  ~PjRtPhaseCompiler() override = default;

  // Returns a vector of strings containing the names of all registered phases
  // in the order they were registered.
  absl::StatusOr<std::vector<std::string>> GetPhaseNames();

  // Compiles a set of input programs by running them through a specified
  // sequence of compilation phases. This function internally calls
  // `ValidatePhase` to trigger plugin-specific validation of input
  // compatibility before invoking the appropriate phase compilers. The output
  // of one phase is passed as input to the next. Returns the vector of
  // `PjRtPartialProgramProto` objects resulting from the last executed phase,
  // or an error status if any validation or compilation step fails.
  absl::StatusOr<std::vector<PjRtPartialProgramProto>> RunPhases(
      CompileOptions options,
      const std::vector<PjRtPartialProgramProto>& input_programs,
      const PjRtTopologyDescription& topology,
      const std::vector<std::string>& phases_to_run);

  // Registers all compilation phases supported by this `PjRtPhaseCompiler`
  // instance. Derived classes must override this method to register their
  // supported compilation phases. This method is expected to be called during
  // the initialization of the `PjRtPhaseCompiler` instance.
  //
  // Implementations of this method typically consist of a series of calls
  // to the `RegisterPhase` method, where each call registers a specific
  // compilation stage with its corresponding `compiler` and `validator`
  // functions.
  virtual absl::Status RegisterAllPhases() = 0;

 protected:
  // Registers a new compilation phase with its corresponding compiler and
  // validator functions encapsulated in a CompilationPhaseFunctions struct.
  // The `phase_name` must be unique. The `compiler` and `validator` functions
  // within the struct must not be null. The order of registered phase names is
  // maintained.
  absl::Status RegisterPhase(const std::string& phase_name,
                             CompilationPhaseFunctions phase_functions);

 private:
  // Maps phase names to their corresponding compiler and validator functions.
  absl::flat_hash_map<std::string, CompilationPhaseFunctions> phase_map_;

  // The names of all registered phases in the order they were registered.
  std::vector<std::string> phase_names_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMPILER_H_
