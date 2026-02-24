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

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

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

// A factory that creates a PjRtCompiler. Creation is deferred to avoid
// violations during program initialization (e.g., RPC or file access during
// global init).
using PjRtCompilerFactory =
    std::function<absl::StatusOr<std::unique_ptr<PjRtCompiler>>()>;

// PjRtCompilerRegistry manages the registration and lifecycle of PjRtCompilers.
// It supports both direct registration of compiler instances and registration
// of factories for deferred initialization.
class PjRtCompilerRegistry {
 public:
  PjRtCompilerRegistry() = default;
  ~PjRtCompilerRegistry() = default;

  // Not copyable or movable.
  PjRtCompilerRegistry(const PjRtCompilerRegistry&) = delete;
  PjRtCompilerRegistry& operator=(const PjRtCompilerRegistry&) = delete;

  // Returns the global singleton instance.
  static PjRtCompilerRegistry& Global();

  // Registers a compiler factory for a specific platform and variant.
  absl::Status RegisterFactory(absl::string_view platform_name,
                               absl::string_view variant_name,
                               PjRtCompilerFactory factory);

  // Registers a compiler instance for a specific platform and variant.
  absl::Status RegisterCompiler(absl::string_view platform_name,
                                absl::string_view variant_name,
                                std::unique_ptr<PjRtCompiler> compiler);

  // Returns the registered compiler for the given platform and variant.
  // Initializes the compiler using the factory if necessary.
  absl::StatusOr<PjRtCompiler*> GetCompiler(absl::string_view platform_name,
                                            absl::string_view variant_name);

  // Explicitly initializes a compiler with a given variant.
  absl::Status InitializeVariant(absl::string_view platform_name,
                                 absl::string_view variant_name);

  // Initializes all registered compiler variants.
  absl::Status InitializeAllVariants();

 private:
  absl::StatusOr<PjRtCompiler*> GetOrCreateCompiler(
      absl::string_view platform_name, absl::string_view variant_name)
      ABSL_LOCKS_EXCLUDED(compiler_mutex_, factory_mutex_);

  absl::Mutex compiler_mutex_;

  absl::Mutex factory_mutex_;

  absl::flat_hash_map<std::pair<std::string, std::string>,
                      std::unique_ptr<PjRtCompiler>>
      compilers_ ABSL_GUARDED_BY(compiler_mutex_);

  absl::flat_hash_map<std::pair<std::string, std::string>, PjRtCompilerFactory>
      factories_ ABSL_GUARDED_BY(factory_mutex_);
};

// Thread-safe. Returns a pointer to the registered compiler for the given
// platform and a default compiler variant.
// Initializes the compiler using the factory if necessary.
absl::StatusOr<PjRtCompiler*> GetDefaultPjRtCompiler(
    absl::string_view platform_name);

// Thread-safe. Returns a pointer to the registered compiler for the given
// platform and compiler variant.
// Initializes the compiler using the factory if necessary.
absl::StatusOr<PjRtCompiler*> GetPjRtCompiler(
    absl::string_view platform_name, absl::string_view compiler_variant);

// Registers a compiler factory for a specific platform and variant.
void PjRtRegisterCompilerFactory(absl::string_view platform_name,
                                 absl::string_view variant_name,
                                 PjRtCompilerFactory factory);

// A compiler variant is a string used to distinguish between different
// compiler implementations registered for the same platform, such as a remote
// compiler service vs in-process compilation.
//
// Explicitly initializes a compiler with a given variant. The corresponding
// factory must have been registered.
// If the compiler is already initialized, this is a no-op.
absl::Status PjRtInitializeCompilerVariant(absl::string_view platform_name,
                                           absl::string_view variant_name);

// Initializes all compiler variants.
absl::Status PjRtInitializeCompilerVariants();

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

  // Returns the number of chips per process.
  virtual absl::StatusOr<int> ChipsPerProcess() const {
    return absl::UnimplementedError("ChipsPerProcess is unsupported.");
  }

  // Returns the number of chips.
  virtual absl::StatusOr<int> ChipCount() const {
    TF_ASSIGN_OR_RETURN(int process_count, ProcessCount());
    TF_ASSIGN_OR_RETURN(int chips_per_process, ChipsPerProcess());
    return process_count * chips_per_process;
  }

  // Returns the total number of cores of the default type.
  virtual absl::StatusOr<int> CoreCountOfDefaultType() const {
    TF_ASSIGN_OR_RETURN(int process_count, ProcessCount());
    TF_ASSIGN_OR_RETURN(int cores_per_process,
                        CoreCountOfDefaultTypePerProcess());
    return process_count * cores_per_process;
  }

  // As above, but returns the number of logical devices per host.
  virtual absl::StatusOr<int> LogicalDeviceCountOfDefaultTypePerProcess()
      const {
    TF_ASSIGN_OR_RETURN(int logical_devices_per_chip,
                        LogicalDeviceCountOfDefaultTypePerChip());
    TF_ASSIGN_OR_RETURN(int chips_per_process, ChipsPerProcess());
    return chips_per_process * logical_devices_per_chip;
  }

  // Returns the total number of logical devices of the default type.
  virtual absl::StatusOr<int> LogicalDeviceCountOfDefaultType() const {
    TF_ASSIGN_OR_RETURN(int process_count, ProcessCount());
    TF_ASSIGN_OR_RETURN(int logical_devices_per_process,
                        LogicalDeviceCountOfDefaultTypePerProcess());
    return process_count * logical_devices_per_process;
  }

  // Returns the number of logical devices of the default type per chip.
  virtual absl::StatusOr<int> LogicalDeviceCountOfDefaultTypePerChip() const {
    return absl::UnimplementedError(
        "LogicalDeviceCountOfDefaultTypePerChip is unsupported.");
  }

  // Returns the number of cores of the default type per process.
  virtual absl::StatusOr<int> CoreCountOfDefaultTypePerProcess() const {
    TF_ASSIGN_OR_RETURN(int cores_per_chip, CoreCountOfDefaultTypePerChip());
    TF_ASSIGN_OR_RETURN(int chips_per_process, ChipsPerProcess());
    return cores_per_chip * chips_per_process;
  }

  // Returns the number of cores per chip for the default type.
  virtual absl::StatusOr<int> CoreCountOfDefaultTypePerChip() const {
    return absl::UnimplementedError(
        "CoreCountOfDefaultTypePerChip is unsupported.");
  }

  // Returns the ids for all processes.
  virtual absl::StatusOr<PjRtIdContainer<ProcessId>> ProcessIds() const {
    return absl::UnimplementedError("ProcessIds is unsupported.");
  }

  // Returns the ids for all the logical devices on a specific process.
  virtual absl::StatusOr<PjRtIdContainer<GlobalDeviceId>>
  LogicalDeviceOfDefaultTypeIdsOnProcess(ProcessId process_id) const {
    return absl::UnimplementedError(
        "LogicalDeviceOfDefaultTypeIdsOnProcess is unsupported.");
  }

  // Returns the process ID and the index of the chip within that process for a
  // given chip.
  virtual absl::StatusOr<std::pair<ProcessId, int>>
  ProcessIdAndIndexOnProcessForChip(GlobalChipId chip_id) const {
    return absl::UnimplementedError(
        "ProcessIdAndIndexOnProcessForChip is unsupported.");
  }

  // Returns the process ID and the index on process for a logical device.
  virtual absl::StatusOr<std::pair<ProcessId, int>>
  ProcessIdAndIndexOnProcessForLogicalDeviceOfDefaultType(
      GlobalDeviceId device_id) const {
    return absl::UnimplementedError(
        "ProcessIdAndIndexOnProcessForLogicalDeviceOfDefaultType is "
        "unsupported.");
  }

  // Returns the coordinates of a process given its ID.
  virtual absl::StatusOr<PjRtDeviceDimensions> ProcessCoordFromId(
      ProcessId process_id) const {
    return absl::UnimplementedError("ProcessCoordForId is unsupported.");
  }

  // Returns the chip ID for a given chip coordinate.
  virtual absl::StatusOr<GlobalChipId> ChipIdFromCoord(
      const PjRtDeviceDimensions& chip) const {
    return absl::UnimplementedError("IdForChip is unsupported.");
  }

  // Returns a unique integer ID for the logical device of the default type on
  // the chip at the given coordinates and with the given core index.
  virtual absl::StatusOr<GlobalDeviceId>
  LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex(
      const PjRtDeviceDimensions& chip, int core_index) const {
    return absl::UnimplementedError(
        "LogicalDeviceOfDefaultTypeIdFromChipCoordAndCoreIndex is "
        "unsupported.");
  }

  // Returns the chip coordinates and core index of the logical device of the
  // default type for the given unique device ID.
  virtual absl::StatusOr<std::pair<PjRtDeviceDimensions, int32_t>>
  ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
      GlobalDeviceId device_id) const {
    return absl::UnimplementedError(
        "LogicalDeviceCoordsOfDefaultTypeForId is unsupported.");
  }

  // Returns the bounds of the chips within a single host.
  // The product of all dimensions should equal to ChipsPerProcess().
  virtual absl::StatusOr<PjRtDeviceDimensions> ChipsPerProcessBounds() const {
    return absl::UnimplementedError("GetChipsPerProcessBounds is unsupported.");
  }

  // Returns the total bounds of all chips in the topology.
  // The product of all dimensions should equal to ChipCount().
  virtual absl::StatusOr<PjRtDeviceDimensions> ChipBounds() const {
    return absl::UnimplementedError("ChipBounds is unsupported.");
  }

  // Returns the total bounds of all hosts in the topology.
  // The product of all dimensions should equal to ProcessCount().
  virtual absl::StatusOr<PjRtDeviceDimensions> ProcessBounds() const {
    return absl::UnimplementedError("ProcessBounds is unsupported.");
  }

  // Serializes the topology for use in cache keys. (No guarantees on
  // stability).
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  // Returns vendor specific attributes about the topology.
  // This map should only include static information available at cross-compile
  // time.
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

  virtual absl::StatusOr<PjRtTopologyDescriptionProto> ToProto() const {
    return absl::UnimplementedError("ToProto is unsupported.");
  }

  // Returns a new `PjRtTopologyDescription` representing a subslice of the
  // current topology, defined by `chips_per_host_bounds` and `host_bounds`.
  virtual absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> Subslice(
      const PjRtDeviceDimensions& chips_per_host_bounds,
      const PjRtDeviceDimensions& host_bounds) const {
    return absl::UnimplementedError("Subslice is not supported.");
  }
};

// Returns true if it's TPU id.
inline bool IsTpuId(PjRtPlatformId platform_id) {
  return platform_id == xla::TpuId();
}

// Returns true if it's GPU id.
inline bool IsGpuId(PjRtPlatformId platform_id) {
  return platform_id == xla::CudaId() || platform_id == xla::RocmId();
}

// Returns true if it's CPU id.
inline bool IsCpuId(PjRtPlatformId platform_id) {
  return platform_id == xla::CpuId();
}

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

  virtual absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
  DeserializePjRtTopologyDescription(const std::string& serialized_topology) {
    return absl::UnimplementedError(
        "DeserializePjRtTopologyDescription is not implemented.");
  }

  // Returns the target runtime ABI version that the compiled executables will
  // be compatible with.
  virtual absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>>
  GetTargetRuntimeAbiVersion() {
    return absl::UnimplementedError(
        "GetTargetRuntimeAbiVersion is not implemented.");
  }
};

// Registers a compiler to compile programs for 'platform_name' with
// a default compiler variant. Takes ownership of 'compiler'.
//
// REQUIRES: No default compiler has been registered for the platform.
void PjRtRegisterDefaultCompiler(absl::string_view platform_name,
                                 std::unique_ptr<PjRtCompiler> compiler);

// Registers a compiler to compile programs for 'platform_name' with
// 'compiler_variant'. Takes ownership of 'compiler'.
//
// REQUIRES: No compiler has been registered for the platform and compiler
// variant yet.
void PjRtRegisterCompiler(absl::string_view platform_name,
                          absl::string_view compiler_variant,
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
  virtual absl::StatusOr<std::vector<std::string>> GetPhaseNames();

  // Compiles a set of input programs by running them through a specified
  // sequence of compilation phases. This function internally calls
  // `ValidatePhase` to trigger plugin-specific validation of input
  // compatibility before invoking the appropriate phase compilers. The output
  // of one phase is passed as input to the next. Returns the vector of
  // `PjRtPartialProgramProto` objects resulting from the last executed phase,
  // or an error status if any validation or compilation step fails.
  virtual absl::StatusOr<std::vector<PjRtPartialProgramProto>> RunPhases(
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

  // PhaseCompiler does not support topology deserialization for now.
  absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
  DeserializePjRtTopologyDescription(
      const std::string& serialized_topology) override {
    return absl::UnimplementedError(
        "DeserializePjRtTopologyDescription is not implemented.");
  }

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
