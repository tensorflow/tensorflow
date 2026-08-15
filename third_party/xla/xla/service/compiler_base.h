/* Copyright 2017 The OpenXLA Authors.

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

// The compiler API is used by the XLA service to generate executables that
// run on a given platform. This is a registry and abstract interface, for
// pluggability by the various platforms.

#ifndef XLA_SERVICE_COMPILER_BASE_H_
#define XLA_SERVICE_COMPILER_BASE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiled_module_base.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/util.h"

namespace xla {

// The following types are used for ahead of time compilation.

class AotCompilationOptionsBase;

// Abstract superclass describing metadata produced during ahead-of-time
// compilation.
class AotCompilationMetadata {
 public:
  AotCompilationMetadata(const AotCompilationMetadata&) = delete;
  AotCompilationMetadata& operator=(AotCompilationMetadata const&) = delete;
  virtual std::string ToString() const { return ""; }
  virtual ~AotCompilationMetadata() = default;

 protected:
  AotCompilationMetadata() = default;
};

// Abstract compiler interface that is subclassed for compilation on a
// particular platform.
//
// The compiler ties together high level optimization (HLO) and low level
// optimization (LLO) / codegen (CG) to generate efficient executables for the
// target platform.
//
// The platform-based compiler singletons are registered via module initializers
// in their corresponding XLA compiler libraries, and are registered via the
// RegisterCompilerFactory API below.
//
// Thread-safety: subclasses of Compiler must be thread-safe, as multiple
// XLA clients may be requesting compilation concurrently for a given
// platform.
class CompilerBase {
 public:
  CompilerBase() = default;
  virtual ~CompilerBase();

  // Runs Hlo passes to optimize the given Hlo module, returns the optimized
  // module.
  virtual absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module) = 0;

  // Compiles the HLO module for execution on a device given by the executor,
  // and returns an executable object or an error status. No HLO passes are
  // applied to module. Generally a module should be passed through RunHloPasses
  // prior to calling this method because some HLO passes are required for
  // correctness. Takes ownership of the HLO module.
  //
  // The compiler may optionally specialize to the individual device
  // (not just type of device) indicated by the executor.
  virtual absl::StatusOr<std::unique_ptr<ExecutableBase>> RunBackend(
      std::unique_ptr<HloModule> module) = 0;

  // The following two interfaces are same as the above two, except they
  // facilitate the loading of buffer assignment from proto if available.

  // Note: The default implementation of the API here does not utilize the given
  // buffer assignment. Different backends are a expected to override the
  // following method to achieve this functionality.
  virtual absl::StatusOr<std::unique_ptr<ExecutableBase>>
  RunBackendWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto) {
    LOG(WARNING) << "Ignoring the buffer assignment proto provided.";
    return RunBackend(std::move(module));
  }

  // Compiles the HLO module for ahead-of-time execution.  This is intended for
  // use in static compilation.
  virtual absl::StatusOr<std::vector<std::unique_ptr<CompiledModuleBase>>>
  CompileAheadOfTime(std::unique_ptr<HloModule> module,
                     const AotCompilationOptionsBase& options) = 0;

  // Similar to CompileAheadOfTime above but AotCompilationMetadata
  // has an argument that can be populated during compilation.
  virtual absl::StatusOr<std::vector<std::unique_ptr<CompiledModuleBase>>>
  CompileAheadOfTime(std::unique_ptr<HloModule> module,
                     const AotCompilationOptionsBase& options,
                     std::unique_ptr<AotCompilationMetadata>* metadata);
};

// Abstract superclass describing options to an ahead-of-time compilation.
class AotCompilationOptionsBase {
 public:
  AotCompilationOptionsBase(const AotCompilationOptionsBase&) = delete;
  AotCompilationOptionsBase& operator=(AotCompilationOptionsBase const&) =
      delete;

  virtual ~AotCompilationOptionsBase() = default;

  virtual int64_t replica_count() const { return 0; }
  virtual int64_t num_cores() const { return 0; }
  virtual bool use_spmd_partitioning() const { return false; }
  virtual bool use_shardy_partitioner() const { return false; }
  virtual bool use_auto_spmd_partitioning() const { return false; }
  virtual std::vector<int64_t> auto_spmd_partitioning_mesh_shape() const {
    return {};
  }
  virtual std::vector<int64_t> auto_spmd_partitioning_mesh_ids() const {
    return {};
  }
  virtual bool deduplicate_hlo() const { return false; }
  virtual PrecisionConfig::Precision matrix_unit_operand_precision() const {
    return PrecisionConfig::DEFAULT;
  }

  const DebugOptions& debug_options() const { return debug_options_; }
  DebugOptions* mutable_debug_options() { return &debug_options_; }

  bool has_static_device_assignment() const {
    return static_device_assignment_.has_value();
  }
  const DeviceAssignment& static_device_assignment() const {
    CHECK(static_device_assignment_.has_value());
    return *static_device_assignment_;
  }
  void set_static_device_assignment(const DeviceAssignment& device_assignment) {
    static_device_assignment_ = device_assignment;
  }

  FusionConfigCollection fusion_config_collection() const {
    return fusion_config_collection_;
  }
  void set_fusion_config_collection(
      FusionConfigCollection fusion_config_collection) {
    fusion_config_collection_ = fusion_config_collection;
  }

  const std::vector<std::vector<bool>>& fusion_config() const {
    return fusion_config_;
  }
  void set_fusion_config(const std::vector<std::vector<bool>>& fusion_config) {
    fusion_config_ = fusion_config;
  }

  // Optional profile_version and cache key may be used to trigger recompilation
  // when a compilation cache is used.
  int64_t profile_version() const { return profile_version_; }
  void set_profile_version(int64_t profile_version) {
    profile_version_ = profile_version;
  }

  absl::string_view cache_key() const { return cache_key_; }
  void set_cache_key(absl::string_view cache_key) {
    cache_key_ = std::string(cache_key);
  }

  bool run_backend_only() const { return run_backend_only_; }
  void set_run_backend_only(bool run_backend_only) {
    run_backend_only_ = run_backend_only;
  }

  enum class EarlyExitPoint {
    kNone,
    kAfterLayoutAssignment,
    kAfterConfigAssignment,
    kAfterBufferAssignment,
  };

  EarlyExitPoint early_exit_point() const { return early_exit_point_; }
  void set_early_exit_point(EarlyExitPoint early_exit_point) {
    early_exit_point_ = early_exit_point;
  }

 protected:
  AotCompilationOptionsBase();

 private:
  DebugOptions debug_options_;
  std::optional<DeviceAssignment> static_device_assignment_;
  std::vector<std::vector<bool>> fusion_config_;
  FusionConfigCollection fusion_config_collection_ =
      FusionConfigCollection::kOff;
  int64_t profile_version_ = 0;
  std::string cache_key_;
  bool run_backend_only_ = false;
  EarlyExitPoint early_exit_point_ = EarlyExitPoint::kNone;
};

}  // namespace xla

#endif  // XLA_SERVICE_COMPILER_BASE_H_
