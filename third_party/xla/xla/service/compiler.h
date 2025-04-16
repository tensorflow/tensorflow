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

#ifndef XLA_SERVICE_COMPILER_H_
#define XLA_SERVICE_COMPILER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/metrics_hook_interface.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/threadpool.h"

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace xla {

// The following types are used for ahead of time compilation.

// Contains the object file data created as a result of ahead-of-time
// computation.
using ObjectFileData = std::vector<char>;

class Compiler;
class AotCompilationOptions;

// Abstract superclass describing the result of an ahead-of-time compilation.
class AotCompilationResult {
 public:
  AotCompilationResult(const AotCompilationResult&) = delete;
  AotCompilationResult& operator=(AotCompilationResult const&) = delete;

  virtual ~AotCompilationResult() = default;

  virtual absl::StatusOr<std::string> SerializeAsString() const {
    return Unimplemented("SerializeAsString unimplemented.");
  }

  virtual absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, const se::StreamExecutor* executor) const&& {
    return Unimplemented("LoadExecutable unimplemented.");
  }

  // Returns the optimized HLO module if one was computed and the implementation
  // supports it.
  virtual const HloModule* optimized_module() const = 0;
  virtual std::unique_ptr<HloModule> consume_optimized_module() = 0;

 protected:
  AotCompilationResult() = default;
};

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
class Compiler {
 public:
  // Description of a target device for compilation.
  struct TargetConfig {
    explicit TargetConfig(const se::GpuTargetConfigProto& proto);
    explicit TargetConfig(se::StreamExecutor* s);

    se::GpuTargetConfigProto ToProto() const;

    bool operator==(const TargetConfig& other) const {
      // TODO(cheshire): More efficient comparator, this is currently just for
      // tests.
      return ToProto().SerializeAsString() ==
             other.ToProto().SerializeAsString();
    }

    std::string ToString() { return ToProto().DebugString(); }

    se::DeviceDescription device_description;
    std::string platform_name;
    se::dnn::VersionInfo dnn_version_info;
    std::string device_description_str;
  };

  struct CompileOptions {
    // If device_allocator is not null, the compiler may use it to allocate temp
    // space on the device for use during compilation.  For example, the
    // compiler may allocate buffers on the device and then run variants of a
    // given algorithm over those buffers, to see which variant is fastest.  Any
    // space allocated will be deallocated before the compilation returns.
    se::DeviceMemoryAllocator* device_allocator = nullptr;

    // An optional thread pool for parallel compilation.
    tsl::thread::ThreadPool* thread_pool = nullptr;

    std::function<absl::StatusOr<std::pair<std::vector<Shape>, Shape>>(
        const HloModule& module)>
        layout_canonicalization_callback = {};

    bool is_autotuning_compilation = false;

    // AOT device description. If provided, used instead of querying the device
    // on which compilation is performed.
    std::optional<TargetConfig> target_config;

    MultiProcessKeyValueStore key_value_store;
  };

  virtual ~Compiler() = default;

  // Returns the ID of the platform that this compiler targets.
  virtual se::Platform::Id PlatformId() const = 0;

  // Runs Hlo passes to optimize the given Hlo module, returns the optimized
  // module.
  virtual absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      const CompileOptions& options) = 0;
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      se::DeviceMemoryAllocator* device_allocator) {
    return RunHloPasses(std::move(module), executor,
                        CompileOptions{device_allocator});
  }

  // Compiles the HLO module for execution on a device given by the executor,
  // and returns an executable object or an error status. No HLO passes are
  // applied to module. Generally a module should be passed through RunHloPasses
  // prior to calling this method because some HLO passes are required for
  // correctness. Takes ownership of the HLO module.
  //
  // The compiler may optionally specialize to the individual device
  // (not just type of device) indicated by the executor.
  virtual absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      const CompileOptions& options) = 0;
  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      se::DeviceMemoryAllocator* device_allocator) {
    return RunBackend(std::move(module), executor,
                      CompileOptions{device_allocator});
  }

  // The following two interfaces are same as the above two, except they
  // facilitate the loading of buffer assignment from proto if available.

  // Note: The default implementation of the API here does not utilize the given
  // buffer assignment. Different backends are a expected to override the
  // following method to achieve this functionality.
  virtual absl::StatusOr<std::unique_ptr<Executable>>
  RunBackendWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* /*buffer_assignment_proto*/,
      se::StreamExecutor* executor, const CompileOptions& options) {
    LOG(WARNING) << "Ignoring the buffer assignment proto provided.";
    return RunBackend(std::move(module), executor, options);
  }

  absl::StatusOr<std::unique_ptr<Executable>> RunBackendWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto,
      se::StreamExecutor* executor,
      se::DeviceMemoryAllocator* device_allocator) {
    return RunBackendWithBufferAssignment(std::move(module),
                                          buffer_assignment_proto, executor,
                                          CompileOptions{device_allocator});
  }

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  virtual absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  LoadAotCompilationResult(const std::string& serialized_aot_result) {
    return Unimplemented("LoadAotCompilationResult unimplemented.");
  }

  // Compiles a set of HLO modules that can run in parallel, potentially
  // communicating data between the modules, and returns a corresponding
  // sequence of executable objects.
  //
  // TODO(b/68666782): Remove this method after adding support for multiple
  // modules to RunHloPasses and RunBackends.
  virtual absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      const CompileOptions& options) = 0;
  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      se::DeviceMemoryAllocator* device_allocator) {
    return Compile(std::move(module_group), stream_exec,
                   CompileOptions{device_allocator});
  }

  // Returns the backend configurations that the backend will consider for the
  // given HLO. Returns no configurations if the backend does not support
  // configurations for the given HLO.
  //
  // The stream executor is passed in to provide information about the hardware
  // that the backend configurations would be targeting.
  virtual std::vector<std::unique_ptr<tsl::protobuf::Message>>
  ComputeBackendConfigs(const HloInstruction& hlo,
                        se::StreamExecutor* executor) const;

  // Returns the backend configuration that the backend chooses by default for
  // the given HLO. Returns no configuration if the backend does not support
  // configurations for the given HLO.
  //
  // The stream executor is passed in to provide information about the hardware
  // that the backend configurations would be targeting.
  virtual std::unique_ptr<tsl::protobuf::Message> ComputeDefaultBackendConfig(
      const HloInstruction& hlo, se::StreamExecutor* executor) const;

  // Compiles the HLO module group for ahead-of-time execution.  This is
  // intended for use in static compilation.
  virtual absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) = 0;

  // Similar to CompileAheadOfTime above but AotCompilationMetadata
  // has an argument that can be populated during compilation.
  virtual absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options,
                     std::unique_ptr<AotCompilationMetadata>* metadata);

  /////
  // The Compiler class also serves as a point to register compiler objects
  // for the various platforms.

  using CompilerFactory = std::function<std::unique_ptr<Compiler>()>;

  // Registers the compiler singleton for the platform. This is assumed to
  // be a singleton, so no ownership is transferred.
  //
  // Precondition: a platform kind must not be registered more than once.
  static void RegisterCompilerFactory(se::Platform::Id platform_id,
                                      CompilerFactory compiler_factory);

  // Returns the compiler singleton pointer if it is available for the given
  // platform, or an error status if it is not.
  static absl::StatusOr<Compiler*> GetForPlatform(const se::Platform* platform);

  // Returns a function that computes the size in bytes of the logical
  // buffer that contains a shape.
  virtual HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const = 0;

  // Returns a function that computes the size in bytes of a given
  // logical buffer.
  std::function<int64_t(const BufferValue&)> BufferSizeBytesFunction() const {
    HloCostAnalysis::ShapeSizeFunction shape_size = ShapeSizeBytesFunction();
    return [shape_size](const BufferValue& buffer) {
      return shape_size(buffer.shape());
    };
  }

  virtual Shape DefaultDeviceShapeRepresentation(const Shape& shape) const {
    return shape;
  }

  // Returns an AotCompilationResult of the executable for serialization.
  virtual absl::StatusOr<std::unique_ptr<AotCompilationResult>> Export(
      Executable* executable) const {
    return Unimplemented("Export unimplemented");
  }

  // Returns a MetricsHookInterface object used to instrument Compiler's
  // compilation stages.
  virtual std::unique_ptr<MetricsHookInterface> CreateMetricsHook(
      absl::string_view filename_prefix) const;

  virtual absl::StatusOr<std::unique_ptr<Executable>> DeserializeExecutable(
      const absl::string_view serialized) const {
    return Unimplemented("DeserializeExecutable unimplemented");
  }

 private:
  // Mutex that guards the platform-compiler map.
  static absl::Mutex platform_compiler_mutex_;

  // Map from platform kind to compiler factory.
  static absl::flat_hash_map<se::Platform::Id, CompilerFactory>*
  GetPlatformCompilerFactories();

  // Map from platform kind to compiler instance, if we made one already (based
  // on the factories above).
  static absl::flat_hash_map<se::Platform::Id, std::unique_ptr<Compiler>>*
  GetPlatformCompilers();
};

// Abstract superclass describing options to an ahead-of-time compilation.
class AotCompilationOptions {
 public:
  AotCompilationOptions(const AotCompilationOptions&) = delete;
  AotCompilationOptions& operator=(AotCompilationOptions const&) = delete;

  explicit AotCompilationOptions(se::Platform::Id platform_id)
      : platform_id_(platform_id), debug_options_(GetDebugOptionsFromFlags()) {}
  virtual ~AotCompilationOptions() = default;

  // Returns the ID of the platform to which these options apply.
  virtual se::Platform::Id PlatformId() const { return platform_id_; }

  virtual int64_t replica_count() const { return 0; }
  virtual int64_t num_cores() const { return 0; }
  virtual bool use_spmd_partitioning() const { return false; }
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

  // Optional allocator that may be used for allocating temp space on the device
  // during compilation.
  se::DeviceMemoryAllocator* device_allocator() const {
    return device_allocator_;
  }
  void set_device_allocator(se::DeviceMemoryAllocator* device_allocator) {
    device_allocator_ = device_allocator;
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

  se::StreamExecutor* executor() const { return executor_; }
  void set_executor(se::StreamExecutor* executor) { executor_ = executor; }

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

  bool sanitize_dataflow() const { return sanitize_dataflow_; }
  void set_sanitize_dataflow(bool sanitize_dataflow) {
    sanitize_dataflow_ = sanitize_dataflow;
  }

  const std::vector<std::string>& sanitize_abilists_dataflow() const {
    return sanitize_abilists_dataflow_;
  }
  void set_sanitize_abilists_dataflow(
      const std::vector<std::string>& abilists) {
    sanitize_abilists_dataflow_ = abilists;
  }

  const std::optional<Compiler::TargetConfig>& target_config() const {
    return target_config_;
  }
  void set_target_config(const Compiler::TargetConfig& target_config) {
    target_config_ = target_config;
  }

 protected:
  AotCompilationOptions();

 private:
  se::Platform::Id platform_id_;
  se::DeviceMemoryAllocator* device_allocator_ = nullptr;
  DebugOptions debug_options_;
  std::optional<DeviceAssignment> static_device_assignment_;
  std::vector<std::vector<bool>> fusion_config_;
  FusionConfigCollection fusion_config_collection_ =
      FusionConfigCollection::kOff;
  se::StreamExecutor* executor_ = nullptr;
  int64_t profile_version_ = 0;
  std::string cache_key_;
  bool run_backend_only_ = false;
  bool sanitize_dataflow_ = false;
  std::vector<std::string> sanitize_abilists_dataflow_;
  // Contains target-specific information required by AOT compilation.
  std::optional<Compiler::TargetConfig> target_config_;
};

}  // namespace xla

#endif  // XLA_SERVICE_COMPILER_H_
