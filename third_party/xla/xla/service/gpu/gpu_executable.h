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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/gpu/gpu_module_globals.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// GPU-targeting implementation of the XLA Executable interface.
//
// Launches the given GPU kernel via the StreamExecutor.
//
// This is an immutable data type after initialization, and thus thread safe.
class GpuExecutable : public Executable {
 public:
  struct NumAdditionalStreams {
    int compute = 0;
    int communication = 0;
    bool need_d2h_stream = false;
    bool need_h2d_stream = false;
  };

  using ConstantInfo = GpuModuleGlobals::ConstantInfo;

  struct OutputInfo {
    // Corresponding allocation index.
    // Note that each output lives on its own allocation, i.e., it is allocated
    // in a slice with offset 0, and size equal to the size of the allocation.
    int allocation_index;

    // Output is passed-through from a parameter.
    bool passthrough = false;

    // Whether this output is hinted to alias a parameter (BufferAllocation*
    // would indicate the aliased parameter), and what kind of alias it is.
    std::optional<HloInputOutputAliasConfig::Alias> alias_config;

    GpuExecutableProto::OutputInfoProto ToProto() const;
    static absl::StatusOr<OutputInfo> FromProto(
        const GpuExecutableProto::OutputInfoProto& proto);

    friend bool operator==(const OutputInfo& lhs, const OutputInfo& rhs) {
      return std::tie(lhs.allocation_index, lhs.passthrough,
                      lhs.alias_config) ==
             std::tie(rhs.allocation_index, rhs.passthrough, rhs.alias_config);
    }

    friend bool operator!=(const OutputInfo& lhs, const OutputInfo& rhs) {
      return !(lhs == rhs);
    }
  };

  struct Params {
    std::vector<uint8_t> binary;
    BinaryMap dnn_compiled_graphs;
    std::unique_ptr<ThunkExecutor> executable;
    std::vector<ConstantInfo> constants;
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
    std::string module_name;
    ProgramShape program_shape;
    std::vector<BufferAllocation> allocations;
    std::unique_ptr<GpuAliasInfo> alias_info;
    DebugOptions debug_options;
    se::DeviceDescription device_description;
    // The module the executable was compiled from, if it's available:
    //
    // - `std::monostate` (or a null `std::shared_ptr`): There's no module.
    // - `std::shared_ptr<HloModule>`: The module. The executable keeps it
    //   alive.
    // - `HloModuleProtoWithConfig`: The serialized module. The executable
    //   only builds the module from it when someone needs it (see
    //   `Executable::GetOrCreateHloModule`). `hlo_module().pjrt_id()` must be
    //   set and non-negative, and it becomes the unique id of the built
    //   module. Serialized executables (`GpuExecutableProto`) don't carry a
    //   `pjrt_id`, so a deserializer has to assign a fresh id (e.g.
    //   `HloModule::GetNextUniqueModuleId()`) to its copy of the module.
    //   Building the module is expensive, so it's observable on
    //   purpose: It's logged together with the reason why the module is
    //   needed, and it increments the counter
    //   `/xla/gpu/hlo_module_materializations{reason}`. If
    //   `xla_gpu_experimental_fail_on_lazy_hlo_module_materialization` is set
    //   in the compile- or load-time options, it fails with
    //   `FailedPrecondition` instead (and still increments the counter).
    std::variant<std::monostate, std::shared_ptr<HloModule>,
                 HloModuleProtoWithConfig>
        hlo_module;
    bool enable_debug_info_manager = true;
    ModuleStats module_stats;
    se::ExecutableAbiVersion executable_abi_version;
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options;
    BufferAssignmentProto buffer_assignment_proto;
    std::string buffer_allocations_debug_summary;
    std::optional<GpuTopology> gpu_topology;
    // Metadata of the module the executable was compiled from. Only used if
    // `hlo_module` doesn't hold a module or a serialized module (e.g. with
    // `embed_hlo_module=false`), since otherwise it's read from there.
    std::optional<DeviceAssignment> static_device_assignment;
    HloBufferDonorConfig buffer_donor_config;
    int64_t partition_size = 0;
    // If not set, the layout of `program_shape` is used.
    std::optional<ComputationLayout> entry_computation_layout;
    // The fingerprint of the optimized HLO module (as returned by
    // `HloModule::GetFingerprint128()`), if it's known (e.g. when
    // deserializing). If it's empty and `hlo_module` holds a module, `Create`
    // computes it from the module.
    std::string fingerprint;
    // Whether collectives should use minimal GPU resources. If not set, the
    // value is computed from the module in `hlo_module` (false if there is no
    // module). It must be set if `hlo_module` holds a serialized module,
    // because computing it would require building the module.
    std::optional<bool> collective_use_minimal_resource;
  };

  static absl::StatusOr<std::unique_ptr<GpuExecutable>> Create(Params params);
  ~GpuExecutable() override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  absl::string_view name() const override { return module_name_; }

  xla::Shape result_shape() const override { return program_shape_.result(); }

  const absl::flat_hash_map<ShapeIndex, OutputInfo>& output_info() const {
    return output_info_;
  }

  ComputationLayout compute_computation_layout() const override {
    return ComputationLayout(program_shape_, /*ignore_layouts=*/false);
  }

  // Returns the binary stored in this GpuExecutable.
  //
  // The binary is cubin in Cuda, and HSA code object in ROCm. It may be empty,
  // in which case compilation is left up to the GPU driver. If both text() and
  // binary() are empty, that means the HLO required no custom kernels to be
  // compiled.
  const std::vector<uint8_t>& binary() const { return binary_; }

  const BinaryMap& dnn_compiled_graphs() const { return dnn_compiled_graphs_; }

  // ExecuteAsyncOnStream will fail if the compute capability of the stream
  // doesn't match the compute capability passed to this object's constructor.
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

  absl::StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments) override;

  using VariantArguments = std::variant<absl::Span<const ShapedBuffer* const>,
                                        absl::Span<ExecutionInput>>;
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStreamImpl(
      const ServiceExecutableRunOptions* run_options,
      VariantArguments arguments);

  absl::Span<const BufferAllocation* absl_nonnull const> GetAllocations()
      const override {
    return allocation_ptrs_;
  }

  absl::Span<const BufferAllocation> allocations() const {
    return allocations_;
  }

  const std::vector<ConstantInfo>& constants() const { return constants_; }

  // Human readable summary of the buffer allocations. Tailored to debugging
  // OOMs, includes the Hlo op metadata for every buffer associated with each
  // allocation.
  const std::string& buffer_allocations_debug_summary() const {
    return buffer_allocations_debug_summary_;
  }

  // Returns the stored buffer assignment proto.
  const BufferAssignmentProto& buffer_assignment_proto() const;

  const GpuAliasInfo* alias_info() const { return alias_info_.get(); }

  const ThunkExecutor& thunk_executor() const { return *thunk_executor_; }

  const ThunkExecutor::DefinitionPlan& definition_plan() const {
    return definition_plan_;
  }

  GpuExecutableBufferAllocator& buffer_allocator() {
    return *buffer_allocator_;
  }
  const GpuExecutableBufferAllocator& buffer_allocator() const {
    return *buffer_allocator_;
  }

  // `persistent_alloc_indices` lists allocations whose device addresses are
  // stable for this execution. Passing std::nullopt disables command buffer
  // lowering for this step (CommandBufferThunk falls back to its thunk
  // sequence); pass a valid - possibly empty - span once the allocation
  // address policy is decided.
  absl::Status ExecuteThunks(
      const BufferAllocations& buffer_allocations,
      const ServiceExecutableRunOptions* run_options,
      std::optional<absl::Span<const BufferAllocation::Index>>
          persistent_alloc_indices = std::nullopt);

  using BufferAllocToDeviceMemoryMap =
      GpuModuleGlobals::BufferAllocToDeviceMemoryMap;

  // Loads the PTX or CUBIN for this executable and initializes all
  // constants that haven't already been initialized by the CUDA driver. Loaded
  // modules are owned by this executable.
  //
  // Returns a map from buffer allocation indices to device memory pointers
  // (only for allocations that contain constants).
  //
  // The returned map is cached. If the above process has already been run for
  // the given stream, it is skipped and the cached map is immediately returned
  // instead.
  absl::StatusOr<const BufferAllocToDeviceMemoryMap*> ResolveConstantGlobals(
      se::Stream* stream);

  // Creates a `GpuExecutable` from its proto representation.
  //
  // If `debug_module` isn't populated, the HLO module is deserialized from the
  // proto, which can be expensive.
  static absl::StatusOr<std::unique_ptr<GpuExecutable>> FromProto(
      const GpuExecutableProto&,
      const se::DeviceDescription& device_description,
      absl::string_view platform, DebugOptions debug_options,
      const std::optional<se::KernelLoaderSpec::SymbolResolver>&
          symbol_resolver = std::nullopt,
      std::shared_ptr<HloModule> debug_module = nullptr);

  absl::StatusOr<GpuExecutableProto> ToProto() const;

  absl::Status DumpExecutableIfEnabled(
      const ExecutableBuildOptions& options,
      const DebugOptions& debug_options) const final;

  absl::StatusOr<se::ExecutableAbiVersion> GetExecutableAbiVersion()
      const override {
    return executable_abi_version_;
  }

  const std::optional<xla::cpu::TargetMachineOptions>&
  cpu_target_machine_options() const {
    return cpu_target_machine_options_;
  }

  // The GPU topology the executable was compiled for. Not set for executables
  // deserialized from protos that predate the `gpu_topology` field.
  const std::optional<GpuTopology>& gpu_topology() const {
    return gpu_topology_;
  }

  // The DebugOptions that execution uses. See `runtime_debug_options_` for
  // which flags are read from here and which ones from the load-time options.
  const DebugOptions& runtime_debug_options() const {
    return runtime_debug_options_;
  }

  // The `partition_size` from the config of the module this executable was
  // compiled from, or 0 if unknown.
  int64_t partition_size() const { return partition_size_; }

  // The static device assignment from the config of the module this executable
  // was compiled from, or nullptr if it doesn't have one.
  const DeviceAssignment* static_device_assignment() const override {
    return static_device_assignment_.has_value() ? &*static_device_assignment_
                                                 : nullptr;
  }

  // Same as the module's input/output alias config, but doesn't need the
  // module.
  const HloInputOutputAliasConfig& input_output_alias_config() const override {
    return input_output_alias_config_;
  }

  // Same as the module's buffer donor config, but doesn't need the module.
  const HloBufferDonorConfig& buffer_donor_config() const override {
    return buffer_donor_config_;
  }

  // Same as the entry computation layout of the module's config, but doesn't
  // need the module.
  const ComputationLayout& entry_computation_layout() const override {
    return entry_computation_layout_;
  }

  // Returns the fingerprint of the optimized HLO module (as returned by
  // `HloModule::GetFingerprint128()`), or an empty string if it's unknown
  // (i.e. the executable was created without a module and without a
  // fingerprint).
  const std::string& fingerprint() const { return fingerprint_; }

  // Prepares profiling for an executable that was created with a serialized
  // module instead of a module, if a profiler is active. The first call that
  // sees an active profiler gets the module (see `GetOrCreateHloModule`, which
  // builds it unless that already happened for another reason), switches
  // executions from the name-only annotations to the full annotations prepared
  // from it and registers it with XlaDebugInfoManager. If building the module
  // fails, it logs a warning and executions keep using the name-only
  // annotations. Does nothing for executables that were created with a
  // module, which are prepared for profiling at construction.
  //
  // Every execution calls this, so there's usually no need to call it
  // directly. It's thread-safe.
  void MaybePrepareProfiling();

 private:
  // The parts of the HloModuleConfig that the executable needs at run time.
  struct ModuleConfigMetadata {
    DebugOptions debug_options;
    int64_t partition_size = 0;
    std::optional<DeviceAssignment> static_device_assignment;
    HloBufferDonorConfig buffer_donor_config;
    // If not set, the layout of `program_shape` is used.
    std::optional<ComputationLayout> entry_computation_layout;
  };

  // Reads the module config metadata from `proto` without deserializing the
  // HLO module. Gives the same result as `HloModule::CreateFromProtoWithConfig`
  // followed by the load-time overrides in `FromProto`.
  static absl::StatusOr<ModuleConfigMetadata> ModuleConfigMetadataFromProto(
      const HloModuleProtoWithConfig& proto,
      const DebugOptions& load_time_debug_options);

  // Builds the input/output alias config of the entry computation from the
  // aliasing stored in `output_info`. This gives the same config as the module
  // the executable was compiled from, because `output_info` has an entry for
  // every index of the result shape (see `GetOutputInfo`).
  static absl::StatusOr<HloInputOutputAliasConfig>
  InputOutputAliasConfigFromOutputInfo(
      const Shape& result_shape,
      const absl::flat_hash_map<ShapeIndex, OutputInfo>& output_info);

  // Like `Create`, but takes the module config metadata explicitly instead of
  // reading it from `params.hlo_module`. If `module_config_metadata` is not
  // set, it's read from the (serialized) module in `params.hlo_module` if
  // available and otherwise defaults to `params.debug_options`.
  static absl::StatusOr<std::unique_ptr<GpuExecutable>> CreateImpl(
      Params params,
      std::optional<ModuleConfigMetadata> module_config_metadata);

  // Additional streams borrowed at run time for the execution.
  struct BorrowedStreams {
    std::vector<se::Stream*> streams;
    std::vector<StreamPool::Ptr> owners;

    // Assigns `stream` to all requested stream slots.
    static BorrowedStreams Assign(se::Stream* stream, int num_streams);
  };

  // Use GpuExecutable::Create() to create an instance. `hlo_module` is
  // either a `std::shared_ptr<HloModule>` (which may be null) or an
  // `HloModuleFactory`, and it's passed on to the matching constructor of
  // `Executable`. `serialized_hlo_module` is the serialized module the factory
  // builds the module from (null if there's no factory).
  template <typename HloModuleOrFactory>
  GpuExecutable(
      HloModuleOrFactory hlo_module,
      std::shared_ptr<const HloModuleProtoWithConfig> serialized_hlo_module,
      std::vector<uint8_t> binary, BinaryMap dnn_compiled_graphs,
      se::DeviceDescription device_description,
      std::unique_ptr<ThunkExecutor> executable, std::string module_name,
      ProgramShape program_shape, std::vector<BufferAllocation> allocations,
      std::deque<BufferAllocation> thunk_pass_allocations,
      std::unique_ptr<GpuAliasInfo> alias_info, DebugOptions debug_options,
      std::vector<ConstantInfo> constants,
      absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
      HloInputOutputAliasConfig input_output_alias_config,
      bool enable_debug_info_manager, ModuleStats module_stats,
      absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto,
      se::ExecutableAbiVersion executable_abi_version,
      std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options,
      BufferAssignmentProto buffer_assignment_proto,
      std::string buffer_allocations_debug_summary,
      bool collective_use_minimal_resource,
      std::optional<GpuTopology> gpu_topology,
      ModuleConfigMetadata module_config_metadata, std::string fingerprint);

  // GpuExecutable check with either AMD's ISA version, or Nvidia's major minor
  // version for compute capability, depending on the hardware.
  absl::Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options);

  static absl::StatusOr<BorrowedStreams> BorrowStreams(
      const ServiceExecutableRunOptions& run_options, int device_ordinal,
      int num_streams, se::StreamPriority priority);

  static absl::Status ExecuteThunksImpl(
      const DebugOptions* debug_options, const std::string& module_name,
      ThunkExecutor& thunk_executor, Thunk::ExecutableSource executable_source,
      const ServiceExecutableRunOptions* run_options,
      const BufferAllocations& buffer_allocations, bool block_host_until_done,
      std::optional<absl::Span<const BufferAllocation::Index>>
          persistent_alloc_indices,
      NumAdditionalStreams num_additional_streams,
      CollectiveMemoryCache& collective_memory_cache,
      bool collective_use_minimal_resource);

  // Compare current allocation's address with previous run's address, and
  // report the allocation info if memory addressed changed. Useful for identify
  // in user's model if it is command buffer perf friendly (no command buffer
  // update cost).
  void LogChangedAllocationsInBetweenExecutions(
      const BufferAllocations& buffer_allocations,
      const ServiceExecutableRunOptions* run_options);

  // The serialized module that the `HloModuleFactory` builds the module from,
  // shared with the factory. Null if the executable was created without a
  // serialized module. `ToProto` writes it as long as the module hasn't been
  // built.
  std::shared_ptr<const HloModuleProtoWithConfig> serialized_hlo_module_;

  // The GPU machine code for the computation, targeting GPUs at
  // compute_capability_.
  //
  // May be empty, in which case we leave compilation up to the GPU driver.
  std::vector<uint8_t> binary_;

  BinaryMap dnn_compiled_graphs_;

  // The GPU version for compute compatibility check.
  se::GpuComputeCapability gpu_version_;

  // The thunks to be invoked by this GpuExecutable. They are generated by the
  // ThunkEmitter.
  std::unique_ptr<ThunkExecutor> thunk_executor_;

  // Definition plan for the finalized thunk tree.
  ThunkExecutor::DefinitionPlan definition_plan_;

  // Number of additional streams available at run time.
  NumAdditionalStreams num_additional_streams_;

  std::string module_name_;

  ProgramShape program_shape_;

  // Provides information for allocating memory for every output/temp buffer.
  //
  // Non-owning pointers - allocation objects reside either in allocations_
  // or buffer_assignment_.
  //
  // A GpuExecutable can get its allocations in three ways:
  // 1 - From a regular compilation that uses allocations from MLIR.
  // 2 - From a regular compilation that uses the original allocations from
  //     the buffer assignment.
  // 3 - From loading the executable from an object file.
  //
  // In cases 1 and 3, the allocations are stored in allocations_ and in
  // case 2, they are part of the buffer_assignment.
  const std::vector<const BufferAllocation*> allocation_ptrs_;

  // The allocations_ object contains allocations that **may** be used to
  // provide information for allocating memory for every output/temp buffer.
  // See the comment on allocation_ptrs_.
  std::vector<BufferAllocation> allocations_;

  // Proto representation of the buffer assignments that was used to compile
  // this executable. The actual BufferAssignment is only available during
  // compilation.
  BufferAssignmentProto buffer_assignment_proto_;

  // Extra allocations added by thunk passes outside of the normal buffer
  // assignment process.
  // std::deque is used to ensure pointer stability.
  const std::deque<BufferAllocation> thunk_pass_allocations_;

  // Backend specific aliasing information whether operands can/should share the
  // buffer with the user.
  std::unique_ptr<GpuAliasInfo> alias_info_;

  // The DebugOptions used at run time. These are the compile-time options from
  // the module config, with `xla_dump_to` and
  // `xla_gpu_experimental_nccl_window_symmetric` taken from the load-time
  // options for loaded executables (see `FromProto`). If the executable has no
  // module config at all, these are the options passed to `Create`.
  //
  // This is a copy taken in `Create`. Changing the module config after that
  // (e.g. through `HloModule::mutable_config()`) doesn't affect execution.
  //
  // Not all flags are read from here. Today's per-flag precedence is:
  //
  // | Consumer                             | Options                  |
  // |--------------------------------------|--------------------------|
  // | ExecuteThunks (NCCL channels, stream | runtime_debug_options_   |
  // | priority, nccl_window_symmetric,     |                          |
  // | progress tracking, timeouts,         |                          |
  // | rendezvous timeouts)                 |                          |
  // | Buffer allocator (command buffer     | runtime_debug_options_   |
  // | update mode)                         |                          |
  // | ModuleAnnotations (trace annotation  | runtime_debug_options_   |
  // | level)                               |                          |
  // | Number of additional streams         | Params::debug_options    |
  // | Thunk passes                         | Params::debug_options    |
  // | debug_buffer_assignment_show_max     | Params::debug_options    |
  // | Thunk deserialization (FromProto)    | Params::debug_options    |
  DebugOptions runtime_debug_options_;

  // `partition_size` from the module config, or 0 if unknown.
  int64_t partition_size_;

  // Static device assignment from the module config, if any.
  std::optional<DeviceAssignment> static_device_assignment_;

  // The input/output alias config and the buffer donor config of the entry
  // computation.
  HloInputOutputAliasConfig input_output_alias_config_;
  HloBufferDonorConfig buffer_donor_config_;

  // The entry computation layout from the module config.
  ComputationLayout entry_computation_layout_;

  ModuleAnnotations module_annotations_;

  // The annotations that executions use: `module_annotations_`, or the full
  // annotations prepared by `MaybePrepareProfiling`.
  std::atomic<const ModuleAnnotations*> current_module_annotations_{
      &module_annotations_};

  // Whether the executable was created with a serialized module instead of a
  // module, i.e. whether the base class builds the module on demand.
  bool builds_hlo_module_on_demand_;

  // Whether `MaybePrepareProfiling` already ran, successfully or not.
  std::atomic<bool> profiling_prepared_{false};
  absl::Mutex profiling_mutex_;
  // The full annotations prepared by `MaybePrepareProfiling`. They refer to
  // the module, which the base class keeps alive.
  std::unique_ptr<ModuleAnnotations> profiling_module_annotations_
      ABSL_GUARDED_BY(profiling_mutex_);

  int64_t debug_buffer_assignment_show_max_;

  // Cache previous memory allocations for current module, this is used to help
  // identify if user's model have unstable pointers by turning on VLOG(5).
  absl::Mutex module_allocations_mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::vector<se::DeviceAddressBase>>
      module_allocations_ ABSL_GUARDED_BY(module_allocations_mutex_);

  std::vector<ConstantInfo> constants_;
  std::unique_ptr<GpuModuleGlobals> module_globals_;
  const absl::flat_hash_map<ShapeIndex, OutputInfo> output_info_;
  bool enable_debug_info_manager_;
  // Whether the module is registered with the XlaDebugInfoManager, either by
  // the constructor or by `MaybePrepareProfiling`. A module that
  // `GetOrCreateHloModule` builds for another reason isn't registered, so the
  // destructor must not rely on `has_module()`. Only set before other threads
  // can access the executable or while holding `profiling_mutex_`.
  bool registered_with_debug_info_manager_ = false;

  std::unique_ptr<GpuExecutableBufferAllocator> buffer_allocator_;

  GpuExecutable(const GpuExecutable&) = delete;
  GpuExecutable& operator=(const GpuExecutable&) = delete;

  // Stores the thunk sequence as a proto from before running the thunk pass.
  // Might contain an error if the given thunk graph is not serializable.
  absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto_;

  se::ExecutableAbiVersion executable_abi_version_;

  std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options_;

  CollectiveMemoryCache collective_memory_cache_;

  // Human readable summary of the buffer allocations. Tailored to debugging
  // OOMs, includes the Hlo op metadata for every buffer associated with each
  // allocation.
  std::string buffer_allocations_debug_summary_;

  const bool collective_use_minimal_resource_;

  // The GPU topology the executable was compiled for, if known.
  std::optional<GpuTopology> gpu_topology_;

  // The fingerprint of the optimized HLO module, or empty if it's unknown.
  std::string fingerprint_;
};

absl::StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
