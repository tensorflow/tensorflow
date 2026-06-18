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

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
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
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/scoped_module_handle.h"
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
  };

  struct ConstantInfo {
    std::string symbol_name;
    DenseDataIntermediate content;
    int allocation_index = -1;

    GpuExecutableProto::ConstantInfoProto ToProto(
        bool skip_content_serialization = false) const;

    static absl::StatusOr<ConstantInfo> FromProto(
        const GpuExecutableProto::ConstantInfoProto& proto,
        const absl::flat_hash_map<std::string,
                                  const HloInstruction*>* absl_nullable
            content_overrides = nullptr);
  };

  struct OutputInfo {
    // Corresponding allocation index.
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
    std::string asm_text;
    std::vector<uint8_t> binary;
    BinaryMap dnn_compiled_graphs;
    std::unique_ptr<ThunkExecutor> executable;
    std::vector<ConstantInfo> constants;
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
    std::string module_name;
    ProgramShape program_shape;
    std::optional<std::vector<BufferAllocation>> mlir_allocations;
    std::unique_ptr<const BufferAssignment> buffer_assignment;
    std::unique_ptr<GpuAliasInfo> alias_info;
    DebugOptions debug_options;
    se::DeviceDescription device_description;
    std::unique_ptr<HloModule> debug_module = nullptr;
    bool enable_debug_info_manager = true;
    ModuleStats module_stats;
    se::ExecutableAbiVersion executable_abi_version;
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options;
    std::optional<BufferAssignmentProto> buffer_assignment_proto;
  };

  static absl::StatusOr<std::unique_ptr<GpuExecutable>> Create(Params params);
  ~GpuExecutable() override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  // This should be called after set_ir_module_string.
  const std::string& ir_module_string() const { return ir_module_string_; }

  absl::string_view name() const override { return module_name_; }

  xla::Shape result_shape() const override { return program_shape_.result(); }

  const absl::flat_hash_map<ShapeIndex, OutputInfo>& output_info() const {
    return output_info_;
  }

  ComputationLayout compute_computation_layout() const override {
    return ComputationLayout(program_shape_, /*ignore_layouts=*/false);
  }

  // This should be called before ExecuteOnStream.
  void set_ir_module_string(const std::string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  // Returns the compiled code for the computation.
  //
  // The compiled code is PTX in Cuda and unused empty string in ROCm.
  // This may be left empty for saving memory if we have a non-empty binary.
  // If both text() and binary() are empty, that means the HLO required no
  // custom kernels to be compiled.
  const std::string& text() const { return text_; }

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

  struct ParameterBuffer {
    se::DeviceAddressBase buffer;
    int64_t parameter_number;
  };

  // Resolves the device address backing an entry-computation-parameter
  // allocation. Returning a null DeviceAddressBase means "leave the buffer
  // unset" (e.g. a skipped tuple index-table allocation). The parameter number
  // is used only for diagnostics.
  using ParameterBufferResolver =
      absl::FunctionRef<absl::StatusOr<ParameterBuffer>(
          const BufferAllocation& allocation)>;

  absl::Span<const BufferAllocation* absl_nonnull const> GetAllocations()
      const override {
    return allocation_ptrs_;
  }

  const std::vector<ConstantInfo>& constants() const { return constants_; }

  // Only returns a non-null pointer if this executable was constructed with a
  // valid BufferAssignment. Deserialized executables do not have a valid
  // BufferAssignment and will return nullptr.
  const BufferAssignment* buffer_assignment() const {
    return buffer_assignment_.get();
  }

  // Returns the proto representation of `buffer_assignment()` if available,
  // otherwise returns the stored buffer assignment proto if available. Returns
  // nullopt if neither is available.
  std::optional<BufferAssignmentProto> buffer_assignment_proto() const;

  const GpuAliasInfo* alias_info() const { return alias_info_.get(); }

  const ThunkExecutor& thunk_executor() const { return *thunk_executor_; }

  absl::Status ExecuteThunks(const BufferAllocations& buffer_allocations,
                             const ServiceExecutableRunOptions* run_options);

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>;

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

  // Builds the BufferAllocations for an execution. Entry-computation-parameter
  // buffers are obtained from `get_parameter_buffer`; all other allocations
  // (thread-local, constant, temp/maybe-live-out) are resolved internally,
  // including collective-memory granularity rounding and alignment checking.
  absl::StatusOr<BufferAllocations> GenerateBufferAllocations(
      const ServiceExecutableRunOptions* run_options,
      ParameterBufferResolver get_parameter_buffer,
      const BufferAllocToDeviceMemoryMap* globals,
      se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

  // Copy-protection for an aliased output that was not donated at runtime:
  // allocates a fresh result buffer for the output at `index`, copies the
  // contents of the aliased buffer (allocation `allocation`) into it, and
  // redirects the aliased entry in `buffer_allocations` to the fresh buffer.
  // Returns the newly allocated result buffer.
  absl::StatusOr<se::DeviceAddressBase> AllocateCopyProtectedOutputBuffer(
      const ServiceExecutableRunOptions* run_options,
      BufferAllocations& buffer_allocations, const ShapeIndex& index,
      const BufferAllocation& allocation, int device_ordinal,
      se::DeviceAddressAllocator* memory_allocator);

  absl::Status VerboseAllocationError(absl::Status s);

  static absl::StatusOr<std::unique_ptr<GpuExecutable>> FromProto(
      const GpuExecutableProto&,
      const se::DeviceDescription& device_description,
      absl::string_view platform, DebugOptions debug_options,
      const std::optional<se::KernelLoaderSpec::SymbolResolver>&
          symbol_resolver = std::nullopt);

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

 private:
  // State for VA remapping of command buffer allocations on a single executor.
  struct VaRanges {
    // Mutex to protect VA range operations (map/execute/unmap) for this
    // executor. This ensures only one thread can use the VA ranges at a time.
    absl::Mutex mutex;

    // Single large virtual address reservation covering all command buffer
    // allocations. nullptr until first use.
    std::unique_ptr<se::MemoryReservation> va_reservation;

    // Event used to synchronize VA range reuse. When the device has completed
    // the task that uses the VA range, it marks the event, letting the host
    // know the VA range can be remapped to other physical addresses.
    std::unique_ptr<se::Event> unmap_event;

    // RAII wrapper that keeps the VA->physical mapping active.
    // Reset (auto-unmapping) before each re-use of the VA range.
    std::optional<se::MemoryReservation::ScopedMapping> scoped_mapping;
  };

  // Additional streams borrowed at run time for the execution.
  struct BorrowedStreams {
    std::vector<se::Stream*> streams;
    std::vector<StreamPool::Ptr> owners;

    // Assigns `stream` to all requested stream slots.
    static BorrowedStreams Assign(se::Stream* stream, int num_streams);
  };

  // Use GpuExecutable::Create() to create an instance.
  explicit GpuExecutable(
      std::unique_ptr<HloModule> debug_module, std::string asm_text,
      std::vector<uint8_t> binary, BinaryMap dnn_compiled_graphs,
      se::DeviceDescription device_description,
      std::unique_ptr<ThunkExecutor> executable, std::string module_name,
      ProgramShape program_shape,
      std::optional<std::vector<BufferAllocation>> mlir_allocations,
      std::unique_ptr<const BufferAssignment> buffer_assignment,
      std::deque<BufferAllocation> thunk_pass_allocations,
      std::unique_ptr<GpuAliasInfo> alias_info, DebugOptions debug_options,
      std::vector<ConstantInfo> constants,
      absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
      bool enable_debug_info_manager, ModuleStats module_stats,
      absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto,
      se::ExecutableAbiVersion executable_abi_version,
      std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options,
      std::optional<BufferAssignmentProto> buffer_assignment_proto);

  // GpuExecutable check with either AMD's ISA version, or Nvidia's major minor
  // version for compute capability, depending on the hardware.
  absl::Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options);

  absl::StatusOr<se::DeviceAddressBase> BufferForAllocation(
      ParameterBufferResolver get_parameter_buffer,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      const BufferAllocation& allocation,
      se::DeviceAddressAllocator* memory_allocator, int device_ordinal,
      int64_t arg_idx,
      const absl::flat_hash_map<LogicalBuffer::Color, int64_t>&
          allocate_granularity);

  static absl::StatusOr<BorrowedStreams> BorrowStreams(
      const ServiceExecutableRunOptions& run_options, int device_ordinal,
      int num_streams, se::StreamPriority priority);

  // Handles the VA remapping path of ExecuteThunks: reserves or remaps the
  // virtual address range for command buffer allocations, then delegates to
  // ExecuteThunksImpl with the remapped BufferAllocations.
  absl::Status ExecuteThunksWithVaRemapping(
      const BufferAllocations& buffer_allocations,
      const ServiceExecutableRunOptions* run_options,
      se::StreamExecutor* executor, int64_t unique_id,
      Thunk::ExecutableSource executable_source, bool block_host_until_done,
      bool collective_use_minimal_resource);

  static absl::Status ExecuteThunksImpl(
      const DebugOptions* debug_options, const std::string& module_name,
      ModuleIdentifier module_id, ThunkExecutor& thunk_executor,
      Thunk::ExecutableSource executable_source,
      const ServiceExecutableRunOptions* run_options,
      const BufferAllocations& buffer_allocations, bool block_host_until_done,
      NumAdditionalStreams num_additional_streams,
      CollectiveMemoryCache& collective_memory_cache,
      bool collective_use_minimal_resource);

  // The LLVM IR, in string format, of the unoptimized module generated for
  // this GpuExecutable. We save a string instead of an llvm::Module* because
  // leaving llvm::Module* in a singleton can cause the heap checker to emit
  // false positives.
  //
  // This string should be modified only before ExecuteOnStream.
  std::string ir_module_string_;

  // The compiled code for the computation.
  const std::string text_;

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
  std::optional<const std::vector<BufferAllocation>> allocations_;

  // The buffer_assignment_ object contains allocations that **may** be used to
  // provide information for allocating memory for every output/temp buffer.
  // See the comment on allocation_ptrs_.
  //
  // This object is also used for dumping debug info.
  std::shared_ptr<const xla::BufferAssignment> buffer_assignment_;

  // A buffer assignment proto may exists when `buffer_assignment_` is nullptr.
  // This happens when the executable is reconstructed from a proto (e.g. AOT).
  // The full BufferAssignment object can't be reconstructed because it requires
  // access to the compiler. But for debugging purposes, the proto is enough.
  std::optional<BufferAssignmentProto> buffer_assignment_proto_;

  // Extra allocations added by thunk passes outside of the normal buffer
  // assignment process.
  // std::deque is used to ensure pointer stability.
  const std::deque<BufferAllocation> thunk_pass_allocations_;

  // Backend specific aliasing information whether operands can/should share the
  // buffer with the user.
  std::unique_ptr<GpuAliasInfo> alias_info_;

  ModuleAnnotations module_annotations_ = [this] {
    if (has_module()) {
      return ModuleAnnotations(module());
    }
    return ModuleAnnotations(module_name_);
  }();

  int64_t debug_buffer_assignment_show_max_;

  absl::Mutex module_handle_mutex_;
  // Cache of module handles. Required to keep loaded modules alive until this
  // executable is destroyed.
  absl::flat_hash_map<se::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ ABSL_GUARDED_BY(module_handle_mutex_);
  // Cache of constant buffer allocation maps used by `ResolveConstantGlobals`.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<BufferAllocToDeviceMemoryMap>>
      module_globals_ ABSL_GUARDED_BY(module_handle_mutex_);

  // Cache previous memory allocations for current module, this is used to help
  // identify if user's model have unstable pointers by turning on VLOG(5).
  absl::flat_hash_map<se::StreamExecutor*, std::vector<se::DeviceAddressBase>>
      module_allocations_ ABSL_GUARDED_BY(module_handle_mutex_);

  std::vector<ConstantInfo> constants_;
  const absl::flat_hash_map<ShapeIndex, OutputInfo> output_info_;
  bool enable_debug_info_manager_;

  // Buffer allocation indices accessed by command buffer thunks. Using
  // btree_set for deterministic iteration order.
  absl::btree_set<BufferAllocation::Index> command_buffer_allocation_indexes_;

  // Separate mutex for VA ranges to avoid contention with module_handle_mutex_
  // during VA remapping operations which may involve GPU synchronization.
  absl::Mutex va_ranges_mutex_;
  absl::node_hash_map<std::pair<se::StreamExecutor*, int>, VaRanges>
      module_va_ranges_ ABSL_GUARDED_BY(va_ranges_mutex_);

  GpuExecutable(const GpuExecutable&) = delete;
  GpuExecutable& operator=(const GpuExecutable&) = delete;

  // Stores the thunk sequence as a proto from before running the thunk pass.
  // Might contain an error if the given thunk graph is not serializable.
  absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto_;

  se::ExecutableAbiVersion executable_abi_version_;

  std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options_;

  CollectiveMemoryCache collective_memory_cache_;
};

absl::StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment);

// Verifies that `buffer` satisfies the alignment required for `allocation`'s
// kind (entry parameter, constant, or XLA-allocated). `arg_idx` is used only
// for error messages.
absl::Status CheckAlignment(const BufferAllocation& allocation,
                            se::DeviceAddressBase buffer, int arg_idx);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
