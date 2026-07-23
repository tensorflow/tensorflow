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
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/gpu/gpu_module_globals.h"
#include "xla/service/gpu/ir_emission_utils.h"
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
    std::unique_ptr<HloModule> debug_module = nullptr;
    bool enable_debug_info_manager = true;
    ModuleStats module_stats;
    se::ExecutableAbiVersion executable_abi_version;
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options;
    BufferAssignmentProto buffer_assignment_proto;
    std::string buffer_allocations_debug_summary;
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

  GpuExecutableBufferAllocator& buffer_allocator() {
    return *buffer_allocator_;
  }
  const GpuExecutableBufferAllocator& buffer_allocator() const {
    return *buffer_allocator_;
  }

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
  // Additional streams borrowed at run time for the execution.
  struct BorrowedStreams {
    std::vector<se::Stream*> streams;
    std::vector<StreamPool::Ptr> owners;

    // Assigns `stream` to all requested stream slots.
    static BorrowedStreams Assign(se::Stream* stream, int num_streams);
  };

  // Use GpuExecutable::Create() to create an instance.
  explicit GpuExecutable(
      std::unique_ptr<HloModule> debug_module, std::vector<uint8_t> binary,
      BinaryMap dnn_compiled_graphs, se::DeviceDescription device_description,
      std::unique_ptr<ThunkExecutor> executable, std::string module_name,
      ProgramShape program_shape, std::vector<BufferAllocation> allocations,
      std::deque<BufferAllocation> thunk_pass_allocations,
      std::unique_ptr<GpuAliasInfo> alias_info, DebugOptions debug_options,
      std::vector<ConstantInfo> constants,
      absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
      bool enable_debug_info_manager, ModuleStats module_stats,
      absl::StatusOr<std::vector<ThunkProto>> thunk_sequence_proto,
      se::ExecutableAbiVersion executable_abi_version,
      std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options,
      BufferAssignmentProto buffer_assignment_proto,
      std::string buffer_allocations_debug_summary,
      bool collective_use_minimal_resource);

  // GpuExecutable check with either AMD's ISA version, or Nvidia's major minor
  // version for compute capability, depending on the hardware.
  absl::Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options);

  static absl::StatusOr<BorrowedStreams> BorrowStreams(
      const ServiceExecutableRunOptions& run_options, int device_ordinal,
      int num_streams, se::StreamPriority priority);

  static absl::Status ExecuteThunksImpl(
      const DebugOptions* debug_options, const std::string& module_name,
      ModuleIdentifier module_id, ThunkExecutor& thunk_executor,
      Thunk::ExecutableSource executable_source,
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

  ModuleAnnotations module_annotations_ = [this] {
    if (has_module()) {
      return ModuleAnnotations(module());
    }
    return ModuleAnnotations(module_name_);
  }();

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
};

absl::StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
