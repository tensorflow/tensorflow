/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace tfrt {
namespace gpu {

class GpuContextCache;

}  // namespace gpu
}  // namespace tfrt

namespace xla {
namespace gpu {

// Returns whether GpuExecutable runs on TFRT (instead of thunks).
bool IsBefExecutableEnabled(const HloModuleConfig& config);

// Returns whether to create BefThunks (if the specific thunk is supported).
bool IsBefThunkEnabled(const HloModuleConfig& config);

inline bool IsBefEnabled(const HloModuleConfig& config) {
  return IsBefExecutableEnabled(config) || IsBefThunkEnabled(config);
}

// Returns whether GpuExecutable runs on TFRT/JitRt.
bool IsJitRtExecutableEnabled(const HloModuleConfig& config);

// GPU-targeting implementation of the XLA Executable interface.
//
// Launches the given GPU kernel via the StreamExecutor.
//
// This is an immutable data type after initialization, and thus thread safe.
class GpuExecutable : public Executable {
  struct BefBufferDeleter {
    void operator()(uint8_t* ptr) const;
    size_t size;
  };

  struct GpuContextCacheDeleter {
    void operator()(tfrt::gpu::GpuContextCache* ptr) const;
  };

 public:
  struct BefExecutable;
  struct JitRtExecutable;

  // Serialized MLIR module prepared for JitRt compilation.
  struct JitRtProgram {
    explicit JitRtProgram(std::string entry_point, std::string module,
                          std::vector<int64_t> buffer_sizes,
                          DebugOptions debug_options)
        : entry_point(std::move(entry_point)),
          module(std::move(module)),
          buffer_sizes(std::move(buffer_sizes)),
          debug_options(std::move(debug_options)) {}

    std::string entry_point;
    std::string module;
    std::vector<int64_t> buffer_sizes;
    DebugOptions debug_options;
  };

  typedef std::unique_ptr<const ThunkSchedule> OwnedThunkSchedule;
  typedef std::unique_ptr<uint8_t, BefBufferDeleter> OwnedBefBuffer;
  typedef std::unique_ptr<JitRtProgram> OwnedJitRtProgram;
  typedef std::unique_ptr<tfrt::gpu::GpuContextCache, GpuContextCacheDeleter>
      OwnedGpuContextCache;

  struct ConstantInfo {
    std::string symbol_name;
    std::vector<uint8_t> content;
    int allocation_index = -1;
  };

  struct OutputInfo {
    // Corresponding allocation index.
    int allocation_index;

    // Output is passed-through from a parameter.
    bool passthrough = false;

    // Whether this output is hinted to alias a parameter (BufferAllocation*
    // would indicate the aliased parameter), and what kind of alias it is.
    std::optional<HloInputOutputAliasConfig::Alias> alias_config;
  };

  struct Params {
    std::string asm_text;
    std::vector<uint8_t> binary;
    GpuVersion gpu_version;
    // The GpuExecutable will either execute Thunks, a whole-program BEF or a
    // JitRt compiled native function depending on which is supplied.
    absl::variant<OwnedThunkSchedule, OwnedBefBuffer, OwnedJitRtProgram>
        executable;
    xla::EntryFunctionAttributes entry_func_attrs;
    std::vector<ConstantInfo> constants;
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
    std::string module_name;
    xla::Shape output_shape;
    std::vector<BufferAllocation> allocations;
    std::unique_ptr<BufferAssignmentProto> debug_buffer_assignment = nullptr;

    // A callable that dumps out a debug string upon device OOM. It's not the
    // string itself, as the string can be huge and increase peak host memory
    // usage for the common (non-OOM) case.
    std::function<std::string()> verbose_buffer_assignment_string_dumper = [] {
      return std::string();
    };

    std::unique_ptr<HloModule> debug_module = nullptr;

    // Only relevant to whole-program BEF execution:
    // Optionally provide a cache of GPU contexts and corresponding
    // tfrt::ResourceContext(s). This can be used to supply
    // tfrt::ResourceContext(s) that are preloaded with GPU resources for given
    // GPU contexts. This isn't required for correct execution. However, it
    // prevents the initial execution step from being slowed down due to
    // initializing GPU resources.
    OwnedGpuContextCache gpu_ctx_cache;
  };

  // TODO(hanbinyoon): Once BEF replaces Thunks, hide this method as an
  // implementation detail of GpuExecutable.
  // Analyze the entry function to construct buffer allocation and other output
  // information. Optionally use buffer_param_offset to indicate the position of
  // buffer parameters in the entry function - in tfrt_gpu dialect, buffer
  // arguments start from the third parameter (after tfrt::Chain and GpuStream).
  static Status SetUpMlirAllocation(
      mlir::func::FuncOp func, llvm::ArrayRef<int64_t> buffer_sizes,
      std::vector<BufferAllocation>* allocations,
      absl::flat_hash_map<ShapeIndex, OutputInfo>* output_info,
      Shape* output_shape, int buffer_param_offset = 0);

  // Returns an Executable that is loaded from a BEF. This BEF must have entry
  // point information set using the 'tfrt-set-entry-point' pass.
  static StatusOr<std::unique_ptr<Executable>> LoadFromBef(
      std::shared_ptr<HloModule> hlo_module, absl::string_view bef,
      xla::EntryFunctionAttributes entry_func_attrs, GpuVersion gpu_version,
      stream_executor::StreamExecutor* executor);

  // Returns a cache of the given StreamExecutor's GPU context and a
  // corresponding tfrt::ResourceContext that is preloaded with the GPU
  // resources needed to run the specified BEF program.
  static StatusOr<OwnedGpuContextCache> CreatePreloadedGpuContextCache(
      llvm::ArrayRef<uint8_t> bef_array,
      stream_executor::StreamExecutor* executor);

  static StatusOr<std::unique_ptr<GpuExecutable>> Create(Params params);
  ~GpuExecutable() override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  // This should be called after set_ir_module_string.
  const std::string& ir_module_string() const { return ir_module_string_; }

  // This should be called before ExecuteOnStream.
  void set_ir_module_string(const std::string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  // Returns the compiled code for the computation. The compiled code is PTX in
  // Cuda and unused empty string in ROCm.
  const std::string& text() const { return text_; }

  // Returns the binary stored in this GpuExecutable. The binary is cubin in
  // Cuda, and HSA code object in ROCm. It may be empty, in which case
  // compilation is left up to the GPU driver.
  const std::vector<uint8_t>& binary() const { return binary_; }

  // ExecuteAsyncOnStream will fail if the compute capability of the stream
  // doesn't match the compute capability passed to this object's constructor.
  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  using VariantArguments = absl::variant<absl::Span<const ShapedBuffer* const>,
                                         absl::Span<ExecutionInput>>;
  StatusOr<ExecutionOutput> ExecuteAsyncOnStreamImpl(
      const ServiceExecutableRunOptions* run_options,
      VariantArguments arguments);

  absl::Span<const BufferAllocation> GetAllocations() const {
    return allocations_;
  }

  const std::vector<ConstantInfo>& constants() const { return constants_; }

 private:
  // Use GpuExecutable::Create() to create an instance.
  explicit GpuExecutable(Params params);

  // Constructor to use when loading a GpuExecutable from a BEF. Omits setting
  // class members that aren't used in BEF execution mode.
  GpuExecutable(std::shared_ptr<HloModule> hlo_module, GpuVersion gpu_version,
                xla::EntryFunctionAttributes entry_func_attrs,
                absl::string_view module_name, Shape xla_output_shape,
                std::vector<BufferAllocation> allocations,
                absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
                BefExecutable* bef_executable);

  // If `block_host_until_done` is false, execution will not block the host
  // until the kernels have completed. This is used as an optimization for
  // clients, such as Tensorflow, that use a single stream of execution for
  // computations, and allow host-side deallocation from the allocator before
  // GPU execution completes.
  Status ExecuteThunksOrBef(const ServiceExecutableRunOptions* run_options,
                            const BufferAllocations& buffer_allocations,
                            bool block_host_until_done);

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceMemoryBase>;

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
  StatusOr<const BufferAllocToDeviceMemoryMap*> ResolveConstantGlobals(
      stream_executor::Stream* stream);

  // GpuExecutable check with either AMD's ISA version, or Nvidia's major minor
  // version for compute capability, depending on the hardware.
  Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options);

  StatusOr<BufferAllocations> GenerateBufferAllocations(
      VariantArguments arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal);

  StatusOr<se::DeviceMemoryBase> BufferForAllocation(
      VariantArguments arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      const BufferAllocation& allocation,
      se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
      int64_t arg_idx);

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
  const std::vector<uint8_t> binary_;

  // The GPU version for compute compatibility check.
  GpuVersion gpu_version_;

  // The thunks to be invoked by this GpuExecutable. They are generated by the
  // IrEmitter.
  OwnedThunkSchedule thunks_;

  xla::EntryFunctionAttributes entry_func_attrs_;

  std::string module_name_;

  xla::Shape output_shape_;

  // Owns the buffer data at runtime. It provides information to allocate
  // memory for every output/temp buffers.
  const std::vector<BufferAllocation> allocations_;

  std::shared_ptr<BufferAssignmentProto> debug_buffer_assignment_;
  std::function<std::string()> verbose_buffer_assignment_string_dumper_;

  absl::Mutex module_handle_mutex_;
  // Cache of module handles. Required to keep loaded modules alive until this
  // executable is destroyed.
  std::map<stream_executor::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ ABSL_GUARDED_BY(module_handle_mutex_);
  // Cache of constant buffer allocation maps used by `ResolveConstantGlobals`.
  std::map<stream_executor::StreamExecutor*, BufferAllocToDeviceMemoryMap>
      module_globals_ ABSL_GUARDED_BY(module_handle_mutex_);

  std::vector<ConstantInfo> constants_;
  const absl::flat_hash_map<ShapeIndex, OutputInfo> output_info_;
  // Retains shared ownership of on-device constants that are managed by XLA and
  // potentially shared with other executables.
  std::vector<std::shared_ptr<se::DeviceMemoryBase>> shared_constants_;

  // Data for bef executable mode only, owned.
  BefExecutable* bef_executable_ = nullptr;

  // JitRt executable if the JitRt mode is on, owned.
  JitRtExecutable* jitrt_executable_ = nullptr;

  GpuExecutable(const GpuExecutable&) = delete;
  GpuExecutable& operator=(const GpuExecutable&) = delete;
};

StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
