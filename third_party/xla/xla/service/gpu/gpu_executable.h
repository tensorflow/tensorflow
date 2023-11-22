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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/non_atomically_upgradeable_rw_lock.h"
#include "xla/service/gpu/runtime/executable.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/shaped_buffer.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Returns whether GpuExecutable runs with Xla Runtime.
bool IsXlaRuntimeExecutableEnabled(const HloModuleConfig& config);

// GPU-targeting implementation of the XLA Executable interface.
//
// Launches the given GPU kernel via the StreamExecutor.
//
// This is an immutable data type after initialization, and thus thread safe.
class GpuExecutable : public Executable {
 public:
  using OwnedThunkSequence = std::unique_ptr<const ThunkSequence>;
  using OwnedGpuRuntimeProgram = std::unique_ptr<GpuRuntimeProgram>;

  struct ConstantInfo {
    std::string symbol_name;
    DenseDataIntermediate content;
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
    se::GpuComputeCapability gpu_version;
    // The GpuExecutable will either execute Thunks, XLA runtime executable
    // (native function) or experimental XLA runtime executable (IREE VM
    // function) depending on which is supplied.
    std::variant<OwnedThunkSequence, OwnedGpuRuntimeProgram> executable;
    std::vector<ConstantInfo> constants;
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
    std::string module_name;
    xla::Shape output_shape;
    std::optional<std::vector<BufferAllocation>> mlir_allocations;
    std::unique_ptr<const BufferAssignment> buffer_assignment;
    bool enable_persistent_temp_buffers;
    int64_t debug_buffer_assignment_show_max;
    std::unique_ptr<HloModule> debug_module = nullptr;
    bool enable_debug_info_manager = true;
  };

  // Analyze the entry function to construct buffer allocation and other output
  // information.
  //
  // TODO(ezhulenev): Once Xla runtime enabled by default, hide this method as
  // an implementation detail of GpuExecutable.
  static Status SetUpMlirAllocation(
      mlir::func::FuncOp func, llvm::ArrayRef<int64_t> buffer_sizes,
      std::vector<BufferAllocation>* allocations,
      absl::flat_hash_map<ShapeIndex, OutputInfo>* output_info,
      Shape* output_shape);

  // Returns an Executable that is loaded from an object file (XLA program
  // compiled to a native function using the XLA Runtime stack).
  static StatusOr<std::unique_ptr<Executable>> LoadFromObjFile(
      std::shared_ptr<HloModule> hlo_module, absl::string_view obj_file,
      absl::string_view mlir_module, DebugOptions debug_options,
      absl::string_view asm_text, absl::string_view binary,
      std::vector<ConstantInfo> constants, se::GpuComputeCapability gpu_version,
      stream_executor::StreamExecutor* executor);

  // Constructor to use when loading a GpuExecutable from an object file (native
  // function compiled for XLA Runtime). Omits setting class members that aren't
  // used in XLA Runtime execution mode.
  GpuExecutable(std::shared_ptr<HloModule> hlo_module, std::string asm_text,
                std::vector<uint8_t> binary,
                std::vector<ConstantInfo> constants,
                se::GpuComputeCapability gpu_version,
                absl::string_view module_name, Shape xla_output_shape,
                std::vector<BufferAllocation> allocations,
                absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
                std::unique_ptr<GpuRuntimeExecutable> runtime_executable);

  static StatusOr<std::unique_ptr<GpuExecutable>> Create(Params params);
  ~GpuExecutable() override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  // This should be called after set_ir_module_string.
  const std::string& ir_module_string() const { return ir_module_string_; }

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

  using VariantArguments = std::variant<absl::Span<const ShapedBuffer* const>,
                                        absl::Span<ExecutionInput>>;
  StatusOr<ExecutionOutput> ExecuteAsyncOnStreamImpl(
      const ServiceExecutableRunOptions* run_options,
      VariantArguments arguments);

  absl::Span<const BufferAllocation> GetAllocations() const {
    // A GpuExecutable can get its allocations in three ways:
    // 1 - From a regular compilation that uses allocations from MLIR.
    // 2 - From a regular compilation that uses the original allocations from
    //     the buffer assignment.
    // 3 - From loading the executable from an object file.
    //
    // In cases 1 and 3, the allocations are stored in allocations_ and in
    // case 2, they are part of the buffer_assignment.
    //
    // This function chooses the correct allocations to be used within the
    // GpuExecutable code.
    return allocations_.has_value() ? *allocations_
                                    : buffer_assignment_->Allocations();
  }

  const std::vector<ConstantInfo>& constants() const { return constants_; }

  StatusOr<std::string_view> GetObjFile() const;
  StatusOr<std::string_view> GetMlirModule() const;

  const BufferAssignment* buffer_assignment() const {
    return buffer_assignment_.get();
  }

 private:
  // Use GpuExecutable::Create() to create an instance.
  explicit GpuExecutable(Params params);

  // If `block_host_until_done` is false, execution will not block the host
  // until the kernels have completed. This is used as an optimization for
  // clients, such as Tensorflow, that use a single stream of execution for
  // computations, and allow host-side deallocation from the allocator before
  // GPU execution completes.
  Status ExecuteThunksOrXlaRuntime(
      const ServiceExecutableRunOptions* run_options,
      const BufferAllocations& buffer_allocations, bool block_host_until_done,
      NonAtomicallyUpgradeableRWLock& gpu_lock);

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

  // Allocate the temp buffers and store them with the GpuExecutable. This
  // function only allocates buffers on the first run for each executor.
  Status PopulatePersistentTempBuffers(se::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(persistent_temp_buffers_mu_);

  // GpuExecutable check with either AMD's ISA version, or Nvidia's major minor
  // version for compute capability, depending on the hardware.
  Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options);

  StatusOr<BufferAllocations> GenerateBufferAllocations(
      VariantArguments arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
      const BufferAllocToDeviceMemoryMap&
          buffer_alloc_to_persistent_memory_map);

  StatusOr<se::DeviceMemoryBase> BufferForAllocation(
      VariantArguments arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      const BufferAllocation& allocation,
      se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
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
#ifdef TENSORFLOW_USE_ROCM
  std::vector<uint8_t> binary_;
#else
  const std::vector<uint8_t> binary_;
#endif
  // The GPU version for compute compatibility check.
  se::GpuComputeCapability gpu_version_;

  // The thunks to be invoked by this GpuExecutable. They are generated by the
  // IrEmitter (null if XLA:GPU runtime is enabled).
  OwnedThunkSequence thunks_;

  // Gpu runtime executable that encapsulates all the state for running Gpu
  // runtime custom calls implementing gpu abstraction layer (available only if
  // Xla runtime is enabled).
  std::unique_ptr<GpuRuntimeExecutable> gpu_runtime_executable_;

  std::string module_name_;

  xla::Shape output_shape_;

  // The allocations_ object contains allocations that **may** be used to
  // provide information for allocating memory for every output/temp buffer.
  // See the comment on GetAllocations().
  std::optional<const std::vector<BufferAllocation>> allocations_;

  // The buffer_assignment_ object contains allocations that **may** be used to
  // provide information for allocating memory for every output/temp buffer.
  // See the comment on GetAllocations().
  //
  // This object is also used for dumping debug info.
  std::unique_ptr<const xla::BufferAssignment> buffer_assignment_;

  bool enable_persistent_temp_buffers_ = false;

  absl::Mutex persistent_temp_buffers_mu_;
  // Temp buffers can be allocated once and be reused whenever the GpuExecutable
  // is executed. The persistent temp buffer is stored in a map that maps from
  // a BufferAllocation to the temp buffer.
  absl::flat_hash_map<stream_executor::StreamExecutor*,
                      BufferAllocToDeviceMemoryMap>
      persistent_temp_buffers_ ABSL_GUARDED_BY(persistent_temp_buffers_mu_);

  int64_t debug_buffer_assignment_show_max_;

  absl::Mutex module_handle_mutex_;
  // Cache of module handles. Required to keep loaded modules alive until this
  // executable is destroyed.
  absl::flat_hash_map<stream_executor::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ ABSL_GUARDED_BY(module_handle_mutex_);
  // Cache of constant buffer allocation maps used by `ResolveConstantGlobals`.
  absl::flat_hash_map<stream_executor::StreamExecutor*,
                      std::unique_ptr<BufferAllocToDeviceMemoryMap>>
      module_globals_ ABSL_GUARDED_BY(module_handle_mutex_);

  std::vector<ConstantInfo> constants_;
  const absl::flat_hash_map<ShapeIndex, OutputInfo> output_info_;
  // Retains shared ownership of on-device constants that are managed by XLA and
  // potentially shared with other executables.
  std::vector<std::shared_ptr<se::DeviceMemoryBase>> shared_constants_;
  bool enable_debug_info_manager_;

  GpuExecutable(const GpuExecutable&) = delete;
  GpuExecutable& operator=(const GpuExecutable&) = delete;
};

StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
