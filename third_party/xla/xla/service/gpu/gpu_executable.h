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
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/runtime/annotation.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// GPU-targeting implementation of the XLA Executable interface.
//
// Launches the given GPU kernel via the StreamExecutor.
//
// This is an immutable data type after initialization, and thus thread safe.
class GpuExecutable : public Executable {
 public:
  using OwnedThunkSequence = std::unique_ptr<const ThunkSequence>;

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
    Thunk::BinaryMap dnn_compiled_graphs;
    se::GpuComputeCapability gpu_version;
    OwnedThunkSequence executable;
    std::vector<ConstantInfo> constants;
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
    std::string module_name;
    xla::Shape output_shape;
    std::optional<std::vector<BufferAllocation>> mlir_allocations;
    std::unique_ptr<const BufferAssignment> buffer_assignment;
    int64_t debug_buffer_assignment_show_max;
    std::unique_ptr<HloModule> debug_module = nullptr;
    bool enable_debug_info_manager = true;
  };

  static absl::StatusOr<std::unique_ptr<GpuExecutable>> Create(Params params);
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

  const Thunk::BinaryMap& dnn_compiled_graphs() const {
    return dnn_compiled_graphs_;
  }

  // ExecuteAsyncOnStream will fail if the compute capability of the stream
  // doesn't match the compute capability passed to this object's constructor.
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  absl::StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  using VariantArguments = std::variant<absl::Span<const ShapedBuffer* const>,
                                        absl::Span<ExecutionInput>>;
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStreamImpl(
      const ServiceExecutableRunOptions* run_options,
      VariantArguments arguments);

  absl::Span<const BufferAllocation> GetAllocations() const override {
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

  const BufferAssignment* buffer_assignment() const {
    return buffer_assignment_.get();
  }

 private:
  // Use GpuExecutable::Create() to create an instance.
  explicit GpuExecutable(Params params);

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
  absl::StatusOr<const BufferAllocToDeviceMemoryMap*> ResolveConstantGlobals(
      stream_executor::Stream* stream);

  // GpuExecutable check with either AMD's ISA version, or Nvidia's major minor
  // version for compute capability, depending on the hardware.
  absl::Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options);

  absl::StatusOr<BufferAllocations> GenerateBufferAllocations(
      VariantArguments arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      se::DeviceMemoryAllocator* memory_allocator, int device_ordinal);

  absl::StatusOr<se::DeviceMemoryBase> BufferForAllocation(
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
  std::vector<uint8_t> binary_;

  Thunk::BinaryMap dnn_compiled_graphs_;

  // The GPU version for compute compatibility check.
  se::GpuComputeCapability gpu_version_;

  // The thunks to be invoked by this GpuExecutable. They are generated by the
  // IrEmitter (null if XLA:GPU runtime is enabled).
  OwnedThunkSequence thunks_;

  // Additional execution streams requested by `thunks_`.
  absl::flat_hash_set<ExecutionStreamId> execution_stream_ids_;

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
  absl::flat_hash_map<stream_executor::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ ABSL_GUARDED_BY(module_handle_mutex_);
  // Cache of constant buffer allocation maps used by `ResolveConstantGlobals`.
  absl::flat_hash_map<stream_executor::StreamExecutor*,
                      std::unique_ptr<BufferAllocToDeviceMemoryMap>>
      module_globals_ ABSL_GUARDED_BY(module_handle_mutex_);

  // Cache previous memory allocations for current module, this is used to help
  // identify if user's model have unstable pointers by turning on VLOG(5).
  absl::flat_hash_map<stream_executor::StreamExecutor*,
                      std::vector<se::DeviceMemoryBase>>
      module_allocations_ ABSL_GUARDED_BY(module_handle_mutex_);

  std::vector<ConstantInfo> constants_;
  const absl::flat_hash_map<ShapeIndex, OutputInfo> output_info_;
  // Retains shared ownership of on-device constants that are managed by XLA and
  // potentially shared with other executables.
  std::vector<std::shared_ptr<se::DeviceMemoryBase>> shared_constants_;
  bool enable_debug_info_manager_;

  GpuExecutable(const GpuExecutable&) = delete;
  GpuExecutable& operator=(const GpuExecutable&) = delete;
};

absl::StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
