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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace xla {
namespace gpu {

class GpuAotCompilationResult : public AotCompilationResult {
 public:
  static StatusOr<std::unique_ptr<GpuAotCompilationResult>> FromString(
      const std::string& serialized) {
    GpuBefExecutableProto bef_executable;
    if (!bef_executable.ParseFromString(serialized)) {
      return InternalError("Failed to parse serialized GpuBefExecutableProto.");
    }
    return std::unique_ptr<GpuAotCompilationResult>(
        new GpuAotCompilationResult(std::move(bef_executable)));
  }

  GpuAotCompilationResult(HloModuleProto hlo_module_proto,
                          const std::string& bef,
                          EntryFunctionAttributes entry_func_attrs) {
    *bef_executable_.mutable_hlo_module_proto() = hlo_module_proto;
    bef_executable_.set_bef(bef);
    *bef_executable_.mutable_entry_func_attrs() = entry_func_attrs;
  }
  ~GpuAotCompilationResult() override = default;

  StatusOr<std::string> SerializeAsString() const override {
    return bef_executable_.SerializeAsString();
  }

  StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, se::StreamExecutor* executor) const override;

  const GpuBefExecutableProto& bef_executable() const {
    return bef_executable_;
  }

 private:
  explicit GpuAotCompilationResult(GpuBefExecutableProto bef_executable)
      : bef_executable_(std::move(bef_executable)) {}

  GpuBefExecutableProto bef_executable_;
};

// The GPU compiler generates efficient GPU executables.
class GpuCompiler : public LLVMCompiler {
 public:
  GpuCompiler(se::Platform::Id platform_id, const char* target_triple,
              const char* data_layout);
  ~GpuCompiler() override {}

  using LLVMCompiler::Compile;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      const HloModule* hlo_module) override;

  virtual GpuVersion GetGpuVersion(se::StreamExecutor* stream_exec) = 0;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<AotCompilationResult>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     AotCompilationOptions const& options) override;

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileToTargetBinary(
      const HloModuleConfig& module_config,
      std::unique_ptr<llvm::Module> llvm_module,
      se::StreamExecutor* stream_exec, const CompileOptions& options,
      const HloModule* debug_module);

  se::Platform::Id PlatformId() const override { return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

 protected:
  virtual Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator);

 private:
  Status OptimizeHloModule(HloModule* hlo_module,
                           se::StreamExecutor* stream_exec,
                           se::DeviceMemoryAllocator* device_allocator);

  virtual Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) = 0;

  virtual HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() {
    return
        [](const HloInstruction*, const HloInstruction*,
           const ShapeIndex&) -> std::optional<bool> { return std::nullopt; };
  }

  // TODO(timshen): Replace `debug_module` with some portable debug information
  // that accommodates both HLO and MLIR.
  virtual StatusOr<std::pair<std::string, std::vector<uint8_t>>>
  CompileTargetBinary(const HloModuleConfig& module_config,
                      llvm::Module* llvm_module, GpuVersion gpu_version,
                      se::StreamExecutor* stream_exec, bool relocatable,
                      const HloModule* debug_module) = 0;

  Status PrepareHloModuleForIrEmitting(HloModule* hlo_module);

  virtual StatusOr<std::vector<uint8_t>> LinkModules(
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules) {
    return Unimplemented("LinkModules is not implemented.");
  }

  se::Platform::Id platform_id_;

  // The triple that represents our target.
  const char* target_triple_;

  // The data layout of the emitted module.
  const char* data_layout_;

  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64_t pointer_size_;

  GpuCompiler(const GpuCompiler&) = delete;
  GpuCompiler& operator=(const GpuCompiler&) = delete;
};

GpuDeviceInfo GetGpuDeviceInfo(se::StreamExecutor* stream_exec);

// Compile `hlo_module` using XLA GPU and return the LLVM module thus generated.
// The GpuExecutable (and the Thunks that are part of it) are not returned.
StatusOr<std::unique_ptr<llvm::Module>> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, const se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability, int pointer_size);

// Compiles the given LMHLO module to an executable.
// ir_emitter_context should be partially populated: buffer_assignment
// or buffer_allocations should not be populated, while other fields should be
// populated (or left empty if that field is optional).
//
// NOTE: buffer_assignment will be gone from ir_emitter_context once LMHLO
// transition is done.
StatusOr<std::unique_ptr<Executable>> CompileLmhloToExecutable(
    GpuCompiler* compiler, mlir::ModuleOp module, std::string module_name,
    const HloModuleConfig& module_config,
    const Compiler::CompileOptions& options,
    absl::string_view entry_function_name, se::StreamExecutor* stream_exec,
    std::unique_ptr<llvm::Module> llvm_module,
    IrEmitterContext* ir_emitter_context);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
