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
#include <string_view>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/executable.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.pb.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

// TODO(b/232263665): It should be shared between GPU and CPU.
class GpuXlaRuntimeAotCompilationResult : public AotCompilationResult {
 public:
  GpuXlaRuntimeAotCompilationResult(
      HloModuleProto hlo, std::string_view obj_file,
      std::string_view mlir_module, EntryFunctionAttributes entry_func_attrs,
      std::string_view gpu_asm_text, absl::Span<const uint8_t> gpu_binary,
      absl::Span<const GpuExecutable::ConstantInfo> constants = {}) {
    XlaRuntimeExecutableProto xla_runtime_executable;
    *xla_runtime_executable.mutable_hlo_module_proto() = hlo;
    xla_runtime_executable.set_obj_file(std::string(obj_file));
    xla_runtime_executable.set_mlir_module(std::string(mlir_module));
    *xla_runtime_gpu_executable_.mutable_xla_runtime_executable() =
        xla_runtime_executable;

    *xla_runtime_gpu_executable_.mutable_entry_func_attrs() = entry_func_attrs;
    xla_runtime_gpu_executable_.set_gpu_asm_text(std::string(gpu_asm_text));
    xla_runtime_gpu_executable_.set_gpu_binary(gpu_binary.data(),
                                               gpu_binary.size());

    for (const GpuExecutable::ConstantInfo& cst : constants) {
      auto* cst_proto = xla_runtime_gpu_executable_.add_constants();
      cst_proto->set_symbol_name(cst.symbol_name);
      cst_proto->set_allocation_index(cst.allocation_index);
      cst_proto->set_content(cst.content.data(), cst.content.size());
    }
  }

  explicit GpuXlaRuntimeAotCompilationResult(
      XlaRuntimeGpuExecutableProto executable)
      : xla_runtime_gpu_executable_(executable) {}

  StatusOr<std::string> SerializeAsString() const override {
    return xla_runtime_gpu_executable_.SerializeAsString();
  }

  static StatusOr<std::unique_ptr<GpuXlaRuntimeAotCompilationResult>>
  FromString(const std::string& serialized) {
    XlaRuntimeGpuExecutableProto xla_runtime_gpu_executable;
    if (!xla_runtime_gpu_executable.ParseFromString(serialized)) {
      return InternalError("Failed to parse serialized JitRtExecutableProto.");
    }
    return std::make_unique<GpuXlaRuntimeAotCompilationResult>(
        xla_runtime_gpu_executable);
  }

  StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, se::StreamExecutor* executor) const override;

 private:
  XlaRuntimeGpuExecutableProto xla_runtime_gpu_executable_;
};

struct GpuTargetConfig {
  GpuTargetConfig() = default;
  explicit GpuTargetConfig(const stream_executor::GpuTargetConfigProto& proto)
      : gpu_device_info(proto.gpu_device_info()),
        cuda_compute_capability(proto.cuda_compute_capability()),
        rocm_compute_capability(proto.rocm_compute_capability()),
        platform_name(proto.platform_name()) {}

  stream_executor::GpuTargetConfigProto ToProto() const {
    stream_executor::GpuTargetConfigProto proto;
    *proto.mutable_gpu_device_info() = gpu_device_info.ToProto();
    *proto.mutable_cuda_compute_capability() =
        cuda_compute_capability.ToProto();
    *proto.mutable_rocm_compute_capability() =
        rocm_compute_capability.ToProto();
    proto.set_platform_name(platform_name);
    return proto;
  }

  GpuDeviceInfo gpu_device_info;
  // CUDA "CC" major value, -1 if not available.
  stream_executor::CudaComputeCapability cuda_compute_capability{-1, -1};
  // ROCm gfx arch,  "gfx000" if not available.
  stream_executor::RocmComputeCapability rocm_compute_capability{"gfx000"};
  std::string platform_name;
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

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     AotCompilationOptions const& options) override;

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileToTargetBinary(
      const HloModuleConfig& module_config,
      std::unique_ptr<llvm::Module> llvm_module, GpuVersion gpu_version,
      se::StreamExecutor* stream_exec, const CompileOptions& options,
      const HloModule* debug_module);

  se::Platform::Id PlatformId() const override { return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  StatusOr<std::unique_ptr<AotCompilationResult>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override {
    return GpuXlaRuntimeAotCompilationResult::FromString(serialized_aot_result);
  }

  StatusOr<std::unique_ptr<AotCompilationResult>> Export(
      Executable* executable) const override;

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
                      bool relocatable, const HloModule* debug_module) = 0;

  Status PrepareHloModuleForIrEmitting(HloModule* hlo_module);

  virtual StatusOr<bool> CanUseLinkModules(const HloModuleConfig& config) {
    return false;
  }

  virtual StatusOr<std::vector<uint8_t>> LinkModules(
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options) {
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
