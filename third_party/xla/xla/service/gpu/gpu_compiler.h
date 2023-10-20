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

#ifndef XLA_SERVICE_GPU_GPU_COMPILER_H_
#define XLA_SERVICE_GPU_GPU_COMPILER_H_

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/autotune_results.pb.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/buffer_sharing.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/llvm_compiler.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

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

// The GPU compiler generates efficient GPU executables.
class GpuCompiler : public LLVMCompiler {
 public:
  GpuCompiler(se::Platform::Id platform_id, const char* target_triple,
              const char* data_layout);

  using LLVMCompiler::Compile;

  // An attached device is passed in via stream_exec. We get GPU configuration
  // from the attached device. GemmAlgorithmPicker and GpuConvAlgorithmPicker
  // can run on the attached device.
  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      HloModule* hlo_module, se::StreamExecutor* stream_exec) override;

  se::GpuComputeCapability GetGpuVersion(se::StreamExecutor* stream_exec);

  TargetConfig GetGpuTargetConfig(se::StreamExecutor* stream_exec) {
    return TargetConfig(stream_exec);
  }

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     AotCompilationOptions const& options) override;

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileToTargetBinary(
      const HloModuleConfig& module_config,
      std::unique_ptr<llvm::Module> llvm_module,
      se::GpuComputeCapability gpu_version, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const HloModule* debug_module);

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

  Status RunPostSchedulingPipelines(HloModule* module,
                                    int64_t scheduler_mem_limit) const;

 protected:
  // During compilation with device, stream_exec != null and autotune_results
  // == null. During deviceless AOT compilation, stream_exec == null and
  // autotune_results != null.
  // thread_pool is used to speed up compilation during autotuning.
  virtual Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const TargetConfig& gpu_target_config,
      tsl::thread::ThreadPool* thread_pool);

  // CollectivesScheduleLinearizer enforces a total ordering between collectives
  // to work around divergence in executables introduced due to auto tuning,
  // specifically the use of extra scratch space for convolutions. This
  // function decided whether to apply this pass. If convolutions are present in
  // the code and we are using "online" autotuning (i.e., not AOT) we need to
  // use the pass, else we do not need to enable the pass.
  virtual bool RequiresCollectiveScheduleLinearizer(
      const HloModule* module, se::StreamExecutor* stream_exec) {
    return false;
  }

  // Add autotuning passes for convolution and gemm (except triton).
  virtual Status AddConvAndGemmAutotuningPasses(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool) {
    return OkStatus();
  }

  // Add autotuning passes for triton gemm.
  virtual Status AddTritonGemmAutotuningPasses(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool) {
    return OkStatus();
  }

 private:
  Status LoadAutotuneResultsFromFile(const DebugOptions& debug_options);
  Status SerializeAutotuneResultsToFile(const DebugOptions& debug_options);

  // During compilation with device, stream_exec != null and autotune_results
  // == null. During deviceless AOT compilation, stream_exec == null and
  // autotune_results != null.
  Status OptimizeHloModule(HloModule* hlo_module,
                           se::StreamExecutor* stream_exec,
                           const CompileOptions& options,
                           const TargetConfig& gpu_target_config);

  virtual Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::GpuComputeCapability gpu_version,
      se::dnn::VersionInfo dnn_version,
      se::DeviceMemoryAllocator* device_allocator) = 0;

  virtual HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() const {
    return &FusionCanShareBufferHint;
  }

  // TODO(timshen): Replace `debug_module` with some portable debug information
  // that accommodates both HLO and MLIR.
  virtual StatusOr<std::pair<std::string, std::vector<uint8_t>>>
  CompileTargetBinary(const HloModuleConfig& module_config,
                      llvm::Module* llvm_module,
                      se::GpuComputeCapability gpu_version, bool relocatable,
                      const HloModule* debug_module,
                      const CompileOptions& options) = 0;

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

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_COMPILER_H_
