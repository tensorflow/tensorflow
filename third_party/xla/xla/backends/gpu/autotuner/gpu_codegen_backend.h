/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_GPU_CODEGEN_BACKEND_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_GPU_CODEGEN_BACKEND_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace  gpu {

// Abstract base class for GPU backends, implementing the Backend interface.
class GpuCodegenBackend : public CodegenBackend {
 public:
  // target_config, debug_options and compiler should outlive the backend.
  // TODO(b/447096292): Remove stream_executor from GpuCodegenBackend.
  GpuCodegenBackend(absl::string_view name, const DebugOptions* debug_options,
                    Compiler* compiler,
                    const Compiler::TargetConfig* target_config,
                    stream_executor::StreamExecutor* stream_executor = nullptr)
      : name_(name),
        stream_executor_(stream_executor),
        target_config_(*target_config),
        debug_options_(*debug_options),
        compiler_(compiler) {}

  absl::string_view name() const override { return name_; }

  const Compiler::TargetConfig& target_config() const { return target_config_; }
  const DebugOptions& debug_options() const { return debug_options_; }
  stream_executor::StreamExecutor* stream_executor() {
    return stream_executor_;
  }

  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& hlo_instruction,
      const BackendConfig& config) override {
    std::unique_ptr<HloModule> hlo_module =
        ExtractInstructionIntoNewModule(hlo_instruction);

    HloComputation* entry_computation = hlo_module->entry_computation();
    HloInstruction* root_instruction = entry_computation->root_instruction();
    TF_RETURN_IF_ERROR(ApplyConfig(*root_instruction, config));

    hlo_module->mutable_config().set_debug_options(debug_options_);
    AdjustDebugOptionsForAutotuning(
        hlo_module->mutable_config().mutable_debug_options(),
        allow_register_spills_);

    Compiler::CompileOptions options;
    options.target_config = target_config_;
    options.is_autotuning_compilation = true;
    TF_ASSIGN_OR_RETURN(auto optimized_module,
                        RunHloPasses(std::move(hlo_module), options));
    return compiler_->RunBackend(std::move(optimized_module), stream_executor_,
                                 options);
  }

  bool CanProduceWrongResults() const override { return false; }
  // TODO b/443207721 - Remove this once we have a better way to handle register
  // spilling during autotuning.
  // Allows compilation to succeed even if kernels spill registers,
  // ignoring the `xla_gpu_filter_kernels_spilling_registers_on_autotuning`
  // flag. If not called, the flag's value is honored.
  void AllowRegisterSpills() { allow_register_spills_ = true; }

  static void AdjustDebugOptionsForAutotuning(
      DebugOptions& debug_options, bool force_allow_register_spills) {
    debug_options.set_xla_enable_dumping(false);
    // Avoid using another thread pool.
    debug_options.set_xla_gpu_force_compilation_parallelism(1);
    debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(false);
    // Avoid using GPU graphs as we don't want to measure graph construction
    // time.
    debug_options.clear_xla_gpu_enable_command_buffer();
    // Avoid using async dot as we don't want to measure event overheads.
    debug_options.set_xla_gpu_async_dot(false);
    debug_options.set_xla_embed_ir_in_executable(false);
    debug_options.set_xla_gpu_kernel_cache_file("");
    if (force_allow_register_spills) {
      debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
          false);
    }
  }

 private:
  // Optimize the HLO module.
  // TODO(b/407494653): Remove this when XLA pipelines is fixed and we autotune
  // only optimized and fused HLOs.
  virtual absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module,
      const Compiler::CompileOptions& options) {
    // No additional passes needed for most backends.
    return hlo_module;
  };

  std::string name_;
  stream_executor::StreamExecutor* stream_executor_;
  const Compiler::TargetConfig& target_config_;
  const DebugOptions& debug_options_;
  // TODO(b/407494653): remove compiler when we don't need to run any HLO passes
  // and the codegen backend can directly produce an executable without a
  // compiler instance.
  Compiler* compiler_;
  bool allow_register_spills_ = false;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_GPU_CODEGEN_BACKEND_H_
