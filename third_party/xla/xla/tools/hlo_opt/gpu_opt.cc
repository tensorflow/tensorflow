/* Copyright 2023 The OpenXLA Authors.

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

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/transforms/all_gather_optimizer.h"
#include "xla/service/gpu/transforms/cudnn_custom_call_converter.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/dot_dimension_sorter.h"
#include "xla/service/gpu/transforms/dot_normalizer.h"
#include "xla/service/gpu/transforms/dot_operand_converter.h"
#include "xla/service/gpu/transforms/gemm_broadcast_folding_rewriter.h"
#include "xla/service/gpu/transforms/gemv_rewriter.h"
#include "xla/service/gpu/transforms/pipelined_p2p_rewriter.h"
#include "xla/service/gpu/transforms/reduce_scatter_creator.h"
#include "xla/service/gpu/transforms/reduction_degenerate_dim_remover.h"
#include "xla/service/gpu/transforms/reduction_dimension_grouper.h"
#include "xla/service/gpu/transforms/reduction_layout_normalizer.h"
#include "xla/service/gpu/transforms/sanitize_constant_names.h"
#include "xla/service/gpu/transforms/topk_specializer.h"
#include "xla/service/gpu/transforms/topk_splitter.h"
#include "xla/service/gpu/transforms/transpose_dimension_grouper.h"
#include "xla/service/gpu/transforms/windowed_einsum_handler.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tools/hlo_opt/compiled_opt_lib.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

class GpuOptProvider : public CompiledOptProvider {
 public:
  GpuOptProvider() : CompiledOptProvider() {}

  absl::StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view s) override {
    if (s == "llvm-before-optimizations") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedHlo(std::move(module)));
      TF_ASSIGN_OR_RETURN(std::string llvm_ir,
                          LlvmIrBeforeOptimizations(optimized_module.get()));
      return llvm_ir;

    } else if (s == "llvm" || s == "llvm-after-optimizations") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          GetExecutable(std::move(module)));
      return static_cast<gpu::GpuExecutable*>(executable.get())
          ->ir_module_string();
    } else if (s == "ptx") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          GetExecutable(std::move(module)));
      return static_cast<gpu::GpuExecutable*>(executable.get())->text();
    } else if (s == "buffer-assignment") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          GetExecutable(std::move(module)));
      return static_cast<gpu::GpuExecutable*>(executable.get())
          ->buffer_assignment()
          ->ToVerboseString(9999);
    } else {
      // Delegate to base class.
      TF_ASSIGN_OR_RETURN(
          std::optional<std::string> out,
          CompiledOptProvider::GenerateStage(std::move(module), s));
      return out;
    }
  }

  std::string GetPlatformName() override { return "gpu"; }

  std::set<std::string> SupportedStages() override {
    std::set<std::string> supported = CompiledOptProvider::SupportedStages();
    supported.insert({"ptx", "llvm", "buffer-assignment",
                      "llvm-before-optimizations", "llvm-after-optimizations"});
    return supported;
  }

  std::string GetRegisteredPassNames() override {
    return GetRegisteredPassNamesHelper(pass_registry_);
  }

  // Register only GPU specific passes here.
  void RegisterProviderPasses(HloModule& module) override {
    // go/keep-sorted start
    RegisterPass<gpu::AllGatherOptimizer>();
    RegisterPass<gpu::CuDnnCustomCallConverter>();
    RegisterPass<gpu::DotAlgorithmRewriter>();
    RegisterPass<gpu::DotDimensionSorter>();
    RegisterPass<gpu::DotNormalizer>();
    RegisterPass<gpu::DotOperandConverter>();
    RegisterPass<gpu::GemmBroadcastFoldingRewriter>();
    RegisterPass<gpu::GemvRewriter>();
    RegisterPass<gpu::PipelinedP2PRewriter>();
    RegisterPass<gpu::ReduceScatterCreator>();
    RegisterPass<gpu::ReductionDegenerateDimRemover>();
    RegisterPass<gpu::ReductionDimensionGrouper>();
    RegisterPass<gpu::ReductionLayoutNormalizer>();
    RegisterPass<gpu::SanitizeConstantNames>();
    RegisterPass<gpu::TopKSplitter>();
    RegisterPass<gpu::TopkSpecializer>();
    RegisterPass<gpu::TransposeDimensionGrouper>();
    RegisterPass<gpu::WindowedEinsumHandler>();
    // go/keep-sorted end
  }

 private:
  absl::StatusOr<std::string> LlvmIrBeforeOptimizations(
      HloModule* optimized_module) {
    Compiler::CompileOptions opts;
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor, GetExecutor());
    TF_ASSIGN_OR_RETURN(
        Compiler::TargetConfig target_config,
        gpu::GpuCompiler::GetTargetConfig(
            opts, optimized_module->config().debug_options(), executor));

    TF_ASSIGN_OR_RETURN(se::Platform * platform,
                        PlatformUtil::GetPlatform(GetPlatformName()));
    TF_ASSIGN_OR_RETURN(Compiler * compiler,
                        Compiler::GetForPlatform(platform));

    auto* gpu_compiler = static_cast<gpu::GpuCompiler*>(compiler);
    if (!optimized_module->has_schedule()) {
      TF_ASSIGN_OR_RETURN(gpu::ScheduleMetadata schedule_metadata,
                          gpu::ScheduleGpuModule(
                              optimized_module, gpu_compiler->GetPointerSize(),
                              target_config.device_description));
      TF_RETURN_IF_ERROR(gpu_compiler->RunPostSchedulingPipelines(
          optimized_module, schedule_metadata.scheduler_mem_limit,
          target_config.device_description));
    }

    llvm::LLVMContext llvm_context;
    TF_ASSIGN_OR_RETURN(
        xla::gpu::CompileModuleResults results,
        xla::gpu::CompileModuleToLlvmIr(
            optimized_module, &llvm_context, gpu_compiler->GetTargetTriple(),
            gpu_compiler->GetDataLayout(), platform->Name(), platform->id(),
            target_config.device_description,
            gpu_compiler->GetCanShareBuffer(target_config.device_description),
            gpu_compiler->BufferSizeBytesFunction()));
    return llvm_ir::DumpToString(results.llvm_module.get());
  }
};

}  // namespace
}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(gpu_opt_provider, {
  xla::OptProvider::RegisterForPlatform(
      "gpu", std::make_unique<xla::GpuOptProvider>());
});
