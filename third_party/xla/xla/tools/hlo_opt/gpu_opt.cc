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

#include "absl/log/log.h"
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
#include "xla/service/gpu/transforms/collectives/all_gather_optimizer.h"
#include "xla/service/gpu/transforms/cudnn_custom_call_converter.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/dot_dimension_sorter.h"
#include "xla/service/gpu/transforms/dot_normalizer.h"
#include "xla/service/gpu/transforms/dot_operand_converter.h"
#include "xla/service/gpu/transforms/gemm_broadcast_folding_rewriter.h"
#include "xla/service/gpu/transforms/gemm_fusion.h"
#include "xla/service/gpu/transforms/gemv_rewriter.h"
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
#include "xla/service/spmd/schedule_aware_collective_ops_cse.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tools/hlo_opt/compiled_opt_lib.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

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

  //////////////////////////////////////////////////////////////////////////////
  // Registration of GPU-specific HLO Passes                                  //
  //////////////////////////////////////////////////////////////////////////////
  void RegisterProviderPasses(HloModule& module) override {
    auto device_description = GetDeviceDescription(&module);
    auto debug_config = module.config().debug_options();
    se::GpuComputeCapability gpu_compute_capability;
    if (device_description.ok()) {
      gpu_compute_capability = device_description->gpu_compute_capability();
    } else {
      LOG(WARNING)
          << "No compute capability specified, defaulting to Hopper. Use "
             "--xla_gpu_target_config_filename= to specify a target config.";
      gpu_compute_capability = stream_executor::CudaComputeCapability::Hopper();
    }
    // go/keep-sorted start
    RegisterPass<gpu::AllGatherOptimizer>();
    RegisterPass<gpu::CuDnnCustomCallConverter>();
    RegisterPass<gpu::DotAlgorithmRewriter>();
    RegisterPass<gpu::DotDimensionSorter>();
    RegisterPass<gpu::DotNormalizer>();
    RegisterPass<gpu::DotOperandConverter>();
    RegisterPass<gpu::GemmBroadcastFoldingRewriter>();
    RegisterPass<gpu::GemmFusion>(gpu_compute_capability);
    RegisterPass<gpu::GemvRewriter>();
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
    if (debug_config.xla_gpu_experimental_collective_cse_distance_threshold() >
        0) {
      RegisterPass<ScheduleAwareCollectiveOpsCSE>(
          debug_config.xla_gpu_experimental_collective_cse_distance_threshold(),
          false);
    }
  }

 private:
  absl::StatusOr<se::DeviceDescription> GetDeviceDescription(
      const HloModule* module) {
    Compiler::CompileOptions opts;
    TF_ASSIGN_OR_RETURN(
        Compiler::TargetConfig target_config,
        gpu::GpuCompiler::GetTargetConfig(
            opts, module->config().debug_options(), /*executor=*/nullptr));
    return target_config.device_description;
  }

  absl::StatusOr<std::string> LlvmIrBeforeOptimizations(
      HloModule* optimized_module) {
    TF_ASSIGN_OR_RETURN(se::DeviceDescription device_description,
                        GetDeviceDescription(optimized_module));
    TF_ASSIGN_OR_RETURN(se::Platform * platform,
                        PlatformUtil::GetPlatform(GetPlatformName()));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler,
                        Compiler::GetForPlatform(platform));

    auto* gpu_compiler = static_cast<gpu::GpuCompiler*>(compiler.get());
    if (!optimized_module->has_schedule()) {
      TF_ASSIGN_OR_RETURN(gpu::ScheduleMetadata schedule_metadata,
                          gpu::ScheduleGpuModule(optimized_module,
                                                 gpu_compiler->GetPointerSize(),
                                                 device_description));
      TF_RETURN_IF_ERROR(gpu_compiler->RunPostSchedulingPipelines(
          optimized_module, schedule_metadata.scheduler_mem_limit,
          device_description));
    }

    llvm::LLVMContext llvm_context;
    TF_ASSIGN_OR_RETURN(
        xla::gpu::CompileModuleResults results,
        xla::gpu::CompileModuleToLlvmIr(
            optimized_module, &llvm_context, gpu_compiler->GetTargetTriple(),
            gpu_compiler->GetDataLayout(), platform, device_description,
            gpu_compiler->GetCanShareBuffer(device_description),
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
