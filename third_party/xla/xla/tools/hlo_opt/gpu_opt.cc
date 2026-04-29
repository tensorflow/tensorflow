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

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/IR/LLVMContext.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/collectives/all_gather_optimizer.h"
#include "xla/backends/gpu/transforms/cudnn_custom_call_converter.h"
#include "xla/backends/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/backends/gpu/transforms/dot_dimension_sorter.h"
#include "xla/backends/gpu/transforms/dot_normalizer.h"
#include "xla/backends/gpu/transforms/dot_operand_converter.h"
#include "xla/backends/gpu/transforms/gemm_broadcast_folding_rewriter.h"
#include "xla/backends/gpu/transforms/gemm_fusion.h"
#include "xla/backends/gpu/transforms/gemv_rewriter.h"
#include "xla/backends/gpu/transforms/reduce_scatter_creator.h"
#include "xla/backends/gpu/transforms/reduction_degenerate_dim_remover.h"
#include "xla/backends/gpu/transforms/reduction_dimension_grouper.h"
#include "xla/backends/gpu/transforms/reduction_layout_normalizer.h"
#include "xla/backends/gpu/transforms/sanitize_constant_names.h"
#include "xla/backends/gpu/transforms/topk_specializer.h"
#include "xla/backends/gpu/transforms/topk_splitter.h"
#include "xla/backends/gpu/transforms/windowed_einsum_handler.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_opt/opt_lib.h"
#include "xla/hlo/transforms/host_offloader.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/layout.h"
#include "xla/service/buffer_value.h"
#include "xla/service/compiler.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/nvptx_alias_info.h"
#include "xla/service/llvm_compiler.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/spmd/schedule_aware_collective_ops_cse.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tools/hlo_opt/compiled_opt_lib.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class GpuOptProvider : public CompiledOptProvider {
 public:
  GpuOptProvider() : CompiledOptProvider() {}

  absl::StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view s) override {
    if (s == "llvm-before-optimizations") {
      ASSIGN_OR_RETURN(std::string llvm_ir,
                       LlvmIrFor(std::move(module), false));
      return llvm_ir;
    }
    if (s == "llvm" || s == "llvm-after-optimizations") {
      ASSIGN_OR_RETURN(std::string llvm_ir, LlvmIrFor(std::move(module), true));
      return llvm_ir;
    }
    if (s == "ptx") {
      ASSIGN_OR_RETURN(std::string ptx, PtxFor(std::move(module)));
      return ptx;
    }
    if (s == "buffer-assignment") {
      ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                       GetExecutable(std::move(module)));
      auto gpu_executable = static_cast<gpu::GpuExecutable*>(executable.get());
      return gpu_executable->buffer_assignment()->ToVerboseString(
          gpu_executable->alias_info(), 9999);
    }
    {
      // Delegate to base class.
      ASSIGN_OR_RETURN(
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
      if (gpu_compute_capability.IsCuda()) {
        alias_info_ =
            std::make_unique<gpu::NVPTXAliasInfo>(*device_description);
      } else {
        alias_info_ = std::make_unique<gpu::GpuAliasInfo>(*device_description);
      }
    } else {
      LOG(WARNING)
          << "No compute capability specified, defaulting to Hopper. Use "
             "--xla_gpu_target_config_filename= to specify a target config.";
      gpu_compute_capability = stream_executor::CudaComputeCapability::Hopper();
    }
    static BufferValue::SizeFunction* const kSizeFunction =
        new BufferValue::SizeFunction([](const BufferValue& buffer) {
          const Shape& shape = buffer.shape();
          if (shape.has_layout() &&
              shape.layout().memory_space() == Layout::kHostMemorySpace) {
            return static_cast<int64_t>(0);
          }
          return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
        });
    // go/keep-sorted start
    RegisterPass<CopyInsertion>(alias_info_.get());
    RegisterPass<HloMemoryScheduler>(alias_info_.get(), kSizeFunction);
    RegisterPass<HostOffloader>(alias_info_.get());
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
    RegisterPass<gpu::TopkSpecializer>(gpu_compute_capability);
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
    ASSIGN_OR_RETURN(
        gpu::GpuTargetConfig target_config,
        gpu::GetTargetConfigFromFile(
            module->config().debug_options().xla_gpu_target_config_filename()));
    return target_config.device_description;
  }

  absl::StatusOr<std::string> LlvmIrFor(std::unique_ptr<HloModule> input_module,
                                        bool optimized) {
    Compiler::CompileOptions opts;
    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                     GetOptimizedHlo(std::move(input_module)));
    ASSIGN_OR_RETURN(se::StreamExecutor * executor, GetExecutor());
    ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler, GetCompiler());

    LLVMCompiler* llvm_compiler =
        absl::down_cast<LLVMCompiler*>(compiler.get());

    llvm::LLVMContext context;
    std::vector<std::unique_ptr<llvm::Module>> modules;
    if (optimized) {
      llvm_compiler->SetPostOptimizationHook([&](const llvm::Module& module) {
        modules.push_back(gpu::CopyToContext(module, context));
      });
    } else {
      llvm_compiler->SetPreOptimizationHook([&](const llvm::Module& module) {
        modules.push_back(gpu::CopyToContext(module, context));
      });
    }

    ASSIGN_OR_RETURN(
        std::unique_ptr<Executable> executable,
        compiler->RunBackend(std::move(optimized_module), executor, opts));

    gpu::LinkLlvmModulesInPlace(modules);

    return llvm_ir::DumpToString(modules[0].get());
  }

  absl::StatusOr<std::string> PtxFor(std::unique_ptr<HloModule> input_module) {
    Compiler::CompileOptions opts;
    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                     GetOptimizedHlo(std::move(input_module)));
    ASSIGN_OR_RETURN(se::StreamExecutor * executor, GetExecutor());
    ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler, GetCompiler());

    gpu::GpuCompiler* gpu_compiler =
        absl::down_cast<gpu::GpuCompiler*>(compiler.get());

    std::string ptx_str = "// GPU Executable\n";
    gpu_compiler->SetAsmHook([&](absl::string_view ptx) { ptx_str += ptx; });

    ASSIGN_OR_RETURN(
        std::unique_ptr<Executable> executable,
        compiler->RunBackend(std::move(optimized_module), executor, opts));

    return ptx_str;
  }
};

}  // namespace
}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(gpu_opt_provider, {
  xla::OptProvider::RegisterForPlatform(
      "gpu", std::make_unique<xla::GpuOptProvider>());
});
