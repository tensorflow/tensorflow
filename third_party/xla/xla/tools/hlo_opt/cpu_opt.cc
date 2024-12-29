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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/transforms/expanders/rng_bit_generator_expander.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_rewriter.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/change_op_data_type.h"
#include "xla/service/cpu/conv_canonicalization.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/service/cpu/cpu_layout_assignment.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/parallel_task_assignment.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/executable.h"
#include "xla/service/gather_expander.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/select_and_scatter_expander.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tools/hlo_opt/compiled_opt_lib.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

class CpuOptProvider : public CompiledOptProvider {
 public:
  CpuOptProvider() : CompiledOptProvider() {}

  absl::StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view s) override {
    if (s == "llvm-before-optimizations") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          GetExecutable(std::move(module)));
      return static_cast<cpu::CpuExecutable*>(executable.get())
          ->ir_module_string();
    }
    return CompiledOptProvider::GenerateStage(std::move(module), s);
  }

  std::set<std::string> SupportedStages() override {
    std::set<std::string> supported = CompiledOptProvider::SupportedStages();
    supported.insert({"llvm-before-optimizations"});
    return supported;
  }

  std::string GetPlatformName() override { return "cpu"; }

  std::string GetRegisteredPassNames() override {
    return GetRegisteredPassNamesHelper(pass_registry_);
  }

  // Register only CPU specific passes here.
  void RegisterProviderPasses(HloModule& module) override {
    // initialize all needed to extract configs for pass registration
    // and pass it to the register function
    DebugOptions debug_opts = GetDebugOptionsFromFlags();
    auto executor = GetExecutor();
    HloModuleConfig module_config = module.config();
    absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> jit_target_machine =
        cpu::JitCompiler::InferTargetMachine(
            CompilerTargetOptions(module_config),
            CodeGenOptLevel(module_config),
            cpu::CpuFeatureFromString(
                module_config.debug_options().xla_cpu_max_isa()));
    if (!jit_target_machine.ok()) {
      LOG(ERROR) << "Failed to infer target machine: "
                 << jit_target_machine.status();
      return;
    }

    cpu::TargetMachineFeatures target_machine_features(
        jit_target_machine->get());

    RegisterPass<ShardingPropagation>(
        /*is_spmd=*/true,
        /*propagate_metadata=*/false,
        module_config.allow_spmd_sharding_propagation_to_output(),
        module_config.allow_spmd_sharding_propagation_to_parameters(),
        /*cse_prevention_only=*/false,
        /*sharding_helper=*/nullptr);

    RegisterPass<spmd::StatefulRngSpmdPartitioner>(
        module_config.num_partitions(), module_config.replica_count());
    RegisterPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);
    RegisterPass<TopkDecomposer>([&](const HloInstruction* instr) {
      return instr->opcode() == HloOpcode::kTopK;
    });
    RegisterPass<BatchNormExpander>(
        /*rewrite_training_op=*/true,
        /*rewrite_inference_op=*/true,
        /*rewrite_grad_op=*/true);
    RegisterPass<HloPassFix<ReduceWindowRewriter>>(
        module_config.debug_options().xla_reduce_window_rewrite_base_length());
    auto dynamic_padder_options = DynamicPadderOptions();
    dynamic_padder_options.shape_check_mode =
        DynamicDimensionInference::ShapeCheckMode::kIgnore;
    RegisterPass<DynamicPadder>(dynamic_padder_options);
    RegisterPass<SelectAndScatterExpander>();
    RegisterPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);
    RegisterPass<ChangeOpDataType>(
        F16, F32, HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kConvolution>);
    AlgebraicSimplifierOptions options;
    options.set_enable_dot_strength_reduction(false);
    options.set_minmax_propagate_nan(false);
    options.set_supports_non_canonical_dots(false);
    options.set_executing_on_cpu(true);
    RegisterPass<AlgebraicSimplifier>(options);
    RegisterPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
    RegisterPass<TransposeFolding>(
        [&](const HloInstruction& dot,
            int64_t operand) -> absl::StatusOr<bool> {
          if (DotImplementationCanHandleTranspose(dot,
                                                  target_machine_features)) {
            return TransposeFolding::IsRowColumnTransposeDotOperand(dot,
                                                                    operand);
          }
          return false;
        },
        TransposeFolding::NeverFoldTranspose);
    RegisterPass<cpu::ConvCanonicalization>(&target_machine_features);

    // Fails to register if module does not have entry computation layout
    if (module.config().has_entry_computation_layout()) {
      RegisterPass<cpu::CpuLayoutAssignment>(
          module.mutable_entry_computation_layout(), &target_machine_features,
          nullptr);
    }

    RegisterPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
    const int max_parallelism =
        module_config.intra_op_parallelism_threads() > 0
            ? module_config.intra_op_parallelism_threads()
            : tsl::port::NumSchedulableCPUs();
    RegisterPass<cpu::ParallelTaskAssigner>(max_parallelism,
                                            cpu::CpuExecutable::ShapeSizeBytes,
                                            &target_machine_features);
    RegisterPass<cpu::CpuInstructionFusion>();
  }

 private:
  llvm::CodeGenOptLevel CodeGenOptLevel(const HloModuleConfig& module_config) {
    VLOG(2) << "backend_optimization_level: "
            << module_config.debug_options().xla_backend_optimization_level();
    switch (module_config.debug_options().xla_backend_optimization_level()) {
      case 1:
        return llvm::CodeGenOptLevel::Less;
      case 2:
        return llvm::CodeGenOptLevel::Default;
      case 3:
        return llvm::CodeGenOptLevel::Aggressive;
      default:
        return llvm::CodeGenOptLevel::None;
    }
  }

  llvm::TargetOptions CompilerTargetOptions(
      const HloModuleConfig& module_config) {
    llvm::TargetOptions target_options;
    // Always allow FMA fusion. This increases precision instead of decreasing
    // it.
    target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    return target_options;
  }
};

}  // namespace
}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(cpu_opt_provider, {
  xla::OptProvider::RegisterForPlatform(
      "cpu", std::make_unique<xla::CpuOptProvider>());
});
