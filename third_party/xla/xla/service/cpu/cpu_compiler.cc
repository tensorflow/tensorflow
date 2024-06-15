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

#include "xla/service/cpu/cpu_compiler.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/Config/Disassemblers.def.inc"
// IWYU pragma: no_include "llvm/Config/Targets.def.inc"

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/mlir_hlo/transforms/passes.h"
#include "xla/primitive_util.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/all_reduce_promotion.h"
#include "xla/service/all_to_all_decomposer.h"
#include "xla/service/batch_dot_simplification.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/broadcast_canonicalizer.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"
#include "xla/service/change_op_data_type.h"
#include "xla/service/cholesky_expander.h"
#include "xla/service/comparison_expander.h"
#include "xla/service/compiler.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/conditional_to_select.h"
#include "xla/service/convolution_group_converter.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu/buffer_info_util.h"
#include "xla/service/cpu/compiler_functor.h"
#include "xla/service/cpu/conv_canonicalization.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/service/cpu/cpu_layout_assignment.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/cpu/parallel_task_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/simple_orc_jit.h"
#include "xla/service/cpu/target_machine_features.h"
#include "xla/service/cpu/thunk_emitter.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/dump.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_dimension_simplifier.h"
#include "xla/service/dynamic_index_splitter.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/eigh_expander.h"
#include "xla/service/executable.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_ordering.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/indexed_array_analysis.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/llvm_compiler.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/logistic_expander.h"
#include "xla/service/map_inliner.h"
#include "xla/service/operand_upcaster.h"
#include "xla/service/optimization_barrier_expander.h"
#include "xla/service/optimize_input_output_buffer_alias.h"
#include "xla/service/qr_expander.h"
#include "xla/service/reduce_decomposer.h"
#include "xla/service/reduce_window_rewriter.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/result_caster.h"
#include "xla/service/rng_bit_generator_expander.h"
#include "xla/service/rng_expander.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/select_and_scatter_expander.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/stochastic_convert_decomposer.h"
#include "xla/service/sub_byte_normalization.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/tree_reduction_rewriter.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_invariant_code_motion.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

#ifdef TF_LLVM_X86_AVAILABLE
#include "llvm/TargetParser/X86TargetParser.h"
#endif

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
#include "xla/service/cpu/cpu_float_support.h"
#include "xla/service/cpu/onednn_matmul_rewriter.h"
#include "xla/service/cpu/onednn_ops_rewriter.h"
#include "xla/service/simplify_fp_conversions.h"
#endif

namespace xla {

namespace {

// For each computation in the module, determines whether that computation
// calls a custom-call function, either directly or indirectly (e.g. because it
// calls another computation that does).
absl::flat_hash_map<const HloComputation*, bool>
ModuleComputationsTransitivelyContainCustomCall(const HloModule& module) {
  absl::flat_hash_map<const HloComputation*, bool> custom_call_map;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(&module);

  // Can never fail because we always return an OK status from the visitor.
  TF_CHECK_OK(call_graph->VisitNodes([&custom_call_map](
                                         const CallGraphNode& node) {
    const HloComputation* computation = node.computation();

    for (const HloInstruction* instruction : computation->instructions()) {
      // The computation contains a custom-call instruction directly.
      if (DynCast<HloCustomCallInstruction>(instruction)) {
        custom_call_map[computation] = true;
        return absl::OkStatus();
      }
      // The computation calls something that contains a custom-call
      // instruction (directly or indirectly). This lookup relies on the call
      // graph traversing callees before callers, so that the map is always
      // populated for all callees at this point.
      for (const HloComputation* callee : instruction->called_computations()) {
        bool callee_contains_custom_call = FindOrDie(custom_call_map, callee);
        if (callee_contains_custom_call) {
          custom_call_map[computation] = true;
          return absl::OkStatus();
        }
      }
    }

    custom_call_map[computation] = false;
    return absl::OkStatus();
  }));

  return custom_call_map;
}

}  // namespace

namespace cpu {
using BufferInfo = cpu_function_runtime::BufferInfo;

CpuAotCompilationOptions::CpuAotCompilationOptions(
    std::string triple, std::string cpu_name, std::string features,
    std::string entry_point_name, RelocationModel relocation_model)
    : triple_(std::move(triple)),
      cpu_name_(std::move(cpu_name)),
      features_(std::move(features)),
      entry_point_name_(std::move(entry_point_name)),
      relocation_model_(relocation_model) {}

CpuAotCompilationOptions::~CpuAotCompilationOptions() = default;

se::Platform::Id CpuAotCompilationOptions::PlatformId() const {
  return se::host::kHostPlatformId;
}

CpuAotCompilationResult::CpuAotCompilationResult(
    ObjectFileData object_file_data, std::vector<BufferInfo> buffer_infos,
    int64_t result_buffer_index, std::unique_ptr<HloModule> module,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data)
    : object_file_data_(std::move(object_file_data)),
      buffer_infos_(std::move(buffer_infos)),
      result_buffer_index_(result_buffer_index),
      module_(std::move(module)),
      hlo_profile_printer_data_(std::move(hlo_profile_printer_data)) {}

const HloModule* CpuAotCompilationResult::optimized_module() const {
  return module_.get();
}

std::unique_ptr<HloModule> CpuAotCompilationResult::consume_optimized_module() {
  return std::move(module_);
}

CpuCompiler::CpuCompiler() {
  // Initialize LLVM the first time the CpuCompiler is initialized.
  static bool llvm_initialized = []() {
    InitializeLLVMTarget();
    return true;
  }();
  (void)llvm_initialized;
}

absl::StatusOr<std::vector<std::unique_ptr<Executable>>> CpuCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs,
    const CompileOptions& options) {
  for (const std::vector<se::StreamExecutor*>& se_vector : stream_execs) {
    if (se_vector.size() != 1) {
      return Unimplemented(
          "Model partitioning not implemented for the CPU compiler");
    }
  }
  return LLVMCompiler::Compile(std::move(module_group), stream_execs, options);
}

/* static */ void CpuCompiler::InitializeLLVMTarget() {
  // Initialize LLVM's MC layer for the native target.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

namespace {

// LLVM makes certain options configurable only through its command-line
// options; it provide the ParseCommandLineOptions function that lets us set
// flags at runtime. However, since these flags are global we want to avoid
// multiple invocations of the LLVM compilation pipeline with a different set of
// flags. Therefore, we only pass command-line flags to LLVM once, before the
// first module is compiled.
absl::once_flag llvm_command_line_options_initialized;

// This visitor records which HLO instructions should have profiling information
// recorded.
class CollectProfileCandidates : public DfsHloVisitorWithDefault {
 public:
  static absl::StatusOr<absl::flat_hash_map<const HloInstruction*, int64_t>>
  GetCandidatesForComputation(
      const HloComputation& computation,
      const absl::flat_hash_map<const HloInstruction*, int64_t>&
          assigned_indices) {
    absl::flat_hash_map<const HloInstruction*, int64_t> hlo_to_profile_idx;
    CollectProfileCandidates profile_candidates_for_computation(
        &hlo_to_profile_idx, assigned_indices);
    TF_RETURN_IF_ERROR(computation.Accept(&profile_candidates_for_computation));
    return hlo_to_profile_idx;
  }

 private:
  CollectProfileCandidates(
      absl::flat_hash_map<const HloInstruction*, int64_t>* hlo_to_profile_idx,
      const absl::flat_hash_map<const HloInstruction*, int64_t>&
          assigned_indices)
      : hlo_to_profile_idx_(hlo_to_profile_idx),
        assigned_indices_(assigned_indices) {}

  absl::Status DefaultAction(HloInstruction* hlo_instruction) override {
    hlo_to_profile_idx_->insert(
        {hlo_instruction, FindOrDie(assigned_indices_, hlo_instruction)});
    return absl::OkStatus();
  }

  absl::Status HandleCall(HloInstruction* call) override {
    TF_RETURN_IF_ERROR(DefaultAction(call));
    CollectProfileCandidates candidates_for_call(hlo_to_profile_idx_,
                                                 assigned_indices_);
    TF_RETURN_IF_ERROR(call->to_apply()->Accept(&candidates_for_call));
    return absl::OkStatus();
  }
  // Recurse into "conditional" so we can profile inside of it.
  absl::Status HandleConditional(HloInstruction* conditional) override {
    TF_RETURN_IF_ERROR(DefaultAction(conditional));

    CollectProfileCandidates candidates_for_true(hlo_to_profile_idx_,
                                                 assigned_indices_);
    TF_RETURN_IF_ERROR(
        conditional->true_computation()->Accept(&candidates_for_true));

    CollectProfileCandidates candidates_for_false(hlo_to_profile_idx_,
                                                  assigned_indices_);
    TF_RETURN_IF_ERROR(
        conditional->false_computation()->Accept(&candidates_for_false));

    return absl::OkStatus();
  }

  // Skip constants, there is nothing to profile.
  absl::Status HandleConstant(HloInstruction*) override {
    return absl::OkStatus();
  }
  // Skip parameters, they are a simple load.
  absl::Status HandleParameter(HloInstruction*) override {
    return absl::OkStatus();
  }
  // It is important to recurse for "while" or else we risk overly coarse
  // profiling information.
  absl::Status HandleWhile(HloInstruction* xla_while) override {
    TF_RETURN_IF_ERROR(DefaultAction(xla_while));

    CollectProfileCandidates candidates_for_condition(hlo_to_profile_idx_,
                                                      assigned_indices_);
    TF_RETURN_IF_ERROR(
        xla_while->while_condition()->Accept(&candidates_for_condition));

    CollectProfileCandidates candidates_for_body(hlo_to_profile_idx_,
                                                 assigned_indices_);
    TF_RETURN_IF_ERROR(xla_while->while_body()->Accept(&candidates_for_body));

    return absl::OkStatus();
  }

  absl::flat_hash_map<const HloInstruction*, int64_t>* hlo_to_profile_idx_;
  const absl::flat_hash_map<const HloInstruction*, int64_t>& assigned_indices_;
};

// Adds the HloVerifier for CPU to the given pipeline.
void AddHloVerifier(HloPassPipeline* pipeline, HloVerifierOpts&& opts = {},
                    bool debug_only = false) {
  auto verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));

  if (debug_only) {
    pipeline->AddInvariantCheckerDebug<HloVerifier>(
        std::move(verifier_metadata), "hlo verifier (debug)");
  } else {
    pipeline->AddInvariantChecker<HloVerifier>(std::move(verifier_metadata),
                                               "hlo verifier");
  }
}

}  // namespace

absl::Status CpuCompiler::RunHloPassesThroughLayoutAssn(
    HloModule* module, bool is_aot_compile,
    LLVMTargetMachineFeatures* target_machine_features, bool is_mlir_compile) {
  const DebugOptions& debug_options = module->config().debug_options();
  const int64_t num_partitions = module->config().num_partitions();
  if (num_partitions > 1) {
    if (!module->config().use_spmd_partitioning()) {
      return InvalidArgument(
          "num_partitions=%d but SPMD partitioning not enabled.",
          num_partitions);
    }
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    // Run some IR cleanup passes before running the SPMD partitioning
    // passes.
    AddHloVerifier(&spmd_pipeline);
    spmd_pipeline.AddPass<CallInliner>();
    spmd_pipeline.AddPass<ZeroSizedHloElimination>();
    spmd_pipeline.AddPass<ConditionalCanonicalizer>();

    spmd_pipeline.AddPass<ShardingPropagation>(
        /*is_spmd=*/true, /*propagate_metadata=*/false,
        module->config().allow_spmd_sharding_propagation_to_output(),
        module->config().allow_spmd_sharding_propagation_to_parameters());
    spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
        num_partitions, module->config().replica_count());
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(module).status());
  } else {
    HloPassPipeline sharding_removal_pipeline("sharding-removal");
    AddHloVerifier(&sharding_removal_pipeline);
    // Remove redundant sharding ops when partition_count == 1.
    sharding_removal_pipeline.AddPass<ShardingRemover>();
    sharding_removal_pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(sharding_removal_pipeline.Run(module).status());
  }

  {
    // SubbytePacker must be run before the rest of the pipeline since it
    // modifies the layout of the entry computation inputs/outputs, which is
    // passed to LayoutAssignment.
    HloPassPipeline subbyte_packer_pipeline("SubbytePacker pipeline");
    subbyte_packer_pipeline.AddPass<SubByteNormalization>(
        SubByteNormalization::SET_ELEMENT_SIZE);
    TF_RETURN_IF_ERROR(subbyte_packer_pipeline.Run(module).status());
  }

  HloPassPipeline pipeline("HLO passes through layout assignment");
  AddHloVerifier(&pipeline);

  pipeline.AddPass<OperandUpcaster>();
  pipeline.AddPass<ResultCaster>();

  // Expand random number generation.
  pipeline.AddPass<RngExpander>();
  if (!is_mlir_compile) {
    pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);
  }

  // Remove zero-sized HLO from the input so that other passes don't have to
  // handle it.
  pipeline.AddPass<ZeroSizedHloElimination>();

  pipeline.AddPass<DynamicIndexSplitter>();

  pipeline.AddPass<ConditionalToSelect>();
  pipeline.AddPass<MapInliner>();

  // The TopkDecomposer generates a compare op with type=TOTALORDER and must
  // run before the ComparisonExpander which rewrites such comparisons.
  pipeline.AddPass<TopkDecomposer>([&](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kTopK;
  });

  pipeline.AddPass<ComparisonExpander>();
  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<QrExpander>();
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<TriangularSolveExpander>();
  pipeline.AddPass<AllToAllDecomposer>();
  pipeline.AddPass<StochasticConvertDecomposer>();

  // Inline computations with a single call site.
  pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  pipeline.AddPass<BatchDotSimplification>();
  pipeline.AddPass<DotDecomposer>();

  // Rewrite to custom calls with target as oneDNN library calls.
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
  // AOT compiled code runs in single thread.
  if (!is_aot_compile) {
    // Placing OneDnnOpsRewriter here to match the flax patterns
    // TODO: Decide where would be the appropriate place for this pass to make
    // it more generic
    // TODO - intel: Name of the pass might seem redundant as oneDnnRewriter,
    // but in future plan to rename oneDNNrewriter to specific to onednn matmul
    pipeline.AddPass<OneDnnOpsRewriter>();
  }
#endif  // INTEL_MKL && ENABLE_ONEDNN_V3

  // Promote BF16 all-reduce to F32.
  const std::pair<PrimitiveType, PrimitiveType> ar_promoted_types[] = {
      {BF16, F32}};
  pipeline.AddPass<AllReducePromotion>(ar_promoted_types);
  // Convert BF16 and F8 operations to F32 and F16 respectively so that the CPU
  // backend can support BF16/F8 operations without directly implementing a
  // BF16/F8 lowering for most ops.
  FloatSupport bf16_support(BF16);
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
  CpuFloatSupport onednn_bf16_support(BF16);
  if (!is_aot_compile) {
    pipeline.AddPass<FloatNormalization>(&onednn_bf16_support);
  } else {
    pipeline.AddPass<FloatNormalization>(&bf16_support);
  }
#else
  pipeline.AddPass<FloatNormalization>(&bf16_support);
#endif
  FloatSupport f8e5m2_support(F8E5M2, F16);
  pipeline.AddPass<FloatNormalization>(&f8e5m2_support);
  FloatSupport f8e4m3fn_support(F8E4M3FN, F16);
  pipeline.AddPass<FloatNormalization>(&f8e4m3fn_support);
  FloatSupport f8e4m3b11fnuz_support(F8E4M3B11FNUZ, F16);
  pipeline.AddPass<FloatNormalization>(&f8e4m3b11fnuz_support);
  FloatSupport f8e5m2fnuz_support(F8E5M2FNUZ, F16);
  pipeline.AddPass<FloatNormalization>(&f8e5m2fnuz_support);
  FloatSupport f8e4m3fnuz_support(F8E4M3FNUZ, F16);
  pipeline.AddPass<FloatNormalization>(&f8e4m3fnuz_support);
  // After canonicalization, there may be more batch dots that can be
  // simplified.
  pipeline.AddPass<BatchDotSimplification>();
  auto cost_model = [](HloInstruction* conv) {
    // We need a cost model for CPUs. Currently, do nothing.
    return false;
  };
  pipeline.AddPass<ConvolutionGroupConverter>(
      /*should_expand=*/[](HloInstruction* conv) { return true; }, cost_model,
      /*convert_batch_groups_only=*/true);
  auto feature_group_should_expand = [](HloInstruction* conv) {
    switch (conv->shape().element_type()) {
      case F16:
      case F32:
        return false;
      default:
        return true;
    }
  };
  pipeline.AddPass<ConvolutionGroupConverter>(
      feature_group_should_expand, cost_model,
      /*convert_batch_groups_only=*/false);
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);
  pipeline.AddPass<LogisticExpander>();
  pipeline.AddPass<ConditionalCanonicalizer>();
  pipeline.AddPass<DynamicDimensionSimplifier>();

  if (debug_options.xla_reduce_window_rewrite_base_length() != 0) {
    pipeline.AddPass<HloPassFix<ReduceWindowRewriter>>(
        debug_options.xla_reduce_window_rewrite_base_length());
  }

  auto dynamic_padder_options = DynamicPadderOptions();
  // TODO(pgavin): ShapeChecks were never implemented correctly by the dynamic
  // padder.  The mode defaults to kIgnore, and it was not overridden for nested
  // computations (such as while bodies or conditional branches), and so cases
  // that could not be proven would still be accepted even with compile-time
  // checks enabled.  Recent changes to the DynamicPadder correctly
  // override the mode.  However, some models have started to rely on the check
  // being ignored, and they would be broken if it is enforced.
  dynamic_padder_options.shape_check_mode =
      DynamicDimensionInference::ShapeCheckMode::kIgnore;
  pipeline.AddPass<DynamicPadder>(dynamic_padder_options);
  if (!is_mlir_compile) {
    pipeline.AddPass<SelectAndScatterExpander>();
    pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);
  }
  pipeline.AddPass<ConvCanonicalization>(target_machine_features);

  // Run fp16 dots/convs in fp32 and then downcast the result to fp16.
  // Justification:
  //
  //   - This is significantly faster on our CPUs today than true fp16.
  //   - It's numerically more accurate.  (Granted, this is not always
  //     desirable, thus the ability to disable this functionality.)
  //   - It matches more closely the GPU's behavior on fp16 dot/conv, where
  //     accumulation happens in f32.
  if (!module->config().debug_options().xla_cpu_strict_dot_conv_math()) {
    pipeline.AddPass<ChangeOpDataType>(
        F16, F32, HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kConvolution>);
  }

  // Run the following passes to a fixed point.
  [&pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification"),
   this] {
    AddHloVerifier(&pipeline, HloVerifierOpts{},
                   /*debug_only=*/true);

    AlgebraicSimplifierOptions options;
    options.set_enable_dot_strength_reduction(false);
    // TODO(b/209827141): XLA:CPU doesn't propagate NaN through min/max, but
    // other platforms do, so it should be changed.
    options.set_minmax_propagate_nan(false);
    options.set_supports_non_canonical_dots(false);
    pipeline.AddPass<AlgebraicSimplifier>(options);
    pipeline.AddPass<SortSimplifier>();
    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);

    // Needs to happen after algebraic simplifier.
    pipeline.AddPass<TreeReductionRewriter>();

    // BatchNormExpander can create zero-sized ops, so zero-sized HLO
    // elimination has to come after that pass.
    pipeline.AddPass<ZeroSizedHloElimination>();

    pipeline.AddPass<WhileLoopInvariantCodeMotion>();
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<WhileLoopConstantSinking>();
    pipeline.AddPass<WhileLoopSimplifier>();

    // TODO(b/134075051): Re-enable after b/134075051 is fixed.
    // pipeline.AddPass<SliceSinker>();

    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<ReshapeMover>();
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<ConditionalSimplifier>();
  }();
  pipeline.AddPass<BitcastDtypesExpander>();

  // XLA lowers topk to a libcall while the MLIR based pipeline does not yet
  // support libcalls. Disable this for now.
  if (!is_mlir_compile) {
    pipeline.AddPass<TopkRewriter>([](const HloSortInstruction* sort, int64_t) {
      return sort->operand(0)->shape().element_type() == F32;
    });
  }
  pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  pipeline.AddPass<TransposeFolding>(
      [&](const HloInstruction& dot, int64_t operand) -> absl::StatusOr<bool> {
        if (DotImplementationCanHandleTranspose(dot,
                                                *target_machine_features)) {
          return TransposeFolding::IsRowColumnTransposeDotOperand(dot, operand);
        }
        return false;
      },
      TransposeFolding::NeverFoldTranspose);
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);

  pipeline.AddPass<OptimizationBarrierExpander>();
  pipeline.AddPass<TupleSimplifier>();

  // Layout assignment uses alias analysis, which requires the call graph to be
  // flattened.
  pipeline.AddPass<FlattenCallGraph>();
  ChannelLayoutConstraints layout_constraints;
  // The MLIR pipeline always uses default layouts, so we don't need to run
  // layout assignment. The exception to this is at the ABI boundary, where
  // custom layouts may be used. The XlaAbiLegalization pass takes care of
  // these.
  if (!is_mlir_compile) {
    pipeline.AddPass<CpuLayoutAssignment>(
        module->mutable_entry_computation_layout(), target_machine_features,
        &layout_constraints);
    // Run SubByteNormalization because CpuLayoutAssignment may modify a
    // Layout's element_size_in_bits field.
    pipeline.AddPass<SubByteNormalization>(
        SubByteNormalization::SET_ELEMENT_SIZE);
  }

  return pipeline.Run(module).status();
}

absl::Status CpuCompiler::RunHloPassesAfterLayoutAssn(
    HloModule* module, bool is_aot_compile,
    LLVMTargetMachineFeatures* target_machine_features,
    const CompileOptions& compile_options, bool is_mlir_compile) {
  HloPassPipeline pipeline("HLO passes after layout assignment");

  // CopyInsertion is still needed by BufferAssignment. MLIR passes will handle
  // everything else done by XLA, but CopyInsertion is needed to interface with
  // the existing runtime.
  if (is_mlir_compile) {
    pipeline.AddPass<CopyInsertion>();
    return pipeline.Run(module).status();
  }

  {
    HloPassPipeline normalization_pipeline("hlo normalization");
    normalization_pipeline.AddPass<ReshapeDecomposer>();
    normalization_pipeline.AddPass<ReduceDecomposer>();
    normalization_pipeline.AddPass<BroadcastCanonicalizer>();
    TF_RETURN_IF_ERROR(normalization_pipeline.Run(module).status());
  }

  // After layout assignment, use a layout-sensitive verifier.
  pipeline.AddPass<HloPassPipeline>("after layout assignment");
  AddHloVerifier(&pipeline, HloVerifierOpts{}.MakeLayoutSensitive(),
                 /*debug_only=*/true);

  pipeline.AddPass<ReshapeDecomposer>();

  const int max_parallelism =
      module->config().intra_op_parallelism_threads() > 0
          ? module->config().intra_op_parallelism_threads()
          : tsl::port::NumSchedulableCPUs();

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
  // AOT compiled code runs in single thread.
  if (!is_aot_compile) {
    auto debug_options = module->config().debug_options();
    // Run SimplifyFPConversions pass to simplify the BF16 pattern and make it
    // easier to match.
    // Remove `f32 -> bf16 -> f32` casts inserted by bf16 normalization.
    if (debug_options.xla_allow_excess_precision()) {
      pipeline.AddPass<SimplifyFPConversions>();
    }
    pipeline.AddPass<OneDnnMatMulRewriter>(max_parallelism,
                                           compile_options.thread_pool);
    // Run SimplifyFPConversions pass again to remove redundant Convert ops
    // that may exist as a result of running OneDnnMatMulRewriter pass.
    if (debug_options.xla_allow_excess_precision()) {
      pipeline.AddPass<SimplifyFPConversions>();
    }
  }
#endif  // INTEL_MKL && ENABLE_ONEDNN_V3

  // Add a fusion pass now that layout assignment is done.
  pipeline.AddPass<CpuInstructionFusion>();

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  // Run this to a fixed point.
  [&pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
       "simplification after layout assignment"),
   this] {
    AddHloVerifier(
        &pipeline,
        HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
            LayoutAssignment::InstructionCanChangeLayout),
        /*debug_only=*/true);
    AlgebraicSimplifierOptions options;
    options.set_is_layout_sensitive(true);
    options.set_supports_non_canonical_dots(false);
    options.set_enable_dot_strength_reduction(false);
    // TODO(b/209827141): XLA:CPU doesn't propagate NaN through min/max, but
    // other platforms do, so it should be changed.
    options.set_minmax_propagate_nan(false);
    pipeline.AddPass<AlgebraicSimplifier>(options);
    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  }();

  // Outline ops in the entry computation into calls to subcomputations.
  if (!is_aot_compile) {
    // Run ParallelTaskAssigner to assign parallel tasks to HLOs in module.
    // Note this is not run for AOT because it would bring in thread pool
    // and thread synchronization dependencies which would likely increase
    // binary size (and most AOT applications are single-threaded).
    // TODO(b/29630486) Support multi-threaded AOT.
    pipeline.AddPass<ParallelTaskAssigner>(
        max_parallelism, ShapeSizeBytesFunction(), target_machine_features);
  }
  // Copy insertion should be performed immediately before IR emission to
  // avoid inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes
  // an instruction which materializes a value). DCE must be run immediately
  // before (and sometime after) copy insertion, to avoid dead code from
  // interfering with the rewrites.
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<OptimizeInputOutputBufferAlias>(true);
  pipeline.AddPass<CopyInsertion>();
  pipeline.AddPass<HloDCE>();
  return pipeline.Run(module).status();
}

absl::Status CpuCompiler::RunHloPasses(HloModule* module, bool is_aot_compile,
                                       llvm::TargetMachine* target_machine,
                                       const CompileOptions& compile_options,
                                       bool is_mlir_compile) {
  LLVMTargetMachineFeatures target_machine_features(target_machine);
  TF_RETURN_IF_ERROR(RunHloPassesThroughLayoutAssn(
      module, is_aot_compile, &target_machine_features, is_mlir_compile));

  return RunHloPassesAfterLayoutAssn(module, is_aot_compile,
                                     &target_machine_features, compile_options,
                                     is_mlir_compile);
}

namespace {

// Align buffers to XLA:CPU minimal alignment.
int64_t memory_alignment(LogicalBuffer::Color) {
  return cpu_function_runtime::MinAlign();
}

llvm::TargetOptions CompilerTargetOptions(
    const HloModuleConfig& module_config) {
  llvm::TargetOptions target_options;
  // Always allow FMA fusion. This increases precision instead of decreasing it.
  target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  return target_options;
}

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

std::pair<LLVMCompiler::ModuleHook, LLVMCompiler::ModuleHook> GetIRModuleHooks(
    const HloModule& hlo_module,
    const LLVMCompiler::ModuleHook& user_pre_optimization_hook,
    const LLVMCompiler::ModuleHook& user_post_optimization_hook) {
  // Create the IR hooks. If applicable, each IR hook does the following:
  //
  //  * Calls the user supplied module hook.
  //  * Writes out the IR to a file in the output directory designated by
  //    --xla_dump_to
  const HloModule* hlo_module_ptr = &hlo_module;
  auto hook = [user_pre_optimization_hook, user_post_optimization_hook,
               hlo_module_ptr](bool optimized,
                               const llvm::Module& llvm_module) {
    const auto& user_hook =
        !optimized ? user_pre_optimization_hook : user_post_optimization_hook;
    if (user_hook) {
      user_hook(llvm_module);
    }
    llvm_ir::DumpIrIfEnabled(*hlo_module_ptr, llvm_module, optimized);
  };
  return {[hook](const llvm::Module& llvm_module) {
            return hook(/*optimized=*/false, llvm_module);
          },
          [hook](const llvm::Module& llvm_module) {
            return hook(/*optimized=*/true, llvm_module);
          }};
}

absl::Status VerifyLlvmModule(const llvm::Module& llvm_module) {
  XLA_SCOPED_LOGGING_TIMER("CpuCompiler - Running LLVM verifier");

  std::string err;
  llvm::raw_string_ostream err_stream(err);

  // verifyModule() returns true if the module is broken.
  TF_RET_CHECK(!llvm::verifyModule(llvm_module, &err_stream))
      << "Invalid LLVM IR before optimizations:\n"
      << err_stream.str()
      << "\nThis probably indicates a bug in the HLO -> LLVM IR lowering. "
         "Rerun with --xla_dump_to to get the IR. ";
  return absl::OkStatus();
}

absl::Status CreateHloProfilingArtifacts(
    const HloModule& module,
    absl::flat_hash_map<const HloInstruction*, int64_t>*
        instruction_to_profile_idx,
    absl::flat_hash_map<const HloComputation*, int64_t>*
        computation_to_profile_idx,
    std::unique_ptr<HloProfileIndexMap>* hlo_profile_index_map,
    std::unique_ptr<HloProfilePrinterData>* hlo_profile_printer_data) {
  *hlo_profile_index_map = std::make_unique<HloProfileIndexMap>(module);
  const HloComputation& entry_computation = *module.entry_computation();

  TF_ASSIGN_OR_RETURN(
      *instruction_to_profile_idx,
      CollectProfileCandidates::GetCandidatesForComputation(
          entry_computation,
          (*hlo_profile_index_map)->instruction_to_profile_idx()));

  auto shape_size_bytes = [](const Shape& shape) {
    // On the cpu, opaques are pointers.
    if (shape.IsOpaque()) {
      return static_cast<int64_t>(sizeof(void*));
    }
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  };

  HloCostAnalysis cost_analysis(shape_size_bytes);
  TF_RETURN_IF_ERROR(entry_computation.Accept(&cost_analysis));
  *hlo_profile_printer_data = CreateHloProfilePrinterData(
      **hlo_profile_index_map, cost_analysis, entry_computation.name());
  *computation_to_profile_idx =
      (*hlo_profile_index_map)->computation_to_profile_idx();

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<HloModule>> CpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* /*stream_exec*/,
    const CompileOptions& options) {
  std::unique_ptr<llvm::TargetMachine> jit_target_machine =
      SimpleOrcJIT::InferTargetMachineForJIT(
          CompilerTargetOptions(module->config()),
          CodeGenOptLevel(module->config()));

  TF_RETURN_IF_ERROR(RunHloPasses(module.get(), /*is_aot_compile=*/false,
                                  jit_target_machine.get(),
                                  /*compile_options=*/options,
                                  /*is_mlir_compile=*/false));
  return std::move(module);
}

namespace {

// Post-compilation callback functor for use by SimpleOrcJIT.
//
// Dumps machine code if dumping is enabled for the module.
static absl::AnyInvocable<void(const llvm::object::ObjectFile& obj_file)>
CreateOrcJITPostCompilationHook(const HloModule* module,
                                std::vector<std::string>* obj_files) {
  return [=](const llvm::object::ObjectFile& obj_file) {
    if (obj_files) obj_files->push_back(obj_file.getData().str());

    if (DumpingEnabledForHloModule(*module)) {
      DumpToFileInDir(*module, /*file_prefix=*/"", /*file_suffix=*/"o",
                      absl::string_view(obj_file.getData().data(),
                                        obj_file.getData().size()));
    }
  };
}

void InitializeLLVMCommandLineOptions(const HloModuleConfig& config) {
  llvm_ir::InitializeLLVMCommandLineOptions(
      config.debug_options().xla_backend_extra_options());
}

struct ComputationToEmit {
  HloComputation* computation;

  // Are we emitting this computation with fast-math reassociation enabled?
  // We enable reassociation for reductions because it has a significant
  // performance impact.
  bool allow_reassociation;

  bool operator==(const ComputationToEmit& other) const {
    return computation == other.computation &&
           allow_reassociation == other.allow_reassociation;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ComputationToEmit& c) {
    return H::combine(std::move(h), c.computation, c.allow_reassociation);
  }
};

std::vector<ComputationToEmit> SubcomputationEmissionOrder(
    HloComputation* root) {
  absl::flat_hash_set<ComputationToEmit> visited;
  std::vector<ComputationToEmit> postorder;

  // agenda of (node, leave) pairs.
  std::stack<std::pair<ComputationToEmit, bool>> agenda;
  agenda.emplace(ComputationToEmit{root, false}, false);
  while (!agenda.empty()) {
    ComputationToEmit c;
    bool leave;
    std::tie(c, leave) = agenda.top();
    agenda.pop();

    if (leave) {
      postorder.push_back(c);
      continue;
    }

    if (visited.insert(c).second) {
      agenda.emplace(c, true);
      for (auto* instruction : c.computation->instructions()) {
        bool allow_reassociation =
            instruction->opcode() == HloOpcode::kAllReduce ||
            instruction->opcode() == HloOpcode::kReduce ||
            instruction->opcode() == HloOpcode::kReduceWindow;
        auto cc = absl::MakeSpan(instruction->called_computations());
        for (auto it = cc.rbegin(); it != cc.rend(); ++it) {
          HloComputation* called_computation = *it;
          ComputationToEmit callee{
              called_computation, c.allow_reassociation || allow_reassociation};
          if (!visited.contains(callee)) {
            agenda.emplace(callee, false);
          }
        }
      }
    }
  }
  DCHECK(!postorder.empty() && postorder.back().computation == root);
  postorder.pop_back();
  return postorder;
}

}  // namespace

static absl::StatusOr<CpuExecutable::ConstantAllocation>
LiteralToConstantAllocation(BufferAllocation::Index index,
                            const Literal& literal) {
  // TODO(ezhulenev): This code is almost identical to code in XLA:GPU, we
  // should standardize it. See `xla/service/gpu/ir_emission_utils.cc`.
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return absl::InternalError(
        "Only array literals can be converted to constant allocations");
  }

  int64_t size_bytes = literal.size_bytes();
  const void* untyped_data = literal.untyped_data();

  // Pack sub-byte types into an XLA storage format.
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    int bit_width = primitive_util::BitWidth(element_type);
    int packed_size_bytes = CeilOfRatio<int64_t>(size_bytes, 8 / bit_width);

    // Use Literal as a storage for packed data as it allocates underlying
    // buffer with correct alignment. Keep it allocated on heap to avoid
    // capturing stack address that will be invalidated by a move below.
    auto packed = std::make_unique<Literal>(
        ShapeUtil::MakeShape(U8, {packed_size_bytes}));

    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(untyped_data), size_bytes),
        absl::MakeSpan(reinterpret_cast<char*>(packed->untyped_data()),
                       packed->size_bytes()));

    return CpuExecutable::ConstantAllocation{index, std::move(packed)};
  }

  // Create a constant allocation from the literal's untyped data.
  return CpuExecutable::ConstantAllocation{
      index, absl::Span<const uint8_t>(
                 reinterpret_cast<const uint8_t*>(untyped_data), size_bytes)};
}

// Creates a vector of constant allocations from the given buffer assignment.
static absl::StatusOr<std::vector<CpuExecutable::ConstantAllocation>>
CreateConstantAllocations(const BufferAssignment& assignment) {
  std::vector<CpuExecutable::ConstantAllocation> constants;

  for (const BufferAllocation& allocation : assignment.Allocations()) {
    if (!allocation.is_constant()) {
      continue;
    }

    // Find the constant instruction defining the value for allocation.
    HloInstruction* const_instr = nullptr;
    for (const auto& [value, _] : allocation.assigned_buffers()) {
      // Multiple aliasing instructions can share the allocation, we need to
      // find the original constant instruction that defines the value.
      if (value->instruction()->opcode() == HloOpcode::kConstant) {
        if (const_instr != nullptr) {
          return absl::InternalError(
              absl::StrCat("Multiple constant instructions define buffer ",
                           allocation.ToString()));
        }
        const_instr = value->instruction();
      }
    }
    if (const_instr == nullptr) {
      return absl::InternalError(
          absl::StrCat("Could not find constant instruction defining buffer ",
                       allocation.ToString()));
    }

    VLOG(3) << "Create constant allocation for index " << allocation.index()
            << " from constant literal " << const_instr->name()
            << "; shape=" << const_instr->literal().shape();
    TF_ASSIGN_OR_RETURN(constants.emplace_back(),
                        LiteralToConstantAllocation(allocation.index(),
                                                    const_instr->literal()));
  }

  return constants;
}

absl::StatusOr<std::unique_ptr<CpuExecutable>>
CpuCompiler::CompileLegacyCpuExecutable(std::unique_ptr<HloModule> module) {
  ModuleHook pre_optimization_ir_hook;
  ModuleHook post_optimization_ir_hook;
  std::tie(pre_optimization_ir_hook, post_optimization_ir_hook) =
      GetIRModuleHooks(*module, user_pre_optimization_hook_,
                       user_post_optimization_hook_);

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::MLIRContext mlir_context;
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module =
      std::make_unique<llvm::Module>("__compute_module", *llvm_context);

  const DebugOptions& debug_options = module->config().debug_options();

  // We collect compiled object files (machine code) so we can export
  // CpuExecutable to an AOT compilation result.
  std::vector<std::string> obj_files;

  auto jit = SimpleOrcJIT::Create(
      CompilerTargetOptions(module->config()),
      CodeGenOptLevel(module->config()),
      options::OptimizeForSizeRequested(module->config()),
      debug_options.xla_llvm_disable_expensive_passes(),
      options::SlpVectorizerDisabled(module->config()),
      llvm_ir::GetCpuFastMathFlags(module->config()), pre_optimization_ir_hook,
      post_optimization_ir_hook,
      CreateOrcJITPostCompilationHook(module.get(), &obj_files));
  if (!jit) {
    return Internal("Creating JIT failed: %s", llvm::toString(jit.takeError()));
  }
  llvm_module->setDataLayout((*jit)->data_layout());
  llvm_module->setTargetTriple((*jit)->target_triple().getTriple());

  HloComputation* entry_computation = module->entry_computation();
  absl::flat_hash_map<const HloInstruction*, int64_t>
      instruction_to_profile_idx;
  absl::flat_hash_map<const HloComputation*, int64_t>
      computation_to_profile_idx;
  std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map;
  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data;
  if (module->config().hlo_profiling_enabled()) {
    TF_RETURN_IF_ERROR(CreateHloProfilingArtifacts(
        *module, &instruction_to_profile_idx, &computation_to_profile_idx,
        &hlo_profile_index_map, &hlo_profile_printer_data));
  }

  // Cache these flags here since we'll want to access them after the module's
  // ownership is std::moved.
  const bool embed_ir_in_executable =
      debug_options.xla_embed_ir_in_executable();

  // Select a memory scheduler optimized for concurrency vs minimal memory.
  auto scheduler =
      debug_options.xla_cpu_enable_concurrency_optimized_scheduler()
          ? BFSMemoryScheduler
          : DFSMemoryScheduler;

  // Select an order for emitting the HLO instructions for each
  // computation. Using this sequence enables tighter buffer liveness analysis
  // and reduced memory usage (as compared to using DependencyHloOrdering).
  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleModule(module.get(), BufferSizeBytesFunction(),
                     ComputationSchedulerToModuleScheduler(scheduler)));
  TF_RETURN_IF_ERROR(module->set_schedule(schedule));

  // Run buffer allocation on the HLO graph.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(module.get(),
                          std::make_unique<SequentialHloOrdering>(schedule),
                          BufferSizeBytesFunction(), memory_alignment,
                          /*allocate_buffers_for_constants=*/true));
  DumpHloModuleIfEnabled(*module, *assignment,
                         absl::StrCat("cpu_", kAfterOptimizationsDumpName));

  // Dump computation proto state and buffer assignment for
  // GetCompiledMemoryStats results.
  auto with_hlo_proto = [&](std::unique_ptr<CpuExecutable> cpu_executable) {
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_hlo_module() = cpu_executable->module().ToProto();
    *hlo_proto->mutable_buffer_assignment() =
        cpu_executable->buffer_assignment().ToProto();
    cpu_executable->set_hlo_proto(std::move(hlo_proto));
    return cpu_executable;
  };

  LLVMTargetMachineFeatures target_machine_features((*jit)->target_machine());

  // TODO(ezhulenev): Once we fully migrate to Thunks current IrEmitter should
  // be renamed to NestedIrEmitter and be used only for emitting nested (aka
  // thread local or embedded) computations (reductions, maps, etc.).

  // (Nested) IrEmitter is responsible for building LLVM module with functions
  // for all HLO computations. In thunk execution mode we only build LLVM
  // functions for embedded computations (e.g. reduction computations) and all
  // high-level operations (fusions, elementwise, etc.) are lowered to kernel
  // functions (which are also LLVM functions, but use a HostKernel ABI).
  IrEmitter nested_ir_emitter(
      &mlir_context, *module, *assignment, llvm_module.get(),
      std::move(instruction_to_profile_idx),
      std::move(computation_to_profile_idx),
      ModuleComputationsTransitivelyContainCustomCall(*module),
      &target_machine_features,
#ifdef MEMORY_SANITIZER
      /*emit_code_for_msan=*/true
#else
      /*emit_code_for_msan=*/false
#endif
  );

  // Emit global variables for constants.
  //
  // TODO(ezhulenev): Figure out how to emit constants that are only needed for
  // thread local computations as with Thunks runtime we keep constants outside
  // of the LLVM module. Currently we end up doubling memory for constants.
  TF_RETURN_IF_ERROR(nested_ir_emitter.EmitConstantGlobals());

  // If we use Thunk runtime then instead of emitting LLVM function for the
  // entry computation we emit a sequence of thunks that implement the
  // computation as a sequence of interpreted commands.
  if (module->config().debug_options().xla_cpu_use_thunk_runtime()) {
    // IR emitter is responsible for building LLVM module with host kernels for
    // corresponding HLO instructions (fusions, elemental instructions, etc.).
    IrEmitter2 ir_emitter2(*module, llvm_module.get(), &nested_ir_emitter);

    // Thunk emitter is responsible for building a Thunk sequence that will
    // resolved kernels in the compiled LLVM module and execute them together
    // with Thunks implemented as library calls (e.g. oneDNN or Eigen).
    ThunkEmitter thunk_emitter(ir_emitter2, *assignment,
                               target_machine_features, module->config());
    TF_ASSIGN_OR_RETURN(ThunkSequence thunks,
                        thunk_emitter.EmitEntryComputation(*module));

    // JIT compile the LLVM IR module to in-memory machine code.
    TF_RETURN_IF_ERROR(VerifyLlvmModule(*llvm_module));
    cantFail((*jit)->AddModule(llvm::orc::ThreadSafeModule(
        std::move(llvm_module), std::move(llvm_context))));

    // TODO(ezhulenev): We should be able to make it lazy on-demand, but today
    // we capture obj_files by reference and it leads to asan errors. Figure out
    // lifetime issues and move compilation to Thunk initialization stage.
    for (const auto& kernel : ir_emitter2.kernels()) {
      if (auto sym = (*jit)->FindCompiledSymbol(kernel.name); !sym) {
        return Internal("Failed to find compiled symbol for kernel %s",
                        kernel.name);
      }
    }

    // Create constant allocations from the buffer assignment.
    TF_ASSIGN_OR_RETURN(
        std::vector<CpuExecutable::ConstantAllocation> constants,
        CreateConstantAllocations(*assignment));

    TF_ASSIGN_OR_RETURN(
        auto cpu_executable,
        CpuExecutable::Create(std::move(*jit), std::move(assignment),
                              std::move(module), std::move(thunks),
                              std::move(constants),
                              std::move(hlo_profile_printer_data),
                              std::move(hlo_profile_index_map)));

    return with_hlo_proto(std::move(cpu_executable));
  }

  // Each computation is a single function.  Emit all embedded computations
  // before the entry computation. The order of computations returned from
  // SubcomputationEmissionOrder guarantees that a called computation occurs
  // before a caller computation.
  for (ComputationToEmit subcomputation :
       SubcomputationEmissionOrder(entry_computation)) {
    if (subcomputation.computation->IsFusionComputation()) {
      continue;
    }
    TF_RETURN_IF_ERROR(
        nested_ir_emitter
            .EmitComputation(
                subcomputation.computation, subcomputation.computation->name(),
                /*is_top_level_computation=*/false,
                schedule.sequence(subcomputation.computation).instructions(),
                subcomputation.allow_reassociation)
            .status());
  }
  absl::string_view function_name_prefix = entry_computation->name().empty()
                                               ? "__compute"
                                               : entry_computation->name();
  TF_ASSIGN_OR_RETURN(llvm::Function * entry_function,
                      nested_ir_emitter.EmitComputation(
                          entry_computation, function_name_prefix,
                          /*is_top_level_computation=*/true,
                          schedule.sequence(entry_computation).instructions(),
                          /*allow_reassociation=*/false));

  std::string function_name = [&]() {
    llvm::SmallVector<char, 40> function_name_vector;
    llvm::Mangler::getNameWithPrefix(
        function_name_vector, entry_function->getName(), (*jit)->data_layout());
    return std::string(function_name_vector.begin(),
                       function_name_vector.end());
  }();

  std::string ir_module_string;
  if (embed_ir_in_executable) {
    ir_module_string = llvm_ir::DumpToString(llvm_module.get());
  }

  TF_RETURN_IF_ERROR(VerifyLlvmModule(*llvm_module));

  // JIT compile the LLVM IR module to in-memory machine code.
  llvm::orc::ThreadSafeModule thread_safe_module(std::move(llvm_module),
                                                 std::move(llvm_context));
  cantFail((*jit)->AddModule(std::move(thread_safe_module)));

  TF_ASSIGN_OR_RETURN(
      auto cpu_executable,
      CpuExecutable::Create(std::move(*jit), std::move(assignment),
                            std::move(module), function_name,
                            std::move(hlo_profile_printer_data),
                            std::move(hlo_profile_index_map)));

  cpu_executable->set_obj_files(std::move(obj_files));

  if (embed_ir_in_executable) {
    cpu_executable->set_ir_module_string(ir_module_string);
  }

  return with_hlo_proto(std::move(cpu_executable));
}

absl::StatusOr<std::unique_ptr<Executable>> CpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    [[maybe_unused]] se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  VLOG(1) << "Compiling: " << module->name();
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrFormat("Compiling [%s] for CPU using JIT", module->name()));
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

  absl::call_once(llvm_command_line_options_initialized,
                  &InitializeLLVMCommandLineOptions, module->config());

  std::unique_ptr<CpuExecutable> cpu_executable;
  TF_ASSIGN_OR_RETURN(cpu_executable,
                      CompileLegacyCpuExecutable(std::move(module)));

  cpu_executable->set_debug_info(
      cpu_executable->buffer_assignment().GetStats().ToString());
  VLOG(1) << "Compilation finished";
  return std::unique_ptr<Executable>(std::move(cpu_executable));
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& aot_options) {
  TF_RET_CHECK(!module_group->empty());
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();

  absl::call_once(llvm_command_line_options_initialized,
                  &InitializeLLVMCommandLineOptions, modules[0]->config());

  // We can pass just one llvm::TargetOptions when we compile the LLVM module,
  // so we bail if the configs have conflicting flags. At the moment, the only
  // flags that need to be consistent are for fast-math.
  for (const auto& fn_and_name :
       {std::make_pair(&DebugOptions::xla_cpu_enable_fast_math,
                       "xla_cpu_enable_fast_math"),
        std::make_pair(&DebugOptions::xla_cpu_fast_math_honor_infs,
                       "xla_cpu_fast_math_honor_infs"),
        std::make_pair(&DebugOptions::xla_cpu_fast_math_honor_nans,
                       "xla_cpu_fast_math_honor_nans")}) {
    // This only works because each of the method pointers above returns a bool.
    // Otherwise we'd have to do some template magic.
    const auto& field_method_ptr = fn_and_name.first;
    const auto& field_name = fn_and_name.second;
    bool first_module_val =
        (modules[0]->config().debug_options().*field_method_ptr)();
    for (int64_t i = 0; i < modules.size(); ++i) {
      bool cur_module_val =
          (modules[i]->config().debug_options().*field_method_ptr)();
      if (first_module_val != cur_module_val) {
        return InvalidArgument(
            "All HLO module configs must have the same value for %s, but "
            "module 0 and %d have different values (%d vs %d).",
            field_name, i, first_module_val, cur_module_val);
      }
    }
  }

  if (aot_options.PlatformId() != se::host::kHostPlatformId) {
    return InvalidArgument("Incompatible AOT compilation platform");
  }
  const CpuAotCompilationOptions& options =
      static_cast<const CpuAotCompilationOptions&>(aot_options);
  llvm::Triple triple(llvm::Triple::normalize(options.triple()));
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), error);
  if (target == nullptr) {
    return Internal("TargetRegistry::lookupTarget failed: %s", error);
  }

  llvm::Reloc::Model reloc_model = llvm::Reloc::Static;
  llvm::PICLevel::Level pic_level = llvm::PICLevel::NotPIC;
  llvm::PIELevel::Level pie_level = llvm::PIELevel::Default;
  switch (options.relocation_model()) {
    case CpuAotCompilationOptions::RelocationModel::Static:
      reloc_model = llvm::Reloc::Static;
      pic_level = llvm::PICLevel::NotPIC;
      pie_level = llvm::PIELevel::Default;
      break;
    case CpuAotCompilationOptions::RelocationModel::SmallPic:
      reloc_model = llvm::Reloc::PIC_;
      pic_level = llvm::PICLevel::SmallPIC;
      pie_level = llvm::PIELevel::Default;
      break;
    case CpuAotCompilationOptions::RelocationModel::BigPic:
      reloc_model = llvm::Reloc::PIC_;
      pic_level = llvm::PICLevel::BigPIC;
      pie_level = llvm::PIELevel::Default;
      break;
    case CpuAotCompilationOptions::RelocationModel::SmallPie:
      reloc_model = llvm::Reloc::PIC_;
      pic_level = llvm::PICLevel::SmallPIC;
      pie_level = llvm::PIELevel::Small;
      break;
    case CpuAotCompilationOptions::RelocationModel::BigPie:
      reloc_model = llvm::Reloc::PIC_;
      pic_level = llvm::PICLevel::BigPIC;
      pie_level = llvm::PIELevel::Large;
      break;
  }
  llvm::CodeGenOptLevel opt_level = CodeGenOptLevel(modules[0]->config());
  std::unique_ptr<llvm::TargetMachine> target_machine =
      absl::WrapUnique(target->createTargetMachine(
          triple.getTriple(), options.cpu_name(), options.features(),
          CompilerTargetOptions(modules[0]->config()), reloc_model,
          std::nullopt, opt_level));

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::MLIRContext mlir_context;
  llvm::LLVMContext llvm_context;

  std::vector<std::unique_ptr<AotCompilationResult>> results;
  for (size_t i = 0; i < modules.size(); ++i) {
    HloModule* module = modules[i].get();
    VLOG(1) << "Compiling ahead-of-time: " << module->name();

    if (!module->has_schedule()) {
      TF_RETURN_IF_ERROR(
          RunHloPasses(module, /*is_aot_compile=*/true, target_machine.get(),
                       /*dummy*/ CompileOptions{},
                       /*is_mlir_compile=*/options.use_mlir_hlo_lowering()));

      TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                          ScheduleModule(module, BufferSizeBytesFunction()));

      // Run buffer analysis on the HLO graph. This analysis figures out which
      // temporary buffers are required to run the computation.
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<BufferAssignment> assignment,
          BufferAssigner::Run(module,
                              std::make_unique<SequentialHloOrdering>(schedule),
                              BufferSizeBytesFunction(), memory_alignment,
                              /*allocate_buffers_for_constants=*/true));
      // BufferAssignment::ToString() includes a header, so no need for us to
      // print one ourselves.
      if (DumpingEnabledForHloModule(*module)) {
        DumpToFileInDirOrStdout(*module, "", "buffer_assignment",
                                assignment->ToString());
      }
      DumpHloModuleIfEnabled(*module, *assignment,
                             absl::StrCat("cpu_", kAfterOptimizationsDumpName));

      absl::flat_hash_map<const HloInstruction*, int64_t>
          instruction_to_profile_idx;
      absl::flat_hash_map<const HloComputation*, int64_t>
          computation_to_profile_idx;
      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map;
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data;

      if (module->config().hlo_profiling_enabled()) {
        TF_RETURN_IF_ERROR(CreateHloProfilingArtifacts(
            *module, &instruction_to_profile_idx, &computation_to_profile_idx,
            &hlo_profile_index_map, &hlo_profile_printer_data));
      }

      LLVMTargetMachineFeatures target_machine_features(target_machine.get());
      std::vector<BufferInfo> buffer_infos =
          CreateBufferInfosFromBufferAssignment(*module, *assignment);
      HloComputation* computation = module->entry_computation();

      // Set required information before emitting IR
      auto llvm_module =
          std::make_unique<llvm::Module>("__compute_module", llvm_context);
      llvm_module->setDataLayout(target_machine->createDataLayout());
      llvm_module->setTargetTriple(triple.getTriple());
      if (pic_level != llvm::PICLevel::NotPIC) {
        llvm_module->setPICLevel(pic_level);
      }
      if (pie_level != llvm::PIELevel::Default) {
        llvm_module->setPIELevel(pie_level);
      }
      IrEmitter ir_emitter(
          &mlir_context, *module, *assignment, llvm_module.get(),
          std::move(instruction_to_profile_idx),
          std::move(computation_to_profile_idx),
          ModuleComputationsTransitivelyContainCustomCall(*module),
          &target_machine_features,
          // TODO(b/66051036): Run full msan for AOT.
          /*emit_code_for_msan=*/false);

      TF_RETURN_IF_ERROR(ir_emitter.EmitConstantGlobals());

      for (ComputationToEmit subcomputation :
           SubcomputationEmissionOrder(computation)) {
        if (subcomputation.computation->IsFusionComputation()) {
          continue;
        }
        TF_RETURN_IF_ERROR(
            ir_emitter
                .EmitComputation(subcomputation.computation,
                                 subcomputation.computation->name(),
                                 /*is_top_level_computation=*/false,
                                 schedule.sequence(subcomputation.computation)
                                     .instructions(),
                                 subcomputation.allow_reassociation)
                .status());
      }
      const std::string& entry_point_name = options.entry_point_name();
      TF_ASSIGN_OR_RETURN(llvm::Function * entry_function,
                          ir_emitter.EmitComputation(
                              computation, entry_point_name,
                              /*is_top_level_computation=*/true,
                              schedule.sequence(computation).instructions(),
                              /*allow_reassociation=*/false));

      CHECK(entry_function->getName() == entry_point_name);

      ModuleHook pre_optimization_ir_hook;
      ModuleHook post_optimization_ir_hook;
      std::tie(pre_optimization_ir_hook, post_optimization_ir_hook) =
          GetIRModuleHooks(*module, user_pre_optimization_hook_,
                           user_post_optimization_hook_);

      // Run the LLVM verifier over the unoptimized LLVM IR.  If it fails, run
      // the pre-optimization IR dump hook before returning.
      {
        absl::Status verify_status = VerifyLlvmModule(*llvm_module);
        if (!verify_status.ok() && pre_optimization_ir_hook) {
          pre_optimization_ir_hook(*llvm_module);
        }
        TF_RETURN_IF_ERROR(verify_status);
      }

      auto post_codegen_hook = [&](const llvm::object::ObjectFile& obj_file) {
        if (!DumpingEnabledForHloModule(*module)) {
          return;
        }
        DumpToFileInDir(*module, /*file_prefix=*/"", /*file_suffix=*/"o",
                        absl::string_view(obj_file.getData().data(),
                                          obj_file.getData().size()));
      };

      CompilerFunctor compiler_functor(
          target_machine.get(), static_cast<int>(opt_level),
          options::OptimizeForSizeRequested(module->config()),
          module->config().debug_options().xla_llvm_disable_expensive_passes(),
          options::SlpVectorizerDisabled(module->config()),
          llvm_ir::GetCpuFastMathFlags(module->config()),
          pre_optimization_ir_hook, post_optimization_ir_hook,
          post_codegen_hook, aot_options.sanitize_dataflow(),
          aot_options.sanitize_abilists_dataflow());
      std::unique_ptr<llvm::MemoryBuffer> object_file =
          cantFail(compiler_functor(*llvm_module));
      ObjectFileData object_file_data(object_file->getBufferStart(),
                                      object_file->getBufferEnd());

      TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                          assignment->GetUniqueTopLevelOutputSlice());

      results.emplace_back(std::make_unique<CpuAotCompilationResult>(
          std::move(object_file_data), std::move(buffer_infos),
          result_slice.index(), std::move(modules[i]),
          std::move(hlo_profile_printer_data)));
    }
  }

  VLOG(1) << "Compilation finished";
  return std::move(results);
}

se::Platform::Id CpuCompiler::PlatformId() const {
  return se::host::kHostPlatformId;
}

HloCostAnalysis::ShapeSizeFunction CpuCompiler::ShapeSizeBytesFunction() const {
  return CpuExecutable::ShapeSizeBytes;
}

namespace {

// This is a result of exporting JIT compiled CpuExecutable to AOT compilation
// result that can be saved on disk and shipped over the wire.
class CpuExecutableAotCompilationResult : public AotCompilationResult {
 public:
  CpuExecutableAotCompilationResult(const HloModule* hlo_module,
                                    const BufferAssignment* buffer_assignment,
                                    std::string_view function_name,
                                    std::string_view obj_file) {
    *proto_.mutable_hlo_module()->mutable_hlo_module() = hlo_module->ToProto();
    *proto_.mutable_buffer_assignment() = buffer_assignment->ToProto();
    proto_.set_entry_function_name(std::string(function_name));
    proto_.set_obj_file(std::string(obj_file));
    *proto_.mutable_hlo_module()->mutable_config() =
        hlo_module->config().ToProto();
    module_ = hlo_module->Clone();
  }

  absl::StatusOr<std::string> SerializeAsString() const override {
    return proto_.SerializeAsString();
  }

  static absl::StatusOr<std::unique_ptr<CpuExecutableAotCompilationResult>>
  FromString(const std::string& serialized) {
    CompilationResultProto proto;
    if (!proto.ParseFromString(serialized)) {
      return Internal(
          "Failed to parse serialized CpuExecutableAotCompilationResult.");
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProtoWithConfig(proto.hlo_module()));

    return std::unique_ptr<CpuExecutableAotCompilationResult>(
        new CpuExecutableAotCompilationResult(proto, std::move(module)));
  }

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, const se::StreamExecutor* stream_exec) const override;

  const HloModule* optimized_module() const override { return module_.get(); }

  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }

 private:
  explicit CpuExecutableAotCompilationResult(CompilationResultProto proto,
                                             std::unique_ptr<HloModule> module)
      : proto_(std::move(proto)), module_(std::move(module)) {}

  CompilationResultProto proto_;
  std::unique_ptr<HloModule> module_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Executable>>
CpuExecutableAotCompilationResult::LoadExecutable(
    Compiler* compiler, const se::StreamExecutor* stream_exec) const {
  // Recreate HloModule from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProtoWithConfig(proto_.hlo_module()));

  // Recreate BufferAssignment from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssignment::FromProto(proto_.buffer_assignment(), module.get(),
                                  compiler->BufferSizeBytesFunction(),
                                  /*can_share_buffer=*/nullptr));

  auto jit = SimpleOrcJIT::Create(
      CompilerTargetOptions(module->config()),
      CodeGenOptLevel(module->config()),
      options::OptimizeForSizeRequested(module->config()),
      module->config().debug_options().xla_llvm_disable_expensive_passes(),
      options::SlpVectorizerDisabled(module->config()),
      llvm_ir::GetCpuFastMathFlags(module->config()),
      /*pre_optimization_hook=*/nullptr, /*post_optimization_hook=*/nullptr,
      /*post_codegen_hook=*/nullptr);
  if (!jit) {
    return Internal("Creating JIT failed: %s", llvm::toString(jit.takeError()));
  }

  // Create a named buffer from compiled object file.
  llvm::StringRef data(proto_.obj_file().data(), proto_.obj_file().size());
  auto obj_file =
      llvm::MemoryBuffer::getMemBuffer(data, proto_.entry_function_name());

  cantFail((*jit)->AddObjFile(std::move(obj_file)));

  TF_ASSIGN_OR_RETURN(
      auto cpu_executable,
      CpuExecutable::Create(std::move(*jit), std::move(buffer_assignment),
                            std::move(module), proto_.entry_function_name(),
                            nullptr, nullptr));

  // Dump computation proto state and buffer assignment for
  // GetCompiledMemoryStats results.
  auto hlo_proto = std::make_unique<HloProto>();
  *hlo_proto->mutable_hlo_module() = cpu_executable->module().ToProto();
  *hlo_proto->mutable_buffer_assignment() =
      cpu_executable->buffer_assignment().ToProto();
  cpu_executable->set_hlo_proto(std::move(hlo_proto));

  return cpu_executable;
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>> CpuCompiler::Export(
    Executable* executable) const {
  auto* cpu_executable = tensorflow::down_cast<CpuExecutable*>(executable);
  if (!cpu_executable)
    return Internal("Could not downcast Executable to CpuExecutable");

  if (cpu_executable->obj_files().size() != 1) {
    return absl::InternalError(
        absl::StrCat("Can't export CPU execuable, expected exactly one object "
                     "file but got: ",
                     cpu_executable->obj_files().size()));
  }

  return {std::make_unique<CpuExecutableAotCompilationResult>(
      &cpu_executable->module(), &cpu_executable->buffer_assignment(),
      cpu_executable->module_name(), cpu_executable->obj_files()[0])};
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
CpuCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  return CpuExecutableAotCompilationResult::FromString(serialized_aot_result);
}

}  // namespace cpu
}  // namespace xla
