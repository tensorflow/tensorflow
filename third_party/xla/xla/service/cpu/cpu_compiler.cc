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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/Config/Disassemblers.def.inc"
// IWYU pragma: no_include "llvm/Config/Targets.def.inc"

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/object_loader.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/transforms/xnn_graph_fusion.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/indexed_array_analysis.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/bitcast_dtypes_expander.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/comparison_expander.h"
#include "xla/hlo/transforms/expanders/dot_decomposer.h"
#include "xla/hlo/transforms/expanders/dynamic_index_splitter.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/logistic_expander.h"
#include "xla/hlo/transforms/expanders/optimization_barrier_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/hlo/transforms/expanders/reduce_decomposer.h"
#include "xla/hlo/transforms/expanders/reshape_decomposer.h"
#include "xla/hlo/transforms/expanders/rng_bit_generator_expander.h"
#include "xla/hlo/transforms/expanders/rng_expander.h"
#include "xla/hlo/transforms/expanders/stochastic_convert_decomposer.h"
#include "xla/hlo/transforms/literal_canonicalizer.h"
#include "xla/hlo/transforms/operand_upcaster.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/batch_dot_simplification.h"
#include "xla/hlo/transforms/simplifiers/broadcast_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/conditional_canonicalizer.h"
#include "xla/hlo/transforms/simplifiers/convolution_group_converter.h"
#include "xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/hlo/transforms/simplifiers/optimize_input_output_buffer_alias.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_rewriter.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/result_caster.h"
#include "xla/hlo/transforms/simplifiers/sort_simplifier.h"
#include "xla/hlo/transforms/simplifiers/sub_byte_normalization.h"
#include "xla/hlo/transforms/simplifiers/tree_reduction_rewriter.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/literal_pool.h"
#include "xla/map_util.h"
#include "xla/mlir_hlo/transforms/passes.h"
#include "xla/service/all_reduce_promotion.h"
#include "xla/service/all_to_all_decomposer.h"
#include "xla/service/batched_gather_scatter_normalizer.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/call_inliner.h"
#include "xla/service/change_op_data_type.h"
#include "xla/service/compiler.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/conditional_to_select.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu/buffer_info_util.h"
#include "xla/service/cpu/conv_canonicalization.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/service/cpu/cpu_layout_assignment.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/ir_emitter2.h"
#include "xla/service/cpu/metrics.h"
#include "xla/service/cpu/parallel_task_assignment.h"
#include "xla/service/cpu/runtime_symbol_generator.h"
#include "xla/service/cpu/small_while_loop_hoisting_pass.h"
#include "xla/service/cpu/thunk_emitter.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/dump.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/executable.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/llvm_compiler.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/map_inliner.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/select_and_scatter_expander.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/service/spmd/shardy/shardy_xla_pass.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_invariant_code_motion.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

#ifdef TF_LLVM_X86_AVAILABLE
#include "llvm/TargetParser/X86TargetParser.h"
#endif

#if defined(INTEL_MKL)
#include "xla/hlo/transforms/simplifiers/simplify_fp_conversions.h"
#include "xla/service/cpu/cpu_float_support.h"
#include "xla/service/cpu/onednn_contraction_rewriter.h"
#include "xla/service/cpu/onednn_ops_rewriter.h"
#endif

namespace xla {
namespace {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

// A module identifier (prefix) for emitted LLVM modules.
static constexpr absl::string_view kXlaModuleIdentifier = "__compute_module";

// Returns a global (per-process) thread pool for XLA CPU compilation tasks.
static tsl::thread::ThreadPool* GetCompilationThreadPool() {
  // LLVM compilation has a lot of memory-bound pointer chasing and not
  // so much CPU-bound work. Based on profiling a few examples, 32 threads seems
  // to be enough to achieve maximum parallel compilation speedup.
  static constexpr int kMaxCompilationThreads = 32;

  // On Mac OS the default stack size is 512KiB, this is too small for compiling
  // reasonably sized programs
  tsl::ThreadOptions thread_options;
  thread_options.stack_size = 4 * 1024 * 1024;  // 4 MB

  static auto* thread_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), thread_options, "xla-cpu-llvm-codegen",
      std::min(kMaxCompilationThreads, tsl::port::MaxParallelism()));
  return thread_pool;
}

// Returns task runner that uses the global compilation thread pool.
static cpu::JitCompiler::TaskRunner GetCompilationTaskRunner() {
  return [](cpu::JitCompiler::Task task) {
    GetCompilationThreadPool()->Schedule(std::move(task));
  };
}

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

std::unique_ptr<HloPassFix<HloPassPipeline>> CreateSimplificationPipeline(
    absl::string_view name, HloModule* module) {
  // Run the following passes to a fixed point.
  auto pipeline =
      std::make_unique<HloPassFix<HloPassPipeline>>(std::string(name));
  AddHloVerifier(pipeline.get(), HloVerifierOpts{},
                 /*debug_only=*/true);

  AlgebraicSimplifierOptions options;
  options.set_enable_dot_strength_reduction(false);
  // TODO(b/209827141): XLA:CPU doesn't propagate NaN through min/max, but
  // other platforms do, so it should be changed.
  options.set_minmax_propagate_nan(false);
  options.set_supports_non_canonical_dots(false);
  options.set_executing_on_cpu(true);
  pipeline->AddPass<AlgebraicSimplifier>(options);
  pipeline->AddPass<SortSimplifier>();
  pipeline->AddPass<HloDCE>();
  pipeline->AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);

  // Needs to happen after algebraic simplifier.
  pipeline->AddPass<TreeReductionRewriter>();

  // BatchNormExpander can create zero-sized ops, so zero-sized HLO
  // elimination has to come after that pass.
  pipeline->AddPass<ZeroSizedHloElimination>();

  pipeline->AddPass<WhileLoopInvariantCodeMotion>();
  pipeline->AddPass<TupleSimplifier>();
  pipeline->AddPass<WhileLoopConstantSinking>();
  pipeline->AddPass<WhileLoopSimplifier>();

  // TODO(b/134075051): Re-enable after b/134075051 is fixed.
  // pipeline->AddPass<SliceSinker>();

  pipeline->AddPass<HloDCE>();
  pipeline->AddPass<ReshapeMover>();
  pipeline->AddPass<HloConstantFolding>(
      options::FoldAllConstants(module->config())
          ? HloConstantFolding::Level::kAgressive
          : HloConstantFolding::Level::kDefault);
  pipeline->AddPass<ConditionalSimplifier>();

  return pipeline;
}

}  // namespace

absl::Status CpuCompiler::RunHloPassesThroughLayoutAssn(
    HloModule* module, bool is_aot_compile,
    TargetMachineFeatures* target_machine_features) {
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
    if (module->config().use_shardy_partitioner()) {
      spmd_pipeline.AddPass<sdy::ShardyXLA>();
    } else {
      spmd_pipeline.AddPass<ShardingPropagation>(
          /*is_spmd=*/true, /*propagate_metadata=*/false,
          module->config().allow_spmd_sharding_propagation_to_output(),
          module->config().allow_spmd_sharding_propagation_to_parameters());
    }
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
  pipeline.AddPass<BatchedGatherScatterNormalizer>();
  pipeline.AddPass<ResultCaster>();
  pipeline.AddPass<OperandUpcaster>();

  // Expand random number generation.
  pipeline.AddPass<RngExpander>();
  pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);

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
#if defined(INTEL_MKL)
  // AOT compiled code runs in single thread.
  bool is_thunk_runtime =
      module->config().debug_options().xla_cpu_use_thunk_runtime();
  if (!is_aot_compile && !is_thunk_runtime) {
    // Placing OneDnnOpsRewriter here to match the flax patterns
    // TODO: Decide where would be the appropriate place for this pass to make
    // it more generic
    // TODO - intel: Name of the pass might seem redundant as oneDnnRewriter,
    // but in future plan to rename oneDNNrewriter to specific to onednn matmul
    pipeline.AddPass<OneDnnOpsRewriter>();
  }
#endif  // INTEL_MKL

  // Promote BF16 all-reduce to F32.
  const std::pair<PrimitiveType, PrimitiveType> ar_promoted_types[] = {
      {BF16, F32}};
  pipeline.AddPass<AllReducePromotion>(ar_promoted_types);
  // Convert BF16 and F8 operations to F32 and F16 respectively so that the CPU
  // backend can support BF16/F8 operations without directly implementing a
  // BF16/F8 lowering for most ops.
  FloatSupport bf16_support(BF16);
#if defined(INTEL_MKL)
  CpuFloatSupport onednn_bf16_support(BF16);
  if (!is_aot_compile && !is_thunk_runtime) {
    pipeline.AddPass<FloatNormalization>(&onednn_bf16_support);
  } else {
    pipeline.AddPass<FloatNormalization>(&bf16_support);
  }
#else
  pipeline.AddPass<FloatNormalization>(&bf16_support);
#endif
  FloatSupport f8e5m2_support(F8E5M2, F16);
  pipeline.AddPass<FloatNormalization>(&f8e5m2_support);
  FloatSupport f8e4m3_support(F8E4M3, F16);
  pipeline.AddPass<FloatNormalization>(&f8e4m3_support);
  FloatSupport f8e4m3fn_support(F8E4M3FN, F16);
  pipeline.AddPass<FloatNormalization>(&f8e4m3fn_support);
  FloatSupport f8e4m3b11fnuz_support(F8E4M3B11FNUZ, F16);
  pipeline.AddPass<FloatNormalization>(&f8e4m3b11fnuz_support);
  FloatSupport f8e5m2fnuz_support(F8E5M2FNUZ, F16);
  pipeline.AddPass<FloatNormalization>(&f8e5m2fnuz_support);
  FloatSupport f8e4m3fnuz_support(F8E4M3FNUZ, F16);
  pipeline.AddPass<FloatNormalization>(&f8e4m3fnuz_support);
  FloatSupport f8e3m4_support(F8E3M4, F16);
  pipeline.AddPass<FloatNormalization>(&f8e3m4_support);
  FloatSupport s4_support(S4, S8);
  pipeline.AddPass<FloatNormalization>(&s4_support);
  FloatSupport u4_support(U4, U8);
  pipeline.AddPass<FloatNormalization>(&u4_support);
  FloatSupport f4e2m1fn_support(F4E2M1FN, F16);
  pipeline.AddPass<FloatNormalization>(&f4e2m1fn_support);
  FloatSupport f8e8m0fnu_support(F8E8M0FNU, F32);
  pipeline.AddPass<FloatNormalization>(&f8e8m0fnu_support);
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

  if (module->config()
          .debug_options()
          .xla_reduce_window_rewrite_base_length() != 0) {
    pipeline.AddPass<HloPassFix<ReduceWindowRewriter>>(
        module->config()
            .debug_options()
            .xla_reduce_window_rewrite_base_length());
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

  pipeline.AddPass(CreateSimplificationPipeline("simplification", module));

  // Scatter expander is sandwiched between two simplification pipelines to
  // enable constant folding with the original scatter instructions (which is
  // more efficient than with the expanded version) but then to also ensure that
  // the resulting while loops are simplified.
  pipeline.AddPass<SelectAndScatterExpander>();
  pipeline.AddPass<ScatterExpander>(ScatterExpander::kEliminateAllScatters);

  pipeline.AddPass(CreateSimplificationPipeline(
      "post_scatter_expansion_simplification", module));

  pipeline.AddPass<BitcastDtypesExpander>();

  pipeline.AddPass<TopkRewriter>([](const HloSortInstruction* sort, int64_t) {
    return sort->operand(0)->shape().element_type() == F32;
  });
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

  // Annotate while loops with statically known trip counts, so that at run time
  // we can avoid running the loop condition computations.
  pipeline.AddPass<WhileLoopTripCountAnnotator>();

  // Layout assignment uses alias analysis, which requires the call graph to be
  // flattened.
  pipeline.AddPass<FlattenCallGraph>();
  ChannelLayoutConstraints layout_constraints;
  pipeline.AddPass<CpuLayoutAssignment>(
      module->mutable_entry_computation_layout(), target_machine_features,
      &layout_constraints);
  // Run SubByteNormalization because CpuLayoutAssignment may modify a
  // Layout's element_size_in_bits field.
  pipeline.AddPass<SubByteNormalization>(
      SubByteNormalization::SET_ELEMENT_SIZE);

  // Finally canonicalize all literals larger than 1024 bytes in the module to
  // reuse the same literal across multiple HLO modules.
  pipeline.AddPass<LiteralCanonicalizer>(LiteralPool::Default(),
                                         /*min_size_bytes=*/1024);

  return pipeline.Run(module).status();
}

absl::Status CpuCompiler::RunHloPassesAfterLayoutAssn(
    HloModule* module, bool is_aot_compile,
    TargetMachineFeatures* target_machine_features,
    const CompileOptions& compile_options) {
  HloPassPipeline pipeline("HLO passes after layout assignment");

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

#if defined(INTEL_MKL)
  auto& debug_options = module->config().debug_options();
  bool is_thunk_runtime = debug_options.xla_cpu_use_thunk_runtime();

  // AOT compiled code runs in single thread.
  if (!is_aot_compile && !is_thunk_runtime) {
    auto debug_options = module->config().debug_options();
    // Run SimplifyFPConversions pass to simplify the BF16 pattern and make it
    // easier to match.
    // Remove `f32 -> bf16 -> f32` casts inserted by bf16 normalization.
    if (debug_options.xla_allow_excess_precision()) {
      pipeline.AddPass<SimplifyFPConversions>();
    }
    pipeline.AddPass<OneDnnContractionRewriter>(max_parallelism,
                                                compile_options.thread_pool);
    // Run SimplifyFPConversions pass again to remove redundant Convert ops
    // that may exist as a result of running OneDnnContractionRewriter pass.
    if (debug_options.xla_allow_excess_precision()) {
      pipeline.AddPass<SimplifyFPConversions>();
    }
  }
#endif  // INTEL_MKL

  if (module->config()
          .debug_options()
          .xla_cpu_experimental_xnn_graph_fusion_mode() !=
      DebugOptions::XNN_GRAPH_FUSION_MODE_DISABLED) {
    pipeline.AddPass<XnnGraphFusion>();
  }

  // Add a fusion pass now that layout assignment is done.
  pipeline.AddPass<CpuInstructionFusion>();

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  // Run this to a fixed point.
  [&pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
       "simplification after layout assignment")] {
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
    options.set_executing_on_cpu(true);
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

  // If enabled we'll use more precise region based analysis for copy removal.
  if (module->config()
          .debug_options()
          .xla_cpu_copy_insertion_use_region_analysis()) {
    pipeline.AddPass<CopyInsertion>(
        /*can_share_buffer=*/nullptr,
        /*use_region_based_live_range_analysis=*/-1);
  } else {
    pipeline.AddPass<CopyInsertion>();
  }

  // The hoisting of small while loops is only useful in the context of the
  // thunk runtime.
  if (module->config().debug_options().xla_cpu_use_thunk_runtime()) {
    pipeline.AddPass<SmallWhileLoopHoistingPass>(
        /*small_buffer_access_size*/ 256);
  }

  pipeline.AddPass<HloDCE>();
  return pipeline.Run(module).status();
}

absl::Status CpuCompiler::RunHloPasses(HloModule* module, bool is_aot_compile,
                                       llvm::TargetMachine* target_machine,
                                       const CompileOptions& compile_options) {
  TargetMachineFeatures target_machine_features(target_machine);
  TF_RETURN_IF_ERROR(RunHloPassesThroughLayoutAssn(module, is_aot_compile,
                                                   &target_machine_features));

  return RunHloPassesAfterLayoutAssn(module, is_aot_compile,
                                     &target_machine_features, compile_options);
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

    // Include LLVM module identifier suffix in case `llvm_module` is just a
    // part of the original LLVM module constructed by the XLA.
    absl::string_view id = llvm_module.getModuleIdentifier();
    size_t pos = std::min(id.size(), 1 + kXlaModuleIdentifier.size());
    llvm_ir::DumpIrIfEnabled(*hlo_module_ptr, llvm_module, optimized,
                             /*filename_suffix=*/id.substr(pos));
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
  auto& config = module->config();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> jit_target_machine,
      JitCompiler::InferTargetMachine(
          CompilerTargetOptions(config), IrCompiler::GetCodeGenOptLevel(config),
          CpuFeatureFromString(config.debug_options().xla_cpu_max_isa())));

  TF_RETURN_IF_ERROR(RunHloPasses(module.get(), /*is_aot_compile=*/false,
                                  jit_target_machine.get(),
                                  /*compile_options=*/options));
  return std::move(module);
}

namespace {

static void DumpModuleToFile(const llvm::Module& llvm_module,
                             const llvm::object::ObjectFile& obj_file,
                             const HloModule& hlo_module) {
  absl::string_view id = llvm_module.getModuleIdentifier();
  size_t pos = std::min(id.size(), 1 + kXlaModuleIdentifier.size());
  auto get_file_suffix = [&]() {
    std::vector<absl::string_view> parts = {"obj-file"};
    parts.reserve(3);
    absl::string_view middle_name = id.substr(pos);
    if (!middle_name.empty()) {
      parts.push_back(middle_name);
    }
    parts.push_back("o");
    return absl::StrJoin(parts, ".");
  };
  DumpToFileInDir(
      hlo_module, /*file_prefix=*/"", get_file_suffix(),
      absl::string_view(obj_file.getData().data(), obj_file.getData().size()));
}

// Post-compilation callback functor for use by SimpleOrcJIT.
//
// Dumps machine code if dumping is enabled for the module.
static std::function<void(const llvm::Module&, const llvm::object::ObjectFile&)>
CreateOrcJITPostCompilationHook(const HloModule* hlo_module,
                                std::vector<std::string>* obj_files) {
  return [=](const llvm::Module& llvm_module,
             const llvm::object::ObjectFile& obj_file) {
    if (obj_files) obj_files->push_back(obj_file.getData().str());

    if (DumpingEnabledForHloModule(*hlo_module)) {
      DumpModuleToFile(llvm_module, obj_file, *hlo_module);
    }
  };
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

// Removes unused globals and function declarations from the LLVM module.
//
// After splitting LLVM module into multiple parts, we end up with unused
// symbols in each part: external globals and function declarations. We don't
// support linking across modules added to SimpleOrcJIT, and we don't need it,
// because we never construct LLVM IR that might require cross-module linking,
// so we can just remove unused symbols from each part.
static void RemoveUnusedSymbols(llvm::Module& module) {
  llvm::SmallVector<llvm::GlobalVariable*> unused_globals;
  llvm::SmallVector<llvm::Function*> unused_functions;

  for (llvm::GlobalVariable& gv : module.globals()) {
    if (gv.use_empty()) unused_globals.push_back(&gv);
  }
  for (llvm::Function& f : module.functions()) {
    if (f.isDeclaration() && f.use_empty()) unused_functions.push_back(&f);
  }

  for (auto* gv : unused_globals) {
    module.eraseGlobalVariable(gv);
  }
  for (auto* f : unused_functions) {
    f->eraseFromParent();
  }
}

// Clones a ThreadSafeModule from the given LLVM module in a new LLVM context.
//
// To enable parallel compilation, each LLVM module has to be owned by a
// separate LLVM context. We take each part of the original module after a
// split, and clone it into a new LLVM context.
static llvm::orc::ThreadSafeModule CloneAsThreadSafeModule(
    int64_t part, std::unique_ptr<llvm::Module> module) {
  TraceMe trace([&] {
    return TraceMeEncode("CpuCompiler::CloneAsThreadSafeModule",
                         {{"part", part}});
  });

  // There is no way to clone a module from one context to another, so we need
  // to serialize the module to bitcode and parse it back into the new context.
  llvm::SmallString<0> bc;
  llvm::raw_svector_ostream bcos(bc);
  llvm::WriteBitcodeToFile(*module, bcos);

  // Parse module back into its own LLVM context.
  auto clone_context = std::make_unique<llvm::LLVMContext>();
  auto clone_module = llvm::parseBitcodeFile(
      llvm::MemoryBufferRef(
          llvm::StringRef(bc.data(), bc.size()),
          absl::StrFormat("%s_part_%02d", kXlaModuleIdentifier, part)),
      *clone_context);

  return llvm::orc::ThreadSafeModule(std::move(*clone_module),
                                     std::move(clone_context));
}

namespace {
// Compiled symbols (kernels and comparators) from a single LLVM module part.
struct CompiledSymbolsPart {
  std::vector<IrEmitter2::KernelInfo> kernels;
  std::vector<IrEmitter2::ComparatorInfo> comparators;
};
}  // namespace

// Collect IrEmitter2 symbols that got into the LLVM module part. We issue
// compilation tasks in parallel, and to maximize concurrency we don't issue
// separate compilation tasks that compile symbols from the same module.
static CompiledSymbolsPart CollectCompiledSymbolsPart(
    const IrEmitter2& ir_emitter, const llvm::Module& module) {
  CompiledSymbolsPart syms;

  auto find_kernel =
      [&](llvm::StringRef name) -> std::optional<IrEmitter2::KernelInfo> {
    for (auto& k : ir_emitter.kernels()) {
      if (k.name == name) return k;
    }
    return std::nullopt;
  };

  auto find_comparator =
      [&](llvm::StringRef name) -> std::optional<IrEmitter2::ComparatorInfo> {
    for (auto& c : ir_emitter.comparators()) {
      if (c.name == name) return c;
    }
    return std::nullopt;
  };

  for (auto& f : module.functions()) {
    if (auto kernel = find_kernel(f.getName())) {
      syms.kernels.push_back(*kernel);
    }
    if (auto comparator = find_comparator(f.getName())) {
      syms.comparators.push_back(*comparator);
    }
  }

  return syms;
}

// If LLVM module has large constants constructed from literals, we don't want
// to split it, because it will cause us to copy large constants across module
// parts. We should not be storing large constants in LLVM IR in a first place,
// but while we do that, we have to be extra-careful, or it leads to extremely
// long compilation times, OOMs and timeouts.
//
// TODO(b/361800465): Figure out how to avoid putting large constants into
// LLVM IR in the first place.
static bool HasLargeConstants(llvm::Module& module) {
  static constexpr int kMaxConstantSize = 10000;
  for (auto& g : module.globals()) {
    if (!g.hasInitializer()) {
      continue;
    }

    llvm::Constant* initializer = g.getInitializer();
    if (auto* arr = llvm::dyn_cast<llvm::ArrayType>(initializer->getType())) {
      if (arr->getNumElements() > kMaxConstantSize) return true;
    }
  }
  return false;
}

inline void VlogMaxIsa(absl::string_view max_cpu_isa) {
  if (VLOG_IS_ON(1) && !max_cpu_isa.empty()) {
    if (tsl::port::IsX86CPU()) {
      VLOG(1) << "`xla_cpu_max_isa` is set. Will not use features newer than: "
              << max_cpu_isa;
    } else {
      VLOG(1) << "`xla_cpu_max_isa` is set to `" << max_cpu_isa
              << "`. This flag is not supported on non-x86 CPUs yet.";
    }
  }
}

// We keep HloProto in the CpuExecutable, but we don't need to keep literals
// payload in it as we use it only for debugging and memory analysis.
static void StripPayloadFromLiteralProto(HloProto& proto) {
  auto* module = proto.mutable_hlo_module();
  for (auto& computation : *module->mutable_computations()) {
    for (auto& instruction : *computation.mutable_instructions()) {
      // We only keep literal shape to correctly estimate memory usage of the
      // HLO module, but we don't need the actual literal data.
      if (instruction.has_literal()) {
        LiteralProto literal;
        *literal.mutable_shape() = instruction.literal().shape();
        *instruction.mutable_literal() = std::move(literal);
      }
    }
  }
}

// Extracts the given set of kernels from the original module.
// Returns a new module with the extracted kernels.
static absl::StatusOr<std::unique_ptr<llvm::Module>> ExtractKernelsFromModule(
    llvm::Module* original_module,
    absl::flat_hash_set<llvm::StringRef> kernels) {
  // Clone into a new module, only keeping definitions of the relevant kernels.
  auto should_clone_definition = [&kernels](const llvm::GlobalValue* gv) {
    if (auto* func = llvm::dyn_cast<llvm::Function>(gv)) {
      return kernels.contains(func->getName());
    }
    return false;
  };
  llvm::ValueToValueMapTy vmap;
  std::unique_ptr<llvm::Module> module =
      llvm::CloneModule(*original_module, vmap, should_clone_definition);

  // Erase the cloned symbols from the original module.
  for (const auto& kernel_name : kernels) {
    llvm::Function* to_be_removed = original_module->getFunction(kernel_name);
    if (to_be_removed == nullptr) {
      return Internal("Cannot remove kernel %s: cannot be found in module %s",
                      kernel_name, original_module->getName());
    }
    to_be_removed->eraseFromParent();
  }
  return module;
}

static void AddXlaBackendExtraOptionsAsModuleFlag(
    llvm::Module* llvm_module, llvm::StringRef backend_extra_options) {
  auto* options_mdstring =
      llvm::MDString::get(llvm_module->getContext(), backend_extra_options);
  llvm_module->addModuleFlag(llvm::Module::Error, "xla_backend_extra_options",
                             options_mdstring);
}

absl::StatusOr<std::unique_ptr<CpuExecutable>>
CpuCompiler::CompileCpuExecutable(std::unique_ptr<HloModule> module) {
  TraceMe trace([&] {
    return TraceMeEncode("CpuCompiler::CompileCpuExecutable",
                         {{"name", module->name()}});
  });

  ModuleHook pre_optimization_ir_hook;
  ModuleHook post_optimization_ir_hook;
  std::tie(pre_optimization_ir_hook, post_optimization_ir_hook) =
      GetIRModuleHooks(*module, user_pre_optimization_hook_,
                       user_post_optimization_hook_);

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::MLIRContext mlir_context;
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module =
      std::make_unique<llvm::Module>(kXlaModuleIdentifier, *llvm_context);

  const DebugOptions& debug_options = module->config().debug_options();

  // We collect compiled object files (machine code) so we can export
  // CpuExecutable to an AOT compilation result.
  std::vector<std::string> obj_files;

  // We split LLVM module and distribute it across separate DyLibs to enable
  // parallel compilation at run time.
  size_t parallel_codegen_split_count =
      debug_options.xla_cpu_parallel_codegen_split_count();
  VlogMaxIsa(debug_options.xla_cpu_max_isa());

  const HloModuleConfig& config = module->config();

  // Options for compiling LLVM IR to machine code.
  IrCompiler::Options ir_compiler_options{
      /*optimization_level=*/IrCompiler::GetCodeGenOptLevel(config),
      /*optimize_for_size=*/options::OptimizeForSizeRequested(config),
      /*fast_math_flags=*/llvm_ir::GetCpuFastMathFlags(config),
      /*disable_expensive_passes=*/
      debug_options.xla_llvm_disable_expensive_passes(),
      /*slp_vectorizer_disabled=*/options::SlpVectorizerDisabled(config),
      /*disable_loop_unrolling=*/options::DisableLoopUnrolling(config),
  };

  // Compiler hooks to intercept compiled LLVM IR modules.
  IrCompiler::CompilationHooks ir_compiler_hooks{
      pre_optimization_ir_hook,
      post_optimization_ir_hook,
      CreateOrcJITPostCompilationHook(module.get(), &obj_files),
  };

  // Definition generator to link with XLA:CPU host runtime symbols.
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const llvm::DataLayout& data_layout) {
        return std::make_unique<RuntimeSymbolGenerator>(data_layout);
      };

  // Options for orchestrating the JIT compilation process.
  JitCompiler::Options jit_compiler_options{
      std::move(ir_compiler_options),
      std::move(ir_compiler_hooks),
      /*num_dylibs=*/parallel_codegen_split_count,
      /*definition_generator=*/std::move(definition_generator),
      /*max_cpu_isa=*/CpuFeatureFromString(debug_options.xla_cpu_max_isa()),
  };

  TF_ASSIGN_OR_RETURN(
      JitCompiler jit_compiler,
      JitCompiler::Create(CompilerTargetOptions(module->config()),
                          std::move(jit_compiler_options),
                          GetCompilationTaskRunner()));

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

  TF_ASSIGN_OR_RETURN(HloSchedule schedule, CreateHloSchedule(*module));
  TF_RETURN_IF_ERROR(module->set_schedule(schedule));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                      CreateBufferAssignment(*module));
  DumpHloModuleIfEnabled(*module, *assignment,
                         absl::StrCat("cpu_", kAfterOptimizationsDumpName));

  // Dump computation proto state and buffer assignment for
  // GetCompiledMemoryStats results.
  auto with_hlo_proto = [&](std::unique_ptr<CpuExecutable> cpu_executable) {
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_hlo_module() = cpu_executable->module().ToProto();
    *hlo_proto->mutable_buffer_assignment() =
        cpu_executable->buffer_assignment().ToProto();
    StripPayloadFromLiteralProto(*hlo_proto);
    cpu_executable->set_hlo_proto(std::move(hlo_proto));
    return cpu_executable;
  };

  TargetMachineFeatures target_machine_features(jit_compiler.target_machine());

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

  // If we use Thunk runtime then instead of emitting LLVM function for the
  // entry computation we emit a sequence of thunks that implement the
  // computation as a sequence of interpreted commands.
  if (module->config().debug_options().xla_cpu_use_thunk_runtime()) {
    // The thunk runtime manages large constants, therefore we only emit
    // small ones.
    TF_RETURN_IF_ERROR(nested_ir_emitter.EmitSmallConstantGlobals());

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

    std::string ir_module_string;
    if (embed_ir_in_executable) {
      std::string emitter2_ir = llvm_ir::DumpToString(llvm_module.get());

      auto thunk_kernel_fmt = [](std::string* out,
                                 const ThunkEmitter::EmittedKernel& kernel) {
        absl::StrAppend(
            out, llvm_ir::DumpToString(kernel.module.getModuleUnlocked()));
      };
      std::string thunks_ir =
          absl::StrJoin(thunk_emitter.kernels(), "\n", thunk_kernel_fmt);

      ir_module_string = absl::StrCat(emitter2_ir, "\n", thunks_ir);
    }

    TF_RETURN_IF_ERROR(VerifyLlvmModule(*llvm_module));
    for (const auto& [name, module] : thunk_emitter.kernels()) {
      TF_RETURN_IF_ERROR(VerifyLlvmModule(*module.getModuleUnlocked()));
    }

    // Some kernels have to be compiled separately because they have
    // extra backend options.
    int num_extra_functions = 0;
    using BackendOptions = llvm::StringRef;
    using Kernel = llvm::StringRef;
    absl::flat_hash_map<BackendOptions, absl::flat_hash_set<Kernel>>
        backend_extra_options_to_kernels;
    for (const auto& k : ir_emitter2.kernels()) {
      if (k.backend_extra_options.empty()) continue;
      auto [_, inserted] =
          backend_extra_options_to_kernels[k.backend_extra_options].insert(
              k.name);
      CHECK(inserted) << "Kernel " << k.name << " is not unique";
      num_extra_functions++;
    }
    const int num_extra_parts = backend_extra_options_to_kernels.size();
    // We assign one dylib to each set of kernels that have the same extra
    // backend options. We do this because we work under the assumption that
    // very few kernels will set extra options, and if they do, the options are
    // likely to be identical.
    if (num_extra_parts >= parallel_codegen_split_count) {
      return Internal(
          "Too many extra compilation parts due to non-default options (%d). "
          "Consider reducing this number or increasing "
          "parallel_codegen_split_count (%d)",
          num_extra_parts, parallel_codegen_split_count);
    }

    // We define the number of module parts based on the total number of
    // compiled functions (kernels and comparators) that are called from thunks,
    // and the maximum number of parts that we want to split the module into.
    size_t num_compiled_functions = ir_emitter2.kernels().size() +
                                    ir_emitter2.comparators().size() +
                                    thunk_emitter.kernels().size();
    size_t num_default_parts =
        std::min(num_compiled_functions - num_extra_functions,
                 parallel_codegen_split_count - num_extra_parts);

    // JIT compile the LLVM IR module to in-memory machine code. We split the
    // module into `num_jit_dylibs` parts to allow parallel compilation. In
    // practice, all of the kernel functions are independent and don't call each
    // other, so we can compile each individual part in parallel. We split
    // module preserving locals, which should guarantee that all thread local
    // computations end up in the same module with the corresponding kernel.

    // Collect all compiled symbols grouped by LLVM module part, so that we can
    // issue compile tasks in parallel without any interference.
    std::vector<CompiledSymbolsPart> compiled_parts;

    VLOG(2) << "Compile LLVM module with " << ir_emitter2.kernels().size()
            << " kernels and " << ir_emitter2.comparators().size()
            << " comparators";

    int dylib_index = 0;
    auto add_jit_module = [&](std::unique_ptr<llvm::Module> llvm_module_part) {
      // Collect symbols that are compiled in this LLVM module part.
      RemoveUnusedSymbols(*llvm_module_part);
      compiled_parts.push_back(
          CollectCompiledSymbolsPart(ir_emitter2, *llvm_module_part));

      std::string dump = llvm_ir::DumpToString(llvm_module_part.get());
      VLOG(5) << "Adding compilation module:\n" << dump;

      // Clone LLVM module part into its own thread safe context.
      auto tsm =
          CloneAsThreadSafeModule(dylib_index, std::move(llvm_module_part));
      TF_CHECK_OK(jit_compiler.AddModule(std::move(tsm), dylib_index++));
    };

    // If there are extra parts, compile them first, since we must
    // remove the affected kernels from the LLVM module.
    if (num_extra_parts > 0) {
      TraceMe trace([&] {
        return TraceMeEncode("CompileExtraKernels",
                             {{"num_extra_parts", num_extra_parts}});
      });
      for (const auto& [backend_extra_options, kernels] :
           backend_extra_options_to_kernels) {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<llvm::Module> new_module,
            ExtractKernelsFromModule(llvm_module.get(), kernels));
        AddXlaBackendExtraOptionsAsModuleFlag(new_module.get(),
                                              backend_extra_options);
        add_jit_module(std::move(new_module));
      }
    }

    if (HasLargeConstants(*llvm_module)) {
      VLOG(3) << "Skip parallel compilation due to large constants";
      num_default_parts = 1;
    }

    if (num_default_parts > 1) {
      VLOG(3) << "Split LLVM module into " << num_default_parts
              << " parts before codegen to enable parallel compilation"
              << " (max split count: " << parallel_codegen_split_count << ")";

      TraceMe trace([&] {
        return TraceMeEncode("SplitModule",
                             {{"num_default_parts", num_default_parts}});
      });

      llvm::SplitModule(*llvm_module, num_default_parts, add_jit_module,
                        /*PreserveLocals=*/true, /*RoundRobin=*/true);
      // Free resources used by the original LLVM module.
      llvm_module.reset();
      llvm_context.reset();

    } else {
      VLOG(3) << "Compile LLVM module without splitting (max split count: "
              << parallel_codegen_split_count << ")";
      compiled_parts.push_back(
          CollectCompiledSymbolsPart(ir_emitter2, *llvm_module));
      TF_CHECK_OK(jit_compiler.AddModule(llvm::orc::ThreadSafeModule(
          std::move(llvm_module), std::move(llvm_context))));
    }

    // Collect compiled symbols from all LLVM module parts.
    std::vector<FunctionLibrary::Symbol> compiled_symbols;

    absl::flat_hash_map<FunctionLibrary::TypeId, SymbolProto::FunctionTypeId>
        symbol_type_id_to_function_type_id;

    VLOG(3) << "Adding " << thunk_emitter.kernels().size()
            << " kernels to the JIT compiler";
    // Make sure we use all the "default" modules for maximum parallelism.
    int num_default_so_far = dylib_index - num_extra_parts;
    int kernel_dylib_index =
        num_default_so_far < num_default_parts ? num_default_so_far : 0;
    for (auto& [name, module] : thunk_emitter.kernels()) {
      compiled_symbols.push_back(
          FunctionLibrary::Sym<FunctionLibrary::Kernel>(name));
      symbol_type_id_to_function_type_id.emplace(
          compiled_symbols.back().type_id, SymbolProto::KERNEL);
      TF_CHECK_OK(jit_compiler.AddModule(std::move(module),
                                         num_extra_parts + kernel_dylib_index));
      // Simply roundrobin the default kernel dylibs
      kernel_dylib_index = (kernel_dylib_index + 1) % num_default_parts;
    }

    for (const CompiledSymbolsPart& part : compiled_parts) {
      for (const IrEmitter2::KernelInfo& kernel : part.kernels) {
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Kernel>(kernel.name));
        symbol_type_id_to_function_type_id.emplace(
            compiled_symbols.back().type_id, SymbolProto::KERNEL);
      }
      for (const IrEmitter2::ComparatorInfo& comparator : part.comparators) {
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Comparator>(comparator.name));
        symbol_type_id_to_function_type_id.emplace(
            compiled_symbols.back().type_id, SymbolProto::COMPARATOR);
      }
    }

    VLOG(3) << "Collected " << compiled_symbols.size() << " compiled symbols";

    TraceMe trace_codegen([&] {
      return TraceMeEncode(
          "Codegen", {{"num_default_parts", num_default_parts},
                      {"num_extra_parts", num_extra_parts},
                      {"num_compiled_functions", num_compiled_functions}});
    });

    TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> function_library,
                        std::move(jit_compiler).Compile(compiled_symbols));

    // Create constant allocations from the buffer assignment.
    TF_ASSIGN_OR_RETURN(std::vector<ConstantAllocation> constants,
                        CreateConstantAllocations(*assignment));

    TF_ASSIGN_OR_RETURN(
        auto cpu_executable,
        CpuExecutable::Create(std::move(function_library),
                              std::move(assignment), std::move(module),
                              std::move(thunks), std::move(constants),
                              std::move(hlo_profile_printer_data),
                              std::move(hlo_profile_index_map)));

    // Save object files to be able to export them to AOT compilation
    // result.
    cpu_executable->set_obj_files(std::move(obj_files));

    // Save compiled symbols to be able to export them to AOT compilation
    // result.
    cpu_executable->set_compiled_symbols(std::move(compiled_symbols));

    // Save mapping between symbol type id and function type id to be able to
    // export them to AOT compilation result.
    cpu_executable->set_symbol_type_id_to_function_type_id(
        symbol_type_id_to_function_type_id);

    if (embed_ir_in_executable) {
      cpu_executable->set_ir_module_string(ir_module_string);
    }

    return with_hlo_proto(std::move(cpu_executable));
  }

  TF_RETURN_IF_ERROR(nested_ir_emitter.EmitAllConstantGlobals());

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

  std::string ir_module_string;
  if (embed_ir_in_executable) {
    ir_module_string = llvm_ir::DumpToString(llvm_module.get());
  }

  TF_RETURN_IF_ERROR(VerifyLlvmModule(*llvm_module));

  // Save entry function name before destroying LLVM module.
  std::string entry_function_name = entry_function->getName().str();

  // JIT compile the LLVM IR module to in-memory machine code.
  llvm::orc::ThreadSafeModule thread_safe_module(std::move(llvm_module),
                                                 std::move(llvm_context));
  TF_RETURN_IF_ERROR(jit_compiler.AddModule(std::move(thread_safe_module)));

  using ComputeFn = std::remove_pointer_t<CpuExecutable::ComputeFunctionType>;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionLibrary> function_library,
      std::move(jit_compiler)
          .Compile({FunctionLibrary::Sym<ComputeFn>(entry_function_name)}));

  TF_ASSIGN_OR_RETURN(
      auto cpu_executable,
      CpuExecutable::Create(std::move(function_library), std::move(assignment),
                            std::move(module), entry_function_name,
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
  TraceMe trace([&] {
    return TraceMeEncode("CpuCompiler::RunBackend", {{"name", module->name()}});
  });

  VLOG(1) << "Compiling: " << module->name();
  RecordCpuCompilerStacktrace();
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrFormat("Compiling [%s] for CPU using JIT", module->name()));
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);
  auto llvm_options = llvm_ir::ExtractXlaBackendExtraOptions(
      module->config().debug_options().xla_backend_extra_options());
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_options);

  std::unique_ptr<CpuExecutable> cpu_executable;
  TF_ASSIGN_OR_RETURN(cpu_executable, CompileCpuExecutable(std::move(module)));

  cpu_executable->set_debug_info(
      cpu_executable->buffer_assignment().StatsString(
          /*report_total_fragmentation=*/true));
  VLOG(1) << "Compilation finished";
  return std::unique_ptr<Executable>(std::move(cpu_executable));
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& aot_options) {
  TF_RET_CHECK(!module_group->empty());
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();

  auto llvm_options = llvm_ir::ExtractXlaBackendExtraOptions(
      modules[0]->config().debug_options().xla_backend_extra_options());
  VlogMaxIsa(modules[0]->config().debug_options().xla_cpu_max_isa());
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_options);

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
    // This only works because each of the method pointers above returns a
    // bool. Otherwise we'd have to do some template magic.
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
  llvm::CodeGenOptLevel opt_level =
      IrCompiler::GetCodeGenOptLevel(modules[0]->config());
  std::shared_ptr<llvm::TargetMachine> target_machine =
      absl::WrapUnique(target->createTargetMachine(
          triple.getTriple(), options.cpu_name(), options.features(),
          CompilerTargetOptions(modules[0]->config()), reloc_model,
          std::nullopt, opt_level));

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::MLIRContext mlir_context;
  llvm::LLVMContext llvm_context;

  std::vector<std::unique_ptr<AotCompilationResult>> results;
  for (auto& hlo_module : modules) {
    VLOG(1) << "Compiling ahead-of-time: " << hlo_module->name();
    if (hlo_module->has_schedule()) {
      continue;
    }

    TF_RETURN_IF_ERROR(RunHloPasses(hlo_module.get(), /*is_aot_compile=*/true,
                                    target_machine.get(),
                                    /*dummy*/ CompileOptions{}));

    if (hlo_module->config().debug_options().xla_cpu_use_thunk_runtime()) {
      TF_ASSIGN_OR_RETURN(
          results.emplace_back(),
          CompileAheadOfTimeThunks(std::move(hlo_module), target_machine,
                                   options, triple, pic_level, pie_level));
    } else {
      TF_ASSIGN_OR_RETURN(
          results.emplace_back(),
          CompileAheadOfTimeLegacy(std::move(hlo_module), target_machine,
                                   options, triple, pic_level, pie_level));
    }
  }

  VLOG(1) << "Compilation finished";
  return std::move(results);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
CpuCompiler::CompileAheadOfTimeLegacy(
    std::unique_ptr<HloModule> module,
    std::shared_ptr<llvm::TargetMachine> target_machine,
    const CpuAotCompilationOptions& aot_options, const llvm::Triple& triple,
    const llvm::PICLevel::Level& pic_level,
    const llvm::PIELevel::Level& pie_level) {
  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      ScheduleModule(module.get(), BufferSizeBytesFunction()));

  // Run buffer analysis on the HLO graph. This analysis figures out which
  // temporary buffers are required to run the computation.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(module.get(),
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

  TargetMachineFeatures target_machine_features(target_machine.get());
  std::vector<cpu_function_runtime::BufferInfo> buffer_infos =
      CreateBufferInfosFromBufferAssignment(*module, *assignment);
  HloComputation* computation = module->entry_computation();

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::MLIRContext mlir_context;
  auto llvm_context = std::make_unique<llvm::LLVMContext>();

  // Set required information before emitting IR
  auto llvm_module =
      std::make_unique<llvm::Module>(kXlaModuleIdentifier, *llvm_context);
  llvm_module->setDataLayout(target_machine->createDataLayout());
  llvm_module->setTargetTriple(triple.getTriple());
  if (pic_level != llvm::PICLevel::NotPIC) {
    llvm_module->setPICLevel(pic_level);
  }
  if (pie_level != llvm::PIELevel::Default) {
    llvm_module->setPIELevel(pie_level);
  }
  IrEmitter ir_emitter(&mlir_context, *module, *assignment, llvm_module.get(),
                       std::move(instruction_to_profile_idx),
                       std::move(computation_to_profile_idx),
                       ModuleComputationsTransitivelyContainCustomCall(*module),
                       &target_machine_features,
                       // TODO(b/66051036): Run full msan for AOT.
                       /*emit_code_for_msan=*/false);

  TF_RETURN_IF_ERROR(ir_emitter.EmitAllConstantGlobals());

  for (ComputationToEmit subcomputation :
       SubcomputationEmissionOrder(computation)) {
    if (subcomputation.computation->IsFusionComputation()) {
      continue;
    }
    TF_RETURN_IF_ERROR(
        ir_emitter
            .EmitComputation(
                subcomputation.computation, subcomputation.computation->name(),
                /*is_top_level_computation=*/false,
                schedule.sequence(subcomputation.computation).instructions(),
                subcomputation.allow_reassociation)
            .status());
  }
  const std::string& entry_point_name = aot_options.entry_point_name();
  TF_ASSIGN_OR_RETURN(
      llvm::Function * entry_function,
      ir_emitter.EmitComputation(computation, entry_point_name,
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

  auto post_codegen_hook = [&](const llvm::Module& llvm_module,
                               const llvm::object::ObjectFile& obj_file) {
    if (!DumpingEnabledForHloModule(*module)) {
      return;
    }
    DumpModuleToFile(llvm_module, obj_file, *module);
  };

  IrCompiler::Options ir_compiler_options = {
      /*optimization_level=*/target_machine->getOptLevel(),
      /*optimize_for_size=*/
      options::OptimizeForSizeRequested(module->config()),
      /*fast_math_flags=*/llvm_ir::GetCpuFastMathFlags(module->config()),
      /*disable_expensive_passes=*/
      module->config().debug_options().xla_llvm_disable_expensive_passes(),
      /*disable_slp_vectorizer=*/
      options::SlpVectorizerDisabled(module->config()),
      /*disable_loop_unrolling=*/
      options::DisableLoopUnrolling(module->config()),
      /*dfsan_enabled=*/aot_options.sanitize_dataflow(),
      /*dfsan_abilists_enabled=*/aot_options.sanitize_abilists_dataflow()};

  IrCompiler::CompilationHooks ir_compiler_hooks = {
      pre_optimization_ir_hook,
      post_optimization_ir_hook,
      post_codegen_hook,
  };

  IrCompiler ir_compiler([&] { return target_machine; },
                         std::move(ir_compiler_options),
                         std::move(ir_compiler_hooks));

  std::unique_ptr<llvm::MemoryBuffer> object_file =
      cantFail(ir_compiler(*llvm_module));
  ObjectFileData object_file_data(object_file->getBufferStart(),
                                  object_file->getBufferEnd());

  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment->GetUniqueTopLevelOutputSlice());

  return std::make_unique<CpuAotCompilationResultLegacy>(
      std::move(object_file_data), std::move(buffer_infos),
      result_slice.index(), std::move(module),
      std::move(hlo_profile_printer_data));
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
CpuCompiler::CompileAheadOfTimeThunks(
    std::unique_ptr<HloModule> module,
    std::shared_ptr<llvm::TargetMachine> target_machine,
    const CpuAotCompilationOptions& aot_options, const llvm::Triple& triple,
    const llvm::PICLevel::Level& pic_level,
    const llvm::PIELevel::Level& pie_level) {
  TraceMe trace([&] {
    return TraceMeEncode("CpuCompiler::CompileAheadOfTimeThunks",
                         {{"name", module->name()}});
  });
  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::MLIRContext mlir_context;
  auto llvm_context = std::make_unique<llvm::LLVMContext>();

  const DebugOptions& debug_options = module->config().debug_options();

  TF_ASSIGN_OR_RETURN(HloSchedule schedule, CreateHloSchedule(*module));
  TF_RETURN_IF_ERROR(module->set_schedule(schedule));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                      CreateBufferAssignment(*module));
  DumpHloModuleIfEnabled(*module, *assignment,
                         absl::StrCat("cpu_aot_", kAfterOptimizationsDumpName));

  // TODO profiling related, probably delete this
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
  // probably delete this end

  TargetMachineFeatures target_machine_features(target_machine.get());

  auto llvm_module =
      std::make_unique<llvm::Module>(kXlaModuleIdentifier, *llvm_context);

  llvm_module->setDataLayout(target_machine->createDataLayout());
  llvm_module->setTargetTriple(triple.getTriple());
  if (pic_level != llvm::PICLevel::NotPIC) {
    llvm_module->setPICLevel(pic_level);
  }
  if (pie_level != llvm::PIELevel::Default) {
    llvm_module->setPIELevel(pie_level);
  }

  // Emitting part
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
      // TODO(b/66051036): Run full msan for AOT.
      /*emit_code_for_msan=*/false);

  // The thunk runtime manages large constants, therefore we only emit
  // small ones.
  TF_RETURN_IF_ERROR(nested_ir_emitter.EmitSmallConstantGlobals());

  // IR emitter is responsible for building LLVM module with host kernels for
  // corresponding HLO instructions (fusions, elemental instructions, etc.).
  IrEmitter2 ir_emitter2(*module, llvm_module.get(), &nested_ir_emitter);

  // Thunk emitter is responsible for building a Thunk sequence that will
  // resolved kernels in the compiled LLVM module and execute them together
  // with Thunks implemented as library calls (e.g. oneDNN or Eigen).
  ThunkEmitter thunk_emitter(ir_emitter2, *assignment, target_machine_features,
                             module->config());
  TF_ASSIGN_OR_RETURN(ThunkSequence thunks,
                      thunk_emitter.EmitEntryComputation(*module));

  // Cache these flags here since we'll want to access them after the module's
  // ownership is std::moved.
  const bool embed_ir_in_executable =
      debug_options.xla_embed_ir_in_executable();

  std::string ir_module_string;
  if (embed_ir_in_executable) {
    std::string emitter2_ir = llvm_ir::DumpToString(llvm_module.get());

    auto thunk_kernel_fmt = [](std::string* out,
                               const ThunkEmitter::EmittedKernel& kernel) {
      absl::StrAppend(out,
                      llvm_ir::DumpToString(kernel.module.getModuleUnlocked()));
    };
    std::string thunks_ir =
        absl::StrJoin(thunk_emitter.kernels(), "\n", thunk_kernel_fmt);

    ir_module_string = absl::StrCat(emitter2_ir, "\n", thunks_ir);
  }

  TF_RETURN_IF_ERROR(VerifyLlvmModule(*llvm_module));
  for (const auto& [name, module] : thunk_emitter.kernels()) {
    TF_RETURN_IF_ERROR(VerifyLlvmModule(*module.getModuleUnlocked()));
  }

  // Compilation part
  ModuleHook pre_optimization_ir_hook;
  ModuleHook post_optimization_ir_hook;
  std::tie(pre_optimization_ir_hook, post_optimization_ir_hook) =
      GetIRModuleHooks(*module, user_pre_optimization_hook_,
                       user_post_optimization_hook_);

  std::vector<std::string> obj_files;
  auto post_codegen_hook = [&](const llvm::Module& llvm_module,
                               const llvm::object::ObjectFile& obj_file) {
    obj_files.push_back(obj_file.getData().str());
    if (!DumpingEnabledForHloModule(*module)) {
      return;
    }
    absl::string_view id = llvm_module.getModuleIdentifier();
    size_t pos = std::min(id.size(), 1 + kXlaModuleIdentifier.size());
    DumpToFileInDir(
        *module, /*file_prefix=*/"",
        /*file_suffix=*/absl::StrCat("obj-file.", id.substr(pos), ".o"),
        absl::string_view(obj_file.getData().data(),
                          obj_file.getData().size()));
  };

  IrCompiler::Options ir_compiler_options = {
      /*optimization_level=*/target_machine->getOptLevel(),
      /*optimize_for_size=*/
      options::OptimizeForSizeRequested(module->config()),
      /*fast_math_flags=*/llvm_ir::GetCpuFastMathFlags(module->config()),
      /*disable_expensive_passes=*/
      module->config().debug_options().xla_llvm_disable_expensive_passes(),
      /*disable_slp_vectorizer=*/
      options::SlpVectorizerDisabled(module->config()),
      /*disable_loop_unrolling=*/
      options::DisableLoopUnrolling(module->config()),
      /*dfsan_enabled=*/aot_options.sanitize_dataflow(),
      /*dfsan_abilists_enabled=*/aot_options.sanitize_abilists_dataflow()};

  IrCompiler::CompilationHooks ir_compiler_hooks = {
      pre_optimization_ir_hook,
      post_optimization_ir_hook,
      post_codegen_hook,
  };

  IrCompiler ir_compiler([&] { return target_machine; },
                         std::move(ir_compiler_options),
                         std::move(ir_compiler_hooks));

  // For simplicity no parallel compilation is used.
  std::vector<CompiledSymbolsPart> compiled_parts;
  compiled_parts.push_back(
      CollectCompiledSymbolsPart(ir_emitter2, *llvm_module));

  // Collect compiled symbols from all LLVM module parts.
  std::vector<FunctionLibrary::Symbol> compiled_symbols;

  absl::flat_hash_map<FunctionLibrary::TypeId, SymbolProto::FunctionTypeId>
      symbol_type_id_to_function_type_id;

  VLOG(3) << "Compiling " << thunk_emitter.kernels().size()
          << " thunk kernels.";

  // We have to clone the LLVM module into a local context to be able to link
  // it with the other modules. This enables us to have one object file for all
  // the kernels.
  auto copy_llvm_module_to_local_context =
      [&llvm_context](llvm::Module& module) {
        // There is no way to clone a module from one context to another, so we
        // need to serialize the module to bitcode and parse it back into the
        // new context.
        llvm::SmallString<0> bc;
        llvm::raw_svector_ostream bcos(bc);
        llvm::WriteBitcodeToFile(module, bcos);

        // Parse module back into its own LLVM context.
        auto clone_module = llvm::parseBitcodeFile(
            llvm::MemoryBufferRef(llvm::StringRef(bc.data(), bc.size()),
                                  absl::StrFormat("%s_cloned_to_local_context",
                                                  kXlaModuleIdentifier)),
            *llvm_context);

        return clone_module;
      };

  llvm::Linker linker(*llvm_module);

  for (auto& [name, module] : thunk_emitter.kernels()) {
    compiled_symbols.push_back(
        FunctionLibrary::Sym<FunctionLibrary::Kernel>(name));
    symbol_type_id_to_function_type_id.emplace(compiled_symbols.back().type_id,
                                               SymbolProto::KERNEL);
    auto cloned_module =
        copy_llvm_module_to_local_context(*module.getModuleUnlocked());
    if (!cloned_module) {
      return Internal("Failed to clone LLVM module.");
    }
    // Match data layouts to avoid warning messages.
    cloned_module->get()->setDataLayout(llvm_module->getDataLayout());
    linker.linkInModule(std::move(cloned_module.get()));
  }

  cantFail(ir_compiler(*llvm_module));

  for (const CompiledSymbolsPart& part : compiled_parts) {
    for (const IrEmitter2::KernelInfo& kernel : part.kernels) {
      compiled_symbols.push_back(
          FunctionLibrary::Sym<FunctionLibrary::Kernel>(kernel.name));
      symbol_type_id_to_function_type_id.emplace(
          compiled_symbols.back().type_id, SymbolProto::KERNEL);
    }
    for (const IrEmitter2::ComparatorInfo& comparator : part.comparators) {
      compiled_symbols.push_back(
          FunctionLibrary::Sym<FunctionLibrary::Comparator>(comparator.name));
      symbol_type_id_to_function_type_id.emplace(
          compiled_symbols.back().type_id, SymbolProto::COMPARATOR);
    }
  }

  VLOG(3) << "Collected " << compiled_symbols.size() << " compiled symbols";

  // Create constant allocations from the buffer assignment.
  TF_ASSIGN_OR_RETURN(std::vector<ConstantAllocation> constants,
                      CreateConstantAllocations(*assignment));

  TF_ASSIGN_OR_RETURN(
      auto cpu_executable,
      CpuExecutable::Create(
          /*function_library=*/nullptr,  // NOTE: We don't need to generate a
                                         // function library as the only purpose
                                         // of this executable is to get
                                         // exported.
          std::move(assignment), std::move(module), std::move(thunks),
          std::move(constants), std::move(hlo_profile_printer_data),
          std::move(hlo_profile_index_map)));

  // Save compiled symbols to be able to export them to AOT compilation
  // result.
  cpu_executable->set_compiled_symbols(std::move(compiled_symbols));

  // Save mapping between symbol type id and function type id to be able to
  // export them to AOT compilation result.
  cpu_executable->set_symbol_type_id_to_function_type_id(
      symbol_type_id_to_function_type_id);

  if (embed_ir_in_executable) {
    cpu_executable->set_ir_module_string(ir_module_string);
  }

  // Dump computation proto state and buffer assignment for
  // GetCompiledMemoryStats results.
  auto with_hlo_proto = [&](std::unique_ptr<CpuExecutable> cpu_executable) {
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_hlo_module() = cpu_executable->module().ToProto();
    *hlo_proto->mutable_buffer_assignment() =
        cpu_executable->buffer_assignment().ToProto();
    StripPayloadFromLiteralProto(*hlo_proto);
    cpu_executable->set_hlo_proto(std::move(hlo_proto));
    return cpu_executable;
  };

  cpu_executable = with_hlo_proto(std::move(cpu_executable));

  const ThunkSequence& thunk_sequence =
      cpu_executable->thunks().thunk_sequence();

  std::unique_ptr<HloProfilePrinterData> executable_hlo_profile_printer_data =
      cpu_executable->module().config().hlo_profiling_enabled()
          ? std::make_unique<HloProfilePrinterData>(
                cpu_executable->hlo_profile_printer_data())
          : nullptr;

  return CpuAotCompilationResultThunks::Create(
      &cpu_executable->module(), &cpu_executable->buffer_assignment(),
      cpu_executable->module_name(), std::move(obj_files),
      cpu_executable->get_compiled_symbols_proto(), thunk_sequence,
      std::move(*cpu_executable).consume_function_library().release(),
      std::move(executable_hlo_profile_printer_data));
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
  static absl::StatusOr<std::unique_ptr<CpuExecutableAotCompilationResult>>
  Create(const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
         absl::string_view function_name, std::vector<std::string> obj_files,
         std::vector<SymbolProto> symbols, const ThunkSequence* thunks,
         CompilationResultProto::ObjFileKind obj_file_kind) {
    std::optional<ThunkSequenceProto> thunk_proto;

    if (thunks != nullptr) {
      ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
          &buffer_assignment->Allocations());
      TF_ASSIGN_OR_RETURN(thunk_proto, thunk_sequence_serdes.ToProto(*thunks));
    }

    return absl::WrapUnique(new CpuExecutableAotCompilationResult(
        hlo_module, buffer_assignment, function_name, std::move(obj_files),
        std::move(symbols), thunk_proto, obj_file_kind));
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
      Compiler* compiler,
      const se::StreamExecutor* stream_exec) const&& override;

  const HloModule* optimized_module() const override { return module_.get(); }

  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }

 private:
  CpuExecutableAotCompilationResult(
      const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
      absl::string_view function_name, std::vector<std::string> obj_files,
      std::vector<SymbolProto> symbols,
      const std::optional<ThunkSequenceProto>& thunks,
      CompilationResultProto::ObjFileKind obj_file_kind) {
    *proto_.mutable_hlo_module()->mutable_hlo_module() = hlo_module->ToProto();
    *proto_.mutable_hlo_module()->mutable_config() =
        hlo_module->config().ToProto();
    *proto_.mutable_buffer_assignment() = buffer_assignment->ToProto();
    proto_.set_entry_function_name(std::string(function_name));
    for (std::string& obj_file : obj_files) {
      proto_.add_obj_files(std::move(obj_file));
    }

    for (const auto& symbol : symbols) {
      auto* symbol_proto = proto_.add_compiled_symbols();
      *symbol_proto = symbol;
    }
    proto_.set_obj_files_kind(obj_file_kind);
    module_ = hlo_module->Clone();

    if (thunks.has_value()) {
      ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
          &buffer_assignment->Allocations());
      *proto_.mutable_thunk_sequence() = *thunks;
    }
  }

  explicit CpuExecutableAotCompilationResult(CompilationResultProto proto,
                                             std::unique_ptr<HloModule> module)
      : proto_(std::move(proto)), module_(std::move(module)) {}

  CompilationResultProto proto_;
  std::unique_ptr<HloModule> module_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Executable>>
CpuExecutableAotCompilationResult::LoadExecutable(
    Compiler* compiler, const se::StreamExecutor* stream_exec) const&& {
  // Recreate HloModule from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProtoWithConfig(proto_.hlo_module()));

  VLOG(2) << "Load XLA:CPU executable for module: " << module->name();

  // Recreate BufferAssignment from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssignment::FromProto(proto_.buffer_assignment(), module.get(),
                                  compiler->BufferSizeBytesFunction(),
                                  /*can_share_buffer=*/nullptr));

  const DebugOptions& debug_options = module->config().debug_options();
  VlogMaxIsa(debug_options.xla_cpu_max_isa());
  const HloModuleConfig& config = module->config();

  // Infer target machine from the current host CPU.
  IrCompiler::TargetMachineBuilder target_machine_builder =
      JitCompiler::InferTargetMachineBuilder(
          std::move(CompilerTargetOptions(module->config())),
          IrCompiler::GetCodeGenOptLevel(config),
          CpuFeatureFromString(debug_options.xla_cpu_max_isa()));
  TF_ASSIGN_OR_RETURN(auto target_machine, target_machine_builder());

  // Definition generator to link with XLA:CPU host runtime symbols.
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const llvm::DataLayout& data_layout) {
        return std::make_unique<RuntimeSymbolGenerator>(data_layout);
      };

  ObjectLoader object_loader(/*num_dylibs=*/1,
                             target_machine->createDataLayout(),
                             definition_generator);

  for (size_t i = 0; i < object_loader.num_dylibs(); ++i) {
    object_loader.dylib(i).value()->addGenerator(
        std::make_unique<RuntimeSymbolGenerator>(
            target_machine->createDataLayout()));
  }

  // We might have an XLA:CPU executable that has only runtime thunks and
  // doesn't have any corresponding object files, and it's absolutely fine.
  VLOG(2) << "Load XLA:CPU executable from " << proto_.obj_files_size()
          << " object files; entry_function_name="
          << proto_.entry_function_name();

  size_t obj_file_index = 0;
  for (auto& obj_file : proto_.obj_files()) {
    llvm::StringRef data(obj_file.data(), obj_file.size());
    TF_RETURN_IF_ERROR(
        object_loader.AddObjFile(llvm::MemoryBuffer::getMemBuffer(
            data, absl::StrCat(proto_.entry_function_name(), "_",
                               obj_file_index++))));
  }

  std::unique_ptr<CpuExecutable> cpu_executable;

  if (proto_.obj_files_kind() == CompilationResultProto::KERNELS) {
    ThunkSequenceSerDesProtobuf thunk_sequence_serdes(
        &buffer_assignment->Allocations());
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<ThunkSequence> thunks,
        thunk_sequence_serdes.FromProto(proto_.thunk_sequence()));

    VLOG(3) << "Loaded " << thunks->size() << " thunks.";

    std::vector<FunctionLibrary::Symbol> compiled_symbols;

    for (const auto& symbol_proto : proto_.compiled_symbols()) {
      switch (symbol_proto.function_type_id()) {
        case SymbolProto::KERNEL:
          compiled_symbols.push_back(
              FunctionLibrary::Sym<FunctionLibrary::Kernel>(
                  symbol_proto.name()));
          break;
        case SymbolProto::COMPARATOR:
          compiled_symbols.push_back(
              FunctionLibrary::Sym<FunctionLibrary::Comparator>(
                  symbol_proto.name()));
          break;
        default:
          return Internal(
              "Unknown function type id %s",
              SymbolProto_FunctionTypeId_Name(symbol_proto.function_type_id()));
      }
    }

    VLOG(3) << "Collected " << compiled_symbols.size() << " compiled symbols";
    TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> function_library,
                        std::move(object_loader).Load(compiled_symbols));

    // Create constant allocations from the buffer assignment.
    TF_ASSIGN_OR_RETURN(std::vector<ConstantAllocation> constants,
                        CreateConstantAllocations(*buffer_assignment));

    TF_ASSIGN_OR_RETURN(
        cpu_executable,
        CpuExecutable::Create(std::move(function_library),
                              std::move(buffer_assignment), std::move(module),
                              std::move(*thunks), std::move(constants), nullptr,
                              nullptr));

  } else if (proto_.obj_files_kind() == CompilationResultProto::CLASSIC) {
    // Create a "classic" CPU executable.
    using ComputeFn = std::remove_pointer_t<CpuExecutable::ComputeFunctionType>;
    TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> function_library,
                        std::move(object_loader)
                            .Load({FunctionLibrary::Sym<ComputeFn>(
                                proto_.entry_function_name())}));

    TF_ASSIGN_OR_RETURN(
        cpu_executable,
        CpuExecutable::Create(std::move(function_library),
                              std::move(buffer_assignment), std::move(module),
                              proto_.entry_function_name(), nullptr, nullptr));

  } else {
    return Internal("Unknown obj file kind");
  }

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

  // Export object files for all dylibs.
  std::vector<std::string> obj_files;
  for (const auto& obj_file : cpu_executable->obj_files()) {
    obj_files.push_back(std::string(obj_file));
  }

  auto kind = cpu_executable->has_thunks() ? CompilationResultProto::KERNELS
                                           : CompilationResultProto::CLASSIC;
  const ThunkSequence* thunk_sequence =
      cpu_executable->has_thunks() ? &cpu_executable->thunks().thunk_sequence()
                                   : nullptr;

  std::vector<SymbolProto> compiled_symbols =
      cpu_executable->get_compiled_symbols_proto();

  return CpuExecutableAotCompilationResult::Create(
      &cpu_executable->module(), &cpu_executable->buffer_assignment(),
      cpu_executable->module_name(), std::move(obj_files),
      std::move(compiled_symbols), thunk_sequence, kind);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
CpuCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  return CpuExecutableAotCompilationResult::FromString(serialized_aot_result);
}

absl::StatusOr<HloSchedule> CpuCompiler::CreateHloSchedule(
    const HloModule& hlo_module) const {
  // Select a memory scheduler optimized for concurrency vs minimal memory.
  auto scheduler = hlo_module.config()
                           .debug_options()
                           .xla_cpu_enable_concurrency_optimized_scheduler()
                       ? BFSMemoryScheduler
                       : DFSMemoryScheduler;

  // Select an order for emitting the HLO instructions for each
  // computation. Using this sequence enables tighter buffer liveness analysis
  // and reduced memory usage (as compared to using `DependencyHloOrdering`).
  return ScheduleModule(&hlo_module, BufferSizeBytesFunction(),
                        ComputationSchedulerToModuleScheduler(scheduler));
}

absl::StatusOr<std::unique_ptr<BufferAssignment>>
CpuCompiler::CreateBufferAssignment(const HloModule& module) const {
  // Run buffer allocation on the HLO graph.
  return BufferAssigner::Run(
      &module, std::make_unique<SequentialHloOrdering>(module.schedule()),
      BufferSizeBytesFunction(), memory_alignment,
      /*allocate_buffers_for_constants=*/true);
}

}  // namespace cpu
}  // namespace xla
