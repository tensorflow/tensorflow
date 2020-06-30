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

#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"

#include <stddef.h>
#include <string.h>

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/Config/Disassemblers.def.inc"
// IWYU pragma: no_include "llvm/Config/Targets.def.inc"
#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/conditional_to_select.h"
#include "tensorflow/compiler/xla/service/convolution_group_converter.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/cpu/buffer_info_util.h"
#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"
#include "tensorflow/compiler/xla/service/cpu/conv_canonicalization.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/dynamic_padder.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/logistic_expander.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/rng_bit_generator_expander.h"
#include "tensorflow/compiler/xla/service/rng_expander.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/slice_sinker.h"
#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tree_reduction_rewriter.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/dynamic_annotations.h"

namespace xla {
namespace cpu {
using BufferInfo = cpu_function_runtime::BufferInfo;

CpuAotCompilationOptions::CpuAotCompilationOptions(
    string triple, string cpu_name, string features, string entry_point_name,
    RelocationModel relocation_model)
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
    int64 result_buffer_index,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data)
    : object_file_data_(std::move(object_file_data)),
      buffer_infos_(std::move(buffer_infos)),
      result_buffer_index_(result_buffer_index),
      hlo_profile_printer_data_(std::move(hlo_profile_printer_data)) {}

CpuAotCompilationResult::~CpuAotCompilationResult() = default;

CpuCompiler::CpuCompiler() {
  // Initialize LLVM the first time the CpuCompiler is initialized.
  static bool llvm_initialized = []() {
    InitializeLLVMTarget();
    return true;
  }();
  (void)llvm_initialized;
}

/* static */ void CpuCompiler::InitializeLLVMTarget() {
  // Initialize LLVM's MC layer for the native target.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerAllDialects();
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
  static StatusOr<std::unordered_map<const HloInstruction*, int64>>
  GetCandidatesForComputation(
      const HloComputation& computation,
      const std::unordered_map<const HloInstruction*, int64>&
          assigned_indices) {
    std::unordered_map<const HloInstruction*, int64> hlo_to_profile_idx;
    CollectProfileCandidates profile_candidates_for_computation(
        &hlo_to_profile_idx, assigned_indices);
    TF_RETURN_IF_ERROR(computation.Accept(&profile_candidates_for_computation));
    return hlo_to_profile_idx;
  }

 private:
  CollectProfileCandidates(
      std::unordered_map<const HloInstruction*, int64>* hlo_to_profile_idx,
      const std::unordered_map<const HloInstruction*, int64>& assigned_indices)
      : hlo_to_profile_idx_(hlo_to_profile_idx),
        assigned_indices_(assigned_indices) {}

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    hlo_to_profile_idx_->insert(
        {hlo_instruction, FindOrDie(assigned_indices_, hlo_instruction)});
    return Status::OK();
  }

  Status HandleCall(HloInstruction* call) override {
    TF_RETURN_IF_ERROR(DefaultAction(call));
    CollectProfileCandidates candidates_for_call(hlo_to_profile_idx_,
                                                 assigned_indices_);
    TF_RETURN_IF_ERROR(call->to_apply()->Accept(&candidates_for_call));
    return Status::OK();
  }

  // Skip constants, there is nothing to profile.
  Status HandleConstant(HloInstruction*) override { return Status::OK(); }
  // Skip parameters, they are a simple load.
  Status HandleParameter(HloInstruction*) override { return Status::OK(); }
  // It is important to recurse for "while" or else we risk overly coarse
  // profiling information.
  Status HandleWhile(HloInstruction* xla_while) override {
    TF_RETURN_IF_ERROR(DefaultAction(xla_while));

    CollectProfileCandidates candidates_for_condition(hlo_to_profile_idx_,
                                                      assigned_indices_);
    TF_RETURN_IF_ERROR(
        xla_while->while_condition()->Accept(&candidates_for_condition));

    CollectProfileCandidates candidates_for_body(hlo_to_profile_idx_,
                                                 assigned_indices_);
    TF_RETURN_IF_ERROR(xla_while->while_body()->Accept(&candidates_for_body));

    return Status::OK();
  }

  std::unordered_map<const HloInstruction*, int64>* hlo_to_profile_idx_;
  const std::unordered_map<const HloInstruction*, int64>& assigned_indices_;
};

}  // namespace

Status CpuCompiler::RunHloPassesThroughLayoutAssn(
    HloModule* module, bool /*is_aot_compile*/,
    LLVMTargetMachineFeatures* target_machine_features) {
  HloPassPipeline pipeline("HLO passes through layout assignment");
  pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                            /*allow_mixed_precision=*/false);
  // Expand random number generation.
  pipeline.AddPass<RngExpander>();
  pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);

  // Remove zero-sized HLO from the input so that other passes don't have to
  // handle it.
  pipeline.AddPass<ZeroSizedHloElimination>();

  pipeline.AddPass<DynamicIndexSplitter>();

  pipeline.AddPass<ConditionalToSelect>();
  pipeline.AddPass<MapInliner>();

  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<TriangularSolveExpander>();

  // Inline computations with a single call site.
  pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  pipeline.AddPass<BatchDotSimplification>();
  pipeline.AddPass<DotDecomposer>();
  // After canonicalization, there may be more batch dots that can be
  // simplified.
  pipeline.AddPass<BatchDotSimplification>();
  auto cost_model = [](HloInstruction* conv) {
    // We need a cost model for CPUs. Currently, do nothing.
    return false;
  };
  pipeline.AddPass<ConvolutionGroupConverter>(
      cost_model,
      /*convert_batch_groups_only=*/true);
  pipeline.AddPass<ConvolutionGroupConverter>(
      cost_model,
      /*convert_batch_groups_only=*/false);
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);
  pipeline.AddPass<LogisticExpander>(
      /*expansion_type=*/LogisticExpansionType::kExp);
  pipeline.AddPass<DynamicPadder>();
  pipeline.AddPass<ScatterExpander>();
  pipeline.AddPass<HloGetDimensionSizeRewriter>();
  pipeline.AddPass<ConvCanonicalization>(target_machine_features);
  {
    auto& pass =
        pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification");
    pass.AddInvariantCheckerDebug<HloVerifier>(/*layout_sensitive=*/false,
                                               /*allow_mixed_precision=*/false);

    pass.AddPass<TreeReductionRewriter>();
    AlgebraicSimplifierOptions options;
    options.set_enable_dot_strength_reduction(false);
    pass.AddPass<AlgebraicSimplifier>(options);
    pass.AddPass<SortSimplifier>();
    pass.AddPass<HloDCE>();

    // BatchNormExpander can create zero-sized ops, so zero-sized HLO
    // elimination has to come after that pass.
    pass.AddPass<ZeroSizedHloElimination>();

    pass.AddPass<WhileLoopInvariantCodeMotion>();
    pass.AddPass<TupleSimplifier>();
    pass.AddPass<WhileLoopConstantSinking>();
    pass.AddPass<WhileLoopSimplifier>();

    // TODO(b/134075051): Re-enable after b/134075051 is fixed.
    // pass.AddPass<SliceSinker>();

    pass.AddPass<HloDCE>();
    pass.AddPass<ReshapeMover>();
    pass.AddPass<HloConstantFolding>();
    pass.AddPass<ConditionalSimplifier>();
  }
  pipeline.AddPass<IndexedArrayAnalysisPrinterPass>();
  pipeline.AddPass<TransposeFolding>(
      [&](const HloInstruction& dot,
          const TransposeFolding::OperandIndices& candidate_operands) {
        return DotImplementationCanHandleTranspose(dot,
                                                   *target_machine_features)
                   ? candidate_operands
                   : TransposeFolding::OperandIndices{};
      },
      TransposeFolding::NeverFoldTranspose);
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);

  // Layout assignment uses alias analysis, which requires the call graph to be
  // flattened.
  pipeline.AddPass<FlattenCallGraph>();
  pipeline.AddPass<CpuLayoutAssignment>(
      module->mutable_entry_computation_layout(),
      LayoutAssignment::InstructionCanChangeLayout, target_machine_features);

  pipeline.AddPass<CpuInstructionFusion>();

  return pipeline.Run(module).status();
}

Status CpuCompiler::RunHloPassesAfterLayoutAssn(
    HloModule* module, bool is_aot_compile,
    LLVMTargetMachineFeatures* target_machine_features) {
  HloPassPipeline pipeline("HLO passes after layout assignment");
  // After layout assignment, use a layout-sensitive verifier.

  pipeline.AddPass<HloPassPipeline>("after layout assignment")
      .AddInvariantCheckerDebug<HloVerifier>(
          /*layout_sensitive=*/true,
          /*allow_mixed_precision=*/false);

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  {
    auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
        "simplification after layout assignment");
    pass.AddInvariantCheckerDebug<HloVerifier>(
        /*layout_sensitive=*/true,
        /*allow_mixed_precision=*/false,
        LayoutAssignment::InstructionCanChangeLayout);
    AlgebraicSimplifierOptions options;
    options.set_is_layout_sensitive(true);
    options.set_enable_dot_strength_reduction(false);
    pass.AddPass<HloPassFix<AlgebraicSimplifier>>(options);
    pass.AddPass<HloDCE>();
    pass.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  }

  pipeline.AddPass<HloElementTypeConverter>(BF16, F32);

  // Outline ops in the entry computation into calls to subcomputations.
  const int max_parallelism =
      module->config().intra_op_parallelism_threads() > 0
          ? module->config().intra_op_parallelism_threads()
          : tensorflow::port::NumSchedulableCPUs();
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
  pipeline.AddPass<CopyInsertion>();
  pipeline.AddPass<HloDCE>();
  return pipeline.Run(module).status();
}

Status CpuCompiler::RunHloPasses(HloModule* module, bool is_aot_compile,
                                 llvm::TargetMachine* target_machine) {
  LLVMTargetMachineFeatures target_machine_features(target_machine);
  TF_RETURN_IF_ERROR(RunHloPassesThroughLayoutAssn(module, is_aot_compile,
                                                   &target_machine_features));
  return RunHloPassesAfterLayoutAssn(module, is_aot_compile,
                                     &target_machine_features);
}

namespace {

// Align buffers to 16-byte boundaries.
int64 memory_alignment(LogicalBuffer::Color) {
  return cpu_function_runtime::kMinAlign;
}

llvm::TargetOptions CompilerTargetOptions(
    const HloModuleConfig& module_config) {
  llvm::TargetOptions target_options;
  // Always allow FMA fusion. This increases precision instead of decreasing it.
  target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  return target_options;
}

llvm::CodeGenOpt::Level CodeGenOptLevel(const HloModuleConfig& module_config) {
  VLOG(2) << "backend_optimization_level: "
          << module_config.debug_options().xla_backend_optimization_level();
  switch (module_config.debug_options().xla_backend_optimization_level()) {
    case 1:
      return llvm::CodeGenOpt::Less;
    case 2:
      return llvm::CodeGenOpt::Default;
    case 3:
      return llvm::CodeGenOpt::Aggressive;
    default:
      return llvm::CodeGenOpt::None;
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

Status VerifyLlvmModule(const llvm::Module& llvm_module) {
  XLA_SCOPED_LOGGING_TIMER("CpuCompiler - Running LLVM verifier");

  std::string err;
  llvm::raw_string_ostream err_stream(err);

  // verifyModule() returns true if the module is broken.
  TF_RET_CHECK(!llvm::verifyModule(llvm_module, &err_stream))
      << "Invalid LLVM IR before optimizations:\n"
      << err_stream.str()
      << "\nThis probably indicates a bug in the HLO -> LLVM IR lowering. "
         "Rerun with --xla_dump_to to get the IR. ";
  return Status::OK();
}

Status CreateHloProfilingArtifacts(
    const HloModule& module,
    std::unordered_map<const HloInstruction*, int64>*
        instruction_to_profile_idx,
    std::unordered_map<const HloComputation*, int64>*
        computation_to_profile_idx,
    std::unique_ptr<HloProfileIndexMap>* hlo_profile_index_map,
    std::unique_ptr<HloProfilePrinterData>* hlo_profile_printer_data) {
  *hlo_profile_index_map = absl::make_unique<HloProfileIndexMap>(module);
  const HloComputation& entry_computation = *module.entry_computation();

  TF_ASSIGN_OR_RETURN(
      *instruction_to_profile_idx,
      CollectProfileCandidates::GetCandidatesForComputation(
          entry_computation,
          (*hlo_profile_index_map)->instruction_to_profile_idx()));

  auto shape_size_bytes = [](const Shape& shape) {
    // On the cpu, opaques are pointers.
    if (shape.IsOpaque()) {
      return static_cast<int64>(sizeof(void*));
    }
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  };

  HloCostAnalysis cost_analysis(shape_size_bytes);
  TF_RETURN_IF_ERROR(entry_computation.Accept(&cost_analysis));
  *hlo_profile_printer_data = CreateHloProfilePrinterData(
      **hlo_profile_index_map, cost_analysis, entry_computation.name());
  *computation_to_profile_idx =
      (*hlo_profile_index_map)->computation_to_profile_idx();

  return Status::OK();
}

}  // namespace

StatusOr<std::unique_ptr<HloModule>> CpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* /*stream_exec*/,
    se::DeviceMemoryAllocator* /*device_allocator*/) {
  std::unique_ptr<llvm::TargetMachine> jit_target_machine =
      SimpleOrcJIT::InferTargetMachineForJIT(
          CompilerTargetOptions(module->config()),
          CodeGenOptLevel(module->config()));

  TF_RETURN_IF_ERROR(RunHloPasses(module.get(), /*is_aot_compile=*/false,
                                  jit_target_machine.get()));
  return std::move(module);
}

StatusOr<
    std::tuple<std::unique_ptr<HloModule>, std::unique_ptr<BufferAssignment>>>
CpuCompiler::RunHloPassesAndBufferAssignement(
    std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
    se::DeviceMemoryAllocator* device_allocator) {
  TF_ASSIGN_OR_RETURN(
      module, RunHloPasses(std::move(module), executor, device_allocator));

  // Select an order for emitting the HLO instructions for each computation.
  // Using this sequence enables tighter buffer liveness analysis and reduced
  // memory usage (as compared to using DependencyHloOrdering).
  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      ScheduleModule(module.get(), BufferSizeBytesFunction(),
                                     ComputationSchedulerToModuleScheduler(
                                         DFSMemoryScheduler)));

  // Run buffer allocation on the HLO graph.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(module.get(),
                          absl::make_unique<SequentialHloOrdering>(schedule),
                          BufferSizeBytesFunction(), memory_alignment,
                          /*allocate_buffers_for_constants=*/true));

  return std::make_tuple(std::move(module), std::move(assignment));
}

namespace {

// Post-compilation callback functor for use by SimpleOrcJIT.
//
// Dumps machine code if dumping is enabled for the module.
struct OrcJITPostCompilationHook {
  // Gets an std::function that implements this hook.
  static std::function<void(const llvm::object::ObjectFile& obj_file)> Create(
      const HloModule* module) {
    // This struct is not copyable, but std::functions must be.  So to create an
    // std::function out of this struct, we have to wrap it in a shared_ptr.
    auto wrapped = std::make_shared<OrcJITPostCompilationHook>(module);
    return [wrapped](const llvm::object::ObjectFile& obj_file) {
      (*wrapped)(obj_file);
    };
  }

  // Constructor can't be private because we want to call it from
  // std::make_shared, but users should call Create() instead.
  explicit OrcJITPostCompilationHook(const HloModule* module)
      : module(module) {}

 private:
  void operator()(const llvm::object::ObjectFile& obj_file) {
    if (!DumpingEnabledForHloModule(*module)) {
      return;
    }
    DumpToFileInDir(*module, /*file_prefix=*/"", /*file_suffix=*/"o",
                    absl::string_view(obj_file.getData().data(),
                                      obj_file.getData().size()));
  }

  const HloModule* module;
};

}  // namespace

StatusOr<std::unique_ptr<Executable>> CpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* /*device_allocator*/) {
  VLOG(1) << "Compiling: " << module->name();
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrFormat("Compiling [%s] for CPU using JIT", module->name()));
  auto slow_compile_alarm = SlowCompilationAlarm();

  TF_RET_CHECK(stream_exec != nullptr);
  absl::call_once(llvm_command_line_options_initialized,
                  &llvm_ir::InitializeLLVMCommandLineOptions, module->config());

  ModuleHook pre_optimization_ir_hook;
  ModuleHook post_optimization_ir_hook;
  std::tie(pre_optimization_ir_hook, post_optimization_ir_hook) =
      GetIRModuleHooks(*module, user_pre_optimization_hook_,
                       user_post_optimization_hook_);

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::MLIRContext mlir_context;
  auto llvm_module = absl::make_unique<llvm::Module>(
      "__compute_module",
      mlir_context.getRegisteredDialect<mlir::LLVM::LLVMDialect>()
          ->getLLVMContext());

  auto jit = absl::make_unique<SimpleOrcJIT>(
      CompilerTargetOptions(module->config()),
      CodeGenOptLevel(module->config()),
      options::OptimizeForSizeRequested(module->config()),
      module->config().debug_options().xla_llvm_disable_expensive_passes(),
      llvm_ir::GetCpuFastMathFlags(module->config()), pre_optimization_ir_hook,
      post_optimization_ir_hook,
      OrcJITPostCompilationHook::Create(module.get()));
  llvm_module->setDataLayout(jit->data_layout());
  llvm_module->setTargetTriple(jit->target_triple().getTriple());

  HloComputation* entry_computation = module->entry_computation();
  std::unordered_map<const HloInstruction*, int64> instruction_to_profile_idx;
  std::unordered_map<const HloComputation*, int64> computation_to_profile_idx;
  std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map;
  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data;
  if (module->config().hlo_profiling_enabled()) {
    TF_RETURN_IF_ERROR(CreateHloProfilingArtifacts(
        *module, &instruction_to_profile_idx, &computation_to_profile_idx,
        &hlo_profile_index_map, &hlo_profile_printer_data));
  }

  std::unique_ptr<Executable> cpu_executable;

  // Cache these flags here since we'll want to access them after the module's
  // ownership is std::moved.
  const bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();

  // Select an order for emitting the HLO instructions for each
  // computation. Using this sequence enables tighter buffer liveness analysis
  // and reduced memory usage (as compared to using DependencyHloOrdering).
  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      ScheduleModule(module.get(), BufferSizeBytesFunction(),
                                     ComputationSchedulerToModuleScheduler(
                                         DFSMemoryScheduler)));

  // Run buffer allocation on the HLO graph.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(module.get(),
                          absl::make_unique<SequentialHloOrdering>(schedule),
                          BufferSizeBytesFunction(), memory_alignment,
                          /*allocate_buffers_for_constants=*/true));
  DumpHloModuleIfEnabled(*module, *assignment, "after_optimizations");

  // Each computation is a single function.  Emit all embedded computations
  // before the entry computation. The order of computations returned from
  // GetEmbeddedComputations guarantees that a called computation occurs
  // before a caller computation.

  LLVMTargetMachineFeatures target_machine_features(jit->target_machine());
  IrEmitter ir_emitter(&mlir_context, *module, *assignment, llvm_module.get(),
                       std::move(instruction_to_profile_idx),
                       std::move(computation_to_profile_idx),
                       &target_machine_features,
#ifdef MEMORY_SANITIZER
                       /*emit_code_for_msan=*/true
#else
                       /*emit_code_for_msan=*/false
#endif
  );

  TF_RETURN_IF_ERROR(ir_emitter.EmitConstantGlobals());

  for (auto embedded_computation :
       entry_computation->MakeEmbeddedComputationsList()) {
    if (embedded_computation->IsFusionComputation()) {
      continue;
    }
    TF_RETURN_IF_ERROR(
        ir_emitter
            .EmitComputation(
                embedded_computation, embedded_computation->name(),
                /*is_top_level_computation=*/false,
                schedule.sequence(embedded_computation).instructions())
            .status());
  }
  string function_name_prefix = entry_computation->name().empty()
                                    ? "__compute"
                                    : entry_computation->name();
  TF_ASSIGN_OR_RETURN(llvm::Function * entry_function,
                      ir_emitter.EmitComputation(
                          entry_computation, function_name_prefix,
                          /*is_top_level_computation=*/true,
                          schedule.sequence(entry_computation).instructions()));

  string function_name = [&]() {
    llvm::SmallVector<char, 40> function_name_vector;
    llvm::Mangler::getNameWithPrefix(
        function_name_vector, entry_function->getName(), jit->data_layout());
    return string(function_name_vector.begin(), function_name_vector.end());
  }();

  string ir_module_string;
  if (embed_ir_in_executable) {
    ir_module_string = llvm_ir::DumpModuleToString(*llvm_module);
  }

  TF_RETURN_IF_ERROR(VerifyLlvmModule(*llvm_module));

  // JIT compile the LLVM IR module to in-memory machine code.
  jit->AddModule(std::move(llvm_module));
  cpu_executable.reset(new CpuExecutable(
      std::move(jit), std::move(assignment), std::move(module), function_name,
      std::move(hlo_profile_printer_data), std::move(hlo_profile_index_map)));

  if (embed_ir_in_executable) {
    static_cast<CpuExecutable&>(*cpu_executable)
        .set_ir_module_string(ir_module_string);
  }

  VLOG(1) << "Compilation finished";
  return std::move(cpu_executable);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& aot_options) {
  TF_RET_CHECK(!module_group->empty());
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();

  absl::call_once(llvm_command_line_options_initialized,
                  &llvm_ir::InitializeLLVMCommandLineOptions,
                  modules[0]->config());

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
    for (int64 i = 0; i < modules.size(); ++i) {
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
    return InternalError("TargetRegistry::lookupTarget failed: %s", error);
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
  llvm::CodeGenOpt::Level opt_level = CodeGenOptLevel(modules[0]->config());
  std::unique_ptr<llvm::TargetMachine> target_machine =
      absl::WrapUnique(target->createTargetMachine(
          triple.getTriple(), options.cpu_name(), options.features(),
          CompilerTargetOptions(modules[0]->config()), reloc_model, llvm::None,
          opt_level));

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::MLIRContext mlir_context;
  llvm::Module llvm_module(
      "__compute_module",
      mlir_context.getRegisteredDialect<mlir::LLVM::LLVMDialect>()
          ->getLLVMContext());
  llvm_module.setDataLayout(target_machine->createDataLayout());
  llvm_module.setTargetTriple(triple.getTriple());
  if (pic_level != llvm::PICLevel::NotPIC) {
    llvm_module.setPICLevel(pic_level);
  }
  if (pie_level != llvm::PIELevel::Default) {
    llvm_module.setPIELevel(pie_level);
  }

  std::vector<std::unique_ptr<AotCompilationResult>> results;
  for (size_t i = 0; i < modules.size(); ++i) {
    HloModule* module = modules[i].get();
    VLOG(1) << "Compiling ahead-of-time: " << module->name();

    TF_RETURN_IF_ERROR(
        RunHloPasses(module, /*is_aot_compile=*/true, target_machine.get()));

    TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                        ScheduleModule(module, BufferSizeBytesFunction()));

    // Run buffer analysis on the HLO graph. This analysis figures out which
    // temporary buffers are required to run the computation.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> assignment,
        BufferAssigner::Run(module,
                            absl::make_unique<SequentialHloOrdering>(schedule),
                            BufferSizeBytesFunction(), memory_alignment,
                            /*allocate_buffers_for_constants=*/true));
    // BufferAssignment::ToString() includes a header, so no need for us to
    // print one ourselves.
    if (DumpingEnabledForHloModule(*module)) {
      DumpToFileInDirOrStdout(*module, "", "buffer_assignment",
                              assignment->ToString());
    }
    DumpHloModuleIfEnabled(*module, *assignment, "after_optimizations");

    std::unordered_map<const HloInstruction*, int64> instruction_to_profile_idx;
    std::unordered_map<const HloComputation*, int64> computation_to_profile_idx;
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map;
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data;

    if (module->config().hlo_profiling_enabled()) {
      TF_RETURN_IF_ERROR(CreateHloProfilingArtifacts(
          *module, &instruction_to_profile_idx, &computation_to_profile_idx,
          &hlo_profile_index_map, &hlo_profile_printer_data));
    }

    LLVMTargetMachineFeatures target_machine_features(target_machine.get());
    IrEmitter ir_emitter(&mlir_context, *module, *assignment, &llvm_module,
                         std::move(instruction_to_profile_idx),
                         std::move(computation_to_profile_idx),
                         &target_machine_features,
                         // TODO(b/66051036): Run full msan for AOT.
                         /*emit_code_for_msan=*/false);

    TF_RETURN_IF_ERROR(ir_emitter.EmitConstantGlobals());

    HloComputation* computation = module->entry_computation();
    for (auto embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
      if (embedded_computation->IsFusionComputation()) {
        continue;
      }
      TF_RETURN_IF_ERROR(
          ir_emitter
              .EmitComputation(
                  embedded_computation, embedded_computation->name(),
                  /*is_top_level_computation=*/false,
                  schedule.sequence(embedded_computation).instructions())
              .status());
    }
    const string& entry_point_name = options.entry_point_name();
    TF_ASSIGN_OR_RETURN(llvm::Function * entry_function,
                        ir_emitter.EmitComputation(
                            computation, entry_point_name,
                            /*is_top_level_computation=*/true,
                            schedule.sequence(computation).instructions()));

    CHECK(entry_function->getName() == entry_point_name);

    ModuleHook pre_optimization_ir_hook;
    ModuleHook post_optimization_ir_hook;
    std::tie(pre_optimization_ir_hook, post_optimization_ir_hook) =
        GetIRModuleHooks(*module, user_pre_optimization_hook_,
                         user_post_optimization_hook_);

    // Run the LLVM verifier over the unoptimized LLVM IR.  If it fails, run the
    // pre-optimization IR dump hook before returning.
    {
      Status verify_status = VerifyLlvmModule(llvm_module);
      if (!verify_status.ok() && pre_optimization_ir_hook) {
        pre_optimization_ir_hook(llvm_module);
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
        target_machine.get(), opt_level,
        options::OptimizeForSizeRequested(module->config()),
        module->config().debug_options().xla_llvm_disable_expensive_passes(),
        llvm_ir::GetCpuFastMathFlags(module->config()),
        pre_optimization_ir_hook, post_optimization_ir_hook, post_codegen_hook);
    std::unique_ptr<llvm::MemoryBuffer> object_file =
        compiler_functor(llvm_module);
    ObjectFileData object_file_data(object_file->getBufferStart(),
                                    object_file->getBufferEnd());

    std::vector<BufferInfo> buffer_infos =
        CreateBufferInfosFromBufferAssignment(*assignment);

    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment->GetUniqueTopLevelOutputSlice());

    results.emplace_back(absl::make_unique<CpuAotCompilationResult>(
        std::move(object_file_data), std::move(buffer_infos),
        result_slice.index(), std::move(hlo_profile_printer_data)));
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

}  // namespace cpu
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::host::kHostPlatformId,
      []() { return absl::make_unique<xla::cpu::CpuCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
