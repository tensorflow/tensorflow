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
#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/Config/Disassemblers.def.inc"
// IWYU pragma: no_include "llvm/Config/Targets.def.inc"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_rewriter.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"
#include "tensorflow/compiler/xla/service/cpu/conv_canonicalization.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_parallelization_preparation.h"
#include "tensorflow/compiler/xla/service/cpu/disassembler.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/layout_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_scheduling.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/inliner.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace cpu {

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
    ObjectFileData object_file_data, BufferSizes buffer_sizes,
    int64 result_buffer_index)
    : object_file_data_(std::move(object_file_data)),
      buffer_sizes_(std::move(buffer_sizes)),
      result_buffer_index_(result_buffer_index) {}

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
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86Disassembler();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();
  LLVMInitializeARMAsmPrinter();
  LLVMInitializeARMDisassembler();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmPrinter();
  LLVMInitializeAArch64Disassembler();
  LLVMInitializePowerPCTarget();
  LLVMInitializePowerPCTargetInfo();
  LLVMInitializePowerPCTargetMC();
  LLVMInitializePowerPCAsmPrinter();
  LLVMInitializePowerPCDisassembler();
}

namespace {

// LLVM makes certain options configurable only through its command-line
// options; it provide the ParseCommandLineOptions function that lets us set
// flags at runtime. However, since these flags are global we want to avoid
// multiple invocations of the LLVM compilation pipeline with a different set of
// flags. Therefore, we only pass command-line flags to LLVM once, before the
// first module is compiled.
std::once_flag llvm_command_line_options_initialized;

void InitializeLLVMCommandLineOptions(const HloModuleConfig& config) {
  auto options = config.debug_options().xla_backend_extra_options();
  if (!options.empty()) {
    std::vector<string> fake_argv_storage;
    fake_argv_storage.push_back("");
    for (const auto& it : options) {
      // Skip options the XLA backend itself consumes.
      if (!tensorflow::StringPiece(it.first).starts_with("xla_")) {
        if (it.second.empty()) {
          fake_argv_storage.push_back(it.first);
        } else {
          fake_argv_storage.push_back(it.first + "=" + it.second);
        }
      }
    }

    VLOG(2) << "Passing argv to LLVM:";
    std::vector<const char*> fake_argv;
    for (const auto& s : fake_argv_storage) {
      fake_argv.push_back(s.c_str());
      VLOG(2) << s;
    }
    llvm::cl::ParseCommandLineOptions(fake_argv.size(), &fake_argv[0]);
  }
}

// This visitor records which HLO instructions should have profiling information
// recorded.
class CollectProfileCandidates : public DfsHloVisitorWithDefault {
 public:
  static StatusOr<std::unordered_map<const HloInstruction*, size_t>>
  GetCandidatesForComputation(HloComputation* computation) {
    std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx;
    CollectProfileCandidates profile_candidates_for_computation(
        &hlo_to_profile_idx);
    TF_RETURN_IF_ERROR(
        computation->Accept(&profile_candidates_for_computation));
    return hlo_to_profile_idx;
  }

 private:
  explicit CollectProfileCandidates(
      std::unordered_map<const HloInstruction*, size_t>* hlo_to_profile_idx)
      : hlo_to_profile_idx_(hlo_to_profile_idx) {}

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    hlo_to_profile_idx_->insert({hlo_instruction, hlo_to_profile_idx_->size()});
    return Status::OK();
  }

  Status HandleCall(HloInstruction* call) override {
    TF_RETURN_IF_ERROR(DefaultAction(call));
    CollectProfileCandidates candidates_for_call(hlo_to_profile_idx_);
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

    CollectProfileCandidates candidates_for_condition(hlo_to_profile_idx_);
    TF_RETURN_IF_ERROR(
        xla_while->while_condition()->Accept(&candidates_for_condition));

    CollectProfileCandidates candidates_for_body(hlo_to_profile_idx_);
    TF_RETURN_IF_ERROR(xla_while->while_body()->Accept(&candidates_for_body));

    return Status::OK();
  }

  std::unordered_map<const HloInstruction*, size_t>* hlo_to_profile_idx_;
};
}  // namespace

Status CpuCompiler::RunHloPasses(HloModule* module, bool is_aot_compile) {
  // Optimization pipeline.
  HloPassPipeline pipeline("CPU");
  pipeline.AddInvariantChecker<HloVerifier>(ShapeSizeBytesFunction());

  ReducePrecisionInsertion::AddPasses(
      &pipeline, module->config().debug_options(),
      ReducePrecisionInsertion::PassTiming::BEFORE_OPTIMIZATION);

  // TODO(b/35786417): Re-enable inliner pass after fixing the bug and deciding
  // where we will take this pass in future.
  // pipeline.AddPass<Inliner>();

  // TODO(b/65775800): Fix wrong output bug in Call and remove the CallInliner
  // pass.
  pipeline.AddPass<CallInliner>();

  pipeline.AddPass<ConvCanonicalization>();
  {
    auto& pass =
        pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification");
    pass.AddInvariantChecker<HloVerifier>(ShapeSizeBytesFunction());

    pass.AddPass<BatchNormRewriter>(
        /*rewrite_training_op=*/true,
        /*rewrite_inference_op=*/true,
        /*rewrite_grad_op=*/true,
        /*use_fusion=*/false);
    pass.AddPass<AlgebraicSimplifier>(
        /*is_layout_sensitive=*/false,
        [](const Shape&, const Shape&) { return false; },
        /*enable_dot_simplification=*/false);
    pass.AddPass<TupleSimplifier>();
    pass.AddPass<WhileLoopSimplifier>();
    pass.AddPass<HloDCE>();
    pass.AddPass<ReshapeMover>();
    pass.AddPass<HloConstantFolding>();
  }
  pipeline.AddPass<TransposeFolding>(
      [](const HloInstruction& dot,
         const TransposeFolding::OperandIndices& candidate_operands) {
        return PotentiallyImplementedAsEigenDot(dot)
                   ? candidate_operands
                   : TransposeFolding::OperandIndices{};
      },
      TransposeFolding::NeverFoldTranspose);
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
  pipeline.AddPass<CpuInstructionFusion>();

  ReducePrecisionInsertion::AddPasses(
      &pipeline, module->config().debug_options(),
      ReducePrecisionInsertion::PassTiming::AFTER_FUSION);

  pipeline.AddPass<CpuLayoutAssignment>(
      module->mutable_entry_computation_layout());
  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
      /*is_layout_sensitive=*/true,
      [](const Shape&, const Shape&) { return true; },
      /*enable_dot_simplification=*/false);
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  // Outline ops in the entry computation into calls to subcomputations.
  const int max_parallelism =
      module->config().intra_op_parallelism_threads() > 0
          ? module->config().intra_op_parallelism_threads()
          : tensorflow::port::NumSchedulableCPUs();
  if (options::CpuParallelBackendRequested(module->config())) {
    pipeline.AddPass<ParallelizationPreparation>(max_parallelism,
                                                 ShapeSizeBytesFunction());
  } else if (!is_aot_compile) {
    // Run ParallelTaskAssigner to assign parallel tasks to HLOs in module.
    // Note this is not run for AOT because it would bring in thread pool
    // and thread synchronization dependencies which would likely increase
    // binary size (and most AOT applications are single-threaded).
    // TODO(29630486) Support multi-threaded AOT.
    pipeline.AddPass<ParallelTaskAssigner>(max_parallelism,
                                           ShapeSizeBytesFunction(), module);
  }
  // Copy insertion should be performed immediately before IR emission to avoid
  // inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes an
  // instruction which materializes a value). DCE must be run immediately before
  // (and sometime after) copy insertion, to avoid dead code from interfering
  // with the rewrites.
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<CopyInsertion>();
  if (options::CpuParallelBackendRequested(module->config())) {
    // Re-run the outlining, in case any copies were inserted into the entry
    // computation.
    pipeline.AddPass<ParallelizationPreparation>(max_parallelism,
                                                 ShapeSizeBytesFunction());
  }
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<FlattenCallGraph>();
  return pipeline.Run(module).status();
}

namespace {

// Align buffers to 16-byte boundaries.
constexpr int64 kMemoryAlignment = 16;
auto memory_alignment = [](LogicalBuffer::Color) { return kMemoryAlignment; };

llvm::TargetOptions CompilerTargetOptions(
    const HloModuleConfig& module_config) {
  llvm::TargetOptions target_options;
  llvm_ir::SetTargetOptions(
      /*fast_math_enabled=*/module_config.debug_options()
          .xla_enable_fast_math(),
      &target_options);
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

Status InitializeModuleHooks(
    const HloModule& hlo_module,
    const LLVMCompiler::ModuleHook& user_pre_optimization_hook,
    const LLVMCompiler::ModuleHook& user_post_optimization_hook,
    LLVMCompiler::ModuleHook* pre_optimization_ir_hook,
    LLVMCompiler::ModuleHook* post_optimization_ir_hook) {
  const string& ir_dump_directory =
      hlo_module.config().debug_options().xla_dump_ir_to();
  if (ir_dump_directory.empty()) {
    *pre_optimization_ir_hook = user_pre_optimization_hook;
    *post_optimization_ir_hook = user_post_optimization_hook;
    return Status::OK();
  }

  const string& hlo_module_name = hlo_module.name();

  // Create the IR hooks. If applicable, each IR hook does the following:
  //
  //  * Calls the user supplied module hook.
  //  * Writes out the IR to a file in the output directory designated by
  //    --xla_dump_ir_to

  *pre_optimization_ir_hook =
      [user_pre_optimization_hook, ir_dump_directory,
       hlo_module_name](const llvm::Module& llvm_module) {
        if (user_pre_optimization_hook) {
          TF_RETURN_IF_ERROR(user_pre_optimization_hook(llvm_module));
        }
        return llvm_ir::DumpIRToDirectory(/*directory_name=*/ir_dump_directory,
                                          /*hlo_module_name=*/hlo_module_name,
                                          llvm_module,
                                          /*optimized=*/false);
      };

  *post_optimization_ir_hook =
      [user_post_optimization_hook, ir_dump_directory,
       hlo_module_name](const llvm::Module& llvm_module) {
        if (user_post_optimization_hook) {
          TF_RETURN_IF_ERROR(user_post_optimization_hook(llvm_module));
        }
        return llvm_ir::DumpIRToDirectory(/*directory_name=*/ir_dump_directory,
                                          /*hlo_module_name=*/hlo_module_name,
                                          llvm_module,
                                          /*optimized=*/true);
      };

  return Status::OK();
}

}  // namespace

StatusOr<std::unique_ptr<Executable>> CpuCompiler::Compile(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec) {
  const string timer_message =
      "Compiling [" + module->name() + "] for CPU using JIT";
  ScopedLoggingTimer compiling_timer(timer_message, 1);

  VLOG(1) << "Compiling: " << module->name();
  TF_RET_CHECK(stream_exec != nullptr);
  std::call_once(llvm_command_line_options_initialized,
                 &InitializeLLVMCommandLineOptions, module->config());

  ModuleHook pre_optimization_ir_hook;
  ModuleHook post_optimization_ir_hook;
  TF_RETURN_IF_ERROR(InitializeModuleHooks(
      *module, user_pre_optimization_hook_, user_post_optimization_hook_,
      &pre_optimization_ir_hook, &post_optimization_ir_hook));

  // Compile must be thread-safe so create a new LLVM context for the module.
  auto llvm_context = MakeUnique<llvm::LLVMContext>();
  auto llvm_module =
      MakeUnique<llvm::Module>("__compute_module", *llvm_context);

  auto jit = MakeUnique<SimpleOrcJIT>(
      CompilerTargetOptions(module->config()),
      CodeGenOptLevel(module->config()),
      options::OptimizeForSizeRequested(module->config()),
      module->config().debug_options().xla_enable_fast_math(),
      module->config().debug_options().xla_llvm_disable_expensive_passes(),
      pre_optimization_ir_hook, post_optimization_ir_hook);
  llvm_module->setDataLayout(jit->data_layout());
  llvm_module->setTargetTriple(jit->target_triple().getTriple());

  VLOG(2) << "Before optimization:";
  XLA_VLOG_LINES(2, module->ToString());

  TF_RETURN_IF_ERROR(RunHloPasses(module.get(), /*is_aot_compile=*/false));

  VLOG(2) << "After optimization:";
  XLA_VLOG_LINES(2, module->ToString());

  HloComputation* computation = module->entry_computation();
  std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx;
  if (module->config().hlo_profiling_enabled()) {
    TF_ASSIGN_OR_RETURN(
        hlo_to_profile_idx,
        CollectProfileCandidates::GetCandidatesForComputation(computation));
  }

  std::unique_ptr<Executable> cpu_executable;

  // Cache these flags here since we'll want to access them after the module's
  // ownership is std::moved.
  const bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  const string xla_dump_hlo_proto_to =
      module->config().debug_options().xla_dump_hlo_proto_to();

  if (options::CpuParallelBackendRequested(module->config())) {
    VLOG(1) << "Using parallel cpu backend";

    // Run buffer analysis on the HLO graph. This analysis figures out which
    // temporary buffers are required to run the computation.
    // DependencyHloOrdering is used for the parallel emitter because the order
    // of HLO instruction execution is not known ahead of time.
    // DependencyHloOrdering is the most conservative partial order and only
    // uses data dependencies for determining order.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> assignment,
        BufferAssigner::Run(module.get(),
                            MakeUnique<DependencyHloOrdering>(module.get()),
                            BufferSizeBytesFunction(), memory_alignment));
    // BufferAssignment::ToString() includes a header, so no need for us to
    // print one ourselves.
    XLA_VLOG_LINES(2, assignment->ToString());

    if (!xla_dump_hlo_proto_to.empty()) {
      HloProto proto = MakeHloProto(*module, *assignment);
      TF_RETURN_IF_ERROR(protobuf_util::DumpProtoToDirectory(
          proto, xla_dump_hlo_proto_to, module->name()));
    }

    // If we are using the parallel CPU backend, we need to create map from
    // HloInstruction to the corresponding generated function name.
    std::map<HloComputation*, HloInstruction*> parallel_computations;
    std::unordered_map<const HloInstruction*, std::unique_ptr<unsigned char[]>>
        aligned_constants;
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // Parameters and constants don't get their own computation.
      if (instruction->opcode() == HloOpcode::kParameter) {
        continue;
      }
      if (instruction->opcode() == HloOpcode::kConstant) {
        // Copy the constant out of the ProtocolBuffer so that we can give it a
        // higher alignment.
        const void* data = instruction->literal().InternalData();
        int64 size = CpuExecutable::ShapeSizeBytes(instruction->shape());
        auto iter = aligned_constants.emplace(
            instruction, MakeUnique<unsigned char[]>(size));
        CHECK_EQ(iter.second, true);
        unsigned char* aligned_data = iter.first->second.get();
        memcpy(aligned_data, data, size);
        continue;
      }
      // The parallel preparation should have ensured that the top-level
      // computation consists solely of Call instructions.
      TF_RET_CHECK(instruction->opcode() == HloOpcode::kCall)
          << module->ToString();
      HloComputation* to_apply = instruction->to_apply();
      parallel_computations.emplace(to_apply, instruction);
    }

    IrEmitter ir_emitter(*module, *assignment, llvm_module.get(),
                         &hlo_to_profile_idx, jit->target_machine(),
                         jit->external_constant_pool());

    std::unique_ptr<HloInstructionMap<string>> function_names(
        new HloInstructionMap<string>());
    for (auto embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
      if (embedded_computation->IsFusionComputation()) {
        continue;
      }
      auto parallel_computation_iter =
          parallel_computations.find(embedded_computation);
      // All parallel computations are considered to be an entry computation for
      // IR generation purposes.
      bool computation_is_parallel =
          parallel_computation_iter != parallel_computations.end();
      TF_ASSIGN_OR_RETURN(
          llvm::Function * ir_function,
          ir_emitter.EmitComputation(
              embedded_computation, embedded_computation->name(),
              /*is_entry_computation=*/computation_is_parallel,
              /*instruction_order=*/nullptr));
      // If this computation is parallel, remember it in the function name map.
      // This way we know what function to execute when we try to run code for
      // the Call instruction.
      if (computation_is_parallel) {
        HloInstruction* call_instruction = parallel_computation_iter->second;
        InsertOrDie(function_names.get(), call_instruction,
                    llvm_ir::AsString(ir_function->getName()));
      }
    }

    string ir_module_string;
    if (embed_ir_in_executable) {
      ir_module_string = llvm_ir::DumpModuleToString(*llvm_module);
    }

    // JIT compile the LLVM IR module to in-memory machine code.
    jit->AddModule(std::move(llvm_module));
    cpu_executable.reset(new ParallelCpuExecutable(
        std::move(jit), std::move(assignment), std::move(module),
        std::move(function_names), std::move(hlo_to_profile_idx),
        std::move(aligned_constants)));

    if (embed_ir_in_executable) {
      static_cast<CpuExecutable&>(*cpu_executable)
          .set_ir_module_string(ir_module_string);
    }
  } else {
    VLOG(1) << "Using sequential cpu backend";

    // Select an order for emitting the HLO instructions for each
    // computation. Using this sequence enables tighter buffer liveness analysis
    // and reduced memory usage (as compared to using DependencyHloOrdering).
    TF_ASSIGN_OR_RETURN(
        SequentialHloOrdering::HloModuleSequence module_sequence,
        CreateMemoryMinimizingSequence(*module, BufferSizeBytesFunction()));

    // Run buffer analysis on the HLO graph. This analysis figures out which
    // temporary buffers are required to run the computation.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> assignment,
        BufferAssigner::Run(
            module.get(),
            MakeUnique<SequentialHloOrdering>(module.get(), module_sequence),
            BufferSizeBytesFunction(), memory_alignment));
    // BufferAssignment::ToString() includes a header, so no need for us to
    // print one ourselves.
    XLA_VLOG_LINES(2, assignment->ToString());

    if (!xla_dump_hlo_proto_to.empty()) {
      HloProto proto = MakeHloProto(*module, *assignment);
      TF_RETURN_IF_ERROR(protobuf_util::DumpProtoToDirectory(
          proto, xla_dump_hlo_proto_to, module->name()));
    }
    // Each computation is a single function.  Emit all embedded computations
    // before the entry computation. The order of computations returned from
    // GetEmbeddedComputations guarantees that a called computation occurs
    // before a caller computation.
    IrEmitter ir_emitter(*module, *assignment, llvm_module.get(),
                         &hlo_to_profile_idx, jit->target_machine(),
                         jit->external_constant_pool());

    for (auto embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
      if (embedded_computation->IsFusionComputation()) {
        continue;
      }
      TF_RETURN_IF_ERROR(
          ir_emitter
              .EmitComputation(embedded_computation,
                               embedded_computation->name(),
                               /*is_entry_computation=*/false,
                               &module_sequence.at(embedded_computation))
              .status());
    }
    string function_name_prefix =
        computation->name().empty() ? "__compute" : computation->name();
    TF_ASSIGN_OR_RETURN(
        llvm::Function * entry_function,
        ir_emitter.EmitComputation(computation, function_name_prefix,
                                   /*is_entry_computation=*/true,
                                   &module_sequence.at(computation)));

    string function_name = llvm_ir::AsString(entry_function->getName());
    string ir_module_string;
    if (embed_ir_in_executable) {
      ir_module_string = llvm_ir::DumpModuleToString(*llvm_module);
    }

    // JIT compile the LLVM IR module to in-memory machine code.
    jit->AddModule(std::move(llvm_module));
    cpu_executable.reset(new CpuExecutable(
        std::move(jit), std::move(assignment), std::move(module), function_name,
        std::move(hlo_to_profile_idx)));

    if (embed_ir_in_executable) {
      static_cast<CpuExecutable&>(*cpu_executable)
          .set_ir_module_string(ir_module_string);
    }
  }

  VLOG(1) << "Compilation finished";
  return std::move(cpu_executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> CpuCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> modules,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs) {
  return Unimplemented(
      "Compilation of multiple HLO modules is not yet supported on CPU.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CpuCompiler::CompileAheadOfTime(std::vector<std::unique_ptr<HloModule>> modules,
                                const AotCompilationOptions& aot_options) {
  TF_RET_CHECK(!modules.empty());
  std::call_once(llvm_command_line_options_initialized,
                 &InitializeLLVMCommandLineOptions, modules[0]->config());

  // We can pass just one llvm::TargetOptions when we compile the LLVM module,
  // so we bail if the configs have conflicting flags. At the moment, the only
  // flag that needs to be consistent is fast-math.
  const bool fast_math_enabled =
      modules[0]->config().debug_options().xla_enable_fast_math();
  for (const auto& module : modules) {
    if (module->config().debug_options().xla_enable_fast_math() !=
        fast_math_enabled) {
      return InvalidArgument(
          "All HLO module configs must have the same value for "
          "xla_enable_fast_math.");
    }
  }

  if (aot_options.PlatformId() != se::host::kHostPlatformId) {
    return InvalidArgument("Incompatible AOT compilation platform");
  }
  const CpuAotCompilationOptions& options =
      static_cast<const CpuAotCompilationOptions&>(aot_options);
  llvm::StringRef target_triple = llvm_ir::AsStringRef(options.triple());
  llvm::Triple triple(llvm::Triple::normalize(target_triple));
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), error);
  if (target == nullptr) {
    return InternalError("TargetRegistry::lookupTarget failed: %s",
                         error.c_str());
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
  llvm::StringRef cpu_name = llvm_ir::AsStringRef(options.cpu_name());
  llvm::StringRef features = llvm_ir::AsStringRef(options.features());
  llvm::CodeGenOpt::Level opt_level = CodeGenOptLevel(modules[0]->config());
  std::unique_ptr<llvm::TargetMachine> target_machine = WrapUnique(
      target->createTargetMachine(triple.getTriple(), cpu_name, features,
                                  CompilerTargetOptions(modules[0]->config()),
                                  reloc_model, llvm::None, opt_level));

  // Compile must be thread-safe so create a new LLVM context for the module.
  llvm::LLVMContext llvm_context;
  llvm::Module llvm_module("__compute_module", llvm_context);
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

    VLOG(2) << "Before optimization:";
    XLA_VLOG_LINES(2, module->ToString());

    TF_RETURN_IF_ERROR(RunHloPasses(module, /*is_aot_compile=*/true));

    VLOG(2) << "After optimization:";
    XLA_VLOG_LINES(2, module->ToString());

    TF_ASSIGN_OR_RETURN(
        SequentialHloOrdering::HloModuleSequence module_sequence,
        CreateMemoryMinimizingSequence(*module, BufferSizeBytesFunction()));

    // Run buffer analysis on the HLO graph. This analysis figures out which
    // temporary buffers are required to run the computation.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> assignment,
        BufferAssigner::Run(
            module, MakeUnique<SequentialHloOrdering>(module, module_sequence),
            BufferSizeBytesFunction(), memory_alignment));
    // BufferAssignment::ToString() includes a header, so no need for us to
    // print one ourselves.
    XLA_VLOG_LINES(2, assignment->ToString());

    const string xla_dump_hlo_proto_to =
        module->config().debug_options().xla_dump_hlo_proto_to();
    if (!xla_dump_hlo_proto_to.empty()) {
      HloProto proto = MakeHloProto(*module, *assignment);
      TF_RETURN_IF_ERROR(protobuf_util::DumpProtoToDirectory(
          proto, xla_dump_hlo_proto_to, module->name()));
    }

    IrEmitter ir_emitter(*module, *assignment, &llvm_module,
                         /*hlo_to_profile_idx=*/nullptr, target_machine.get(),
                         /*external_constant_pool=*/nullptr);
    HloComputation* computation = module->entry_computation();
    for (auto embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
      if (embedded_computation->IsFusionComputation()) {
        continue;
      }
      TF_RETURN_IF_ERROR(
          ir_emitter
              .EmitComputation(embedded_computation,
                               embedded_computation->name(),
                               /*is_entry_computation=*/false,
                               &module_sequence.at(embedded_computation))
              .status());
    }
    const string& entry_point_name = options.entry_point_name();
    TF_ASSIGN_OR_RETURN(
        llvm::Function * entry_function,
        ir_emitter.EmitComputation(computation, entry_point_name,
                                   /*is_entry_computation=*/true,
                                   &module_sequence.at(computation)));

    CHECK(entry_function->getName() == llvm_ir::AsStringRef(entry_point_name));

    ModuleHook pre_optimization_ir_dump_hook;
    ModuleHook post_optimization_ir_dump_hook;
    TF_RETURN_IF_ERROR(InitializeModuleHooks(
        *module, user_pre_optimization_hook_, user_post_optimization_hook_,
        &pre_optimization_ir_dump_hook, &post_optimization_ir_dump_hook));

    Disassembler disassembler(*target_machine);
    CompilerFunctor compiler_functor(
        target_machine.get(), &disassembler, opt_level,
        options::OptimizeForSizeRequested(module->config()),
        module->config().debug_options().xla_enable_fast_math(),
        module->config().debug_options().xla_llvm_disable_expensive_passes(),
        CompilerFunctor::AllIntrinsics(), pre_optimization_ir_dump_hook,
        post_optimization_ir_dump_hook);
    llvm::object::OwningBinary<llvm::object::ObjectFile> object_file =
        compiler_functor(llvm_module);
    llvm::StringRef object_file_data_ref = object_file.getBinary()->getData();
    ObjectFileData object_file_data(object_file_data_ref.begin(),
                                    object_file_data_ref.end());

    BufferSizes buffer_sizes;
    for (const BufferAllocation& allocation : assignment->Allocations()) {
      // Callers don't need to allocate temporary buffers for parameters.
      if (allocation.is_entry_computation_parameter()) {
        buffer_sizes.push_back(-1);
        continue;
      }
      // Callers don't need to allocate anything for thread-local temporary
      // buffers.  They are lowered to allocas.
      if (allocation.is_thread_local()) {
        buffer_sizes.push_back(-1);
        continue;
      }
      buffer_sizes.push_back(allocation.size());
    }

    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment->GetUniqueTopLevelOutputSlice());

    results.emplace_back(MakeUnique<CpuAotCompilationResult>(
        std::move(object_file_data), std::move(buffer_sizes),
        result_slice.index()));
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
  xla::Compiler::RegisterCompilerFactory(se::host::kHostPlatformId, []() {
    return xla::MakeUnique<xla::cpu::CpuCompiler>();
  });
  return true;
}
static bool module_initialized = InitModule();
