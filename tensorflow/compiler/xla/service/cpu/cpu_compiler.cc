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
#include "external/llvm/include/llvm/ADT/StringRef.h"
#include "external/llvm/include/llvm/ADT/Triple.h"
#include "external/llvm/include/llvm/IR/Function.h"
#include "external/llvm/include/llvm/IR/LLVMContext.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "external/llvm/include/llvm/Object/ObjectFile.h"
#include "external/llvm/include/llvm/Support/CommandLine.h"
#include "external/llvm/include/llvm/Support/TargetRegistry.h"
#include "external/llvm/include/llvm/Support/TargetSelect.h"
#include "external/llvm/include/llvm/Target/TargetMachine.h"
#include "external/llvm/include/llvm/Target/TargetOptions.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/port/initialize.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"
#include "tensorflow/compiler/xla/service/cpu/conv_canonicalization.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_parallelization_preparation.h"
#include "tensorflow/compiler/xla/service/cpu/disassembler.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/layout_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/inliner.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"

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

  // LLVM command-line flags are global, so set them during initialization.
  legacy_flags::CpuCompilerFlags* flags = legacy_flags::GetCpuCompilerFlags();
  if (!flags->xla_cpu_llvm_cl_opts.empty()) {
    std::vector<string> opts =
        tensorflow::str_util::Split(flags->xla_cpu_llvm_cl_opts, ',');
    std::vector<const char*> fake_argv;
    fake_argv.push_back("");
    for (const string& opt : opts) {
      fake_argv.push_back(opt.c_str());
    }
    llvm::cl::ParseCommandLineOptions(fake_argv.size(), &fake_argv[0]);
  }
}

namespace {
// This visitor records which HLO instructions should have profiling information
// recorded.
class CollectProfileCandidates : public DfsHloVisitorWithDefault {
 public:
  static StatusOr<std::unordered_map<const HloInstruction*, size_t>>
  GetCandidatesForComputation(HloComputation* computation) {
    std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx;
    CollectProfileCandidates profile_candidates_for_computation(
        &hlo_to_profile_idx);
    TF_RETURN_IF_ERROR(computation->root_instruction()->Accept(
        &profile_candidates_for_computation));
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
  // Skip constants, there is nothing to profile.
  Status HandleConstant(HloInstruction* /*constant*/,
                        const Literal& /*literal*/) override {
    return Status::OK();
  }
  // Skip parameters, they are a simple load.
  Status HandleParameter(HloInstruction* /*parameter*/) override {
    return Status::OK();
  }
  // It is important to recurse for "while" or else we risk overly coarse
  // profiling information.
  Status HandleWhile(HloInstruction* xla_while, HloInstruction* /*init*/,
                     HloComputation* condition, HloComputation* body) override {
    TF_RETURN_IF_ERROR(DefaultAction(xla_while));

    CollectProfileCandidates candidates_for_condition(hlo_to_profile_idx_);
    TF_RETURN_IF_ERROR(
        condition->root_instruction()->Accept(&candidates_for_condition));

    CollectProfileCandidates candidates_for_body(hlo_to_profile_idx_);
    TF_RETURN_IF_ERROR(body->root_instruction()->Accept(&candidates_for_body));

    return Status::OK();
  }

  std::unordered_map<const HloInstruction*, size_t>* hlo_to_profile_idx_;
};
}  // namespace

Status CpuCompiler::RunHloPasses(HloModule* hlo_module,
                                 HloModuleConfig* module_config,
                                 HloDumper dump_hlo) {
  // Optimization pipeline.
  HloPassPipeline pipeline("CPU", dump_hlo);

  // TODO(b/35786417): Re-enable inliner pass after fixing the bug and deciding
  // where we will take this pass in future.
  // pipeline.AddPass<Inliner>();

  pipeline.AddPass<ConvCanonicalization>();
  {
    auto& pass = pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification",
                                                               dump_hlo);
    pass.AddPass<AlgebraicSimplifier>(
        /*is_layout_sensitive=*/false,
        [](const Shape&, const Shape&) { return false; },
        /*enable_dot_simplification=*/false);
    pass.AddPass<ReshapeMover>();
    pass.AddPass<HloConstantFolding>();
  }
  pipeline.AddPass<TransposeFolding>(PotentiallyImplementedAsEigenDot);
  pipeline.AddPass<HloSubcomputationUnification>();
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
  pipeline.AddPass<CpuInstructionFusion>();
  pipeline.AddPass<CpuLayoutAssignment>(
      module_config->mutable_entry_computation_layout());
  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
      /*is_layout_sensitive=*/true,
      [](const Shape&, const Shape&) { return true; },
      /*enable_dot_simplification=*/false);
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  // Outline ops in the entry computation into calls to subcomputations.
  legacy_flags::CpuCompilerFlags* flags = legacy_flags::GetCpuCompilerFlags();
  if (flags->xla_cpu_parallel) {
    pipeline.AddPass<ParallelizationPreparation>();
  }
  // Copy insertion should be performed immediately before IR emission to
  // avoid inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes
  // an instruction which materializes a value).
  pipeline.AddPass<CopyInsertion>();
  if (flags->xla_cpu_parallel) {
    // Re-run the outlining, in case any copies were inserted into the entry
    // computation.
    pipeline.AddPass<ParallelizationPreparation>();
  }
  pipeline.AddPass<HloDCE>();
  return pipeline.Run(hlo_module).status();
}

namespace {

// Align buffers to 16-byte boundaries.
constexpr int64 kMemoryAlignment = 16;

llvm::TargetOptions CompilerTargetOptions(
    const HloModuleConfig& execution_options) {
  llvm::TargetOptions target_options;
  llvm_ir::SetTargetOptions(execution_options, &target_options);
  return target_options;
}

llvm::CodeGenOpt::Level CodeGenOptLevel() {
  legacy_flags::CpuCompilerFlags* flags = legacy_flags::GetCpuCompilerFlags();
  switch (flags->xla_cpu_llvm_opt_level) {
    case 1:
      return llvm::CodeGenOpt::Less;
    case 2:
      return llvm::CodeGenOpt::Default;
      break;
    case 3:
      return llvm::CodeGenOpt::Aggressive;
      break;
    default:
      return llvm::CodeGenOpt::None;
  }
}

}  // namespace

StatusOr<std::unique_ptr<Executable>> CpuCompiler::Compile(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloModuleConfig> module_config, HloDumper dump_hlo,
    se::StreamExecutor* stream_exec) {
  TF_RET_CHECK(stream_exec != nullptr);

  // Compile must be thread-safe so create a new LLVM context for the module.
  auto llvm_context = MakeUnique<llvm::LLVMContext>();
  auto llvm_module =
      MakeUnique<llvm::Module>("__compute_module", *llvm_context);
  auto jit = MakeUnique<SimpleOrcJIT>(CompilerTargetOptions(*module_config),
                                      CodeGenOptLevel());
  llvm_module->setDataLayout(jit->data_layout());
  llvm_module->setTargetTriple(jit->target_triple().getTriple());

  TF_RETURN_IF_ERROR(
      RunHloPasses(hlo_module.get(), module_config.get(), dump_hlo));

  HloComputation* computation = hlo_module->entry_computation();
  std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx;
  if (module_config->hlo_profiling_enabled()) {
    TF_ASSIGN_OR_RETURN(
        hlo_to_profile_idx,
        CollectProfileCandidates::GetCandidatesForComputation(computation));
  }

  std::unique_ptr<Executable> cpu_executable;
  legacy_flags::CpuCompilerFlags* flags = legacy_flags::GetCpuCompilerFlags();
  if (flags->xla_cpu_parallel) {
    // Run buffer analysis on the HLO graph. This analysis figures out which
    // temporary buffers are required to run the computation.
    // DependencyHloOrdering is used for the parallel emitter because the order
    // of HLO instruction execution is not known ahead of time.
    // DependencyHloOrdering is the most conservative partial order and only
    // uses data dependencies for determining order.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> assignment,
        BufferAssigner::Run(hlo_module.get(),
                            MakeUnique<DependencyHloOrdering>(hlo_module.get()),
                            [this](const LogicalBuffer& buffer) {
                              return ShapeSizeBytes(buffer.shape());
                            },
                            kMemoryAlignment));

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
        const void* data = LiteralUtil::InternalData(instruction->literal());
        int64 size = ShapeSizeBytes(instruction->shape());
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
          << hlo_module->ToString();
      HloComputation* to_apply = instruction->to_apply();
      parallel_computations.emplace(to_apply, instruction);
    }

    IrEmitter ir_emitter(*hlo_module, *module_config, *assignment,
                         llvm_module.get(), &hlo_to_profile_idx);
    std::unique_ptr<std::map<HloInstruction*, string>> function_names(
        new std::map<HloInstruction*, string>());
    for (auto embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
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
              /*is_entry_computation=*/computation_is_parallel));
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
    if (flags->xla_cpu_embed_ir) {
      ir_module_string = llvm_ir::DumpModuleToString(*llvm_module);
    }

    // JIT compile the LLVM IR module to in-memory machine code.
    jit->AddModule(std::move(llvm_module));
    cpu_executable.reset(new ParallelCpuExecutable(
        std::move(jit), std::move(assignment), std::move(hlo_module),
        std::move(module_config), std::move(function_names),
        std::move(hlo_to_profile_idx), std::move(aligned_constants)));

    if (flags->xla_cpu_embed_ir) {
      static_cast<CpuExecutable&>(*cpu_executable)
          .set_ir_module_string(ir_module_string);
    }
  } else {
    // Select an order for emitting the HLO instructions for each
    // computation. Using this sequence enables tighter buffer liveness analysis
    // and reduced memory usage (as compared to using DependencyHloOrdering).
    TF_ASSIGN_OR_RETURN(
        SequentialHloOrdering::HloModuleSequence module_sequence,
        CreateMemoryMinimizingSequence(*hlo_module,
                                       [this](const LogicalBuffer& buffer) {
                                         return ShapeSizeBytes(buffer.shape());
                                       }));

    // Run buffer analysis on the HLO graph. This analysis figures out which
    // temporary buffers are required to run the computation.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> assignment,
        BufferAssigner::Run(hlo_module.get(),
                            MakeUnique<SequentialHloOrdering>(hlo_module.get(),
                                                              module_sequence),
                            [this](const LogicalBuffer& buffer) {
                              return ShapeSizeBytes(buffer.shape());
                            },
                            kMemoryAlignment));

    // Each computation is a single function.  Emit all embedded computations
    // before the entry computation. The order of computations returned from
    // GetEmbeddedComputations guarantees that a called computation occurs
    // before a caller computation.
    IrEmitter ir_emitter(*hlo_module, *module_config, *assignment,
                         llvm_module.get(), &hlo_to_profile_idx);
    for (auto embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
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
    if (flags->xla_cpu_embed_ir) {
      ir_module_string = llvm_ir::DumpModuleToString(*llvm_module);
    }

    // JIT compile the LLVM IR module to in-memory machine code.
    jit->AddModule(std::move(llvm_module));
    cpu_executable.reset(
        new CpuExecutable(std::move(jit), std::move(assignment),
                          std::move(hlo_module), std::move(module_config),
                          function_name, std::move(hlo_to_profile_idx)));

    if (flags->xla_cpu_embed_ir) {
      static_cast<CpuExecutable&>(*cpu_executable)
          .set_ir_module_string(ir_module_string);
    }
  }

  return std::move(cpu_executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> CpuCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    HloDumper dump_hlos, std::vector<se::StreamExecutor*> stream_execs) {
  return Unimplemented(
      "Compilation of multiple HLO modules is not yet supported on CPU.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CpuCompiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    HloDumper dump_hlo, const AotCompilationOptions& aot_options) {
  TF_RET_CHECK(hlo_modules.size() == module_configs.size());
  TF_RET_CHECK(!hlo_modules.empty());

  // We can pass just one llvm::TargetOptions when we compile the LLVM module,
  // so we bail if the configs have conflicting flags. At the moment, the only
  // flag that needs to be consistent is fast-math.
  bool fast_math_disabled = module_configs[0]->fast_math_disabled();
  for (const auto& module_config : module_configs) {
    if (module_config->fast_math_disabled() != fast_math_disabled) {
      return InvalidArgument(
          "All HLO module configs must have the same value for "
          "fast_math_disabled.");
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
  llvm::CodeGenOpt::Level opt_level = CodeGenOptLevel();
  std::unique_ptr<llvm::TargetMachine> target_machine =
      WrapUnique(target->createTargetMachine(
          triple.getTriple(), cpu_name, features,
          CompilerTargetOptions(*module_configs[0]), reloc_model,
          llvm::CodeModel::Default, opt_level));

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
  for (std::vector<std::unique_ptr<HloModule>>::size_type i = 0;
       i < hlo_modules.size(); ++i) {
    HloModule* hlo_module = hlo_modules[i].get();
    HloModuleConfig* module_config = module_configs[i].get();

    TF_RETURN_IF_ERROR(RunHloPasses(hlo_module, module_config, dump_hlo));

    TF_ASSIGN_OR_RETURN(
        SequentialHloOrdering::HloModuleSequence module_sequence,
        CreateMemoryMinimizingSequence(*hlo_module,
                                       [this](const LogicalBuffer& buffer) {
                                         return ShapeSizeBytes(buffer.shape());
                                       }));

    // Run buffer analysis on the HLO graph. This analysis figures out which
    // temporary buffers are required to run the computation.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> assignment,
        BufferAssigner::Run(
            hlo_module,
            MakeUnique<SequentialHloOrdering>(hlo_module, module_sequence),
            [this](const LogicalBuffer& buffer) {
              return ShapeSizeBytes(buffer.shape());
            },
            kMemoryAlignment));

    IrEmitter ir_emitter(*hlo_module, *module_config, *assignment, &llvm_module,
                         /*hlo_to_profile_idx=*/nullptr);
    HloComputation* computation = hlo_module->entry_computation();
    for (auto embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
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
                                   /*is_entry_computation=*/true));

    entry_function->setName(llvm_ir::AsStringRef(entry_point_name));

    Disassembler disassembler(*target_machine);
    CompilerFunctor compiler_functor(target_machine.get(), &disassembler,
                                     opt_level,
                                     CompilerFunctor::AllIntrinsics());
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
  return std::move(results);
}

se::Platform::Id CpuCompiler::PlatformId() const {
  return se::host::kHostPlatformId;
}

int64 CpuCompiler::ShapeSizeBytes(const Shape& shape) const {
  // On the cpu, opaques are pointers.
  if (ShapeUtil::IsOpaque(shape)) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace cpu
}  // namespace xla

REGISTER_MODULE_INITIALIZER(cpu_compiler, {
  xla::Compiler::RegisterCompilerFactory(se::host::kHostPlatformId, []() {
    return xla::MakeUnique<xla::cpu::CpuCompiler>();
  });
});
