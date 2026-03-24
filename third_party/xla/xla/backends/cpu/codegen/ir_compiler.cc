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

#include "xla/backends/cpu/codegen/ir_compiler.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm-c/Target.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/polynomial_approximations.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/intrinsic_compiler_lib.h"
#include "xla/codegen/intrinsic_lib.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla::cpu {

namespace internal {

static absl::once_flag targets_init;
static void InitializeTargets() {
#if XLA_LLVM_AARCH32_AVAILABLE
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();
  LLVMInitializeARMAsmParser();
  LLVMInitializeARMAsmPrinter();
#endif
#if XLA_LLVM_AARCH64_AVAILABLE
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmParser();
  LLVMInitializeAArch64AsmPrinter();
#endif
#if XLA_LLVM_POWERPC_AVAILABLE
  LLVMInitializePowerPCTarget();
  LLVMInitializePowerPCTargetInfo();
  LLVMInitializePowerPCTargetMC();
  LLVMInitializePowerPCAsmParser();
  LLVMInitializePowerPCAsmPrinter();
#endif
#if XLA_LLVM_S390X_AVAILABLE
  LLVMInitializeSystemZTarget();
  LLVMInitializeSystemZTargetInfo();
  LLVMInitializeSystemZTargetMC();
  LLVMInitializeSystemZAsmParser();
  LLVMInitializeSystemZAsmPrinter();
#endif
#if XLA_LLVM_X86_AVAILABLE
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86AsmPrinter();
#endif
}

}  // namespace internal

void SetXlaCpuBackendOptions(llvm::Module& llvm_module,
                             const LlvmKernelOptions& options) {
  std::vector<std::string> llvm_kernel_options;
  if (options.optimize_for_size()) {
    llvm_kernel_options.emplace_back(options::kXlaOptimizeForSizeCpuOption);
  }
  if (options.disable_loop_unrolling()) {
    llvm_kernel_options.emplace_back(options::kDisableLoopUnrolling);
  }
  if (options.slp_vectorizer_disabled()) {
    llvm_kernel_options.emplace_back(options::kDisableSlpVectorizer);
  }
  if (options.disable_platform_dependent_math()) {
    llvm_kernel_options.emplace_back(options::kDisablePlatformDependentMath);
  }

  llvm::MDString* options_mdstring = llvm::MDString::get(
      llvm_module.getContext(), absl::StrJoin(llvm_kernel_options, ","));
  llvm_module.addModuleFlag(llvm::Module::Error, "xla_backend_extra_options",
                            options_mdstring);
}

static llvm::OptimizationLevel GetOptimizationLevel(
    IrCompiler::Options options) {
  if (options.optimize_for_size) {
    return llvm::OptimizationLevel::Os;
  }

  switch (options.opt_level) {
    case llvm::CodeGenOptLevel::None:
      return llvm::OptimizationLevel::O0;
    case llvm::CodeGenOptLevel::Less:
      return llvm::OptimizationLevel::O1;
    case llvm::CodeGenOptLevel::Default:
      return llvm::OptimizationLevel::O2;
    case llvm::CodeGenOptLevel::Aggressive:
      return llvm::OptimizationLevel::O3;
  }
}

static std::unique_ptr<HloModuleConfig> ParseXlaBackendExtraOptions(
    absl::string_view config_csv) {
  auto module_config = std::make_unique<HloModuleConfig>();
  DebugOptions& debug_options = module_config->mutable_debug_options();
  auto* map = debug_options.mutable_xla_backend_extra_options();
  std::vector<absl::string_view> vec =
      absl::StrSplit(config_csv, ',', absl::SkipEmpty());
  for (const auto& v : vec) {
    std::vector<absl::string_view> kv = absl::StrSplit(v, '=');
    (*map)[kv[0]] = kv.size() == 1 ? "" : kv[1];
  }
  return module_config;
}

// Returns an HloModuleConfig with its DebugOptions.xla_backend_extra_options
// set by the values embedded in the LLVM module. The rest of the fields
// of the proto should be ignored since they're just the default values.
// We could instead return an unordered_map<str, str>, but we already have
// helpers that expect a DebugOptions, so this ends up being simpler.
static absl_nullable std::unique_ptr<HloModuleConfig> GetXlaBackendExtraOptions(
    const llvm::Module& llvm_module) {
  llvm::Metadata* md = llvm_module.getModuleFlag("xla_backend_extra_options");
  if (md == nullptr) return nullptr;
  auto* md_string = llvm::dyn_cast<llvm::MDString>(md);
  if (md_string == nullptr) return nullptr;
  std::string config_csv = md_string->getString().str();
  return ParseXlaBackendExtraOptions(config_csv);
}

static llvm::PipelineTuningOptions GetPipelineTuningOptions(
    const llvm::Module& module, IrCompiler::Options options,
    const llvm::TargetMachine* target_machine) {
  auto pto_from_options = [&](const IrCompiler::Options opts) {
    llvm::PipelineTuningOptions pto;
    pto.LoopVectorization = !opts.optimize_for_size;
    pto.SLPVectorization =
        !opts.optimize_for_size && !opts.disable_slp_vectorizer;
    pto.LoopUnrolling = !opts.disable_loop_unrolling;

    // TODO(b/411125413): Re-enable SLPVectorization once the LLVM bug is fixed.
    pto.SLPVectorization = false;

    return pto;
  };

  std::unique_ptr<HloModuleConfig> config = GetXlaBackendExtraOptions(module);
  if (config == nullptr) {
    return pto_from_options(options);
  }

  // Apply overrides from the embedded config.
  IrCompiler::Options with_overrides(options);
  if (options::OptimizeForSizeRequested(*config)) {
    with_overrides.optimize_for_size = true;
  }
  if (options::SlpVectorizerDisabled(*config)) {
    with_overrides.disable_slp_vectorizer = true;
  }
  if (options::DisableLoopUnrolling(*config)) {
    with_overrides.disable_loop_unrolling = true;
  }
  return pto_from_options(with_overrides);
}

static bool FunctionHasInternalLinkage(const llvm::Function& function) {
  return function.hasInternalLinkage();
}

std::unique_ptr<IrCompiler> IrCompiler::Create(
    llvm::TargetOptions target_options, Options options,
    CompilationHooks hooks) {
  TargetMachineBuilder target_machine_builder =
      IrCompiler::InferTargetMachineBuilder(std::move(target_options),
                                            options.opt_level,
                                            options.target_machine_options);

  return std::make_unique<IrCompiler>(target_machine_builder,
                                      std::move(options), std::move(hooks));
}

IrCompiler::IrCompiler(TargetMachineBuilder target_machine_builder,
                       Options options, CompilationHooks hooks)
    : IRCompiler(llvm::orc::IRSymbolMapper::ManglingOptions()),
      target_machine_builder_(std::move(target_machine_builder)),
      options_(std::move(options)),
      hooks_(std::move(hooks)) {}

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
IrCompiler::InferTargetMachine(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    const TargetMachineOptions& target_machine_options) {
  auto attrs_vec = target_machine_options.GetTargetMachineFeaturesVector();
  llvm::SmallVector<std::string> attrs(attrs_vec.begin(), attrs_vec.end());

  absl::call_once(internal::targets_init, &internal::InitializeTargets);
  std::unique_ptr<llvm::TargetMachine> target_machine(
      llvm::EngineBuilder()
          .setTargetOptions(target_options)
          .setOptLevel(opt_level)
          .selectTarget(
              /*TargetTriple=*/llvm::Triple(target_machine_options.triple()),
              /*MArch=*/"",
              /*MCPU=*/target_machine_options.cpu(),
              /*MAttrs=*/attrs));

  if (target_machine == nullptr) {
    return Internal("Failed to create target machine for CPU %s",
                    target_machine_options.ToProto().DebugString());
  }

  return std::move(target_machine);
}

IrCompiler::TargetMachineBuilder IrCompiler::InferTargetMachineBuilder(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    const TargetMachineOptions& target_machine_options) {
  return [target_options, opt_level, target_machine_options] {
    return InferTargetMachine(target_options, opt_level,
                              target_machine_options);
  };
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> IrCompiler::operator()(
    llvm::Module& module) {
  absl::string_view module_name = module.getName();
  XLA_SCOPED_LOGGING_TIMER_LEVEL(
      absl::StrCat("Compiled LLVM module: ", module_name), 1);

  VLOG(3) << "IR before optimizations";
  XLA_VLOG_LINES(3, llvm_ir::DumpToString(&module));

  // Get a target machine for compilation. If compilations run concurrently on
  // multiple threads, `IrCompiler` user (in most cases `SimpleOrcJIT`)
  // must guarantee that target machine builder will return a unique
  // TargetMachine for each compilation, as it is not thread safe.
  absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> target_machine =
      build_target_machine();

  if (!target_machine.ok()) {
    return llvm::make_error<llvm::StringError>(
        llvm::errc::invalid_argument,
        absl::StrFormat(
            "Failed to create target machine for IR compilation: %s",
            target_machine.status().message()));
  }

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(mutex_);
    if (hooks_.pre_optimization) {
      hooks_.pre_optimization(module);
    }
  }

  if (llvm::Error ir_passes_error =
          RunIrPasses(module, target_machine->get())) {
    return ir_passes_error;
  }

  VLOG(3) << "IR after optimizations";
  XLA_VLOG_LINES(3, llvm_ir::DumpToString(&module));

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(mutex_);
    if (hooks_.post_optimization) {
      hooks_.post_optimization(module);
    }
  }

  std::unique_ptr<llvm::MemoryBuffer> mc_memory_buffer =
      EmitMachineCode(module, target_machine->get());

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(mutex_);
    if (hooks_.post_codegen) {
      llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_file =
          llvm::object::ObjectFile::createObjectFile(*mc_memory_buffer);
      if (obj_file) {
        hooks_.post_codegen(module, *obj_file.get());
      } else {
        LOG(WARNING) << "Could not convert memory buffer to object file";
      }
    }
  }

  return std::move(mc_memory_buffer);
}

llvm::Error IrCompiler::RunIrPasses(llvm::Module& module,
                                    llvm::TargetMachine* target_machine) const {
  if (absl::c_any_of(module.getFunctionList(), FunctionHasInternalLinkage)) {
    codegen::intrinsic::RunInlineAndOptPasses(module);
  }

  llvm::PipelineTuningOptions pto =
      GetPipelineTuningOptions(module, options_, target_machine);
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si(module.getContext(), false);
  si.registerCallbacks(pic, &mam);

  llvm::PassBuilder pb(target_machine, pto, {}, &pic);

  // Add the appropriate TargetLibraryInfo.
  llvm::Triple target_triple(target_machine->getTargetTriple());
  auto target_library_info_impl =
      std::make_unique<llvm::TargetLibraryInfoImpl>(target_triple);
  target_library_info_impl->addVectorizableFunctions(
      PolynomialApproximationsVectorization());

  xla::codegen::intrinsics::DeviceType device_type;
  if (target_triple.isX86()) {
    // As a heuristic, we check for SSE4a to determine if we are on AMD.
    // This feature was added in 2007 and is set on all AMD CPUs since then, and
    // no intel cpus. This is a bit of a hack though, as there is no strict link
    // between increased precision and SSE4a; Intel could decide to add it in
    // the future but they are very unlikely to do so as they haven't in the
    // past 18 years.
    if (target_machine->getTargetFeatureString().contains("+sse4a")) {
      device_type = xla::codegen::intrinsics::DeviceType::kAmdCpu;
    } else {
      device_type = xla::codegen::intrinsics::DeviceType::kIntelCpu;
    }
  } else if (target_triple.isAArch64() || target_triple.isARM()) {
    device_type = xla::codegen::intrinsics::DeviceType::kArmCpu;
  } else if (target_triple.isSystemZ()) {
    device_type = xla::codegen::intrinsics::DeviceType::kSystemZCpu;
  } else {
    LOG(FATAL) << "Unsupported CPU type: " << target_triple.str();
  }

  codegen::IntrinsicFunctionLib intrinsic_lib(
      {target_machine->getTargetFeatureString().str(), device_type,
       /*disable_platform_dependent_math=*/
       options_.disable_platform_dependent_math});
  target_library_info_impl->addVectorizableFunctions(
      intrinsic_lib.Vectorizations());

  fam.registerPass(
      [&] { return llvm::TargetLibraryAnalysis(*target_library_info_impl); });

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager pm;

  if (options_.dfsan_enabled) {
    pm.addPass(llvm::DataFlowSanitizerPass(options_.dfsan_abi_list_files));
  }

  llvm::OptimizationLevel opt_level = GetOptimizationLevel(options_);
  if (opt_level == llvm::OptimizationLevel::O0) {
    pm.addPass(pb.buildO0DefaultPipeline(opt_level));
  } else {
    pm.addPass(pb.buildPerModuleDefaultPipeline(opt_level));
  }

  {
    std::string error_string;
    llvm::raw_string_ostream error_stream(error_string);
    if (llvm::verifyModule(module, &error_stream)) {
      return llvm::make_error<llvm::StringError>(
          llvm::errc::invalid_argument,
          absl::StrFormat("Invalid LLVM IR before optimizations:\n%s",
                          error_stream.str()));
    }
  }

  pm.run(module, mam);

  {
    std::string error_string;
    llvm::raw_string_ostream error_stream(error_string);
    if (llvm::verifyModule(module, &error_stream)) {
      return llvm::make_error<llvm::StringError>(
          llvm::errc::invalid_argument,
          absl::StrFormat("Invalid LLVM IR after optimizations:\n%s\n",
                          error_stream.str()));
    }
  }

  auto replaced_functions = intrinsic_lib.DefineIntrinsicFunctions(module);
  RewriteToPolynomialApproximations(&module, options_.fast_math_flags);
  if (!replaced_functions.empty()) {
    codegen::intrinsic::RemoveFromCompilerUsed(
        module, [&](auto n) { return intrinsic_lib.IsIntrinsicFunction(n); });
    codegen::intrinsic::RunInlineAndOptPasses(module);
  }

  return llvm::Error::success();
}

std::unique_ptr<llvm::MemoryBuffer> IrCompiler::EmitMachineCode(
    llvm::Module& module, llvm::TargetMachine* target_machine) const {
  // Buffer for holding machine code prior to constructing the ObjectFile.
  llvm::SmallVector<char, 0> mc_stream_buffer;
  llvm::raw_svector_ostream ostream(mc_stream_buffer);

  // Generate code.
  llvm::MCContext* mc_context;
  llvm::legacy::PassManager codegen_passes;
  codegen_passes.add(new llvm::RuntimeLibraryInfoWrapper(
      module.getTargetTriple(), target_machine->Options.ExceptionModel,
      target_machine->Options.FloatABIType, target_machine->Options.EABIVersion,
      target_machine->Options.MCOptions.ABIName,
      target_machine->Options.VecLib));
  target_machine->addPassesToEmitMC(codegen_passes, mc_context, ostream);
  codegen_passes.run(module);

  llvm::NamedMDNode* memory_region_name_md =
      module.getNamedMetadata(std::string(kMemoryRegionNameMetadataName));
  CHECK(memory_region_name_md != nullptr)
      << "Memory region name metadata not found in LLVM module.";
  CHECK_GT(memory_region_name_md->getNumOperands(), 0);
  llvm::MDNode* node = memory_region_name_md->getOperand(0);
  CHECK(node != nullptr);
  CHECK_GT(node->getNumOperands(), 0);
  llvm::MDString* md_str = llvm::dyn_cast<llvm::MDString>(node->getOperand(0));
  CHECK(md_str != nullptr);
  llvm::StringRef mem_region_name_str = md_str->getString();

  return std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(mc_stream_buffer), mem_region_name_str);
}

llvm::CodeGenOptLevel IrCompiler::GetCodeGenOptLevel(
    const HloModuleConfig& module_config) {
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

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
IrCompiler::build_target_machine() const {
  return target_machine_builder_();
}

}  // namespace xla::cpu
