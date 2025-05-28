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

#include "absl/base/call_once.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
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
#include "llvm/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/polynomial_approximations.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

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

    // TODO(b/419635451): Without AVX512 loop unrolling leads to LLVM generating
    // enormous IR that later times out during code generation (AVX2 doesn't
    // have masked SIMD instructions, and control flow ends up vectorizing to a
    // lot of scalar loads and stores, which takes forever to codegen in machine
    // instruction selection). As a workaround, disable loop unrolling when
    // AVX512 is not available. Revisit this decision once we migrate to new
    // fusion emitters that do not rely on LLVM that much.
    auto target_features = target_machine->getTargetFeatureString();
    if (target_features.contains("+avx2") &&
        !target_features.contains("+avx512")) {
      pto.LoopUnrolling = false;
    }

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

std::unique_ptr<IrCompiler> IrCompiler::Create(
    llvm::TargetOptions target_options, Options options,
    CompilationHooks hooks) {
  TargetMachineBuilder target_machine_builder =
      IrCompiler::InferTargetMachineBuilder(std::move(target_options),
                                            options.opt_level,
                                            options.max_cpu_feature);

  return std::make_unique<IrCompiler>(target_machine_builder,
                                      std::move(options), std::move(hooks));
}

IrCompiler::IrCompiler(TargetMachineBuilder target_machine_builder,
                       Options options, CompilationHooks hooks)
    : IRCompiler(llvm::orc::IRSymbolMapper::ManglingOptions()),
      target_machine_builder_(std::move(target_machine_builder)),
      options_(std::move(options)),
      hooks_(std::move(hooks)) {}

// Initialize LLVM the first time `InferTargetMachine` is called.
static void InitializeLLVMTarget() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

absl::once_flag initialize_llvm_flag;

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
IrCompiler::InferTargetMachine(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  // Detect machine attributes for the target CPU.
  auto result = DetectMachineAttributes(max_cpu_feature);
  llvm::SmallVector<std::string> attrs(result.features.begin(),
                                       result.features.end());

  // If `max_cpu_feature` is newer than the host CPU, we should keep the host
  // CPU name, e.g., we don't want to set the target CPU to Skylake when we are
  // on a Broadwell host.
  absl::string_view cpu = result.num_filtered_features
                              ? CpuTargetFromMaxFeature(*max_cpu_feature)
                              : absl::string_view(llvm::sys::getHostCPUName());

  absl::call_once(initialize_llvm_flag, InitializeLLVMTarget);
  std::unique_ptr<llvm::TargetMachine> target_machine(
      llvm::EngineBuilder()
          .setTargetOptions(target_options)
          .setOptLevel(opt_level)
          .selectTarget(
              /*TargetTriple=*/llvm::Triple(), /*MArch=*/"",
              /*MCPU=*/cpu,
              /*MAttrs=*/attrs));

  if (target_machine == nullptr) {
    return Internal("Failed to create target machine for CPU %s", cpu);
  }

  return std::move(target_machine);
}

IrCompiler::TargetMachineBuilder IrCompiler::InferTargetMachineBuilder(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  return [target_options, opt_level, max_cpu_feature] {
    return InferTargetMachine(target_options, opt_level, max_cpu_feature);
  };
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> IrCompiler::operator()(
    llvm::Module& module) {
  VLOG(2) << "IR before optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpToString(&module));

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
    absl::MutexLock lock(&mutex_);
    if (hooks_.pre_optimization) {
      hooks_.pre_optimization(module);
    }
  }

  if (llvm::Error ir_passes_error =
          RunIrPasses(module, target_machine->get())) {
    return ir_passes_error;
  }

  VLOG(2) << "IR after optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpToString(&module));

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(&mutex_);
    if (hooks_.post_optimization) {
      hooks_.post_optimization(module);
    }
  }

  std::unique_ptr<llvm::MemoryBuffer> mc_memory_buffer =
      EmitMachineCode(module, target_machine->get());

  {  // Synchronize access to user-defined hooks.
    absl::MutexLock lock(&mutex_);
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
          absl::StrFormat("Invalid LLVM IR after optimizations:\n%s",
                          error_stream.str()));
    }
  }

  RewriteToPolynomialApproximations(&module, options_.fast_math_flags);

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
  target_machine->addPassesToEmitMC(codegen_passes, mc_context, ostream);
  codegen_passes.run(module);

  return std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(mc_stream_buffer));
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

}  // namespace xla::cpu
