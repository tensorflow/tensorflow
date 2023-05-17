/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/runtime/transforms/jit_compiler.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compiler.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/passes.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

using absl::InternalError;
using absl::StrCat;
using absl::StrFormat;

static bool DebugJitCompiler() {
#if defined(DEBUG_XLA_RUNTIME_COMPILER)
  return true;
#endif
  return VLOG_IS_ON(5);
}

static bool EnablePassTiming() {
#if defined(ENABLE_XLAR_RUNTIME_PASS_TIMING)
  return true;
#endif
  return DebugJitCompiler();
}

//===----------------------------------------------------------------------===//
// Setup MLIR pass pipeline to lower to LLVM dialect, and use ORC JIT to codegen
// functions at runtime.
//===----------------------------------------------------------------------===//

static void InitializeLlvmCompiler() {
  static const bool initialized = ([] {
    llvm::InitializeNativeTarget();
    // Initialize asm printer and parser so that we can handle the inline
    // assembly generated in MLIR for some operations.
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
  })();
  (void)initialized;
}

static void SetupPassDebugging(MLIRContext* context, mlir::PassManager& pm) {
  // Print IR after all passes.
  if (DebugJitCompiler()) {
    context->disableMultithreading();
    // Do not print large constants.
    mlir::OpPrintingFlags printing_flags;
    printing_flags.elideLargeElementsAttrs(32);
    pm.enableIRPrinting([](Pass*, Operation*) { return false; },
                        [](Pass*, Operation*) { return true; },
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/false, llvm::errs(),
                        printing_flags);
  }
}

static void PrintPassPipeline(const mlir::PassManager& pm) {
  llvm::errs() << "MLIR Pass Pipeline:\n";
  pm.printAsTextualPipeline(llvm::errs());
  llvm::errs() << "\n";
}

static LogicalResult RunPipeline(
    ModuleOp module, const std::function<void(PassManager&)>& create_pipeline) {
  if (!create_pipeline) return success();

  // Instrument the pass manager to capture timing information.
  DefaultTimingManager tm;
  TimingScope timing;

  mlir::PassManager pm(module.getContext());
  SetupPassDebugging(module.getContext(), pm);

  if (EnablePassTiming()) {
    tm.setEnabled(true);
    timing = tm.getRootScope();
    pm.enableTiming(timing);
  }
  PassManager passes(&pm);
  create_pipeline(passes);

  if (DebugJitCompiler()) {
    PrintPassPipeline(pm);
  }

  return pm.run(module);
}

// Runs the user-provided compilation pipeline to compile the module to LLVM.
static LogicalResult RunCompilationPipeline(ModuleOp module,
                                            const JitCompiler::Options& opts) {
  return RunPipeline(module, opts.create_compilation_pipeline);
}

// Runs the user-provided specialization pipeline.
static LogicalResult RunSpecializationPipeline(
    ModuleOp module, const JitCompiler::Options& opts) {
  return RunPipeline(module, opts.create_specialization_pipeline);
}

//===----------------------------------------------------------------------===//

// Configures MLIR Context and registers all the dialects that are expected
// in the compiled module.
static void ConfigureMlirContext(MLIRContext* context,
                                 const JitCompiler::Options& opts) {
  if (opts.register_dialects) {
    // Call user-provided callback to register all required dialects.
    DialectRegistry dialects;
    opts.register_dialects(dialects);
    context->appendDialectRegistry(*dialects);
    context->loadAllAvailableDialects();
  }
}

//===----------------------------------------------------------------------===//
// JitCompiler implementation.
//===----------------------------------------------------------------------===//

JitCompiler::JitCompiler(JitCompiler::Options opts,
                         std::string_view mlir_module)
    : opts_(std::move(opts)),
      owned_context_(
          std::make_unique<MLIRContext>(MLIRContext::Threading::DISABLED)),
      context_(owned_context_.get()),
      diagnostic_os_(diagnostic_),
      handler_(source_mgr_, context_, diagnostic_os_),
      specialized_(false) {
  ConfigureMlirContext(context_, opts_);
  source_mgr_.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlir_module, "xla.program"),
      llvm::SMLoc());
  module_ = parseSourceFile<ModuleOp>(source_mgr_, context_);
}

JitCompiler::JitCompiler(JitCompiler::Options opts, mlir::ModuleOp mlir_module)
    : opts_(std::move(opts)),
      context_(mlir_module.getContext()),
      diagnostic_os_(diagnostic_),
      handler_(source_mgr_, context_, diagnostic_os_),
      module_(mlir_module),
      specialized_(false) {
  ConfigureMlirContext(context_, opts_);
}

absl::Status JitCompiler::ComputeOrdinalsForExportedFunctions(
    absl::Span<const std::string_view> exported) {
  SymbolTable sym_table(*module_);

  // Add `rt.export` operations for all explicitly exported functions.
  for (const auto& indexed : llvm::enumerate(exported)) {
    if (auto func = sym_table.lookup<FunctionOpInterface>(indexed.value())) {
      OpBuilder(func).create<ExportOp>(func.getLoc(), func, indexed.index());
      continue;
    }
    return InvalidArgument("exported function %s not found", indexed.value());
  }

  // Assign unique ordinals to all exported functions, including functions that
  // were already exported with `rt.export` operations in the input IR.
  mlir::PassManager pm(module_->getContext());
  pm.addPass(CreateOrdinalAssignmentPass());
  if (failed(pm.run(*module_)))
    return Error("failed to run ordinal assignment pass");

  // Resolve all functions exported from the module indexed by ordinal.
  for (ExportOp op : module_->getOps<ExportOp>()) {
    unsigned ordinal = *op.ordinal();
    if (ordinal >= exported_.size()) exported_.resize(ordinal + 1);
    exported_[ordinal] = op.exported(sym_table);
  }

  return absl::OkStatus();
}

absl::Status ExportMainWithOrdinal0(mlir::ModuleOp module,
                                    mlir::MLIRContext& mlir_context) {
  SymbolTable sym_table(module);

  // Add `rt.export` operations for all explicitly exported functions.
  if (auto func = sym_table.lookup<FunctionOpInterface>("main")) {
    OpBuilder(func).create<ExportOp>(func.getLoc(), func, 0);
  }
  mlir::PassManager pm(&mlir_context);
  pm.addPass(CreateOrdinalAssignmentPass());
  if (failed(pm.run(module)))
    return absl::InternalError("failed to run ordinal assignment pass");
  return absl::OkStatus();
}

/*static*/ absl::StatusOr<std::unique_ptr<JitCompiler>>
JitCompiler::Instantiate(JitCompiler::Options opts,
                         std::string_view mlir_module,
                         absl::Span<const std::string_view> exported) {
  std::unique_ptr<JitCompiler> compiler(
      new JitCompiler(std::move(opts), mlir_module));

  // Check that mlir source was parsed into module operation.
  if (!compiler->module_)
    return compiler->Error("failed to parse the mlir source");

  auto status = compiler->ComputeOrdinalsForExportedFunctions(exported);
  if (!status.ok()) return status;

  // Initialize LLVM compiler internals.
  InitializeLlvmCompiler();

  return {std::move(compiler)};
}

/*static*/ absl::StatusOr<std::unique_ptr<JitCompiler>>
JitCompiler::Instantiate(JitCompiler::Options opts, ModuleOp mlir_module,
                         absl::Span<const std::string_view> exported) {
  std::unique_ptr<JitCompiler> compiler(
      new JitCompiler(std::move(opts), mlir_module));

  auto status = compiler->ComputeOrdinalsForExportedFunctions(exported);
  if (!status.ok()) return status;

  // Initialize LLVM compiler internals.
  InitializeLlvmCompiler();

  return {std::move(compiler)};
}

static std::function<llvm::Error(llvm::Module*)>
MakeOptimizingTransformerForJit(llvm::TargetMachine* targetMachine) {
  return [targetMachine](llvm::Module* m) -> llvm::Error {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PipelineTuningOptions tuningOptions;
    // LLVM's loop unrolling isn't well tuned for the loops we emit. Turn it off
    // as it consumes compile time with little benefit.
    tuningOptions.LoopUnrolling = false;
    // Vectorization happens at the MLIR level.
    tuningOptions.LoopVectorization = false;
    llvm::PassBuilder pb(targetMachine, tuningOptions);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm;
    mpm.addPass(pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2));
    mpm.run(*m, mam);
    return llvm::Error::success();
  };
}

/*static*/ absl::StatusOr<Executable> JitCompiler::Compile(
    std::unique_ptr<JitCompiler> compiler, std::string_view memory_region_name,
    std::optional<size_t> specialization) {
  // We track end-to-end time to compile the final executable.
  auto compilation_start = std::chrono::steady_clock::now();

  const JitCompiler::Options& opts = compiler->options();

  // Calling convention must be defined so we can get the run-time signature.
  if (!opts.calling_convention)
    return compiler->Error("calling convention is not defined");

  // Prepare exported functions that will be handed to the Executable.
  std::vector<Executable::Function> functions;
  std::vector<std::string_view> exported;  // names of exported functions

  for (const auto& indexed : llvm::enumerate(compiler->exported())) {
    auto func = indexed.value();
    std::string_view name = exported.emplace_back(func.getName());

    // Get the signature of the exported function.
    auto signature = opts.type_converter.Convert(
        llvm::cast<mlir::FunctionType>(func.getFunctionType()));
    if (!signature.ok()) return signature.status();

    // Calling convention conversion can fail if some types are not supported.
    auto runtime_type = opts.calling_convention(
        llvm::cast<mlir::FunctionType>(func.getFunctionType()));
    if (!runtime_type)
      return compiler->Error(StrFormat(
          "calling convention failed to convert function type for %s", name));

    // Get the runtime signature of the exported function.
    auto runtime_signature = opts.type_converter.Convert(runtime_type);
    if (!runtime_signature.ok()) return runtime_signature.status();

    // Get the memory layout for passing function arguments.
    auto arguments_memory_layout =
        Executable::GetArgumentsMemoryLayout(*runtime_signature);
    if (!arguments_memory_layout.ok()) return arguments_memory_layout.status();

    // Get the memory layout for returning function results.
    auto results_memory_layout =
        Executable::GetResultsMemoryLayout(*runtime_signature);
    if (!results_memory_layout.ok()) return results_memory_layout.status();

    bool requires_blas = false;
    if (Attribute requires_blas_attr = func->getAttr("xla.requires_blas")) {
      requires_blas = cast<BoolAttr>(requires_blas_attr).getValue();
    }

    // Add function with an unresolved function pointer; it will be updated once
    // we compile the input module to the native executable.
    functions.push_back(Executable::Function(
        name,
        /*fptr=*/nullptr, std::move(*signature), std::move(*runtime_signature),
        std::move(*arguments_memory_layout), std::move(*results_memory_layout),
        requires_blas));
  }

  // Run the compilation pipeline to lower the module to LLVM dialect.
  if (failed(RunCompilationPipeline(compiler->module(), opts)))
    return compiler->Error("failed to run compilation pipeline");

  if (EnablePassTiming()) llvm::TimePassesIsEnabled = true;

  // Prepare JIT target machine for code generation.
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) return InternalError(toString(builder.takeError()));

  auto target_machine = builder->createTargetMachine();
  if (!target_machine)
    return InternalError(toString(target_machine.takeError()));

  // Name of the compiled module if available.
  auto module_name = compiler->module().getSymName().value_or("<unknown>");

  // Memory region name to mmap executable code.
  std::string mapper_name = llvm::formatv(
      "/xla{0}{1}:@{2}::@{3}", memory_region_name.empty() ? "" : ":",
      EscapeMemRegionName(memory_region_name), module_name,
      specialization.has_value() ? "specialized" : "default");

  // Custom memory mapper to tag memory allocated for XLA executables.
  std::unique_ptr<XlaRuntimeMemoryMapper> memory_mapper =
      XlaRuntimeMemoryMapper::Create(std::move(mapper_name));

  // Register symbols required for running XLA Executable.
  ExecutionEngine::SymbolsBinding symbols =
      RuntimeSymbolsBinding(compiler->options().symbols_binding);

  // Construct options for the XLA runtime execution engine.
  ExecutionEngine::JitOptions engine_options;
  engine_options.opt_level = compiler->options().jit_code_opt_level;
  engine_options.target_machine = target_machine->get();
  engine_options.make_optimizing_transformer = MakeOptimizingTransformerForJit;
  engine_options.section_memory_mapper = memory_mapper.get();
  engine_options.symbols_binding = std::move(symbols);

  // Translate MLIR module to the LLVM module.
  auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = translateModuleToLLVMIR(compiler->module(), *llvm_ctx);
  if (!llvm_module)
    return compiler->Error("failed to translate module to LLVM IR");

  // Compile input module to the native function.
  auto engine = ExecutionEngine::CreateFromModule(
      std::move(llvm_ctx), std::move(llvm_module), engine_options, exported);
  if (!engine.ok()) return engine.status();

  // At this point compilation is completed, and all symbols in the LLVM module
  // materialized as addresses (all exported functions have a corresponding
  // function pointer).
  auto time_to_compile = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - compilation_start);

  if (EnablePassTiming()) llvm::reportAndResetTimings();

  // Resolve all exported functions to function pointers.
  for (unsigned i = 0; i < exported.size(); ++i)
    functions[i].fptr = (*engine)->exported(i);

  return Executable(compiler->name(), std::move(memory_mapper),
                    std::move(*engine), std::move(functions), specialization,
                    time_to_compile);
}

// TODO(ezhulenev): Currently it's possible to specialize only one function. It
// should be possible to specialize multiple functions, and run specialization
// pipeline once all specialized functions signatures are updated.

absl::Status JitCompiler::Specialize(unsigned ordinal, ArgumentsRef arguments,
                                     ArrayRef<SymbolicShape> symbolic_shapes,
                                     ArrayRef<ArgumentConstraint> constraints,
                                     const SpecializationListener* listener) {
  assert(!specialized_ && "can specialize executable only once");
  specialized_ = true;

  auto func = exported(ordinal);

  // Update function signature and sink constant arguments into the body.
  if (auto specialized = SpecializeFunction(func, arguments, symbolic_shapes,
                                            constraints, listener);
      !specialized.ok()) {
    // No need to call this->Error() because we don't have diagnostic to report
    // in case of a failed specialization.
    return InternalError(
        StrCat("failed to specialize: ", specialized.message()));
  }

  // Run the user-provided specialization pipeline to take advantage of the
  // specialized operands and sunk constants.
  if (failed(RunSpecializationPipeline(*module_, opts_)))
    return Error("failed to run specialization pipeline");

  return absl::OkStatus();
}

}  // namespace runtime
}  // namespace xla
