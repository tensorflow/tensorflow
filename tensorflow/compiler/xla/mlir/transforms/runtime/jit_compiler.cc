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

#include "tensorflow/compiler/xla/mlir/transforms/runtime/jit_compiler.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/runtime/passes.h"
#include "tensorflow/compiler/xla/mlir/utils/to_string.h"
#include "tensorflow/compiler/xla/runtime/symbolic_shape.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

using absl::InternalError;
using absl::StrCat;

static bool DebugJitCompiler() {
#if defined(DEBUG_XLA_RUNTIME_COMPILER)
  return true;
#endif
  return false;
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

static void InitializeCompiler() {
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

static void SetupPassDebugging(MLIRContext* context, PassManager& pm) {
  // Print IR after all passes.
  if (DebugJitCompiler()) {
    context->disableMultithreading();
    pm.enableIRPrinting([](Pass*, Operation*) { return false; },
                        [](Pass*, Operation*) { return true; },
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
}

static LogicalResult RunPipeline(
    ModuleOp module, const std::function<void(PassManager&)>& create_pipeline) {
  if (!create_pipeline) return success();

  PassManager pm(module.getContext());
  SetupPassDebugging(module.getContext(), pm);

  // Instrument the pass manager to capture timing information.
  DefaultTimingManager tm;
  TimingScope timing;
  if (EnablePassTiming()) {
    tm.setEnabled(true);
    timing = tm.getRootScope();
    pm.enableTiming(timing);
  }

  create_pipeline(pm);

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

// Creates a new MLIR Context and registers all the dialects that are expected
// in the compiled module.
static std::unique_ptr<MLIRContext> CreateMlirContext(
    const JitCompiler::Options& opts) {
  DialectRegistry registry;

  // Call user-provided callback to register all required dialects.
  if (opts.register_dialects) opts.register_dialects(registry);

  auto threading = MLIRContext::Threading::DISABLED;
  auto ctx = std::make_unique<MLIRContext>(registry, threading);
  ctx->loadAllAvailableDialects();
  return ctx;
}

//===----------------------------------------------------------------------===//
// JitCompiler implementation.
//===----------------------------------------------------------------------===//

JitCompiler::JitCompiler(JitCompiler::Options opts,
                         std::string_view mlir_module,
                         std::string_view entrypoint)
    : opts_(std::move(opts)),
      context_(CreateMlirContext(opts_)),
      diagnostic_os_(diagnostic_),
      handler_(source_mgr_, context_.get(), diagnostic_os_),
      specialized_(false) {
  source_mgr_.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlir_module, "xla.program"),
      llvm::SMLoc());

  module_ = parseSourceFile<ModuleOp>(source_mgr_, context_.get());
  if (module_) entrypoint_ = module_->lookupSymbol<func::FuncOp>(entrypoint);
}

/*static*/ absl::StatusOr<std::unique_ptr<JitCompiler>>
JitCompiler::Instantiate(JitCompiler::Options opts,
                         std::string_view mlir_module,
                         std::string_view entrypoint) {
  std::unique_ptr<JitCompiler> context(
      new JitCompiler(std::move(opts), mlir_module, entrypoint));
  if (!context->module_)
    return context->Error("failed to parse the mlir source");
  if (!context->entrypoint_)
    return context->Error("failed to resolve entrypoint function");

  InitializeCompiler();

  return {std::move(context)};
}

/*static*/ absl::StatusOr<Executable> JitCompiler::Compile(
    std::unique_ptr<JitCompiler> compiler, std::string_view memory_region_name,
    llvm::Optional<size_t> specialization) {
  const JitCompiler::Options& opts = compiler->options();
  func::FuncOp entry_func = compiler->entrypoint();
  std::string entrypoint = entry_func.getName().str();

  // We track end-to-end time to compile the final executable.
  auto compilation_start = std::chrono::steady_clock::now();

  // Get the signature of the entrypoint function.
  auto signature = opts.type_converter.Convert(entry_func.getFunctionType());
  if (!signature.ok()) return signature.status();

  // Get the calling convention for the entrypoint function.
  if (!opts.calling_convention)
    return compiler->Error("calling convention is not defined");

  // Calling convention conversion can fail if some types are not supported.
  auto runtime_type = opts.calling_convention(entry_func.getFunctionType());
  if (!runtime_type)
    return compiler->Error(
        "calling convention failed to convert entrypoint type");

  // Get the runtime signature of the entrypoint function.
  auto runtime_signature = opts.type_converter.Convert(runtime_type);
  if (!runtime_signature.ok()) return runtime_signature.status();

  // Get the memory layout for passing function arguments.
  auto arguments_memory_layout =
      Executable::GetArgumentsMemoryLayout(*runtime_signature);
  if (auto err = arguments_memory_layout.takeError())
    return InternalError(ToString(std::move(err)));

  // Get the memory layout for returning function results.
  auto results_memory_layout =
      Executable::GetResultsMemoryLayout(*runtime_signature);
  if (auto err = results_memory_layout.takeError())
    return InternalError(ToString(std::move(err)));

  // Mark entry function with an attribute, so it can be converted to an Xla
  // entrypoint (see `rt-convert-to-entrypoint` pass).
  auto unit_attr = UnitAttr::get(entry_func.getContext());
  entry_func->setAttr(kEntrypointAttrName, unit_attr);

  // Run the compilation pipeline to lower the module to LLVM dialect.
  if (failed(RunCompilationPipeline(compiler->module(), opts)))
    return compiler->Error("failed to run compilation pipeline");

  if (EnablePassTiming()) llvm::TimePassesIsEnabled = true;

  // Prepare JIT target machine for code generation.
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) return InternalError(ToString(builder.takeError()));

  auto target_machine = builder->createTargetMachine();
  if (!target_machine)
    return InternalError(ToString(target_machine.takeError()));

  // Name of the compiled module if available.
  auto module_name = compiler->module().getSymName().value_or("<unknown>");

  // Memory region name to mmap executable code.
  std::string mapper_name = llvm::formatv(
      "/xla{0}{1}:@{2}::@{3}:{4}", memory_region_name.empty() ? "" : ":",
      EscapeMemRegionName(memory_region_name), module_name, entrypoint,
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
  engine_options.make_optimizing_transformer = makeOptimizingTransformer;
  engine_options.section_memory_mapper = memory_mapper.get();
  engine_options.symbols_binding = std::move(symbols);

  // Translate MLIR module to the LLVM module.
  auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = translateModuleToLLVMIR(compiler->module(), *llvm_ctx);
  if (!llvm_module)
    return InternalError("failed to translate module to LLVM IR");

  // Compile input module to the native function.
  auto engine = ExecutionEngine::CreateFromModule(
      std::move(llvm_ctx), std::move(llvm_module), entrypoint, engine_options);
  if (auto err = engine.takeError())
    return InternalError(ToString(std::move(err)));

  // At this point compilation is completed, and all symbols in the LLVM module
  // materialized as addresses (entrypoint is an executable function pointer).
  auto time_to_compile = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - compilation_start);

  if (EnablePassTiming()) llvm::reportAndResetTimings();

  return Executable(
      compiler->name().str(), std::move(memory_mapper), std::move(*engine),
      std::move(*signature), std::move(*runtime_signature),
      std::move(*arguments_memory_layout), std::move(*results_memory_layout),
      specialization, time_to_compile);
}

absl::Status JitCompiler::Specialize(ArgumentsRef arguments,
                                     ArrayRef<SymbolicShape> symbolic_shapes,
                                     ArrayRef<ArgumentConstraint> constraints,
                                     const SpecializationListener* listener) {
  assert(!specialized_ && "can specialize executable only once");
  specialized_ = true;

  func::FuncOp func = entrypoint();

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
