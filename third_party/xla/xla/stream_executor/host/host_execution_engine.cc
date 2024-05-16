/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/host/host_execution_engine.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace host {

using absl::InternalError;
using absl::StrFormat;

using llvm::Expected;
using llvm::MemoryBuffer;
using llvm::SectionMemoryManager;
using llvm::Triple;

using llvm::orc::ExecutionSession;
using llvm::orc::ExecutorAddr;
using llvm::orc::InPlaceTaskDispatcher;
using llvm::orc::IRCompileLayer;
using llvm::orc::JITTargetMachineBuilder;
using llvm::orc::RTDyldObjectLinkingLayer;
using llvm::orc::SelfExecutorProcessControl;
using llvm::orc::SimpleCompiler;
using llvm::orc::SymbolMap;
using llvm::orc::ThreadSafeModule;

namespace {

// This compiler keeps weak pointers to the TargetMachine and the ObjectCache.
//
// This allows releasing the memory of those objects, even though the LLJIT
// keeps the compiler alive.
//
// We wrote this class based on the code of llvm::orc::ConcurrentIRCompiler.
class WeakCompiler : public IRCompileLayer::IRCompiler {
 public:
  static llvm::orc::IRSymbolMapper::ManglingOptions
  IrManglingOptionsForWeakTargetMachine(
      std::weak_ptr<llvm::TargetMachine> weak_target_machine) {
    std::shared_ptr<llvm::TargetMachine> target_machine =
        weak_target_machine.lock();
    CHECK(target_machine != nullptr)
        << "Compiler should not be used after the TargetMachine is destroyed.";

    return llvm::orc::irManglingOptionsFromTargetOptions(
        target_machine->Options);
  }

  // It's not recommended to allocate the parameters with std::make_shared,
  // because that would allocate the object and the control block in one
  // allocation, so the weak_ptr would keep alive the memory of the object as
  // well.
  explicit WeakCompiler(std::weak_ptr<llvm::TargetMachine> weak_target_machine)
      : IRCompiler(IrManglingOptionsForWeakTargetMachine(weak_target_machine)),
        weak_target_machine_(std::move(weak_target_machine)) {}

  Expected<std::unique_ptr<MemoryBuffer>> operator()(
      llvm::Module &module) override {
    std::shared_ptr<llvm::TargetMachine> target_machine =
        weak_target_machine_.lock();
    CHECK(target_machine != nullptr)
        << "Compiler should not be used after the TargetMachine is destroyed.";

    SimpleCompiler compiler(*target_machine);
    return compiler(module);
  }

 private:
  std::weak_ptr<llvm::TargetMachine> weak_target_machine_;
};

}  // namespace

RuntimeExecutionEngine::RuntimeExecutionEngine(bool enable_gdb_listener,
                                               bool enable_perf_listener) {
  if (enable_gdb_listener)
    gdb_listener_ = llvm::JITEventListener::createGDBRegistrationListener();
  if (enable_perf_listener)
    perf_listener_ = llvm::JITEventListener::createPerfJITEventListener();
}

std::unique_ptr<MemoryBuffer> RuntimeExecutionEngine::obj_file() const {
  return obj_file_ ? MemoryBuffer::getMemBuffer(obj_file_->getMemBufferRef())
                   : nullptr;
}

// -------------------------------------------------------------------------- //

namespace {
using llvm::DenseMap;

// Intercept object compilation to save the object file corresponding to the
// XLA executable in the execution engine.
class ExecutionEngineObjectCache : public llvm::ObjectCache {
 public:
  void notifyObjectCompiled(const llvm::Module *m,
                            llvm::MemoryBufferRef objBuffer) override;

  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *m) override;

  // Transfer memory buffer from the cache to the caller.
  std::unique_ptr<llvm::MemoryBuffer> stealObject(const llvm::Module *m);

 private:
  DenseMap<const llvm::Module *, std::unique_ptr<llvm::MemoryBuffer>> objs_;
};
}  // namespace

void ExecutionEngineObjectCache::notifyObjectCompiled(
    const llvm::Module *m, llvm::MemoryBufferRef objBuffer) {
  objs_[m] = llvm::MemoryBuffer::getMemBufferCopy(
      objBuffer.getBuffer(), objBuffer.getBufferIdentifier());
}

std::unique_ptr<llvm::MemoryBuffer> ExecutionEngineObjectCache::getObject(
    const llvm::Module *m) {
  auto it = objs_.find(m);
  if (it == objs_.end()) return nullptr;
  return llvm::MemoryBuffer::getMemBuffer(it->second->getMemBufferRef());
}

std::unique_ptr<llvm::MemoryBuffer> ExecutionEngineObjectCache::stealObject(
    const llvm::Module *m) {
  auto it = objs_.find(m);
  if (it == objs_.end()) return nullptr;
  return std::move(it->second);
}

// -------------------------------------------------------------------------- //

// llvm_ir::DumpToString() is not used here, because we don't want to add too
// many dependencies to the runtime.
static std::string ToString(const llvm::Error &err) {
  std::string str;
  llvm::raw_string_ostream(str) << err;
  return str;
}

/*static*/ absl::StatusOr<std::unique_ptr<RuntimeExecutionEngine>>
RuntimeExecutionEngine::CreateFromModule(
    std::unique_ptr<llvm::LLVMContext> ctx,
    std::unique_ptr<llvm::Module> module, JitOptions options,
    absl::Span<const std::string_view> exported) {
  auto engine =
      std::unique_ptr<RuntimeExecutionEngine>(new RuntimeExecutionEngine(
          options.enable_gdb_listener, options.enable_perf_listener));

  // We'll need module pointer later to lookup object file in the cache.
  llvm::Module *module_ptr = module.get();

  // Set up the target machine details.
  if (!options.target_machine)
    return InternalError("target machine was not provided");
  module->setDataLayout(options.target_machine->createDataLayout());
  module->setTargetTriple(options.target_machine->getTargetTriple().str());

  // Run an optimization pipeline over the LLVM module (alway run with default
  // opt level independent of the options).
  //
  // TODO(ezhulenev): We should have out own optimizing transformer pipelines
  // for different Xla backends, e.g. there is absolutely no need to run
  // SLV vectorizer for Xla Gpi host side executable.
  auto transformer =
      options.make_optimizing_transformer(options.target_machine.get());
  if (auto err = transformer(module_ptr))
    return InternalError(
        StrFormat("failed to run optimization pipeline: %s", ToString(err)));

  // Callback to create the object layer with a user-provided section memory
  // mapper and JIT event listeners.
  auto obj_layer_creator = [&](ExecutionSession &session, const Triple &tt) {
    auto obj_layer = std::make_unique<RTDyldObjectLinkingLayer>(
        session, [section_memory_mapper = options.section_memory_mapper]() {
          return std::make_unique<SectionMemoryManager>(section_memory_mapper);
        });

    // Register JIT event listeners if they are enabled.
    if (engine->gdb_listener_)
      obj_layer->registerJITEventListener(*engine->gdb_listener_);
    if (engine->perf_listener_)
      obj_layer->registerJITEventListener(*engine->perf_listener_);

    return obj_layer;
  };

  // Callback to compile IR module on demand.
  auto compile_function_creator =
      [weak_target_machine = std::weak_ptr<llvm::TargetMachine>(
           options.target_machine)](JITTargetMachineBuilder)
      -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
    return std::make_unique<WeakCompiler>(weak_target_machine);
  };

  // Use in-process executor process control with in-place task dispatcher.
  auto executorProcessControl = SelfExecutorProcessControl::Create(
      nullptr, std::make_unique<InPlaceTaskDispatcher>());

  if (auto err = executorProcessControl.takeError())
    return InternalError(StrFormat(
        "failed to create executor process control: %s", ToString(err)));

  // TODO(b/286475799): Concurrent compilation leads to spurious memory
  // corruptions and segfaults at run time, however nothing shows up in tsan
  // or asan builds. This is a hack that for some unknown reason helps.
  static auto *lljit_mu = new absl::Mutex();
  std::optional<absl::MutexLock> lljit_lock(lljit_mu);

  // Construct the LLJIT with the given compiler and object linking layers.
  auto jit = llvm::orc::LLJITBuilder()
                 .setCompileFunctionCreator(std::move(compile_function_creator))
                 .setObjectLinkingLayerCreator(obj_layer_creator)
                 .setExecutorProcessControl(std::move(*executorProcessControl))
                 .setNumCompileThreads(0)  // disable multi-threading
                 .create();

  if (auto err = jit.takeError())
    return InternalError(
        StrFormat("failed to construct LLJIT: %s", ToString(err)));

  lljit_lock.reset();

  // Register input module with the LLJIT.
  ThreadSafeModule tsm(std::move(module), std::move(ctx));
  if (auto err = (*jit)->addIRModule(std::move(tsm)))
    return InternalError(
        StrFormat("failed to add source module: %s", ToString(err)));

  llvm::DataLayout data_layout = (*jit)->getDataLayout();

  // Resolve all exported functions to function pointers.
  for (std::string_view name : exported) {
    // Trigger compilation by looking up the exported function.
    // TODO(tsilytskyi):
    //   - Do we need to mangle function name?
    //   - Do we need to verify/adapt function proto to expected API?
    Expected<ExecutorAddr> addr = (*jit)->lookup(name);
    if (auto err = addr.takeError())
      return InternalError(StrFormat(
          "failed to compile exported function %s: %s", name, ToString(err)));

    // Check that we found an address of an exported function.
    auto ptr = addr->toPtr<ExportedFunctionPtr>();
    if (!ptr)
      return InternalError(
          StrFormat("exported function %s resolved to null", name));

    engine->exported_.push_back(ptr);
  }

  // Fill remaining fields and return constructed ExecutionEngine to the caller.
  engine->jit_ = std::move(*jit);
  return std::move(engine);
}

static std::function<llvm::Error(llvm::Module *)>
MakeOptimizingTransformerForJit(llvm::TargetMachine *targetMachine) {
  return [targetMachine](llvm::Module *m) -> llvm::Error {
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

absl::StatusOr<std::unique_ptr<LlvmExecutionEngine>>
LlvmExecutionEngine::CreateFromLlvmIr(absl::string_view name,
                                      absl::string_view entry,
                                      absl::string_view ir,
                                      absl::Span<const std::string> options) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic diagnostic;
  llvm::MemoryBufferRef ir_buffer(ir, name);
  std::unique_ptr<llvm::Module> llvm_module =
      llvm::parseAssembly(ir_buffer, diagnostic, *llvm_ctx, nullptr);

  // Prepare JIT target machine for code generation.
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) return absl::InternalError(toString(builder.takeError()));

  llvm::Expected<std::unique_ptr<llvm::TargetMachine>> target_machine =
      builder->createTargetMachine();
  if (!target_machine)
    return absl::InternalError(toString(target_machine.takeError()));

  // Set target triple
  llvm_module->setTargetTriple(
      llvm::StringRef(target_machine.get()->getTargetTriple().getTriple()));

  // Construct options for the XLA runtime execution engine.
  RuntimeExecutionEngine::JitOptions engine_options;
  engine_options.target_machine = std::move(target_machine.get());
  engine_options.make_optimizing_transformer = MakeOptimizingTransformerForJit;

  std::vector<std::string_view> exported = {entry};

  // Compile input module to the native function.
  TF_ASSIGN_OR_RETURN(auto engine,
                      RuntimeExecutionEngine::CreateFromModule(
                          std::move(llvm_ctx), std::move(llvm_module),
                          std::move(engine_options), exported));

  return std::unique_ptr<LlvmExecutionEngine>(
      new LlvmExecutionEngine(std::move(engine)));
}

}  // namespace host
}  // namespace stream_executor
