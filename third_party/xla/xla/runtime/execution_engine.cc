/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/runtime/execution_engine.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "xla/runtime/errors.h"

namespace xla {
namespace runtime {

using absl::StatusOr;
using absl::StrFormat;

using llvm::cast;

using llvm::Expected;
using llvm::MemoryBuffer;
using llvm::SectionMemoryManager;
using llvm::Triple;

using llvm::orc::DynamicLibrarySearchGenerator;
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
  explicit WeakCompiler(std::weak_ptr<llvm::TargetMachine> weak_target_machine,
                        std::weak_ptr<llvm::ObjectCache> weak_object_cache = {})
      : IRCompiler(IrManglingOptionsForWeakTargetMachine(weak_target_machine)),
        weak_target_machine_(weak_target_machine),
        weak_object_cache_(weak_object_cache) {}

  Expected<std::unique_ptr<MemoryBuffer>> operator()(
      llvm::Module &module) override {
    std::shared_ptr<llvm::TargetMachine> target_machine =
        weak_target_machine_.lock();
    CHECK(target_machine != nullptr)
        << "Compiler should not be used after the TargetMachine is destroyed.";

    // This may be nullptr, and that's fine.
    std::shared_ptr<llvm::ObjectCache> object_cache = weak_object_cache_.lock();

    SimpleCompiler compiler(*target_machine, object_cache.get());
    return compiler(module);
  }

 private:
  std::weak_ptr<llvm::TargetMachine> weak_target_machine_;
  std::weak_ptr<llvm::ObjectCache> weak_object_cache_;
};

}  // namespace

ExecutionEngine::ExecutionEngine(bool enable_gdb_listener,
                                 bool enable_perf_listener) {
  if (enable_gdb_listener)
    gdb_listener_ = llvm::JITEventListener::createGDBRegistrationListener();
  if (enable_perf_listener)
    perf_listener_ = llvm::JITEventListener::createPerfJITEventListener();
}

/*static*/ ExecutionEngine::SymbolsBinding ExecutionEngine::BindAll(
    std::vector<SymbolsBinding> bindings) {
  return [b = std::move(bindings)](llvm::orc::MangleAndInterner mangle) {
    llvm::orc::SymbolMap symbol_map;

    for (const SymbolsBinding &binding : b) {
      if (!binding) continue;
      auto symbols = binding(mangle);
      symbol_map.insert(symbols.begin(), symbols.end());
    }

    return symbol_map;
  };
}

std::unique_ptr<MemoryBuffer> ExecutionEngine::obj_file() const {
  return obj_file_ ? MemoryBuffer::getMemBuffer(obj_file_->getMemBufferRef())
                   : nullptr;
}

// -------------------------------------------------------------------------- //

static std::string GetExportedName(std::string_view name) {
  return StrFormat("__xla__%s", name);
}

absl::Status ExportWithXlaRuntimeAbi(llvm::Module &module,
                                     std::string_view original_name,
                                     std::string_view exported_name) {
  llvm::IRBuilder<> builder(module.getContext());

  // Check that we have a function with a valid type.
  llvm::Function *func = module.getFunction(original_name);
  if (!func)
    return Internal("exported function not found: %s", original_name);
  if (!func->getReturnType()->isVoidTy())
    return Internal("exported function must return void");

  // Add an XLA interface function for the exported function.
  llvm::FunctionType *xla_runtime_type =
      llvm::FunctionType::get(builder.getVoidTy(), builder.getPtrTy(),
                              /*isVarArg=*/false);

  llvm::FunctionCallee xla_runtime_func =
      module.getOrInsertFunction(exported_name, xla_runtime_type);

  llvm::Function *callee = cast<llvm::Function>(xla_runtime_func.getCallee());
  llvm::Value *packed_args = callee->arg_begin();

  // Load arguments from the type erased pointer array and cast them to the
  // original type.
  llvm::BasicBlock *bb = llvm::BasicBlock::Create(builder.getContext());
  bb->insertInto(callee);
  builder.SetInsertPoint(bb);

  llvm::SmallVector<llvm::Value *> args;
  args.reserve(llvm::size(func->args()));

  for (const auto &indexed_arg : llvm::enumerate(func->args())) {
    llvm::Type *art_ty = indexed_arg.value().getType();

    llvm::Value *arg_ptr_gep = builder.CreateConstGEP1_64(
        builder.getPtrTy(), packed_args, indexed_arg.index());
    llvm::LoadInst *arg_ptr_load =
        builder.CreateLoad(builder.getPtrTy(), arg_ptr_gep);
    llvm::LoadInst *arg_load = builder.CreateLoad(art_ty, arg_ptr_load);

    args.emplace_back(arg_load);
  }

  // Call the implementation function with the extracted arguments.
  auto *call = builder.CreateCall(func, args);
  builder.CreateRetVoid();

  // Make sure that we do not keep exported function in the binary if we do not
  // have any other callers.
  func->setLinkage(llvm::GlobalValue::LinkageTypes::PrivateLinkage);

  // Explicitly inline implementation function into the interface function,
  // because it potentially can have thousands of arguments and it interacts
  // badly with various SCCP passes in LLVM.
  llvm::InlineFunctionInfo ifi;

  // If inlined function is a coroutine (result of lowering async function),
  // then we have to mark the interface function as a corotuine as well.
  bool is_coro = func->isPresplitCoroutine();
  if (auto inlined = llvm::InlineFunction(*call, ifi); inlined.isSuccess()) {
    if (is_coro) callee->setPresplitCoroutine();
  }

  // Always keep the frame pointer inside jit-compiled modules, so that we can
  // correctly walk the stack when collecting profiles at run time.
  for (llvm::Function &fn : module.functions()) {
    if (!fn.isDeclaration()) fn.addFnAttr("frame-pointer", "all");
  }

  return absl::OkStatus();
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
std::string ToString(const llvm::Error &err) {
  std::string str;
  llvm::raw_string_ostream(str) << err;
  return str;
}

/*static*/ StatusOr<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::CreateFromModule(std::unique_ptr<llvm::LLVMContext> ctx,
                                  std::unique_ptr<llvm::Module> module,
                                  JitOptions options,
                                  absl::Span<const std::string_view> exported) {
  auto engine = std::unique_ptr<ExecutionEngine>(new ExecutionEngine(
      options.enable_gdb_listener, options.enable_perf_listener));

  // We'll need module pointer later to lookup object file in the cache.
  llvm::Module *module_ptr = module.get();

  // Set up the target machine details.
  if (!options.target_machine)
    return Internal("target machine was not provided");
  module->setDataLayout(options.target_machine->createDataLayout());
  module->setTargetTriple(options.target_machine->getTargetTriple().str());

  // Set up exported functions interface functions in the LLVM module.
  for (std::string_view name : exported) {
    if (auto status =
            ExportWithXlaRuntimeAbi(*module, name, GetExportedName(name));
        !status.ok()) {
      return Internal(
          "failed to set up exported function %s interface: %s", name,
          status.message());
    }
  }

  // Run an optimization pipeline over the LLVM module (alway run with default
  // opt level independent of the options).
  //
  // TODO(ezhulenev): We should have out own optimizing transformer pipelines
  // for different Xla backends, e.g. there is absolutely no need to run
  // SLV vectorizer for Xla Gpi host side executable.
  auto transformer =
      options.make_optimizing_transformer(options.target_machine.get());
  if (auto err = transformer(module_ptr))
    return Internal("failed to run optimization pipeline: %s",
                         ToString(err));

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

  // Optionally enable cache for compiled object files.
  // Not using std::make_shared to allocate the control block and the object in
  // separate allocations.
  std::shared_ptr<ExecutionEngineObjectCache> obj_cache =
      options.save_compiled_obj_file
          ? std::shared_ptr<ExecutionEngineObjectCache>(
                std::make_unique<ExecutionEngineObjectCache>())
          : nullptr;

  // Callback to compile IR module on demand.
  auto compile_function_creator =
      [weak_target_machine =
           std::weak_ptr<llvm::TargetMachine>(options.target_machine),
       weak_obj_cache =
           std::weak_ptr<llvm::ObjectCache>(obj_cache)](JITTargetMachineBuilder)
      -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
    return std::make_unique<WeakCompiler>(weak_target_machine, weak_obj_cache);
  };

  // Use in-process executor process control with in-place task dispatcher.
  auto executorProcessControl = SelfExecutorProcessControl::Create(
      nullptr, std::make_unique<InPlaceTaskDispatcher>());

  if (auto err = executorProcessControl.takeError())
    return Internal("failed to create executor process control: %s",
                         ToString(err));

  // TODO(b/286475799): Concurrent compilation leads to spurious memory
  // corruptions and segfaults at run time, however nothing shows up in tsan
  // or asan builds. This is a hack that for some unknown reason helps.
  static auto *lljit_mu = new absl::Mutex();
  std::optional<absl::MutexLock> lljit_lock(lljit_mu);

  // Construct the LLJIT with the given compiler and object linking layers.
  auto jit = llvm::orc::LLJITBuilder()
                 .setCompileFunctionCreator(compile_function_creator)
                 .setObjectLinkingLayerCreator(obj_layer_creator)
                 .setExecutorProcessControl(std::move(*executorProcessControl))
                 .setNumCompileThreads(0)  // disable multi-threading
                 .create();

  if (auto err = jit.takeError())
    return Internal("failed to construct LLJIT: %s", ToString(err));

  lljit_lock.reset();

  // Register input module with the LLJIT.
  ThreadSafeModule tsm(std::move(module), std::move(ctx));
  if (auto err = (*jit)->addIRModule(std::move(tsm)))
    return Internal("failed to add source module: %s", ToString(err));

  llvm::orc::JITDylib &main_jd = (*jit)->getMainJITDylib();
  llvm::DataLayout data_layout = (*jit)->getDataLayout();

  // Register user-provided symbols.
  if (options.symbols_binding) {
    auto mangle = llvm::orc::MangleAndInterner(main_jd.getExecutionSession(),
                                               data_layout);
    auto symbols = absoluteSymbols(options.symbols_binding(mangle));
    if (auto err = main_jd.define(symbols))
      return Internal("failed to add symbols bindings: %s", ToString(err));
  }

  // Resolve all exported functions to function pointers.
  for (std::string_view name : exported) {
    // Trigger compilation by looking up the exported function.
    Expected<ExecutorAddr> addr = (*jit)->lookup(GetExportedName(name));
    if (auto err = addr.takeError())
      return Internal("failed to compile exported function %s: %s", name,
                           ToString(err));

    // Check that we found an address of an exported function.
    auto ptr = addr->toPtr<ExportedFunctionPtr>();
    if (!ptr)
      return Internal("exported function %s resolved to null", name);

    engine->exported_.push_back(ptr);
  }

  // Check that if we enabled object cache we have an object file for the
  // compiled module.
  std::unique_ptr<llvm::MemoryBuffer> obj_file =
      options.save_compiled_obj_file ? obj_cache->stealObject(module_ptr)
                                     : nullptr;
  if (options.save_compiled_obj_file && !obj_file)
    return Internal("could not find object file for the XLA module");

  // Fill remaining fields and return constructed ExecutionEngine to the caller.
  engine->jit_ = std::move(*jit);
  engine->obj_file_ = std::move(obj_file);
  return std::move(engine);
}

static void InitializeLlvmNativeTarget() {
  static const bool initialized = [] {
    llvm::InitializeNativeTarget();
    return true;
  }();
  (void)initialized;
}

/*static*/ StatusOr<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::CreateFromObjFile(
    std::unique_ptr<llvm::MemoryBuffer> obj_file, AotOptions options,
    absl::Span<const std::string_view> exported) {
  auto engine = std::unique_ptr<ExecutionEngine>(new ExecutionEngine(
      options.enable_gdb_listener, options.enable_perf_listener));

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

  // Initialize LLVM native target before constructing LLJIT.
  InitializeLlvmNativeTarget();

  // Construct the LLJIT with the given compiler and object linking layers.
  auto jit = llvm::orc::LLJITBuilder()
                 .setObjectLinkingLayerCreator(obj_layer_creator)
                 .create();
  if (auto err = jit.takeError())
    return Internal("failed to construct LLJIT: %s", ToString(err));

  if (auto err = (*jit)->addObjectFile(std::move(obj_file)))
    return Internal("failed to add object file: %s", ToString(err));

  llvm::orc::JITDylib &main_jd = (*jit)->getMainJITDylib();
  llvm::DataLayout data_layout = (*jit)->getDataLayout();

  // Register symbols that are statically linked in the current process.
  auto generator = DynamicLibrarySearchGenerator::GetForCurrentProcess(
      data_layout.getGlobalPrefix());
  if (auto err = generator.takeError())
    return Internal("failed to construct DyLib search generator");
  main_jd.addGenerator(std::move(*generator));

  // Register user-provided symbols.
  if (options.symbols_binding) {
    auto mangle = llvm::orc::MangleAndInterner(main_jd.getExecutionSession(),
                                               data_layout);
    auto symbols = absoluteSymbols(options.symbols_binding(mangle));
    if (auto err = main_jd.define(symbols))
      return Internal("failed to add symbols bindings: %s", ToString(err));
  }

  // Resolve all exported functions to function pointers.
  for (std::string_view name : exported) {
    // Lookup exported function in the loaded object file.
    Expected<ExecutorAddr> addr = (*jit)->lookup(GetExportedName(name));
    if (auto err = addr.takeError())
      return Internal("failed to look up the exported function %s: %s",
                           name, ToString(err));

    // Check that we found an address of an exported function.
    auto ptr = addr->toPtr<ExportedFunctionPtr>();
    if (!ptr)
      return Internal("exported function %s resolved to null", name);

    engine->exported_.push_back(ptr);
  }

  // Fill remaining fields and return constructed ExecutionEngine to the caller.
  engine->jit_ = std::move(*jit);
  return std::move(engine);
}

}  // namespace runtime
}  // namespace xla
