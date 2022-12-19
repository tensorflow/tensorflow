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

#include "tensorflow/compiler/xla/runtime/execution_engine.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "tensorflow/compiler/xla/runtime/errors.h"

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
using llvm::orc::IRCompileLayer;
using llvm::orc::JITTargetMachineBuilder;
using llvm::orc::RTDyldObjectLinkingLayer;
using llvm::orc::SymbolMap;
using llvm::orc::ThreadSafeModule;
using llvm::orc::TMOwningSimpleCompiler;

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

// Converts exported function to an interface function that wraps all the
// arguments of the original function into an i8** pointer to provide a function
// with trivial ABI.
static absl::Status SetUpExportedFunction(llvm::Module &module,
                                          std::string_view function_name) {
  llvm::IRBuilder<> builder(module.getContext());

  // Check that we have a function with a valid type.
  llvm::Function *func = module.getFunction(function_name);
  if (!func)
    return InternalError("exported function not found: %s", function_name);
  if (!func->getReturnType()->isVoidTy())
    return InternalError("exported function must return void");

  // Add an XLA interface function for the exported function.
  llvm::FunctionType *xla_runtime_type = llvm::FunctionType::get(
      builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
      /*isVarArg=*/false);

  llvm::FunctionCallee xla_runtime_func = module.getOrInsertFunction(
      GetExportedName(func->getName()), xla_runtime_type);

  llvm::Function *callee = cast<llvm::Function>(xla_runtime_func.getCallee());
  llvm::Value *packed_args = callee->arg_begin();

  // Load arguments from the type erased pointer array and cast them to the
  // original type.
  llvm::BasicBlock *bb = llvm::BasicBlock::Create(builder.getContext());
  bb->insertInto(callee);
  builder.SetInsertPoint(bb);

  llvm::SmallVector<llvm::Value *, 8> args;
  args.reserve(llvm::size(func->args()));

  for (auto &indexed_arg : llvm::enumerate(func->args())) {
    llvm::Value *arg_idx = llvm::Constant::getIntegerValue(
        builder.getInt64Ty(), llvm::APInt(64, indexed_arg.index()));
    llvm::Value *arg_ptr_ptr =
        builder.CreateGEP(builder.getInt8PtrTy(), packed_args, arg_idx);
    llvm::Value *arg_ptr =
        builder.CreateLoad(builder.getInt8PtrTy(), arg_ptr_ptr);
    llvm::Type *art_ty = indexed_arg.value().getType();
    arg_ptr = builder.CreateBitCast(arg_ptr, art_ty->getPointerTo());
    llvm::Value *arg = builder.CreateLoad(art_ty, arg_ptr);
    args.push_back(arg);
  }

  // Call the implementation function with the extracted arguments.
  auto *call = builder.CreateCall(func, args);

  // Force LLVM to inline original function into the interface function.
  call->addFnAttr(llvm::Attribute::AlwaysInline);

  // And make sure that we do not keep exported function in the binary if we do
  // not have other callers.
  func->setLinkage(llvm::GlobalValue::LinkageTypes::PrivateLinkage);

  builder.CreateRetVoid();

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
    return InternalError("target machine was not provided");
  module->setDataLayout(options.target_machine->createDataLayout());
  module->setTargetTriple(options.target_machine->getTargetTriple().str());

  // Run an optimization pipeline over the LLVM module.
  auto transformer = options.make_optimizing_transformer(
      options.opt_level, /*sizeLevel=*/0, options.target_machine);
  if (auto err = transformer(module_ptr))
    return InternalError("failed to run optimization pipeline: %s",
                         ToString(err));

  // Set up exported functions interface functions in the LLVM module.
  for (std::string_view name : exported) {
    if (auto status = SetUpExportedFunction(*module, name); !status.ok())
      return InternalError(
          "failed to set up exported function %s interface: %s", name,
          status.message());
  }

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
  std::unique_ptr<ExecutionEngineObjectCache> obj_cache =
      options.save_compiled_obj_file
          ? std::make_unique<ExecutionEngineObjectCache>()
          : nullptr;

  // Callback to compile IR module on demand.
  auto compile_function_creator = [&](JITTargetMachineBuilder jtmb)
      -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
    jtmb.setCodeGenOptLevel(options.opt_level);
    auto tm = jtmb.createTargetMachine();
    if (!tm) return tm.takeError();
    return std::make_unique<TMOwningSimpleCompiler>(std::move(*tm),
                                                    obj_cache.get());
  };

  // Construct the LLJIT with the given compiler and object linking layers.
  auto jit = llvm::orc::LLJITBuilder()
                 .setCompileFunctionCreator(compile_function_creator)
                 .setObjectLinkingLayerCreator(obj_layer_creator)
                 .create();
  if (auto err = jit.takeError())
    return InternalError("failed to construct LLJIT: %s", ToString(err));

  // Register input module with the LLJIT.
  ThreadSafeModule tsm(std::move(module), std::move(ctx));
  if (auto err = (*jit)->addIRModule(std::move(tsm)))
    return InternalError("failed to add source module: %s", ToString(err));

  llvm::orc::JITDylib &main_jd = (*jit)->getMainJITDylib();
  llvm::DataLayout data_layout = (*jit)->getDataLayout();

  // Register symbols that are statically linked in the current process.
  auto generator = DynamicLibrarySearchGenerator::GetForCurrentProcess(
      data_layout.getGlobalPrefix());
  if (auto err = generator.takeError())
    return InternalError("failed to construct DyLib search generator");
  main_jd.addGenerator(std::move(*generator));

  // Register user-provided symbols.
  if (options.symbols_binding) {
    auto mangle = llvm::orc::MangleAndInterner(main_jd.getExecutionSession(),
                                               data_layout);
    auto symbols = absoluteSymbols(options.symbols_binding(mangle));
    if (auto err = main_jd.define(symbols))
      return InternalError("failed to add symbols bindings: %s", ToString(err));
  }

  // Resolve all exported functions to function pointers.
  for (std::string_view name : exported) {
    // Trigger compilation by looking up the exported function.
    Expected<ExecutorAddr> addr = (*jit)->lookup(GetExportedName(name));
    if (auto err = addr.takeError())
      return InternalError("failed to compile exported function %s: %s", name,
                           ToString(err));

    // Check that we found an address of an exported function.
    auto ptr = addr->toPtr<ExportedFunctionPtr>();
    if (!ptr)
      return InternalError("exported function %s resolved to null", name);

    engine->exported_.push_back(ptr);
  }

  // Check that if we enabled object cache we have an object file for the
  // compiled module.
  std::unique_ptr<llvm::MemoryBuffer> obj_file =
      options.save_compiled_obj_file ? obj_cache->stealObject(module_ptr)
                                     : nullptr;
  if (options.save_compiled_obj_file && !obj_file)
    return InternalError("could not find object file for the XLA module");

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
    return InternalError("failed to construct LLJIT: %s", ToString(err));

  if (auto err = (*jit)->addObjectFile(std::move(obj_file)))
    return InternalError("failed to add object file: %s", ToString(err));

  llvm::orc::JITDylib &main_jd = (*jit)->getMainJITDylib();
  llvm::DataLayout data_layout = (*jit)->getDataLayout();

  // Register symbols that are statically linked in the current process.
  auto generator = DynamicLibrarySearchGenerator::GetForCurrentProcess(
      data_layout.getGlobalPrefix());
  if (auto err = generator.takeError())
    return InternalError("failed to construct DyLib search generator");
  main_jd.addGenerator(std::move(*generator));

  // Register user-provided symbols.
  if (options.symbols_binding) {
    auto mangle = llvm::orc::MangleAndInterner(main_jd.getExecutionSession(),
                                               data_layout);
    auto symbols = absoluteSymbols(options.symbols_binding(mangle));
    if (auto err = main_jd.define(symbols))
      return InternalError("failed to add symbols bindings: %s", ToString(err));
  }

  // Resolve all exported functions to function pointers.
  for (std::string_view name : exported) {
    // Lookup exported function in the loaded object file.
    Expected<ExecutorAddr> addr = (*jit)->lookup(GetExportedName(name));
    if (auto err = addr.takeError())
      return InternalError("failed to look up the exported function %s: %s",
                           name, ToString(err));

    // Check that we found an address of an exported function.
    auto ptr = addr->toPtr<ExportedFunctionPtr>();
    if (!ptr)
      return InternalError("exported function %s resolved to null", name);

    engine->exported_.push_back(ptr);
  }

  // Fill remaining fields and return constructed ExecutionEngine to the caller.
  engine->jit_ = std::move(*jit);
  return std::move(engine);
}

}  // namespace runtime
}  // namespace xla
