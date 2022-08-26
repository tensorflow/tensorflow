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

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "tensorflow/compiler/xla/runtime/errors.h"

namespace xla {
namespace runtime {

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

static std::string GetEntrypointName(std::string_view name) {
  return llvm::formatv("__xla__{0}", name);
}

// Converts entrypoint function to an interface function that wraps all the
// arguments of the original function into an i8** pointer to provide a function
// with trivial ABI.
static llvm::Error SetUpEntrypointFunction(llvm::Module &module,
                                           std::string_view entrypoint) {
  llvm::IRBuilder<> builder(module.getContext());

  // Check that we have an entrypoint function with a valid type.
  llvm::Function *func = module.getFunction(entrypoint);
  if (!func)
    return MakeStringError("entrypoint function not found: ", entrypoint);
  if (!func->getReturnType()->isVoidTy())
    return MakeStringError("entrypoint function must return void");

  // Add an XLA interface function for the entrypoint.
  llvm::FunctionType *xla_runtime_type = llvm::FunctionType::get(
      builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
      /*isVarArg=*/false);

  llvm::FunctionCallee xla_runtime_func = module.getOrInsertFunction(
      GetEntrypointName(func->getName()), xla_runtime_type);

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
  builder.CreateCall(func, args);
  builder.CreateRetVoid();

  return llvm::Error::success();
}

// -------------------------------------------------------------------------- //

namespace {
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
  llvm::DenseMap<const llvm::Module *, std::unique_ptr<llvm::MemoryBuffer>>
      objs_;
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

/*static*/ Expected<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::CreateFromModule(std::unique_ptr<llvm::LLVMContext> ctx,
                                  std::unique_ptr<llvm::Module> module,
                                  std::string_view entrypoint,
                                  JitOptions options) {
  auto engine = std::unique_ptr<ExecutionEngine>(new ExecutionEngine(
      options.enable_gdb_listener, options.enable_perf_listener));

  // We'll need module pointer later to lookup object file in the cache.
  llvm::Module *module_ptr = module.get();

  // Set up the target machine details.
  if (!options.target_machine)
    return MakeStringError("target machine was not provided");
  module->setDataLayout(options.target_machine->createDataLayout());
  module->setTargetTriple(options.target_machine->getTargetTriple().str());

  // Run an optimization pipeline over the LLVM module.
  auto transformer = options.make_optimizing_transformer(
      options.opt_level, /*sizeLevel=*/0, options.target_machine);
  if (auto err = transformer(module_ptr))
    return MakeStringError("failed to run optimization pipeline: ", err);

  // Set up the entry point function compatible with XLA ABI.
  if (auto err = SetUpEntrypointFunction(*module, entrypoint))
    return MakeStringError("failed to set up entrypoint ABI: ", err);

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
    return MakeStringError("failed to construct LLJIT: ", err);

  // Register input module with the LLJIT.
  ThreadSafeModule tsm(std::move(module), std::move(ctx));
  if (auto err = (*jit)->addIRModule(std::move(tsm)))
    return MakeStringError("failed to add source module: ", err);

  llvm::orc::JITDylib &main_jd = (*jit)->getMainJITDylib();
  llvm::DataLayout data_layout = (*jit)->getDataLayout();

  // Register symbols that are statically linked in the current process.
  auto generator = DynamicLibrarySearchGenerator::GetForCurrentProcess(
      data_layout.getGlobalPrefix());
  if (auto err = generator.takeError())
    return MakeStringError("failed to construct DyLib search generator");
  main_jd.addGenerator(std::move(*generator));

  // Register user-provided symbols.
  if (options.symbols_binding) {
    auto mangle = llvm::orc::MangleAndInterner(main_jd.getExecutionSession(),
                                               data_layout);
    auto symbols = absoluteSymbols(options.symbols_binding(mangle));
    if (auto err = main_jd.define(symbols))
      return MakeStringError("failed to add symbols bindings: ", err);
  }

  // Trigger compilation by looking up the entrypoint function.
  Expected<ExecutorAddr> addr = (*jit)->lookup(GetEntrypointName(entrypoint));
  if (auto err = addr.takeError())
    return MakeStringError("failed to compile the entrypoint: ", err);

  // Check that we found an address of an entrypoint function.
  auto ptr = addr->toPtr<EntrypointFunctionPtr>();
  if (!ptr) return MakeStringError("entrypoint function resolved to null");

  // Check that if we enabled object cache we have an object file for the
  // compiled module.
  std::unique_ptr<llvm::MemoryBuffer> obj_file =
      options.save_compiled_obj_file ? obj_cache->stealObject(module_ptr)
                                     : nullptr;
  if (options.save_compiled_obj_file && !obj_file)
    return MakeStringError("could not find object file for the XLA module");

  // Fill remaining fields and return constructed ExecutionEngine to the caller.
  engine->jit_ = std::move(*jit);
  engine->entrypoint_ptr_ = ptr;
  engine->obj_file_ = std::move(obj_file);
  return std::move(engine);
}

/*static*/ Expected<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::CreateFromObjFile(std::unique_ptr<llvm::MemoryBuffer> obj_file,
                                   std::string_view entrypoint,
                                   AotOptions options) {
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

  // Construct the LLJIT with the given compiler and object linking layers.
  auto jit = llvm::orc::LLJITBuilder()
                 .setObjectLinkingLayerCreator(obj_layer_creator)
                 .create();
  if (auto err = jit.takeError())
    return MakeStringError("failed to construct LLJIT: ", err);

  if (auto err = (*jit)->addObjectFile(std::move(obj_file)))
    return MakeStringError("failed to add object file: ", err);

  llvm::orc::JITDylib &main_jd = (*jit)->getMainJITDylib();
  llvm::DataLayout data_layout = (*jit)->getDataLayout();

  // Register symbols that are statically linked in the current process.
  auto generator = DynamicLibrarySearchGenerator::GetForCurrentProcess(
      data_layout.getGlobalPrefix());
  if (auto err = generator.takeError())
    return MakeStringError("failed to construct DyLib search generator");
  main_jd.addGenerator(std::move(*generator));

  // Register user-provided symbols.
  if (options.symbols_binding) {
    auto mangle = llvm::orc::MangleAndInterner(main_jd.getExecutionSession(),
                                               data_layout);
    auto symbols = absoluteSymbols(options.symbols_binding(mangle));
    if (auto err = main_jd.define(symbols))
      return MakeStringError("failed to add symbols bindings: ", err);
  }

  // Lookup entrypoint in the loaded object file.
  Expected<ExecutorAddr> addr = (*jit)->lookup(GetEntrypointName(entrypoint));
  if (auto err = addr.takeError())
    return MakeStringError("failed to lookup the entrypoint: ", err);

  // Check that we found an address of an entrypoint function.
  auto ptr = addr->toPtr<EntrypointFunctionPtr>();
  if (!ptr) return MakeStringError("entrypoint function resolved to null");

  // Fill remaining fields and return constructed ExecutionEngine to the caller.
  engine->jit_ = std::move(*jit);
  engine->entrypoint_ptr_ = ptr;
  return std::move(engine);
}

}  // namespace runtime
}  // namespace xla
