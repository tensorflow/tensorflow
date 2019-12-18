//===- ExecutionEngine.cpp - MLIR Execution engine and utils --------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the execution engine for MLIR modules based on LLVM Orc
// JIT engine.
//
//===----------------------------------------------------------------------===//
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "execution-engine"

using namespace mlir;
using llvm::dbgs;
using llvm::Error;
using llvm::errs;
using llvm::Expected;
using llvm::LLVMContext;
using llvm::MemoryBuffer;
using llvm::MemoryBufferRef;
using llvm::Module;
using llvm::SectionMemoryManager;
using llvm::StringError;
using llvm::Triple;
using llvm::orc::DynamicLibrarySearchGenerator;
using llvm::orc::ExecutionSession;
using llvm::orc::IRCompileLayer;
using llvm::orc::JITTargetMachineBuilder;
using llvm::orc::RTDyldObjectLinkingLayer;
using llvm::orc::ThreadSafeModule;
using llvm::orc::TMOwningSimpleCompiler;

/// Wrap a string into an llvm::StringError.
static Error make_string_error(const Twine &message) {
  return llvm::make_error<StringError>(message.str(),
                                       llvm::inconvertibleErrorCode());
}

void SimpleObjectCache::notifyObjectCompiled(const Module *M,
                                             MemoryBufferRef ObjBuffer) {
  cachedObjects[M->getModuleIdentifier()] = MemoryBuffer::getMemBufferCopy(
      ObjBuffer.getBuffer(), ObjBuffer.getBufferIdentifier());
}

std::unique_ptr<MemoryBuffer> SimpleObjectCache::getObject(const Module *M) {
  auto I = cachedObjects.find(M->getModuleIdentifier());
  if (I == cachedObjects.end()) {
    LLVM_DEBUG(dbgs() << "No object for " << M->getModuleIdentifier()
                      << " in cache. Compiling.\n");
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "Object for " << M->getModuleIdentifier()
                    << " loaded from cache.\n");
  return MemoryBuffer::getMemBuffer(I->second->getMemBufferRef());
}

void SimpleObjectCache::dumpToObjectFile(StringRef outputFilename) {
  // Set up the output file.
  std::string errorMessage;
  auto file = openOutputFile(outputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return;
  }

  // Dump the object generated for a single module to the output file.
  assert(cachedObjects.size() == 1 && "Expected only one object entry.");
  auto &cachedObject = cachedObjects.begin()->second;
  file->os() << cachedObject->getBuffer();
  file->keep();
}

void ExecutionEngine::dumpToObjectFile(StringRef filename) {
  cache->dumpToObjectFile(filename);
}

// Setup LLVM target triple from the current machine.
bool ExecutionEngine::setupTargetTriple(Module *llvmModule) {
  // Setup the machine properties from the current architecture.
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    errs() << "NO target: " << errorMessage << "\n";
    return true;
  }
  std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(targetTriple, "generic", "", {}, {}));
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
  return false;
}

static std::string makePackedFunctionName(StringRef name) {
  return "_mlir_" + name.str();
}

// For each function in the LLVM module, define an interface function that wraps
// all the arguments of the original function and all its results into an i8**
// pointer to provide a unified invocation interface.
void packFunctionArguments(Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto newType = llvm::FunctionType::get(
        builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
        /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc = cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto &indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, indexedArg.index()));
      llvm::Value *argPtrPtr = builder.CreateGEP(argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(argPtrPtr);
      argPtr = builder.CreateBitCast(
          argPtr, indexedArg.value().getType()->getPointerTo());
      llvm::Value *arg = builder.CreateLoad(argPtr);
      args.push_back(arg);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr = builder.CreateGEP(argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}

ExecutionEngine::ExecutionEngine(bool enableObjectCache)
    : cache(enableObjectCache ? nullptr : new SimpleObjectCache()) {}

Expected<std::unique_ptr<ExecutionEngine>> ExecutionEngine::create(
    ModuleOp m, std::function<Error(llvm::Module *)> transformer,
    Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel,
    ArrayRef<StringRef> sharedLibPaths, bool enableObjectCache) {
  auto engine = std::make_unique<ExecutionEngine>(enableObjectCache);

  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = translateModuleToLLVMIR(m);
  if (!llvmModule)
    return make_string_error("could not convert to LLVM IR");
  // FIXME: the triple should be passed to the translation or dialect conversion
  // instead of this.  Currently, the LLVM module created above has no triple
  // associated with it.
  setupTargetTriple(llvmModule.get());
  packFunctionArguments(llvmModule.get());

  // Clone module in a new LLVMContext since translateModuleToLLVMIR buries
  // ownership too deeply.
  // TODO(zinenko): Reevaluate model of ownership of LLVMContext in LLVMDialect.
  SmallVector<char, 1> buffer;
  {
    llvm::raw_svector_ostream os(buffer);
    WriteBitcodeToFile(*llvmModule, os);
  }
  llvm::MemoryBufferRef bufferRef(StringRef(buffer.data(), buffer.size()),
                                  "cloned module buffer");
  auto expectedModule = parseBitcodeFile(bufferRef, *ctx);
  if (!expectedModule)
    return expectedModule.takeError();
  std::unique_ptr<Module> deserModule = std::move(*expectedModule);

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](ExecutionSession &session,
                                       const Triple &TT) {
    auto objectLayer = std::make_unique<RTDyldObjectLinkingLayer>(
        session, []() { return std::make_unique<SectionMemoryManager>(); });
    auto dataLayout = deserModule->getDataLayout();
    llvm::orc::JITDylib *mainJD = session.getJITDylibByName("<main>");
    if (!mainJD)
      mainJD = &session.createJITDylib("<main>");

    // Resolve symbols that are statically linked in the current process.
    mainJD->addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            dataLayout.getGlobalPrefix())));

    // Resolve symbols from shared libraries.
    for (auto libPath : sharedLibPaths) {
      auto mb = llvm::MemoryBuffer::getFile(libPath);
      if (!mb) {
        errs() << "Fail to create MemoryBuffer for: " << libPath << "\n";
        continue;
      }
      auto &JD = session.createJITDylib(libPath);
      auto loaded = DynamicLibrarySearchGenerator::Load(
          libPath.data(), dataLayout.getGlobalPrefix());
      if (!loaded) {
        errs() << "Could not load " << libPath << ":\n  " << loaded.takeError()
               << "\n";
        continue;
      }
      JD.addGenerator(std::move(*loaded));
      cantFail(objectLayer->add(JD, std::move(mb.get())));
    }

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows Lang's
  // LLJITWithObjectCache example.
  auto compileFunctionCreator = [&](JITTargetMachineBuilder JTMB)
      -> Expected<IRCompileLayer::CompileFunction> {
    if (jitCodeGenOptLevel)
      JTMB.setCodeGenOptLevel(jitCodeGenOptLevel.getValue());
    auto TM = JTMB.createTargetMachine();
    if (!TM)
      return TM.takeError();
    return IRCompileLayer::CompileFunction(
        TMOwningSimpleCompiler(std::move(*TM), engine->cache.get()));
  };

  // Create the LLJIT by calling the LLJITBuilder with 2 callbacks.
  auto jit =
      cantFail(llvm::orc::LLJITBuilder()
                   .setCompileFunctionCreator(compileFunctionCreator)
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                   .create());

  // Add a ThreadSafemodule to the engine and return.
  ThreadSafeModule tsm(std::move(deserModule), std::move(ctx));
  if (transformer)
    cantFail(tsm.withModuleDo(
        [&](llvm::Module &module) { return transformer(&module); }));
  cantFail(jit->addIRModule(std::move(tsm)));
  engine->jit = std::move(jit);

  return std::move(engine);
}

Expected<void (*)(void **)> ExecutionEngine::lookup(StringRef name) const {
  auto expectedSymbol = jit->lookup(makePackedFunctionName(name));
  if (!expectedSymbol)
    return expectedSymbol.takeError();
  auto rawFPtr = expectedSymbol->getAddress();
  auto fptr = reinterpret_cast<void (*)(void **)>(rawFPtr);
  if (!fptr)
    return make_string_error("looked up function is null");
  return fptr;
}

Error ExecutionEngine::invoke(StringRef name, MutableArrayRef<void *> args) {
  auto expectedFPtr = lookup(name);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  auto fptr = *expectedFPtr;

  (*fptr)(args.data());

  return Error::success();
}
