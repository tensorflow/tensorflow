//===- mlir-cpu-runner.cpp - MLIR CPU Execution Driver---------------------===//
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
// This is a command line utility that executes an MLIR file on the CPU by
// translating MLIR to LLVM IR before JIT-compiling and executing the latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using llvm::Error;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));
static llvm::cl::opt<std::string>
    initValue("init-value", llvm::cl::desc("Initial value of MemRef elements"),
              llvm::cl::value_desc("<float value>"), llvm::cl::init("0.0"));
static llvm::cl::opt<std::string>
    mainFuncName("e", llvm::cl::desc("The function to be called"),
                 llvm::cl::value_desc("<function name>"),
                 llvm::cl::init("main"));

static std::unique_ptr<Module> parseMLIRInput(StringRef inputFilename,
                                              MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return std::unique_ptr<Module>(parseSourceFile(sourceMgr, context));
}

// Initialize the relevant subsystems of LLVM.
static void initializeLLVM() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

static inline Error make_string_error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

static void printOneMemRef(Type t, void *val) {
  auto memRefType = t.cast<MemRefType>();
  auto shape = memRefType.getShape();
  int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<int64_t>());
  for (int64_t i = 0; i < size; ++i) {
    llvm::outs() << reinterpret_cast<StaticFloatMemRef *>(val)->data[i] << ' ';
  }
  llvm::outs() << '\n';
}

static void printMemRefArguments(const Function *func, ArrayRef<void *> args) {
  auto properArgs = args.take_front(func->getNumArguments());
  for (const auto &kvp : llvm::zip(func->getArguments(), properArgs)) {
    auto arg = std::get<0>(kvp);
    auto val = std::get<1>(kvp);
    printOneMemRef(arg->getType(), val);
  }

  auto results = args.drop_front(func->getNumArguments());
  for (const auto &kvp : llvm::zip(func->getType().getResults(), results)) {
    auto type = std::get<0>(kvp);
    auto val = std::get<1>(kvp);
    printOneMemRef(type, val);
  }
}

static Error compileAndExecute(Module *module, StringRef entryPoint) {
  Function *mainFunction = module->getNamedFunction(entryPoint);
  if (!mainFunction || mainFunction->getBlocks().empty()) {
    return make_string_error("entry point not found");
  }

  float init = std::stof(initValue.getValue());

  auto expectedArguments = allocateMemRefArguments(mainFunction, init);
  if (!expectedArguments)
    return expectedArguments.takeError();

  auto expectedEngine = mlir::ExecutionEngine::create(module);
  if (!expectedEngine)
    return expectedEngine.takeError();

  auto engine = std::move(*expectedEngine);
  auto expectedFPtr = engine->lookup(entryPoint);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  void (*fptr)(void **) = *expectedFPtr;
  (*fptr)(expectedArguments->data());
  printMemRefArguments(mainFunction, *expectedArguments);
  freeMemRefArguments(*expectedArguments);

  return Error::success();
}

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  initializeLLVM();

  MLIRContext context;
  auto m = parseMLIRInput(inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }
  auto error = compileAndExecute(m.get(), mainFuncName.getValue());
  int exitCode = EXIT_SUCCESS;
  llvm::handleAllErrors(std::move(error),
                        [&exitCode](const llvm::ErrorInfoBase &info) {
                          llvm::errs() << "Error: ";
                          info.log(llvm::errs());
                          llvm::errs() << '\n';
                          exitCode = EXIT_FAILURE;
                        });

  return exitCode;
}
