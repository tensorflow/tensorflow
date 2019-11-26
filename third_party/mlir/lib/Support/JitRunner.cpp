//===- jit-runner.cpp - MLIR CPU Execution Driver Library -----------------===//
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
// This is a library that provides a shared implementation for command line
// utilities that execute an MLIR file on the CPU by translating MLIR to LLVM
// IR before JIT-compiling and executing the latter.
//
// The translation can be customized by providing an MLIR to MLIR
// transformation.
//===----------------------------------------------------------------------===//

#include "mlir/Support/JitRunner.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include <numeric>

using namespace mlir;
using llvm::Error;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));
static llvm::cl::opt<std::string>
    mainFuncName("e", llvm::cl::desc("The function to be called"),
                 llvm::cl::value_desc("<function name>"),
                 llvm::cl::init("main"));
static llvm::cl::opt<std::string> mainFuncType(
    "entry-point-result",
    llvm::cl::desc("Textual description of the function type to be called"),
    llvm::cl::value_desc("f32 | void"), llvm::cl::init("f32"));

static llvm::cl::OptionCategory optFlags("opt-like flags");

// CLI list of pass information
static llvm::cl::list<const llvm::PassInfo *, bool, llvm::PassNameParser>
    llvmPasses(llvm::cl::desc("LLVM optimizing passes to run"),
               llvm::cl::cat(optFlags));

// CLI variables for -On options.
static llvm::cl::opt<bool>
    optO0("O0", llvm::cl::desc("Run opt passes and codegen at O0"),
          llvm::cl::cat(optFlags));
static llvm::cl::opt<bool>
    optO1("O1", llvm::cl::desc("Run opt passes and codegen at O1"),
          llvm::cl::cat(optFlags));
static llvm::cl::opt<bool>
    optO2("O2", llvm::cl::desc("Run opt passes and codegen at O2"),
          llvm::cl::cat(optFlags));
static llvm::cl::opt<bool>
    optO3("O3", llvm::cl::desc("Run opt passes and codegen at O3"),
          llvm::cl::cat(optFlags));

static llvm::cl::OptionCategory clOptionsCategory("linking options");
static llvm::cl::list<std::string>
    clSharedLibs("shared-libs", llvm::cl::desc("Libraries to link dynamically"),
                 llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
                 llvm::cl::cat(clOptionsCategory));

// CLI variables for debugging.
static llvm::cl::opt<bool> dumpObjectFile(
    "dump-object-file",
    llvm::cl::desc("Dump JITted-compiled object to file specified with "
                   "-object-filename (<input file>.o by default)."));

static llvm::cl::opt<std::string> objectFilename(
    "object-filename",
    llvm::cl::desc("Dump JITted-compiled object to file <input file>.o"));

static OwningModuleRef parseMLIRInput(StringRef inputFilename,
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
  return OwningModuleRef(parseSourceFile(sourceMgr, context));
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

static llvm::Optional<unsigned> getCommandLineOptLevel() {
  llvm::Optional<unsigned> optLevel;
  llvm::SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      optO0, optO1, optO2, optO3};

  // Determine if there is an optimization flag present.
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optLevel = j;
      break;
    }
  }
  return optLevel;
}

// JIT-compile the given module and run "entryPoint" with "args" as arguments.
static Error
compileAndExecute(ModuleOp module, StringRef entryPoint,
                  std::function<llvm::Error(llvm::Module *)> transformer,
                  void **args) {
  Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel;
  if (auto clOptLevel = getCommandLineOptLevel())
    jitCodeGenOptLevel =
        static_cast<llvm::CodeGenOpt::Level>(clOptLevel.getValue());
  SmallVector<StringRef, 4> libs(clSharedLibs.begin(), clSharedLibs.end());
  auto expectedEngine = mlir::ExecutionEngine::create(module, transformer,
                                                      jitCodeGenOptLevel, libs);
  if (!expectedEngine)
    return expectedEngine.takeError();

  auto engine = std::move(*expectedEngine);
  auto expectedFPtr = engine->lookup(entryPoint);
  if (!expectedFPtr)
    return expectedFPtr.takeError();

  if (dumpObjectFile)
    engine->dumpToObjectFile(objectFilename.empty() ? inputFilename + ".o"
                                                    : objectFilename);

  void (*fptr)(void **) = *expectedFPtr;
  (*fptr)(args);

  return Error::success();
}

static Error compileAndExecuteVoidFunction(
    ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.getBlocks().empty())
    return make_string_error("entry point not found");
  void *empty = nullptr;
  return compileAndExecute(module, entryPoint, transformer, &empty);
}

static Error compileAndExecuteSingleFloatReturnFunction(
    ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.isExternal())
    return make_string_error("entry point not found");

  if (mainFunction.getType().getFunctionNumParams() != 0)
    return make_string_error("function inputs not supported");

  if (!mainFunction.getType().getFunctionResultType().isFloatTy())
    return make_string_error("only single llvm.f32 function result supported");

  float res;
  struct {
    void *data;
  } data;
  data.data = &res;
  if (auto error =
          compileAndExecute(module, entryPoint, transformer, (void **)&data))
    return error;

  // Intentional printing of the output so we can test.
  llvm::outs() << res << '\n';

  return Error::success();
}

// Entry point for all CPU runners. Expects the common argc/argv arguments for
// standard C++ main functions and an mlirTransformer.
// The latter is applied after parsing the input into MLIR IR and before passing
// the MLIR module to the ExecutionEngine.
int mlir::JitRunnerMain(
    int argc, char **argv,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer) {
  llvm::InitLLVM y(argc, argv);

  initializeLLVM();
  mlir::initializeLLVMPasses();

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  llvm::Optional<unsigned> optLevel = getCommandLineOptLevel();
  llvm::SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      optO0, optO1, optO2, optO3};
  unsigned optCLIPosition = 0;
  // Determine if there is an optimization flag present, and its CLI position
  // (optCLIPosition).
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optCLIPosition = flag.getPosition();
      break;
    }
  }
  // Generate vector of pass information, plus the index at which we should
  // insert any optimization passes in that vector (optPosition).
  llvm::SmallVector<const llvm::PassInfo *, 4> passes;
  unsigned optPosition = 0;
  for (unsigned i = 0, e = llvmPasses.size(); i < e; ++i) {
    passes.push_back(llvmPasses[i]);
    if (optCLIPosition < llvmPasses.getPosition(i)) {
      optPosition = i;
      optCLIPosition = UINT_MAX; // To ensure we never insert again
    }
  }

  MLIRContext context;
  auto m = parseMLIRInput(inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  if (mlirTransformer)
    if (failed(mlirTransformer(m.get())))
      return EXIT_FAILURE;

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return EXIT_FAILURE;
  }
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    return EXIT_FAILURE;
  }

  auto transformer = mlir::makeLLVMPassesTransformer(
      passes, optLevel, /*targetMachine=*/tmOrError->get(), optPosition);

  // Get the function used to compile and execute the module.
  using CompileAndExecuteFnT = Error (*)(
      ModuleOp, StringRef, std::function<llvm::Error(llvm::Module *)>);
  auto compileAndExecuteFn =
      llvm::StringSwitch<CompileAndExecuteFnT>(mainFuncType.getValue())
          .Case("f32", compileAndExecuteSingleFloatReturnFunction)
          .Case("void", compileAndExecuteVoidFunction)
          .Default(nullptr);

  Error error =
      compileAndExecuteFn
          ? compileAndExecuteFn(m.get(), mainFuncName.getValue(), transformer)
          : make_string_error("unsupported function type");

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
