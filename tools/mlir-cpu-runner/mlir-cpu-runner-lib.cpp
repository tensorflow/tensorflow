//===- mlir-cpu-runner-lib.cpp - MLIR CPU Execution Driver Library --------===//
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

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
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
    initValue("init-value", llvm::cl::desc("Initial value of MemRef elements"),
              llvm::cl::value_desc("<float value>"), llvm::cl::init("0.0"));
static llvm::cl::opt<std::string>
    mainFuncName("e", llvm::cl::desc("The function to be called"),
                 llvm::cl::value_desc("<function name>"),
                 llvm::cl::init("main"));
static llvm::cl::opt<std::string> mainFuncType(
    "entry-point-result",
    llvm::cl::desc("Textual description of the function type to be called"),
    llvm::cl::value_desc("f32 or memrefs"), llvm::cl::init("memrefs"));

static llvm::cl::OptionCategory optFlags("opt-like flags");

// CLI list of pass information
static llvm::cl::list<const llvm::PassInfo *, bool, llvm::PassNameParser>
    llvmPasses(llvm::cl::desc("LLVM optimizing passes to run"),
               llvm::cl::cat(optFlags));

// CLI variables for -On options.
static llvm::cl::opt<bool> optO0("O0", llvm::cl::desc("Run opt O0 passes"),
                                 llvm::cl::cat(optFlags));
static llvm::cl::opt<bool> optO1("O1", llvm::cl::desc("Run opt O1 passes"),
                                 llvm::cl::cat(optFlags));
static llvm::cl::opt<bool> optO2("O2", llvm::cl::desc("Run opt O2 passes"),
                                 llvm::cl::cat(optFlags));
static llvm::cl::opt<bool> optO3("O3", llvm::cl::desc("Run opt O3 passes"),
                                 llvm::cl::cat(optFlags));

static llvm::cl::OptionCategory clOptionsCategory("linking options");
static llvm::cl::list<std::string>
    clSharedLibs("shared-libs", llvm::cl::desc("Libraries to link dynamically"),
                 llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
                 llvm::cl::cat(clOptionsCategory));

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

static void printMemRefArguments(ArrayRef<Type> argTypes,
                                 ArrayRef<Type> resTypes,
                                 ArrayRef<void *> args) {
  auto properArgs = args.take_front(argTypes.size());
  for (const auto &kvp : llvm::zip(argTypes, properArgs)) {
    auto type = std::get<0>(kvp);
    auto val = std::get<1>(kvp);
    printOneMemRef(type, val);
  }

  auto results = args.drop_front(argTypes.size());
  for (const auto &kvp : llvm::zip(resTypes, results)) {
    auto type = std::get<0>(kvp);
    auto val = std::get<1>(kvp);
    printOneMemRef(type, val);
  }
}

// Calls the passes necessary to convert affine and standard dialects to the
// LLVM IR dialect.
// Currently, these passes are:
// - CSE
// - canonicalization
// - affine to standard lowering
// - standard to llvm lowering
static LogicalResult convertAffineStandardToLLVMIR(ModuleOp module) {
  PassManager manager;
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::createConvertToLLVMIRPass());
  return manager.run(module);
}

static Error compileAndExecuteFunctionWithMemRefs(
    ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  FuncOp mainFunction = module.lookupSymbol<FuncOp>(entryPoint);
  if (!mainFunction || mainFunction.getBlocks().empty()) {
    return make_string_error("entry point not found");
  }

  // Store argument and result types of the original function necessary to
  // pretty print the results, because the function itself will be rewritten
  // to use the LLVM dialect.
  SmallVector<Type, 8> argTypes =
      llvm::to_vector<8>(mainFunction.getType().getInputs());
  SmallVector<Type, 8> resTypes =
      llvm::to_vector<8>(mainFunction.getType().getResults());

  float init = std::stof(initValue.getValue());

  auto expectedArguments = allocateMemRefArguments(mainFunction, init);
  if (!expectedArguments)
    return expectedArguments.takeError();

  if (failed(convertAffineStandardToLLVMIR(module)))
    return make_string_error("conversion to the LLVM IR dialect failed");

  SmallVector<StringRef, 4> libs(clSharedLibs.begin(), clSharedLibs.end());
  auto expectedEngine =
      mlir::ExecutionEngine::create(module, transformer, libs);
  if (!expectedEngine)
    return expectedEngine.takeError();

  auto engine = std::move(*expectedEngine);
  auto expectedFPtr = engine->lookup(entryPoint);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  void (*fptr)(void **) = *expectedFPtr;
  (*fptr)(expectedArguments->data());
  printMemRefArguments(argTypes, resTypes, *expectedArguments);
  freeMemRefArguments(*expectedArguments);

  return Error::success();
}

static Error compileAndExecuteSingleFloatReturnFunction(
    ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  FuncOp mainFunction = module.lookupSymbol<FuncOp>(entryPoint);
  if (!mainFunction || mainFunction.isExternal()) {
    return make_string_error("entry point not found");
  }

  if (!mainFunction.getType().getInputs().empty())
    return make_string_error("function inputs not supported");

  if (mainFunction.getType().getResults().size() != 1)
    return make_string_error("only single f32 function result supported");

  auto t = mainFunction.getType().getResults()[0].dyn_cast<LLVM::LLVMType>();
  if (!t)
    return make_string_error("only single llvm.f32 function result supported");
  auto *llvmTy = t.getUnderlyingType();
  if (llvmTy != llvmTy->getFloatTy(llvmTy->getContext()))
    return make_string_error("only single llvm.f32 function result supported");

  SmallVector<StringRef, 4> libs(clSharedLibs.begin(), clSharedLibs.end());
  auto expectedEngine =
      mlir::ExecutionEngine::create(module, transformer, libs);
  if (!expectedEngine)
    return expectedEngine.takeError();

  auto engine = std::move(*expectedEngine);
  auto expectedFPtr = engine->lookup(entryPoint);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  void (*fptr)(void **) = *expectedFPtr;

  float res;
  struct {
    void *data;
  } data;
  data.data = &res;
  (*fptr)((void **)&data);

  // Intentional printing of the output so we can test.
  llvm::outs() << res;

  return Error::success();
}

// Entry point for all CPU runners. Expects the common argc/argv arguments for
// standard C++ main functions and an mlirTransformer.
// The latter is applied after parsing the input into MLIR IR and before passing
// the MLIR module to the ExecutionEngine.
int run(int argc, char **argv,
        llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);

  initializeLLVM();
  mlir::initializeLLVMPasses();

  llvm::SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      optO0, optO1, optO2, optO3};

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  llvm::SmallVector<const llvm::PassInfo *, 4> passes;
  llvm::Optional<unsigned> optLevel;
  unsigned optCLIPosition = 0;
  // Determine if there is an optimization flag present, and its CLI position
  // (optCLIPosition).
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optLevel = j;
      optCLIPosition = flag.getPosition();
      break;
    }
  }
  // Generate vector of pass information, plus the index at which we should
  // insert any optimization passes in that vector (optPosition).
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

  auto transformer =
      mlir::makeLLVMPassesTransformer(passes, optLevel, optPosition);
  auto error = mainFuncType.getValue() == "f32"
                   ? compileAndExecuteSingleFloatReturnFunction(
                         m.get(), mainFuncName.getValue(), transformer)
                   : compileAndExecuteFunctionWithMemRefs(
                         m.get(), mainFuncName.getValue(), transformer);
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
