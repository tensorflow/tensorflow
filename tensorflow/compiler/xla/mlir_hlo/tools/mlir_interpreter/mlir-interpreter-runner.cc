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

#include <memory>
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "lhlo/IR/lhlo_ops.h"
#include "lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilities.h"
#include "thlo/IR/thlo_ops.h"
#include "tools/mlir_interpreter/framework/interpreter.h"

struct Options {
  llvm::cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-")};
  llvm::cl::opt<bool> runAllFunctions{
      "run-all", llvm::cl::desc("Run all functions in the module"),
      llvm::cl::init(false)};
};

static mlir::OwningOpRef<mlir::Operation *> parseMlirInput(
    llvm::StringRef inputFilename, mlir::MLIRContext *context) {
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return {};
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), mlir::SMLoc());
  return mlir::parseSourceFileForTool(sourceMgr, context,
                                      /*insertImplicitModule=*/true);
}

static mlir::LogicalResult run(mlir::ModuleOp module,
                               mlir::func::FuncOp function) {
  llvm::outs() << "@" << function.getName().str() << "()\n";
  if (function.getBody().getBlocks().front().getNumArguments() > 0) {
    llvm::errs() << "Function arguments are not supported.";
    return mlir::failure();
  }

  mlir::SymbolTable symbolTable{module};
  auto results =
      mlir::interpreter::runInterpreter(symbolTable, function, {}, {});
  if (!mlir::succeeded(results)) {
    llvm::errs() << "Interpreter failed\n";
    return mlir::failure();
  }

  if (!results->empty()) {
    llvm::outs() << "Results:\n";
    for (const auto &result : *results) {
      llvm::outs() << result.toString() << "\n";
    }
  }

  return mlir::success();
}

int main(int argc, char *argv[]) {
  // Flush llvm::outs before writing errors.
  llvm::errs().tie(&llvm::outs());

  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::mhlo::registerAllMhloPasses();
  mlir::registerAllPasses();
  registry.insert<mlir::lmhlo::LmhloDialect, mlir::lmhlo_gpu::LmhloGpuDialect,
                  mlir::gml_st::GmlStDialect, mlir::thlo::THLODialect>();

  mlir::MLIRContext context(registry);
  auto parsedInput = parseMlirInput(options.inputFilename, &context);
  if (!parsedInput) {
    llvm::errs() << "Failed to parse module.\n";
    return 1;
  }
  auto module = llvm::dyn_cast<mlir::ModuleOp>(**parsedInput);
  if (!module) {
    llvm::errs() << "Parsing returned something that's not a module.\n";
    return 1;
  }

  if (options.runAllFunctions) {
    bool allSucceeded = true;
    module.walk([&](mlir::func::FuncOp function) {
      if (!function.isPrivate()) {
        allSucceeded &= run(module, function).succeeded();
      }
    });
    if (!allSucceeded) {
      return 1;
    }
  } else {
    auto *main = module.lookupSymbol("main");
    if (!main) {
      llvm::errs() << "no main function found.\n";
      return 1;
    }
    if (!run(module, llvm::cast<mlir::func::FuncOp>(main)).succeeded()) {
      return 1;
    }
  }
  return 0;
}
