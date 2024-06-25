/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#include "mhlo/IR/register.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/ParseUtilities.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

struct Options {
  llvm::cl::opt<std::string> input_filename{llvm::cl::Positional,
                                            llvm::cl::desc("<input file>"),
                                            llvm::cl::init("-")};
  llvm::cl::opt<bool> run_all_functions{
      "run-all", llvm::cl::desc("Run all functions in the module"),
      llvm::cl::init(false)};
};

static mlir::OwningOpRef<mlir::Operation *> ParseMlirInput(
    llvm::StringRef input_filename, mlir::MLIRContext *context) {
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!file) {
    llvm::errs() << error_message << "\n";
    return {};
  }

  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(file), mlir::SMLoc());
  return mlir::parseSourceFileForTool(source_mgr, context,
                                      /*insertImplicitModule=*/true);
}

static mlir::LogicalResult Run(mlir::ModuleOp module,
                               mlir::func::FuncOp function) {
  llvm::outs() << "@" << function.getName().str() << "()\n";
  if (function.getBody().getBlocks().front().getNumArguments() > 0) {
    llvm::errs() << "Function arguments are not supported.";
    return mlir::failure();
  }

  mlir::SymbolTable symbol_table{module};
  auto results =
      mlir::interpreter::RunInterpreter(symbol_table, function, {}, {});
  if (!results.ok()) {
    llvm::errs() << "Interpreter failed\n";
    return mlir::failure();
  }

  if (!results->empty()) {
    llvm::outs() << "Results:\n";
    for (const auto &result : *results) {
      llvm::outs() << result.ToString() << "\n";
    }
  }

  return mlir::success();
}

int main(int argc, char *argv[]) {
  // Flush llvm::outs before writing errors.
  llvm::errs().tie(&llvm::outs());

  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR interpreter driver\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::mhlo::registerAllMhloPasses();
  mlir::registerAllPasses();

  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();
  auto parsed_input = ParseMlirInput(options.input_filename, &context);
  if (!parsed_input) {
    llvm::errs() << "Failed to parse module.\n";
    return 1;
  }
  auto module = llvm::dyn_cast<mlir::ModuleOp>(**parsed_input);
  if (!module) {
    llvm::errs() << "Parsing returned something that's not a module.\n";
    return 1;
  }

  if (options.run_all_functions) {
    bool all_succeeded = true;
    module.walk([&](mlir::func::FuncOp function) {
      if (!function.isPrivate()) {
        all_succeeded &= Run(module, function).succeeded();
      }
    });
    if (!all_succeeded) {
      return 1;
    }
  } else {
    auto *main = module.lookupSymbol("main");
    if (!main) {
      llvm::errs() << "no main function found.\n";
      return 1;
    }
    if (!Run(module, llvm::cast<mlir::func::FuncOp>(main)).succeeded()) {
      return 1;
    }
  }
  return 0;
}
