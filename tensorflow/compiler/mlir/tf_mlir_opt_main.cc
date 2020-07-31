/* Copyright 2019 Google Inc. All Rights Reserved.

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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/MlirOptMain.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> output_filename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> split_input_file(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> verify_diagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> verify_passes(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

int main(int argc, char **argv) {
  tensorflow::InitMlir y(&argc, &argv);

  // Register various MLIR command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  mlir::PassPipelineCLParser pass_pipeline("", "Compiler passes to run");

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TF MLIR modular optimizer driver\n");

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  QCHECK(file) << error_message;

  auto output = mlir::openOutputFile(output_filename, &error_message);
  QCHECK(output) << error_message;

  if (failed(mlir::MlirOptMain(output->os(), std::move(file), pass_pipeline,
                               split_input_file, verify_diagnostics,
                               verify_passes, allowUnregisteredDialects)))
    return 1;
  output->keep();
  return 0;
}
