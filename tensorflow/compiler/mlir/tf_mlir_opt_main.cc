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
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "mlir/Support/MlirOptMain.h"  // TF:local_config_mlir
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

static std::vector<const mlir::PassRegistryEntry *> *pass_list;

int main(int argc, char **argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::list<const mlir::PassRegistryEntry *, bool, mlir::PassNameParser>
      pass_list("", llvm::cl::desc("Compiler passes to run"));
  ::pass_list = &pass_list;
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TF MLIR modular optimizer driver\n");

  // TODO(jpienaar): Enable command line parsing for both sides.
  int fake_argc = 1;
  tensorflow::port::InitMain(argv[0], &fake_argc, &argv);

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  QCHECK(file) << error_message;

  auto output = mlir::openOutputFile(output_filename, &error_message);
  QCHECK(output) << error_message;

  return failed(mlir::MlirOptMain(output->os(), std::move(file), pass_list,
                                  split_input_file, verify_diagnostics,
                                  verify_passes));
}
