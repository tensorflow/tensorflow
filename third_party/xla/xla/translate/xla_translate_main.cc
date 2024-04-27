/* Copyright 2019 The OpenXLA Authors.

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
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/ToolUtilities.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tsl/platform/init_main.h"

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> output_filename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each chunk "
                   "independently"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  int dummyArgc = 1;
  tsl::port::InitMain(argv[0], &dummyArgc, &argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const mlir::Translation*, false, mlir::TranslationParser>
      requested_translation("", llvm::cl::desc("Translation to perform"));
  mlir::registerAsmPrinterCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "XLA translation driver\n");

  std::string error_message;
  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!output) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  auto input = mlir::openInputFile(input_filename, &error_message);
  if (!input) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  // Processes the memory buffer with a new MLIRContext.
  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           llvm::raw_ostream& os) {
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
    mlir::MLIRContext context;

    if (!verifyDiagnostics) {
      mlir::SourceMgrDiagnosticHandler diagnostic_handler(*sourceMgr, &context);
      return (*requested_translation)(sourceMgr, os, &context);
    }

    context.printOpOnDiagnostic(false);
    mlir::SourceMgrDiagnosticVerifierHandler diagnostic_handler(*sourceMgr,
                                                                &context);
    (void)(*requested_translation)(sourceMgr, os, &context);
    return diagnostic_handler.verify();
  };

  if (splitInputFile) {
    if (failed(mlir::splitAndProcessBuffer(std::move(input), processBuffer,
                                           output->os())))
      return 1;
  } else {
    if (failed(processBuffer(std::move(input), output->os()))) return 1;
  }

  output->keep();
  return 0;
}
