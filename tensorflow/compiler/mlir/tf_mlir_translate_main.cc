/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "mlir/Support/TranslateClParser.h"  // TF:local_config_mlir
#include "tensorflow/core/platform/init_main.h"

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> output_filename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

int main(int argc, char** argv) {
  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::InitLLVM y(argc, argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const mlir::TranslateFunction*, false, mlir::TranslationParser>
      requested_translation("", llvm::cl::desc("Translation to perform"),
                            llvm::cl::Required);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TF MLIR translation driver\n");

  // TODO(jpienaar): Enable command line parsing for both sides.
  int fake_argc = 1;
  tensorflow::port::InitMain(argv[0], &fake_argc, &argv);

  mlir::MLIRContext context;
  return failed(
      (*requested_translation)(input_filename, output_filename, &context));
}
