/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_compiler.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/xla_gpu_opt.h"
#include "tensorflow/compiler/xla/status.h"
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
static llvm::cl::opt<bool> verify_errors(
    "verify-errors",
    llvm::cl::desc("Whether we expect errors which should be verified"),
    llvm::cl::init(false));

static llvm::cl::opt<xla::mlir_gpu::MlirCompiler::IRHook::LoweringStage>
    // NOLINTNEXTLINE
    lowering_stage(
        "lowering-stage",
        llvm::cl::desc(
            "The lowering stage up to which the compiler will be run"),
        llvm::cl::values(
            clEnumValN(xla::mlir_gpu::MlirCompiler::IRHook::LoweringStage::LHLO,
                       "LHLO", "LHLO"),
            clEnumValN(xla::mlir_gpu::MlirCompiler::IRHook::LoweringStage::GPU,
                       "GPU", "GPU"),
            clEnumValN(xla::mlir_gpu::MlirCompiler::IRHook::LoweringStage::LLVM,
                       "LLVM", "LLVM"),
            clEnumValN(
                xla::mlir_gpu::MlirCompiler::IRHook::LoweringStage::KERNEL,
                "KERNEL", "Kernel")),
        llvm::cl::init(
            xla::mlir_gpu::MlirCompiler::IRHook::LoweringStage::LHLO));

int main(int argc, char **argv) {
  tensorflow::InitMlir y(&argc, &argv);
  mlir::registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "XLA GPU modular optimizer driver\n");

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  QCHECK(file) << error_message;

  auto output = mlir::openOutputFile(output_filename, &error_message);
  QCHECK(output) << error_message;

  xla::mlir_gpu::XlaGpuOpt opt;
  xla::Status status =
      verify_errors ? opt.CompileAndExpectErrors(file->getBuffer().str(),
                                                 output->os(), lowering_stage)
                    : opt.CompileAndOutputIr(file->getBuffer().str(),
                                             output->os(), lowering_stage);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
    return 1;
  }
  output->keep();
  return 0;
}
