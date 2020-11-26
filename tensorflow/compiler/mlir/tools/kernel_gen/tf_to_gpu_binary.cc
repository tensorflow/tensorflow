// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- tf_to_gpu_binary.cc --------------------------------------*- C++ -*-===//
//
// This file implements the entry point to compile a tf op to a gpu binary
//
//===----------------------------------------------------------------------===//
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/crash_handler.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace kernel_gen {
namespace {

xla::Status Run(llvm::StringRef input_file, llvm::StringRef output_file,
                std::string architecture, llvm::ArrayRef<uint32_t> tile_sizes,
                llvm::ArrayRef<uint32_t> unroll_factors) {
  // Read TF code.
  std::string tf_code;
  TF_RETURN_IF_ERROR(
      ReadFileToString(Env::Default(), input_file.str(), &tf_code));
  // Compile.
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(
      mlir::OwningModuleRef module,
      GenerateKernelForTfCode(context, tf_code, /*gpu_binary_only=*/true,
                              architecture, tile_sizes, unroll_factors,
                              /*embed_memref_prints=*/false,
                              /*generate_fatbin=*/false));
  // Extract gpu_binary.
  TF_ASSIGN_OR_RETURN(std::string gpu_binary, ExtractGpuBinary(*module));

  // Write gpu_binary blob.
  TF_RETURN_IF_ERROR(
      WriteStringToFile(Env::Default(), output_file.str(), gpu_binary));
  return xla::Status::OK();
}

}  // namespace
}  // namespace kernel_gen
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::kernel_gen::SetCrashReportMessage();
  llvm::cl::opt<std::string> input_file("input", llvm::cl::desc("input file"),
                                        llvm::cl::value_desc("filename"),
                                        llvm::cl::init("foo.mlir"));
  llvm::cl::opt<std::string> output_file(
      "output", llvm::cl::desc("output file"), llvm::cl::value_desc("filename"),
      llvm::cl::init("foo.bin"));
  llvm::cl::opt<std::string> architecture(
      "arch", llvm::cl::desc("target architecture (e.g. sm_50)"),
      llvm::cl::init("sm_50"));
  llvm::cl::list<uint32_t> tile_sizes(
      "tile_sizes", llvm::cl::desc("tile sizes to use"), llvm::cl::ZeroOrMore,
      llvm::cl::CommaSeparated);
  llvm::cl::list<uint32_t> unroll_factors(
      "unroll_factors",
      llvm::cl::desc("factors to unroll by, separated by commas"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

  tensorflow::InitMlir y(&argc, &argv);
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "TF op GPU kernel generator\n");

  auto status = tensorflow::kernel_gen::Run(
      input_file, output_file, architecture, tile_sizes, unroll_factors);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }
  return 0;
}
