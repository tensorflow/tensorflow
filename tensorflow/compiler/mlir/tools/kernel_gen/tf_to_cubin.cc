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

//===- tf_to_cubin.cc -------------------------------------------*- C++ -*-===//
//
// This file implements the entry point to compile a tf op to a cubin file.
//
//===----------------------------------------------------------------------===//
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/Support/CommandLine.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/cubin_creator.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

int main(int argc, char** argv) {
  llvm::cl::opt<std::string> input_file("input", llvm::cl::desc("input file"),
                                        llvm::cl::value_desc("filename"),
                                        llvm::cl::init("foo.mlir"));
  llvm::cl::opt<std::string> output_file(
      "output", llvm::cl::desc("output file"), llvm::cl::value_desc("filename"),
      llvm::cl::init("foo.bin"));
  llvm::cl::opt<int32_t> architecture(
      "arch", llvm::cl::desc("target architecture (e.g. 50 for sm_50)"),
      llvm::cl::init(50));
  llvm::cl::list<uint32_t> tile_sizes(
      "tile_sizes", llvm::cl::desc("tile sizes to use"), llvm::cl::ZeroOrMore,
      llvm::cl::CommaSeparated);
  llvm::cl::list<uint32_t> unroll_factors(
      "unroll_factors",
      llvm::cl::desc("factors to unroll by, separated by commas"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);
  llvm::cl::list<uint32_t> same_shape(
      "same_shape",
      llvm::cl::desc("arguments with same shape, separated by commas"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

  tensorflow::InitMlir y(&argc, &argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TF op GPU kernel generator\n");

  std::pair<int32_t, int32_t> compute_capability(architecture / 10,
                                                 architecture % 10);

  std::string tf_code;
  auto read_status = tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  input_file, &tf_code);
  if (!read_status.ok()) {
    LOG(ERROR) << read_status;
    return 1;
  }

  auto cubin = tensorflow::kernel_gen::GenerateCubinForTfCode(
      tf_code, compute_capability, tile_sizes, same_shape, unroll_factors);

  if (!cubin.ok()) {
    LOG(ERROR) << cubin.status();
    return 1;
  }

  std::vector<uint8_t> cubin_data = cubin.ConsumeValueOrDie();

  auto status = tensorflow::WriteStringToFile(
      tensorflow::Env::Default(), output_file,
      absl::string_view{reinterpret_cast<char*>(cubin_data.data()),
                        cubin_data.size()});

  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }

  return 0;
}
