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

#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"

#include <functional>
#include <memory>
#include <string>

#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"

namespace tensorflow {
DataDumperLoggerConfig::DataDumperLoggerConfig(
    std::function<std::string(const std::string &, mlir::Operation *op)>
        get_filename,
    const std::string &pass_prefix, bool print_module_scope,
    bool print_after_only_on_change, mlir::OpPrintingFlags op_printing_flags)
    : ::tensorflow::BridgeLoggerConfig(
          print_module_scope, print_after_only_on_change, op_printing_flags),
      get_filename_(get_filename),
      pass_prefix_(pass_prefix) {}

void DataDumperLoggerConfig::printBeforeIfEnabled(
    mlir::Pass *pass, mlir::Operation *op, PrintCallbackFn print_callback) {
  std::string pass_name = pass->getName().str();
  std::string filename =
      get_filename_(pass_prefix_ + "before_" + pass_name, op);

  if (ShouldPrint(pass, op)) DumpMlir(filename, print_callback);
}

void DataDumperLoggerConfig::printAfterIfEnabled(
    mlir::Pass *pass, mlir::Operation *op, PrintCallbackFn print_callback) {
  std::string pass_name = pass->getName().str();
  std::string filename = get_filename_(pass_prefix_ + "after_" + pass_name, op);

  if (ShouldPrint(pass, op)) DumpMlir(filename, print_callback);
}

void DataDumperLoggerConfig::DumpMlir(
    const std::string &filename,
    BridgeLoggerConfig::PrintCallbackFn print_callback) {
  std::unique_ptr<llvm::raw_ostream> os;
  std::string filepath;
  if (tensorflow::CreateFileForDumping(filename, &os, &filepath).ok()) {
    print_callback(*os);
    LOG(INFO) << "Dumped MLIR module to " << filepath;
  }
}
}  // namespace tensorflow
