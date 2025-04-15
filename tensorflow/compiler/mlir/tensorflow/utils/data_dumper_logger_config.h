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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DATA_DUMPER_LOGGER_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DATA_DUMPER_LOGGER_CONFIG_H_

#include <functional>
#include <string>

#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"

namespace tensorflow {

class DataDumperLoggerConfig : public ::tensorflow::BridgeLoggerConfig {
 public:
  explicit DataDumperLoggerConfig(
      std::function<std::string(const std::string &, mlir::Operation *op)>
          get_filename,
      const std::string &pass_prefix = "", bool print_module_scope = false,
      bool print_after_only_on_change = true,
      mlir::OpPrintingFlags op_printing_flags = mlir::OpPrintingFlags());

  void printBeforeIfEnabled(mlir::Pass *pass, mlir::Operation *op,
                            PrintCallbackFn print_callback) override;

  void printAfterIfEnabled(mlir::Pass *pass, mlir::Operation *op,
                           PrintCallbackFn print_callback) override;

 private:
  static void DumpMlir(const std::string &filename,
                       BridgeLoggerConfig::PrintCallbackFn print_callback);

  // The function to dump the target MLIR string to file.
  // The parameter that will be sent to the dump_func_ is:
  // The pass name (std::string)
  std::function<std::string(const std::string &, mlir::Operation *op)>
      get_filename_;

  // The pass prefix.
  std::string pass_prefix_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DATA_DUMPER_LOGGER_CONFIG_H_
