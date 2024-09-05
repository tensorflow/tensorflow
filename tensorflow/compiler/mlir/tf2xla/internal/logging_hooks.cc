/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/logging_hooks.h"

#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/core/util/debug_data_dumper.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

using mlir::PassManager;

// Setup the input pass manager to enable IR dumping after each pass.
// Note a side effect of this method is that multi threading will be disabled.
void EnablePassIRPrinting(PassManager& pm, const std::string& dump_group_name,
                          llvm::StringRef module_name) {
  // Print the whole module after each pass, which requires disabling
  // multi-threading as well.
  pm.getContext()->disableMultithreading();
  pm.enableIRPrinting(std::make_unique<::tensorflow::DataDumperLoggerConfig>(
      [module_name, dump_group_name](const std::string& pass_tag_name,
                                     mlir::Operation* op) {
        return DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), dump_group_name, pass_tag_name);
      },
      /*pass_prefix=*/"",
      /*print_module_scope=*/true));
  pm.enableTiming();
}

};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow
