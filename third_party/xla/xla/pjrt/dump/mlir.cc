/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/pjrt/dump/mlir.h"

#include <string>

#include "absl/status/status.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "xla/tsl/platform/env.h"

namespace pjrt {

absl::Status MlirModuleToFile(mlir::ModuleOp module, std::string file_path) {
  std::string module_string = MlirModuleToString(module);
  return tsl::WriteStringToFile(tsl::Env::Default(), file_path, module_string);
}

std::string MlirModuleToString(mlir::ModuleOp module) {
  std::string module_string;
  llvm::raw_string_ostream os(module_string);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(os, flags);
  return module_string;
}

}  // namespace pjrt
