/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/translate/portable_api.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "tsl/platform/statusor.h"

namespace xla {

std::string PrintModule(mlir::ModuleOp module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo();
  module->print(os, flags);
  return s;
}

void LoadHloDialects(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  context.appendDialectRegistry(registry);
}

absl::StatusOr<std::string> SerializeUsingBytecode(mlir::ModuleOp module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  if (mlir::failed(mlir::writeBytecodeToFile(module, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
}

absl::StatusOr<std::string> ConvertHloToStablehlo(
    xla::HloModule const& hlo_module, bool emit_bytecode) {
  mlir::MLIRContext context;
  LoadHloDialects(context);
  TF_ASSIGN_OR_RETURN(auto module, ConvertHloToStablehlo(context, &hlo_module));
  if (emit_bytecode) return SerializeUsingBytecode(*module);
  return PrintModule(*module);
}

}  // namespace xla
