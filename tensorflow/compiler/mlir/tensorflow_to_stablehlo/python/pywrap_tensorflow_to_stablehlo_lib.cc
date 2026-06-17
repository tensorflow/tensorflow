/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow_to_stablehlo/python/pywrap_tensorflow_to_stablehlo_lib.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow_to_stablehlo/tf_to_stablehlo.h"
#include "tensorflow/core/platform/path.h"

namespace mlir::tensorflow_to_stablehlo::pywrap {

absl::StatusOr<std::string> ModuleToBytecode(ModuleOp module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  if (mlir::failed(mlir::writeBytecodeToFile(module, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
}

absl::StatusOr<std::string> ExportModule(ModuleOp module) {
  const std::string output_filename = tensorflow::io::GetTempFilename(".mlir");
  std::string error_msg;
  auto output = openOutputFile(output_filename, &error_msg);
  if (output == nullptr) {
    return absl::UnknownError(
        absl::StrCat("Unable to open output path: ", error_msg));
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  OpPrintingFlags printing_flags;
  module.print(os, printing_flags);

  output->os() << result;
  output->keep();

  return output_filename;
}

absl::StatusOr<std::string> PywrapSavedModelToStablehlo(
    absl::string_view input_path,
    const std::vector<std::string>& exported_model_signatures,
    const std::vector<std::string>& tag_names,
    absl::string_view input_arg_shapes_str) {
  mlir::DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module =
      TfToStablehlo(input_path, &context, exported_model_signatures, tag_names,
                    input_arg_shapes_str, /*is_input_mlir_module=*/false);

  if (!module.ok()) {
    return absl::UnknownError(
        absl::StrCat("Failed to convert SavedModel to StableHLO: ",
                     module.status().message()));
  }

  auto bytecode = ModuleToBytecode(module.value().get());
  if (!bytecode.ok()) {
    return absl::UnknownError(
        absl::StrCat("Failed to serialize MLIR module to bytecode: ",
                     bytecode.status().message()));
  }

  return bytecode.value();
}

absl::StatusOr<std::string> PywrapTfModuleToStablehlo(
    absl::string_view module_op_str, absl::string_view input_arg_shapes_str) {
  mlir::DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto tf_module = mlir::parseSourceString<ModuleOp>(module_op_str, &context);
  if (!tf_module) {
    return absl::UnknownError("Failed to parse MLIR module");
  }

  auto mlir_file_path = ExportModule(*tf_module);
  if (!mlir_file_path.ok()) {
    return absl::UnknownError(
        absl::StrCat("Failed to write MLIR module to file.",
                     mlir_file_path.status().message()));
  }

  auto module = TfToStablehlo(*mlir_file_path, &context,
                              /*exported_model_signatures=*/{},
                              /*tag_names=*/{}, input_arg_shapes_str,
                              /*is_input_mlir_module=*/true);

  if (!module.ok()) {
    return absl::UnknownError(
        absl::StrCat(" Failed to convert SavedModel to StableHLO: ",
                     module.status().message()));
  }

  auto bytecode = ModuleToBytecode(module.value().get());
  if (!bytecode.ok()) {
    return absl::UnknownError(
        absl::StrCat("Failed to serialize MLIR module to bytecode: ",
                     bytecode.status().message()));
  }

  return bytecode.value();
}

}  // namespace mlir::tensorflow_to_stablehlo::pywrap
