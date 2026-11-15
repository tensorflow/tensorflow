/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/python/mlir_bytecode_to_tflite.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/converter_v2_options.pb.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/python/slim_model_importer.h"
#include "tensorflow/compiler/mlir/lite/python/stablehlo_tfl_pipeline.h"

namespace tensorflow::internal {

absl::Status ConvertMlirBytecodeToTFLite(
    const tflite::ConverterV2Options& options, absl::string_view model_dir,
    llvm::raw_pwrite_stream& export_stream) {
  mlir::DialectRegistry registry;
  registry
      .insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
              mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect,
              mlir::TFL::TensorFlowLiteDialect, mlir::quant::QuantDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();

  auto module_or = LoadSlimModel(model_dir, context.get());
  if (!module_or.ok()) return module_or.status();
  auto module = std::move(module_or).value();

  return mlir::TFL::ConvertStableHloToTFLite(*module, options, export_stream);
}

absl::Status ConvertMlirBytecodeToTFLite(
    const tflite::ConverterV2Options& options, absl::string_view model_dir,
    absl::string_view output_file_path) {
  std::string error_msg;
  auto output = mlir::openOutputFile(output_file_path, &error_msg);
  if (!output) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open output file: ", error_msg));
  }
  auto status = ConvertMlirBytecodeToTFLite(options, model_dir, output->os());
  if (!status.ok()) return status;
  output->os().close();
  if (output->os().has_error()) {
    return absl::InternalError("Error writing to output file.");
  }
  output->keep();
  return absl::OkStatus();
}

}  // namespace tensorflow::internal
