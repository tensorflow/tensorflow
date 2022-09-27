/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/experimental/tac/utils/utils.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace TFL {
namespace tac {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportFlatbufferOrMlir(
    const std::string& input_filename, bool input_mlir,
    llvm::SourceMgr* source_mgr, mlir::MLIRContext* context) {
  std::string error;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      mlir::openInputFile(input_filename, &error);
  if (buffer == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot open input file: %s. %s", input_filename, error));
  }

  if (input_mlir) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::TFL::TensorFlowLiteDialect,
                    mlir::arith::ArithmeticDialect, mlir::func::FuncDialect>();
    context->appendDialectRegistry(registry);
    source_mgr->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    return mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, context));
  }

  mlir::Location loc =
      mlir::FileLineColLoc::get(context, input_filename, /*line=*/0,
                                /*column=*/0);
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  return tflite::FlatBufferToMlir(
      absl::string_view(buffer->getBufferStart(), buffer->getBufferSize()),
      context, loc, /*use_external_constant=*/false, inputs, outputs,
      /*experimental_prune_unreachable_nodes_unconditionally=*/true);
}

absl::Status ExportFlatbufferOrMlir(const std::string& output_filename,
                                    bool output_mlir, mlir::ModuleOp module) {
  std::string error_msg;
  auto output = mlir::openOutputFile(output_filename, &error_msg);
  if (output == nullptr) {
    llvm::errs() << error_msg << '\n';
    return absl::InvalidArgumentError("cannot open output file.");
  }

  std::string result;
  if (output_mlir) {
    llvm::raw_string_ostream os(result);
    module.print(os);
    os.flush();
  } else {
    tflite::FlatbufferExportOptions options;
    options.toco_flags.set_force_select_tf_ops(false);
    options.toco_flags.set_enable_select_tf_ops(false);
    options.toco_flags.set_allow_custom_ops(true);
    if (!tflite::MlirToFlatBufferTranslateFunction(module, options, &result)) {
      return absl::UnknownError("Failed to export tflite file.");
    }
  }

  output->os() << result;
  output->keep();
  return absl::OkStatus();
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
