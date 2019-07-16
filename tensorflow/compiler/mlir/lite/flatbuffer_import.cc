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

#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"

#include <iostream>
#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "mlir/Translation.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

using mlir::Location;
using mlir::MLIRContext;
using mlir::OwningModuleRef;

namespace tflite {
OwningModuleRef FlatBufferToMlir(absl::string_view buffer, MLIRContext* context,
                                 Location base_loc) {
  auto model_ptr =
      FlatBufferModel::VerifyAndBuildFromBuffer(buffer.data(), buffer.length());
  if (nullptr == model_ptr) {
    return emitError(base_loc, "Couldn't parse flatbuffer"), nullptr;
  }

  std::unique_ptr<ModelT> model(model_ptr->GetModel()->UnPack());
  std::cout << "Model version: " << model->version << std::endl;

  for (auto& subgraph : model->subgraphs) {
    std::cout << "Subgraph name: " << subgraph->name << std::endl;
    for (auto& input : subgraph->inputs) {
      std::cout << "  Subgraph input: " << input << std::endl;
    }
    for (auto& output : subgraph->outputs) {
      std::cout << "  Subgraph output: " << output << std::endl;
    }
  }

  mlir::Builder builder(context);
  return OwningModuleRef(mlir::ModuleOp::create(base_loc));
}

}  // namespace tflite

static OwningModuleRef FlatBufferFileToMlirTrans(llvm::StringRef filename,
                                                 MLIRContext* context) {
  std::string error;
  auto loc = mlir::FileLineColLoc::get(filename, 0, 0, context);
  auto buffer = mlir::openInputFile(filename, &error);
  if (nullptr == buffer) {
    return emitError(loc, error), nullptr;
  }

  return tflite::FlatBufferToMlir(
      absl::string_view(buffer->getBufferStart(), buffer->getBufferSize()),
      context, loc);
}

static mlir::TranslateToMLIRRegistration FlatBufferFileToMlirTransReg(
    "tflite-flatbuffer-to-mlir", FlatBufferFileToMlirTrans);
