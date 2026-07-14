/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_SAVEDMODEL_EXPORT_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_SAVEDMODEL_EXPORT_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace mlir {
namespace tfg {

// Given an MLIR module, returns a `output_saved_model` SavedModel.
// The module must contain at most a single Graph operation and zero or more
// TFFunc operations. `original_saved_model` is used as only a GraphDef portion
// of a saved model represented in the MLIR module.
absl::Status ExportMlirToSavedModel(
    mlir::ModuleOp module, const tensorflow::SavedModel &original_saved_model,
    tensorflow::SavedModel *output_saved_model);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_SAVEDMODEL_EXPORT_H_
