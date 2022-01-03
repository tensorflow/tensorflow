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

#ifndef TENSORFLOW_TOOLS_TFG_GRAPH_TRANSFORMS_IMPORT_H_
#define TENSORFLOW_TOOLS_TFG_GRAPH_TRANSFORMS_IMPORT_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

// Convert a saved model from `saved_model_file` to a MLIR module expressed in
// TFG dialect. Only the GraphDef portion of the saved model gets imported into
// MLIR TFG dialect.
tensorflow::StatusOr<mlir::OwningModuleRef> ImportSavedModel(
    mlir::MLIRContext* context, const std::string& saved_model_file);

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_TOOLS_TFG_GRAPH_TRANSFORMS_IMPORT_H_
