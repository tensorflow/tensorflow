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

#ifndef TENSORFLOW_COMPILER_MLIR_UTILS_SAVED_MODEL_CONVERTER_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_UTILS_SAVED_MODEL_CONVERTER_UTILS_H_

#include <memory>
#include <string>
#include <unordered_set>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/mlir_roundtrip_flags.h"

namespace tensorflow {
namespace utils {

// 'saved_model_bundle' will be initialized if V1 model was loaded.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    const std::string& input_filename, int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<const std::string> extra_tf_opdefs,
    absl::Span<std::string> exported_names, const GraphImportConfig& specs,
    bool enable_variable_lifting, mlir::MLIRContext* context,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle);

}  // namespace utils
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_UTILS_SAVED_MODEL_CONVERTER_UTILS_H_
