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

#include "tensorflow/compiler/mlir/utils/saved_model_converter_utils.h"

#include <stdlib.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "absl/log/log.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/mlir_roundtrip_flags.h"


namespace tensorflow {
namespace utils {

// Util that registers 'extra_tf_opdefs' to the TF global registry.
// Return OK on success, failure if registering failed.
absl::Status RegisterExtraTfOpDefs(
    absl::Span<const std::string> extra_tf_opdefs) {
  for (const auto& tf_opdefs_string : extra_tf_opdefs) {
    OpDef opdef;
    // NOLINTNEXTLINE: Use tsl::protobuf to be compatible with OSS.
    if (!tsl::protobuf::TextFormat::ParseFromString(tf_opdefs_string, &opdef)) {
      LOG(ERROR) << "OpDef parsing failed for: " << tf_opdefs_string;
      return absl::InvalidArgumentError("fail to parse extra OpDef");
    }
    // Register extra opdefs.
    // TODO: b/133770952 - Support shape functions.
    OpRegistry::Global()->Register(
        [opdef](OpRegistrationData* op_reg_data) -> absl::Status {
          *op_reg_data = OpRegistrationData(opdef);
          return absl::OkStatus();
        });
  }
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    const std::string& input_filename, const int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<const std::string> extra_tf_opdefs,
    absl::Span<std::string> exported_names, const GraphImportConfig& specs,
    bool enable_variable_lifting, mlir::MLIRContext* context,
    std::unique_ptr<SavedModelBundle>* saved_model_bundle) {
  // Register extra TF ops passed as OpDef.
  auto extra_opdefs_status = RegisterExtraTfOpDefs(extra_tf_opdefs);
  if (!extra_opdefs_status.ok()) return extra_opdefs_status;

  if (saved_model_version == 2) {
    auto module_or = SavedModelObjectGraphToMlirImport(
        input_filename, tags, exported_names, context,
        /*unconditionally_use_set_output_shapes=*/true);
    if (!module_or.status().ok()) return module_or.status();
    return std::move(module_or).value();
  } else if (saved_model_version == 1) {
    MLIRImportOptions options;
    options.upgrade_legacy = specs.upgrade_legacy;
    options.unconditionally_use_set_output_shapes = true;
    options.lift_variables = enable_variable_lifting;
    auto module_or = SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, context, options,
        saved_model_bundle);

    if (!module_or.status().ok()) return module_or.status();
    return std::move(module_or).value();
  } else {
    return absl::InvalidArgumentError("Should be either saved model v1 or v2.");
  }
}

}  // namespace utils
}  // namespace tensorflow
