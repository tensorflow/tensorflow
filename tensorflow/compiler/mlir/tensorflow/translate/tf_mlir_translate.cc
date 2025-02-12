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

#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelObjectGraphToMlirImport(absl::string_view saved_model_dir,
                                  const std::unordered_set<std::string>& tags,
                                  absl::Span<std::string> exported_names,
                                  mlir::MLIRContext* context,
                                  bool unconditionally_use_set_output_shapes,
                                  bool import_variables_as_dense_resources) {
  tensorflow::SavedModelV2Bundle bundle;
  auto load_status = tensorflow::SavedModelV2Bundle::Load(
      std::string(saved_model_dir.data(), saved_model_dir.length()), &bundle);
  if (!load_status.ok()) {
    LOG(ERROR) << "Failed to load saved model '" << saved_model_dir
               << "': " << load_status;
    return load_status;
  }

  MLIRImportOptions options;
  options.add_default_attributes = true;
  options.unconditionally_use_set_output_shapes =
      unconditionally_use_set_output_shapes;
  options.import_variables_as_dense_resources =
      import_variables_as_dense_resources;

  auto module_or =
      ConvertSavedModelToMlir(&bundle, context, exported_names, options);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "SavedModel import failed: " << module_or.status();
  }
  return module_or;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle) {
  // Create local bundle if no one is provided to use.
  std::unique_ptr<tensorflow::SavedModelBundle> bundle;
  if (saved_model_bundle == nullptr) {
    bundle = std::make_unique<tensorflow::SavedModelBundle>();
  } else if (*saved_model_bundle == nullptr) {
    *saved_model_bundle = std::make_unique<tensorflow::SavedModelBundle>();
  }
  SavedModelBundle* bundle_ptr =
      saved_model_bundle ? saved_model_bundle->get() : bundle.get();
  tensorflow::SessionOptions session_options;

  // Force saved model states to be restored to CPU.
  (*session_options.config.mutable_device_count())["GPU"] = 0;
  auto load_status = tensorflow::LoadSavedModel(
      session_options, /* run_options = */ {}, std::string(saved_model_dir),
      tags, bundle_ptr);
  if (!load_status.ok()) {
    LOG(ERROR) << "Failed to load saved model v1 '" << saved_model_dir
               << "': " << load_status;
    return load_status;
  }

  auto module_or =
      ConvertSavedModelV1ToMlir(*bundle_ptr, exported_names, context, options);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "SavedModel V1 import failed: " << module_or.status();
  }
  return module_or;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImportLite(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options) {
  MetaGraphDef meta_graph_def;
  auto status =
      ReadMetaGraphDefFromSavedModel(saved_model_dir, tags, &meta_graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to load saved model v1 '" << saved_model_dir
               << "': " << status;
    return status;
  }

  std::optional<absl::Span<const std::string>> optional_exported_names;
  if (!exported_names.empty()) optional_exported_names = exported_names;

  // TODO(b/186898924): debug info in the savedmodel should not be ignored and
  // should be passed here.
  auto module_or =
      ConvertSavedModelV1ToMlirLite(meta_graph_def, /*debug_info=*/{},
                                    optional_exported_names, context, options);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "SavedModel V1 import failed: " << module_or.status();
  }
  return module_or;
}

}  // namespace tensorflow
