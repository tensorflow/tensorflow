/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"

namespace tensorflow {

std::string ImportGraphDef(const std::string &proto,
                           const std::string &pass_pipeline,
                           TF_Status *status) {
  GraphDef graphdef;
  auto s = tensorflow::LoadProtoFromBuffer(proto, &graphdef);
  if (!s.ok()) {
    Set_TF_Status_from_Status(status, s);
    return "// error";
  }
  GraphDebugInfo debug_info;
  GraphImportConfig specs;
  mlir::MLIRContext context;
  auto module = ConvertGraphdefToMlir(graphdef, debug_info, specs, &context);
  if (!module.ok()) {
    Set_TF_Status_from_Status(status, module.status());
    return "// error";
  }

  // Run the pass_pipeline on the module if not empty.
  if (!pass_pipeline.empty()) {
    mlir::PassManager pm(&context);
    std::string error;
    llvm::raw_string_ostream error_stream(error);
    if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   ("Invalid pass_pipeline: " + error_stream.str()).c_str());
      return "// error";
    }

    mlir::StatusScopedDiagnosticHandler statusHandler(&context);
    if (failed(pm.run(*module.ValueOrDie()))) {
      Set_TF_Status_from_Status(status, statusHandler.ConsumeStatus());
      return "// error";
    }
  }
  return MlirModuleToString(*module.ConsumeValueOrDie());
}

std::string ExperimentalConvertSavedModelToMlir(
    const std::string &saved_model_path, const std::string &exported_names_str,
    bool show_debug_info, TF_Status *status) {
  // Load the saved model into a SavedModelV2Bundle.

  tensorflow::SavedModelV2Bundle bundle;
  auto load_status =
      tensorflow::SavedModelV2Bundle::Load(saved_model_path, &bundle);
  if (!load_status.ok()) {
    Set_TF_Status_from_Status(status, load_status);
    return "// error";
  }

  // Convert the SavedModelV2Bundle to an MLIR module.

  std::vector<string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  mlir::MLIRContext context;
  auto module_or = ConvertSavedModelToMlir(
      &bundle, &context, absl::Span<std::string>(exported_names));
  if (!module_or.status().ok()) {
    Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  return MlirModuleToString(*module_or.ConsumeValueOrDie(), show_debug_info);
}

std::string ExperimentalConvertSavedModelV1ToMlir(
    const std::string &saved_model_path, const std::string &tags,
    bool show_debug_info, TF_Status *status) {
  // Load the saved model into a SavedModelBundle.

  std::unordered_set<string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());

  tensorflow::SavedModelBundle bundle;
  auto load_status =
      tensorflow::LoadSavedModel({}, {}, saved_model_path, tag_set, &bundle);
  if (!load_status.ok()) {
    Set_TF_Status_from_Status(status, load_status);
    return "// error";
  }

  // Convert the SavedModelBundle to an MLIR module.

  mlir::MLIRContext context;
  auto module_or = ConvertSavedModelV1ToMlir(bundle, {}, &context);
  if (!module_or.status().ok()) {
    Set_TF_Status_from_Status(status, module_or.status());
    return "// error";
  }

  return MlirModuleToString(*module_or.ConsumeValueOrDie(), show_debug_info);
}

std::string ExperimentalRunPassPipeline(const std::string &mlir_txt,
                                        const std::string &pass_pipeline,
                                        bool show_debug_info,
                                        TF_Status *status) {
  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  {
    mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
    module = mlir::parseSourceString(mlir_txt, &context);
    if (!module) {
      Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
      return "// error";
    }
  }

  // Run the pass_pipeline on the module.
  mlir::PassManager pm(&context);
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  if (failed(mlir::parsePassPipeline(pass_pipeline, pm, error_stream))) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Invalid pass_pipeline: " + error_stream.str()).c_str());
    return "// error";
  }

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module))) {
    Set_TF_Status_from_Status(status, diagnostic_handler.ConsumeStatus());
    return "// error";
  }
  return MlirModuleToString(*module, show_debug_info);
}

}  // namespace tensorflow
