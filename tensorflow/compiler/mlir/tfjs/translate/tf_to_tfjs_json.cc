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

#include "tensorflow/compiler/mlir/tfjs/translate/tf_to_tfjs_json.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tfjs/translate/json_translate.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningOpRef;
using stream_executor::port::StatusOr;

namespace {
tensorflow::Status RegisterCustomOps(
    const std::vector<std::string>& extra_tf_opdefs) {
  for (const auto& tf_opdefs_string : extra_tf_opdefs) {
    tensorflow::OpDef opdef;
    if (!tensorflow::protobuf::TextFormat::ParseFromString(tf_opdefs_string,
                                                           &opdef)) {
      LOG(ERROR) << "OpDef parsing failed for: " << tf_opdefs_string;
      return errors::InvalidArgument("fail to parse extra OpDef");
    }
    // Register extra opdefs.
    tensorflow::OpRegistry::Global()->Register(
        [opdef](tensorflow::OpRegistrationData* op_reg_data) -> Status {
          *op_reg_data = tensorflow::OpRegistrationData(opdef);
          return OkStatus();
        });
  }
  return OkStatus();
}
}  // namespace

StatusOr<OwningOpRef<ModuleOp>> LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    const std::vector<std::string>& extra_tf_opdefs,
    absl::string_view debug_info_file, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, bool prune_unused_nodes,
    llvm::SourceMgr* source_mgr, MLIRContext* context) {
  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!file) {
    llvm::errs() << error_message << "\n";
    return errors::InvalidArgument("fail to open input file");
  }

  if (input_mlir) {
    source_mgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return OwningOpRef<ModuleOp>(
        mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, context));
  }

  TF_RETURN_IF_ERROR(RegisterCustomOps(extra_tf_opdefs));

  return tensorflow::GraphdefToMlirTranslateFunction(
      file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
      input_shapes, output_arrays, /*control_output_arrays=*/"",
      prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
      /*graph_as_function=*/false, /*upgrade_legacy=*/true,
      /*enable_shape_inference=*/true,
      /*unconditionally_use_set_output_shapes=*/false, context);
}

Status ConvertTFOpsToTfjsJSON(mlir::ModuleOp module, bool export_to_mlir,
                              std::string* result,
                              mlir::PassManager* pass_manager) {
  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
                                                    /*propagate=*/true);
  if (failed(pass_manager->run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return OkStatus();
  }

  return tfjs::MlirToJSONTranslateFunction(module, result)
             ? OkStatus()
             : statusHandler.ConsumeStatus();
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    bool import_saved_model, bool import_saved_model_v1,
    const std::vector<std::string>& extra_tf_opdefs,
    const std::string& input_filename, const std::string& saved_model_tags,
    const std::string& saved_model_exported_names, mlir::MLIRContext* context) {
  std::unordered_set<std::string> tags = absl::StrSplit(saved_model_tags, ',');
  std::vector<std::string> exported_names_in_vector =
      absl::StrSplit(saved_model_exported_names, ',', absl::SkipEmpty());
  absl::Span<std::string> exported_names(exported_names_in_vector);
  if (import_saved_model) {
    auto module_or = tensorflow::SavedModelObjectGraphToMlirImport(
        input_filename, tags, absl::Span<std::string>(exported_names), context);
    if (!module_or.status().ok()) return module_or.status();
    TF_RETURN_IF_ERROR(RegisterCustomOps(extra_tf_opdefs));
    return std::move(module_or).value();
  } else if (import_saved_model_v1) {
    tensorflow::MLIRImportOptions import_options;
    auto module_or = tensorflow::SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, context, import_options);

    if (!module_or.status().ok()) return module_or.status();
    TF_RETURN_IF_ERROR(RegisterCustomOps(extra_tf_opdefs));
    return std::move(module_or).value();
  } else {
    return tensorflow::errors::InvalidArgument(
        "Should be either saved model v1 or v2");
  }
}

}  // namespace tensorflow
