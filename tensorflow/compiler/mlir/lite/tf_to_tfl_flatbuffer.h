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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

// Load a TF model from a GraphDef definition or a TF control flow dialect MLIR
// source into a MLIR module. If `input_mlir` is true, load from a MLIR source
// file; otherwise, load from a GraphDef.
// Setting prune_unused_nodes to true, would prune unreachable nodes if
// output_arrays is specified.
tsl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    bool use_splatted_constant, const std::vector<std::string>& extra_tf_opdefs,
    const GraphImportConfig& specs, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view control_output_arrays, llvm::SourceMgr* source_mgr,
    mlir::MLIRContext* context);

// Load Saved model (either v1 or v2) into MLIR.
// 'saved_model_bundle' will be initialized if V1 model was loaded.
tsl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    const std::string& input_filename, int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<const std::string> extra_tf_opdefs,
    absl::Span<std::string> exported_names, const GraphImportConfig& specs,
    bool enable_variable_lifting, mlir::MLIRContext* context,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle);

Status ConvertTFExecutorToStablehloFlatbuffer(
    mlir::PassManager& pass_manager, mlir::ModuleOp module, bool export_to_mlir,
    mlir::StatusScopedDiagnosticHandler& statusHandler,
    const toco::TocoFlags& toco_flags, const mlir::TFL::PassConfig& pass_config,
    std::optional<tensorflow::Session*> session, std::string* result);

// Taking a MLIR module in TF executor dialect and a set of parameters,
// applies a set of passes (configured accordingly to the provided
// `pass_config`) to convert the module to TF Lite dialect and serializes the
// result to a string. Depending on an attribute in the module main function,
// full integer quantization is applied.
// * `quantizated_buffer_type` can be set to INT8 or FLOAT16 to trigger the
// corresponding weight quantization.
// * `export_to_mlir` enables exporting to MLIR text format, otherwise exported
// in flat buffer. If the
// * `session` pointer may provided, it will be used to freeze resource
// variables. If the `saved_model_dir` directory path is provided, then the
// `tf_saved_model.asset` ops will be freezed.
Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, toco::TocoFlags& toco_flags,
    const mlir::TFL::PassConfig& pass_config,
    const std::unordered_set<std::string>& saved_model_tags,
    llvm::StringRef saved_model_dir, SavedModelBundle* saved_model_bundle,
    std::string* result, bool serialize_stablehlo_ops = false,
    const quantization::PyFunctionLibrary* quantization_py_function_lib =
        nullptr);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_
