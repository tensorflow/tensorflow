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

#include <unordered_set>

#include "absl/types/span.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// Load a TF model from a GraphDef definition or a TF control flow dialect MLIR
// source into a MLIR module. If `input_mlir` is true, load from a MLIR source
// file; otherwise, load from a GraphDef.
// Setting prune_unused_nodes to true, would prune unreachable nodes if
// output_arrays is specified.
stream_executor::port::StatusOr<mlir::OwningModuleRef>
LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    bool use_splatted_constant, const std::vector<std::string>& extra_tf_opdefs,
    const GraphImportConfig& specs, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view control_output_arrays, llvm::SourceMgr* source_mgr,
    mlir::MLIRContext* context);

// Load Saved model (either v1 or v2) into MLIR.
// 'saved_model_bundle' will be initialized if V1 model was loaded.
stream_executor::port::StatusOr<mlir::OwningModuleRef> ImportSavedModel(
    const std::string& input_filename, const int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<const std::string> extra_tf_opdefs,
    absl::Span<std::string> exported_names, const GraphImportConfig& specs,
    bool enable_variable_lifting, mlir::MLIRContext* context,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle);

// Taking a MLIR module in TF executor dialect and a set of parameters,
// applies a set of passes to convert the module to TF Lite dialect and
// serializes the result to a string. Depending on an attribute in the module
// main function, full integer quantization is applied.
// If `quantizated_buffer_type` is provided as INT8 or FLOAT16, the
// corresponding weight quantization will take place.
// If `export_to_mlir` is true, the
// result is exported in MLIR text format, otherwise exported in flat buffer.
Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, bool emit_builtin_tflite_ops,
    bool emit_select_tf_ops, bool emit_custom_ops, bool allow_all_select_tf_ops,
    const std::unordered_set<std::string>& select_user_tf_ops,
    const mlir::TFL::QuantizationSpecs& quant_specs,
    const std::unordered_set<std::string>& saved_model_tags,
    std::string* result, mlir::PassManager* pass_manager);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TO_TFL_FLATBUFFER_H_
