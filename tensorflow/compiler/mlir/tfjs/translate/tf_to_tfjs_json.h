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

#ifndef TENSORFLOW_COMPILER_MLIR_TFJS_TRANSLATE_TF_TO_TFJS_JSON_H_
#define TENSORFLOW_COMPILER_MLIR_TFJS_TRANSLATE_TF_TO_TFJS_JSON_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"
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
    const std::vector<std::string>& extra_tf_opdefs,
    absl::string_view debug_info_file, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, bool prune_unused_nodes,
    llvm::SourceMgr* source_mgr, mlir::MLIRContext* context);

// Load Saved model (either v1 or v2) into MLIR.
stream_executor::port::StatusOr<mlir::OwningModuleRef> ImportSavedModel(
    bool import_saved_model, bool import_saved_model_v1,
    const std::vector<std::string>& extra_tf_opdefs,
    const std::string& input_filename, const std::string& saved_model_tags,
    const std::string& saved_model_exported_names, mlir::MLIRContext* context);

// Taking a MLIR module in TF executor dialect and a set of parameters,
// applies a set of passes to convert the module to TFJS dialect and
// serializes the result to JSON string.
// If `export_to_mlir` is true, the result is exported in MLIR text format,
// otherwise exported in JSON.
Status ConvertTFOpsToTfjsJSON(mlir::ModuleOp module, bool export_to_mlir,
                              std::string* result,
                              mlir::PassManager* pass_manager);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFJS_TRANSLATE_TF_TO_TFJS_JSON_H_
