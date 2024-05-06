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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_

#include <string>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

inline constexpr absl::string_view kImportModelDefaultGraphFuncName = "main";

// Given a GraphDef, returns a MLIR module containing the graph, expressed with
// tf_executor dialect.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphdefToMlir(
    const GraphDef& graphdef, const GraphDebugInfo& debug_info,
    const GraphImportConfig& specs, mlir::MLIRContext* context);

// Given a Graph, returns a MLIR module containing the graph, expressed with
// tf_executor dialect.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphToMlir(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context);

// [Experimental]
// Given a Function, returns a MLIR module containing the graph, expressed with
// tf_executor dialect.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertFunctionToMlir(
    const FunctionBody* fbody, const FunctionLibraryDefinition& flib_def,
    mlir::MLIRContext* context);

// Given a SavedModel, returns a MLIR module containing the functions, expressed
// with tf_executor dialect.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelToMlir(
    SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
    absl::Span<std::string> exported_names, MLIRImportOptions options = {});

// Given a V1 SavedModel, returns a MLIR module containing the functions,
// expressed with tf_executor dialect.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlir(
    const SavedModelBundle& saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options = {});

// Given a V1 SavedModel, returns a MLIR module containing the functions,
// expressed with tf_executor dialect. It does not require a session to be
// created and it does not perform any graph transformation. If `exported_names`
// is std::nullopt, all signatures will be imported. Otherwise, only names
// in `exported_names` are imported.
//
// Note that the word `Lite` means it is a lighter version compared to
// ConvertSavedModelV1ToMlir(), and is not related to TFLite.
//
// TODO(b/179683149): Rename this class to avoid confusion with TFLite.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlirLite(
    const MetaGraphDef& meta_graph_def, const GraphDebugInfo& debug_info,
    std::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options);

// SavedModelMLIRImportInput is an adapter class for users to inject custom
// graph transformation logic on Tensorflow graphs before importing to MLIR. It
// serves as the source that provides the subgraphs requested by the savedmodel
// MLIR importer, and at the same time it allows the implementation of this
// class to transform the graph before feeding it to the importer.
class SavedModelMLIRImportInput {
 public:
  SavedModelMLIRImportInput(const MetaGraphDef* meta_graph_def,
                            const GraphDebugInfo& debug_info)
      : meta_graph_def_(meta_graph_def), debug_info_(debug_info) {
    DCHECK(meta_graph_def);
  }

  virtual ~SavedModelMLIRImportInput();

  // The original MetaGraphDef of the savedmodel.
  const MetaGraphDef& meta_graph_def() const { return *meta_graph_def_; }

  const GraphDebugInfo& debug_info() const { return debug_info_; }

  // GetSubGraph() is expected to return a tensorflow::Graph that contains the
  // node set specified in `specs`. The implementation is free to transform the
  // graph in the original savedmodel as needed, as long as it produces the same
  // results and effects. If the transformation requires some configs in `spec`
  // (e.g., control_outputs) to be changed, they should be updated accordingly
  // and remain valid for the graph.
  // `name` is a unique identifier for this subgraph, so the implementation can
  // use it for eg. debugging or caching compilation results.
  virtual absl::StatusOr<const Graph*> GetSubGraph(
      absl::string_view name, GraphImportConfig& specs) = 0;

 private:
  const MetaGraphDef* meta_graph_def_ = nullptr;
  GraphDebugInfo debug_info_;
};

// Given the SavedModelMLIRImportInput for a saved model, returns a MLIR module
// containing the functions, expressed with tf_executor dialect. It does not
// require a session to be created. If `exported_names` is std::nullopt, all
// signatures will be imported. Otherwise, only names in `exported_names` are
// imported.

//
// Note that the word `Lite` means it is a lighter version compared to
// ConvertSavedModelV1ToMlir(), and is not related to TFLite.
//
// TODO(b/179683149): Rename this class to avoid confusion with TFLite.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlirLite(
    SavedModelMLIRImportInput& input,
    std::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context,
    bool unconditionally_use_set_output_shapes = false);

// Serialize a MLIR module to a string.
std::string MlirModuleToString(mlir::ModuleOp module,
                               mlir::OpPrintingFlags flags);
std::string MlirModuleToString(mlir::ModuleOp m, bool show_debug_info = false);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_IMPORT_MODEL_H_
