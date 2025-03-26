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
#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_IMPORT_INPUT_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_IMPORT_INPUT_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"

namespace tensorflow {
namespace tfrt_stub {

// TfrtSavedModelMLIRImportInput implements SavedModelMLIRImportInput, so that
// it can perform customization (eg. Placer and Grappler) on the input graph to
// the MLIR importer.
class TfrtSavedModelMLIRImportInput : public SavedModelMLIRImportInput {
 public:
  static absl::StatusOr<TfrtSavedModelMLIRImportInput> Create(
      const FallbackState& fallback_state, const MetaGraphDef* meta_graph_def,
      const GraphDebugInfo& debug_info,
      bool run_placer_grappler_on_nested_functions = false,
      tensorflow::tfrt_stub::RuntimeConfig* runtime_config = nullptr);

  TfrtSavedModelMLIRImportInput(
      const MetaGraphDef* meta_graph_def, const GraphDebugInfo& debug_info,
      std::unique_ptr<TfrtGraphExecutionState> graph_execution_state);

  absl::StatusOr<const tensorflow::Graph*> GetSubGraph(
      absl::string_view name, GraphImportConfig& graph_import_config) override;

  // Return the time used by grappler.
  absl::Duration GetGrapplerDuration() const { return grappler_duration_; }

  // Return the time used by functionalization.
  absl::Duration GetFunctionalizationDuration() const {
    return functionalization_duration_;
  }

 private:
  std::unique_ptr<TfrtGraphExecutionState> graph_execution_state_;
  absl::flat_hash_map<std::string, std::unique_ptr<tensorflow::Graph>>
      optimized_graphs_;

  absl::Duration functionalization_duration_;
  absl::Duration grappler_duration_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_IMPORT_INPUT_H_
