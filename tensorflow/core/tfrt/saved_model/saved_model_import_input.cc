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
#include "tensorflow/core/tfrt/saved_model/saved_model_import_input.h"

#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace tfrt_stub {

StatusOr<TfrtSavedModelMLIRImportInput> TfrtSavedModelMLIRImportInput::Create(
    const FallbackState& fallback_state, const MetaGraphDef* meta_graph_def,
    const GraphDebugInfo& debug_info,
    bool run_placer_grappler_on_nested_functions) {
  DCHECK(meta_graph_def);

  TF_ASSIGN_OR_RETURN(auto graph_execution_state,
                      TfrtGraphExecutionState::Create(
                          meta_graph_def->graph_def(), fallback_state,
                          run_placer_grappler_on_nested_functions));

  return TfrtSavedModelMLIRImportInput(meta_graph_def, debug_info,
                                       std::move(graph_execution_state));
}

TfrtSavedModelMLIRImportInput::TfrtSavedModelMLIRImportInput(
    const MetaGraphDef* meta_graph_def, const GraphDebugInfo& debug_info,
    std::unique_ptr<TfrtGraphExecutionState> graph_execution_state)
    : SavedModelMLIRImportInput(meta_graph_def, debug_info),
      graph_execution_state_(std::move(graph_execution_state)) {}

StatusOr<const tensorflow::Graph*> TfrtSavedModelMLIRImportInput::GetSubGraph(
    absl::string_view name, const GraphImportConfig& graph_import_config) {
  LOG(INFO) << "TFRT importing savedmodel signature: " << name;

  auto iter = optimized_graphs_.find(name);
  if (iter != optimized_graphs_.end()) return iter->second.get();

  TF_ASSIGN_OR_RETURN(
      auto optimization_result,
      graph_execution_state_->CreateOptimizedGraph(graph_import_config));

  functionalization_duration_ += optimization_result.functionalization_duration;
  grappler_duration_ += optimization_result.grappler_duration;

  const auto* optimized_graph_ptr = optimization_result.graph.get();
  DCHECK(optimized_graph_ptr);
  optimized_graphs_[name] = std::move(optimization_result.graph);
  return optimized_graph_ptr;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
