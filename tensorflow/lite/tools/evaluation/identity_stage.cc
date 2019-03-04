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
#include "tensorflow/lite/tools/evaluation/identity_stage.h"

#include <ctime>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session_options.h"

namespace tflite {
namespace evaluation {

using ::tensorflow::Scope;
using ::tensorflow::SessionOptions;
using ::tensorflow::Tensor;
using ::tensorflow::ops::Identity;
using ::tensorflow::ops::Placeholder;

IdentityStage::IdentityStage(const EvaluationStageConfig& config)
    : EvaluationStage(config) {
  stage_input_name_ = config_.name() + "_identity_input";
  stage_output_name_ = config_.name() + "_identity_output";
}

bool IdentityStage::DoInit(
    absl::flat_hash_map<std::string, void*>& object_map) {
  // Initialize TF Graph.
  const Scope scope = Scope::NewRootScope();
  if (!GetObjectFromTag(kInputTypeTag, object_map, &input_type_)) {
    return false;
  }
  auto input_placeholder =
      Placeholder(scope.WithOpName(stage_input_name_), *input_type_);
  stage_output_ =
      Identity(scope.WithOpName(stage_output_name_), input_placeholder);
  if (!scope.status().ok() || !scope.ToGraphDef(&graph_def_).ok()) {
    return false;
  }

  // Initialize TF Session.
  session_.reset(NewSession(SessionOptions()));
  if (!session_->Create(graph_def_).ok()) {
    return false;
  }

  return true;
}

bool IdentityStage::Run(absl::flat_hash_map<std::string, void*>& object_map,
                        EvaluationStageMetrics& metrics) {
  std::vector<Tensor>* input_tensors;
  if (!GetObjectFromTag(kInputTensorsTag, object_map, &input_tensors)) {
    return false;
  }
  tensor_outputs_.clear();
  // TODO(b/122482115): Encapsulate timing into its own helper.
  std::clock_t start = std::clock();
  if (!session_
           ->Run({{stage_input_name_, input_tensors->at(0)}},
                 {stage_output_name_}, {}, &tensor_outputs_)
           .ok()) {
    return false;
  }
  metrics.set_total_latency_ms(
      static_cast<float>((std::clock() - start) / (CLOCKS_PER_SEC / 1000)));

  if (!AssignObjectToTag(kOutputTensorsTag, &tensor_outputs_, object_map)) {
    return false;
  }
  return true;
}

const char IdentityStage::kInputTypeTag[] = "INPUT_TYPE";
const char IdentityStage::kInputTensorsTag[] = "INPUT_TENSORS";
const char IdentityStage::kOutputTensorsTag[] = "OUTPUT_TENSORS";

}  // namespace evaluation
}  // namespace tflite
