/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/accuracy/eval_pipeline_builder.h"

#include "absl/memory/memory.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace metrics {

EvalPipelineBuilder& EvalPipelineBuilder::WithInputStage(Stage* input_stage) {
  input_stage_ = input_stage;
  return *this;
}

EvalPipelineBuilder& EvalPipelineBuilder::WithPreprocessingStage(
    Stage* preprocessing_stage) {
  preprocessing_stage_ = preprocessing_stage;
  return *this;
}

EvalPipelineBuilder& EvalPipelineBuilder::WithRunModelStage(
    Stage* run_model_stage) {
  run_model_stage_ = run_model_stage;
  return *this;
}

EvalPipelineBuilder& EvalPipelineBuilder::WithAccuracyEval(
    AccuracyEval* accuracy_eval) {
  accuracy_eval_ = accuracy_eval;
  return *this;
}

EvalPipelineBuilder& EvalPipelineBuilder::WithInput(const string& input_name,
                                                    DataType input_type) {
  input_name_ = input_name;
  input_type_ = input_type;
  return *this;
}

Status EvalPipelineBuilder::Build(
    const Scope& scope, std::unique_ptr<EvalPipeline>* eval_pipeline) {
  if (input_stage_ == nullptr) {
    return errors::InvalidArgument("Input stage is null.");
  }
  if (preprocessing_stage_ == nullptr) {
    return errors::InvalidArgument("Preprocessing stage is null.");
  }
  if (run_model_stage_ == nullptr) {
    return errors::InvalidArgument("Run model stage is null.");
  }
  if (accuracy_eval_ == nullptr) {
    return errors::InvalidArgument("accuracy_eval is null.");
  }
  if (input_name_.empty()) {
    return errors::InvalidArgument("input name is not set.");
  }
  if (input_type_ == DT_INVALID) {
    return errors::InvalidArgument("input type is not set.");
  }

  auto input_placeholder =
      ops::Placeholder(scope.WithOpName(input_name_), input_type_);
  TF_RETURN_IF_ERROR(scope.status());

  input_stage_->AddToGraph(scope, input_placeholder);
  TF_RETURN_IF_ERROR(scope.status());

  preprocessing_stage_->AddToGraph(scope, input_stage_->Output());
  TF_RETURN_IF_ERROR(scope.status());

  run_model_stage_->AddToGraph(scope, preprocessing_stage_->Output());
  TF_RETURN_IF_ERROR(scope.status());

  GraphDef graph_def;
  TF_RETURN_IF_ERROR(scope.ToGraphDef(&graph_def));
  EvalPipeline::Params params;
  params.model_input_node_name = input_name_;
  params.model_output_node_name = run_model_stage_->output_name();
  *eval_pipeline =
      absl::make_unique<EvalPipeline>(graph_def, params, accuracy_eval_);

  return Status::OK();
}

}  //  namespace metrics
}  //  namespace tensorflow
