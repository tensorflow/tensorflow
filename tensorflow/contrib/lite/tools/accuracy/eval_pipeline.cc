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

#include "tensorflow/contrib/lite/tools/accuracy/eval_pipeline.h"

namespace tensorflow {
namespace metrics {

Status EvalPipeline::AttachSession(std::unique_ptr<Session> session) {
  session_ = std::move(session);
  TF_RETURN_IF_ERROR(session_->Create(model_graph_));
  return Status::OK();
}

Status EvalPipeline::Run(const Tensor& input, const Tensor& ground_truth) {
  if (session_ == nullptr) {
    return errors::Internal("No session is associated with the graph.");
  }
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(session_->Run({{params_.model_input_node_name, input}},
                                   {params_.model_output_node_name}, {},
                                   &outputs));
  TF_RETURN_IF_ERROR(eval_->ComputeEval(outputs, ground_truth));
  return Status::OK();
}
}  //  namespace metrics
}  //  namespace tensorflow
