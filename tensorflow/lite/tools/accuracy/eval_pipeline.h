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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_EVAL_PIPELINE_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_EVAL_PIPELINE_H_

#include <string>

#include "tensorflow/lite/tools/accuracy/accuracy_eval_stage.h"
#include "tensorflow/lite/tools/accuracy/stage.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace metrics {

// Pipeline for evaluating a model.
// Runs the graph and passes the output of graph to
// the provided instance of AccuracyEval.
// Example usage:
// AccuracyEval *eval;
// GraphDef graph_def;
// ... populate graph_def...
//
// EvalPipeline eval_pipeline(&graph_def,
//    {.model_input_node_name = "model_input",
//     .model_output_node_name = "model_output"},
//     eval);
//  std::unique_ptr<Session> session(NewSession(SessionOptions()));
//  TF_CHECK_OK(eval_pipeline.AttachSession(std::move(session)));
//  Tensor input = ... read input for the model ...
//  Tensor ground_truth = ... read ground truth for the model ...
//  TF_CHECK_OK(eval_pipeline.Run(input, ground_truth));
//
class EvalPipeline {
 public:
  struct Params {
    string model_input_node_name;
    string model_output_node_name;
  };

  // Creates a new `EvalPipeline` object. The ownership of the `accuracy_eval`
  // is retained by the caller. Lifetime of `accuracy_eval` instance should
  // be longer than the lifetime of this instance of pipeline.
  EvalPipeline(const GraphDef& graph, const Params& params,
               AccuracyEval* accuracy_eval)
      : model_graph_(graph),
        params_(params),
        eval_(accuracy_eval),
        session_(nullptr) {}

  EvalPipeline(const EvalPipeline&) = delete;
  EvalPipeline& operator=(const EvalPipeline&) = delete;

  EvalPipeline(const EvalPipeline&&) = delete;
  EvalPipeline& operator=(const EvalPipeline&&) = delete;

  // Attaches the given session to this instance of pipeline.
  // The provided session object will be reused for subsequent calls to
  // EvalPipeline::Run.
  Status AttachSession(std::unique_ptr<Session> session);

  // Runs the model by feeding `input` and then passes the output of the model
  // along with provided `ground_truth` to the AccuracyEval instance by calling
  // AccuracyEval::ComputeEval.
  Status Run(const Tensor& input, const Tensor& ground_truth);

 private:
  GraphDef model_graph_;
  Params params_;
  AccuracyEval* eval_;
  std::unique_ptr<Session> session_;
};
}  //  namespace metrics
}  //  namespace tensorflow
#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_EVAL_PIPELINE_H_
