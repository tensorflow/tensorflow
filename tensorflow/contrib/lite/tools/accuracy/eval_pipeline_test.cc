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
#include <gtest/gtest.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace metrics {
namespace {

Tensor CreateFloatTensor(float value) {
  Tensor tensor(DT_FLOAT, TensorShape({}));
  tensor.scalar<float>()() = value;
  return tensor;
}

class NoOpAccuracyEval : public AccuracyEval {
 public:
  explicit NoOpAccuracyEval(const Status& status_to_return)
      : status_to_return_(status_to_return) {}

  Status ComputeEval(const std::vector<Tensor>& model_outputs,
                     const Tensor& ground_truth) override {
    model_outputs_ = model_outputs;
    ground_truth_ = ground_truth;
    was_called_ = true;
    return status_to_return_;
  }

  bool WasCalled() { return was_called_; }
  std::vector<Tensor> model_outputs() { return model_outputs_; }
  Tensor ground_truth() { return ground_truth_; }

 private:
  std::vector<Tensor> model_outputs_;
  Tensor ground_truth_;
  Status status_to_return_;
  bool was_called_ = false;
};

TEST(EvalPipeline, AccuracyEvalIsCalled) {
  Scope scope = Scope::NewRootScope();
  // A graph that adds 1 to input.
  auto input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);
  auto add_node = ops::Add(scope.WithOpName("output"), input, 1.0f);
  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  EvalPipeline::Params params;
  params.model_input_node_name = "input";
  params.model_output_node_name = "output";
  NoOpAccuracyEval accuracy_eval(Status::OK());

  EvalPipeline eval_pipeline(graph_def, params, &accuracy_eval);
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(eval_pipeline.AttachSession(std::move(session)));
  TF_CHECK_OK(eval_pipeline.Run(CreateFloatTensor(5), CreateFloatTensor(27)));

  EXPECT_TRUE(accuracy_eval.WasCalled());
  auto outputs = accuracy_eval.model_outputs();
  ASSERT_EQ(1, outputs.size());
  EXPECT_EQ(6.0f, outputs[0].scalar<float>()());
  // Ground truth is unchanged.
  EXPECT_EQ(27, accuracy_eval.ground_truth().scalar<float>()());
}

TEST(EvalPipeline, EvalIsNotCalledOnGraphRunFailure) {
  Scope scope = Scope::NewRootScope();
  // A graph that adds 1 to input.
  auto input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);
  auto add_node = ops::Add(scope.WithOpName("output"), input, 1.0f);
  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  EvalPipeline::Params params;
  params.model_input_node_name = "input";
  params.model_output_node_name = "output";
  NoOpAccuracyEval accuracy_eval(Status::OK());

  EvalPipeline eval_pipeline(graph_def, params, &accuracy_eval);
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(eval_pipeline.AttachSession(std::move(session)));

  // Pass a string tensor instead of a float tensor.
  Tensor string_tensor(DT_STRING, TensorShape{});
  auto status = eval_pipeline.Run(string_tensor, CreateFloatTensor(27));
  EXPECT_FALSE(accuracy_eval.WasCalled());
  EXPECT_FALSE(status.ok());
}

TEST(EvalPipeline, AccuracyEvalFailureResultsInFailure) {
  Scope scope = Scope::NewRootScope();
  // A graph that adds 1 to input.
  auto input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);
  auto add_node = ops::Add(scope.WithOpName("output"), input, 1.0f);
  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  EvalPipeline::Params params;
  params.model_input_node_name = "input";
  params.model_output_node_name = "output";
  NoOpAccuracyEval accuracy_eval(errors::Internal("accuracy_fail"));

  EvalPipeline eval_pipeline(graph_def, params, &accuracy_eval);
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(eval_pipeline.AttachSession(std::move(session)));
  auto status = eval_pipeline.Run(CreateFloatTensor(5), CreateFloatTensor(27));

  EXPECT_TRUE(accuracy_eval.WasCalled());
  EXPECT_FALSE(status.ok());
}

}  // namespace

}  // namespace metrics
}  // namespace tensorflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
