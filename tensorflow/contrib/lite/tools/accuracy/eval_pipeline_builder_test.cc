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

#include "tensorflow/contrib/lite/tools/accuracy/eval_pipeline_builder.h"
#include <gtest/gtest.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace metrics {
namespace {

class IdentityStage : public Stage {
 public:
  IdentityStage(const string& name, const string& output)
      : name_(name), output_(output) {}

  void AddToGraph(const Scope& scope, const Input& input) override {
    called_count_++;
    inputs_.push_back(input.node()->name());
    stage_output_ = ops::Identity(scope.WithOpName(output_), input);
  }

  string name() const override { return name_; }
  string output_name() const override { return output_; }

  int times_called() const { return called_count_; }

  const std::vector<string> input_params() { return inputs_; }

 private:
  string name_;
  string output_;
  int called_count_ = 0;
  std::vector<string> inputs_;
};

class FailingStage : public Stage {
 public:
  FailingStage(const string& name, const string& output)
      : name_(name), output_(output) {}

  void AddToGraph(const Scope& scope, const Input& input) override {
    called_count_++;
    scope.UpdateStatus(errors::Internal("Stage failed:", name_));
  }

  string name() const override { return name_; }
  string output_name() const override { return output_; }

  int times_called() const { return called_count_; }

 private:
  string name_;
  string output_;
  int called_count_ = 0;
};

class SimpleAccuracyEval : public AccuracyEval {
 public:
  SimpleAccuracyEval() {}

  Status ComputeEval(const std::vector<Tensor>& model_outputs,
                     const Tensor& ground_truth) override {
    return Status::OK();
  }
};

TEST(EvalPipelineBuilder, MissingPipelineStages) {
  IdentityStage input_stage("input_stage", "input_stage_out");
  IdentityStage run_model_stage("run_model", "run_model_out");
  IdentityStage preprocess_stage("preprocess_stage", "preprocess_stage_out");
  const string pipeline_input = "pipeline_input";

  SimpleAccuracyEval eval;

  Scope scope = Scope::NewRootScope();
  std::unique_ptr<EvalPipeline> eval_pipeline;
  EvalPipelineBuilder builder;
  auto status =
      builder.WithInputStage(&input_stage).Build(scope, &eval_pipeline);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(eval_pipeline);

  status =
      builder.WithRunModelStage(&run_model_stage).Build(scope, &eval_pipeline);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(eval_pipeline);

  status = builder.WithPreprocessingStage(&preprocess_stage)
               .Build(scope, &eval_pipeline);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(eval_pipeline);

  status =
      builder.WithInput(pipeline_input, DT_FLOAT).Build(scope, &eval_pipeline);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(eval_pipeline);

  status = builder.WithAccuracyEval(&eval).Build(scope, &eval_pipeline);
  TF_CHECK_OK(status);
  EXPECT_TRUE(eval_pipeline);
}

TEST(EvalPipeline, InputStageFailure) {
  FailingStage input_stage("input_stage", "input_stage_out");
  IdentityStage run_model_stage("run_model", "run_model_out");
  IdentityStage preprocess_stage("preprocess_stage", "preprocess_stage_out");
  const string pipeline_input = "pipeline_input";

  SimpleAccuracyEval eval;

  Scope scope = Scope::NewRootScope();
  std::unique_ptr<EvalPipeline> eval_pipeline;
  EvalPipelineBuilder builder;
  auto status = builder.WithInputStage(&input_stage)
                    .WithRunModelStage(&run_model_stage)
                    .WithPreprocessingStage(&preprocess_stage)
                    .WithInput(pipeline_input, DT_FLOAT)
                    .WithAccuracyEval(&eval)
                    .Build(scope, &eval_pipeline);

  EXPECT_FALSE(scope.status().ok());
  // None of the other stages would have been called.
  EXPECT_EQ(1, input_stage.times_called());
  EXPECT_EQ(0, preprocess_stage.times_called());
  EXPECT_EQ(0, run_model_stage.times_called());
}

TEST(EvalPipeline, PreprocessingFailure) {
  IdentityStage input_stage("input_stage", "input_stage_out");
  FailingStage preprocess_stage("preprocess_stage", "preprocess_stage_out");
  IdentityStage run_model_stage("run_model", "run_model_out");
  const string pipeline_input = "pipeline_input";

  SimpleAccuracyEval eval;

  Scope scope = Scope::NewRootScope();
  std::unique_ptr<EvalPipeline> eval_pipeline;
  EvalPipelineBuilder builder;
  auto status = builder.WithInputStage(&input_stage)
                    .WithRunModelStage(&run_model_stage)
                    .WithPreprocessingStage(&preprocess_stage)
                    .WithInput(pipeline_input, DT_FLOAT)
                    .WithAccuracyEval(&eval)
                    .Build(scope, &eval_pipeline);

  EXPECT_FALSE(status.ok());
  // None of the other stages would have been called.
  EXPECT_EQ(1, input_stage.times_called());
  EXPECT_EQ(1, preprocess_stage.times_called());
  EXPECT_EQ(0, run_model_stage.times_called());
}

TEST(EvalPipeline, GraphEvalFailure) {
  IdentityStage input_stage("input_stage", "input_stage_out");
  IdentityStage preprocess_stage("preprocess_stage", "preprocess_stage_out");
  FailingStage run_model_stage("run_model", "run_model_out");
  const string pipeline_input = "pipeline_input";

  SimpleAccuracyEval eval;

  Scope scope = Scope::NewRootScope();
  std::unique_ptr<EvalPipeline> eval_pipeline;
  EvalPipelineBuilder builder;
  auto status = builder.WithInputStage(&input_stage)
                    .WithRunModelStage(&run_model_stage)
                    .WithPreprocessingStage(&preprocess_stage)
                    .WithInput(pipeline_input, DT_FLOAT)
                    .WithAccuracyEval(&eval)
                    .Build(scope, &eval_pipeline);

  EXPECT_FALSE(status.ok());
  // None of the other stages would have been called.
  EXPECT_EQ(1, input_stage.times_called());
  EXPECT_EQ(1, preprocess_stage.times_called());
  EXPECT_EQ(1, run_model_stage.times_called());
}

TEST(EvalPipeline, PipelineHasCorrectSequence) {
  IdentityStage input_stage("input_stage", "input_stage_out");
  IdentityStage preprocess_stage("preprocess_stage", "preprocess_stage_out");
  IdentityStage run_model_stage("run_model", "run_model_out");
  const string pipeline_input = "pipeline_input";

  SimpleAccuracyEval eval;

  Scope scope = Scope::NewRootScope();
  std::unique_ptr<EvalPipeline> eval_pipeline;
  EvalPipelineBuilder builder;
  auto status = builder.WithInputStage(&input_stage)
                    .WithRunModelStage(&run_model_stage)
                    .WithPreprocessingStage(&preprocess_stage)
                    .WithInput(pipeline_input, DT_FLOAT)
                    .WithAccuracyEval(&eval)
                    .Build(scope, &eval_pipeline);
  TF_CHECK_OK(status);

  ASSERT_EQ(1, input_stage.times_called());
  ASSERT_EQ(1, run_model_stage.times_called());
  ASSERT_EQ(1, preprocess_stage.times_called());

  EXPECT_EQ(pipeline_input, input_stage.input_params()[0]);
  EXPECT_EQ(input_stage.output_name(), preprocess_stage.input_params()[0]);
  EXPECT_EQ(preprocess_stage.output_name(), run_model_stage.input_params()[0]);
}

}  // namespace

}  // namespace metrics
}  // namespace tensorflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
