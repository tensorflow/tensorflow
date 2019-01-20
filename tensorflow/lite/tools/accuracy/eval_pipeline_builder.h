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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_EVAL_PIPELINE_BUILDER_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_EVAL_PIPELINE_BUILDER_H_

#include <memory>
#include <string>

#include "tensorflow/lite/tools/accuracy/accuracy_eval_stage.h"
#include "tensorflow/lite/tools/accuracy/eval_pipeline.h"
#include "tensorflow/lite/tools/accuracy/stage.h"

namespace tensorflow {
namespace metrics {

// A builder to simplify construction of an `EvalPipeline` instance.
// The `Build` method creates an |EvalPipeline| with the following structure:
// |input| -> |input_stage|
//               |--> |preprocessing_stage|
//                         |--> |run_model_stage| ->  |accuracy_eval_stage|.
// The stages are chained in the order shown above. Any missing stage results in
// an error. The ownership of the stage object is retained by the caller. Stage
// objects need to exist until the |Build| method is called.
//
// Currently only single inputs are supported.
//
// Example Usage:
// EvalPipelineBuilder builder;
// std::unique_ptr<EvalPipeline> eval_pipeline;
// auto status = builder.WithInput("pipeline_input", DT_FLOAT)
//      .WithInputStage(&input_stage)
//      .WithRunModelStage(&run_model_stage)
//      .WithPreprocessingStage(&preprocess_stage)
//      .WithAccuracyEval(&eval)
//      .Build(scope, &eval_pipeline);
// TF_CHECK_OK(status);
class EvalPipelineBuilder {
 public:
  EvalPipelineBuilder() = default;
  EvalPipelineBuilder(const EvalPipelineBuilder&) = delete;
  EvalPipeline& operator=(const EvalPipelineBuilder&) = delete;

  EvalPipelineBuilder(const EvalPipelineBuilder&&) = delete;
  EvalPipeline& operator=(const EvalPipelineBuilder&&) = delete;

  // Sets the input stage for the pipeline.
  // Input stage converts the input, say filename into appropriate format
  // that can be consumed by the preprocessing stage.
  EvalPipelineBuilder& WithInputStage(Stage* input_stage);

  // Sets the preprocessing stage for the pipeline.
  // Preprocessing stage converts the input into a format that can be used to
  // run the model.
  EvalPipelineBuilder& WithPreprocessingStage(Stage* preprocessing_stage);

  // Sets the run model stage for the pipeline.
  // This stage receives the preprocessing input and output of this stage is
  // fed to the accuracy eval stage.
  EvalPipelineBuilder& WithRunModelStage(Stage* run_model_stage);

  // Sets the accuracy eval for the pipeline.
  // Results of evaluating the pipeline are fed to the `accuracy_eval` instance.
  EvalPipelineBuilder& WithAccuracyEval(AccuracyEval* accuracy_eval);

  // Sets the name and type of input for the pipeline.
  // TODO(shashishekhar): Support multiple inputs for the pipeline, use a vector
  // here.
  EvalPipelineBuilder& WithInput(const string& input_name, DataType input_type);

  // Builds the pipeline and assigns the pipeline to `eval_pipeline`.
  // If the pipeline creation fails `eval_pipeline` is untouched.
  Status Build(const Scope& scope,
               std::unique_ptr<EvalPipeline>* eval_pipeline);

 private:
  Stage* input_stage_ = nullptr;
  Stage* preprocessing_stage_ = nullptr;
  Stage* run_model_stage_ = nullptr;
  AccuracyEval* accuracy_eval_ = nullptr;
  string input_name_;
  DataType input_type_ = DT_INVALID;
};

}  //  namespace metrics
}  //  namespace tensorflow
#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_EVAL_PIPELINE_BUILDER_H_
