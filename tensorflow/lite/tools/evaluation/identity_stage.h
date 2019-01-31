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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_IDENTITY_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_IDENTITY_STAGE_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

namespace tflite {
namespace evaluation {

// Simple EvaluationStage subclass that encapsulates the functionality of
// tensorflow::ops::Identity. Primarily used for tests.
// Initializer TAGs (Object Class): INPUT_TYPE (DataType)
// Input TAGs (Object Class): INPUT_TENSORS (std::vector<Tensor>)
// Output TAGs (Object Class): OUTPUT_TENSORS (std::vector<Tensor>)
// TODO(b/122482115): Migrate common TF-related code into an abstract class.
class IdentityStage : public EvaluationStage {
 public:
  explicit IdentityStage(const EvaluationStageConfig& config);

  bool Run(absl::flat_hash_map<std::string, void*>& object_map,
           EvaluationStageMetrics& metrics) override;

  ~IdentityStage() {}

 protected:
  bool DoInit(absl::flat_hash_map<std::string, void*>& object_map) override;

  std::vector<std::string> GetInitializerTags() override {
    return {kInputTypeTag};
  }
  std::vector<std::string> GetInputTags() override {
    return {kInputTensorsTag};
  }
  std::vector<std::string> GetOutputTags() override {
    return {kOutputTensorsTag};
  }

 private:
  ::tensorflow::DataType* input_type_;
  ::tensorflow::GraphDef graph_def_;
  ::tensorflow::Output stage_output_;
  std::unique_ptr<::tensorflow::Session> session_;
  std::vector<::tensorflow::Tensor> tensor_outputs_;
  std::string stage_input_name_;
  std::string stage_output_name_;

  static const char kInputTypeTag[];
  static const char kInputTensorsTag[];
  static const char kOutputTensorsTag[];
};

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_IDENTITY_STAGE_H_
