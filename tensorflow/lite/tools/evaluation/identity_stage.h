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
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

namespace tflite {
namespace evaluation {

// Simple EvaluationStage that passes INPUT_VALUE to OUTPUT_VALUE if the former
// is non-zero, DEFAULT_VALUE otherwise. Primarily used for tests.
// Initializer TAGs (Object Class): DEFAULT_VALUE (float)
// Input TAGs (Object Class): INPUT_VALUE (float)
// Output TAGs (Object Class): OUTPUT_VALUE (float)
class IdentityStage : public EvaluationStage {
 public:
  explicit IdentityStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  bool Run(absl::flat_hash_map<std::string, void*>& object_map) override;

  EvaluationStageMetrics LatestMetrics() override;

  ~IdentityStage() {}

 protected:
  bool DoInit(absl::flat_hash_map<std::string, void*>& object_map) override;

  std::vector<std::string> GetInitializerTags() override {
    return {kDefaultValueTag};
  }
  std::vector<std::string> GetInputTags() override { return {kInputValueTag}; }
  std::vector<std::string> GetOutputTags() override {
    return {kOutputValueTag};
  }

 private:
  float default_value_ = 0;
  float current_value_ = 0;
  int num_runs_ = 0;

  static const char kDefaultValueTag[];
  static const char kInputValueTag[];
  static const char kOutputValueTag[];
};

DECLARE_FACTORY(IdentityStage);

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_IDENTITY_STAGE_H_
