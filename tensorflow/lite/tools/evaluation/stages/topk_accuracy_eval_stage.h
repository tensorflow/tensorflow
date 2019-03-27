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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_TOPK_ACCURACY_EVAL_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_TOPK_ACCURACY_EVAL_STAGE_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

namespace tflite {
namespace evaluation {

// EvaluationStage to compute top-K accuracy of a classification model.
// The computed weights in the model output should be in the same order
// as ALL_LABELS provided during Init.
// GROUND_TRUTH_LABEL must be one of ALL_LABELS.
// Current accuracies can be obtained with GetLatestMetrics().
// Note: MODEL_OUTPUT_* are taken as Initializers & not in the config, since
// they are typically computed by inference stage after analyzing a tflite file.
//
// Initializer TAGs (Object Class): ALL_LABELS (std::vector<string>*),
//                                  MODEL_OUTPUT_TYPE (tflite::TfLiteType*),
//                                  MODEL_OUTPUT_SHAPE (tflite::TfLiteIntArray*)
// Input TAGs (Object Class): MODEL_OUTPUT (pointer to model output, based on
//                                TopkAccuracyEvalParams.output_type),
//                            GROUND_TRUTH_LABEL (std::string*)
class TopkAccuracyEvalStage : public EvaluationStage {
 public:
  explicit TopkAccuracyEvalStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  bool Run(absl::flat_hash_map<std::string, void*>& object_map) override;

  EvaluationStageMetrics LatestMetrics() override;

  ~TopkAccuracyEvalStage() {}

 protected:
  bool DoInit(absl::flat_hash_map<std::string, void*>& object_map) override;

  std::vector<std::string> GetInitializerTags() override {
    return {kAllLabelsTag, kModelOutputTypeTag, kModelOutputShapeTag};
  }
  std::vector<std::string> GetInputTags() override {
    return {kModelOutputTag, kGroundTruthLabelTag};
  }
  std::vector<std::string> GetOutputTags() override { return {}; }

 private:
  // Returns the index of label from ground_truth_labels_.
  int GroundTruthIndex(const std::string& label) const;
  // Updates accuracy_counts_ based on the top k indices & index of the ground
  // truth.
  void UpdateCounts(const std::vector<int>& topk_indices,
                    int ground_truth_index);

  int k_;
  std::vector<std::string> ground_truth_labels_;
  int num_total_labels_;
  // Equal to number of samples evaluated so far.
  int num_runs_;
  // Stores |k_| values, where the ith value denotes number of samples (out of
  // num_runs_) for which correct label appears in the top (i+1) model outputs.
  std::vector<int> accuracy_counts_;
  // Output type of model.
  TfLiteType output_type_;

  // Initializers.
  static const char kAllLabelsTag[];
  static const char kModelOutputTypeTag[];
  static const char kModelOutputShapeTag[];
  // Inputs.
  static const char kModelOutputTag[];
  static const char kGroundTruthLabelTag[];
};

DECLARE_FACTORY(TopkAccuracyEvalStage);

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_TOPK_ACCURACY_EVAL_STAGE_H_
