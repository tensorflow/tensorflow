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

#include <string>
#include <vector>

#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

namespace tflite {
namespace evaluation {

// EvaluationStage to compute top-K accuracy of a classification model.
// The computed weights in the model output should be in the same order
// as the vector provided during SetAllLabels
// Ground truth label must be one of provided labels.
// Current accuracies can be obtained with GetLatestMetrics().
class TopkAccuracyEvalStage : public EvaluationStage {
 public:
  explicit TopkAccuracyEvalStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  TfLiteStatus Init() override;

  TfLiteStatus Run() override;

  EvaluationStageMetrics LatestMetrics() override;

  ~TopkAccuracyEvalStage() override {}

  // Call before Init().
  // model_output_shape is not owned, so this class does not free the
  // TfLiteIntArray.
  void SetTaskInfo(const std::vector<std::string>& all_labels,
                   TfLiteType model_output_type,
                   TfLiteIntArray* model_output_shape) {
    // We copy ground_truth_labels to ensure we can access the data throughout
    // the lifetime of this evaluation stage.
    ground_truth_labels_ = all_labels;
    model_output_type_ = model_output_type;
    model_output_shape_ = model_output_shape;
  }

  // Call before Run().
  void SetEvalInputs(void* model_raw_output, std::string* ground_truth_label) {
    model_output_ = model_raw_output;
    ground_truth_label_ = ground_truth_label;
  }

 private:
  // Updates accuracy_counts_ based on comparing top k labels and the
  // groundtruth one. Using string comparison since there are some duplicate
  // labels in the imagenet dataset.
  void UpdateCounts(const std::vector<int>& topk_indices);

  std::vector<std::string> ground_truth_labels_;
  TfLiteType model_output_type_ = kTfLiteNoType;
  TfLiteIntArray* model_output_shape_ = nullptr;
  int num_total_labels_;
  void* model_output_ = nullptr;
  std::string* ground_truth_label_ = nullptr;

  // Equal to number of samples evaluated so far.
  int num_runs_;
  // Stores |k_| values, where the ith value denotes number of samples (out of
  // num_runs_) for which correct label appears in the top (i+1) model outputs.
  std::vector<int> accuracy_counts_;
};

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_TOPK_ACCURACY_EVAL_STAGE_H_
