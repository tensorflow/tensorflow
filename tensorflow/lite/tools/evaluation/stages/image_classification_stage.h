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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_CLASSIFICATION_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_CLASSIFICATION_STAGE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"
#include "tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.h"

namespace tflite {
namespace evaluation {

// An EvaluationStage to encapsulate the complete Image Classification task.
// Utilizes ImagePreprocessingStage, TfLiteInferenceStage &
// TopkAccuracyEvalStage for individual sub-tasks.
class ImageClassificationStage : public EvaluationStage {
 public:
  explicit ImageClassificationStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  TfLiteStatus Init() override { return Init(nullptr); }
  TfLiteStatus Init(const DelegateProviders* delegate_providers);

  TfLiteStatus Run() override;

  EvaluationStageMetrics LatestMetrics() override;

  // Call before Init(), if topk_accuracy_eval_params is set in
  // ImageClassificationParams. all_labels should contain the labels
  // corresponding to model's output, in the same order. all_labels should
  // outlive the call to Init().
  void SetAllLabels(const std::vector<std::string>& all_labels) {
    all_labels_ = &all_labels;
  }

  // Call before Run().
  // If accuracy eval is not being performed, ground_truth_label is ignored.
  void SetInputs(const std::string& image_path,
                 const std::string& ground_truth_label) {
    image_path_ = image_path;
    ground_truth_label_ = ground_truth_label;
  }

  // Provides a pointer to the underlying TfLiteInferenceStage.
  // Returns non-null value only if this stage has been initialized.
  TfliteInferenceStage* const GetInferenceStage() {
    return inference_stage_.get();
  }

 private:
  const std::vector<std::string>* all_labels_ = nullptr;
  std::unique_ptr<ImagePreprocessingStage> preprocessing_stage_;
  std::unique_ptr<TfliteInferenceStage> inference_stage_;
  std::unique_ptr<TopkAccuracyEvalStage> accuracy_eval_stage_;
  std::string image_path_;
  std::string ground_truth_label_;
};

struct ImageLabel {
  std::string image;
  std::string label;
};

// Reads a file containing newline-separated blacklisted image indices and
// filters them out from image_labels.
TfLiteStatus FilterBlackListedImages(const std::string& blacklist_file_path,
                                     std::vector<ImageLabel>* image_labels);

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_CLASSIFICATION_STAGE_H_
