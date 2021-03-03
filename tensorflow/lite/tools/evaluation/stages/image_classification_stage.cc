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
#include "tensorflow/lite/tools/evaluation/stages/image_classification_stage.h"

#include <algorithm>
#include <iterator>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace evaluation {
namespace {
// Default cropping fraction value.
const float kCroppingFraction = 0.875;
}  // namespace

TfLiteStatus ImageClassificationStage::Init(
    const DelegateProviders* delegate_providers) {
  // Ensure inference params are provided.
  if (!config_.specification().has_image_classification_params()) {
    LOG(ERROR) << "ImageClassificationParams not provided";
    return kTfLiteError;
  }
  auto& params = config_.specification().image_classification_params();
  if (!params.has_inference_params()) {
    LOG(ERROR) << "Inference_params not provided";
    return kTfLiteError;
  }

  // TfliteInferenceStage.
  EvaluationStageConfig tflite_inference_config;
  tflite_inference_config.set_name("tflite_inference");
  *tflite_inference_config.mutable_specification()
       ->mutable_tflite_inference_params() = params.inference_params();
  inference_stage_.reset(new TfliteInferenceStage(tflite_inference_config));
  if (inference_stage_->Init(delegate_providers) != kTfLiteOk)
    return kTfLiteError;

  // Validate model inputs.
  const TfLiteModelInfo* model_info = inference_stage_->GetModelInfo();
  if (model_info->inputs.size() != 1 || model_info->outputs.size() != 1) {
    LOG(ERROR) << "Model must have 1 input & 1 output";
    return kTfLiteError;
  }
  TfLiteType input_type = model_info->inputs[0]->type;
  auto* input_shape = model_info->inputs[0]->dims;
  // Input should be of the shape {1, height, width, 3}
  if (input_shape->size != 4 || input_shape->data[0] != 1 ||
      input_shape->data[3] != 3) {
    LOG(ERROR) << "Invalid input shape for model";
    return kTfLiteError;
  }

  // ImagePreprocessingStage
  if (!config_.specification().has_image_preprocessing_params()) {
    tflite::evaluation::ImagePreprocessingConfigBuilder builder(
        "image_preprocessing", input_type);
    builder.AddCroppingStep(kCroppingFraction, true /*square*/);
    builder.AddResizingStep(input_shape->data[2], input_shape->data[1], false);
    builder.AddDefaultNormalizationStep();
    preprocessing_stage_.reset(new ImagePreprocessingStage(builder.build()));
  } else {
    preprocessing_stage_.reset(new ImagePreprocessingStage(config_));
  }
  if (preprocessing_stage_->Init() != kTfLiteOk) return kTfLiteError;

  // TopkAccuracyEvalStage.
  if (params.has_topk_accuracy_eval_params()) {
    EvaluationStageConfig topk_accuracy_eval_config;
    topk_accuracy_eval_config.set_name("topk_accuracy");
    *topk_accuracy_eval_config.mutable_specification()
         ->mutable_topk_accuracy_eval_params() =
        params.topk_accuracy_eval_params();
    if (!all_labels_) {
      LOG(ERROR) << "all_labels not set for TopkAccuracyEvalStage";
      return kTfLiteError;
    }
    accuracy_eval_stage_.reset(
        new TopkAccuracyEvalStage(topk_accuracy_eval_config));
    accuracy_eval_stage_->SetTaskInfo(*all_labels_, input_type,
                                      model_info->outputs[0]->dims);
    if (accuracy_eval_stage_->Init() != kTfLiteOk) return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus ImageClassificationStage::Run() {
  if (image_path_.empty()) {
    LOG(ERROR) << "Input image not set";
    return kTfLiteError;
  }

  // Preprocessing.
  preprocessing_stage_->SetImagePath(&image_path_);
  if (preprocessing_stage_->Run() != kTfLiteOk) return kTfLiteError;
  // Inference.
  std::vector<void*> data_ptrs = {};
  data_ptrs.push_back(preprocessing_stage_->GetPreprocessedImageData());
  inference_stage_->SetInputs(data_ptrs);
  if (inference_stage_->Run() != kTfLiteOk) return kTfLiteError;
  // Accuracy Eval.
  if (accuracy_eval_stage_) {
    if (ground_truth_label_.empty()) {
      LOG(ERROR) << "Ground truth label not provided";
      return kTfLiteError;
    }
    accuracy_eval_stage_->SetEvalInputs(inference_stage_->GetOutputs()->at(0),
                                        &ground_truth_label_);
    if (accuracy_eval_stage_->Run() != kTfLiteOk) return kTfLiteError;
  }

  return kTfLiteOk;
}

EvaluationStageMetrics ImageClassificationStage::LatestMetrics() {
  EvaluationStageMetrics metrics;
  auto* classification_metrics =
      metrics.mutable_process_metrics()->mutable_image_classification_metrics();

  *classification_metrics->mutable_pre_processing_latency() =
      preprocessing_stage_->LatestMetrics().process_metrics().total_latency();
  EvaluationStageMetrics inference_metrics = inference_stage_->LatestMetrics();
  *classification_metrics->mutable_inference_latency() =
      inference_metrics.process_metrics().total_latency();
  *classification_metrics->mutable_inference_metrics() =
      inference_metrics.process_metrics().tflite_inference_metrics();
  if (accuracy_eval_stage_) {
    *classification_metrics->mutable_topk_accuracy_metrics() =
        accuracy_eval_stage_->LatestMetrics()
            .process_metrics()
            .topk_accuracy_metrics();
  }
  metrics.set_num_runs(inference_metrics.num_runs());
  return metrics;
}

TfLiteStatus FilterDenyListedImages(const std::string& denylist_file_path,
                                    std::vector<ImageLabel>* image_labels) {
  if (!denylist_file_path.empty()) {
    std::vector<std::string> lines;
    if (!tflite::evaluation::ReadFileLines(denylist_file_path, &lines)) {
      LOG(ERROR) << "Could not read: " << denylist_file_path;
      return kTfLiteError;
    }
    std::vector<int> denylist_ids;
    denylist_ids.reserve(lines.size());
    // Populate denylist_ids with indices of images.
    std::transform(lines.begin(), lines.end(), std::back_inserter(denylist_ids),
                   [](const std::string& val) { return std::stoi(val) - 1; });

    std::vector<ImageLabel> filtered_images;
    std::sort(denylist_ids.begin(), denylist_ids.end());
    const size_t size_post_filtering =
        image_labels->size() - denylist_ids.size();
    filtered_images.reserve(size_post_filtering);
    int denylist_index = 0;
    for (int image_index = 0; image_index < image_labels->size();
         image_index++) {
      if (denylist_index < denylist_ids.size() &&
          denylist_ids[denylist_index] == image_index) {
        denylist_index++;
        continue;
      }
      filtered_images.push_back((*image_labels)[image_index]);
    }

    if (filtered_images.size() != size_post_filtering) {
      LOG(ERROR) << "Invalid number of filtered images";
      return kTfLiteError;
    }
    *image_labels = filtered_images;
  }
  return kTfLiteOk;
}

}  // namespace evaluation
}  // namespace tflite
