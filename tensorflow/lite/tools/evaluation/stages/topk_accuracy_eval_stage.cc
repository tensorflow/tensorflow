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
#include "tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.h"

#include <stdint.h>
#include <numeric>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

std::vector<int> GetTopKIndices(const std::vector<float>& values, int k) {
  std::vector<int> indices(values.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&values](int a, int b) { return values[a] > values[b]; });
  indices.resize(k);
  return indices;
}

}  // namespace

TfLiteStatus TopkAccuracyEvalStage::Init() {
  num_runs_ = 0;
  auto& params = config_.specification().topk_accuracy_eval_params();
  if (!params.has_k()) {
    LOG(ERROR) << "Value of k not provided for TopkAccuracyEvalStage";
    return kTfLiteError;
  }
  accuracy_counts_ = std::vector<int>(params.k(), 0);

  if (ground_truth_labels_.empty()) {
    LOG(ERROR) << "Ground-truth labels are empty";
    return kTfLiteError;
  }
  num_total_labels_ = ground_truth_labels_.size();
  if (params.k() > num_total_labels_) {
    LOG(ERROR) << "k is too large";
    return kTfLiteError;
  }

  if (!model_output_shape_) {
    LOG(ERROR) << "Model output details not correctly set";
    return kTfLiteError;
  }
  // Ensure model output is of shape (1, num_total_labels_).
  if (!(model_output_shape_->size == 2) ||
      !(model_output_shape_->data[0] == 1) ||
      !(model_output_shape_->data[1] == num_total_labels_)) {
    LOG(ERROR) << "Invalid model_output_shape_";
    return kTfLiteError;
  }
  if (model_output_type_ != kTfLiteFloat32 &&
      model_output_type_ != kTfLiteUInt8 && model_output_type_ != kTfLiteInt8) {
    LOG(ERROR) << "model_output_type_ not supported";
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus TopkAccuracyEvalStage::Run() {
  if (!model_output_) {
    LOG(ERROR) << "model_output_ not set correctly";
    return kTfLiteError;
  }
  if (!ground_truth_label_) {
    LOG(ERROR) << "ground_truth_label_ not provided";
    return kTfLiteError;
  }
  auto& params = config_.specification().topk_accuracy_eval_params();

  std::vector<float> probabilities;
  probabilities.reserve(num_total_labels_);
  if (model_output_type_ == kTfLiteFloat32) {
    auto probs = static_cast<float*>(model_output_);
    for (size_t i = 0; i < num_total_labels_; i++) {
      probabilities.push_back(probs[i]);
    }
  } else if (model_output_type_ == kTfLiteUInt8) {
    auto probs = static_cast<uint8_t*>(model_output_);
    for (size_t i = 0; i < num_total_labels_; i++) {
      probabilities.push_back(probs[i]);
    }
  } else if (model_output_type_ == kTfLiteInt8) {
    auto probs = static_cast<int8_t*>(model_output_);
    for (size_t i = 0; i < num_total_labels_; i++) {
      probabilities.push_back(probs[i]);
    }
  }

  std::vector<int> top_k = GetTopKIndices(probabilities, params.k());
  int ground_truth_index = GroundTruthIndex(*ground_truth_label_);
  if (ground_truth_index < 0) {
    LOG(ERROR) << "Invalid ground truth label";
    return kTfLiteError;
  }
  UpdateCounts(top_k, ground_truth_index);
  return kTfLiteOk;
}

EvaluationStageMetrics TopkAccuracyEvalStage::LatestMetrics() {
  EvaluationStageMetrics metrics;
  if (num_runs_ == 0) return metrics;

  metrics.set_num_runs(num_runs_);
  auto* topk_metrics =
      metrics.mutable_process_metrics()->mutable_topk_accuracy_metrics();
  for (const auto& count : accuracy_counts_) {
    topk_metrics->add_topk_accuracy_percentages(static_cast<float>(count) /
                                                num_runs_);
  }
  return metrics;
}

void TopkAccuracyEvalStage::UpdateCounts(const std::vector<int>& topk_indices,
                                         int ground_truth_index) {
  for (size_t i = 0; i < topk_indices.size(); ++i) {
    if (ground_truth_index == topk_indices[i]) {
      for (size_t j = i; j < topk_indices.size(); j++) {
        accuracy_counts_[j] += 1;
      }
      break;
    }
  }
  num_runs_++;
}

int TopkAccuracyEvalStage::GroundTruthIndex(const std::string& label) const {
  auto index = std::find(ground_truth_labels_.cbegin(),
                         ground_truth_labels_.cend(), label);
  if (index == ground_truth_labels_.end()) {
    LOG(ERROR) << "Invalid label: " << label;
    return -1;
  }
  return std::distance(ground_truth_labels_.cbegin(), index);
}

}  // namespace evaluation
}  // namespace tflite
