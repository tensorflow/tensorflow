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
#include "tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.h"

#include <cmath>
#include <limits>
#include <random>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

// Parameters for a simple Gaussian distribution to generate values roughly in
// [0, 1).
constexpr float kGaussianFloatMean = 0.5;
constexpr float kGaussianStdDev = 1.0 / 3;

// TODO(b/131420973): Reconcile these with the functionality in
// testing/kernel_test.
template <typename T>
void GenerateRandomGaussianData(int64_t num_elements, float min, float max,
                                std::vector<T>* data) {
  data->clear();
  data->reserve(num_elements);

  static std::normal_distribution<double> distribution(kGaussianFloatMean,
                                                       kGaussianStdDev);
  static std::default_random_engine generator;
  for (int i = 0; i < num_elements; ++i) {
    auto rand_n = distribution(generator);
    while (rand_n < 0 || rand_n >= 1) {
      rand_n = distribution(generator);
    }
    auto rand_float = min + (max - min) * static_cast<float>(rand_n);
    data->push_back(static_cast<T>(rand_float));
  }
}

template <typename T>
float CalculateAverageError(T* reference, T* test, int64_t num_elements) {
  float error = 0;

  for (int i = 0; i < num_elements; i++) {
    float test_value = static_cast<float>(test[i]);
    float reference_value = static_cast<float>(reference[i]);
    error += std::abs(test_value - reference_value);
  }
  error /= num_elements;

  return error;
}

}  // namespace

TfLiteStatus InferenceProfilerStage::Init() {
  // Initialize TfliteInferenceStage with the user-provided
  // TfliteInferenceParams.
  test_stage_.reset(new TfliteInferenceStage(config_));
  if (test_stage_->Init() != kTfLiteOk) return kTfLiteError;

  // Initialize a reference TfliteInferenceStage that uses the given model &
  // num_runs, but maintains the rest of TfliteInferenceParams to default.
  EvaluationStageConfig reference_config;
  reference_config.set_name("reference_inference");
  auto* params = reference_config.mutable_specification()
                     ->mutable_tflite_inference_params();
  params->set_model_file_path(
      config_.specification().tflite_inference_params().model_file_path());
  params->set_invocations_per_run(
      config_.specification().tflite_inference_params().invocations_per_run());
  reference_stage_.reset(new TfliteInferenceStage(reference_config));
  if (reference_stage_->Init() != kTfLiteOk) return kTfLiteError;

  model_info_ = reference_stage_->GetModelInfo();

  // Preprocess model input metadata for generating random data later.
  for (int i = 0; i < model_info_->inputs.size(); ++i) {
    const TfLiteType model_input_type = model_info_->inputs[i]->type;
    if (model_input_type == kTfLiteUInt8 || model_input_type == kTfLiteInt8 ||
        model_input_type == kTfLiteFloat32) {
    } else {
      LOG(ERROR) << "InferenceProfilerStage only supports float/int8/uint8 "
                    "input types";
      return kTfLiteError;
    }
    auto* input_shape = model_info_->inputs[i]->dims;
    int64_t total_num_elements = 1;
    for (int i = 0; i < input_shape->size; i++) {
      total_num_elements *= input_shape->data[i];
    }
    input_num_elements_.push_back(total_num_elements);
    float_tensors_.emplace_back();
    uint8_tensors_.emplace_back();
    int8_tensors_.emplace_back();
  }
  // Preprocess output metadata for calculating diffs later.
  for (int i = 0; i < model_info_->outputs.size(); ++i) {
    const TfLiteType model_output_type = model_info_->outputs[i]->type;
    if (model_output_type == kTfLiteUInt8 || model_output_type == kTfLiteInt8 ||
        model_output_type == kTfLiteFloat32) {
    } else {
      LOG(ERROR) << "InferenceProfilerStage only supports float/int8/uint8 "
                    "output types";
      return kTfLiteError;
    }
    auto* output_shape = model_info_->outputs[i]->dims;
    int64_t total_num_elements = 1;
    for (int i = 0; i < output_shape->size; i++) {
      total_num_elements *= output_shape->data[i];
    }
    output_num_elements_.push_back(total_num_elements);

    error_stats_.emplace_back();
  }

  return kTfLiteOk;
}

TfLiteStatus InferenceProfilerStage::Run() {
  // Generate random inputs.
  std::vector<void*> input_ptrs;
  for (int i = 0; i < model_info_->inputs.size(); ++i) {
    const TfLiteType model_input_type = model_info_->inputs[i]->type;
    if (model_input_type == kTfLiteUInt8) {
      GenerateRandomGaussianData(
          input_num_elements_[i], std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(), &uint8_tensors_[i]);
      input_ptrs.push_back(uint8_tensors_[i].data());
    } else if (model_input_type == kTfLiteInt8) {
      GenerateRandomGaussianData(
          input_num_elements_[i], std::numeric_limits<int8_t>::min(),
          std::numeric_limits<int8_t>::max(), &int8_tensors_[i]);
      input_ptrs.push_back(int8_tensors_[i].data());
    } else if (model_input_type == kTfLiteFloat32) {
      GenerateRandomGaussianData(input_num_elements_[i], -1, 1,
                                 &(float_tensors_[i]));
      input_ptrs.push_back(float_tensors_[i].data());
    }
  }

  // Run both inference stages.
  test_stage_->SetInputs(input_ptrs);
  reference_stage_->SetInputs(input_ptrs);
  if (test_stage_->Run() != kTfLiteOk) return kTfLiteError;
  if (reference_stage_->Run() != kTfLiteOk) return kTfLiteError;

  // Calculate errors per output vector.
  for (int i = 0; i < model_info_->outputs.size(); ++i) {
    const TfLiteType model_output_type = model_info_->outputs[i]->type;
    void* reference_ptr = reference_stage_->GetOutputs()->at(i);
    void* test_ptr = test_stage_->GetOutputs()->at(i);
    float output_diff = 0;
    if (model_output_type == kTfLiteUInt8) {
      output_diff = CalculateAverageError(static_cast<uint8_t*>(reference_ptr),
                                          static_cast<uint8_t*>(test_ptr),
                                          output_num_elements_[i]);
    } else if (model_output_type == kTfLiteInt8) {
      output_diff = CalculateAverageError(static_cast<int8_t*>(reference_ptr),
                                          static_cast<int8_t*>(test_ptr),
                                          output_num_elements_[i]);
    } else if (model_output_type == kTfLiteFloat32) {
      output_diff = CalculateAverageError(static_cast<float*>(reference_ptr),
                                          static_cast<float*>(test_ptr),
                                          output_num_elements_[i]);
    }
    error_stats_[i].UpdateStat(output_diff);
  }

  return kTfLiteOk;
}

EvaluationStageMetrics InferenceProfilerStage::LatestMetrics() {
  EvaluationStageMetrics metrics;
  const auto& reference_metrics = reference_stage_->LatestMetrics();
  metrics.set_num_runs(reference_metrics.num_runs());
  auto* inference_profiler_metrics =
      metrics.mutable_process_metrics()->mutable_inference_profiler_metrics();

  *inference_profiler_metrics->mutable_reference_latency() =
      reference_metrics.process_metrics().total_latency();
  *inference_profiler_metrics->mutable_test_latency() =
      test_stage_->LatestMetrics().process_metrics().total_latency();

  for (int i = 0; i < error_stats_.size(); ++i) {
    AccuracyMetrics* diff = inference_profiler_metrics->add_output_errors();
    diff->set_avg_value(error_stats_[i].avg());
    diff->set_std_deviation(error_stats_[i].std_deviation());
    diff->set_min_value(error_stats_[i].min());
    // Avoiding the small positive values contained in max() even when avg() ==
    // 0.
    if (error_stats_[i].avg() != 0) {
      diff->set_max_value(error_stats_[i].max());
    } else {
      diff->set_max_value(0);
    }
  }

  return metrics;
}

}  // namespace evaluation
}  // namespace tflite
