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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_PREPROCESSING_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_PREPROCESSING_STAGE_H_

#include <stdint.h>

#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/preprocessing_steps.pb.h"

namespace tflite {
namespace evaluation {

// EvaluationStage to read contents of an image and preprocess it for inference.
// Currently only supports JPEGs.
class ImagePreprocessingStage : public EvaluationStage {
 public:
  explicit ImagePreprocessingStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  TfLiteStatus Init() override;

  TfLiteStatus Run() override;

  EvaluationStageMetrics LatestMetrics() override;

  ~ImagePreprocessingStage() override {}

  // Call before Run().
  void SetImagePath(std::string* image_path) { image_path_ = image_path; }

  // Provides preprocessing output.
  void* GetPreprocessedImageData();

 private:
  std::string* image_path_ = nullptr;
  TfLiteType output_type_;
  tensorflow::Stat<int64_t> latency_stats_;

  // One of the following 3 vectors will be populated based on output_type_.
  std::vector<float> float_preprocessed_image_;
  std::vector<int8_t> int8_preprocessed_image_;
  std::vector<uint8_t> uint8_preprocessed_image_;
};

// Helper class to build a new ImagePreprocessingParams.
class ImagePreprocessingConfigBuilder {
 public:
  ImagePreprocessingConfigBuilder(const std::string& name,
                                  TfLiteType output_type) {
    config_.set_name(name);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->set_output_type(static_cast<int>(output_type));
  }

  // Adds a cropping step with cropping fraction.
  void AddCroppingStep(float cropping_fraction) {
    ImagePreprocessingStepParams params;
    params.mutable_cropping_params()->set_cropping_fraction(cropping_fraction);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a cropping step with target size.
  void AddCroppingStep(uint32_t width, uint32_t height) {
    ImagePreprocessingStepParams params;
    params.mutable_cropping_params()->mutable_target_size()->set_height(height);
    params.mutable_cropping_params()->mutable_target_size()->set_width(width);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a square cropping step.
  void AddSquareCroppingStep() {
    ImagePreprocessingStepParams params;
    params.mutable_cropping_params()->set_square_cropping(true);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a resizing step.
  void AddResizingStep(uint32_t width, uint32_t height,
                       bool aspect_preserving) {
    ImagePreprocessingStepParams params;
    params.mutable_resizing_params()->set_aspect_preserving(aspect_preserving);
    params.mutable_resizing_params()->mutable_target_size()->set_height(height);
    params.mutable_resizing_params()->mutable_target_size()->set_width(width);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a padding step.
  void AddPaddingStep(uint32_t width, uint32_t height, int value) {
    ImagePreprocessingStepParams params;
    params.mutable_padding_params()->mutable_target_size()->set_height(height);
    params.mutable_padding_params()->mutable_target_size()->set_width(width);
    params.mutable_padding_params()->set_padding_value(value);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a square padding step.
  void AddSquarePaddingStep(int value) {
    ImagePreprocessingStepParams params;
    params.mutable_padding_params()->set_square_padding(true);
    params.mutable_padding_params()->set_padding_value(value);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a subtracting means step.
  void AddPerChannelNormalizationStep(float r_mean, float g_mean, float b_mean,
                                      float scale) {
    ImagePreprocessingStepParams params;
    params.mutable_normalization_params()->mutable_means()->set_r_mean(r_mean);
    params.mutable_normalization_params()->mutable_means()->set_g_mean(g_mean);
    params.mutable_normalization_params()->mutable_means()->set_b_mean(b_mean);
    params.mutable_normalization_params()->set_scale(scale);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a normalization step.
  void AddNormalizationStep(float mean, float scale) {
    ImagePreprocessingStepParams params;
    params.mutable_normalization_params()->set_channelwise_mean(mean);
    params.mutable_normalization_params()->set_scale(scale);
    config_.mutable_specification()
        ->mutable_image_preprocessing_params()
        ->mutable_steps()
        ->Add(std::move(params));
  }

  // Adds a normalization step with default value.
  void AddDefaultNormalizationStep() {
    switch (
        config_.specification().image_preprocessing_params().output_type()) {
      case kTfLiteFloat32:
        AddNormalizationStep(127.5, 1.0 / 127.5);
        break;
      case kTfLiteUInt8:
        break;
      case kTfLiteInt8:
        AddNormalizationStep(128.0, 1.0);
        break;
      default:
        LOG(ERROR) << "Type not supported";
        break;
    }
  }

  EvaluationStageConfig build() { return std::move(config_); }

 private:
  EvaluationStageConfig config_;
};

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_PREPROCESSING_STAGE_H_
