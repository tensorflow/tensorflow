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

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

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

  ~ImagePreprocessingStage() {}

  // Call before Run().
  void SetImagePath(std::string* image_path) { image_path_ = image_path; }

  // Provides preprocessing output.
  void* GetPreprocessedImageData();

 private:
  std::string* image_path_ = nullptr;
  float cropping_fraction_;
  float input_mean_value_;
  float scale_;
  int total_size_;
  TfLiteType output_type_;
  tensorflow::Stat<int64_t> latency_stats_;

  // One of the following 3 vectors will be populated based on output_type_.
  std::vector<float> float_preprocessed_image_;
  std::vector<int8_t> int8_preprocessed_image_;
  std::vector<uint8_t> uint8_preprocessed_image_;
};

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_PREPROCESSING_STAGE_H_
