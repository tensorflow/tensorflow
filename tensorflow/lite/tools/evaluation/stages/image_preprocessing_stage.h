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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

namespace tflite {
namespace evaluation {

// EvaluationStage to read contents of an image and preprocess it for inference.
// Currently only supports JPEGs.
// Input TAGs (Object Class): IMAGE_PATH (std::string*)
// Output TAGs (Object Class): PREPROCESSED_IMAGE (pointer to image based on
//                                                 output_type)
// For more information on TAGs, please see EvaluationStageConfig proto.
class ImagePreprocessingStage : public EvaluationStage {
 public:
  explicit ImagePreprocessingStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  bool Run(absl::flat_hash_map<std::string, void*>& object_map) override;

  EvaluationStageMetrics LatestMetrics() override;

  ~ImagePreprocessingStage() {}

 protected:
  bool DoInit(absl::flat_hash_map<std::string, void*>& object_map) override;

  std::vector<std::string> GetInitializerTags() override { return {}; }
  std::vector<std::string> GetInputTags() override { return {kImagePathTag}; }
  std::vector<std::string> GetOutputTags() override {
    return {kPreprocessedImageTag};
  }

 private:
  float cropping_fraction_;
  float input_mean_value_;
  float scale_;
  int image_height_, image_width_, total_size_;
  TfLiteType output_type_;
  tensorflow::Stat<int64_t> latency_stats_;

  // One of the following 3 vectors will be populated based on output_type_.
  std::vector<float> float_preprocessed_image_;
  std::vector<int8_t> int8_preprocessed_image_;
  std::vector<uint8_t> uint8_preprocessed_image_;

  static const char kImagePathTag[];
  static const char kPreprocessedImageTag[];
};

DECLARE_FACTORY(ImagePreprocessingStage);

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_IMAGE_PREPROCESSING_STAGE_H_
