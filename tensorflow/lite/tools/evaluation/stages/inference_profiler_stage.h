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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_INFERENCE_PROFILER_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_INFERENCE_PROFILER_STAGE_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "xla/tsl/util/stats_calculator.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"

namespace tflite {
namespace evaluation {

// An EvaluationStage to profile a custom TFLite inference config by comparing
// performance in two settings:
// 1. User-defined TfliteInferenceParams (The 'test' setting)
// 2. Default TfliteInferenceParams (The 'reference' setting)
// The latter essentially implies single-threaded CPU execution.
class InferenceProfilerStage : public EvaluationStage {
 public:
  explicit InferenceProfilerStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  TfLiteStatus Init() override { return Init(nullptr); }
  TfLiteStatus Init(const DelegateProviders* delegate_providers);

  // New Gaussian random data is used as input for each Run.
  TfLiteStatus Run() override;

  EvaluationStageMetrics LatestMetrics() override;

 private:
  std::unique_ptr<TfliteInferenceStage> reference_stage_;
  std::unique_ptr<TfliteInferenceStage> test_stage_;

  const TfLiteModelInfo* model_info_;
  std::vector<int64_t> input_num_elements_;
  std::vector<int64_t> output_num_elements_;

  // One Stat for each model output.
  std::vector<tsl::Stat<float>> error_stats_;

  // One of the following vectors will be populated based on model_input_type_,
  // and used as the input for the underlying model.
  std::vector<std::vector<float>> float_tensors_;
  std::vector<std::vector<int8_t>> int8_tensors_;
  std::vector<std::vector<uint8_t>> uint8_tensors_;
  std::vector<std::vector<uint16_t>> float16_tensors_;
  std::vector<std::vector<int64_t>> int64_tensors_;
};

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_INFERENCE_PROFILER_STAGE_H_
