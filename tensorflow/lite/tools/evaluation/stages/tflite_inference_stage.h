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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_TFLITE_INFERENCE_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_TFLITE_INFERENCE_STAGE_H_

#include <stdint.h>

#include <vector>

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

namespace tflite {
namespace evaluation {

struct TfLiteModelInfo {
  std::vector<const TfLiteTensor*> inputs;
  std::vector<const TfLiteTensor*> outputs;
};

// EvaluationStage to run inference using TFLite.
class TfliteInferenceStage : public EvaluationStage {
 public:
  explicit TfliteInferenceStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  TfLiteStatus Init() override { return Init(nullptr); }
  TfLiteStatus Init(const DelegateProviders* delegate_providers);

  TfLiteStatus Run() override;

  // EvaluationStageMetrics.num_runs denotes the number of inferences run.
  EvaluationStageMetrics LatestMetrics() override;

  ~TfliteInferenceStage() override {}

  // Call before Run().
  // This class does not take ownership of raw_input_ptrs.
  void SetInputs(const std::vector<void*>& raw_input_ptrs) {
    inputs_ = &raw_input_ptrs;
  }

  // Resize input tensors with given shapes.
  TfLiteStatus ResizeInputs(const std::vector<std::vector<int>>& shapes);

  // Applies provided delegate to the underlying TFLite Interpreter.
  TfLiteStatus ApplyCustomDelegate(Interpreter::TfLiteDelegatePtr delegate);

  // Read-only view of a TfliteModelInfo. TfliteInferenceStage retains
  // ownership.
  // Only available after Init is done.
  const TfLiteModelInfo* GetModelInfo() const { return &model_info_; }

  // Provides a read-only view to the model's output tensor(s). Retains
  // ownership of object.
  const std::vector<void*>* GetOutputs() const { return &outputs_; }

 private:
  // Sets model_info_ & outputs_ after interpreter tensors are (re)allocated.
  void UpdateModelInfo();

  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<ops::builtin::BuiltinOpResolver> resolver_;
  std::unique_ptr<Interpreter> interpreter_;
  std::vector<Interpreter::TfLiteDelegatePtr> delegates_;

  TfLiteModelInfo model_info_;
  const std::vector<void*>* inputs_ = nullptr;
  std::vector<void*> outputs_;

  tensorflow::Stat<int64_t> latency_stats_;
};

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_TFLITE_INFERENCE_STAGE_H_
