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
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);

// Version with Weak linker attribute doing nothing: if someone links this
// library with another definition of this function (presumably to actually
// register custom ops), that version will be used instead.
void ABSL_ATTRIBUTE_WEAK
RegisterSelectedOps(::tflite::MutableOpResolver* resolver) {}

namespace tflite {
namespace evaluation {
namespace {

TfLiteModelInfo GetTfliteModelInfo(const Interpreter& interpreter) {
  TfLiteModelInfo model_info;
  for (int i : interpreter.inputs()) {
    model_info.inputs.push_back(interpreter.tensor(i));
  }
  for (int i : interpreter.outputs()) {
    model_info.outputs.push_back(interpreter.tensor(i));
  }
  return model_info;
}

}  // namespace

void TfliteInferenceStage::UpdateModelInfo() {
  model_info_ = GetTfliteModelInfo(*interpreter_);

  outputs_.clear();
  outputs_.reserve(interpreter_->outputs().size());
  for (int i : interpreter_->outputs()) {
    TfLiteTensor* tensor = interpreter_->tensor(i);
    outputs_.push_back(tensor->data.raw);
  }
}

TfLiteStatus TfliteInferenceStage::ResizeInputs(
    const std::vector<std::vector<int>>& shapes) {
  const std::vector<int>& interpreter_inputs = interpreter_->inputs();
  if (interpreter_inputs.size() != shapes.size()) {
    LOG(ERROR) << "New shape is not compatible";
    return kTfLiteError;
  }

  for (int j = 0; j < shapes.size(); ++j) {
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (t->type != kTfLiteString) {
      TF_LITE_ENSURE_STATUS(interpreter_->ResizeInputTensor(i, shapes[j]));
    }
  }

  TF_LITE_ENSURE_STATUS(interpreter_->AllocateTensors());
  UpdateModelInfo();
  return kTfLiteOk;
}

TfLiteStatus TfliteInferenceStage::ApplyCustomDelegate(
    Interpreter::TfLiteDelegatePtr delegate) {
  if (!interpreter_) {
    LOG(ERROR) << "Stage not initialized before calling ApplyCustomDelegate";
    return kTfLiteError;
  }
  // Skip if delegate is a nullptr.
  if (!delegate) {
    LOG(WARNING)
        << "Tried to apply null TfLiteDelegatePtr to TfliteInferenceStage";
    return kTfLiteOk;
  }
  delegates_.push_back(std::move(delegate));
  TF_LITE_ENSURE_STATUS(
      interpreter_->ModifyGraphWithDelegate(delegates_.back().get()));
  UpdateModelInfo();
  return kTfLiteOk;
}

TfLiteStatus TfliteInferenceStage::Init(
    const DelegateProviders* delegate_providers) {
  if (!config_.specification().has_tflite_inference_params()) {
    LOG(ERROR) << "TfliteInferenceParams not provided";
    return kTfLiteError;
  }
  auto& params = config_.specification().tflite_inference_params();
  if (!params.has_model_file_path()) {
    LOG(ERROR) << "Model path not provided";
    return kTfLiteError;
  }
  std::ifstream model_check(params.model_file_path());
  if (!model_check.good()) {
    LOG(ERROR) << "Model file not found";
    return kTfLiteError;
  }

  model_ = FlatBufferModel::BuildFromFile(params.model_file_path().c_str());

  bool apply_default_delegates = true;
  if (delegate_providers != nullptr) {
    const auto& provider_params = delegate_providers->GetAllParams();
    // When --use_xnnpack is explicitly set to false, to honor this, skip
    // applying the XNNPACK delegate by default in TfLite runtime.
    if (provider_params.HasParam("use_xnnpack") &&
        provider_params.HasValueSet<bool>("use_xnnpack") &&
        !provider_params.Get<bool>("use_xnnpack")) {
      apply_default_delegates = false;
    }
  }

  if (apply_default_delegates) {
    resolver_ = std::make_unique<ops::builtin::BuiltinOpResolver>();
  } else {
    resolver_ = std::make_unique<
        ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>();
  }
  RegisterSelectedOps(resolver_.get());
  InterpreterBuilder(*model_, *resolver_)(&interpreter_);
  if (!interpreter_) {
    LOG(ERROR) << "Could not build interpreter";
    return kTfLiteError;
  }
  interpreter_->SetNumThreads(params.num_threads());

  if (!delegate_providers) {
    std::string error_message;
    auto delegate = CreateTfLiteDelegate(params, &error_message);
    if (delegate) {
      delegates_.push_back(std::move(delegate));
      LOG(INFO) << "Successfully created "
                << params.Delegate_Name(params.delegate()) << " delegate.";
    } else {
      LOG(WARNING) << error_message;
    }
  } else {
    auto delegates = delegate_providers->CreateAllDelegates(params);
    for (auto& one : delegates) delegates_.push_back(std::move(one.delegate));
  }

  for (int i = 0; i < delegates_.size(); ++i) {
    if (interpreter_->ModifyGraphWithDelegate(delegates_[i].get()) !=
        kTfLiteOk) {
      LOG(FATAL) << "Failed to apply delegate " << i;
    }
  }
  interpreter_->AllocateTensors();
  UpdateModelInfo();

  return kTfLiteOk;
}

TfLiteStatus TfliteInferenceStage::Run() {
  if (!inputs_) {
    LOG(ERROR) << "Input data not set";
    return kTfLiteError;
  }

  // Copy input data.
  for (int i = 0; i < interpreter_->inputs().size(); ++i) {
    TfLiteTensor* tensor = interpreter_->tensor(interpreter_->inputs()[i]);
    tensor->data.raw = static_cast<char*>(inputs_->at(i));
  }

  // Invoke.
  auto& params = config_.specification().tflite_inference_params();
  for (int i = 0; i < params.invocations_per_run(); ++i) {
    int64_t start_us = profiling::time::NowMicros();
    if (interpreter_->Invoke() != kTfLiteOk) {
      LOG(ERROR) << "TFLite interpreter failed to invoke at run " << i;
      return kTfLiteError;
    }
    latency_stats_.UpdateStat(profiling::time::NowMicros() - start_us);
  }

  return kTfLiteOk;
}

EvaluationStageMetrics TfliteInferenceStage::LatestMetrics() {
  auto& params = config_.specification().tflite_inference_params();
  EvaluationStageMetrics metrics;
  auto* latency_metrics =
      metrics.mutable_process_metrics()->mutable_total_latency();
  latency_metrics->set_last_us(latency_stats_.newest());
  latency_metrics->set_max_us(latency_stats_.max());
  latency_metrics->set_min_us(latency_stats_.min());
  latency_metrics->set_sum_us(latency_stats_.sum());
  latency_metrics->set_avg_us(latency_stats_.avg());
  latency_metrics->set_std_deviation_us(latency_stats_.std_deviation());
  metrics.set_num_runs(
      static_cast<int>(latency_stats_.count() / params.invocations_per_run()));
  auto* inference_metrics =
      metrics.mutable_process_metrics()->mutable_tflite_inference_metrics();
  inference_metrics->set_num_inferences(latency_stats_.count());
  return metrics;
}

}  // namespace evaluation
}  // namespace tflite
