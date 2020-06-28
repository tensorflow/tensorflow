/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <fstream>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.h"
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kNumRunsFlag[] = "num_runs";
constexpr char kInterpreterThreadsFlag[] = "num_interpreter_threads";
constexpr char kDelegateFlag[] = "delegate";

class InferenceDiff : public TaskExecutor {
 public:
  InferenceDiff() : num_runs_(50), num_interpreter_threads_(1) {}
  ~InferenceDiff() override {}

 protected:
  std::vector<Flag> GetFlags() final;

  // If the run is successful, the latest metrics will be returned.
  absl::optional<EvaluationStageMetrics> RunImpl() final;

 private:
  void OutputResult(const EvaluationStageMetrics& latest_metrics) const;
  std::string model_file_path_;
  std::string output_file_path_;
  std::string delegate_;
  int num_runs_;
  int num_interpreter_threads_;
};

std::vector<Flag> InferenceDiff::GetFlags() {
  // Command Line Flags.
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path_,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(kOutputFilePathFlag, &output_file_path_,
                               "File to output metrics proto to."),
      tflite::Flag::CreateFlag(kNumRunsFlag, &num_runs_,
                               "Number of runs of test & reference inference "
                               "each. Default value: 50"),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads_,
          "Number of interpreter threads to use for test inference."),
      tflite::Flag::CreateFlag(
          kDelegateFlag, &delegate_,
          "Delegate to use for test inference, if available. "
          "Must be one of {'nnapi', 'gpu', 'hexagon', 'xnnpack'}"),
  };

  return flag_list;
}

absl::optional<EvaluationStageMetrics> InferenceDiff::RunImpl() {
  // Initialize evaluation stage.
  EvaluationStageConfig eval_config;
  eval_config.set_name("inference_profiling");
  auto* inference_params =
      eval_config.mutable_specification()->mutable_tflite_inference_params();
  inference_params->set_model_file_path(model_file_path_);
  inference_params->set_num_threads(num_interpreter_threads_);
  // This ensures that latency measurement isn't hampered by the time spent in
  // generating random data.
  inference_params->set_invocations_per_run(3);
  inference_params->set_delegate(ParseStringToDelegateType(delegate_));
  if (!delegate_.empty() &&
      inference_params->delegate() == TfliteInferenceParams::NONE) {
    TFLITE_LOG(WARN) << "Unsupported TFLite delegate: " << delegate_;
    return absl::nullopt;
  }

  InferenceProfilerStage eval(eval_config);
  if (eval.Init(&delegate_providers_) != kTfLiteOk) return absl::nullopt;

  // Run inference & check diff for specified number of runs.
  for (int i = 0; i < num_runs_; ++i) {
    if (eval.Run() != kTfLiteOk) return absl::nullopt;
  }

  const auto latest_metrics = eval.LatestMetrics();
  OutputResult(latest_metrics);
  return absl::make_optional(latest_metrics);
}

void InferenceDiff::OutputResult(
    const EvaluationStageMetrics& latest_metrics) const {
  // Output latency & diff metrics.
  if (!output_file_path_.empty()) {
    std::ofstream metrics_ofile;
    metrics_ofile.open(output_file_path_, std::ios::out);
    metrics_ofile << latest_metrics.SerializeAsString();
    metrics_ofile.close();
  }

  TFLITE_LOG(INFO) << "Num evaluation runs: " << latest_metrics.num_runs();
  const auto& metrics =
      latest_metrics.process_metrics().inference_profiler_metrics();
  const auto& ref_latency = metrics.reference_latency();
  TFLITE_LOG(INFO) << "Reference run latency: avg=" << ref_latency.avg_us()
                   << "(us), std_dev=" << ref_latency.std_deviation_us()
                   << "(us)";
  const auto& test_latency = metrics.test_latency();
  TFLITE_LOG(INFO) << "Test run latency: avg=" << test_latency.avg_us()
                   << "(us), std_dev=" << test_latency.std_deviation_us()
                   << "(us)";
  const auto& output_errors = metrics.output_errors();
  for (int i = 0; i < output_errors.size(); ++i) {
    const auto& error = output_errors.at(i);
    TFLITE_LOG(INFO) << "OutputDiff[" << i
                     << "]: avg_error=" << error.avg_value()
                     << ", std_dev=" << error.std_deviation();
  }
}

std::unique_ptr<TaskExecutor> CreateTaskExecutor() {
  return std::unique_ptr<TaskExecutor>(new InferenceDiff());
}

}  // namespace evaluation
}  // namespace tflite
