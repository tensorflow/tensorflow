/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/evaluation/tasks/task_executor_c_api.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"
#include "tensorflow/lite/tools/logging.h"

extern "C" {

struct TfLiteEvaluationMetrics {
  tflite::evaluation::EvaluationStageMetrics metrics;
};

struct TfLiteEvaluationTask {
  std::unique_ptr<tflite::evaluation::TaskExecutor> task_executor;
};

int32_t TfLiteEvaluationMetricsGetNumRuns(
    const TfLiteEvaluationMetrics* metrics) {
  return metrics->metrics.num_runs();
}

extern TfLiteEvaluationMetricsLatency TfLiteEvaluationMetricsGetTestLatency(
    const TfLiteEvaluationMetrics* metrics) {
  const auto& latency = metrics->metrics.process_metrics()
                            .inference_profiler_metrics()
                            .test_latency();
  return {latency.last_us(), latency.max_us(), latency.min_us(),
          latency.sum_us(),  latency.avg_us(), latency.std_deviation_us()};
}

extern TfLiteEvaluationMetricsLatency
TfLiteEvaluationMetricsGetReferenceLatency(
    const TfLiteEvaluationMetrics* metrics) {
  const auto& latency = metrics->metrics.process_metrics()
                            .inference_profiler_metrics()
                            .reference_latency();
  return {latency.last_us(), latency.max_us(), latency.min_us(),
          latency.sum_us(),  latency.avg_us(), latency.std_deviation_us()};
}

extern size_t TfLiteEvaluationMetricsGetOutputErrorCount(
    const TfLiteEvaluationMetrics* metrics) {
  return metrics->metrics.process_metrics()
      .inference_profiler_metrics()
      .output_errors()
      .size();
}

extern TfLiteEvaluationMetricsAccuracy TfLiteEvaluationMetricsGetOutputError(
    const TfLiteEvaluationMetrics* metrics, int32_t output_error_index) {
  int32_t output_count = TfLiteEvaluationMetricsGetOutputErrorCount(metrics);
  if (output_error_index < 0 || output_error_index >= output_count) {
    return {};
  }
  const auto& accuracy = metrics->metrics.process_metrics()
                             .inference_profiler_metrics()
                             .output_errors()
                             .at(output_error_index);
  return {accuracy.max_value(), accuracy.min_value(), accuracy.avg_value(),
          accuracy.std_deviation()};
}

TfLiteEvaluationTask* TfLiteEvaluationTaskCreate() {
  auto task_executor = tflite::evaluation::CreateTaskExecutor();
  return new TfLiteEvaluationTask{std::move(task_executor)};
}

TfLiteEvaluationMetrics* TfLiteEvaluationTaskRunWithArgs(
    TfLiteEvaluationTask* evaluation_task, int argc, char** argv) {
  const auto metrics_optional =
      evaluation_task->task_executor->Run(&argc, argv);
  if (!metrics_optional.has_value()) {
    TFLITE_LOG(ERROR) << "Failed to run the task evaluation!";
    return new TfLiteEvaluationMetrics{};
  }
  return new TfLiteEvaluationMetrics{std::move(metrics_optional.value())};
}

}  // extern "C"
