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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_C_API_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_C_API_H_

#include <cstddef>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::evaluation::LatencyMetrics proto.
// -----------------------------------------------------------------------------

typedef struct TfLiteEvaluationMetricsLatency {
  // Latency for the last Run.
  int64_t last_us;
  // Maximum latency observed for any Run.
  int64_t max_us;
  // Minimum latency observed for any Run.
  int64_t min_us;
  // Sum of all Run latencies.
  int64_t sum_us;
  // Average latency across all Runs.
  double avg_us;
  // Standard deviation for latency across all Runs.
  int64_t std_deviation_us;
} TfLiteEvaluationMetricsLatency;

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::evaluation::AccuracyMetrics proto.
// -----------------------------------------------------------------------------

typedef struct TfLiteEvaluationMetricsAccuracy {
  // Maximum value observed for any Run.
  float max_value;
  // Minimum value observed for any Run.
  float min_value;
  // Average value across all Runs.
  double avg_value;
  // Standard deviation across all Runs.
  float std_deviation;
} TfLiteEvaluationMetricsAccuracy;

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::evaluation::EvaluationStageMetrics type.
// -----------------------------------------------------------------------------
typedef struct TfLiteEvaluationMetrics TfLiteEvaluationMetrics;

extern int32_t TfLiteEvaluationMetricsGetNumRuns(
    const TfLiteEvaluationMetrics* metrics);

extern TfLiteEvaluationMetricsLatency TfLiteEvaluationMetricsGetTestLatency(
    const TfLiteEvaluationMetrics* metrics);

extern TfLiteEvaluationMetricsLatency
TfLiteEvaluationMetricsGetReferenceLatency(
    const TfLiteEvaluationMetrics* metrics);

extern size_t TfLiteEvaluationMetricsGetOutputErrorCount(
    const TfLiteEvaluationMetrics* metrics);

extern TfLiteEvaluationMetricsAccuracy TfLiteEvaluationMetricsGetOutputError(
    const TfLiteEvaluationMetrics* metrics, int32_t output_error_index);

// -----------------------------------------------------------------------------
// C APIs corresponding to tflite::evaluation::TaskExecutor type.
// -----------------------------------------------------------------------------
typedef struct TfLiteEvaluationTask TfLiteEvaluationTask;

extern TfLiteEvaluationTask* TfLiteEvaluationTaskCreate();

extern TfLiteEvaluationMetrics* TfLiteEvaluationTaskRunWithArgs(
    TfLiteEvaluationTask* evaluation_task, int argc, char** argv);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_TASKS_TASK_EXECUTOR_C_API_H_
