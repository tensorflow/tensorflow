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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_INPUT_PIPELINE_ANALYSIS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_INPUT_PIPELINE_ANALYSIS_H_

#include "google/protobuf/any.pb.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"

namespace tensorflow {
namespace profiler {

// Computes the summary of step time in milliseconds.
StepSummary ComputeStepTimeSummaryInMs(
    const ::tensorflow::protobuf::RepeatedPtrField<PerCoreStepInfo>&
        grouped_by_step);

void GenerateHostResult(const OpMetricsDb& host_tf_metrics_db,
                        InputPipelineAnalysisResult* result);

InputPipelineAnalysisRecommendation GenerateRecommendation();

// Returns the performance bottleneck of the program executed.
BottleneckAnalysis ComputeBottleneckAnalysis(
    const ::tensorflow::protobuf::RepeatedPtrField<::google::protobuf::Any>&
        any_step_details);

InputPipelineAnalysisResult ConvertOpStatsToInputPipelineAnalysis(
    const OpStats& op_stats, const HardwareType& hardware_type);

void InfeedAnalysis(double infeed_percent, string* input_classification,
                    string* input_statement);

string GetSummaryNextStep(absl::string_view input_classification,
                          const InputTimeBreakdown& breakdown);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_INPUT_PIPELINE_ANALYSIS_H_
