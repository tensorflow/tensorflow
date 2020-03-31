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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_OVERVIEW_PAGE_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_OVERVIEW_PAGE_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"

namespace tensorflow {
namespace profiler {

void SetCommonRecommendation(const string& input_classification,
                             const string& input_statement,
                             HardwareType hardware_type,
                             OverviewPageRecommendation* re);

OverviewPageRecommendation ComputeGenericRecommendation(
    const BottleneckAnalysis& bottleneck,
    const PrecisionStats& precision_stats);

OverviewPageAnalysis ComputeAnalysisResult(const OpStats& op_stats);

OverviewPageRunEnvironment ComputeRunEnvironment(
    const RunEnvironment& run_environment);

OverviewPage ConvertOpStatsToOverviewPage(const OpStats& op_stats,
                                          HardwareType hardware_type);

void SetRemarks(const OpStats& op_stats, OverviewPageAnalysis* analysis);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_TO_OVERVIEW_PAGE_H_
