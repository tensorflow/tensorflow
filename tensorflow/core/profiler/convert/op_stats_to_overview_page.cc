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

#include "tensorflow/core/profiler/convert/op_stats_to_overview_page.h"

#include <algorithm>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

OverviewPageTip MakeOverviewPageTip(const string& text) {
  OverviewPageTip tip;
  tip.set_link(text);
  return tip;
}

string AnchorElement(const string& url, const string& text) {
  return absl::StrCat("<a href=\"", url, "\" target=\"_blank\">", text, "</a>");
}

// Makes a recommendation for looking up a document.
// doc_url is expected to be already be escaped suitably for use in an HTML
// attribute.
OverviewPageTip MakeOverviewPageTipDocLink(const string& doc_url,
                                           const string& text) {
  OverviewPageTip tip;
  tip.set_link(AnchorElement(doc_url, text));
  return tip;
}

void ComputeHostTips(OverviewPageRecommendation* re) {
  *re->add_host_tips() = MakeOverviewPageTip(
      "input_pipeline_analyzer (especially Section 3 for the breakdown of "
      "input operations on the Host)");
  *re->add_host_tips() = MakeOverviewPageTip(
      "trace_viewer (look at the activities on the timeline of each Host "
      "Thread near the bottom of the trace view)");
}

void ComputeDeviceTips(HardwareType hardware_type,
                       OverviewPageRecommendation* re) {
  const string& device_name = HardwareType_Name(hardware_type);
  string timeline_name =
      (hardware_type == tensorflow::profiler::TPU) ? "TPU core" : device_name;
  *re->add_device_tips() = MakeOverviewPageTip(absl::StrCat(
      "op_profile (identify the time-consuming operations executed on the ",
      device_name, ")"));
  *re->add_device_tips() = MakeOverviewPageTip(absl::StrCat(
      "trace_viewer (look at the activities on the timeline of each ",
      timeline_name, " in the trace view)"));
}

void ComputeFaqTips(OverviewPageRecommendation* re) {
  *re->add_faq_tips() = MakeOverviewPageTip("Refer to the Cloud tools FAQ");
}

void ComputeDocumentationTips(OverviewPageRecommendation* re) {
  *re->add_documentation_tips() = MakeOverviewPageTipDocLink(
      "https://www.tensorflow.org/guide/"
      "data_performance",
      "Better performance with the tf.data API");
}

}  // namespace

void SetCommonRecommendation(const CommonBottleneck& bottleneck,
                             HardwareType hardware_type,
                             OverviewPageRecommendation* re) {
  re->set_bottleneck(bottleneck.input_classification);
  re->set_statement(bottleneck.input_statement);
  ComputeHostTips(re);
  ComputeDeviceTips(hardware_type, re);
  ComputeDocumentationTips(re);
  ComputeFaqTips(re);
}

OverviewPageRecommendation ComputeGenericRecommendation(
    const GenericBottleneck& bottleneck) {
  OverviewPageRecommendation re;
  GenericRecommendation generic;
  generic.set_kernel_launch_bottleneck(bottleneck.kernel_launch_classification);
  generic.set_kernel_launch_statement(bottleneck.kernel_launch_statement);
  generic.set_all_other_bottleneck(bottleneck.all_other_classification);
  generic.set_all_other_statement(bottleneck.all_other_statement);
  re.mutable_recommendation()->PackFrom(generic);
  return re;
}

OverviewPageAnalysis ComputeAnalysisResult(const OpStats& op_stats) {
  OverviewPageAnalysis analysis;
  OpMetricsDb metrics_db =
      CreateTfMetricsDbFromHloMetricsDb(op_stats.device_op_metrics_db());
  uint64 total_device_time_ps = metrics_db.total_time_ps();
  constexpr int kNumTopOpsShown = 10;
  double device_cumulative_fraction = 0.0;
  for (const OpMetrics* metrics :
       SortedOpMetricsDb(metrics_db, kNumTopOpsShown)) {
    OverviewTfOp* op = analysis.add_top_device_ops();
    op->set_name(metrics->name());
    op->set_category(metrics->category());
    op->set_self_time_fraction(
        SafeDivide(metrics->self_time_ps(), total_device_time_ps));
    device_cumulative_fraction += op->self_time_fraction();
    op->set_cumulative_time_fraction(device_cumulative_fraction);
    op->set_flop_rate(
        SafeDivide(metrics->flops(), PicosToNanos(metrics->time_ps())));
  }
  return analysis;
}

// Converts from HostIndependentJobInfo to OverviewPageHostIndependentJobInfo.
OverviewPageHostIndependentJobInfo ToOverviewPageHostIndependentJobInfo(
    const HostIndependentJobInfoResult& host_independent_job_info) {
  OverviewPageHostIndependentJobInfo result;
  result.set_change_list(host_independent_job_info.change_list());
  result.set_build_time(host_independent_job_info.build_time());
  result.set_build_target(host_independent_job_info.build_target());
  result.set_profile_duration_ms(
      host_independent_job_info.profile_duration_ms());
  return result;
}

// Converts from HostDependentJobInfo to OverviewPageHostDependentJobInfo.
OverviewPageHostDependentJobInfo ToOverviewPageHostDependentJobInfo(
    const HostDependentJobInfoResult& host_dependent_job_info) {
  OverviewPageHostDependentJobInfo result;
  result.set_host_id(host_dependent_job_info.host_id());
  result.set_command_line(host_dependent_job_info.command_line());
  result.set_start_time(host_dependent_job_info.start_time());
  result.set_bns_address(host_dependent_job_info.bns_address());
  result.set_profile_time_ns(host_dependent_job_info.profile_time_ns());
  return result;
}

OverviewPageRunEnvironment ComputeRunEnvironment(
    const RunEnvironment& run_environment) {
  OverviewPageRunEnvironment re;
  re.set_host_count(run_environment.host_count());
  re.set_task_count(run_environment.task_count());
  re.set_device_type(run_environment.device_type());
  re.set_device_core_count(run_environment.device_core_count());
  re.set_per_core_batch_size(run_environment.per_core_batch_size());
  re.set_replica_count(run_environment.replica_count());
  re.set_num_cores_per_replica(run_environment.num_cores_per_replica());
  *re.mutable_host_independent_job_info() =
      ToOverviewPageHostIndependentJobInfo(
          run_environment.host_independent_job_info());
  for (const auto& host_dependent_job_info :
       run_environment.host_dependent_job_info()) {
    *re.add_host_dependent_job_info() =
        ToOverviewPageHostDependentJobInfo(host_dependent_job_info);
  }
  return re;
}

OverviewPage ConvertOpStatsToOverviewPage(const OpStats& op_stats,
                                          HardwareType hardware_type) {
  OverviewPageAnalysis analysis = ComputeAnalysisResult(op_stats);
  InputPipelineAnalysisResult input_analysis =
      ConvertOpStatsToInputPipelineAnalysis(op_stats, hardware_type);
  GenericBottleneck bottleneck = GenericOverallBottleneck(input_analysis);
  OverviewPageRecommendation recommendation =
      ComputeGenericRecommendation(bottleneck);
  SetCommonRecommendation(bottleneck.common, hardware_type, &recommendation);

  OverviewPage overview_page;
  *overview_page.mutable_run_environment() =
      ComputeRunEnvironment(op_stats.run_environment());
  *overview_page.mutable_analysis() = analysis;
  *overview_page.mutable_input_analysis() = input_analysis;
  *overview_page.mutable_recommendation() = recommendation;
  return overview_page;
}

}  // namespace profiler
}  // namespace tensorflow
