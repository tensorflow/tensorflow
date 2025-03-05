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
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/format_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/power_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/html_utils.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

using tsl::profiler::OneDigit;

// If the use of low-precision ops is less than this percentage threshold, a
// statement of suggestion will be made.
constexpr double kLowPrecisionPercentThreshold = 10;

struct TfFunctionInfo {
  absl::string_view function_name;
  double expensive_call_percent;
};

OverviewPageTip MakeOverviewPageTip(std::string text) {
  OverviewPageTip tip;
  tip.set_link(std::move(text));
  return tip;
}

// Makes a recommendation for looking up a document.
// doc_url is expected to be already be escaped suitably for use in an HTML
// attribute.
OverviewPageTip MakeOverviewPageTipDocLink(absl::string_view doc_url,
                                           absl::string_view text) {
  return MakeOverviewPageTip(AnchorElement(doc_url, text));
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
  absl::string_view device_name = HardwareType_Name(hardware_type);
  absl::string_view timeline_name = device_name;
  absl::string_view op_stats_toolname = "framework_op_stats";
  if (hardware_type == tensorflow::profiler::TPU) {
    timeline_name = "TPU core";
    op_stats_toolname = "op_profile";
  }
  *re->add_device_tips() = MakeOverviewPageTip(
      absl::StrCat(op_stats_toolname,
                   " (identify the time-consuming operations "
                   "executed on the ",
                   device_name, ")"));
  *re->add_device_tips() = MakeOverviewPageTip(absl::StrCat(
      "trace_viewer (look at the activities on the timeline of each ",
      timeline_name, " in the trace view)"));
}

void ComputeFaqTips(OverviewPageRecommendation* re) {
  *re->add_faq_tips() = MakeOverviewPageTip("Refer to the TF2 Profiler FAQ");
}

void ComputeDocumentationTips(OverviewPageRecommendation* re) {
  *re->add_documentation_tips() = MakeOverviewPageTipDocLink(
      "https://www.tensorflow.org/guide/data_performance_analysis",
      "Analyze tf.data performance with the TF Profiler");
  *re->add_documentation_tips() = MakeOverviewPageTipDocLink(
      "https://www.tensorflow.org/guide/"
      "data_performance",
      "Better performance with the tf.data API");
}

std::string GeneratePrecisionStatement(const PrecisionStats& precision_stats) {
  uint64 total_compute_ps =
      precision_stats.compute_16bit_ps() + precision_stats.compute_32bit_ps();
  if (total_compute_ps > 0) {
    double percent_16bit =
        (100.0 * precision_stats.compute_16bit_ps()) / total_compute_ps;
    if (percent_16bit < kLowPrecisionPercentThreshold) {
      return absl::StrCat(
          "Only ", OneDigit(percent_16bit),
          "% of device computation is 16 bit. So you might want to replace "
          "more 32-bit Ops by 16-bit Ops to improve performance (if the "
          "reduced accuracy is acceptable).");
    }
  }
  return "";
}

}  // namespace

void SetCommonRecommendation(
    absl::string_view input_classification, absl::string_view input_statement,
    absl::string_view output_statement, HardwareType hardware_type,
    absl::string_view tf_function_statement_html,
    absl::string_view eager_statement_html,
    absl::string_view outside_compilation_statement_html,
    OverviewPageRecommendation* re) {
  re->set_bottleneck(std::string(input_classification));
  re->set_statement(std::string(input_statement));
  re->set_output_statement(std::string(output_statement));
  re->set_tf_function_statement_html(std::string(tf_function_statement_html));
  re->set_eager_statement_html(std::string(eager_statement_html));
  re->set_outside_compilation_statement_html(
      std::string(outside_compilation_statement_html));
  ComputeHostTips(re);
  ComputeDeviceTips(hardware_type, re);
  ComputeDocumentationTips(re);
  ComputeFaqTips(re);
}

OverviewPageRecommendation ComputeGenericRecommendation(
    const BottleneckAnalysis& bottleneck,
    const PrecisionStats& precision_stats) {
  OverviewPageRecommendation re;
  GenericRecommendation generic;
  generic.set_device_collectives_bottleneck(
      bottleneck.device_collectives_classification());
  generic.set_device_collectives_statement(
      bottleneck.device_collectives_statement());
  generic.set_kernel_launch_bottleneck(
      bottleneck.kernel_launch_classification());
  generic.set_kernel_launch_statement(bottleneck.kernel_launch_statement());
  generic.set_all_other_bottleneck(bottleneck.all_other_classification());
  generic.set_all_other_statement(bottleneck.all_other_statement());
  generic.set_precision_statement(GeneratePrecisionStatement(precision_stats));
  re.mutable_recommendation()->PackFrom(generic);
  return re;
}

OverviewPageAnalysis ComputeAnalysisResult(const OpStats& op_stats) {
  OverviewPageAnalysis analysis;
  OpMetricsDb device_tf_op_metrics_db = CreateTfMetricsDbFromDeviceOpMetricsDb(
      op_stats.device_op_metrics_db(), /*with_idle=*/false);
  KernelStatsByOpName kernel_stats_by_op_name =
      GroupKernelReportsByOpName(op_stats.kernel_stats_db());
  uint64 total_device_time_ps = device_tf_op_metrics_db.total_time_ps();
  constexpr int kNumTopOpsShown = 10;
  double device_cumulative_fraction = 0.0;
  for (const OpMetrics* metrics :
       SortedOpMetricsDb(device_tf_op_metrics_db, kNumTopOpsShown)) {
    OverviewTfOp* op = analysis.add_top_device_ops();
    op->set_name(metrics->name());
    op->set_category(metrics->category());
    op->set_self_time_fraction(tsl::profiler::SafeDivide(
        metrics->self_time_ps(), total_device_time_ps));
    device_cumulative_fraction += op->self_time_fraction();
    op->set_cumulative_time_fraction(device_cumulative_fraction);
    op->set_flop_rate(tsl::profiler::SafeDivide(
        metrics->flops(), tsl::profiler::PicoToNano(metrics->time_ps())));
    auto iter = kernel_stats_by_op_name.find(op->name());
    if (iter != kernel_stats_by_op_name.end()) {
      op->set_is_op_tensorcore_eligible(
          iter->second.is_op_tensor_core_eligible);
      op->set_is_op_using_tensorcore(iter->second.tensor_core_duration_ns != 0);
    }
  }
  uint64 total_device_compute_ps =
      op_stats.device_op_metrics_db().precision_stats().compute_16bit_ps() +
      op_stats.device_op_metrics_db().precision_stats().compute_32bit_ps();
  analysis.set_device_compute_16bit_percent(
      100.0 *
      tsl::profiler::SafeDivide(
          op_stats.device_op_metrics_db().precision_stats().compute_16bit_ps(),
          total_device_compute_ps));
  analysis.set_device_compute_32bit_percent(
      100.0 *
      tsl::profiler::SafeDivide(
          op_stats.device_op_metrics_db().precision_stats().compute_32bit_ps(),
          total_device_compute_ps));

  uint64 num_host_tf_ops = 0;
  uint64 total_host_op_time_ps_exclude_idle = 0;
  uint64 eager_host_op_time_ps = 0;
  for (const OpMetrics& metrics : op_stats.host_op_metrics_db().metrics_db()) {
    num_host_tf_ops += metrics.occurrences();
    if (!IsIdleOp(metrics)) {
      total_host_op_time_ps_exclude_idle += metrics.self_time_ps();
      if (metrics.is_eager()) eager_host_op_time_ps += metrics.self_time_ps();
    }
  }
  uint64 num_device_tf_ops = 0;
  uint64 total_device_op_time_ps_exclude_idle = 0;
  uint64 eager_device_op_time_ps = 0;
  for (const OpMetrics& metrics : device_tf_op_metrics_db.metrics_db()) {
    num_device_tf_ops += metrics.occurrences();
    if (!IsIdleOp(metrics)) {
      total_device_op_time_ps_exclude_idle += metrics.self_time_ps();
      if (metrics.is_eager()) eager_device_op_time_ps += metrics.self_time_ps();
    }
  }
  // Figures out outside_compilation time from
  // op_stats.device_op_metrics_db().metrics_db(). We don't use the
  // {metrics.provenance(), metrics.name()} from
  // device_tf_op_metrics_db.metrics_db(), because metrics.provenance() there is
  // not set and metrics.name() can be either HLO-Op name or TF-Op name, which
  // will confuse tsl::profiler::IsOutsideCompilationOp().
  uint64 outside_compilation_device_op_time_ps = 0;
  for (const OpMetrics& metrics :
       op_stats.device_op_metrics_db().metrics_db()) {
    if (!tsl::profiler::IsOutsideCompilationOp(metrics.provenance(),
                                               metrics.long_name()))
      continue;
    outside_compilation_device_op_time_ps += metrics.self_time_ps();
  }
  uint64 num_total_tf_ops = num_host_tf_ops + num_device_tf_ops;
  analysis.set_host_tf_op_percent(
      100.0 * tsl::profiler::SafeDivide(num_host_tf_ops, num_total_tf_ops));
  analysis.set_device_tf_op_percent(
      100.0 * tsl::profiler::SafeDivide(num_device_tf_ops, num_total_tf_ops));
  analysis.set_host_trace_level(op_stats.run_environment().host_trace_level());
  analysis.set_host_op_time_eager_percent(
      100.0 * tsl::profiler::SafeDivide(eager_host_op_time_ps,
                                        total_host_op_time_ps_exclude_idle));
  analysis.set_device_op_time_eager_percent(
      100.0 * tsl::profiler::SafeDivide(eager_device_op_time_ps,
                                        total_device_op_time_ps_exclude_idle));
  analysis.set_device_op_time_outside_compilation_percent(
      100.0 * tsl::profiler::SafeDivide(outside_compilation_device_op_time_ps,
                                        total_device_op_time_ps_exclude_idle));
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
  re.set_replica_count(run_environment.replica_count());
  re.set_num_cores_per_replica(run_environment.num_cores_per_replica());
  re.set_is_training(run_environment.is_training());
  if (run_environment.has_power_metrics()) {
    *re.mutable_power_metrics() = run_environment.power_metrics();
  }
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

std::string TfFunctionRecommendationHtml(const TfFunctionDb& tf_function_db) {
  std::vector<TfFunctionInfo> candidates;
  for (const auto& name_fun : tf_function_db.tf_functions()) {
    const auto& fun = name_fun.second;
    if (fun.expensive_call_percent() >= kTfFunctionReportThresholdInPercent) {
      candidates.push_back({name_fun.first, fun.expensive_call_percent()});
    }
  }
  if (candidates.empty()) return "";
  auto cmp = [](const TfFunctionInfo& a, const TfFunctionInfo& b) {
    return a.expensive_call_percent > b.expensive_call_percent;
  };
  // Sorts candidates in descending order of expensive_call_percent.
  absl::c_sort(candidates, cmp);
  std::string expensive_functions = "";
  auto num_functions_shown = std::min(
      static_cast<decltype(candidates)::size_type>(3), candidates.size());

  for (decltype(candidates)::size_type i = 0; i < num_functions_shown; i++) {
    if (i > 0) absl::StrAppend(&expensive_functions, ", ");
    absl::StrAppend(&expensive_functions, "\"", candidates[i].function_name,
                    "\"");
  }
  if (candidates.size() > num_functions_shown)
    absl::StrAppend(&expensive_functions, " and more");
  return absl::StrCat("Expensive tf-functions detected (", expensive_functions,
                      ") due to either retracing or eager execution.");
}

std::string EagerRecommendationHtml(double host_op_time_eager_percent,
                                    double device_op_time_eager_percent) {
  std::string recommendation = "";
  if (host_op_time_eager_percent > kEagerReportThresholdInPercent)
    absl::StrAppend(&recommendation, OneDigit(host_op_time_eager_percent),
                    "% of Op time on the host used eager execution. ");
  if (device_op_time_eager_percent > kEagerReportThresholdInPercent)
    absl::StrAppend(&recommendation, OneDigit(device_op_time_eager_percent),
                    "% of Op time on the device used eager execution. ");
  if (!recommendation.empty())
    absl::StrAppend(&recommendation, "Performance could be improved with ",
                    AnchorElement("https://www.tensorflow.org/guide/function",
                                  "tf.function."));
  return recommendation;
}

std::string OutsideCompilationRecommendationHtml(
    double device_op_time_outside_compilation_percent) {
  if (device_op_time_outside_compilation_percent <=
      kOutsideCompilationThresholdInPercent)
    return "";
  return absl::StrCat(
      OneDigit(device_op_time_outside_compilation_percent),
      " % of Op time on the device are for outside compilation. Performance "
      "could be improved by avoiding outside compilation.");
}

OverviewPage ConvertOpStatsToOverviewPage(const OpStats& op_stats) {
  OverviewPage overview_page;
  *overview_page.mutable_run_environment() =
      ComputeRunEnvironment(op_stats.run_environment());
  *overview_page.mutable_analysis() = ComputeAnalysisResult(op_stats);
  *overview_page.mutable_input_analysis() =
      ConvertOpStatsToInputPipelineAnalysis(op_stats);
  BottleneckAnalysis bottleneck = ComputeBottleneckAnalysis(
      overview_page.input_analysis().input_time_breakdown(),
      overview_page.input_analysis().step_details());
  *overview_page.mutable_recommendation() = ComputeGenericRecommendation(
      bottleneck, op_stats.device_op_metrics_db().precision_stats());
  SetCommonRecommendation(
      bottleneck.input_classification(), bottleneck.input_statement(), "",
      ParseHardwareType(op_stats.run_environment().device_type()),
      TfFunctionRecommendationHtml(op_stats.tf_function_db()),
      EagerRecommendationHtml(
          overview_page.analysis().host_op_time_eager_percent(),
          overview_page.analysis().device_op_time_eager_percent()),
      OutsideCompilationRecommendationHtml(
          overview_page.analysis()
              .device_op_time_outside_compilation_percent()),
      overview_page.mutable_recommendation());
  PopulateOverviewDiagnostics(op_stats, overview_page.mutable_diagnostics());
  overview_page.mutable_analysis()->set_mxu_utilization_percent(
      op_stats.performance_counter_result().matrix_unit_utilization_percent());
  return overview_page;
}

}  // namespace profiler
}  // namespace tensorflow
