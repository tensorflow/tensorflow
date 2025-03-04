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

#include "tensorflow/core/profiler/convert/op_stats_to_input_pipeline_analysis.h"

#include <math.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/format_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/util/stats_calculator.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/convert/profile_time_breakdown.h"
#include "tensorflow/core/profiler/convert/step_events_to_steps_db.h"
#include "tensorflow/core/profiler/convert/tpu_input_pipeline_analysis_constants.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/html_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/tpu_step_breakdown_utils.h"
#include "tensorflow/core/profiler/utils/tpu_step_details_utils.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace profiler {

namespace {

using tsl::profiler::OneDigit;

// If the percentage of step time that spends on SparseCoreV0 is more than
// kModeratelySparseCoreV0BoundThresholdInPercent, it is considered highly
// SparseCoreV0 bound.
constexpr double kModeratelySparseCoreV0BoundThresholdInPercent = 10;
// If the percentage of step time that spends on all-reduce is more than
// kAllReduceBoundThresholdInPercent, it is considered all-reduce bound.
constexpr double kAllReduceBoundThresholdInPercent = 6;
// If the percentage of step time that is idle due to host overhead (but not
// input-related) is >= kTcIdleThresholdInPercent, it will be highlighted in the
// recommendation section of the Overview Page.
constexpr double kTcIdleThresholdInPercent = 3;
// Public doc on how to run multiple steps in a tf-function.
constexpr absl::string_view kMultipleStepsInTffunctionDoc =
    "https://www.tensorflow.org/guide/"
    "tpu#improving_performance_by_multiple_steps_within_tffunction";

const double kNumPsPerMs = 1000000000.0;

// If the percentage of step time that is due to infeed is less than
// kModeratelyInfeedBoundThresholdInPercent, it is considered NOT
// input-bound; else if it is less than
// kHighlyInfeedBoundThresholdInPercent, it is considered MODERATELY
// input-bound; else if it is considered HIGHLY input-bound.
constexpr double kModeratelyInfeedBoundThresholdInPercent = 5;
constexpr double kHighlyInfeedBoundThresholdInPercent = 20;

// If the percentage of step time that is due to outfeed is less than
// kModeratelyOutfeedBoundThresholdInPercent, it is considered NOT
// output-bound; else if it is less than
// kHighlyOutfeedBoundThresholdInPercent, it is considered MODERATELY
// output-bound; else if it is considered HIGHLY output-bound.
constexpr double kModeratelyOutfeedBoundThresholdInPercent = 5;
constexpr double kHighlyOutfeedBoundThresholdInPercent = 20;

// If the percentage of step time that is due to kernel launch is less than
// kModeratelyKernelLaunchBoundThresholdInPercent, it is considered NOT
// kernel-launch bound; else if it is less than
// kHighlyKernelLaunchBoundThresholdInPercent, it is considered MODERATELY
// kernel-launch bound; else if it is considered HIGHLY kernel-launch bound.
constexpr double kModeratelyKernelLaunchBoundThresholdInPercent = 3;
constexpr double kHighlyKernelLaunchBoundThresholdInPercent = 15;

// If the percentage of step time that is due to all other time is less than
// kModeratelyAllOtherBoundThresholdInPercent, it is considered NOT
// all-other bound; else if it is less than
// kHighlyAllOtherBoundThresholdInPercent, it is considered MODERATELY
// all-other bound; else if it is considered HIGHLY all-other bound.
constexpr double kModeratelyAllOtherBoundThresholdInPercent = 3;
constexpr double kHighlyAllOtherBoundThresholdInPercent = 15;

// If the percentage of step time that is due to device collectives is less than
// kModeratelyDeviceCollectivesBoundThresholdInPercent, it is considered NOT
// device-collectives bound; else if it is less than
// kHighlyDeviceCollectivesBoundThresholdInPercent, it is considered MODERATELY
// device-collectives  bound; else if it is considered HIGHLY device-collectives
// bound.
constexpr double kModeratelyDeviceCollectivesBoundThresholdInPercent = 3;
constexpr double kHighlyDeviceCollectivesBoundThresholdInPercent = 15;

// Section number of the host-analysis section in the input-pipeline analysis.
constexpr int kHostAnalysisSectionNumber = 3;
// Python-only explanation for "All Others" time.
const char* kAllOthersPythonExplanation =
    " % of the total step time sampled is spent on 'All Others' time. "
    "This could be due to Python execution overhead.";
// Explanation for "Kernel Launch" time due to CPU contention with tf.data.
const char* kKernelLaunchTfDataContention =
    " It could be due to CPU contention with tf.data. In this case, you may "
    "try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.";

template <class Collection>
double GetTimeInMs(const Collection& type_ps, EventType event_type) {
  return tsl::profiler::PicoToMilli(
      gtl::FindWithDefault(type_ps, event_type, /*value=*/0));
}

GenericStepTimeBreakdown ComputeGenericStepTimeBreakdownInMs(
    const InputPipelineAnalysisResult& analysis) {
  tsl::Stat<double> unknown_time_ms;
  tsl::Stat<double> host_wait_input_ms;
  tsl::Stat<double> host_to_device_ms;
  tsl::Stat<double> input_ms;
  tsl::Stat<double> output_ms;
  tsl::Stat<double> device_compute_ms;
  tsl::Stat<double> device_to_device_ms;
  tsl::Stat<double> device_collectives_ms;
  tsl::Stat<double> host_compute_ms;
  tsl::Stat<double> host_prepare_ms;
  tsl::Stat<double> host_compile_ms;
  GenericStepTimeBreakdown result;

  for (const google::protobuf::Any& step_details : analysis.step_details()) {
    PerGenericStepDetails details;
    bool success = step_details.UnpackTo(&details);
    if (!success && !step_details.type_url().empty()) {
      LOG(ERROR) << "Unable to unpack step_breakdown. Expected: generic"
                 << std::endl;
      return {};
    }
    unknown_time_ms.UpdateStat(details.unknown_time_ms());
    host_wait_input_ms.UpdateStat(details.host_wait_input_ms());
    host_to_device_ms.UpdateStat(details.host_to_device_ms());
    input_ms.UpdateStat(details.host_wait_input_ms() +
                        details.host_to_device_ms());
    output_ms.UpdateStat(details.output_ms());
    device_compute_ms.UpdateStat(details.device_compute_ms());
    device_to_device_ms.UpdateStat(details.device_to_device_ms());
    device_collectives_ms.UpdateStat(details.device_collectives_ms());
    host_compute_ms.UpdateStat(details.host_compute_ms());
    host_prepare_ms.UpdateStat(details.host_prepare_ms());
    host_compile_ms.UpdateStat(details.host_compile_ms());
  }
  *result.mutable_unknown_time_ms_summary() =
      GetStepSummaryForSampleStats(unknown_time_ms);
  *result.mutable_host_wait_input_ms_summary() =
      GetStepSummaryForSampleStats(host_wait_input_ms);
  *result.mutable_host_to_device_ms_summary() =
      GetStepSummaryForSampleStats(host_to_device_ms);
  *result.mutable_input_ms_summary() = GetStepSummaryForSampleStats(input_ms);
  *result.mutable_output_ms_summary() = GetStepSummaryForSampleStats(output_ms);
  *result.mutable_device_compute_ms_summary() =
      GetStepSummaryForSampleStats(device_compute_ms);
  *result.mutable_device_to_device_ms_summary() =
      GetStepSummaryForSampleStats(device_to_device_ms);
  *result.mutable_device_collectives_ms_summary() =
      GetStepSummaryForSampleStats(device_collectives_ms);
  *result.mutable_host_compute_ms_summary() =
      GetStepSummaryForSampleStats(host_compute_ms);
  *result.mutable_host_prepare_ms_summary() =
      GetStepSummaryForSampleStats(host_prepare_ms);
  *result.mutable_host_compile_ms_summary() =
      GetStepSummaryForSampleStats(host_compile_ms);
  return result;
}

InputPipelineAnalysisResult ComputeGenericInputPipelineAnalysisResult(
    const tsl::protobuf::RepeatedPtrField<PerCoreStepInfo>& grouped_by_step) {
  InputPipelineAnalysisResult result;
  result.set_tag(false);

  // Computes the summary of step time in ms.
  *result.mutable_step_time_summary() =
      ComputeStepTimeSummaryInMs(grouped_by_step);

  tsl::Stat<double> input_summary_stats_in_percent;
  for (const auto& coreid_stepinfo_map : grouped_by_step) {
    // Iterates over each step.
    const auto* ptr = gtl::FindOrNull(coreid_stepinfo_map.step_info_per_core(),
                                      kDefaultGpuLocalCoreId);
    if (ptr == nullptr) {
      // For generic hardware, all step-info is put under core-0. If ptr
      // is nullptr, it means there is no step at all.
      continue;
    }
    const StepInfoResult& step_info = *ptr;
    // Adds the details for a new step.
    PerGenericStepDetails details;
    details.set_step_number(step_info.step_num());
    if (step_info.step_name().empty()) {
      details.set_step_name(absl::StrCat(step_info.step_num()));
    } else {
      details.set_step_name(step_info.step_name());
    }
    details.set_step_time_ms(
        tsl::profiler::PicoToMilli(step_info.duration_ps()));
    GenericStepBreakdown generic;
    bool success = step_info.step_breakdown().UnpackTo(&generic);
    if (!success && !step_info.step_breakdown().type_url().empty()) {
      LOG(ERROR) << "Unable to unpack step_breakdown. Expected: generic"
                 << std::endl;
      return {};
    }
    const auto& type_ps = generic.type_ps();
    details.set_unknown_time_ms(GetTimeInMs(type_ps, UNKNOWN_TIME));
    details.set_host_wait_input_ms(GetTimeInMs(type_ps, HOST_WAIT_INPUT));
    details.set_host_to_device_ms(GetTimeInMs(type_ps, HOST_TO_DEVICE) +
                                  GetTimeInMs(type_ps, DEVICE_WAIT_HOST));
    details.set_output_ms(GetTimeInMs(type_ps, DEVICE_TO_HOST));
    details.set_device_compute_ms(GetTimeInMs(type_ps, DEVICE_COMPUTE_16) +
                                  GetTimeInMs(type_ps, DEVICE_COMPUTE_32));
    details.set_device_to_device_ms(GetTimeInMs(type_ps, DEVICE_TO_DEVICE) +
                                    GetTimeInMs(type_ps, DEVICE_WAIT_DEVICE));
    details.set_device_collectives_ms(GetTimeInMs(type_ps, DEVICE_COLLECTIVES));
    details.set_host_compute_ms(GetTimeInMs(type_ps, HOST_COMPUTE));
    details.set_host_prepare_ms(GetTimeInMs(type_ps, HOST_PREPARE));
    details.set_host_compile_ms(GetTimeInMs(type_ps, HOST_COMPILE));
    result.add_step_details()->PackFrom(details);

    const double input_percent_of_step_time =
        100.0 * tsl::profiler::SafeDivide(
                    details.host_wait_input_ms() + details.host_to_device_ms(),
                    details.step_time_ms());
    input_summary_stats_in_percent.UpdateStat(input_percent_of_step_time);
  }

  // Computes the summary of input time as percentage of step time.
  *result.mutable_input_percent_summary() =
      GetStepSummaryForSampleStats(input_summary_stats_in_percent);

  // Computes the breakdown of step time.
  GenericStepTimeBreakdown generic_step_time_breakdown =
      ComputeGenericStepTimeBreakdownInMs(result);
  result.mutable_step_time_breakdown()->PackFrom(generic_step_time_breakdown);

  return result;
}

// Classification of input processing on the host.
enum class InputOpCategory {
  kEnqueue,           // enqueue data to be transferred to device.
  kDemandedFileRead,  // demanded read from file.
  kAdvancedFileRead,  // advanced read from file (including cached,
                      // prefetch, parallel-map, interleave).
  kPreprocessing      // data preprocessing.
};

std::string InputOpCategoryString(InputOpCategory category) {
  switch (category) {
    case InputOpCategory::kEnqueue:
      return "Enqueue";
    case InputOpCategory::kDemandedFileRead:
      return "Demanded file read";
    case InputOpCategory::kAdvancedFileRead:
      return "Advanced file read";
    case InputOpCategory::kPreprocessing:
      return "Preprocessing";
  }
}

inline bool IsInputOp(absl::string_view category) {
  // Do not include "IteratorGetNext*" here, because IteratorGetNext is an Op
  // that experiences the install stall, not an Op that causes the input stall.
  return tsl::profiler::IsInfeedEnqueueOp(category) ||
         tsl::profiler::IsDatasetOp(category) ||
         tsl::profiler::IsMemcpyHToDOp(category);
}

// TODO(ckluk):
//   Confirm with the tf.data team if the classification below is correct.
InputOpCategory CategorizeInputOp(absl::string_view name,
                                  absl::string_view category) {
  if (tsl::profiler::IsInfeedEnqueueOp(category) ||
      tsl::profiler::IsMemcpyHToDOp(category)) {
    // Ops for sending input from host to device.
    return InputOpCategory::kEnqueue;
  }
  DCHECK(tsl::profiler::IsDatasetOp(category));
  if (absl::EndsWith(name, "::TFRecord") ||
      absl::EndsWith(name, "::TextLine") ||
      absl::EndsWith(name, "::FixedLengthRecord") ||
      absl::EndsWith(name, "::SSTable") || absl::EndsWith(name, "::RecordIO")) {
    // Ops that read files.
    if (absl::StrContains(name, "::MemoryReader") ||
        absl::StrContains(name, "::MemoryWriter") ||
        absl::StrContains(name, "::Interleave") ||
        absl::StrContains(name, "::Prefetch") ||
        absl::StrContains(name, "::ParallelMap")) {
      // Ops that read files in advance, including caching, interleaving, and
      // prefetching.
      return InputOpCategory::kAdvancedFileRead;
    } else {
      // Ops that read files on demand.
      return InputOpCategory::kDemandedFileRead;
    }
  } else {
    // All other ops are classified as preprocessing.
    return InputOpCategory::kPreprocessing;
  }
}

struct InputOpMetrics {
  std::vector<const OpMetrics*> input_op_metrics;
  uint64 input_op_time_ps = 0;
};

InputOpMetrics SelectInputOpMetrics(const OpMetricsDb& all_op_metrics) {
  InputOpMetrics input_op_metrics;
  for (const OpMetrics* op_metrics : SortedOpMetricsDb(all_op_metrics)) {
    if (IsInputOp(op_metrics->category())) {
      input_op_metrics.input_op_metrics.push_back(op_metrics);
      input_op_metrics.input_op_time_ps += op_metrics->self_time_ps();
    }
  }
  return input_op_metrics;
}

InputOpDetails ConvertOpMetricsToInputOpDetails(const OpMetrics& op_metrics,
                                                uint64 input_op_time_ps,
                                                InputOpCategory category) {
  InputOpDetails details;
  details.set_op_name(op_metrics.name());
  details.set_count(op_metrics.occurrences());
  details.set_time_in_ms(tsl::profiler::PicoToMilli(op_metrics.time_ps()));
  details.set_self_time_in_ms(
      tsl::profiler::PicoToMilli(op_metrics.self_time_ps()));
  details.set_time_in_percent(
      100.0 *
      tsl::profiler::SafeDivide(op_metrics.time_ps(), input_op_time_ps));
  details.set_self_time_in_percent(
      100.0 *
      tsl::profiler::SafeDivide(op_metrics.self_time_ps(), input_op_time_ps));
  details.set_category(InputOpCategoryString(category));
  return details;
}

// Returns the ratio of the host-to-device time in each step to the step-time.
double RatioOfHostToDeviceTimeToStepTime(
    const OpMetricsDb& host_tf_metrics_db,
    const InputPipelineAnalysisResult& input_pipeline_analysis) {
  // For TPU execution that uses infeed.
  std::optional<double> host_infeed_enqueue_ratio =
      HostInfeedEnqueueRatio(host_tf_metrics_db);
  if (host_infeed_enqueue_ratio.has_value()) {
    return host_infeed_enqueue_ratio.value();
  }
  // For GPU and TPU execution that do not use infeed.
  double avg_step_time_ms =
      input_pipeline_analysis.step_time_summary().average();
  if (avg_step_time_ms > 0) {
    // Uses the on-device step time.
    GenericStepTimeBreakdown generic_breakdown;
    if (input_pipeline_analysis.step_time_breakdown().UnpackTo(
            &generic_breakdown)) {
      double avg_host_to_device_time_ms =
          generic_breakdown.host_to_device_ms_summary().average();
      return tsl::profiler::SafeDivide(avg_host_to_device_time_ms,
                                       avg_step_time_ms);
    }
  }
  return 0.0;
}

void DeviceCollectivesAnalysis(double device_collectives_percent,
                               std::string* device_collectives_classification,
                               std::string* device_collectives_statement) {
  if (device_collectives_percent >=
      kHighlyDeviceCollectivesBoundThresholdInPercent) {
    *device_collectives_classification = "high";
    *device_collectives_statement =
        absl::StrCat(OneDigit(device_collectives_percent),
                     " % of the total step time sampled is spent on 'Device "
                     "Collective Communication'.");
  } else if (device_collectives_percent >=
             kModeratelyDeviceCollectivesBoundThresholdInPercent) {
    *device_collectives_classification = "moderate";
    *device_collectives_statement =
        absl::StrCat(OneDigit(device_collectives_percent),
                     " % of the total step time sampled is spent on 'Device "
                     "Collective Communication'.");
  } else {
    *device_collectives_classification = "no";
    *device_collectives_statement = "";
  }
}

void KernelLaunchAnalysis(bool tfdata_used, double kernel_launch_percent,
                          std::string* kernel_launch_classification,
                          std::string* kernel_launch_statement) {
  if (kernel_launch_percent >= kHighlyKernelLaunchBoundThresholdInPercent) {
    *kernel_launch_classification = "high";
    *kernel_launch_statement = absl::StrCat(
        OneDigit(kernel_launch_percent),
        " % of the total step time sampled is spent on 'Kernel Launch'.");
    if (tfdata_used) {
      absl::StrAppend(kernel_launch_statement, kKernelLaunchTfDataContention);
    }
  } else if (kernel_launch_percent >=
             kModeratelyKernelLaunchBoundThresholdInPercent) {
    *kernel_launch_classification = "moderate";
    *kernel_launch_statement = absl::StrCat(
        OneDigit(kernel_launch_percent),
        " % of the total step time sampled is spent on 'Kernel Launch'.");
    if (tfdata_used) {
      absl::StrAppend(kernel_launch_statement, kKernelLaunchTfDataContention);
    }
  } else {
    *kernel_launch_classification = "no";
    *kernel_launch_statement = "";
  }
}

void AllOtherAnalysis(bool all_other_reported, double all_other_percent,
                      std::string* all_other_classification,
                      std::string* all_other_statement) {
  if (all_other_reported) {
    *all_other_classification = "no";
    *all_other_statement = "";
    return;
  }
  if (all_other_percent >= kHighlyAllOtherBoundThresholdInPercent) {
    *all_other_classification = "high";
    *all_other_statement =
        absl::StrCat(OneDigit(all_other_percent), kAllOthersPythonExplanation);
  } else if (all_other_percent >= kModeratelyAllOtherBoundThresholdInPercent) {
    *all_other_classification = "moderate";
    *all_other_statement =
        absl::StrCat(OneDigit(all_other_percent), kAllOthersPythonExplanation);
  } else {
    *all_other_classification = "no";
    *all_other_statement = "";
  }
}

// Tests if tf.data API is in use.
bool TfDataInUse(const InputTimeBreakdown& breakdown) {
  // Do not include enqueue_us because the "enqueue" Op that Xprof recognizes is
  // not part of tf.data.
  return breakdown.demanded_file_read_us() > 0 ||
         breakdown.advanced_file_read_us() > 0 ||
         breakdown.preprocessing_us() > 0;
}

// Returns a HTML link with the given text.
std::string MakeDocLink(absl::string_view doc_link, absl::string_view text) {
  return absl::StrCat("<a href=\"", doc_link, "\" target=\"_blank\">", text,
                      "</a>");
}

// Returns the HTML link to the introduction to the tf.data API.
std::string DatasetIntroDoc() {
  return "https://www.tensorflow.org/guide/data";
}

struct WaitForScV0Breakdown {
  uint64_t DurationPs() const {
    return scv0_infeed_duration_ps + scv0_compute_duration_ps;
  }

  uint64_t scv0_infeed_duration_ps = 0;
  uint64_t scv0_compute_duration_ps = 0;
};

struct TcInfeed {
  std::optional<uint32_t> core_id;
  uint64_t duration_ps = 0;
};

void ConvertGenericStepBreakdownToTpuStepBreakdown(
    const tensorflow::profiler::GenericStepBreakdown& generic_step_breakdown,
    uint64_t step_time_ps, TpuStepBreakdown& tpu_step_breakdown) {
  auto& category_ps = generic_step_breakdown.category_ps();
  tensorflow::profiler::ProfileTimeBreakdown time_breakdown;
  for (const auto& [category, time_ps] : category_ps) {
    // Don't add idle time to time_breakdown as the idle time is inferred.
    if (category == "IDLE") continue;
    time_breakdown.IncrementCategoryTimePs(category, time_ps);
  }
  time_breakdown.SetProfileTimePs(step_time_ps);
  time_breakdown.BreakdownSparseCoreV0Infeed();

  tpu_step_breakdown.set_infeed_duration_ps(time_breakdown.InfeedTimePs());
  tpu_step_breakdown.set_host_outfeed_ps(time_breakdown.OutfeedTimePs());
  tpu_step_breakdown.set_wait_for_scv0_duration_ps(
      time_breakdown.SparseCoreV0InfeedWaitTimePs());
  tpu_step_breakdown.set_scv0_infeed_transform_ps(
      time_breakdown.SparseCoreV0InfeedTransformTimePs());
  tpu_step_breakdown.set_scv0_outfeed_ps(
      time_breakdown.SparseCoreV0OutfeedTimePs());
  tpu_step_breakdown.set_crs_duration_ps(
      time_breakdown.AllReduceOrAllToAllTimePs());
  tpu_step_breakdown.set_send_duration_ps(time_breakdown.SendTimePs());
  tpu_step_breakdown.set_recv_duration_ps(time_breakdown.RecvTimePs());
  tpu_step_breakdown.set_host_send_duration_ps(time_breakdown.HostSendTimePs());
  tpu_step_breakdown.set_host_recv_duration_ps(time_breakdown.HostRecvTimePs());
  tpu_step_breakdown.set_wait_for_megacore_fusion_peer_duration_ps(
      time_breakdown.MegacoreFusionTimePs());
  tpu_step_breakdown.set_high_flops_compute_ps(
      time_breakdown.HighFlopsComputeTimePs());
  tpu_step_breakdown.set_tc_idle_ps(time_breakdown.IdleTimePs());
  tpu_step_breakdown.set_tc_busy_ps(time_breakdown.TensorCoreBusyTimePs());
}
// Computes the fields in PerStepData by considering the different StepInfos
// of the same step across cores.
PerTpuStepDetails ComputeTpuPerStepDataAcrossCores(
    const PerCoreStepInfo& coreid_stepinfo_map,
    const tsl::protobuf::Map<uint32_t, tensorflow::profiler::CoreDetails>&
        core_details_map) {
  PerTpuStepDetails per_step_data;

  PerCoreAllReduceBreakdown all_reduce_breakdown =
      ComputePerStepAllReduceBreakdownAcrossCores(coreid_stepinfo_map);

  tsl::Stat<double> infeed_percent_stats;
  tsl::Stat<uint64_t> step_stats_in_ps;
  tsl::Stat<uint64_t> optimal_step_time_ps;
  // Take the average TC outfeed time in result.
  tsl::Stat<uint64_t> tc_outfeed_time_in_ps;
  tsl::Stat<uint64_t> sc_optimal_step_time_ps;
  tsl::Stat<uint64_t> sc_step_stats_in_ps;
  tsl::Stat<uint64_t> sc_outfeed_time_in_ps;
  tsl::Stat<uint64_t> sc_infeed_time_in_ps;
  tsl::Stat<uint64_t> sc_idle_time_in_ps;

  tsl::Stat<uint64_t> host_send_recv_time_ps;

  // For the core with the max wait-for-scv0 duration, breakdown to compute and
  // infeed time.
  WaitForScV0Breakdown max_wait_for_scv0;

  TcInfeed max_infeed;

  // For the core with the max all reduce duration, breakdown to compute and
  // synchronization time.
  AllReduceBreakdown max_all_reduce;

  per_step_data.set_step_number(-1);
  auto process_step_for_sc =
      [&](const tensorflow::profiler::StepInfoResult& step_info,
          const SparseCoreStepBreakdown& sc_step) {
        if (per_step_data.step_number() < 0) {
          per_step_data.set_step_number(step_info.step_num());
        } else {
          if (per_step_data.step_number() != step_info.step_num()) {
            VLOG(1) << "Inconsistent step numbers across cores ("
                    << per_step_data.step_number() << " vs. "
                    << step_info.step_num() << ").";
          }
        }
        sc_step_stats_in_ps.UpdateStat(step_info.duration_ps());
        sc_outfeed_time_in_ps.UpdateStat(sc_step.sc_outfeed_ps());
        sc_infeed_time_in_ps.UpdateStat(sc_step.sc_infeed_ps());
        sc_optimal_step_time_ps.UpdateStat(step_info.duration_ps() -
                                           sc_step.sc_infeed_ps() -
                                           sc_step.sc_outfeed_ps());
        sc_idle_time_in_ps.UpdateStat(sc_step.sc_idle_ps());
      };
  for (const auto& [core_id, step_info] :
       coreid_stepinfo_map.step_info_per_core()) {
    // iterates over each core.
    TpuStepBreakdown tpu;
    if (!step_info.step_breakdown().UnpackTo(&tpu)) {
      VLOG(1) << "Unable to unpack step_breakdown from tpu, try unpacking from "
                 "generic";
      tensorflow::profiler::GenericStepBreakdown generic_step_breakdown;
      if (!step_info.step_breakdown().UnpackTo(&generic_step_breakdown)) {
        SparseCoreStepBreakdown sc_step;
        if (step_info.step_breakdown().UnpackTo(&sc_step)) {
          process_step_for_sc(step_info, sc_step);
          continue;
        } else {
          LOG(ERROR) << "Unable to unpack step_breakdown from "
                        "GenericStepBreakdown or SparseCoreStepBreakdown";
          // TODO(b/302086111): Switch back to DFATAL once absl is updated.
          DCHECK(false);
          return per_step_data;
        }
      }
      if (core_id >= kSparseCoreIndexStart) {
        // Sparse core step breakdown from xspace.
        uint64_t total_time_ps = step_info.duration_ps();
        uint64_t idle_time_ps =
            generic_step_breakdown.category_ps().find("IDLE")->second;
        sc_step_stats_in_ps.UpdateStat(total_time_ps);
        sc_idle_time_in_ps.UpdateStat(idle_time_ps);
        continue;
      } else {
        // Tensor core step breakdown from xspace.
        ConvertGenericStepBreakdownToTpuStepBreakdown(
            generic_step_breakdown, step_info.duration_ps(), tpu);
      }
    }
    step_stats_in_ps.UpdateStat(step_info.duration_ps());
    if (tpu.wait_for_scv0_duration_ps() > max_wait_for_scv0.DurationPs()) {
      max_wait_for_scv0.scv0_infeed_duration_ps = ScV0InfeedDurationPs(tpu);
      max_wait_for_scv0.scv0_compute_duration_ps = ScV0ComputeDurationPs(tpu);
    }

    tc_outfeed_time_in_ps.UpdateStat(tpu.host_outfeed_ps());

    const AllReduceBreakdown& breakdown = all_reduce_breakdown[core_id];
    if (breakdown.DurationPs() > max_all_reduce.DurationPs()) {
      max_all_reduce = breakdown;
    }

    infeed_percent_stats.UpdateStat(100.0 * TcPlusScV0InfeedDurationPs(tpu) /
                                    step_info.duration_ps());
    // The optimal step time is the actual step time minus the time tensor
    // core spends waiting for host or sparsecorev0 (but not other tensor
    // cores).
    optimal_step_time_ps.UpdateStat(step_info.duration_ps() -
                                    WaitForHostOrScV0DurationPs(tpu));
    host_send_recv_time_ps.UpdateStat(HostSendRecvDurationPs(tpu));

    if (per_step_data.step_number() < 0) {
      // Sets the step number of the current step from the first core.
      per_step_data.set_step_number(step_info.step_num());
    } else {
      // The step number of the current step is already set. Checks if it is
      // the same across cores. In case of multi-host tracing, we may have
      // some inconsistent steps as tracing is not exactly guaranteed to be
      // synchronized across all hosts.
      if (per_step_data.step_number() != step_info.step_num()) {
        VLOG(1) << "Inconsistent step numbers across cores ("
                << per_step_data.step_number() << " vs. "
                << step_info.step_num() << ").";
      }
    }
    if (tpu.infeed_duration_ps() > max_infeed.duration_ps) {
      max_infeed.core_id = core_id;
      max_infeed.duration_ps = tpu.infeed_duration_ps();
    }
  }

  per_step_data.set_tc_outfeed_time_ms(
      tsl::profiler::PicoToMilli(tc_outfeed_time_in_ps.avg()));
  // The TC compute time is the minimum of the optimal step time across cores.
  per_step_data.set_tc_compute_time_ms(
      tsl::profiler::PicoToMilli(optimal_step_time_ps.min()));
  per_step_data.set_host_transfer_ms(
      tsl::profiler::PicoToMilli(host_send_recv_time_ps.max()));
  // TODO(b/153730997): Use the maximum step time.
  // The infeed time is the step time across cores minus all other times.
  // Previously, we used the maximum step time but changed to use the minimum
  // step time to work around b/153730997.
  // Uses the max TC infeed duration across cores as the step's TC infeed
  // duration.
  per_step_data.set_tc_infeed_time_ms(
      tsl::profiler::PicoToMilli(max_infeed.duration_ps));
  if (max_infeed.core_id.has_value()) {
    per_step_data.set_coreid_max_infeed_time(max_infeed.core_id.value());
    if (core_details_map.contains(max_infeed.core_id.value())) {
      const CoreDetails& core_details =
          core_details_map.at(max_infeed.core_id.value());
      per_step_data.set_max_infeed_time_core_name(absl::StrCat(
          core_details.hostname(), ":", core_details.device_ordinal()));
    }
  }

  per_step_data.set_scv0_compute_time_ms(
      tsl::profiler::PicoToMilli(max_wait_for_scv0.scv0_compute_duration_ps));
  per_step_data.set_scv0_infeed_time_ms(
      tsl::profiler::PicoToMilli(max_wait_for_scv0.scv0_infeed_duration_ps));

  // The TC idle time is the time TC spends waiting for the host but not
  // waiting for input.
  per_step_data.set_tc_idle_time_ms(
      tsl::profiler::PicoToMilli(step_stats_in_ps.min()) -
      NonIdleTimeMs(per_step_data));
  if (per_step_data.tc_idle_time_ms() < 0) {
    per_step_data.set_tc_idle_time_ms(0);
  }

  per_step_data.set_all_reduce_compute_time_ms(
      tsl::profiler::PicoToMilli(max_all_reduce.compute_duration_ps));
  per_step_data.set_all_reduce_sync_time_ms(
      tsl::profiler::PicoToMilli(max_all_reduce.sync_duration_ps));

  per_step_data.set_infeed_percent_average(infeed_percent_stats.avg());
  per_step_data.set_infeed_percent_minimum(infeed_percent_stats.min());
  per_step_data.set_infeed_percent_maximum(infeed_percent_stats.max());

  per_step_data.set_sc_infeed_time_ms(
      tsl::profiler::PicoToMilli(sc_infeed_time_in_ps.avg()));
  per_step_data.set_sc_outfeed_time_ms(
      tsl::profiler::PicoToMilli(sc_outfeed_time_in_ps.avg()));
  per_step_data.set_sc_compute_time_ms(
      tsl::profiler::PicoToMilli(sc_optimal_step_time_ps.min()));
  per_step_data.set_sc_idle_time_ms(
      tsl::profiler::PicoToMilli(sc_idle_time_in_ps.avg()));
  per_step_data.set_sc_step_time_ms(
      tsl::profiler::PicoToMilli(sc_step_stats_in_ps.avg()));
  if (per_step_data.sc_idle_time_ms() < 0) {
    per_step_data.set_sc_idle_time_ms(0);
  }
  return per_step_data;
}

TpuStepTimeBreakdown ComputeTpuStepTimeBreakdownInMs(
    const InputPipelineAnalysisResult& analysis, bool has_sparse_core) {
  tsl::Stat<double> tc_compute_ms;
  tsl::Stat<double> tc_infeed_ms;
  tsl::Stat<double> tc_outfeed_ms;
  tsl::Stat<double> tc_idle_ms;
  tsl::Stat<double> scv0_compute_ms;
  tsl::Stat<double> scv0_infeed_ms;
  tsl::Stat<double> host_transfer_ms;
  tsl::Stat<double> sc_compute_ms;
  tsl::Stat<double> sc_infeed_ms;
  tsl::Stat<double> sc_outfeed_ms;
  tsl::Stat<double> sc_idle_ms;
  tsl::Stat<double> sc_step_time_ms;
  TpuStepTimeBreakdown result;

  for (const google::protobuf::Any& step_details : analysis.step_details()) {
    PerTpuStepDetails details;
    if (!step_details.UnpackTo(&details)) {
      LOG(ERROR) << "Unable to unpack step_details. Expected: tpu";
      // TODO(b/302086111): Switch back to DFATAL once absl is updated.
      DCHECK(false);
      return result;
    }
    tc_compute_ms.UpdateStat(details.tc_compute_time_ms());
    tc_idle_ms.UpdateStat(details.tc_idle_time_ms());
    tc_infeed_ms.UpdateStat(details.tc_infeed_time_ms());
    tc_outfeed_ms.UpdateStat(details.tc_outfeed_time_ms());
    scv0_compute_ms.UpdateStat(details.scv0_compute_time_ms());
    scv0_infeed_ms.UpdateStat(details.scv0_infeed_time_ms());
    host_transfer_ms.UpdateStat(details.host_transfer_ms());
    sc_compute_ms.UpdateStat(details.sc_compute_time_ms());
    sc_idle_ms.UpdateStat(details.sc_idle_time_ms());
    sc_infeed_ms.UpdateStat(details.sc_infeed_time_ms());
    sc_outfeed_ms.UpdateStat(details.sc_outfeed_time_ms());
    sc_step_time_ms.UpdateStat(details.sc_step_time_ms());
  }
  *result.mutable_tc_compute_ms_summary() =
      GetStepSummaryForSampleStats(tc_compute_ms);
  *result.mutable_scv0_compute_ms_summary() =
      GetStepSummaryForSampleStats(scv0_compute_ms);
  *result.mutable_tc_infeed_ms_summary() =
      GetStepSummaryForSampleStats(tc_infeed_ms);
  *result.mutable_tc_outfeed_ms_summary() =
      GetStepSummaryForSampleStats(tc_outfeed_ms);
  *result.mutable_scv0_infeed_ms_summary() =
      GetStepSummaryForSampleStats(scv0_infeed_ms);
  *result.mutable_tc_idle_ms_summary() =
      GetStepSummaryForSampleStats(tc_idle_ms);
  *result.mutable_host_transfer_ms_summary() =
      GetStepSummaryForSampleStats(host_transfer_ms);
  if (has_sparse_core) {
    auto* sparse_core_step_summary = result.mutable_sparse_core_step_summary();
    *sparse_core_step_summary->mutable_sc_compute_ms_summary() =
        GetStepSummaryForSampleStats(sc_compute_ms);
    *sparse_core_step_summary->mutable_sc_infeed_ms_summary() =
        GetStepSummaryForSampleStats(sc_infeed_ms);
    *sparse_core_step_summary->mutable_sc_outfeed_ms_summary() =
        GetStepSummaryForSampleStats(sc_outfeed_ms);
    *sparse_core_step_summary->mutable_sc_idle_ms_summary() =
        GetStepSummaryForSampleStats(sc_idle_ms);
    *sparse_core_step_summary->mutable_sc_step_time_ms_summary() =
        GetStepSummaryForSampleStats(sc_step_time_ms);
  }
  return result;
}

// Given the step sequence on each core, computes the result proto of the
// input-pipeline analysis tool (the InputPipelineAnalysisResult defined in
// input_pipeline.proto).
// Note on grouped_by_step: There is one element for each step executed (on
// multiple cores). Each element is a map from the core_id to the information
// of the step that runs on that core. Elements are in the same order that the
// steps are executed over time.
InputPipelineAnalysisResult ComputeTpuInputPipelineAnalysisResult(
    const tsl::protobuf::RepeatedPtrField<PerCoreStepInfo>& grouped_by_step,
    const tsl::protobuf::Map<uint32_t, tensorflow::profiler::CoreDetails>&
        core_details_map) {
  InputPipelineAnalysisResult result;
  bool has_sparse_core = false;
  for (const auto& [core_id, core_details] : core_details_map) {
    has_sparse_core |= core_details.is_sparse_core();
  }

  // Computes the summary of step time in ms.
  *result.mutable_step_time_summary() =
      ComputeStepTimeSummaryInMs(grouped_by_step);

  // Summary of the statistics of infeed time as percentage of the step
  // time.
  tsl::Stat<double> infeed_summary_stats_in_percent;
  for (const auto& coreid_stepinfo_map : grouped_by_step) {
    // Compute each TPU step stats.
    const PerTpuStepDetails& per_step_data =
        ComputeTpuPerStepDataAcrossCores(coreid_stepinfo_map, core_details_map);
    result.add_step_details()->PackFrom(per_step_data);

    // The infeed summary is based on the maximum infeed time across cores at
    // each step.
    infeed_summary_stats_in_percent.UpdateStat(
        per_step_data.infeed_percent_maximum());
  }

  // Computes the summary of infeed time as percentage of step time.
  *result.mutable_input_percent_summary() =
      GetStepSummaryForSampleStats(infeed_summary_stats_in_percent);

  // Computes the breakdown of step time
  TpuStepTimeBreakdown tpu_step_time_breakdown =
      ComputeTpuStepTimeBreakdownInMs(result, has_sparse_core);
  result.mutable_step_time_breakdown()->PackFrom(tpu_step_time_breakdown);
  result.set_tag(true);

  return result;
}

// Returns true if device_op_metrics_db contains an infeed op.
bool HasTpuInfeedOp(const OpMetricsDb& device_op_metrics_db) {
  for (const OpMetrics& metrics : device_op_metrics_db.metrics_db()) {
    if (tsl::profiler::IsHostOrSparseCoreV0Infeed(metrics.category())) {
      return true;
    }
  }
  return false;
}

// Returns the time spent waiting for input for generic hardware.
uint64_t TotalInputPs(const StepDetails& step_details) {
  uint64_t total_input_ps = 0;
  for (const auto& event : step_details.Events()) {
    if (event.type == HOST_WAIT_INPUT || event.type == HOST_TO_DEVICE) {
      // Includes both the time where the host was waiting input and the time
      // where the host was sending data to the device.
      total_input_ps += event.span.duration_ps();
    }
  }
  return total_input_ps;
}

void TensorCoreIdleAnalysis(bool all_cores_profiled, double tc_idle_percent,
                            std::string* input_classification,
                            std::string* input_statement,
                            std::string* tc_idle_classification,
                            std::string* tc_idle_statement) {
  // In MayFixTpuStepAnalysis(), we have already separated the idle time from
  // the input time. So, we don't need to substract the input time from the
  // idle time here.
  if (tc_idle_percent < kTcIdleThresholdInPercent) {
    *tc_idle_classification = "no";
    *tc_idle_statement = "";
    return;
  }
  std::string idle_percent_str = absl::StrFormat("%.1lf", tc_idle_percent);
  if (all_cores_profiled) {
    // Significant idle time with all cores profiled.
    *tc_idle_classification = "yes";
    *tc_idle_statement =
        absl::StrCat(idle_percent_str,
                     " % of the total step time sampled is due to host "
                     "overhead that is not input-related. For TF 2.x, you may "
                     "want to use a ",
                     AnchorElement(kMultipleStepsInTffunctionDoc,
                                   "host-training loop (i.e. running multiple "
                                   "steps within a tf.function)."));
    return;
  }

  // Significant idle time without all cores profiled.
  if (*input_classification == "host") {
    // We've already identified that it is input bound. So, no need to issue
    // more warnings.
    *tc_idle_classification = "no";
    *tc_idle_statement = "";
    return;
  }

  *input_classification = "host";  // focuses on "host" first.
  *input_statement = absl::StrCat(
      "Your program COULD be input-bound because ", idle_percent_str,
      "% of the total step time is idle. This may be a manifestation of an "
      "input issue on a worker "
      "machine that was not profiled. To be certain, please profile ALL "
      "worker machines in your job by following ",
      AnchorElement(kProfileAllHostsDoc, "this instruction."));
  *tc_idle_classification = "no";
  *tc_idle_statement = "";
}

void AllReduceAnalysis(bool all_cores_profiled,
                       double all_reduce_compute_percent,
                       double all_reduce_sync_percent, double input_percent,
                       std::string* input_classification,
                       std::string* input_statement,
                       std::string* all_reduce_classification,
                       std::string* all_reduce_statement) {
  double all_reduce_percent =
      all_reduce_compute_percent + all_reduce_sync_percent;
  // Since all-reduce time is overlapped with the input time, we consider the
  // all-reduce time that is not input related.
  double all_reduce_not_input_related_percent =
      all_reduce_percent - input_percent;

  if (all_reduce_not_input_related_percent <
      kAllReduceBoundThresholdInPercent) {
    // Insignificant time spent on all-reduce.
    *all_reduce_classification = "no";
    *all_reduce_statement = "";
    return;
  }

  if (all_cores_profiled) {
    // Significant time spent on all-reduce with all cores profiled.
    std::string all_reduce_compute_percent_str =
        absl::StrFormat("%.1lf", all_reduce_compute_percent);
    std::string all_reduce_sync_percent_str =
        absl::StrFormat("%.1lf", all_reduce_sync_percent);
    *all_reduce_classification = "yes";
    *all_reduce_statement = absl::StrCat(
        "Also, ", all_reduce_sync_percent_str,
        " % of the total step time sampled is spent on synchronization with "
        "other TPU cores, and ",
        all_reduce_compute_percent_str,
        " % of the total step time sampled is spent on actual AllReduce.");
    return;
  }

  // Significant time spent on all-reduce and not all cores were profiled.
  std::string all_reduce_percent_str =
      absl::StrFormat("%.1lf", all_reduce_percent);

  if (*input_classification != "device") {
    // InputAnalysis() already indicates some potential input issue. So, we
    // can focus on all-reduce performance.
    *all_reduce_classification = "yes";
    *all_reduce_statement = absl::StrCat(
        "Also, ", all_reduce_percent_str,
        " % of the total step time sampled is spent on synchronization "
        "with "
        "other TPU cores and AllReduce. Not all worker machines are "
        "profiled, "
        "therefore "
        "we "
        "cannot disambiguate the actual time for AllReduce from the "
        "synchronization. To be certain, please profile ALL "
        "worker machines in your job by following ",
        AnchorElement(kProfileAllHostsDoc, "this instruction."));
    return;
  }

  // InputAnalysis() indicates that it is NOT input-bound. However, it may
  // be because the input delay is manifested as all-reduce time. So,
  // attribute it to a possible input issue.
  *input_classification = "host";  // focuses on "host" first.
  *input_statement = absl::StrCat(
      "Your program COULD be input-bound because ", all_reduce_percent_str,
      "% of the total step time is spent on synchronization with other "
      "TPU cores. This may be a manifestation of an input issue on a "
      "worker "
      "machine that was not profiled. To be certain, please profile ALL "
      "worker machines in your job by following ",
      AnchorElement(kProfileAllHostsDoc, "this instruction."));
  *all_reduce_classification = "no";
  *all_reduce_statement = "";
}

void ScV0Analysis(double scv0_percent, std::string* scv0_classification,
                  std::string* scv0_statement) {
  if (scv0_percent == 0) {
    *scv0_classification = "no";
    *scv0_statement = "";
    return;
  }
  std::string scv0_percent_str = absl::StrFormat("%.1lf", scv0_percent);
  if (scv0_percent < kModeratelySparseCoreV0BoundThresholdInPercent) {
    *scv0_classification = "moderate";
    *scv0_statement = absl::StrCat(
        "Also, ", scv0_percent_str,
        " % of the total step time sampled is spent on the ", kSparseCoreV0Name,
        " compute. You may also want to reduce the ", kSparseCoreV0Name,
        " compute time.");
    return;
  }
  *scv0_classification = "high";
  *scv0_statement = absl::StrCat(
      "Also, ", scv0_percent_str,
      " % of the total step time sampled is spent on the ", kSparseCoreV0Name,
      " compute. You should focus on reducing the ", kSparseCoreV0Name,
      " compute time as well.");
}

// A map keeps track of the minimum value associated with an id.
class MinMap {
 public:
  void Observe(uint64_t id, uint64_t value) {
    auto [iter, inserted] = min_map_.try_emplace(id, value);
    if (!inserted && iter->second > value) {
      iter->second = value;
    }
  }

  uint64_t Min(uint64_t id) const {
    auto iter = min_map_.find(id);
    return (iter != min_map_.end()) ? iter->second : 0;
  }

 private:
  absl::flat_hash_map<uint64_t /*id*/, uint64_t /*min*/> min_map_;
};

}  // namespace

StepSummary GetStepSummaryForSampleStats(
    const tsl::Stat<double>& sample_stats) {
  StepSummary step_time_summary;
  double avg, sdv, min, max;
  if (sample_stats.empty()) {
    // If sample_stats is empty, sample_stats.avg() will return NaN. However, we
    // prefer to show an 0 instead.
    avg = sdv = min = max = 0.0;
  } else {
    avg = sample_stats.avg();
    sdv = sqrt(sample_stats.sample_variance());
    min = sample_stats.min();
    max = sample_stats.max();
  }
  step_time_summary.set_average(avg);
  step_time_summary.set_standard_deviation(sdv);
  step_time_summary.set_minimum(min);
  step_time_summary.set_maximum(max);
  return step_time_summary;
}

PerCoreAllReduceBreakdown ComputePerStepAllReduceBreakdownAcrossCores(
    const PerCoreStepInfo& coreid_stepinfo_map) {
  PerCoreAllReduceBreakdown result;
  MinMap min_duration_map;
  for (const auto& [core_id, all_reduce_db] :
       coreid_stepinfo_map.all_reduce_db_per_core()) {
    for (const auto& all_reduce : all_reduce_db.all_reduce_info()) {
      uint64_t duration_ps =
          all_reduce.end_time_ps() - all_reduce.start_time_ps();
      min_duration_map.Observe(all_reduce.id(), duration_ps);
    }
  }
  for (const auto& [core_id, all_reduce_db] :
       coreid_stepinfo_map.all_reduce_db_per_core()) {
    AllReduceBreakdown& breakdown = result[core_id];
    for (const auto& all_reduce : all_reduce_db.all_reduce_info()) {
      uint64_t duration_ps =
          all_reduce.end_time_ps() - all_reduce.start_time_ps();
      uint64_t min_duration_ps = min_duration_map.Min(all_reduce.id());
      breakdown.compute_duration_ps += min_duration_ps;
      breakdown.sync_duration_ps += duration_ps - min_duration_ps;
    }
  }
  return result;
}

void MayFixTpuStepAnalysis(
    const StepEvents& host_step_events, const OpMetricsDb& device_op_metrics_db,
    StepDatabaseResult& step_db,
    const tsl::protobuf::Map<uint32_t, tensorflow::profiler::CoreDetails>&
        core_details_map) {
  // This code is only applicable when input is received by the tensor core
  // from the host without the use of infeed. If the tensor core receives
  // input via host infeed or via sparsecorev0 infeed, there's nothing to do.
  if (HasTpuInfeedOp(device_op_metrics_db)) return;

  for (PerCoreStepInfo& per_core_step_info :
       *(step_db.mutable_step_sequence())) {
    uint32_t step_num = per_core_step_info.step_num();
    // TODO(ckluk): step_num is obtained from tf_op_stats, which is based on the
    // step-tracking mechanism with the on-device training loop. However, this
    // step_num is different from the group_id. So, what we are doing here is
    // only an approximation, assuming that all steps exhibit similar
    // breakdown. Once grouping works on TPU device, we need to replace step_num
    // by the group_id from TPU device.
    const StepDetails* step_details =
        gtl::FindOrNull(host_step_events, step_num);
    if (step_details == nullptr) {
      continue;  // step_num not in host_step_events, we don't know how to fix.
    }
    uint64_t total_input_ps = TotalInputPs(*step_details);
    if (total_input_ps == 0) {
      continue;  // no host input events.
    }
    PerTpuStepDetails tpu_step_data =
        ComputeTpuPerStepDataAcrossCores(per_core_step_info, core_details_map);
    double tc_idle_ms = tpu_step_data.tc_idle_time_ms();
    double adjusted_input_ratio =
        std::min(tsl::profiler::SafeDivide(
                     tsl::profiler::PicoToMilli(total_input_ps), tc_idle_ms),
                 1.0);
    for (auto& [core_id, step_info] :
         *per_core_step_info.mutable_step_info_per_core()) {
      // skip sparse cores for this.
      if (core_id >= kSparseCoreIndexStart) continue;
      TpuStepBreakdown tpu;
      if (TpuStepBreakdown tpu; step_info.step_breakdown().UnpackTo(&tpu)) {
        DCHECK_EQ(tpu.infeed_duration_ps(), 0);
        if (tpu.tc_idle_ps() > 0) {
          // Extract the infeed fraction of idle time.
          tpu.set_infeed_duration_ps(tpu.tc_idle_ps() * adjusted_input_ratio);
          tpu.set_tc_idle_ps(tpu.tc_idle_ps() - tpu.infeed_duration_ps());
          step_info.mutable_step_breakdown()->PackFrom(tpu);
        }
      } else if (tensorflow::profiler::GenericStepBreakdown generic;
                 step_info.step_breakdown().UnpackTo(&generic)) {
        uint64_t& infeed_time_ps =
            (*generic.mutable_category_ps())[xla::HloOpcodeString(
                xla::HloOpcode::kInfeed)];
        uint64_t& idle_time_ps =
            (*generic.mutable_category_ps())[tensorflow::profiler::kIdle];
        DCHECK_EQ(infeed_time_ps, 0);
        if (idle_time_ps > 0) {
          infeed_time_ps = idle_time_ps * adjusted_input_ratio;
          idle_time_ps -= infeed_time_ps;
          step_info.mutable_step_breakdown()->PackFrom(generic);
        }
      } else {
        // Likely encountered an ScStepBreakdown instance which can be skipped
        // as we only care about attributing TC idle time to host.
        LOG(INFO) << "Unable to unpack step_breakdown.";
      }
    }
  }
}

TpuBottleneckAnalysis ComputeTpuBottleneckAnalysis(
    bool all_cores_profiled, const InputPipelineAnalysisResult& result) {
  double total_step_time_ms = 0;
  double total_infeed_time_ms = 0;
  double total_tc_outfeed_time_ms = 0;
  double total_scv0_compute_time_ms = 0;
  double total_all_reduce_compute_time_ms = 0;
  double total_all_reduce_sync_time_ms = 0;
  double total_tc_idle_time_ms = 0;

  TpuBottleneckAnalysis analysis;
  for (const google::protobuf::Any& step_details : result.step_details()) {
    PerTpuStepDetails details;
    if (!step_details.UnpackTo(&details)) {
      LOG(ERROR) << "Unable to unpack step_details. Expected: tpu";
      // TODO(b/302086111): Switch back to DFATAL once absl is updated.
      DCHECK(false);
      return analysis;
    }
    total_step_time_ms += StepTimeMs(details);
    total_infeed_time_ms += InfeedTimeMs(details);
    total_tc_outfeed_time_ms += details.tc_outfeed_time_ms();
    total_scv0_compute_time_ms += details.scv0_compute_time_ms();
    total_all_reduce_compute_time_ms += details.all_reduce_compute_time_ms();
    total_all_reduce_sync_time_ms += details.all_reduce_sync_time_ms();
    total_tc_idle_time_ms += details.tc_idle_time_ms();
  }
  if (total_step_time_ms == 0) {
    analysis.set_input_classification("unknown");
    analysis.set_input_statement(
        "No step time measured. Therefore we cannot tell where the performance "
        "bottleneck is.");
    analysis.set_tc_idle_classification("no"),
        analysis.set_tc_idle_statement("");
    analysis.set_scv0_classification("no");
    analysis.set_scv0_statement("");
    analysis.set_all_reduce_classification("no");
    analysis.set_all_reduce_statement("");
    return analysis;
  }

  double infeed_percent = 100.0 * total_infeed_time_ms / total_step_time_ms;
  std::string input_classification;
  std::string input_statement;
  InputAnalysis(infeed_percent, /*all_other_percent=*/0, &input_classification,
                &input_statement);

  double tc_outfeed_percent =
      100.0 * total_tc_outfeed_time_ms / total_step_time_ms;
  std::string output_classification;
  std::string output_statement;
  OutputAnalysis(tc_outfeed_percent, &output_classification, &output_statement);

  double tc_idle_percent = 100.0 * total_tc_idle_time_ms / total_step_time_ms;
  std::string tc_idle_classification;
  std::string tc_idle_statement;
  TensorCoreIdleAnalysis(all_cores_profiled, tc_idle_percent,
                         &input_classification, &input_statement,
                         &tc_idle_classification, &tc_idle_statement);

  double all_reduce_compute_percent =
      100.0 * total_all_reduce_compute_time_ms / total_step_time_ms;
  double all_reduce_sync_percent =
      100.0 * total_all_reduce_sync_time_ms / total_step_time_ms;
  std::string all_reduce_classification;
  std::string all_reduce_statement;
  AllReduceAnalysis(all_cores_profiled, all_reduce_compute_percent,
                    all_reduce_sync_percent, infeed_percent,
                    &input_classification, &input_statement,
                    &all_reduce_classification, &all_reduce_statement);

  double scv0_percent = 100.0 * total_scv0_compute_time_ms / total_step_time_ms;
  std::string scv0_classification;
  std::string scv0_statement;
  ScV0Analysis(scv0_percent, &scv0_classification, &scv0_statement);

  // compute_percent includes both TC and ScV0 compute.
  double compute_percent = std::max(
      0.0, 100.0 - infeed_percent - tc_outfeed_percent - tc_idle_percent);

  analysis.set_compute_percent(compute_percent);
  analysis.set_input_percent(infeed_percent);
  analysis.set_output_percent(tc_outfeed_percent);
  analysis.set_tc_idle_percent(tc_idle_percent);
  analysis.set_input_classification(input_classification);
  analysis.set_input_statement(input_statement);
  analysis.set_output_statement(output_statement);
  analysis.set_tc_idle_classification(tc_idle_classification),
      analysis.set_tc_idle_statement(tc_idle_statement);
  analysis.set_scv0_classification(scv0_classification);
  analysis.set_scv0_statement(scv0_statement);
  analysis.set_all_reduce_classification(all_reduce_classification);
  analysis.set_all_reduce_statement(all_reduce_statement);
  return analysis;
}

void GenerateHostResult(const OpMetricsDb& host_tf_metrics_db,
                        InputPipelineAnalysisResult* result) {
  InputOpMetrics input_op_metrics = SelectInputOpMetrics(host_tf_metrics_db);
  // Returns if the program is not using an input pipeline with
  // instrumentation and hence no input ops are found.
  if (input_op_metrics.input_op_metrics.empty()) return;

  absl::flat_hash_map<InputOpCategory, double> aggregated_input_op_times_us;
  for (const OpMetrics* op_metrics : input_op_metrics.input_op_metrics) {
    InputOpCategory category =
        CategorizeInputOp(op_metrics->name(), op_metrics->category());
    *result->add_input_op_details() = ConvertOpMetricsToInputOpDetails(
        *op_metrics, input_op_metrics.input_op_time_ps, category);
    aggregated_input_op_times_us[category] +=
        tsl::profiler::PicoToMicro(op_metrics->self_time_ps());
  }

  double enqueue_time_us =
      aggregated_input_op_times_us[InputOpCategory::kEnqueue];
  double total_input_op_time_us =
      aggregated_input_op_times_us[InputOpCategory::kDemandedFileRead] +
      aggregated_input_op_times_us[InputOpCategory::kAdvancedFileRead] +
      aggregated_input_op_times_us[InputOpCategory::kPreprocessing];

  double ratio = std::min(
      1.0, RatioOfHostToDeviceTimeToStepTime(host_tf_metrics_db, *result));
  DCHECK_GE(ratio, 0.0);
  double non_enqueue_time_us = (ratio != 0.0)
                                   ? (enqueue_time_us * (1.0 - ratio) / ratio)
                                   : total_input_op_time_us;

  // Scales the various input-time components wrt to non_enqueue_time_us.
  double scaled_demanded_fileread_time_us = tsl::profiler::SafeDivide(
      non_enqueue_time_us *
          aggregated_input_op_times_us[InputOpCategory::kDemandedFileRead],
      total_input_op_time_us);
  double scaled_advanced_fileread_time_us = tsl::profiler::SafeDivide(
      non_enqueue_time_us *
          aggregated_input_op_times_us[InputOpCategory::kAdvancedFileRead],
      total_input_op_time_us);
  double scaled_preprocessing_time_us = tsl::profiler::SafeDivide(
      non_enqueue_time_us *
          aggregated_input_op_times_us[InputOpCategory::kPreprocessing],
      total_input_op_time_us);
  double unclassified_non_enqueue_time_us = std::max(
      0.0, non_enqueue_time_us - scaled_demanded_fileread_time_us -
               scaled_advanced_fileread_time_us - scaled_preprocessing_time_us);

  InputTimeBreakdown* input_time_breakdown =
      result->mutable_input_time_breakdown();
  input_time_breakdown->set_enqueue_us(enqueue_time_us);
  input_time_breakdown->set_demanded_file_read_us(
      scaled_demanded_fileread_time_us);
  input_time_breakdown->set_advanced_file_read_us(
      scaled_advanced_fileread_time_us);
  input_time_breakdown->set_preprocessing_us(scaled_preprocessing_time_us);
  input_time_breakdown->set_unclassified_non_enqueue_us(
      unclassified_non_enqueue_time_us);
}

InputPipelineAnalysisRecommendation GenerateRecommendation() {
  const absl::string_view kDatasetIntro =
      "https://www.tensorflow.org/programmers_guide/datasets";

  const absl::string_view kDatasetTopic =
      "https://www.tensorflow.org/api_docs/python/tf/data/Dataset#";

  const absl::string_view kTfRecordDataset =
      "https://www.tensorflow.org/api_docs/python/tf/data/"
      "TFRecordDataset#class_tfrecorddataset";

  InputPipelineAnalysisRecommendation recommendation;
  *recommendation.add_details() =
      "Enqueuing data: you may want to combine small input data chunks "
      "into fewer "
      "but larger chunks.";
  *recommendation.add_details() = absl::StrCat(
      "Data preprocessing: you may increase num_parallel_calls in ",
      AnchorElement(absl::StrCat(kDatasetTopic, "map"), "Dataset map()"),
      " or preprocess the data OFFLINE.");
  *recommendation.add_details() = absl::StrCat(
      "Reading data from files in advance: you may tune parameters in the "
      "following tf.data API (",
      AnchorElement(absl::StrCat(kDatasetTopic, "prefetch"), "prefetch size"),
      ", ",
      AnchorElement(absl::StrCat(kDatasetTopic, "interleave"),
                    "interleave cycle_length"),
      ", ", AnchorElement(kTfRecordDataset, "reader buffer_size"), ")");
  *recommendation.add_details() = absl::StrCat(
      "Reading data from files on demand: you should read data IN ADVANCE "
      "using the following tf.data API (",
      AnchorElement(absl::StrCat(kDatasetTopic, "prefetch"), "prefetch"), ", ",
      AnchorElement(absl::StrCat(kDatasetTopic, "interleave"), "interleave"),
      ", ", AnchorElement(kTfRecordDataset, "reader buffer"), ")");
  *recommendation.add_details() = absl::StrCat(
      "Other data reading or processing: you may consider using the ",
      AnchorElement(kDatasetIntro, "tf.data API"),
      " (if you are not using it now)");

  return recommendation;
}

StepSummary ComputeStepTimeSummaryInMs(
    const tsl::protobuf::RepeatedPtrField<PerCoreStepInfo>& grouped_by_step) {
  tsl::Stat<double> total_step_stats_in_ms;
  // iterates over each step.
  for (const auto& coreid_stepinfo_map : grouped_by_step) {
    double max_per_step_stats_in_ms = 0.0;
    // iterates over each core.
    for (const auto& coreid_and_stepinfo :
         coreid_stepinfo_map.step_info_per_core()) {
      if (coreid_and_stepinfo.first >= kSparseCoreIndexStart) continue;
      const auto& step_info = coreid_and_stepinfo.second;
      max_per_step_stats_in_ms = std::max(step_info.duration_ps() / kNumPsPerMs,
                                          max_per_step_stats_in_ms);
    }
    // Step time of each step is determined by the slowest core.
    total_step_stats_in_ms.UpdateStat(max_per_step_stats_in_ms);
  }

  return GetStepSummaryForSampleStats(total_step_stats_in_ms);
}

InputPipelineAnalysisResult ConvertOpStatsToInputPipelineAnalysis(
    const OpStats& op_stats) {
  const HardwareType hardware_type = op_stats.run_environment().hardware_type();

  InputPipelineAnalysisResult result;
  if (hardware_type == tensorflow::profiler::TPU) {
    result = ComputeTpuInputPipelineAnalysisResult(
        op_stats.step_db().step_sequence(), op_stats.core_id_to_details());
  } else {
    result = ComputeGenericInputPipelineAnalysisResult(
        op_stats.step_db().step_sequence());
  }
  result.set_hardware_type(HardwareType_Name(hardware_type));

  PopulateStepDiagnostics(op_stats, result.mutable_diagnostics());
  GenerateHostResult(op_stats.host_op_metrics_db(), &result);

  InputPipelineAnalysisRecommendation recommendation = GenerateRecommendation();
  if (hardware_type == tensorflow::profiler::TPU) {
    TpuBottleneckAnalysis bottleneck_analysis = ComputeTpuBottleneckAnalysis(
        /*all_cores_profiled=*/true, result);
    result.set_input_percent(bottleneck_analysis.input_percent());
    result.set_output_percent(bottleneck_analysis.output_percent());
    result.set_idle_percent(bottleneck_analysis.tc_idle_percent());
    result.set_compute_percent(bottleneck_analysis.compute_percent());

    recommendation.mutable_bottleneck_analysis()->PackFrom(bottleneck_analysis);
    *recommendation.mutable_summary_next_step() =
        GetSummaryNextStep(bottleneck_analysis.input_classification(),
                           result.input_time_breakdown());
  } else {
    BottleneckAnalysis bottleneck_analysis = ComputeBottleneckAnalysis(
        result.input_time_breakdown(), result.step_details());
    result.set_input_percent(bottleneck_analysis.input_percent());
    result.set_output_percent(bottleneck_analysis.output_percent());
    result.set_idle_percent(bottleneck_analysis.idle_percent());
    result.set_compute_percent(bottleneck_analysis.compute_percent());
    recommendation.mutable_bottleneck_analysis()->PackFrom(bottleneck_analysis);
    *recommendation.mutable_summary_next_step() =
        GetSummaryNextStep(bottleneck_analysis.input_classification(),
                           result.input_time_breakdown());
  }

  *result.mutable_recommendation() = recommendation;
  return result;
}

bool InputAnalysis(double input_percent, double all_other_percent,
                   std::string* input_classification,
                   std::string* input_statement) {
  absl::string_view non_input_time = "other time";
  if (input_percent >= kHighlyInfeedBoundThresholdInPercent) {
    *input_classification = "host";
    *input_statement = absl::StrCat(
        "Your program is HIGHLY input-bound because ", OneDigit(input_percent),
        "% of the total step time sampled is waiting for input. Therefore, you "
        "should first focus on reducing the input time.");
    return false;
  } else if (input_percent >= kModeratelyInfeedBoundThresholdInPercent) {
    *input_classification = "both";
    *input_statement = absl::StrCat(
        "Your program is MODERATELY input-bound because ",
        OneDigit(input_percent),
        "% of the total step time sampled is waiting for input. Therefore, "
        "you would need to reduce both the input time and ",
        non_input_time, ".");
    return false;
  } else if (all_other_percent >= kModeratelyAllOtherBoundThresholdInPercent) {
    // Input analysis says it is not input-bound, but "All-Other" time
    // is significant. It could still be input-bound (or Python overhead).
    *input_classification = "both";
    *input_statement = absl::StrCat(
        "Your program is POTENTIALLY input-bound because ",
        OneDigit(all_other_percent),
        "% of the total step time sampled is spent on 'All Others' time (which "
        "could be due to I/O or Python execution or both).");
    return true;
  } else {
    // Definitely not input-bound.
    *input_classification = "device";
    *input_statement =
        absl::StrCat("Your program is NOT input-bound because only ",
                     OneDigit(input_percent),
                     "% of the total step time sampled is waiting for "
                     "input. Therefore, you should focus on "
                     "reducing ",
                     non_input_time, ".");
    return false;
  }
}

void OutputAnalysis(double output_percent, std::string* output_classification,
                    std::string* output_statement) {
  if (output_percent >= kHighlyOutfeedBoundThresholdInPercent) {
    *output_classification = "host";
    *output_statement = absl::StrCat(
        "Your program is HIGHLY output-bound because ",
        OneDigit(output_percent),
        "% of the total step time sampled is spent on output. Therefore, you "
        "should first focus on reducing the output time.");
  } else if (output_percent >= kModeratelyOutfeedBoundThresholdInPercent) {
    *output_classification = "both";
    *output_statement = absl::StrCat(
        "Your program is MODERATELY output-bound because ",
        OneDigit(output_percent),
        "% of the total step time sampled is spent on output. Therefore, "
        "you would need to reduce both the output time and other time.");
  } else {
    *output_classification = "device";
    *output_statement = "";
  }
}

BottleneckAnalysis ComputeBottleneckAnalysis(
    const InputTimeBreakdown& input_time_breakdown,
    const tsl::protobuf::RepeatedPtrField<::google::protobuf::Any>&
        any_step_details) {
  double total_step_time_ms = 0;
  double total_input_ms = 0;
  double total_output_ms = 0;
  double total_host_compute_ms = 0;
  double total_host_prepare_ms = 0;
  double total_host_compile_ms = 0;
  double total_device_compute_ms = 0;
  double total_device_to_device_ms = 0;
  double total_device_collectives_ms = 0;
  double total_unknown_ms = 0;

  for (const google::protobuf::Any& step_details : any_step_details) {
    PerGenericStepDetails details;
    bool success = step_details.UnpackTo(&details);
    if (!success && !step_details.type_url().empty()) {
      LOG(ERROR) << "Unable to unpack step_breakdown. Expected: generic"
                 << std::endl;
      return {};
    }
    total_step_time_ms += details.step_time_ms();
    total_input_ms +=
        details.host_wait_input_ms() + details.host_to_device_ms();
    total_output_ms += details.output_ms();
    total_host_prepare_ms += details.host_prepare_ms();
    total_device_compute_ms += details.device_compute_ms();
    total_device_to_device_ms += details.device_to_device_ms();
    total_device_collectives_ms += details.device_collectives_ms();
    total_host_compute_ms += details.host_compute_ms();
    total_host_compile_ms += details.host_compile_ms();
    total_unknown_ms += details.unknown_time_ms();
  }

  if (total_step_time_ms == 0) {
    BottleneckAnalysis analysis;
    analysis.set_input_classification("unknown");
    analysis.set_input_statement(
        "No step time measured. Therefore we cannot tell where the "
        "performance bottleneck is.");
    analysis.set_kernel_launch_classification("no");
    analysis.set_kernel_launch_statement("");
    analysis.set_all_other_classification("no");
    analysis.set_all_other_statement("");
    analysis.set_device_collectives_classification("no");
    analysis.set_device_collectives_statement("");
    return analysis;
  }
  double input_percent = 100.0 * total_input_ms / total_step_time_ms;
  double output_percent = 100.0 * total_output_ms / total_step_time_ms;
  double compute_percent = 100.0 * total_device_compute_ms / total_step_time_ms;
  double device_collectives_percent =
      100.0 * total_device_collectives_ms / total_step_time_ms;

  // idle_percent includes host_prepare (i.e. kernel launch, device-to-device,
  // host compute, host compile, and unknown.
  double idle_percent =
      std::max(0.0, 100.0 - input_percent - output_percent - compute_percent -
                        device_collectives_percent);
  double kernel_launch_percent =
      100.0 * total_host_prepare_ms / total_step_time_ms;
  double all_other_percent = 100.0 * total_unknown_ms / total_step_time_ms;

  std::string input_classification;
  std::string input_statement;
  bool all_other_reported =
      InputAnalysis(input_percent, all_other_percent, &input_classification,
                    &input_statement);

  std::string device_collectives_classification;
  std::string device_collectives_statement;
  DeviceCollectivesAnalysis(device_collectives_percent,
                            &device_collectives_classification,
                            &device_collectives_statement);

  std::string kernel_launch_classification;
  std::string kernel_launch_statement;
  KernelLaunchAnalysis(TfDataInUse(input_time_breakdown), kernel_launch_percent,
                       &kernel_launch_classification, &kernel_launch_statement);

  std::string all_other_classification;
  std::string all_other_statement;
  AllOtherAnalysis(all_other_reported, all_other_percent,
                   &all_other_classification, &all_other_statement);

  BottleneckAnalysis analysis;
  analysis.set_input_percent(input_percent);
  analysis.set_output_percent(output_percent);
  analysis.set_idle_percent(idle_percent);
  analysis.set_compute_percent(compute_percent);

  analysis.set_input_classification(input_classification);
  analysis.set_input_statement(input_statement);
  analysis.set_kernel_launch_classification(kernel_launch_classification);
  analysis.set_kernel_launch_statement(kernel_launch_statement);
  analysis.set_all_other_classification(all_other_classification);
  analysis.set_all_other_statement(all_other_statement);
  analysis.set_device_collectives_classification(
      device_collectives_classification);
  analysis.set_device_collectives_statement(device_collectives_statement);

  return analysis;
}

std::string GetSummaryNextStep(absl::string_view input_classification,
                               const InputTimeBreakdown& breakdown) {
  std::string summary_next_step;
  if (input_classification == "host" || input_classification == "both") {
    if (!TfDataInUse(breakdown)) {
      summary_next_step = absl::StrCat(
          "Consider using ", MakeDocLink(DatasetIntroDoc(), "the tf.data API"),
          " to enable profiler's host-side analysis for input pipeline. "
          "Profiler currently does not support custom input pipeline (please "
          "ignore "
          "Section ",
          kHostAnalysisSectionNumber, " below).");
    } else {
      summary_next_step =
          absl::StrCat("Look at Section ", kHostAnalysisSectionNumber,
                       " for the breakdown of input time on the host.");
    }
  } else {
    summary_next_step = "You may skip the rest of this page.";
  }

  return summary_next_step;
}

double HostToDeviceTransferAsPercentOfInputTime(
    const InputTimeBreakdown& breakdown) {
  // Thanks to the scaling trick we did in GenerateHostResult(), we can
  // estimate the percentage of input-time spent on host-to-device transfer in
  // the following way.
  double total_input_time_us =
      breakdown.demanded_file_read_us() + breakdown.advanced_file_read_us() +
      breakdown.preprocessing_us() + breakdown.enqueue_us() +
      breakdown.unclassified_non_enqueue_us();
  return 100.0 *
         tsl::profiler::SafeDivide(breakdown.enqueue_us(), total_input_time_us);
}

}  // namespace profiler
}  // namespace tensorflow
