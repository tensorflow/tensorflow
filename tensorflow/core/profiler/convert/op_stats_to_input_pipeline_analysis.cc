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

#include <algorithm>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/util/stats_calculator.h"

namespace tensorflow {
namespace profiler {

namespace {

const double kNumPsPerMs = 1000000000.0;

// If the percentage of step time that is due to infeed is less than
// kModeratelyInfeedBoundThresholdInPercent, it is considered NOT
// input-bound; else if it is less than
// kHighlyInfeedBoundThresholdInPercent, it is considered MODERATELY
// input-bound; else if it is considered HIGHLY input-bound.
constexpr double kModeratelyInfeedBoundThresholdInPercent = 5;
constexpr double kHighlyInfeedBoundThresholdInPercent = 20;
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

template <class Collection>
double GetTimeInMs(const Collection& type_ps, EventType event_type) {
  return PicosToMillis(gtl::FindWithDefault(type_ps, event_type, /*value=*/0));
}

StepSummary GetStepSummaryForSampleStats(const Stat<double>& sample_stats) {
  StepSummary step_time_summary;
  double avg, sdv, min, max;
  if (sample_stats.empty()) {
    // If sample_stats is empty, sample_stats.avg() will return NaN. However, we
    // prefer to show an 0 instead.
    avg = sdv = min = max = 0.0;
  } else {
    avg = sample_stats.avg();
    sdv = std::sqrt(sample_stats.sample_variance());
    min = sample_stats.min();
    max = sample_stats.max();
  }
  step_time_summary.set_average(avg);
  step_time_summary.set_standard_deviation(sdv);
  step_time_summary.set_minimum(min);
  step_time_summary.set_maximum(max);
  return step_time_summary;
}

GenericStepTimeBreakdown ComputeGenericStepTimeBreakdownInMs(
    const InputPipelineAnalysisResult& analysis) {
  Stat<double> unknown_time_ms;
  Stat<double> host_wait_input_ms;
  Stat<double> host_to_device_ms;
  Stat<double> input_ms;
  Stat<double> output_ms;
  Stat<double> device_compute_ms;
  Stat<double> device_to_device_ms;
  Stat<double> host_compute_ms;
  Stat<double> host_prepare_ms;
  Stat<double> host_compile_ms;
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
  *result.mutable_host_compute_ms_summary() =
      GetStepSummaryForSampleStats(host_compute_ms);
  *result.mutable_host_prepare_ms_summary() =
      GetStepSummaryForSampleStats(host_prepare_ms);
  *result.mutable_host_compile_ms_summary() =
      GetStepSummaryForSampleStats(host_compile_ms);
  return result;
}

InputPipelineAnalysisResult ComputeGenericInputPipelineAnalysisResult(
    const protobuf::RepeatedPtrField<PerCoreStepInfo>& grouped_by_step) {
  InputPipelineAnalysisResult result;

  // Computes the summary of step time in ms.
  *result.mutable_step_time_summary() =
      ComputeStepTimeSummaryInMs(grouped_by_step);

  Stat<double> input_summary_stats_in_percent;
  for (const auto& coreid_stepinfo_map : grouped_by_step) {
    // Iterates over each step.
    const auto* ptr =
        gtl::FindOrNull(coreid_stepinfo_map.step_info_per_core(), 0);
    if (ptr == nullptr) {
      // For generic hardware, all step-info is put under core-0. If ptr
      // is nullptr, it means there is no step at all.
      continue;
    }
    const StepInfoResult& step_info = *ptr;
    // Adds the details for a new step.
    PerGenericStepDetails details;
    details.set_step_number(step_info.step_num());
    details.set_step_time_ms(PicosToMillis(step_info.duration_ps()));
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
    details.set_device_compute_ms(GetTimeInMs(type_ps, DEVICE_COMPUTE));
    details.set_device_to_device_ms(GetTimeInMs(type_ps, DEVICE_TO_DEVICE) +
                                    GetTimeInMs(type_ps, DEVICE_WAIT_DEVICE));
    details.set_host_compute_ms(GetTimeInMs(type_ps, HOST_COMPUTE));
    details.set_host_prepare_ms(GetTimeInMs(type_ps, HOST_PREPARE));
    details.set_host_compile_ms(GetTimeInMs(type_ps, HOST_COMPILE));

    result.add_step_details()->PackFrom(details);

    const double input_percent_of_step_time =
        100.0 *
        SafeDivide(details.host_wait_input_ms() + details.host_to_device_ms(),
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

string InputOpCategoryString(InputOpCategory category) {
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
  return IsInfeedEnqueueOp(category) || IsDatasetOp(category) ||
         IsMemcpyHToDOp(category);
}

// TODO(ckluk):
//   Confirm with the tf.data team if the classification below is correct.
InputOpCategory CategorizeInputOp(absl::string_view name,
                                  absl::string_view category) {
  if (IsInfeedEnqueueOp(category) || IsMemcpyHToDOp(category)) {
    // Ops for sending input from host to device.
    return InputOpCategory::kEnqueue;
  }
  DCHECK(IsDatasetOp(category));
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
  details.set_time_in_ms(PicosToMillis(op_metrics.time_ps()));
  details.set_self_time_in_ms(PicosToMillis(op_metrics.self_time_ps()));
  details.set_time_in_percent(
      100.0 * SafeDivide(op_metrics.time_ps(), input_op_time_ps));
  details.set_self_time_in_percent(
      100.0 * SafeDivide(op_metrics.self_time_ps(), input_op_time_ps));
  details.set_category(InputOpCategoryString(category));
  return details;
}

string AnchorElement(absl::string_view url, absl::string_view text) {
  return absl::StrCat("<a href=\"", url, "\" target=\"_blank\">", text, "</a>");
}

// Returns the ratio of the host-to-device time in each step to the step-time.
double RatioOfHostToDeviceTimeToStepTime(
    const OpMetricsDb& host_tf_metrics_db,
    const InputPipelineAnalysisResult& input_pipeline_analysis) {
  if (host_tf_metrics_db.total_host_infeed_enq_start_timestamp_ps_diff() > 0) {
    // For TPU execution that uses infeed.
    //    We use total_host_infeed_enq_start_timestamp_ps_diff_ to approximate
    //    the total host step time.
    return std::min(
        1.0, SafeDivide(host_tf_metrics_db.total_host_infeed_enq_duration_ps(),
                        host_tf_metrics_db
                            .total_host_infeed_enq_start_timestamp_ps_diff()));
  }
  // For GPU and TPU execution that doesn't use infeed.
  double avg_step_time_ms =
      input_pipeline_analysis.step_time_summary().average();
  if (avg_step_time_ms > 0) {
    // Uses the on-device step time.
    GenericStepTimeBreakdown generic_breakdown;
    if (input_pipeline_analysis.step_time_breakdown().UnpackTo(
            &generic_breakdown)) {
      double avg_host_to_device_time_ms =
          generic_breakdown.host_to_device_ms_summary().average();
      return std::min(1.0,
                      SafeDivide(avg_host_to_device_time_ms, avg_step_time_ms));
    }
  }
  return 0.0;
}

void KernelLaunchAnalysis(double kernel_launch_percent, int* observation_index,
                          string* kernel_launch_classification,
                          string* kernel_launch_statement) {
  string percent_str = absl::StrFormat("%.1lf", kernel_launch_percent);
  if (kernel_launch_percent >= kHighlyKernelLaunchBoundThresholdInPercent) {
    *kernel_launch_classification = "high";
    *kernel_launch_statement = absl::StrCat(
        "(", ++*observation_index, ") ", percent_str,
        " % of the total step time sampled is spent on Kernel Launch.");
  } else if (kernel_launch_percent >=
             kModeratelyKernelLaunchBoundThresholdInPercent) {
    *kernel_launch_classification = "moderate";
    *kernel_launch_statement = absl::StrCat(
        "(", ++*observation_index, ") ", percent_str,
        " % of the total step time sampled is spent on Kernel Launch.");
  } else {
    *kernel_launch_classification = "no";
    *kernel_launch_statement = "";
  }
}

void AllOtherAnalysis(double all_other_percent, int* observation_index,
                      string* all_other_classification,
                      string* all_other_statement) {
  string percent_str = absl::StrFormat("%.1lf", all_other_percent);
  if (all_other_percent >= kHighlyAllOtherBoundThresholdInPercent) {
    *all_other_classification = "high";
    *all_other_statement = absl::StrCat(
        "(", ++*observation_index, ") ", percent_str,
        " % of the total step time sampled is spent on All Others time.");
  } else if (all_other_percent >= kModeratelyAllOtherBoundThresholdInPercent) {
    *all_other_classification = "moderate";
    *all_other_statement = absl::StrCat(
        "(", ++*observation_index, ") ", percent_str,
        " % of the total step time sampled is spent on All Others time.");
  } else {
    *all_other_classification = "no";
    *all_other_statement = "";
  }
}

}  // namespace

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
        PicosToMicros(op_metrics->self_time_ps());
  }

  double enqueue_time_us =
      aggregated_input_op_times_us[InputOpCategory::kEnqueue];
  double total_input_op_time_us =
      aggregated_input_op_times_us[InputOpCategory::kDemandedFileRead] +
      aggregated_input_op_times_us[InputOpCategory::kAdvancedFileRead] +
      aggregated_input_op_times_us[InputOpCategory::kPreprocessing];

  double ratio = RatioOfHostToDeviceTimeToStepTime(host_tf_metrics_db, *result);
  DCHECK_LE(ratio, 1.0);
  DCHECK_GE(ratio, 0.0);
  double non_enqueue_time_us = (ratio != 0.0)
                                   ? (enqueue_time_us * (1.0 - ratio) / ratio)
                                   : total_input_op_time_us;

  // Scales the various input-time components wrt to non_enqueue_time_us.
  double scaled_demanded_fileread_time_us = SafeDivide(
      non_enqueue_time_us *
          aggregated_input_op_times_us[InputOpCategory::kDemandedFileRead],
      total_input_op_time_us);
  double scaled_advanced_fileread_time_us = SafeDivide(
      non_enqueue_time_us *
          aggregated_input_op_times_us[InputOpCategory::kAdvancedFileRead],
      total_input_op_time_us);
  double scaled_preprocessing_time_us = SafeDivide(
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
      "following Dataset API (",
      AnchorElement(absl::StrCat(kDatasetTopic, "prefetch"), "prefetch size"),
      ", ",
      AnchorElement(absl::StrCat(kDatasetTopic, "interleave"),
                    "interleave cycle_length"),
      ", ", AnchorElement(kTfRecordDataset, "reader buffer_size"), ")");
  *recommendation.add_details() = absl::StrCat(
      "Reading data from files on demand: you should read data IN ADVANCE "
      "using the following Dataset API (",
      AnchorElement(absl::StrCat(kDatasetTopic, "prefetch"), "prefetch"), ", ",
      AnchorElement(absl::StrCat(kDatasetTopic, "interleave"), "interleave"),
      ", ", AnchorElement(kTfRecordDataset, "reader buffer"), ")");
  *recommendation.add_details() = absl::StrCat(
      "Other data reading or processing: you may consider using the ",
      AnchorElement(kDatasetIntro, "Dataset API"),
      " (if you are not using it now)");
  return recommendation;
}

StepSummary ComputeStepTimeSummaryInMs(
    const protobuf::RepeatedPtrField<PerCoreStepInfo>& grouped_by_step) {
  Stat<double> total_step_stats_in_ms;
  // iterates over each step.
  for (const auto& coreid_stepinfo_map : grouped_by_step) {
    double max_per_step_stats_in_ms = 0.0;
    // iterates over each core.
    for (const auto& coreid_and_stepinfo :
         coreid_stepinfo_map.step_info_per_core()) {
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
    const OpStats& op_stats, const HardwareType& hardware_type) {
  InputPipelineAnalysisResult result =
      ComputeGenericInputPipelineAnalysisResult(
          op_stats.step_db().step_sequence());
  result.set_hardware_type(hardware_type);
  GenerateHostResult(op_stats.host_op_metrics_db(), &result);
  *result.mutable_recommendation() = GenerateRecommendation();
  return result;
}

void InfeedAnalysis(HardwareType hardware_type, double infeed_percent,
                    int* observation_index, string* input_classification,
                    string* input_statement) {
  absl::string_view non_input_time = "other time";
  string infeed_percent_str = absl::StrFormat("%.1lf", infeed_percent);
  if (infeed_percent >= kHighlyInfeedBoundThresholdInPercent) {
    *input_classification = "host";
    *input_statement = absl::StrCat(
        "(", ++*observation_index, ") ",
        "Your program is HIGHLY input-bound because ", infeed_percent_str,
        "% of the total step time sampled is waiting for input. Therefore, "
        "you should first focus on reducing the input time.");
  } else if (infeed_percent >= kModeratelyInfeedBoundThresholdInPercent) {
    *input_classification = "both";
    *input_statement = absl::StrCat(
        "(", ++*observation_index, ") ",
        "Your program is MODERATELY input-bound because ", infeed_percent_str,
        "% of the total step time sampled is waiting for input. Therefore, "
        "you would need to reduce both the input time and ",
        non_input_time, ".");
  } else {
    *input_classification = "device";
    *input_statement = absl::StrCat(
        "(", ++*observation_index, ") ",
        "Your program is NOT input-bound because only ", infeed_percent_str,
        "% of the total step time sampled is waiting for "
        "input. Therefore, you should focus on "
        "reducing ",
        non_input_time, ".");
  }
}

GenericBottleneck GenericOverallBottleneck(
    const InputPipelineAnalysisResult& result) {
  double total_step_time_ms = 0;
  double total_input_ms = 0;
  double total_output_ms = 0;
  double total_host_compute_ms = 0;
  double total_host_prepare_ms = 0;
  double total_host_compile_ms = 0;
  double total_device_to_device_ms = 0;
  double total_unknown_ms = 0;
  for (const google::protobuf::Any& step_details : result.step_details()) {
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
    total_device_to_device_ms += details.device_to_device_ms();
    total_host_compute_ms += details.host_compute_ms();
    total_host_compile_ms += details.host_compile_ms();
    total_unknown_ms += details.unknown_time_ms();
  }
  if (total_step_time_ms == 0) {
    return {{"unknown",
             "No step time measured. Therefore we cannot tell where the "
             "performance bottleneck is."},
            "no",
            "",
            "no",
            ""};
  }
  double input_percent = 100.0 * total_input_ms / total_step_time_ms;
  double kernel_launch_percent =
      100.0 * total_host_prepare_ms / total_step_time_ms;
  double all_other_percent = 100.0 * total_unknown_ms / total_step_time_ms;
  int observation_index = 0;
  string input_classification;
  string input_statement;
  InfeedAnalysis(result.hardware_type(), input_percent, &observation_index,
                 &input_classification, &input_statement);

  string kernel_launch_classification;
  string kernel_launch_statement;
  KernelLaunchAnalysis(kernel_launch_percent, &observation_index,
                       &kernel_launch_classification, &kernel_launch_statement);

  string all_other_classification;
  string all_other_statement;
  AllOtherAnalysis(all_other_percent, &observation_index,
                   &all_other_classification, &all_other_statement);

  return {{
              input_classification,
              input_statement,
          },
          kernel_launch_classification,
          kernel_launch_statement,
          all_other_classification,
          all_other_statement};
}

}  // namespace profiler
}  // namespace tensorflow
