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
#include <string>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/convert/step_events_to_steps_db.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/format_utils.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/html_utils.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
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

GenericStepTimeBreakdown ComputeGenericStepTimeBreakdownInMs(
    const InputPipelineAnalysisResult& analysis) {
  Stat<double> unknown_time_ms;
  Stat<double> host_wait_input_ms;
  Stat<double> host_to_device_ms;
  Stat<double> input_ms;
  Stat<double> output_ms;
  Stat<double> device_compute_ms;
  Stat<double> device_to_device_ms;
  Stat<double> device_collectives_ms;
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
    const protobuf::RepeatedPtrField<PerCoreStepInfo>& grouped_by_step) {
  InputPipelineAnalysisResult result;

  // Computes the summary of step time in ms.
  *result.mutable_step_time_summary() =
      ComputeStepTimeSummaryInMs(grouped_by_step);

  Stat<double> input_summary_stats_in_percent;
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

// Returns the ratio of the host-to-device time in each step to the step-time.
double RatioOfHostToDeviceTimeToStepTime(
    const OpMetricsDb& host_tf_metrics_db,
    const InputPipelineAnalysisResult& input_pipeline_analysis) {
  // For TPU execution that uses infeed.
  absl::optional<double> host_infeed_enqueue_ratio =
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
      return SafeDivide(avg_host_to_device_time_ms, avg_step_time_ms);
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

  double ratio = std::min(
      1.0, RatioOfHostToDeviceTimeToStepTime(host_tf_metrics_db, *result));
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
    const OpStats& op_stats) {
  InputPipelineAnalysisResult result =
      ComputeGenericInputPipelineAnalysisResult(
          op_stats.step_db().step_sequence());
  PopulateStepDiagnostics(op_stats, result.mutable_diagnostics());
  result.set_hardware_type(HardwareType_Name(
      ParseHardwareType(op_stats.run_environment().device_type())));
  GenerateHostResult(op_stats.host_op_metrics_db(), &result);

  InputPipelineAnalysisRecommendation recommendation = GenerateRecommendation();
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
    // Defintely not input-bound.
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
    const ::tensorflow::protobuf::RepeatedPtrField<::google::protobuf::Any>&
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
  return 100.0 * SafeDivide(breakdown.enqueue_us(), total_input_time_us);
}

}  // namespace profiler
}  // namespace tensorflow
