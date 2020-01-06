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

template <class Collection>
double GetTimeInMs(const Collection& type_ps,
                   EventType event_type) {
  return PicosToMillis(gtl::FindWithDefault(type_ps, event_type, /*value=*/0));
}

StepSummary GetStepSummaryForSampleStats(const Stat<double>& sample_stats) {
  StepSummary step_time_summary;
  step_time_summary.set_average(sample_stats.avg());
  step_time_summary.set_standard_deviation(
      std::sqrt(sample_stats.sample_variance()));
  step_time_summary.set_minimum(sample_stats.min());
  step_time_summary.set_maximum(sample_stats.max());
  return step_time_summary;
}

GenericStepTimeBreakdown ComputeGenericStepTimeBreakdownInMs(
    const InputPipelineAnalysisResult& analysis) {
  Stat<double> unknown_time_ms;
  Stat<double> infeed_ms;
  Stat<double> outfeed_ms;
  Stat<double> device_compute_ms;
  Stat<double> device_to_device_ms;
  Stat<double> host_compute_ms;
  Stat<double> host_prepare_ms;
  Stat<double> host_compile_ms;
  GenericStepTimeBreakdown result;

  for (const google::protobuf::Any& step_details : analysis.step_details()) {
    PerGenericStepDetails details;
    bool success = step_details.UnpackTo(&details);
    if (!success) {
      LOG(ERROR) << "Unable to unpack step_breakdown. Expected: generic"
                 << std::endl;
      return {};
    }
    unknown_time_ms.UpdateStat(details.unknown_time_ms());
    infeed_ms.UpdateStat(details.infeed_ms());
    outfeed_ms.UpdateStat(details.outfeed_ms());
    device_compute_ms.UpdateStat(details.device_compute_ms());
    device_to_device_ms.UpdateStat(details.device_to_device_ms());
    host_compute_ms.UpdateStat(details.host_compute_ms());
    host_prepare_ms.UpdateStat(details.host_prepare_ms());
    host_compile_ms.UpdateStat(details.host_compile_ms());
  }
  *result.mutable_unknown_time_ms_summary() =
      GetStepSummaryForSampleStats(unknown_time_ms);
  *result.mutable_infeed_ms_summary() = GetStepSummaryForSampleStats(infeed_ms);
  *result.mutable_outfeed_ms_summary() =
      GetStepSummaryForSampleStats(outfeed_ms);
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

  Stat<double> infeed_summary_stats_in_percent;
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
    if (!success) {
      LOG(ERROR) << "Unable to unpack step_breakdown. Expected: generic"
                 << std::endl;
      return {};
    }
    const auto& type_ps = generic.type_ps();
    details.set_unknown_time_ms(GetTimeInMs(type_ps, UNKNOWN_TIME));
    // To be consistent with TPU case, the infeed time includes the time that
    // the host is reading files, preprocessing, and the time to transfer the
    // data to the device.
    details.set_infeed_ms(GetTimeInMs(type_ps, HOST_WAIT_INPUT) +
                          GetTimeInMs(type_ps, HOST_TO_DEVICE) +
                          GetTimeInMs(type_ps, DEVICE_WAIT_HOST));
    details.set_outfeed_ms(GetTimeInMs(type_ps, DEVICE_TO_HOST));
    details.set_device_compute_ms(GetTimeInMs(type_ps, DEVICE_COMPUTE));
    details.set_device_to_device_ms(GetTimeInMs(type_ps, DEVICE_TO_DEVICE) +
                                    GetTimeInMs(type_ps, DEVICE_WAIT_DEVICE));
    details.set_host_compute_ms(GetTimeInMs(type_ps, HOST_COMPUTE));
    details.set_host_prepare_ms(GetTimeInMs(type_ps, HOST_PREPARE));
    details.set_host_compile_ms(GetTimeInMs(type_ps, HOST_COMPILE));

    result.add_step_details()->PackFrom(details);

    const double infeed_pct_of_step_time =
        100.0 * SafeDivide(details.infeed_ms(), details.step_time_ms());
    infeed_summary_stats_in_percent.UpdateStat(infeed_pct_of_step_time);
  }

  // Computes the summary of infeed time as percentage of step time.
  *result.mutable_infeed_percent_summary() =
      GetStepSummaryForSampleStats(infeed_summary_stats_in_percent);

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
  return IsInfeedEnqueueOp(category) || IsDatasetOp(category);
}

InputOpCategory CategorizeInputOp(absl::string_view name,
                                  absl::string_view category) {
  if (IsInfeedEnqueueOp(category)) {
    return InputOpCategory::kEnqueue;
  }
  DCHECK(IsDatasetOp(category));
  if (absl::EndsWith(name, "::TFRecord") ||
      absl::EndsWith(name, "::TextLine") ||
      absl::EndsWith(name, "::FixedLengthRecord") ||
      absl::EndsWith(name, "::SSTable") || absl::EndsWith(name, "::RecordIO")) {
    if (absl::StrContains(name, "::MemoryReader") ||
        absl::StrContains(name, "::MemoryWriter") ||
        absl::StrContains(name, "::Interleave") ||
        absl::StrContains(name, "::Prefetch") ||
        absl::StrContains(name, "::ParallelMap")) {
      return InputOpCategory::kAdvancedFileRead;
    } else {
      return InputOpCategory::kDemandedFileRead;
    }
  } else {
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

void GenerateHostResult(const OpMetricsDb& host_tf_metrics_db,
                        InputPipelineAnalysisResult* result) {
  InputOpMetrics input_op_metrics = SelectInputOpMetrics(host_tf_metrics_db);
  // Return if the program is not using an input pipeline with xprof
  // instrumentation and no input ops are found.
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

  // We use total_host_infeed_enq_start_timestamp_ps_diff_ to approximate the
  // total host step time.
  double ratio = SafeDivide(
      host_tf_metrics_db.total_host_infeed_enq_duration_ps(),
      host_tf_metrics_db.total_host_infeed_enq_start_timestamp_ps_diff());
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

string AnchorElement(absl::string_view url, absl::string_view text) {
  return absl::StrCat("<a href=\"", url, "\" target=\"_blank\">", text, "</a>");
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

}  // namespace

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

}  // namespace profiler
}  // namespace tensorflow
