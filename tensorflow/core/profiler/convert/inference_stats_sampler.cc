/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/inference_stats_sampler.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {

namespace {

using ::tensorflow::profiler::BatchDetail;
using ::tensorflow::profiler::InferenceStats;
using ::tensorflow::profiler::PerModelInferenceStats;
using ::tensorflow::profiler::RequestDetail;

// Column names that can be used to do percentile selection.
// For request:
constexpr char kColumnLatencyUs[] = "Latency";
constexpr char kColumnBatchingRequestDelayUs[] = "Request delay for batching";
constexpr char kColumnBatchingRequestSize[] = "Request size";
constexpr char kColumnHostPreprocessing[] = "Host preprocess";
constexpr char kColumnHostBatchFormation[] = "Host batch formation";
constexpr char kColumnHostRuntime[] = "Host runtime";
constexpr char kColumnHostToDevice[] = "Data transfer H2D";
constexpr char kColumnDeviceToHost[] = "Data transfer D2H";
constexpr char kColumnDeviceCompute[] = "Device compute";
constexpr char kColumnHostPostprocessing[] = "Host postprocess";
constexpr char kColumnIdleTime[] = "Idle time";
// For batch:
constexpr char kColumnBatchingDelayUs[] = "Batching delay";
constexpr char kColumnPaddingAmount[] = "Padding amount";
constexpr char kColumnBatchSizeAfterPadding[] = "Batch size after padding";
constexpr char kColumnBatchingEfficiency[] = "Batching efficiency";

double CalculateBatchingEfficiency(const BatchDetail& batch) {
  return tsl::profiler::SafeDivide(
      static_cast<double>(batch.batch_size_after_padding() -
                          batch.padding_amount()),
      static_cast<double>(batch.batch_size_after_padding()));
}

// Comparator for RequestDetail proto.
bool CompareByRequestLatency(const RequestDetail* a, const RequestDetail* b) {
  return (a->end_time_ps() - a->start_time_ps()) <
         (b->end_time_ps() - b->start_time_ps());
}
bool CompareByBatchingRequestDelay(const RequestDetail* a,
                                   const RequestDetail* b) {
  return a->batching_request_delay_ps() < b->batching_request_delay_ps();
}
bool CompareByBatchingRequestSize(const RequestDetail* a,
                                  const RequestDetail* b) {
  return a->batching_request_size() < b->batching_request_size();
}
bool CompareByHostPreprocessing(const RequestDetail* a,
                                const RequestDetail* b) {
  return a->host_preprocessing_ps() < b->host_preprocessing_ps();
}
bool CompareByHostBatchFormation(const RequestDetail* a,
                                 const RequestDetail* b) {
  return a->host_batch_formation_ps() < b->host_batch_formation_ps();
}
bool CompareByHostRuntime(const RequestDetail* a, const RequestDetail* b) {
  return a->host_runtime_ps() < b->host_runtime_ps();
}
bool CompareByHostToDevice(const RequestDetail* a, const RequestDetail* b) {
  return a->write_to_device_time_ps() < b->write_to_device_time_ps();
}
bool CompareByDeviceToHost(const RequestDetail* a, const RequestDetail* b) {
  return a->read_from_device_time_ps() < b->read_from_device_time_ps();
}
bool CompareByDeviceCompute(const RequestDetail* a, const RequestDetail* b) {
  return a->device_time_ps() < b->device_time_ps();
}
bool CompareByPostProcessing(const RequestDetail* a, const RequestDetail* b) {
  return a->host_postprocessing_ps() < b->host_postprocessing_ps();
}
bool CompareByIdleTime(const RequestDetail* a, const RequestDetail* b) {
  return a->idle_time_ps() < b->idle_time_ps();
}
// Use percentile column name to get the corresponding compare function.
std::function<bool(const RequestDetail*, const RequestDetail*)>
GetRequestCompareFunction(absl::string_view column_name) {
  if (column_name == kColumnBatchingRequestDelayUs) {
    return CompareByBatchingRequestDelay;
  } else if (column_name == kColumnBatchingRequestSize) {
    return CompareByBatchingRequestSize;
  } else if (column_name == kColumnHostPreprocessing) {
    return CompareByHostPreprocessing;
  } else if (column_name == kColumnHostBatchFormation) {
    return CompareByHostBatchFormation;
  } else if (column_name == kColumnHostRuntime) {
    return CompareByHostRuntime;
  } else if (column_name == kColumnHostToDevice) {
    return CompareByHostToDevice;
  } else if (column_name == kColumnDeviceToHost) {
    return CompareByDeviceToHost;
  } else if (column_name == kColumnDeviceCompute) {
    return CompareByDeviceCompute;
  } else if (column_name == kColumnHostPostprocessing) {
    return CompareByPostProcessing;
  } else if (column_name == kColumnIdleTime) {
    return CompareByIdleTime;
  } else {
    // Return CompareByRequestLatency by default.
    return CompareByRequestLatency;
  }
}

// Comparator for BatchDetail proto.
bool CompareByBatchLatency(const BatchDetail* a, const BatchDetail* b) {
  return (a->end_time_ps() - a->start_time_ps()) <
         (b->end_time_ps() - b->start_time_ps());
}
bool CompareByBatchDelay(const BatchDetail* a, const BatchDetail* b) {
  return a->batch_delay_ps() < b->batch_delay_ps();
}
bool CompareByPaddingAmount(const BatchDetail* a, const BatchDetail* b) {
  return a->padding_amount() < b->padding_amount();
}
bool CompareByBatchSizeAfterPadding(const BatchDetail* a,
                                    const BatchDetail* b) {
  return a->batch_size_after_padding() < b->batch_size_after_padding();
}
bool CompareByBatchingEfficiency(const BatchDetail* a, const BatchDetail* b) {
  return CalculateBatchingEfficiency(*a) < CalculateBatchingEfficiency(*b);
}
// Use percentile column name to get the corresponding compare function.
std::function<bool(const BatchDetail*, const BatchDetail*)>
GetBatchCompareFunction(absl::string_view column_name) {
  if (column_name == kColumnBatchingDelayUs) {
    return CompareByBatchDelay;
  } else if (column_name == kColumnPaddingAmount) {
    return CompareByPaddingAmount;
  } else if (column_name == kColumnBatchSizeAfterPadding) {
    return CompareByBatchSizeAfterPadding;
  } else if (column_name == kColumnBatchingEfficiency) {
    return CompareByBatchingEfficiency;
  } else {
    // Return CompareByBatchLatency by default.
    return CompareByBatchLatency;
  }
}

// A static helper class to select a subset of inference data (request or batch)
// to show in the frontend.
// DataType can be either RequestDetail or BatchDetail.
template <typename DataType>
class PercentileSelector {
 public:
  // The range of values in [percentile, perentile+error) are still regarded as
  // percentile.
  struct PercentileRange {
    double percentile;
    double error;
  };

  // The percentiles (with the corresponding error bounds) that will be included
  // in inference profile result.
  static constexpr std::array<PercentileRange, 6> kWantedPercentiles = {
      {{50.0, 1},
       {75.0, 1},
       {90.0, 1},
       {99.0, 0.5},
       {99.9, 0.05},
       {99.99, 0.005}}};

  // Maximum number of values included for each percentile range.
  static constexpr size_t kMaxNumDataSelectedPerPercentile = 10;

  // Select a subset of data from <all_data>, return pointer to the original
  // data and the percentile.
  static std::vector<std::pair<const DataType*, double>> Select(
      const std::vector<const DataType*>& all_data) {
    return SelectInternal(all_data);
  }

 private:
  static bool GreaterThan(double percentile, const PercentileRange& wanted) {
    // Uses ">=" instead of ">" so that the round-up value is not included.
    return percentile >= (wanted.percentile + wanted.error);
  }

  static bool LessThan(double percentile, const PercentileRange& wanted) {
    return percentile < wanted.percentile;
  }

  static bool WithinRange(double percentile, const PercentileRange& wanted) {
    return !GreaterThan(percentile, wanted) && !LessThan(percentile, wanted);
  }

  static std::vector<std::pair<const DataType*, double>> SelectInternal(
      const std::vector<const DataType*>& all_data) {
    std::vector<std::pair<const DataType*, double>> result;
    // If the number of data points is too small (smaller than the result size
    // when select by percentile, like in a unit test), it does not make sense
    // to select by percentile, just select all the data points and the frontend
    // is able to display all of them.
    if (all_data.size() <=
        kWantedPercentiles.size() * kMaxNumDataSelectedPerPercentile) {
      for (size_t i = 0; i < all_data.size(); i++) {
        double percentile = 100.0 * i / all_data.size();
        result.push_back(std::make_pair(all_data[i], percentile));
      }
      return result;
    }

    // Select by percentile.
    size_t idx_to_next_data = 0;
    for (size_t i = 0; i < kWantedPercentiles.size(); i++) {
      const auto& wanted = kWantedPercentiles[i];
      size_t num_data_selected = 0;
      for (size_t k = idx_to_next_data; k < all_data.size(); k++) {
        double percentile = 100.0 * k / all_data.size();
        if (GreaterThan(percentile, wanted)) {
          // Updates idx_to_next_data to k so that when we select data for the
          // next percentile we don't need to consider the data with smaller
          // latenices than that for the next percentile.
          idx_to_next_data = k;
          break;
        }
        if (WithinRange(percentile, wanted)) {
          if (num_data_selected < kMaxNumDataSelectedPerPercentile) {
            // Selects this data only if we have not hit the limit for this
            // percentile.
            result.push_back(std::make_pair(all_data[k], percentile));
            ++num_data_selected;
          }
        }
      }
    }
    return result;
  }
};

// Sample the requests and batches in <per_model_stats> using sampling column
// <request_percentile_column> and <batch_percentile_column>.
void SamplePerModelInferenceStats(
    absl::string_view request_percentile_column,
    absl::string_view batch_percentile_column,
    const PerModelInferenceStats& per_model_stats,
    SampledPerModelInferenceStats* sampled_per_model_stats) {
  // Select a subset of requests and batches based on percentile and generate
  // final result.
  std::vector<const RequestDetail*> requests(
      per_model_stats.request_details_size());
  for (size_t i = 0; i < per_model_stats.request_details_size(); i++) {
    requests[i] = &per_model_stats.request_details(i);
  }
  // Requests in per model stats are already sorted by latency. Only redo the
  // sorting when percentile column is not latency.
  if (request_percentile_column != kColumnLatencyUs) {
    std::sort(requests.begin(), requests.end(),
              GetRequestCompareFunction(request_percentile_column));
  }
  sampled_per_model_stats->sampled_requests =
      PercentileSelector<RequestDetail>::Select(requests);

  std::vector<const BatchDetail*> batches(per_model_stats.batch_details_size());
  for (size_t i = 0; i < per_model_stats.batch_details_size(); i++) {
    batches[i] = &per_model_stats.batch_details(i);
  }
  // Batches in per model stats are already sorted by latency. Only redo the
  // sorting when percentile column is not latency.
  if (batch_percentile_column != kColumnLatencyUs) {
    std::sort(batches.begin(), batches.end(),
              GetBatchCompareFunction(batch_percentile_column));
  }
  sampled_per_model_stats->sampled_batches =
      PercentileSelector<BatchDetail>::Select(batches);
}

}  // namespace

SampledInferenceStats SampleInferenceStats(
    absl::string_view request_percentile_column,
    absl::string_view batch_percentile_column,
    const InferenceStats& inference_stats) {
  SampledInferenceStats result;
  for (const auto& [model_index, model_inference_stats] :
       inference_stats.inference_stats_per_model()) {
    SamplePerModelInferenceStats(request_percentile_column,
                                 batch_percentile_column, model_inference_stats,
                                 &(result[model_index]));
  }

  return result;
}

}  // namespace tensorflow::profiler
