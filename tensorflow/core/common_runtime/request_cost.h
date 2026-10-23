/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/core/common_runtime/request_cost.pb.h"

namespace tensorflow {

// RequestCost collects the costs and metrics for processing an rpc request.
class RequestCost {
 public:
  // Records costs. The inputs should be pairs of cost type and cost.
  // It's thread-safe, and can be called from different threads.
  void RecordCost(
      const std::vector<std::pair<absl::string_view, absl::Duration>>& costs);

  // Gets all types of costs for processing an rpc request.
  // It's thread-safe. It's expected to be called at the end of processing an
  // rpc request, when all the costs have been collected.
  absl::flat_hash_map<std::string, absl::Duration> GetCosts() const;

  // A cost broken down by named dimensions (e.g. different unit types).
  // Keys are dimension names, values are the cost in that dimension.
  using StructuredCost = absl::flat_hash_map<std::string, absl::Duration>;

  // Records structured costs (cost type -> StructuredCost) collected during
  // request execution or aggregated from sub-requests (e.g. via
  // GetStructuredCosts()).
  // It's thread-safe, and can be called from different threads. Costs are
  // summed up if recorded with the same key. Passing an rvalue vector or
  // absl::flat_hash_map moves the StructuredCost map into RequestCost when
  // inserting a new key without copying the inner dimension map.
  //
  // The absl::flat_hash_map overload is constrained as a template forwarding to
  // RecordStructuredCostsMap so braced-initializer-list calls such as
  // RecordStructuredCosts({{"cost", dims}}) unambiguously resolve to the
  // std::vector overload.
  void RecordStructuredCosts(
      const std::vector<std::pair<absl::string_view, StructuredCost>>& costs);
  void RecordStructuredCosts(
      std::vector<std::pair<absl::string_view, StructuredCost>>&& costs);
  template <
      typename Map,
      typename = std::enable_if_t<std::is_same_v<
          std::decay_t<Map>, absl::flat_hash_map<std::string, StructuredCost>>>>
  void RecordStructuredCosts(Map&& costs) {
    RecordStructuredCostsMap(std::forward<Map>(costs));
  }

  // Gets all types of structured costs for processing an rpc request.
  // It's thread-safe. It's expected to be called at the end of processing an
  // rpc request, when all the costs have been collected.
  absl::flat_hash_map<std::string, StructuredCost> GetStructuredCosts() const;

  // Records metrics. The inputs should be pairs of metric name and value.
  // It's thread-safe, and can be called from different threads. Unlike
  // RecordCosts where costs are summed up if recorded with the same key,
  // metrics are replaced.
  void RecordMetrics(
      const std::vector<std::pair<absl::string_view, double>>& metrics);

  // A metric that carries either an array of doubles or an array of byte
  // strings, but not both.
  struct StructuredMetric {
    std::variant<std::vector<double>, std::vector<std::string>> values;
  };

  // Records structured metrics (name -> StructuredMetric) collected during
  // request execution or aggregated from sub-requests (e.g. via
  // GetStructuredMetrics()).
  // It's thread-safe, and can be called from different threads. Metrics are
  // replaced if recorded with the same key. Passing an rvalue vector or
  // absl::flat_hash_map moves the metric values into RequestCost without
  // copying the underlying vectors.
  //
  // The absl::flat_hash_map overload is constrained as a template forwarding to
  // RecordStructuredMetricsMap so braced-initializer-list calls such as
  // RecordStructuredMetrics({{"name", metric}}) unambiguously resolve to the
  // std::vector overload.
  void RecordStructuredMetrics(
      const std::vector<std::pair<std::string, StructuredMetric>>&
          structured_metrics);
  void RecordStructuredMetrics(
      std::vector<std::pair<std::string, StructuredMetric>>&&
          structured_metrics);
  template <typename Map,
            typename = std::enable_if_t<std::is_same_v<
                std::decay_t<Map>,
                absl::flat_hash_map<std::string, StructuredMetric>>>>
  void RecordStructuredMetrics(Map&& structured_metrics) {
    RecordStructuredMetricsMap(std::forward<Map>(structured_metrics));
  }

  // Gets all structured metrics for processing an rpc request.
  // It's thread-safe. It's expected to be called at the end of processing an
  // rpc request, when all the structured metrics have been collected.
  absl::flat_hash_map<std::string, StructuredMetric> GetStructuredMetrics()
      const;

  // Gets all types of metrics for processing an rpc request.
  // It's thread-safe. It's expected to be called at the end of processing an
  // rpc request, when all the metrics have been collected.
  absl::flat_hash_map<std::string, double> GetMetrics() const;

  // Metrics of each batch that processes this rpc request.
  struct BatchMetrics {
    // Size of the batch.
    int64_t processed_size = 0;
    // In this batch, input size from this rpc request.
    int64_t input_size = 0;
    // In this batch, the padding amount.
    int64_t padding_size = 0;
    // Costs for processing this batch.
    absl::flat_hash_map<std::string, absl::Duration> batch_costs;
  };

  // Records the metrics of a batch.
  // It's thread-safe, and can be called from different threads. It may be
  // called multiple times if a request is processed by more than one batches.
  void RecordBatchMetrics(const BatchMetrics& batch_metrics);

  // Scales costs of all the batches that process this rpc request.
  // It's thread-safe. It's expected to be called at the end of processing an
  // rpc request, when all batch processing has completed.
  void ScaleBatchCosts(int scale_factor);

  // Get metrics of all the batches that process this rpc request.
  // It's thread-safe. It's expected to be called at the end of processing an
  // rpc request, when all batch processing has completed.
  std::vector<BatchMetrics> GetBatchMetrics() const;

  // Serializes the costs and metrics into `proto`.
  // It's thread-safe.
  void ToProto(RequestCostProto& proto) const;
  RequestCostProto ToProto() const;

  // Replaces the costs and metrics with those from `proto`.
  // It's thread-safe.
  void FromProto(const RequestCostProto& proto);

 private:
  // Out-of-line helpers backing the template RecordStructuredCosts(Map&&) and
  // RecordStructuredMetrics(Map&&) overloads so mutex acquisition and map
  // updates stay in request_cost.cc.
  void RecordStructuredCostsMap(
      const absl::flat_hash_map<std::string, StructuredCost>& costs);
  void RecordStructuredCostsMap(
      absl::flat_hash_map<std::string, StructuredCost>&& costs);
  void RecordStructuredMetricsMap(
      const absl::flat_hash_map<std::string, StructuredMetric>&
          structured_metrics);
  void RecordStructuredMetricsMap(
      absl::flat_hash_map<std::string, StructuredMetric>&& structured_metrics);

  mutable absl::Mutex mutex_;

  // Query costs. Map from cost type to cost.
  absl::flat_hash_map<std::string, absl::Duration> cost_map_
      ABSL_GUARDED_BY(mutex_);
  // Structured costs. Map from cost type to StructuredCost.
  absl::flat_hash_map<std::string, StructuredCost> structured_cost_map_
      ABSL_GUARDED_BY(mutex_);
  // Query metrics. Map from metric name to value.
  absl::flat_hash_map<std::string, double> metric_map_ ABSL_GUARDED_BY(mutex_);
  // Structured metrics. Map from metric name to StructuredMetric.
  absl::flat_hash_map<std::string, StructuredMetric> structured_metric_map_
      ABSL_GUARDED_BY(mutex_);
  // Metrics of batches that process this rpc request.
  std::vector<BatchMetrics> batch_metrics_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_
