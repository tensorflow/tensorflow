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

#include "tensorflow/core/common_runtime/request_cost.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/overload.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/request_cost.pb.h"

namespace tensorflow {

void RequestCost::RecordCost(
    const std::vector<std::pair<absl::string_view, absl::Duration>>& costs) {
  absl::MutexLock lock(mutex_);
  for (const auto& cost : costs) {
    cost_map_[cost.first] += cost.second;
  }
}

absl::flat_hash_map<std::string, absl::Duration> RequestCost::GetCosts() const {
  absl::MutexLock lock(mutex_);
  return cost_map_;
}

void RequestCost::RecordStructuredCosts(
    const std::vector<std::pair<absl::string_view, StructuredCost>>& costs) {
  absl::MutexLock lock(mutex_);
  for (const auto& [name, cost] : costs) {
    auto& entry = structured_cost_map_[name];
    for (const auto& [dim, value] : cost) {
      entry[dim] += value;
    }
  }
}

absl::flat_hash_map<std::string, RequestCost::StructuredCost>
RequestCost::GetStructuredCosts() const {
  absl::MutexLock lock(mutex_);
  return structured_cost_map_;
}

void RequestCost::RecordMetrics(
    const std::vector<std::pair<absl::string_view, double>>& metrics) {
  absl::MutexLock lock(mutex_);
  for (const auto& metric : metrics) {
    metric_map_[metric.first] = metric.second;
  }
}

absl::flat_hash_map<std::string, double> RequestCost::GetMetrics() const {
  absl::MutexLock lock(mutex_);
  return metric_map_;
}

void RequestCost::RecordStructuredMetrics(
    const std::vector<std::pair<std::string, StructuredMetric>>&
        structured_metrics) {
  absl::MutexLock lock(mutex_);
  for (const auto& [name, metric] : structured_metrics) {
    structured_metric_map_[name] = metric;
  }
}

absl::flat_hash_map<std::string, RequestCost::StructuredMetric>
RequestCost::GetStructuredMetrics() const {
  absl::MutexLock lock(mutex_);
  return structured_metric_map_;
}

void RequestCost::RecordBatchMetrics(const BatchMetrics& batch_metrics) {
  absl::MutexLock lock(mutex_);
  batch_metrics_.push_back(batch_metrics);
}

void RequestCost::ScaleBatchCosts(int scale_factor) {
  absl::MutexLock lock(mutex_);
  for (auto& batch_metrics : batch_metrics_) {
    for (auto& [cost_type, cost] : batch_metrics.batch_costs) {
      batch_metrics.batch_costs[cost_type] *= scale_factor;
    }
  }
}

std::vector<RequestCost::BatchMetrics> RequestCost::GetBatchMetrics() const {
  absl::MutexLock lock(mutex_);
  return batch_metrics_;
}

void RequestCost::ToProto(RequestCostProto& proto) const {
  absl::MutexLock lock(mutex_);
  proto.Clear();
  for (const auto& [cost_type, cost] : cost_map_) {
    proto.mutable_costs()->insert(
        {cost_type, absl::ToDoubleMicroseconds(cost)});
  }
  for (const auto& [cost_type, dims] : structured_cost_map_) {
    auto& structured = (*proto.mutable_structured_costs())[cost_type];
    for (const auto& [dim, value] : dims) {
      structured.mutable_dimensions()->insert(
          {dim, absl::ToDoubleMicroseconds(value)});
    }
  }
  for (const auto& [metric_type, metric_value] : metric_map_) {
    proto.mutable_metrics()->insert({metric_type, metric_value});
  }
  for (const auto& [name, structured] : structured_metric_map_) {
    RequestCostProto::StructuredMetric& structured_proto =
        (*proto.mutable_structured_metrics())[name];
    std::visit(absl::Overload{
                   [&structured_proto](absl::Span<const double> values) {
                     structured_proto.mutable_double_values()->Assign(
                         values.begin(), values.end());
                   },
                   [&structured_proto](absl::Span<const std::string> values) {
                     structured_proto.mutable_bytes_values()->Assign(
                         values.begin(), values.end());
                   },
               },
               structured.values);
  }
  for (const auto& bm : batch_metrics_) {
    RequestCostProto::BatchMetrics* bm_proto = proto.add_batch_metrics();
    bm_proto->set_processed_size(bm.processed_size);
    bm_proto->set_input_size(bm.input_size);
    bm_proto->set_padding_size(bm.padding_size);
    for (const auto& [cost_type, cost] : bm.batch_costs) {
      bm_proto->mutable_batch_costs()->insert(
          {cost_type, absl::ToDoubleMicroseconds(cost)});
    }
  }
}

RequestCostProto RequestCost::ToProto() const {
  RequestCostProto proto;
  ToProto(proto);
  return proto;
}

void RequestCost::FromProto(const RequestCostProto& proto) {
  absl::MutexLock lock(mutex_);
  cost_map_.clear();
  structured_cost_map_.clear();
  metric_map_.clear();
  structured_metric_map_.clear();
  batch_metrics_.clear();

  for (const auto& [name, value] : proto.costs()) {
    cost_map_[name] = absl::Microseconds(value);
  }
  for (const auto& [name, structured] : proto.structured_costs()) {
    StructuredCost dims;
    for (const auto& [dim, value] : structured.dimensions()) {
      dims[dim] = absl::Microseconds(value);
    }
    structured_cost_map_[name] = std::move(dims);
  }
  for (const auto& [name, value] : proto.metrics()) {
    metric_map_[name] = value;
  }
  for (const auto& [name, src] : proto.structured_metrics()) {
    StructuredMetric metric;
    if (!src.bytes_values().empty()) {
      metric.values.emplace<1>(src.bytes_values().begin(),
                               src.bytes_values().end());
    } else {
      metric.values.emplace<0>(src.double_values().begin(),
                               src.double_values().end());
    }
    structured_metric_map_[name] = std::move(metric);
  }
  batch_metrics_.reserve(proto.batch_metrics_size());
  for (const auto& bm_proto : proto.batch_metrics()) {
    BatchMetrics bm;
    bm.processed_size = bm_proto.processed_size();
    bm.input_size = bm_proto.input_size();
    bm.padding_size = bm_proto.padding_size();
    for (const auto& [cost_type, cost] : bm_proto.batch_costs()) {
      bm.batch_costs[cost_type] = absl::Microseconds(cost);
    }
    batch_metrics_.push_back(std::move(bm));
  }
}

}  // namespace tensorflow
