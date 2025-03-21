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
#include "tensorflow/core/profiler/convert/inference_stats_grouping.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow::profiler {

namespace {

using ::tensorflow::profiler::BatchDetail;
using ::tensorflow::profiler::InferenceStats;
using ::tensorflow::profiler::ModelIdDatabase;
using ::tensorflow::profiler::PerBatchSizeAggregatedResult;
using ::tensorflow::profiler::PerModelInferenceStats;
using ::tensorflow::profiler::RequestDetail;
using ::tensorflow::profiler::TensorEventDetail;
using ::tsl::profiler::Timespan;

template <typename RandIt, typename Compare>
void push_down_heap(size_t hole, RandIt first, RandIt last, Compare comp) {
  size_t size = last - first;
  assert(hole < size);
  auto value = std::move(first[hole]);
  while (true) {
    size_t l_child = 2 * hole + 1;
    size_t r_child = l_child + 1;
    size_t max_child = l_child;
    if (r_child < size && comp(first[l_child], first[r_child])) {
      max_child = r_child;
    }
    if (max_child >= size) break;
    if (!comp(value, first[max_child])) break;
    first[hole] = std::move(first[max_child]);
    hole = max_child;
  }
  first[hole] = std::move(value);
}
// Pushes the root down the heap.
template <typename RandIt, typename Compare>
void push_root_heap(RandIt first, RandIt last, Compare comp) {
  push_down_heap(0, std::move(first), std::move(last), std::move(comp));
}

template <typename ContainerContainer, typename Out, typename Cmp>
Out nway_merge(const ContainerContainer& containers, Out out, Cmp cmp) {
  using std::begin;
  using std::end;
  using In = decltype(begin(*begin(containers)));  // The input iterator type.
  using Range = std::pair<In, In>;
  std::vector<Range> sources;
  for (const auto& container : containers) {
    Range r(begin(container), end(container));
    if (r.first != r.second) sources.push_back(std::move(r));
  }
  // Zero, one or two collections can be merged without a priority queue.
  switch (sources.size()) {
    case 0:
      return out;
    case 1:
      return std::copy(sources[0].first, sources[0].second, out);
    case 2:
      return std::merge(sources[0].first, sources[0].second, sources[1].first,
                        sources[1].second, out, cmp);
  }
  // Take a comparator for T and produce an inverse comparator
  // for std::pair<In<T>, In<T>>, inverted so as to produce a min-heap.
  auto heap_cmp = [&](const Range& a, const Range& b) {
    // Compares b < a instead of a < b.
    return cmp(*b.first, *a.first);
  };
  auto heap_data = sources.data();
  auto heap_size = sources.size();
  std::make_heap(heap_data, heap_data + heap_size, heap_cmp);
  auto& top = sources.front();
  auto pop = [&]() {
    *out = *top.first;
    ++out;
    ++top.first;
  };

  for (; heap_size > 2;) {
    for (pop(); top.first != top.second; pop()) {
      push_root_heap(heap_data, heap_data + heap_size, heap_cmp);
    }
    top = std::move(sources[--heap_size]);
    push_root_heap(heap_data, heap_data + heap_size, heap_cmp);
  }

  return std::merge(sources[0].first, sources[0].second, sources[1].first,
                    sources[1].second, out, cmp);
}

double GetThroughput(size_t data_size, uint64_t start_time_ps,
                     uint64_t end_time_ps) {
  return data_size / tsl::profiler::PicoToUni(end_time_ps - start_time_ps);
}

// Compute throughput and average latency.
// DataType can either be RequestDetail or BatchDetail.
template <typename DataType>
std::pair<double, double> ComputeThroughputAndAverageLatencyUs(
    const std::vector<const DataType*>& all_data) {
  if (all_data.empty()) {
    // Return 0 immediately to avoid divide by zero error.
    return std::make_pair(0.0, 0.0);
  }

  uint64_t min_start_time_ps = std::numeric_limits<uint64_t>::max();
  uint64_t max_end_time_ps = 0;
  uint64_t total_latency_ps = 0;

  for (const DataType* data : all_data) {
    min_start_time_ps = std::min(min_start_time_ps, data->start_time_ps());
    max_end_time_ps = std::max(max_end_time_ps, data->end_time_ps());
    total_latency_ps += (data->end_time_ps() - data->start_time_ps());
  }

  double throughput =
      GetThroughput(all_data.size(), min_start_time_ps, max_end_time_ps);
  double average_latency_us =
      tsl::profiler::PicoToMicro(total_latency_ps) / all_data.size();
  return std::make_pair(throughput, average_latency_us);
}

template <typename DataType>
bool CompareByDuration(const DataType* a, const DataType* b) {
  return Timespan::ByDuration(
      Timespan::FromEndPoints(a->start_time_ps(), a->end_time_ps()),
      Timespan::FromEndPoints(b->start_time_ps(), b->end_time_ps()));
}

// Regroup data in <data_by_host> using model id for future analysis.
// DataType can be either RequestDetail or BatchDetail.
template <typename DataType>
void RegroupDataByModelId(
    const ModelIdDatabase& model_id_db,
    const std::vector<const tsl::protobuf::RepeatedPtrField<DataType>*>&
        data_by_host,
    std::vector<std::vector<const DataType*>>* data_by_model_id) {
  // First group data by model_id and host.
  std::vector<std::vector<std::vector<const DataType*>>>
      data_by_model_id_by_host;

  // If model_id_db is empty, this means model_id is not available in the trace,
  // so we simply consider the entire execution as a single model_id.
  bool no_model_id = model_id_db.ids_size() == 0;
  int model_index_size = no_model_id ? 1 : model_id_db.ids_size();
  int host_index_size = data_by_host.size();
  data_by_model_id_by_host.resize(model_index_size);
  for (size_t model_index = 0; model_index < model_index_size; ++model_index) {
    data_by_model_id_by_host[model_index].resize(host_index_size);
  }

  int32_t host_index = 0;
  for (const tsl::protobuf::RepeatedPtrField<DataType>* single_host_data :
       data_by_host) {
    for (const DataType& data : *single_host_data) {
      int model_index = no_model_id ? 0 : data.model_id_index();
      // If model_id_db is not empty, and a session/batch does not have
      // model_id, ignore it in per model analysis.
      if (model_index == -1) {
        continue;
      }
      data_by_model_id_by_host[model_index][host_index].push_back(&data);
    }
    ++host_index;
  }

  // data_by_host is already sorted by the latency, so
  // data_by_model_id_by_host is also sorted by the latency. Therefore,
  // we just need to do a n way merge instead of a real sorting.
  data_by_model_id->resize(model_index_size);
  for (size_t model_index = 0; model_index < model_index_size; ++model_index) {
    int total_size = 0;
    for (const auto& per_model_per_host :
         data_by_model_id_by_host[model_index]) {
      total_size += per_model_per_host.size();
    }
    data_by_model_id->at(model_index).reserve(total_size);
  }
  for (size_t model_index = 0; model_index < model_index_size; ++model_index) {
    nway_merge(data_by_model_id_by_host[model_index],
               std::back_inserter(data_by_model_id->at(model_index)),
               CompareByDuration<DataType>);
  }
}

// Generates the tensor transfer aggregated result using the per model data in
// <per_model>.
void GenerateTensorTransferAggregatedResult(PerModelInferenceStats* per_model) {
  absl::flat_hash_map<int32_t, std::vector<const TensorEventDetail*>>
      tensor_events_by_index;
  // For requests, only count the tensor events with owner REQUEST, because if
  // inference batching is enabled, there will be tensor events that are owned
  // by batches and just inherited by requests. Counting these tensor events
  // will lead to double counting.
  for (const auto& request : per_model->request_details()) {
    for (const auto& tensor_event : request.tensor_event_details()) {
      if (tensor_event.owner() == TensorEventDetail::REQUEST) {
        tensor_events_by_index[tensor_event.tensor_pattern_index()].push_back(
            &tensor_event);
      }
    }
  }
  for (const auto& batch : per_model->batch_details()) {
    if (batch.has_tensor_event_detail()) {
      tensor_events_by_index[batch.tensor_event_detail().tensor_pattern_index()]
          .push_back(&batch.tensor_event_detail());
    }
  }

  if (tensor_events_by_index.empty()) return;

  static constexpr double kPercentiles[] = {50.0, 75.0, 90.0, 95.0, 99.0, 99.9};
  for (auto& [index, events] : tensor_events_by_index) {
    auto* tensor_pattern_result =
        per_model->mutable_tensor_transfer_aggregated_result()
            ->add_tensor_pattern_results();
    tensor_pattern_result->set_tensor_pattern_index(index);
    tensor_pattern_result->set_count(events.size());
    std::sort(events.begin(), events.end(),
              [](const TensorEventDetail* a, const TensorEventDetail* b) {
                return a->linearize_delinearize_time_ps() <
                       b->linearize_delinearize_time_ps();
              });
    for (const double percentile : kPercentiles) {
      int index = static_cast<int>(percentile / 100.0 * events.size());
      auto* percentile_time =
          tensor_pattern_result->add_linearize_delinearize_percentile_time();
      percentile_time->set_percentile(percentile);
      percentile_time->set_time_ps(
          events[index]->linearize_delinearize_time_ps());
    }
  }
}

void AggregateRequest(const RequestDetail& input, RequestDetail* result) {
  // In aggregated result, start_time is set to 0, and end time is set to the
  // sum of the duration of the input requests.
  result->set_end_time_ps(input.end_time_ps() - input.start_time_ps() +
                          result->end_time_ps());
  result->set_device_time_ps(result->device_time_ps() + input.device_time_ps());
  result->set_read_from_device_time_ps(result->read_from_device_time_ps() +
                                       input.read_from_device_time_ps());
  result->set_write_to_device_time_ps(result->write_to_device_time_ps() +
                                      input.write_to_device_time_ps());
  result->set_batching_request_delay_ps(result->batching_request_delay_ps() +
                                        input.batching_request_delay_ps());
  result->set_batching_request_size(result->batching_request_size() +
                                    input.batching_request_size());
  result->set_host_preprocessing_ps(result->host_preprocessing_ps() +
                                    input.host_preprocessing_ps());
  result->set_host_batch_formation_ps(result->host_batch_formation_ps() +
                                      input.host_batch_formation_ps());
  result->set_host_runtime_ps(result->host_runtime_ps() +
                              input.host_runtime_ps());
  result->set_host_postprocessing_ps(result->host_postprocessing_ps() +
                                     input.host_postprocessing_ps());
  result->set_idle_time_ps(result->idle_time_ps() + input.idle_time_ps());
}

RequestDetail GetAverageRequestDetails(const RequestDetail& request,
                                       int64_t size) {
  RequestDetail result;
  if (size == 0) return result;
  // Average request detail does not have a request ID.
  result.set_request_id(-1);
  result.set_start_time_ps(0);
  // Calculating average by dividing aggregated request by size.
  result.set_end_time_ps(request.end_time_ps() / size);
  result.set_device_time_ps(request.device_time_ps() / size);
  result.set_write_to_device_time_ps(request.write_to_device_time_ps() / size);
  result.set_read_from_device_time_ps(request.read_from_device_time_ps() /
                                      size);
  result.set_batching_request_delay_ps(request.batching_request_delay_ps() /
                                       size);
  result.set_batching_request_size(request.batching_request_size() / size);
  result.set_host_preprocessing_ps(request.host_preprocessing_ps() / size);
  result.set_host_batch_formation_ps(request.host_batch_formation_ps() / size);
  result.set_host_runtime_ps(request.host_runtime_ps() / size);
  result.set_host_postprocessing_ps(request.host_postprocessing_ps() / size);
  result.set_idle_time_ps(request.idle_time_ps() / size);
  return result;
}

void AggregateBatch(const BatchDetail& input, BatchDetail* result) {
  // In aggregated result, start_time is set to 0, and end time is set to the
  // sum of the duration of the input batches.
  result->set_end_time_ps(input.end_time_ps() - input.start_time_ps() +
                          result->end_time_ps());
  result->set_batch_delay_ps(result->batch_delay_ps() + input.batch_delay_ps());
  result->set_padding_amount(result->padding_amount() + input.padding_amount());
  result->set_batch_size_after_padding(result->batch_size_after_padding() +
                                       input.batch_size_after_padding());
  result->set_device_time_ps(result->device_time_ps() + input.device_time_ps());
}

BatchDetail GetAverageBatchDetails(const BatchDetail& batch, int64_t size) {
  BatchDetail result;
  if (size == 0) return result;
  // Average batch detail does not have a batch ID.
  result.set_batch_id(-1);
  result.set_start_time_ps(0);
  // Calculating average by dividing aggregated batch by size.
  result.set_end_time_ps(batch.end_time_ps() / size);
  result.set_batch_delay_ps(batch.batch_delay_ps() / size);
  result.set_padding_amount(batch.padding_amount() / size);
  result.set_batch_size_after_padding(batch.batch_size_after_padding() / size);
  result.set_device_time_ps(batch.device_time_ps() / size);
  return result;
}

void AggregatePerModelInferenceStats(InferenceStats* inference_stats) {
  for (auto& [model_index, per_model_stats] :
       *inference_stats->mutable_inference_stats_per_model()) {
    // TODO: remove batch size aggregation from request table.
    absl::flat_hash_map<int /*batch_id*/, const BatchDetail*> batch_id_to_batch;
    for (const BatchDetail& b : per_model_stats.batch_details()) {
      batch_id_to_batch[b.batch_id()] = &b;
    }

    // Aggregated result for all data.
    RequestDetail aggregated_r;
    BatchDetail aggregated_b;

    struct PerBatchSizeInfo {
      PerBatchSizeAggregatedResult result;
      int request_count;
      int batch_count;
    };
    // Aggregated result per batch size.
    absl::flat_hash_map<int /*batch_id*/, PerBatchSizeInfo> per_batch_size_info;

    for (const RequestDetail& r : per_model_stats.request_details()) {
      // Aggregate all data.
      AggregateRequest(r, &aggregated_r);
      // Aggregate per batch size.
      // TODO: remove batch size aggregation from request table.
      for (const auto batch_id : r.related_batch_ids()) {
        if (const BatchDetail* batch =
                ::tsl::gtl::FindPtrOrNull(batch_id_to_batch, batch_id)) {
          int batch_size = batch->batch_size_after_padding();
          auto& info = per_batch_size_info[batch_size];
          AggregateRequest(r, info.result.mutable_aggregated_request_result());
          info.request_count++;
        }
      }
    }

    for (const BatchDetail& b : per_model_stats.batch_details()) {
      // Aggregate all data.
      AggregateBatch(b, &aggregated_b);
      // Aggregate per batch size.
      int batch_size = b.batch_size_after_padding();
      auto& info = per_batch_size_info[batch_size];
      AggregateBatch(b, info.result.mutable_aggregated_batch_result());
      info.batch_count++;
    }

    *per_model_stats.mutable_aggregated_request_detail() =
        GetAverageRequestDetails(aggregated_r,
                                 per_model_stats.request_details().size());
    *per_model_stats.mutable_aggregated_batch_detail() = GetAverageBatchDetails(
        aggregated_b, per_model_stats.batch_details().size());

    std::vector<int> sorted_batch_sizes;
    for (const auto& [batch_size, _] : per_batch_size_info) {
      sorted_batch_sizes.push_back(batch_size);
    }
    std::sort(sorted_batch_sizes.begin(), sorted_batch_sizes.end());
    for (const int batch_size : sorted_batch_sizes) {
      auto* result = per_model_stats.add_per_batch_size_aggregated_result();
      result->set_batch_size(batch_size);
      auto& info = per_batch_size_info[batch_size];
      *result->mutable_aggregated_request_result() = GetAverageRequestDetails(
          info.result.aggregated_request_result(), info.request_count);
      result->set_request_throughput(info.request_count *
                                     per_model_stats.request_throughput() /
                                     per_model_stats.request_details_size());
      *result->mutable_aggregated_batch_result() = GetAverageBatchDetails(
          info.result.aggregated_batch_result(), info.batch_count);
      result->set_batch_throughput(info.batch_count *
                                   per_model_stats.batch_throughput() /
                                   per_model_stats.batch_details_size());
    }
  }
}

}  // namespace

void RegroupInferenceStatsByModel(InferenceStats* inference_stats) {
  if (inference_stats->inference_stats_per_host().empty()) {
    return;
  }
  std::vector<const tsl::protobuf::RepeatedPtrField<RequestDetail>*>
      all_requests_by_host;
  for (const auto& [host_id, per_host_inference_stats] :
       inference_stats->inference_stats_per_host()) {
    all_requests_by_host.push_back(&per_host_inference_stats.request_details());
  }
  std::vector<std::vector<const RequestDetail*>> requests_by_model_id;
  RegroupDataByModelId(inference_stats->model_id_db(), all_requests_by_host,
                       &requests_by_model_id);

  std::vector<const tsl::protobuf::RepeatedPtrField<BatchDetail>*>
      all_batches_by_host;
  for (const auto& [host_id, per_host_inference_stats] :
       inference_stats->inference_stats_per_host()) {
    all_batches_by_host.push_back(&per_host_inference_stats.batch_details());
  }
  std::vector<std::vector<const BatchDetail*>> batches_by_model_id;
  RegroupDataByModelId(inference_stats->model_id_db(), all_batches_by_host,
                       &batches_by_model_id);

  for (size_t index = 0; index < requests_by_model_id.size(); index++) {
    auto* per_model =
        &(*inference_stats->mutable_inference_stats_per_model())[index];
    for (const RequestDetail* request : requests_by_model_id[index]) {
      *per_model->add_request_details() = *request;
    }
    for (const BatchDetail* batch : batches_by_model_id[index]) {
      *per_model->add_batch_details() = *batch;
    }
    auto [request_throughput, request_latency] =
        ComputeThroughputAndAverageLatencyUs(requests_by_model_id[index]);
    per_model->set_request_throughput(request_throughput);
    per_model->set_request_average_latency_us(request_latency);
    auto [batch_throughput, batch_latency] =
        ComputeThroughputAndAverageLatencyUs(batches_by_model_id[index]);
    per_model->set_batch_throughput(batch_throughput);
    per_model->set_batch_average_latency_us(batch_latency);
    GenerateTensorTransferAggregatedResult(per_model);
  }

  AggregatePerModelInferenceStats(inference_stats);

  // If there is no model id provided by user, create a fake "ALL" model id to
  // represent all the requests during profiling.
  // This ALL model id is mapped to index 0, which is consistent with the index
  // used by RegroupDataByModelId.
  if (inference_stats->model_id_db().ids().empty()) {
    inference_stats->mutable_model_id_db()->add_ids("ALL");
    inference_stats->mutable_model_id_db()->mutable_id_to_index()->insert(
        {"ALL", 0});
  }
  inference_stats->clear_inference_stats_per_host();
}

}  // namespace tensorflow::profiler
