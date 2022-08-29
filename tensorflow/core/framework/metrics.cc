/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/metrics.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace metrics {
namespace {

auto* graph_runs = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_runs",
    "The number of graph executions used to collect "
    "/tensorflow/core/graph_run_time_usecs");

auto* graph_run_time_usecs = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_run_time_usecs",
    "The total time spent on executing graphs in microseconds.");

auto* graph_run_time_usecs_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_time_usecs_histogram",
     "The wall-clock time spent on executing graphs in microseconds."},
    // Power of 2 with bucket count 20 (> 17 minutes)
    {monitoring::Buckets::Exponential(1000, 2, 20)});

auto* graph_pending_queue_length_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_pending_queue_length_histogram",
     "The number of pending (ready but not running) tasks in graph executor."},
    // Power of 1.5 with bucket count 30 (> 191k)
    {monitoring::Buckets::Exponential(1, 1.5, 30)});

auto* graph_run_input_tensor_bytes = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_input_tensor_bytes",
     "The size of input tensors in bytes."},
    // Power of 2 with bucket count 14 (256MB)
    {monitoring::Buckets::Exponential(1, 4, 14)});

auto* graph_run_output_tensor_bytes = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_output_tensor_bytes",
     "The size of output tensors in bytes."},
    // Power of 2 with bucket count 14 (256MB)
    {monitoring::Buckets::Exponential(1, 4, 14)});

auto* graph_unused_outputs = monitoring::Counter<1>::New(
    "/tensorflow/core/graph_unused_outputs",
    "The number of unused outputs for ops of a given type.", "name");

auto* tf_data_autotune_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/autotune", "tf.data autotuning", "name");

auto* tf_data_bytes_consumed_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_consumed",
    "The number of bytes consumed by a tf.data Dataset.", "name");

auto* tf_data_bytes_produced_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_produced",
    "The number of bytes produced by a tf.data Dataset.", "name");

auto* tf_data_bytes_read_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_read",
    "The number of bytes read by tf.data Dataset sources.", "name");

auto* tf_data_bytes_fetched_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/bytes_fetched",
    "The number of bytes fetched from tf.data Dataset iterator.");

auto* tf_data_elements_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/elements", "tf.data elements", "name");

auto* tf_data_experiment_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/experiment",
    "The number of times tf.data experiment is applied to input pipelines.",
    "name");

auto* tf_data_fingerprint_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/fingerprint", "tf.data fingerprint", "name");

auto* tf_data_get_next_duration_usecs_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/data/getnext_duration",
     "Microseconds spent fetching an element from tf.data iterator."},
    // Power of 2 with bucket count 10 (1024 microseconds) and 1 second.
    {monitoring::Buckets::Explicit(
        {2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., 1e6})});

auto* tf_data_used_vs_budget_ratio_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/data/used_vs_budget_ratio",
     "Ratio of tf.data used ram over ram budget when running optimization."},
    // Uniform linear buckets with count 10 from 0 to 2
    {monitoring::Buckets::Explicit(
        {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0})});

auto* tf_data_buffered_vs_budget_ratio_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/data/buffered_vs_budget_ratio",
     "Ratio of tf.data max buffer bytes over ram budget when running "
     "optimization."},
    // Uniform linear buckets with count 10 from 0 to 2
    {monitoring::Buckets::Explicit(
        {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0})});

auto* tf_data_iterator_busy_counter =
    monitoring::Counter<0>::New("/tensorflow/data/iterator_busy",
                                "The time (in microseconds) during which a "
                                "tf.data iterator was busy processing at "
                                "least one `GetNext()` request.");

auto* tf_data_iterator_lifetime_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/iterator_lifetime",
    "The time (in microseconds) between a tf.data iterator receiving the first "
    "`GetNext()` request and responding to the last `GetNext()` request.");

auto* tf_data_iterator_gap_msec_histogram = monitoring::Sampler<0>::New(
    {"/tensorflow/data/iterator_gap",
     "The time (in milliseconds) between a tf.data iterator responding to a "
     "`GetNext()` request and receiving the next `GetNext()` request."},
    // Power of 1.5 with bucket count of 20 (from 1 msec to about 2.2 secs).
    {monitoring::Buckets::Exponential(1, 1.5, 20)});

auto* tf_data_optimization_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/optimization", "tf.data optimization", "name");

auto* tf_data_service_workers_created_counter =
    monitoring::Counter<0>::New("/tensorflow/data/service/workers_created",
                                "Number of tf.data service workers created");

auto* tf_data_service_jobs_created_counter = monitoring::Counter<2>::New(
    "/tensorflow/data/service/jobs_created", "Number of tf.data service jobs.",
    "processing_mode", "coordinated_read");

auto* tf_data_service_client_iterators_counter = monitoring::Counter<4>::New(
    "/tensorflow/data/service/client_iterators",
    "Number of tf.data service client iterators created.", "worker_uid",
    "deployment_mode", "processing_mode", "is_coordinated_read");

auto* tf_data_service_cross_trainer_cache_queries_counter =
    monitoring::Counter<1>::New(
        "/tensorflow/data/service/cross_trainer_cache_queries",
        "tf.data service cross-trainer cache queries counter. The result can "
        "be hit or miss.",
        "cache_hit");

auto* tf_data_service_cross_trainer_cache_size_bytes =
    monitoring::Gauge<int64_t, 0>::New(
        "/tensorflow/data/service/cross_trainer_cache_size_bytes",
        "tf.data service cross-trainer cache memory usage in bytes.");

auto* tf_data_filename_counter = monitoring::Counter<2>::New(
    "/tensorflow/data/filename", "The file name read by a tf.data Dataset.",
    "name", "filename");

auto* tf_data_model_gauge =
    monitoring::Gauge<std::function<std::string()>, 1>::New(
        "/tensorflow/data/model", "tf.data autotuning model proto.", "id");

auto* tf_data_auto_shard = monitoring::Gauge<int64, 2>::New(
    "/tensorflow/data/autoshard", "tf.data autoshard statistics.", "id",
    "name");

auto* tf_data_auto_shard_rewrite_batch_size_eligible =
    monitoring::Counter<1>::New(
        "/tensorflow/data/autoshard_rewrite_batch_size/eligible",
        "Whether tf.data pipelines that are eligible for autoshard "
        "to rewrite the batch size.",
        "eligible");

auto* tf_data_auto_shard_rewrite_batch_size_reason =
    monitoring::Counter<1>::New(
        "/tensorflow/data/autoshard_rewrite_batch_size/reason",
        "The reasons that tf.data pipelines are ineligible for autoshard "
        "to rewrite the batch size.",
        "reason");

auto* tf_data_autotune_stopping_criteria_counter =
    monitoring::Counter<1>::New("/tensorflow/data/autotune_stopping_criteria",
                                "The number of times each tf.data autotune "
                                "algorithm stopping criterion is met.",
                                "name");

auto* parse_dense_feature_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/dense_feature",
    "The number of dense features parsed by ops for parsing tf.Example.");

auto* parse_sparse_feature_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/sparse_feature",
    "The number of sparse features parsed by ops for parsing tf.Example.");

auto* parse_ragged_feature_counter = monitoring::Counter<0>::New(
    "/tensorflow/data/ragged_feature",
    "The number of ragged features parsed by ops for parsing tf.Example.");

auto* build_graph_calls = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_build_calls",
    "The number of times TensorFlow has created a new client graph. "
    "A client graph is a sub-graph of the full graph, induced by a set of "
    "options, including the requested feeds and fetches. It includes time "
    "spent optimizing the graph with Grappler, and time spent pruning the "
    "sub-graph.");

auto* build_graph_time_usecs = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_build_time_usecs",
    "The amount of time TensorFlow has spent creating new client graphs in "
    "microseconds. "
    "A client graph is a sub-graph of the full graph, induced by a set of "
    "options, including the requested feeds and fetches. It includes time "
    "spent optimizing the graph with Grappler, and time spent pruning the "
    "sub-graph.");

auto* xla_compilations = monitoring::Counter<0>::New(
    "/tensorflow/core/xla_compilations",
    "The number of XLA compilations used to collect "
    "/tensorflow/core/xla_compilation_time_usecs");

auto* xla_compilation_time_usecs = monitoring::Counter<0>::New(
    "/tensorflow/core/xla_compilation_time_usecs",
    "The total time spent on compiling XLA graphs in microseconds.");

auto* xla_tpu_spmd_cores_per_replica = monitoring::Counter<1>::New(
    "/tensorflow/tpu/xla_spmd_cores_per_replica",
    "The number of cores used by XLA SPMD-replicated models.", "cores");

auto* bfc_allocator_delay =
    monitoring::Counter<0>::New("/tensorflow/core/bfc_allocator_delay",
                                "The total time spent running each graph "
                                "optimization pass in microseconds.");

auto* tpu_variable_distribution_time_usecs = monitoring::Counter<0>::New(
    "/tensorflow/tpu/variable_distribution_time",
    "Time spent sending variables from primary task to other worker tasks "
    "at the start of a call to TPUExecute.  Timer starts at RunGraph "
    "invocation and ends when TPUExecute args are ready on the current task.");

auto* test_counters =
    monitoring::Counter<2>::New("/tensorflow/core/test_counters",
                                "Counters used for testing.", "name", "label");

}  // namespace

auto* tpu_op_error_counter = monitoring::Counter<2>::New(
    "/tensorflow/tpu/op_error_count",
    "Count the tpu related errors by op and error_type.", "op", "error_type");

auto* eager_client_error_counter = monitoring::Counter<2>::New(
    "/tensorflow/core/eager_client_error_count",
    "Count the errors in eager client as a central place.", "error_source",
    "error_type");

monitoring::Counter<2>* GetGraphOptimizationCounter() {
  static auto* graph_optimization_counter =
      monitoring::Counter<2>::New("/tensorflow/core/graph_optimization_usecs",
                                  "The total time spent running each graph "
                                  "optimization pass in microseconds.",
                                  "kind", "name");
  return graph_optimization_counter;
}

void RecordTFDataAutotune(const string& name) {
  tf_data_autotune_counter->GetCell(name)->IncrementBy(1);
}

monitoring::CounterCell* GetTFDataBytesConsumedCounter(const string& name) {
  return tf_data_bytes_consumed_counter->GetCell(name);
}

monitoring::CounterCell* GetTFDataBytesProducedCounter(const string& name) {
  return tf_data_bytes_produced_counter->GetCell(name);
}

monitoring::CounterCell* GetTFDataBytesReadCounter(const string& name) {
  return tf_data_bytes_read_counter->GetCell(name);
}

monitoring::CounterCell* GetTFDataElementsCounter(const string& name) {
  return tf_data_elements_counter->GetCell(name);
}

monitoring::GaugeCell<std::function<std::string()>>* GetTFDataModelGauge(
    const string& id) {
  return tf_data_model_gauge->GetCell(id);
}

void RecordTFDataBytesFetched(int64_t num_bytes) {
  tf_data_bytes_fetched_counter->GetCell()->IncrementBy(num_bytes);
}

void RecordTFDataExperiment(const string& name) {
  tf_data_experiment_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataFingerprint(const string& name) {
  tf_data_fingerprint_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataGetNextDuration(uint64 duration_us) {
  static auto* tf_data_get_next_duration_cell =
      tf_data_get_next_duration_usecs_histogram->GetCell();
  tf_data_get_next_duration_cell->Add(duration_us);
}

void RecordTFDataAutotuneUsedRamBudgetRatio(const double ratio) {
  static auto* tf_data_used_vs_budget_ratio_histogram_cell =
      tf_data_used_vs_budget_ratio_histogram->GetCell();
  tf_data_used_vs_budget_ratio_histogram_cell->Add(ratio);
}

void RecordTFDataAutotuneMaxBufferBudgetRatio(const double ratio) {
  static auto* tf_data_buffered_vs_budget_ratio_histogram_cell =
      tf_data_buffered_vs_budget_ratio_histogram->GetCell();
  tf_data_buffered_vs_budget_ratio_histogram_cell->Add(ratio);
}

void RecordTFDataIteratorBusy(uint64 duration_us) {
  static auto* tf_data_iterator_busy_cell =
      tf_data_iterator_busy_counter->GetCell();
  tf_data_iterator_busy_cell->IncrementBy(duration_us);
}

void RecordTFDataIteratorLifetime(uint64 duration_us) {
  static auto* tf_data_iterator_lifetime_cell =
      tf_data_iterator_lifetime_counter->GetCell();
  tf_data_iterator_lifetime_cell->IncrementBy(duration_us);
}

void RecordTFDataIteratorGap(uint64 duration_us) {
  static auto* tf_data_iterator_gap_msec_histogram_cell =
      tf_data_iterator_gap_msec_histogram->GetCell();
  tf_data_iterator_gap_msec_histogram_cell->Add(duration_us * 0.001);
}

void RecordTFDataOptimization(const string& name, int64_t num_changes) {
  tf_data_optimization_counter->GetCell(name)->IncrementBy(num_changes);
}

void RecordTFDataServiceWorkerCreated() {
  tf_data_service_workers_created_counter->GetCell()->IncrementBy(1);
}

void RecordTFDataServiceJobsCreated(
    const tensorflow::data::ProcessingModeDef& processing_mode,
    bool is_coordinated_read) {
  const std::string sharding_policy_str =
      data::ProcessingModeDef::ShardingPolicy_Name(
          processing_mode.sharding_policy());
  const std::string coordinated_read_str =
      is_coordinated_read ? "true" : "false";
  tf_data_service_jobs_created_counter
      ->GetCell(sharding_policy_str, coordinated_read_str)
      ->IncrementBy(1);
}

void RecordTFDataServiceClientIterators(
    int64_t worker_uid, tensorflow::data::DeploymentMode deployment_mode,
    const tensorflow::data::ProcessingModeDef& processing_mode,
    bool is_coordinated_read) {
  const std::string deployment_mode_str =
      tensorflow::data::DeploymentMode_Name(deployment_mode);
  const std::string sharding_policy_str =
      data::ProcessingModeDef::ShardingPolicy_Name(
          processing_mode.sharding_policy());
  const std::string coordinated_read_str =
      is_coordinated_read ? "true" : "false";
  tf_data_service_client_iterators_counter
      ->GetCell(absl::StrCat(worker_uid), deployment_mode_str,
                sharding_policy_str, coordinated_read_str)
      ->IncrementBy(1);
}

void RecordTFDataServiceCrossTrainerCacheQuery(bool cache_hit) {
  std::string cache_hit_str = cache_hit ? "true" : "false";
  tf_data_service_cross_trainer_cache_queries_counter->GetCell(cache_hit_str)
      ->IncrementBy(1);
}

void RecordTFDataServiceCrossTrainerCacheSizeBytes(size_t bytes) {
  tf_data_service_cross_trainer_cache_size_bytes->GetCell()->Set(
      static_cast<int64_t>(bytes));
}

void RecordTFDataFilename(const string& name, const string& filename) {
  tf_data_filename_counter->GetCell(name, filename)->IncrementBy(1);
}

void RecordTFDataAutoShard(const string& id, data::AutoShardPolicy policy,
                           int64 num_workers, int64 num_replicas) {
  tf_data_auto_shard->GetCell(id, "policy")->Set(static_cast<int64_t>(policy));
  tf_data_auto_shard->GetCell(id, "num_workers")->Set(num_workers);
  tf_data_auto_shard->GetCell(id, "num_replicas")->Set(num_replicas);
}

void RecordTFDataAutoShardRewriteBatchSize(
    bool eligible, const std::vector<string>& ineligible_reason) {
  tf_data_auto_shard_rewrite_batch_size_eligible
      ->GetCell(eligible ? "true" : "false")
      ->IncrementBy(1);
  for (const string& reason : ineligible_reason) {
    tf_data_auto_shard_rewrite_batch_size_reason->GetCell(reason)->IncrementBy(
        1);
  }
}

void RecordTFDataAutotuneStoppingCriteria(const string& name) {
  tf_data_autotune_stopping_criteria_counter->GetCell(name)->IncrementBy(1);
}

void RecordParseDenseFeature(int64 num_features) {
  static auto* parse_dense_feature_counter_cell =
      parse_dense_feature_counter->GetCell();
  parse_dense_feature_counter_cell->IncrementBy(num_features);
}

void RecordParseSparseFeature(int64_t num_features) {
  static auto* parse_sparse_feature_counter_cell =
      parse_sparse_feature_counter->GetCell();
  parse_sparse_feature_counter_cell->IncrementBy(num_features);
}

void RecordParseRaggedFeature(int64_t num_features) {
  static auto* parse_ragged_feature_counter_cell =
      parse_ragged_feature_counter->GetCell();
  parse_ragged_feature_counter_cell->IncrementBy(num_features);
}

void RecordGraphInputTensors(const size_t size) {
  static auto* graph_run_input_tensor_bytes_cell =
      graph_run_input_tensor_bytes->GetCell();
  graph_run_input_tensor_bytes_cell->Add(size);
}

void RecordGraphOutputTensors(const size_t size) {
  static auto* graph_run_output_tensor_bytes_cell =
      graph_run_output_tensor_bytes->GetCell();
  graph_run_output_tensor_bytes_cell->Add(size);
}

void RecordTPUXlaSpmdCoresPerReplica(int64_t cores_per_replica) {
  xla_tpu_spmd_cores_per_replica->GetCell(absl::StrCat(cores_per_replica))
      ->IncrementBy(1);
}

void UpdateGraphExecTime(const uint64 running_time_usecs) {
  if (running_time_usecs > 0) {
    static auto* graph_runs_cell = graph_runs->GetCell();
    static auto* graph_run_time_usecs_cell = graph_run_time_usecs->GetCell();
    static auto* graph_run_time_usecs_histogram_cell =
        graph_run_time_usecs_histogram->GetCell();
    graph_runs_cell->IncrementBy(1);
    graph_run_time_usecs_cell->IncrementBy(running_time_usecs);
    graph_run_time_usecs_histogram_cell->Add(running_time_usecs);
  }
}

void UpdateGraphPendingQueueLength(uint64 len) {
  static auto* graph_pending_queue_length_cell =
      graph_pending_queue_length_histogram->GetCell();
  graph_pending_queue_length_cell->Add(len);
}

void UpdateGraphBuildTime(const uint64 running_time_usecs) {
  if (running_time_usecs > 0) {
    static auto* build_graph_calls_cell = build_graph_calls->GetCell();
    static auto* build_graph_time_usecs_cell =
        build_graph_time_usecs->GetCell();
    build_graph_calls_cell->IncrementBy(1);
    build_graph_time_usecs_cell->IncrementBy(running_time_usecs);
  }
}

void UpdateTpuVariableDistributionTime(const uint64 distribution_time_usecs) {
  if (distribution_time_usecs > 0) {
    tpu_variable_distribution_time_usecs->GetCell()->IncrementBy(
        distribution_time_usecs);
  }
}

void UpdateXlaCompilationTime(const uint64 compilation_time_usecs) {
  if (compilation_time_usecs > 0) {
    static auto* xla_compilations_cell = xla_compilations->GetCell();
    static auto* xla_compilation_time_usecs_cell =
        xla_compilation_time_usecs->GetCell();
    xla_compilations_cell->IncrementBy(1);
    xla_compilation_time_usecs_cell->IncrementBy(compilation_time_usecs);
  }
}

void UpdateBfcAllocatorDelayTime(const uint64 delay_usecs) {
  static auto* bfc_allocator_delay_cell = bfc_allocator_delay->GetCell();
  if (delay_usecs > 0) {
    bfc_allocator_delay_cell->IncrementBy(delay_usecs);
  }
}

void RecordUnusedOutput(const string& op_name) {
  graph_unused_outputs->GetCell(op_name)->IncrementBy(1);
}

void IncrementTestCounter(const string& name, const string& label) {
  test_counters->GetCell(name, label)->IncrementBy(1);
}

const monitoring::CounterCell* TestCounter(const string& name,
                                           const string& label) {
  return test_counters->GetCell(name, label);
}

TestDelta::TestDelta(const string& name, const string& label)
    : cell_(TestCounter(name, label)) {
  Reset();
}

void TestDelta::Reset() { last_value_ = cell_->value(); }

int64 TestDelta::Get() { return cell_->value() - last_value_; }

void UpdateTfMlirBridgeFirstPhaseCounter(const std::string& device_type,
                                         const std::string& bridge_version,
                                         bool fallback_enabled,
                                         const std::string& result) {
  static auto* metric = monitoring::Counter<4>::New(
      "/tensorflow/core/tf_mlir_bridge_first_phase_count",
      "Tracks processing state in first phase of mlir bridge", "device",
      "version", "fallback", "result");
  std::string fallback_status =
      fallback_enabled ? "fallback_enabled" : "fallback_disabled";
  metric->GetCell(device_type, bridge_version, fallback_status, result)
      ->IncrementBy(1);
}

void UpdateTpuErrorCounter(const string& op, const string& error_type) {
  tpu_op_error_counter->GetCell(op, error_type)->IncrementBy(1);
}

void UpdateEagerClientErrorCounter(const string& error_source,
                                   const string& error_type) {
  eager_client_error_counter->GetCell(error_source, error_type)->IncrementBy(1);
}

void UpdateTfMlirBridgeGraphAnalysisPerOp(
    const std::string& op_name, const std::string& construction_context,
    bool is_single_core_inference_mode, const std::string& num_replicas,
    const std::string& num_cores_per_replica, const std::string& use_tpu,
    const std::string& allow_soft_placement,
    const std::string& use_spmd_for_xla_partitioning,
    const std::string& unsupported_reason, bool has_unsupported_features) {
  static auto* metric = monitoring::Counter<10>::New(
      "/tensorflow/core/tf_mlir_bridge_graph_analysis_per_op",
      "Tracks processing state per op in first phase of mlir bridge", "op_name",
      "construction_context", "is_single_core_inference_mode", "num_replicas",
      "num_cores_per_replica", "use_tpu", "allow_soft_placement",
      "use_spmd_for_xla_partitioning", "unsupported_reason",
      "has_unsupported_features");

  metric
      ->GetCell(op_name, construction_context,
                is_single_core_inference_mode ? "Yes" : "No", num_replicas,
                num_cores_per_replica, use_tpu, allow_soft_placement,
                use_spmd_for_xla_partitioning, unsupported_reason,
                has_unsupported_features ? "Yes" : "No")
      ->IncrementBy(1);
}

}  // namespace metrics
}  // namespace tensorflow
