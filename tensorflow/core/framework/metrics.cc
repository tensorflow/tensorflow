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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/monitoring/counter.h"
#include "xla/tsl/lib/monitoring/gauge.h"
#include "xla/tsl/lib/monitoring/sampler.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace metrics {
namespace {

auto* persistent_cache_load_count = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/persistent_cache_load_count",
    "The number of times a binary is loaded from the persistent cache.");

auto* aot_bef_mlir_load_count = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/aot_bef_mlir_load_count",
    "The number of times BEF and MLIR are deserialized instead of generated "
    "and used.");

auto* graph_runs = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/graph_runs",
    "The number of graph executions used to collect "
    "/tensorflow/core/graph_run_time_usecs");

auto* graph_run_time_usecs = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/graph_run_time_usecs",
    "The total time spent on executing graphs in microseconds.");

auto* graph_run_time_usecs_histogram = tsl::monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_time_usecs_histogram",
     "The wall-clock time spent on executing graphs in microseconds."},
    // Power of 2 with bucket count 20 (> 17 minutes)
    {tsl::monitoring::Buckets::Exponential(1000, 2, 20)});

auto* graph_pending_queue_length_histogram = tsl::monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_pending_queue_length_histogram",
     "The number of pending (ready but not running) tasks in graph executor."},
    // Power of 1.5 with bucket count 30 (> 191k)
    {tsl::monitoring::Buckets::Exponential(1, 1.5, 30)});

auto* graph_run_input_tensor_bytes = tsl::monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_input_tensor_bytes",
     "The size of input tensors in bytes."},
    // Power of 2 with bucket count 14 (256MB)
    {tsl::monitoring::Buckets::Exponential(1, 4, 14)});

auto* graph_run_output_tensor_bytes = tsl::monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_output_tensor_bytes",
     "The size of output tensors in bytes."},
    // Power of 2 with bucket count 14 (256MB)
    {tsl::monitoring::Buckets::Exponential(1, 4, 14)});

auto* graph_unused_outputs = tsl::monitoring::Counter<1>::New(
    "/tensorflow/core/graph_unused_outputs",
    "The number of unused outputs for ops of a given type.", "name");

auto* tf_data_fetch_op_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/fetch_op",
    "The number of times a tf.data operation that fetches output(s) of a "
    "tf.data input pipeline (e.g. `IteratorGetNext`) was executed.",
    "fetch_op");

auto* tf_data_autotune_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/autotune", "tf.data autotuning", "name");

auto* tf_data_bytes_consumed_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_consumed",
    "The number of bytes consumed by a tf.data Dataset.", "name");

auto* tf_data_bytes_produced_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_produced",
    "The number of bytes produced by a tf.data Dataset.", "name");

auto* tf_data_bytes_read_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_read",
    "The number of bytes read by tf.data Dataset sources.", "name");

auto* tf_data_bytes_fetched_counter = tsl::monitoring::Counter<0>::New(
    "/tensorflow/data/bytes_fetched",
    "The number of bytes fetched from tf.data Dataset iterator.");

auto* tf_data_elements_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/elements", "tf.data elements", "name");

auto* tf_data_experiment_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/experiment",
    "The number of times a tf.data experiment was applied.", "name");

auto* tf_data_experiment_live_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/experiment_live",
    "The number of times a tf.data experiment could have been applied.",
    "name");

auto* tf_data_experiment_opt_in_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/experiment_opt_in",
    "The number of times a tf.data experiment was opted into. Values are "
    "either (1) the name of the experiment or (2) `\"all\"` (for all "
    "experiments in `/tensorflow/data/experiment_live`).",
    "name");

auto* tf_data_experiment_opt_out_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/experiment_opt_out",
    "The number of times a tf.data experiment was opted out of. Values are (1) "
    "the name of the experiment, (2) `\"all\"` (for all experiments in "
    "`/tensorflow/data/experiment_live`), or (3) `\"all_except_opt_in\"` (for "
    "all experiments in `/tensorflow/data/experiment_live` and not in "
    "`/tensor/data/experiment_opt_out`).",
    "name");

auto* tf_data_fingerprint_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/fingerprint", "tf.data fingerprint", "name");

auto* tf_data_service_compression = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/service/compression",
    "The number of times a tf.data service pipeline performed a "
    "compression-related action {'disabled_at_runtime', "
    "'not_disabled_at_runtime', 'not_eligible'}.",
    "action");

auto* tf_data_service_get_element_duration_usecs_histogram =
    tsl::monitoring::Sampler<1>::New(
        {"/tensorflow/data/getelement_duration",
         "Microseconds spent generating an element and transferring it over "
         "the network for the given protocol.",
         "data_transfer_protocol"},
        // Power of 2 with bucket count 10 (1024 microseconds) and 10-1000 ms.
        {tsl::monitoring::Buckets::Explicit({2., 4., 8., 16., 32., 64., 128.,
                                             256., 512., 1024., 1e4, 1e5,
                                             1e6})});

auto* tf_data_get_next_duration_usecs_histogram =
    tsl::monitoring::Sampler<0>::New(
        {"/tensorflow/data/getnext_duration",
         "Microseconds spent fetching an element from tf.data iterator."},
        // Power of 2 with bucket count 10 (1024 microseconds) and 1 second.
        {tsl::monitoring::Buckets::Explicit(
            {2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., 1e6})});

auto* tf_data_used_vs_budget_ratio_histogram = tsl::monitoring::Sampler<0>::New(
    {"/tensorflow/data/used_vs_budget_ratio",
     "Ratio of tf.data used ram over ram budget when running optimization."},
    // Uniform linear buckets with count 10 from 0 to 2
    {tsl::monitoring::Buckets::Explicit(
        {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0})});

auto* tf_data_buffered_vs_budget_ratio_histogram =
    tsl::monitoring::Sampler<0>::New(
        {"/tensorflow/data/buffered_vs_budget_ratio",
         "Ratio of tf.data max buffer bytes over ram budget when running "
         "optimization."},
        // Uniform linear buckets with count 10 from 0 to 2
        {tsl::monitoring::Buckets::Explicit(
            {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0})});

auto* tf_data_iterator_busy_counter = tsl::monitoring::Counter<0>::New(
    "/tensorflow/data/iterator_busy",
    "The time (in microseconds) during which a "
    "tf.data iterator was busy processing at "
    "least one `GetNext()` request.");

auto* tf_data_iterator_lifetime_counter = tsl::monitoring::Counter<0>::New(
    "/tensorflow/data/iterator_lifetime",
    "The time (in microseconds) between a tf.data iterator receiving the first "
    "`GetNext()` request and responding to the last `GetNext()` request.");

auto* tf_data_iterator_gap_msec_histogram = tsl::monitoring::Sampler<0>::New(
    {"/tensorflow/data/iterator_gap",
     "The time (in milliseconds) between a tf.data iterator responding to a "
     "`GetNext()` request and receiving the next `GetNext()` request."},
    // Power of 1.5 with bucket count of 20 (from 1 msec to about 2.2 secs).
    {tsl::monitoring::Buckets::Exponential(1, 1.5, 20)});

auto* tf_data_optimization_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/optimization", "tf.data optimization", "name");

auto* tf_data_service_workers_created_counter =
    tsl::monitoring::Counter<0>::New(
        "/tensorflow/data/service/workers_created",
        "Number of tf.data service workers created");

auto* tf_data_service_jobs_created_counter = tsl::monitoring::Counter<2>::New(
    "/tensorflow/data/service/jobs_created", "Number of tf.data service jobs.",
    "processing_mode", "coordinated_read");

auto* tf_data_service_client_iterators_counter =
    tsl::monitoring::Counter<4>::New(
        "/tensorflow/data/service/client_iterators",
        "Number of tf.data service client iterators created.", "worker_uid",
        "deployment_mode", "processing_mode", "is_coordinated_read");

auto* tf_data_service_cross_trainer_cache_queries_counter =
    tsl::monitoring::Counter<1>::New(
        "/tensorflow/data/service/cross_trainer_cache_queries",
        "tf.data service cross-trainer cache queries counter. The result can "
        "be hit or miss.",
        "cache_hit");

auto* tf_data_service_cross_trainer_cache_size_bytes =
    tsl::monitoring::Gauge<int64_t, 0>::New(
        "/tensorflow/data/service/cross_trainer_cache_size_bytes",
        "tf.data service cross-trainer cache memory usage in bytes.");

auto* tf_data_service_snapshot_bytes_committed =
    tsl::monitoring::Counter<0>::New(
        "/tensorflow/data/service/snapshot_bytes_committed",
        "tf.data service distributed snapshot committed bytes.");

auto* tf_data_service_snapshot_ops_counter = tsl::monitoring::Counter<2>::New(
    "/tensorflow/data/service/snapshot_ops",
    "Number times a tf.data snapshot is saved/loaded.", "path", "op");

auto* tf_data_service_data_transfer_protocol_used =
    tsl::monitoring::Counter<1>::New(
        "/tensorflow/data/service/data_transfer_protocol_used",
        "The number of tf.data service worker clients created that use this "
        "data transfer protocol.",
        "data_transfer_protocol");

auto* tf_data_service_data_transfer_protocol_used_by_nature =
    tsl::monitoring::Counter<2>::New(
        "/tensorflow/data/service/data_transfer_protocol_used_by_nature",
        "The number of tf.data service worker clients created that use this "
        "data transfer protocol and the nature ('default' or 'specified') "
        "under which this protocol was chosen.",
        "data_transfer_protocol", "nature");

auto* tf_data_service_data_transfer_protocol_fallback =
    tsl::monitoring::Counter<3>::New(
        "/tensorflow/data/service/data_transfer_protocol_fallback",
        "The number of tf.data service worker clients created that fell back "
        "from using this data transfer protocol for this reason.",
        "data_transfer_protocol", "error_type", "error_message");

auto* tf_data_service_data_transfer_protocol_error =
    tsl::monitoring::Counter<3>::New(
        "/tensorflow/data/service/data_transfer_protocol_error",
        "The number of times a tf.data service worker client got this type "
        "of non-retriable error with this message when using this protocol.",
        "data_transfer_protocol", "error_type", "error_message");

auto* tf_data_service_optimal_number_of_workers =
    monitoring::Gauge<int64_t, 0>::New(
        "/tensorflow/data/service/optimal_number_of_workers",
        "Estimated optimal number of tf.data service workers based on the "
        "current workload.");

auto* tf_data_filename_counter = tsl::monitoring::Counter<2>::New(
    "/tensorflow/data/filename", "The file name read by a tf.data Dataset.",
    "name", "filename");

auto* tf_data_file_logger_attempts_counter = tsl::monitoring::Counter<0>::New(
    "/tensorflow/data/file_logger_attempts",
    "The number of times a file logger attempted to log filenames.");

auto* tf_data_file_logger_errors_counter = tsl::monitoring::Counter<2>::New(
    "/tensorflow/data/file_logger_errors",
    "The number of times file logger got error of this type and message.",
    "error_code", "error_message");

auto* tf_data_file_logger_attempted_num_files_counter =
    tsl::monitoring::Counter<0>::New(
        "/tensorflow/data/file_logger_attempts_num_files",
        "The number of files that were attempted to be logged by the file "
        "logger.");

auto* tf_data_file_logger_errors_num_files_counter =
    tsl::monitoring::Counter<2>::New(
        "/tensorflow/data/file_logger_errors_num_files",
        "The number of files that encountered errors of this type and message "
        "during logging by the file logger.",
        "error_code", "error_message");

auto* tf_data_model_gauge =
    tsl::monitoring::Gauge<std::function<std::string()>, 1>::New(
        "/tensorflow/data/model", "tf.data autotuning model proto.", "id");

auto* tf_data_pipeline_processing_time = tsl::monitoring::Gauge<double, 1>::New(
    "/tensorflow/data/pipeline_processing_time",
    "The total processing time of the slowest stage in the input pipeline "
    "in microseconds",
    "id");

auto* tf_data_auto_shard = tsl::monitoring::Gauge<int64, 2>::New(
    "/tensorflow/data/autoshard", "tf.data autoshard statistics.", "id",
    "name");

auto* tf_data_auto_shard_rewrite_batch_size_eligible =
    tsl::monitoring::Counter<1>::New(
        "/tensorflow/data/autoshard_rewrite_batch_size/eligible",
        "Whether tf.data pipelines that are eligible for autoshard "
        "to rewrite the batch size.",
        "eligible");

auto* tf_data_auto_shard_rewrite_batch_size_reason =
    tsl::monitoring::Counter<1>::New(
        "/tensorflow/data/autoshard_rewrite_batch_size/reason",
        "The reasons that tf.data pipelines are ineligible for autoshard "
        "to rewrite the batch size.",
        "reason");

auto* tf_data_autotune_stopping_criteria_counter =
    tsl::monitoring::Counter<1>::New(
        "/tensorflow/data/autotune_stopping_criteria",
        "The number of times each tf.data autotune "
        "algorithm stopping criterion is met.",
        "name");

auto* tf_data_debug = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/debug",
    "The number of times this event occured, for debugging.", "event");

auto* tf_data_error = tsl::monitoring::Counter<2>::New(
    "/tensorflow/data/error",
    "The number of times an error of this type occurred with this status code.",
    "error_type", "status_code");

auto* tf_data_framework_type = tsl::monitoring::Counter<1>::New(
    "/tensorflow/data/framework_type",
    "The framework type used to build the tf.data.Dataset.", "name");

auto* parse_dense_feature_counter = tsl::monitoring::Counter<0>::New(
    "/tensorflow/data/dense_feature",
    "The number of dense features parsed by ops for parsing tf.Example.");

auto* parse_sparse_feature_counter = tsl::monitoring::Counter<0>::New(
    "/tensorflow/data/sparse_feature",
    "The number of sparse features parsed by ops for parsing tf.Example.");

auto* parse_ragged_feature_counter = tsl::monitoring::Counter<0>::New(
    "/tensorflow/data/ragged_feature",
    "The number of ragged features parsed by ops for parsing tf.Example.");

auto* build_graph_calls = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/graph_build_calls",
    "The number of times TensorFlow has created a new client graph. "
    "A client graph is a sub-graph of the full graph, induced by a set of "
    "options, including the requested feeds and fetches. It includes time "
    "spent optimizing the graph with Grappler, and time spent pruning the "
    "sub-graph.");

auto* build_graph_time_usecs = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/graph_build_time_usecs",
    "The amount of time TensorFlow has spent creating new client graphs in "
    "microseconds. "
    "A client graph is a sub-graph of the full graph, induced by a set of "
    "options, including the requested feeds and fetches. It includes time "
    "spent optimizing the graph with Grappler, and time spent pruning the "
    "sub-graph.");

auto* function_graph_optimization_time_usecs = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/function_graph_optimization_time_usecs",
    "The amount of time TensorFlow has spent optimizing function graphs, in "
    "microseconds. ");

auto* graph_optimization_saving_time_usecs = tsl::monitoring::Counter<1>::New(
    "/tensorflow/core/graph_optimization_saving_time_usec",
    "The amount of time TensorFlow has saved by caching the optimized "
    "function graph, in microseconds",  // metric description
    "source"                            // graph optimization source
);

auto* graph_optimization_cache_hit_count = tsl::monitoring::Counter<1>::New(
    "/tensorflow/core/graph_optimization_cache_hit_count",
    "The number of times the cache for the graph optimization is hit.",
    "source"  // graph optimization source
);

auto* graph_optimization_cache_failure_count = tsl::monitoring::Counter<1>::New(
    "/tensorflow/core/graph_optimization_cache_failure_count",
    "The number of times restoring from the graph optimization cache "
    "fails.",
    "source"  // graph optimization source
);

auto* graph_optimization_cache_miss_count = tsl::monitoring::Counter<1>::New(
    "/tensorflow/core/graph_optimization_cache_miss_count",
    "The number of times the cache for the graph optimization is missed.",
    "source"  // graph optimization source
);

auto* graph_optimization_cache_load_count = tsl::monitoring::Counter<1>::New(
    "/tensorflow/core/graph_optimization_cache_load_count",
    "The number of times loading an optimized function graph to RAM.",
    "source"  // graph optimization source
);

auto* xla_compilations = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/xla_compilations",
    "The number of XLA compilations used to collect "
    "/tensorflow/core/xla_compilation_time_usecs");

auto* xla_compilation_time_usecs = tsl::monitoring::Counter<0>::New(
    "/tensorflow/core/xla_compilation_time_usecs",
    "The total time spent on compiling XLA graphs in microseconds.");

auto* xla_tpu_spmd_cores_per_replica = tsl::monitoring::Counter<1>::New(
    "/tensorflow/tpu/xla_spmd_cores_per_replica",
    "The number of cores used by XLA SPMD-replicated models.", "cores");

auto* tpu_variable_distribution_time_usecs = tsl::monitoring::Counter<0>::New(
    "/tensorflow/tpu/variable_distribution_time",
    "Time spent sending variables from primary task to other worker tasks "
    "at the start of a call to TPUExecute.  Timer starts at RunGraph "
    "invocation and ends when TPUExecute args are ready on the current task.");

auto* test_counters = tsl::monitoring::Counter<2>::New(
    "/tensorflow/core/test_counters", "Counters used for testing.", "name",
    "label");

}  // namespace

auto* tpu_op_error_counter = tsl::monitoring::Counter<2>::New(
    "/tensorflow/tpu/op_error_count",
    "Count the tpu related errors by op and error_type.", "op", "error_type");

auto* eager_client_error_counter = tsl::monitoring::Counter<2>::New(
    "/tensorflow/core/eager_client_error_count",
    "Count the errors in eager client as a central place.", "error_source",
    "error_type");

auto* mlir_bridge_first_phase_counter = tsl::monitoring::Counter<5>::New(
    "/tensorflow/core/tf_mlir_bridge_first_phase_v2_count",
    "Tracks processing state in first phase of mlir bridge", "bridge",
    "version", "device", "fallback", "result");

auto* mlir_second_phase_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/api/v2/phase2_compilation_status" /*metric_name*/,
    "Counts the number of graphs that were analyzed prior deciding whether "
    "the MLIR or the old bridge will be used" /* metric description */,
    "status" /* metric label */);

auto* phase_2_xla_compiler_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/compiler/tf2xla/xla_compiler/"
    "compilation_status" /*metric_name*/,
    "Counts the number of times the xla builder vs mlir was "
    "used for XlaCompiler entry points." /* metric description*/,
    "status" /* metric label */);

auto* tf1_features_by_graph_count = tsl::monitoring::Counter<5>::New(
    "/tensorflow/core/tf1_features_by_graph_count",
    "Marks which tf1 feature (if any) a graph contains.", "device", "context",
    "control_flow", "ref_variable", "manual_control_deps");

tsl::monitoring::Counter<2>* GetGraphOptimizationCounter() {
  static auto* graph_optimization_counter = tsl::monitoring::Counter<2>::New(
      "/tensorflow/core/graph_optimization_usecs",
      "The total time spent running each graph "
      "optimization pass in microseconds.",
      "kind", "name");
  return graph_optimization_counter;
}

std::string GraphOptimizationSourceMapping(GraphOptimizationSource source) {
  switch (source) {
    case GraphOptimizationSource::kJit:
      return "jit";
    case GraphOptimizationSource::kAot:
      return "aot";
    case GraphOptimizationSource::kUnknown:
      return "unknown";
    default:
      return "";
      LOG(ERROR) << "Unexpected value for GraphOptimizationSource: "
                 << absl::StrCat(source);
  }
}

void RecordTFDataFetchOp(const string& name) {
  tf_data_fetch_op_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataAutotune(const string& name) {
  tf_data_autotune_counter->GetCell(name)->IncrementBy(1);
}

tsl::monitoring::CounterCell* GetTFDataBytesConsumedCounter(
    const string& name) {
  return tf_data_bytes_consumed_counter->GetCell(name);
}

tsl::monitoring::CounterCell* GetTFDataBytesProducedCounter(
    const string& name) {
  return tf_data_bytes_produced_counter->GetCell(name);
}

tsl::monitoring::CounterCell* GetTFDataBytesReadCounter(const string& name) {
  return tf_data_bytes_read_counter->GetCell(name);
}

tsl::monitoring::CounterCell* GetTFDataElementsCounter(const string& name) {
  return tf_data_elements_counter->GetCell(name);
}

tsl::monitoring::GaugeCell<std::function<std::string()>>* GetTFDataModelGauge(
    const string& id) {
  return tf_data_model_gauge->GetCell(id);
}

tsl::monitoring::GaugeCell<double>* GetTFDataPipelineProcessingTimeGauge(
    const string& id) {
  return tf_data_pipeline_processing_time->GetCell(id);
}

void RecordTFDataBytesFetched(int64_t num_bytes) {
  tf_data_bytes_fetched_counter->GetCell()->IncrementBy(num_bytes);
}

void RecordTFDataExperiment(const string& name) {
  tf_data_experiment_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataExperimentLive(const string& name) {
  tf_data_experiment_live_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataExperimentOptIn(const string& name) {
  tf_data_experiment_opt_in_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataExperimentOptOut(const string& name) {
  tf_data_experiment_opt_out_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataFingerprint(const string& name) {
  tf_data_fingerprint_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataServiceRuntimeCompressionDecision(bool compression_disabled) {
  tf_data_service_compression
      ->GetCell(compression_disabled ? "disabled_at_runtime"
                                     : "not_disabled_at_runtime")
      ->IncrementBy(1);
}

void RecordTFDataServiceCompressionAction(const string& action) {
  tf_data_service_compression->GetCell(action)->IncrementBy(1);
}

void RecordTFDataServiceGetElementDuration(const string& data_transfer_protocol,
                                           uint64 duration_us) {
  tf_data_service_get_element_duration_usecs_histogram
      ->GetCell(data_transfer_protocol)
      ->Add(duration_us);
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
    const data::ProcessingModeDef& processing_mode, bool is_coordinated_read) {
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
    int64_t worker_uid, data::DeploymentMode deployment_mode,
    const data::ProcessingModeDef& processing_mode, bool is_coordinated_read) {
  const std::string deployment_mode_str =
      data::DeploymentMode_Name(deployment_mode);
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

void RecordTFDataServiceDataTransferProtocolUsed(
    const string& data_transfer_protocol, bool user_specified) {
  std::string nature = user_specified ? "specified" : "default";
  tf_data_service_data_transfer_protocol_used_by_nature
      ->GetCell(data_transfer_protocol, nature)
      ->IncrementBy(1);
}

void RecordTFDataServiceDataTransferProtocolFallback(
    const string& data_transfer_protocol, error::Code code,
    const string& error_message) {
  tf_data_service_data_transfer_protocol_fallback
      ->GetCell(data_transfer_protocol, error::Code_Name(code), error_message)
      ->IncrementBy(1);
}

void RecordTFDataServiceDataTransferProtocolError(
    const string& data_transfer_protocol, error::Code code,
    const string& error_message) {
  tf_data_service_data_transfer_protocol_error
      ->GetCell(data_transfer_protocol, error::Code_Name(code), error_message)
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

void RecordTFDataServiceSnapshotBytesCommitted(int64_t bytes) {
  tf_data_service_snapshot_bytes_committed->GetCell()->IncrementBy(bytes);
}

void RecordTFDataServiceSnapshotOp(const std::string& path,
                                   const std::string& op) {
  tf_data_service_snapshot_ops_counter->GetCell(path, op)->IncrementBy(1);
}

void RecordTFDataServiceOptimalNumberOfWorkers(int64_t number_of_workers) {
  tf_data_service_optimal_number_of_workers->GetCell()->Set(number_of_workers);
}

void RecordTFDataFilename(const string& name, const string& filename) {
  tf_data_filename_counter->GetCell(name, filename)->IncrementBy(1);
}

void RecordTFDataFileLoggerAttempts() {
  tf_data_file_logger_attempts_counter->GetCell()->IncrementBy(1);
}

void RecordTFDataFileLoggerErrors(error::Code error_code,
                                  const string& error_message) {
  tf_data_file_logger_errors_counter
      ->GetCell(error::Code_Name(error_code), error_message)
      ->IncrementBy(1);
}

void RecordTFDataFileLoggerAttemptedNumFiles(size_t num_files) {
  tf_data_file_logger_attempted_num_files_counter->GetCell()->IncrementBy(
      num_files);
}

void RecordTFDataFileLoggerErrorsNumFiles(size_t num_files,
                                          error::Code error_code,
                                          const string& error_message) {
  tf_data_file_logger_errors_num_files_counter
      ->GetCell(error::Code_Name(error_code), error_message)
      ->IncrementBy(num_files);
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

void RecordTFDataDebug(const string& event) {
  tf_data_debug->GetCell(event)->IncrementBy(1);
}

void RecordTFDataError(const string& error_type, const string& status_code) {
  tf_data_error->GetCell(error_type, status_code)->IncrementBy(1);
}

void RecordTFDataFrameworkType(const std::string& framework_type) {
  tf_data_framework_type->GetCell(framework_type)->IncrementBy(1);
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

void UpdatePersistentCacheLoadCount() {
  static auto* persistent_cache_load_count_cell =
      persistent_cache_load_count->GetCell();
  persistent_cache_load_count_cell->IncrementBy(1);
}

void UpdateAotBefMlirLoadCount() {
  static auto* aot_bef_mlir_load_count_cell =
      aot_bef_mlir_load_count->GetCell();
  aot_bef_mlir_load_count_cell->IncrementBy(1);
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

void UpdateFunctionGraphOptimizationTime(const uint64 running_time_usecs) {
  if (running_time_usecs > 0) {
    static auto* function_graph_optimization_time_usecs_cell =
        function_graph_optimization_time_usecs->GetCell();
    function_graph_optimization_time_usecs_cell->IncrementBy(
        running_time_usecs);
  }
}

void UpdateFunctionGraphOptimizationSavingTime(const uint64 saving_time_usecs,
                                               GraphOptimizationSource source) {
  if (saving_time_usecs > 0) {
    std::string mapped_source = GraphOptimizationSourceMapping(source);
    static auto* function_graph_optimization_saving_time_usecs_cell =
        graph_optimization_saving_time_usecs->GetCell(mapped_source);
    function_graph_optimization_saving_time_usecs_cell->IncrementBy(
        saving_time_usecs);
  }
}

uint64 GetFunctionGraphOptimizationSavingTimeUsecs(
    GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  return graph_optimization_saving_time_usecs->GetCell(mapped_source)->value();
}

void IncrementFunctionGraphOptimizationCacheHitCount(
    const int count, GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  graph_optimization_cache_hit_count->GetCell(mapped_source)
      ->IncrementBy(count);
}

int64_t GetFunctionGraphOptimizationCacheHitCount(
    GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  return graph_optimization_cache_hit_count->GetCell(mapped_source)->value();
}

void IncrementFunctionGraphOptimizationCacheFailureCount(
    const int count, GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  graph_optimization_cache_failure_count->GetCell(mapped_source)
      ->IncrementBy(count);
}

int64_t GetFunctionGraphOptimizationCacheFailureCount(
    GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  return graph_optimization_cache_failure_count->GetCell(mapped_source)
      ->value();
}

void IncrementFunctionGraphOptimizationCacheMissCount(
    const int count, GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  graph_optimization_cache_miss_count->GetCell(mapped_source)
      ->IncrementBy(count);
}

int64_t GetFunctionGraphOptimizationCacheMissCount(
    GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  return graph_optimization_cache_miss_count->GetCell(mapped_source)->value();
}

void IncrementFunctionGraphOptimizationCacheLoadCount(
    int count, GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  graph_optimization_cache_load_count->GetCell(mapped_source)
      ->IncrementBy(count);
}

int64_t GetFunctionGraphOptimizationCacheLoadCount(
    GraphOptimizationSource source) {
  std::string mapped_source = GraphOptimizationSourceMapping(source);
  return graph_optimization_cache_load_count->GetCell(mapped_source)->value();
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

void RecordUnusedOutput(const string& op_name) {
  graph_unused_outputs->GetCell(op_name)->IncrementBy(1);
}

void RecordPipelineProcessingTime(const string& id,
                                  double pipeline_processing_time_usec) {
  GetTFDataPipelineProcessingTimeGauge(id)->Set(pipeline_processing_time_usec);
}

void IncrementTestCounter(const string& name, const string& label) {
  test_counters->GetCell(name, label)->IncrementBy(1);
}

const tsl::monitoring::CounterCell* TestCounter(const string& name,
                                                const string& label) {
  return test_counters->GetCell(name, label);
}

TestDelta::TestDelta(const string& name, const string& label)
    : cell_(TestCounter(name, label)) {
  Reset();
}

void TestDelta::Reset() { last_value_ = cell_->value(); }

int64 TestDelta::Get() { return cell_->value() - last_value_; }

void UpdateTfMlirBridgeFirstPhaseCounter(const std::string& bridge_type,
                                         const std::string& bridge_version,
                                         const std::string& device_type,
                                         bool fallback_enabled,
                                         const std::string& result) {
  std::string fallback_status =
      fallback_enabled ? "fallback_enabled" : "fallback_disabled";
  mlir_bridge_first_phase_counter
      ->GetCell(bridge_type, bridge_version, device_type, fallback_status,
                result)
      ->IncrementBy(1);
}

// Records the activity of the second phase of the mlir bridge.
void IncrementTfMlirBridgeSecondPhaseCounter(
    MlirBridgeSecondPhaseMetric metric) {
  static auto* mlir_bridge_second_phase_metric_names =
      new absl::flat_hash_map<MlirBridgeSecondPhaseMetric, absl::string_view>{
          {MlirBridgeSecondPhaseMetric::kMlirWithFallbackModeSuccess,
           "kMlirWithFallbackModeSuccess"},
          {MlirBridgeSecondPhaseMetric::kMlirWithFallbackModeFailure,
           "kMlirWithFallbackModeFailure"},
          {MlirBridgeSecondPhaseMetric::kMlirModeSuccess, "kMlirModeSuccess"},
          {MlirBridgeSecondPhaseMetric::kMlirModeFailure, "kMlirModeFailure"},
          {MlirBridgeSecondPhaseMetric::kOldBridgeMlirFilteredSuccess,
           "kOldBridgeMlirFilteredSuccess"},
          {MlirBridgeSecondPhaseMetric::kOldBridgeMlirFilteredFailure,
           "kOldBridgeMlirFilteredFailure"},
          {MlirBridgeSecondPhaseMetric::kOldBridgeWithFallbackModeSuccess,
           "kOldBridgeWithFallbackModeSuccess"},
          {MlirBridgeSecondPhaseMetric::kOldBridgeWithFallbackModeFailure,
           "kOldBridgeWithFallbackModeFailure"},
          {MlirBridgeSecondPhaseMetric::kMlirCombinedMlirSuccess,
           "kMlirCombinedMlirSuccess"},
          {MlirBridgeSecondPhaseMetric::kMlirCombinedMlirFailure,
           "kMlirCombinedMlirFailure"},
          {MlirBridgeSecondPhaseMetric::kMlirCombinedOldSuccess,
           "kMlirCombinedOldSuccess"},
          {MlirBridgeSecondPhaseMetric::kMlirCombinedOldFailure,
           "kMlirCombinedOldFailure"},
      };

  mlir_second_phase_count
      ->GetCell(std::string(mlir_bridge_second_phase_metric_names->at(metric)))
      ->IncrementBy(1);
}

void IncrementPhase2XlaCompilerCounter(Phase2XlaCompilerMetric metric) {
  static auto* metric_names =
      new absl::flat_hash_map<Phase2XlaCompilerMetric, absl::string_view>{
          {Phase2XlaCompilerMetric::kCompileSingleOpXlaBuilderSuccess,
           "kCompileSingleOpXlaBuilderSuccess"},
          {Phase2XlaCompilerMetric::kCompileSingleOpXlaBuilderFailure,
           "kCompileSingleOpXlaBuilderFailure"},
          {Phase2XlaCompilerMetric::kCompileSingleOpMlirSuccess,
           "kCompileSingleOpMlirSuccess"},
          {Phase2XlaCompilerMetric::kCompileSingleOpMlirFailure,
           "kCompileSingleOpMlirFailure"},
          {Phase2XlaCompilerMetric::kCompileFunctionXlaBuilderSuccess,
           "kCompileFunctionXlaBuilderSuccess"},
          {Phase2XlaCompilerMetric::kCompileFunctionXlaBuilderFailure,
           "kCompileFunctionXlaBuilderFailure"},
          {Phase2XlaCompilerMetric::kCompileFunctionMlirSuccess,
           "kCompileFunctionMlirSuccess"},
          {Phase2XlaCompilerMetric::kCompileFunctionMlirFailure,
           "kCompileFunctionMlirFailure"},
      };

  phase_2_xla_compiler_count->GetCell(std::string(metric_names->at(metric)))
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
  static auto* metric = tsl::monitoring::Counter<10>::New(
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

void RecordTFVersionByGraphFeatures(const std::string& device,
                                    const std::string& context,
                                    bool hasControlFlowV1,
                                    bool hasReferenceVariables,
                                    bool hasManualControlDeps) {
  tf1_features_by_graph_count
      ->GetCell(device, context, hasControlFlowV1 ? "true" : "false",
                hasReferenceVariables ? "true" : "false",
                hasManualControlDeps ? "true" : "false")
      ->IncrementBy(1);
}

}  // namespace metrics
}  // namespace tensorflow
