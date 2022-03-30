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

#ifndef TENSORFLOW_CORE_FRAMEWORK_METRICS_H_
#define TENSORFLOW_CORE_FRAMEWORK_METRICS_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace metrics {

// Records that a tf.data.Dataset executed by the program used autotuning.
//
// The `name` argument identifies the Dataset type (e.g. "ParallelMap").
void RecordTFDataAutotune(const string& name);

// Returns a counter that can be used to record the number of bytes produced by
// a tf.data.Dataset.
//
// The `name` argument identifies the Dataset type (e.g. "Batch" or "Map").
monitoring::CounterCell* GetTFDataBytesConsumedCounter(const string& name);

// Returns a counter that can be used to record the number of bytes produced by
// a tf.data.Dataset.
//
// The `name` argument identifies the Dataset type (e.g. "Batch" or "Map").
monitoring::CounterCell* GetTFDataBytesProducedCounter(const string& name);

// Returns a counter than can be used to record the number of bytes read from
// the filesystem by a tf.data.Dataset source.
//
// The `name` argument identifies the Dataset type (e.g. "TFRecordDataset").
//
// TODO(jsimsa): Remove this now that we have GetTFDataBytesConsumedCounter?
monitoring::CounterCell* GetTFDataBytesReadCounter(const string& name);

// Returns a counter than can be used to record the number of elements produced
// by a tf.data.Dataset.
//
// The `name` argument identifies the Dataset type (e.g. "Batch" or "Map").
monitoring::CounterCell* GetTFDataElementsCounter(const string& name);

// Returns a gauge than can be used to record the performance model information.
//
// The `id` argument represents the (unique) model ID.
monitoring::GaugeCell<std::function<std::string()>>* GetTFDataModelGauge(
    const string& id);

// Records the number of bytes fetched from tf.data.Dataset iterator.
void RecordTFDataBytesFetched(int64_t num_bytes);

// Records the number of times tf.data experiment is applied to input pipelines.
void RecordTFDataExperiment(const string& name);

// Records the time (in microseconds) spent in a single invocation of
// `ItertatorResource::GetNext()`.
void RecordTFDataGetNextDuration(uint64 duration_us);

// Records the histogram of ratios of tf.data autotune algorithm used RAM over
// the ram budget.
void RecordTFDataAutotuneUsedRamBudgetRatio(const double ratio);

// Records the histogram of ratios of tf.data autotune algorithm max buffer
// bytes over the ram budget.
void RecordTFDataAutotuneMaxBufferBudgetRatio(const double ratio);

// Records the number of times each tf.data fingerprint is used
// to measure duplicate pre-processing.
//
// The `name` argument identifies the Dataset graph fingerprint,
// created using GraphHash().
void RecordTFDataFingerprint(const string& name);

// Records the time (in microseconds) during which `IteratorResource` was busy
// processing at least one `GetNext()` request.
void RecordTFDataIteratorBusy(uint64 duration_us);

// Records the time (in microseconds) between `IteratorResource` receiving the
// first `GetNext()` request and responding to the last `GetNext()` request.
void RecordTFDataIteratorLifetime(uint64 duration_us);

// Records the time histogram (in microseconds) between `IteratorResource`
// responding to a `GetNext()` request and receiving the next `GetNext()`
// request.
void RecordTFDataIteratorGap(uint64 duration_us);

// Records the number of independent graph changes resulting from the
// application of a tf.data optimization.
//
// The `name` argument identifies the optimization (e.g. "noop_elimination").
void RecordTFDataOptimization(const string& name, int64_t num_changes);

// Records that a tf.data service worker has been created.
void RecordTFDataServiceWorkerCreated();

// Records that a tf.data service job has been created.
void RecordTFDataServiceJobsCreated(
    const tensorflow::data::ProcessingModeDef& processing_mode,
    bool is_coordinated_read);

// Records tf.data service iterators created by clients.
void RecordTFDataServiceClientIterators(
    int64_t worker_uid, tensorflow::data::DeploymentMode deployment_mode,
    const tensorflow::data::ProcessingModeDef& processing_mode,
    bool is_coordinated_read);

// Records tf.data service multi-trainer cache queries.
void RecordTFDataServiceMultiTrainerCacheQuery(bool cache_hit);

// Records the file name read by a tf.data Dataset.
//
// The `name` argument identifies the Dataset type (e.g. "TFRecordDataset").
void RecordTFDataFilename(const string& name, const string& filename);

// Records statistics of tf.data auto sharding.
//
// The `id` is a unique identifier of the input pipeline. The `policy`
// identifies the auto-sharding policy used, the `num_workers` identifies the
// number of workers, and `num_replicas` identifies the number of replicas.
void RecordTFDataAutoShard(const string& id, data::AutoShardPolicy policy,
                           int64 num_workers, int64 num_replicas);

// Records statistics of whether we can rewrite batch size in tf.data auto
// sharding.
//
// The `id` is a unique identifier of the input pipeline. The `eligible`
// indicates whether the input pipeline is eligible for the rewrite. The
// `ineligible_reason` is the reason if the input pipeline is ineligible.
void RecordTFDataAutoShardRewriteBatchSize(
    bool eligible, const std::vector<string>& ineligible_reason);

// Records the number of times each tf.data autotuning algorithm stopping
// criterion is met.
void RecordTFDataAutotuneStoppingCriteria(const string& name);

// Records parsing of dense tensor features.
void RecordParseDenseFeature(int64_t num_features);

// Records parsing of sparse tensor features.
void RecordParseSparseFeature(int64_t num_features);

// Records parsing of ragged tensor features.
void RecordParseRaggedFeature(int64_t num_features);

// Records the size of input/output tensors in bytes.
void RecordGraphInputTensors(const size_t size);
void RecordGraphOutputTensors(const size_t size);

// Records the number of cores requested by graphs with XLA SPMD enabled.
void RecordTPUXlaSpmdCoresPerReplica(int64_t cores_per_replica);

void UpdateGraphExecTime(const uint64 running_time_usecs);
void UpdateGraphPendingQueueLength(uint64 len);

// Records that one output of an op of type `op_name` was unused.
void RecordUnusedOutput(const string& op_name);

// Updates the metrics stored about time spent building graphs.
//
// By "GraphBuild", we refer to building a client graph, which is a sub-graph of
// the full graph, induced by a set of options. In particular, these options
// include the feeds and fetches requested.
//
// This includes time spent:
//   * optimizing the graphs with Grappler
//   * pruning the sub-graph (unless the place_pruned_graph option is set)
//
// When executing eagerly, this will not record any activity.
//
// TODO(jtkeeling): Should we record building/optimizing tf.functions?
void UpdateGraphBuildTime(const uint64 running_time_usecs);

// Records the status of a graph passing through various states/stages of
// TfMlirGraphOptimizationPass processing using
// tf_metadata.tf_mlir_update_graph_optimization_pass_state_counter metric.
// 'pass_state' identifies the state of the pass
// (or "PassState" metric field) and 'processing_state' refers to the stage
// in the process the graph is at (or "ProcessingState" metric field).
void UpdateTfMlirGraphOptimizationPassStateCounter(
    const std::string& pass_state, const std::string& processing_state);

// Records the activity of the first phase of the mlir bridge using the
// tf_metadata.tf_mlir_bridge_first_phase_count metric.
// device_type: tpu, cpu, gpu, etc.
// bridge_version: v1 compat, v2, etc.
// fallback_enabled: true if fallback will happen, false if not
// result: outcome of bridge (success, failure, disabled, invalid_graph, etc.)
void UpdateTfMlirBridgeFirstPhaseCounter(const std::string& device_type,
                                         const std::string& bridge_version,
                                         bool fallback_enabled,
                                         const std::string& result);

// Convenience class allowing RAII style of reporting for a monitoring::Counter.
template <int NumLabels>
class ScopedCounter final {
 public:
  ScopedCounter(monitoring::Counter<NumLabels>* const counter,
                const std::array<std::string, NumLabels>& labels)
      : counter_(counter), labels_(labels) {
    Init();
  }

  // Report counter and stop it. Counter needs to be reset to perform
  // next measurement.
  void ReportAndStop() {
    if (started_) {
      started_ = false;
      ReportInternal(std::make_index_sequence<NumLabels>());
    }
  }

  // Start the measurement with the new set of labels.
  void Reset(const std::array<std::string, NumLabels>& labels) {
    labels_ = labels;
    Init();
  }

  // Start the measurement with the existing set of labels.
  void Reset() { Init(); }

  // Returns duration of the current interval in case the timer has started.
  // Returns nullopt otherwise.
  absl::optional<uint64> DurationMicroSec() const {
    return started_ ? absl::optional<uint64>(
                          accumulated_time_ +
                          tensorflow::Env::Default()->NowMicros() - start_time_)
                    : absl::nullopt;
  }

  // Temporarily stop the timer, but keep accumulated time.
  void AccumulateAndStop() {
    if (started_) {
      accumulated_time_ = tensorflow::Env::Default()->NowMicros() - start_time_;
      started_ = false;
    }
  }

  // Start previously stopped timer.
  void Start() {
    if (started_) return;

    // Keep previously accumulated time if any.
    start_time_ = tensorflow::Env::Default()->NowMicros();
    started_ = true;
  }

  ~ScopedCounter() { ReportAndStop(); }

 private:
  template <std::size_t... S>
  void ReportInternal(std::index_sequence<S...>) {
    uint64 time_interval =
        tensorflow::Env::Default()->NowMicros() - start_time_;
    time_interval += accumulated_time_;
    if (time_interval > 0) {
      counter_->GetCell(labels_[S]...)->IncrementBy(time_interval);
    }
  }

  void Init() {
    start_time_ = tensorflow::Env::Default()->NowMicros();
    started_ = true;
    accumulated_time_ = 0;
  }

  monitoring::Counter<NumLabels>* counter_;
  std::array<std::string, NumLabels> labels_;
  bool started_{false};
  uint64 start_time_;
  uint64 accumulated_time_;
};

// Returns a counter used to capture timing metrics for graph optimization
// passes.
monitoring::Counter<2>* GetGraphOptimizationCounter();

// Updates metrics for time to distribute variables to all TPU hosts.
void UpdateTpuVariableDistributionTime(const uint64 distribution_time_usecs);

// Updates the metrics stored about time XLA spents compiling graphs.
void UpdateXlaCompilationTime(const uint64 compilation_time_usecs);

// Updates the metrics stored about time BFC allocator spents during delay.
void UpdateBfcAllocatorDelayTime(const uint64 delay_usecs);

// Increments (by 1) a simple integer counter that is exposed for testing.
void IncrementTestCounter(const string& name, const string& label);

// Read-only access to a counter for testing.
const monitoring::CounterCell* TestCounter(const string& name,
                                           const string& label);

// Read-only wrapper for a TestCounter to track increments between calls.
class TestDelta {
 public:
  TestDelta(const string& name, const string& label);
  void Reset();
  int64 Get();

 private:
  const monitoring::CounterCell* cell_;
  int64 last_value_;
};
void UpdateTpuErrorCounter(const string& op, const string& error_type);

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_METRICS_H_
