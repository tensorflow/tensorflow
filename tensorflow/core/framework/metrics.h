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

#include <cstdint>
#include <string>

#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace metrics {
enum class GraphOptimizationSource {
  kUnknown,
  kJit,
  kAot,
};

// Records when a data-fetching tf.data operation is executed.
//
// The `name` argument identifies the operation type (e.g. "ToSingleElementOp").
void RecordTFDataFetchOp(const string& name);

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

// Records the number of times a tf.data experiment was applied.
void RecordTFDataExperiment(const string& name);

// Records the number of times a tf.data experiment could have been applied.
void RecordTFDataExperimentLive(const string& name);

// Records the number of times a tf.data experiment was opted into.
void RecordTFDataExperimentOptIn(const string& experiment_name);

// Records the number of times a tf.data experiment was opted out of.
void RecordTFDataExperimentOptOut(const string& experiment_name);

// Records the time (in microseconds) spent generating an element and
// transferring it over the network for the given protocol.
void RecordTFDataServiceGetElementDuration(const string& data_transfer_protocol,
                                           uint64 duration_us);

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

// Records the event of a tf.data service pipeline getting a runtime
// compression decision.
void RecordTFDataServiceRuntimeCompressionDecision(bool compression_decision);

// Records the event of a tf.data service pipeline making the compression
// related action.
void RecordTFDataServiceCompressionAction(const string& action);

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
    const data::ProcessingModeDef& processing_mode, bool is_coordinated_read);

// Records tf.data service iterators created by clients.
void RecordTFDataServiceClientIterators(
    int64_t worker_uid, data::DeploymentMode deployment_mode,
    const data::ProcessingModeDef& processing_mode, bool is_coordinated_read);

// Records that a tf.data service worker client has been created that will use
// `data_transfer_protocol` to get data from the worker server and whether or
// not the user explicitly specified the protocol.
void RecordTFDataServiceDataTransferProtocolUsed(
    const string& data_transfer_protocol, bool user_specified);

// Records that a tf.data service worker client fell back to gRPC rather than
// use `data_transfer_protocol` because of an error of type `code` with message
// `error_message`.
void RecordTFDataServiceDataTransferProtocolFallback(
    const string& data_transfer_protocol, error::Code code,
    const string& error_message);

// Records that a tf.data service worker client got an error of non-retriable
// type `code` with message `error_message` when trying to transfer data over
// `data_transfer_protocol`.
void RecordTFDataServiceDataTransferProtocolError(
    const string& data_transfer_protocol, error::Code code,
    const string& error_message);

// Records tf.data service cross-trainer cache queries.
void RecordTFDataServiceCrossTrainerCacheQuery(bool cache_hit);

// Records tf.data service cross-trainer cache memory usage in bytes.
void RecordTFDataServiceCrossTrainerCacheSizeBytes(size_t bytes);

// Records tf.data distributed snapshot bytes committed.
void RecordTFDataServiceSnapshotBytesCommitted(int64_t bytes);

// Records tf.data distributed snapshot save/load ops.
void RecordTFDataServiceSnapshotOp(const std::string& path,
                                   const std::string& op);

// Records the current estimated optimal number of tf.data service workers.
void RecordTFDataServiceOptimalNumberOfWorkers(int64_t number_of_workers);

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

// Records the number of times this event occured, for debugging.
void RecordTFDataDebug(const string& event);

// Records the number of times an error of this type occurred with this status
// code.
void RecordTFDataError(const string& error_type, const string& error_code);

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

// Records the pipeline processing time in microseconds
void RecordPipelineProcessingTime(const string& id,
                                  double pipeline_processing_time_usec);

// Increments the count of binaries loaded from the persistent cache.
void UpdatePersistentCacheLoadCount();

// Increments the count of BEF and MLIR deserialized.
void UpdateAotBefMlirLoadCount();

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

// Updates the metric stored for time spent optimizing function graphs.
void UpdateFunctionGraphOptimizationTime(const uint64 running_time_usecs);

// Updates the metric stored for time saved by caching graph optimization.
void UpdateFunctionGraphOptimizationSavingTime(uint64 saving_time_usec,
                                               GraphOptimizationSource source);

// Retrieves the total time saved by the graph optimization caching.
uint64 GetFunctionGraphOptimizationSavingTimeUsecs(
    GraphOptimizationSource source);

// Increments the hit count for the graph optimization cache.
void IncrementFunctionGraphOptimizationCacheHitCount(
    int count, GraphOptimizationSource source);

// Gets the hit count for the graph optimization cache.
int64_t GetFunctionGraphOptimizationCacheHitCount(
    GraphOptimizationSource source);

// Increments the failure count for the graph optimization cache restoring.
void IncrementFunctionGraphOptimizationCacheFailureCount(
    int count, GraphOptimizationSource source);

// Gets the failure count for the graph optimization cache.
int64_t GetFunctionGraphOptimizationCacheFailureCount(
    GraphOptimizationSource source);

// Increments the miss count for the graph optimization cache.
void IncrementFunctionGraphOptimizationCacheMissCount(
    int count, GraphOptimizationSource source);

// Gets the miss count for the graph optimization cache.
int64_t GetFunctionGraphOptimizationCacheMissCount(
    GraphOptimizationSource source);

// Increments the number of restoring function graph optimization cache.
void IncrementFunctionGraphOptimizationCacheLoadCount(
    int count, GraphOptimizationSource source);

int64_t GetFunctionGraphOptimizationCacheLoadCount(
    GraphOptimizationSource source);

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

enum class MlirBridgeSecondPhaseMetric {
  // MLIR bridge phase 2 was executed and the graph was processed successfully
  // (fallback enabled).
  kMlirWithFallbackModeSuccess,
  // MLIR bridge phase 2 compilation was failure (fallback enabled).
  kMlirWithFallbackModeFailure,
  // MLIR bridge phase 2 compilation was successful (manually enabled).
  kMlirModeSuccess,
  // MLIR bridge phase 2 compilation fails (manually enabled)
  kMlirModeFailure,
  // Old bridge compilation was run successfully (was run because MLIR bridge
  // could not process the graph).
  kOldBridgeMlirFilteredSuccess,
  // Old bridge failed (was run b/c MLIR bridge could not process the graph).
  kOldBridgeMlirFilteredFailure,
  // Old bridge compilation was successfully run after MLIR bridge ran and
  // failed.
  kOldBridgeWithFallbackModeSuccess,
  // Old Bridge failed in fallback (was run because MLIR bridge failed first).
  kOldBridgeWithFallbackModeFailure,
  // MLIR bridge phase 2 Combined Bridge MLIR was successful
  kMlirCombinedMlirSuccess,
  // MLIR bridge phase 2 Combined Bridge MLIR failed
  kMlirCombinedMlirFailure,
  // MLIR bridge phase 2 Combined Bridge Old bridge was successful
  kMlirCombinedOldSuccess,
  // MLIR bridge phase 2 Combined Bridge Old bridge was successful
  kMlirCombinedOldFailure,
};

// Records the activity of the second phase of the mlir bridge.
void IncrementTfMlirBridgeSecondPhaseCounter(
    MlirBridgeSecondPhaseMetric metric);

// Records the activity per op using the
// tf_metadata.tf_mlir_bridge_graph_analysis_per_op.
// op_name: the name of op.
// construction_context: eager, session, Not tracked.
// is_single_core_inference_mode: true, false.
// unsupported_reason: the reason why the graph is not supported in MLIR-based
// bridge, like invalid graph, has unsupported ops, etc.
// has_unsupported_features: true indicates MLIR-based bridge is disabled,
// false indicates MLIR-based bridge is enabled.

void UpdateTfMlirBridgeGraphAnalysisPerOp(
    const std::string& op_name, const std::string& construction_context,
    bool is_single_core_inference_mode, const std::string& num_replicas,
    const std::string& num_cores_per_replica, const std::string& use_tpu,
    const std::string& allow_soft_placement,
    const std::string& use_spmd_for_xla_partitioning,
    const std::string& unsupported_reason, bool has_unsupported_features);

// Records whether a graph contains any of the TF1 features
void RecordTFVersionByGraphFeatures(const std::string& device,
                                    const std::string& context,
                                    bool hasControlFlowV1,
                                    bool hasReferenceVariables,
                                    bool hasManualControlDeps);

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
  std::optional<uint64> DurationMicroSec() const {
    return started_ ? std::optional<uint64>(accumulated_time_ +
                                            Env::Default()->NowMicros() -
                                            start_time_)
                    : std::nullopt;
  }

  // Temporarily stop the timer, but keep accumulated time.
  void AccumulateAndStop() {
    if (started_) {
      accumulated_time_ = Env::Default()->NowMicros() - start_time_;
      started_ = false;
    }
  }

  // Start previously stopped timer.
  void Start() {
    if (started_) return;

    // Keep previously accumulated time if any.
    start_time_ = Env::Default()->NowMicros();
    started_ = true;
  }

  ~ScopedCounter() { ReportAndStop(); }

 private:
  template <std::size_t... S>
  void ReportInternal(std::index_sequence<S...>) {
    uint64 time_interval = Env::Default()->NowMicros() - start_time_;
    time_interval += accumulated_time_;
    if (time_interval > 0) {
      counter_->GetCell(labels_[S]...)->IncrementBy(time_interval);
    }
  }

  void Init() {
    start_time_ = Env::Default()->NowMicros();
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
void UpdateEagerClientErrorCounter(const string& error_source,
                                   const string& error_type);

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_METRICS_H_
