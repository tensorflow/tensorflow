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

#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/types.h"

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

// Records the number of bytes fetched from tf.data.Dataset iterator.
void RecordTFDataBytesFetched(int64 num_bytes);

// Records the number of times tf.data experiment is applied to input pipelines.
void RecordTFDataExperiment(const string& name);

// Records the time spent in ItertatorResource::GetNext() in microseconds.
void RecordTFDataGetNextDuration(uint64 duration_us);

// Records the time spent between IteratorResource::GetNext() calls
// in microseconds. Time is measured from the point of returning data from
// GetNext() to the point of new data being requested.
// This elapsed time corresponds to time spent outside the GetNext() function.
void RecordTFDataGetNextTimeBetween(uint64 duration_us);

// Records the number of times each tf.data fingerprint is used
// to measure duplicate pre-processing.
//
// The `name` argument identifies the Dataset graph fingerprint,
// created using GraphHash().
void RecordTFDataFingerprint(const string& name);

// Records the number of independent graph changes resulting from the
// application of a tf.data optimization.
//
// The `name` argument identifies the optimization (e.g. "noop_elimination").
void RecordTFDataOptimization(const string& name, int64 num_changes);

// Records parsing of dense tensor features.
void RecordParseDenseFeature(int64 num_features);

// Records parsing of sparse tensor features.
void RecordParseSparseFeature(int64 num_features);

// Records parsing of ragged tensor features.
void RecordParseRaggedFeature(int64 num_features);

// Records the size of input/output tensors in bytes.
void RecordGraphInputTensors(const size_t size);
void RecordGraphOutputTensors(const size_t size);

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

// Updates the metrics stored about graph optimizations.
void UpdateGraphOptimizationPassTime(const string& pass_name,
                                     const uint64 running_time_usecs);
void UpdateGrapplerPassTime(const string& pass_name,
                            const uint64 running_time_usecs);

// Updates the metrics stored about time XLA spents compiling graphs.
void UpdateXlaCompilationTime(const uint64 compilation_time_usecs);

// Updates the metrics stored about time BFC allocator spents during delay.
void UpdateBfcAllocatorDelayTime(const uint64 delay_usecs);

// Increment the number of jobs that failed during import to mlir.
void IncrementMLIRImportFailureCount();

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_METRICS_H_
