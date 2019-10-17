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

#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"

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

auto* graph_run_input_tensor_bytes = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_input_tensor_bytes",
     "The size of input tensors in bytes."},
    // Power of 2 with bucket count 14 (256G)
    {monitoring::Buckets::Exponential(1, 4, 20)});

auto* graph_run_output_tensor_bytes = monitoring::Sampler<0>::New(
    {"/tensorflow/core/graph_run_output_tensor_bytes",
     "The size of output tensors in bytes."},
    // Power of 2 with bucket count 14 (256G)
    {monitoring::Buckets::Exponential(1, 4, 14)});

auto* tf_data_autotune_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/autotune", "tf.data autotuning", "name");

auto* tf_data_bytes_read_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/bytes_read",
    "The number of bytes read by tf.data Dataset sources.", "name");

auto* tf_data_elements_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/elements", "tf.data elements", "name");

auto* tf_data_fingerprint_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/fingerprint", "tf.data fingerprint", "name");

auto* tf_data_optimization_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/optimization", "tf.data optimization", "name");

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

auto* mlir_import_failure_count = monitoring::Counter<0>::New(
    "/tensorflow/mlir/import_failure_count",
    "The number of jobs that failed during mlir import or verification.");

}  // namespace

void RecordTFDataAutotune(const string& name) {
  tf_data_autotune_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataBytesRead(const string& name, int64 num_bytes) {
  tf_data_bytes_read_counter->GetCell(name)->IncrementBy(num_bytes);
}

void RecordTFDataElements(const string& name, int64 num_elements) {
  tf_data_elements_counter->GetCell(name)->IncrementBy(num_elements);
}

void RecordTFDataFingerprint(const string& name) {
  tf_data_fingerprint_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataOptimization(const string& name, int64 num_changes) {
  tf_data_optimization_counter->GetCell(name)->IncrementBy(num_changes);
}

void RecordParseDenseFeature(int64 num_features) {
  parse_dense_feature_counter->GetCell()->IncrementBy(num_features);
}

void RecordParseSparseFeature(int64 num_features) {
  parse_sparse_feature_counter->GetCell()->IncrementBy(num_features);
}

void RecordParseRaggedFeature(int64 num_features) {
  parse_ragged_feature_counter->GetCell()->IncrementBy(num_features);
}

void RecordGraphInputTensors(const size_t size) {
  graph_run_input_tensor_bytes->GetCell()->Add(size);
}

void RecordGraphOutputTensors(const size_t size) {
  graph_run_output_tensor_bytes->GetCell()->Add(size);
}

void UpdateGraphExecTime(const uint64 running_time_usecs) {
  if (running_time_usecs > 0) {
    graph_runs->GetCell()->IncrementBy(1);
    graph_run_time_usecs->GetCell()->IncrementBy(running_time_usecs);
    graph_run_time_usecs_histogram->GetCell()->Add(running_time_usecs);
  }
}

void UpdateGraphBuildTime(const uint64 running_time_usecs) {
  if (running_time_usecs > 0) {
    build_graph_calls->GetCell()->IncrementBy(1);
    build_graph_time_usecs->GetCell()->IncrementBy(running_time_usecs);
  }
}

void UpdateXlaCompilationTime(const uint64 compilation_time_usecs) {
  if (compilation_time_usecs > 0) {
    xla_compilations->GetCell()->IncrementBy(1);
    xla_compilation_time_usecs->GetCell()->IncrementBy(compilation_time_usecs);
  }
}

void IncrementMLIRImportFailureCount() {
  mlir_import_failure_count->GetCell()->IncrementBy(1);
}

}  // namespace metrics
}  // namespace tensorflow
