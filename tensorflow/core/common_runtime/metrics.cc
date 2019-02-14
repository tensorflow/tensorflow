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

auto* tf_data_autotune_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/autotune", "tf.data autotuning", "name");

auto* tf_data_elements_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/elements", "tf.data elements", "name");

auto* tf_data_optimization_counter = monitoring::Counter<1>::New(
    "/tensorflow/data/optimization", "tf.data optimization", "name");

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

}  // namespace

void RecordTFDataAutotune(const string& name) {
  tf_data_autotune_counter->GetCell(name)->IncrementBy(1);
}

void RecordTFDataElements(const string& name, int64 num_elements) {
  tf_data_elements_counter->GetCell(name)->IncrementBy(num_elements);
}

void RecordTFDataOptimization(const string& name, int64 num_changes) {
  tf_data_optimization_counter->GetCell(name)->IncrementBy(num_changes);
}

void UpdateGraphExecTime(const uint64 running_time_usecs) {
  if (running_time_usecs > 0) {
    graph_runs->GetCell()->IncrementBy(1);
    graph_run_time_usecs->GetCell()->IncrementBy(running_time_usecs);
  }
}

void UpdateGraphBuildTime(const uint64 running_time_usecs) {
  if (running_time_usecs > 0) {
    build_graph_calls->GetCell()->IncrementBy(1);
    build_graph_time_usecs->GetCell()->IncrementBy(running_time_usecs);
  }
}

}  // namespace metrics
}  // namespace tensorflow
