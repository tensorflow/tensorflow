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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METRICS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METRICS_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace metrics {

// Records that a tf.data dataset op executed by the program used autotuning.
//
// The `name` argument identifies the dataset (e.g. "ParallelMap").
void RecordTFDataAutotune(const string& name);

// Records the number of elements produced by a tf.data dataset.
//
// The `name` argument identifies the dataset (e.g. "Batch" or "Map").
void RecordTFDataElements(const string& name, int64 num_elements);

// Records the number of independent graph changes resulting from the applicaton
// of a tf.data optimization.
//
// The `name` argument identifies the optimization (e.g. "noop_eliminiation").
void RecordTFDataOptimization(const string& name, int64 num_changes);

void UpdateGraphExecTime(const uint64 running_time_usecs);

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

}  // namespace metrics
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METRICS_H_
