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

namespace {

auto* graph_runs = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_runs",
    "The number of graph executions used to collect "
    "/tensorflow/core/graph_run_time_msecs");

auto* graph_run_time_msecs = monitoring::Counter<0>::New(
    "/tensorflow/core/graph_run_time_msecs",
    "The total time spent on executing graphs in milliseconds.");
}  // namespace

void UpdateGraphExecutionTime(const absl::Duration running_time) {
  if (running_time > absl::ZeroDuration()) {
    graph_runs->GetCell()->IncrementBy(1);
    graph_run_time_msecs->GetCell()->IncrementBy(running_time /
                                                 absl::Milliseconds(1));
  }
}

}  // namespace tensorflow
