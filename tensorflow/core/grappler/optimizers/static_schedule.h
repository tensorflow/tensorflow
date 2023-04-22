/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_STATIC_SCHEDULE_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_STATIC_SCHEDULE_H_

#include <unordered_map>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

// Compute the earliest time at which the execution of each node in the graph
// can complete.
// In our estimation, we ensure that each node takes at least one nanosecond to
// execute: therefore the execution times can be used to derive a topological
// ordering of the graph (at least as long as there is no loop in the graph).
Status EstimateEarliestExecutionTimes(
    const GrapplerItem& item, const Cluster* cluster,
    std::unordered_map<const NodeDef*, Costs::NanoSeconds>* execution_times);

// Compute the time by which the execution of each node must complete to ensure
// the subsequent nodes can still be executed by the times predicted by the
// EstimateEarliestExecutionTimes function.
Status EstimateRequiredTimes(
    const GrapplerItem& item, const Cluster* cluster,
    const std::unordered_map<const NodeDef*, Costs::NanoSeconds>&
        execution_times,
    std::unordered_map<const NodeDef*, Costs::NanoSeconds>* required_times);

}  // namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_STATIC_SCHEDULE_H_
