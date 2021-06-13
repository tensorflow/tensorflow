/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METRIC_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METRIC_UTIL_H_

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Determines whether to branch and execute metrics logging.
// Currently requires an exact match on the node type and a substring match on
// the node name. This function is meant to be low latency, as it is called from
// within the executor loop.
bool ShouldLogLatencyMetrics(const NodeDef& ndef);

// Uses the supplied NodeDef to log the latency metric of interest.
void LogLatencyMetrics(const NodeDef& ndef, const int64 cur_time_usecs,
                       const int64 start_time_usecs);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METRIC_UTIL_H_
