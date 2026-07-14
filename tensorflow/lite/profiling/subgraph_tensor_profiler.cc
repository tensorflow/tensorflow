/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/profiling/subgraph_tensor_profiler.h"

#include <cstring>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite::profiling {

SubgraphTensorProfiler::SubgraphTensorProfiler(const Interpreter& interpreter,
                                               CallbackT callback)
    : interpreter_(interpreter), callback_(callback) {
  events_.reserve(interpreter.subgraphs_size());
}

// This Subgraph aware profiler may be invoked at multiple points throughout the
// execution of a subgraph. Only handle subgraph Invoke tagged events.
uint32_t SubgraphTensorProfiler::BeginEvent(const char* tag,
                                            EventType event_type,
                                            int64_t event_metadata1,
                                            int64_t event_metadata2) {
  // Only listen to the "Invoke" event triggered by Subgraph::InvokeImpl().
  if (strcmp(tag, "Invoke")) {
    return 0;
  }

  events_.push_back(/*subgraph_index=*/event_metadata2);
  return events_.size();
}

// Store tensors used during the subgraph's Invoke event for later retrieval.
void SubgraphTensorProfiler::EndEvent(uint32_t event_handle) {
  if (!event_handle || events_.size() < event_handle) {
    return;
  }

  const Subgraph* subgraph = interpreter_.subgraph(events_[event_handle - 1]);

  for (int i = 0; i < subgraph->tensors_size(); ++i) {
    callback_(subgraph->tensor(i));
  }
}

}  // namespace tflite::profiling
