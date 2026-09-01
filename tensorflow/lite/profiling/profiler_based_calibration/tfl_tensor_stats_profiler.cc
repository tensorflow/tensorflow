/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/profiling/profiler_based_calibration/tfl_tensor_stats_profiler.h"

#include <cstdint>
#include <cstring>
#include <utility>

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

namespace odml {

TensorStatsProfiler::TensorStatsProfiler(const tflite::Interpreter& interpreter,
                                         Callback callback)
    : interpreter_(interpreter), callback_(callback) {}

uint32_t TensorStatsProfiler::BeginEvent(const char* tag, EventType event_type,
                                         int64_t event_metadata1,
                                         int64_t event_metadata2) {
  // Process subgraph inputs at the beginning of each subgraph invoke.
  if (tag && strcmp(tag, "Invoke") == 0) {
    const int64_t subgraph_idx = event_metadata2;
    const tflite::Subgraph* subgraph = interpreter_.subgraph(subgraph_idx);
    if (subgraph) {
      for (const int input_tensor_index : subgraph->inputs()) {
        if (input_tensor_index != kTfLiteOptionalTensor) {
          callback_(subgraph->tensor(input_tensor_index));
        }
      }
    } else {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "TensorStatsProfiler: subgraph %d not found.",
                      subgraph_idx);
    }
    return 0;  // No event handle for input tensors.
  }

  // Only capture operator invoke events for intermediate/output activations.
  if (event_type != EventType::OPERATOR_INVOKE_EVENT) {
    return 0;
  }

  const int64_t node_index = event_metadata1;
  const int64_t subgraph_index = event_metadata2;

  // Store the subgraph and node index for later retrieval.
  events_.push_back({
      .subgraph_index = subgraph_index,
      .node_index = node_index,
  });
  return events_.size();
}

void TensorStatsProfiler::EndEvent(uint32_t event_handle) {
  if (!event_handle || events_.size() < event_handle) {
    return;
  }

  // Retrieve the node from the event handle.
  const auto& event = events_[event_handle - 1];
  const tflite::Subgraph* subgraph =
      interpreter_.subgraph(event.subgraph_index);
  if (!subgraph) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "TensorStatsProfiler: subgraph %d not found.",
                    event.subgraph_index);
    return;
  }
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
      subgraph->node_and_registration(event.node_index);
  if (!node_and_reg) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                    "TensorStatsProfiler: node %d in subgraph %d not found.",
                    event.node_index, event.subgraph_index);
    return;
  }
  const TfLiteNode& node = node_and_reg->first;

  // Invoke the callback for each output tensor of the operator.
  for (int i = 0; i < node.outputs->size; ++i) {
    const int tensor_index = node.outputs->data[i];
    if (tensor_index != kTfLiteOptionalTensor) {
      callback_(subgraph->tensor(tensor_index));
    }
  }
}

}  // namespace odml
