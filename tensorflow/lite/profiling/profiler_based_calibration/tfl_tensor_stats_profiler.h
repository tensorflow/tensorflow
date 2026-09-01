/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_TENSOR_STATS_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_TENSOR_STATS_PROFILER_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"

namespace odml {

// The TensorStatsProfiler collects stats for each non-constant tensor in a
// subgraph during interpreter invocation.
class TensorStatsProfiler : public tflite::Profiler {
 public:
  using Callback = std::function<void(const TfLiteTensor*)>;

  TensorStatsProfiler(const tflite::Interpreter& interpreter,
                      Callback callback);

  // This profiler may be invoked at multiple points throughout the
  // execution of a subgraph. At the beginning of each subgraph invoke,
  // capture the input tensor stats with the provided callback. At the beginning
  // of each operator invoke, stores the subgraph and node index for later
  // retrieval.
  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  // At the end of an operator invoke event, calculates the tensor stats for the
  // operator's output tensors with the provided callback.
  void EndEvent(uint32_t event_handle) override;

 private:
  struct EventMetadata {
    int64_t subgraph_index;
    int64_t node_index;
  };
  // A mapping between event IDs and (subgraph_index, node_index).
  std::vector<EventMetadata> events_;

  // A handle to the active TFLite interpreter.
  const tflite::Interpreter& interpreter_;

  // A user provided callback to calculate tensor stats for a given tensor. The
  // callback signature is:
  //  void Callback(const TfLiteTensor* tensor);
  Callback callback_;
};

}  // namespace odml

#endif  // TENSORFLOW_LITE_PROFILING_PROFILER_BASED_CALIBRATION_TFL_TENSOR_STATS_PROFILER_H_
