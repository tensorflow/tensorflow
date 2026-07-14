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
#ifndef TENSORFLOW_LITE_PROFILING_SUBGRAPH_TENSOR_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_SUBGRAPH_TENSOR_PROFILER_H_

#include <functional>
#include <vector>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite::profiling {

// The SubgraphTensorProfiler is invoked for every tensor in a subgraph at the
// end of the subgraph's execution. This profiler is constructed with a user
// provided callback to run on each tensor in the subgraph.
class SubgraphTensorProfiler : public tflite::Profiler {
 public:
  using CallbackT = std::function<void(const TfLiteTensor*)>;

  SubgraphTensorProfiler(const Interpreter& interpreter, CallbackT callback);

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  void EndEvent(uint32_t event_handle) override;

 private:
  // A mapping between event IDs and the subgraph that owns the event ID.
  std::vector<int64_t> events_;

  // A handle to the active TFLite interpreter.
  const Interpreter& interpreter_;

  // A user provided callback to run on each tensor in the subgraph. The
  // callback signature is:
  //
  //  void Callback(const TfLiteTensor* tensor);
  CallbackT callback_;
};

}  // namespace tflite::profiling

#endif  // TENSORFLOW_LITE_PROFILING_SUBGRAPH_TENSOR_PROFILER_H_
