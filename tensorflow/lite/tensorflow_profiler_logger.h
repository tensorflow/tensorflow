/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TENSORFLOW_PROFILER_LOGGER_H_
#define TENSORFLOW_LITE_TENSORFLOW_PROFILER_LOGGER_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "tensorflow/lite/core/macros.h"

struct TfLiteTensor;

namespace tsl {
namespace profiler {
class TraceMe;
}  // namespace profiler
}  // namespace tsl

namespace tensorflow {
namespace profiler {

using tsl::profiler::TraceMe;

}  // namespace profiler
}  // namespace tensorflow

namespace tflite {

// Records an op preparation with `op_name` and `node_index`.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteOpPrepare(const char* op_name,
                                             int subgraph_index,
                                             int node_index);

// Returns a `TraceMe` pointer to record a subgraph invocation with
// `subgraph_name` and `subgraph_index`.
TFLITE_ATTRIBUTE_WEAK tensorflow::profiler::TraceMe* OnTfLiteSubgraphInvoke(
    const char* subgraph_name, int subgraph_index);

// Records an end of the subgraph invocation with the given `TraceMe` pointer.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteSubgraphInvokeEnd(
    tensorflow::profiler::TraceMe* trace_me);

// Returns a `TraceMe` pointer to record an op invocation with `op_name` and
// `node_index`.
TFLITE_ATTRIBUTE_WEAK tensorflow::profiler::TraceMe* OnTfLiteOpInvoke(
    const char* op_name, int subgraph_index, int node_index);

// Records an end of the op invocation with the given `TraceMe` pointer.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteOpInvokeEnd(
    tensorflow::profiler::TraceMe* trace_me);

// Records an event of `num_bytes` of memory allocated for `tensor`.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteTensorAlloc(TfLiteTensor* tensor,
                                               size_t num_bytes);

// Records an event of memory deallocated for `tensor`.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteTensorDealloc(TfLiteTensor* tensor);

// Records an event of `num_bytes` of memory allocated for arena.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteArenaAlloc(int subgraph_index, int arena_id,
                                              size_t num_bytes);

// Records an event of `num_bytes` of memory deallocated for arena.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteArenaDealloc(int subgraph_index,
                                                int arena_id, size_t num_bytes);

// Pause / resume heap monitoring via malloc/free hooks.
TFLITE_ATTRIBUTE_WEAK void PauseHeapMonitoring(bool pause);

// Records end of Interpreter so logger can report saved heap allocations.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteInterpreterEnd();

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TENSORFLOW_PROFILER_LOGGER_H_
