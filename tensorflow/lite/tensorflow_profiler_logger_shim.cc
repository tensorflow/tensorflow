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

#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/tensorflow_profiler_logger.h"

// Use weak symbols here (even though they are guarded by macros) to avoid
// build breakage when building a benchmark requires TFLite runs. The main
// benchmark library should have tensor_profiler_logger dependency.
// Strong symbol definitions can be found in tensorflow_profiler_logger.cc.

namespace tflite {

// No-op for the weak symbol. Overridden by a strong symbol in
// tensorflow_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteOpPrepare(const char* op_name,
                                             const int node_index) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// tensorflow_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteOpInvoke(const char* op_name,
                                            const int node_index) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// tensorflow_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteTensorAlloc(TfLiteTensor* tensor,
                                               size_t num_bytes) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// tensorflow_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteTensorDealloc(TfLiteTensor* tensor) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// tensorflow_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteArenaAlloc(int subgraph_index, int arena_id,
                                              size_t num_bytes) {}

// No-op for the weak symbol. Overridden by a strong symbol in
// tensorflow_profiler_logger.cc.
TFLITE_ATTRIBUTE_WEAK void OnTfLiteArenaDealloc(int subgraph_index,
                                                int arena_id,
                                                size_t num_bytes) {}

}  // namespace tflite
