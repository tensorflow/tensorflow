/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONTROL_EDGES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONTROL_EDGES_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace tflite {

// LINT.IfChange

using ControlEdge = std::pair<int32_t, int32_t>;
using ControlEdges = std::vector<ControlEdge>;

// LINT.ThenChange(//tensorflow/lite/graph_info.h)

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_CONTROL_EDGES_H_
