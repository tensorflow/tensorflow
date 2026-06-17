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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_GRAPH_DEF_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_GRAPH_DEF_H_

#include <type_traits>

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace stablehlo::quantization {

// Mutates all `NodeDef`s in `graph_def` by applying `func`. It modifies the
// top-level `NodeDef`s as well as all `NodeDef`s in the function library.
// `func` should accept a `NodeDef` reference.
template <typename FuncT, typename = std::enable_if_t<std::is_invocable_r_v<
                              void, FuncT, tensorflow::NodeDef&>>>
void MutateNodeDefs(tensorflow::GraphDef& graph_def, FuncT&& func) {
  for (tensorflow::NodeDef& node_def : *graph_def.mutable_node()) {
    func(node_def);
  }

  for (tensorflow::FunctionDef& function_def :
       *graph_def.mutable_library()->mutable_function()) {
    for (tensorflow::NodeDef& node_def : *function_def.mutable_node_def()) {
      func(node_def);
    }
  }
}

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_GRAPH_DEF_H_
