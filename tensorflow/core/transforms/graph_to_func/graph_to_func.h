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

#ifndef TENSORFLOW_CORE_TRANSFORMS_GRAPH_TO_FUNC_H_
#define TENSORFLOW_CORE_TRANSFORMS_GRAPH_TO_FUNC_H_

#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/status.h"

namespace mlir {
namespace tfg {

// Lifts a graph into a function, using the provided array of `feeds` for
// function arguments, `fetches` for function returned values, and
// `control_rets` for returned control values. The Graph op is replaced in-place
// by a GraphFuncOp with a name defined in the dialect.
tensorflow::Status GraphToFunc(GraphOp graph, ArrayRef<Value> feeds,
                               ArrayRef<Value> fetches,
                               ArrayRef<Value> control_rets);

// Lifts a graph into a function, using the provided array of `feeds` for
// function arguments, `fetches` for function returned values, and
// `control_rets` for returned control values. The Graph op is replaced in-place
// by a GraphFuncOp with a name defined in the dialect.
tensorflow::Status GraphToFunc(GraphOp graph, ArrayRef<std::string> feeds_names,
                               ArrayRef<std::string> fetches_names,
                               ArrayRef<std::string> control_rets);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_TRANSFORMS_GRAPH_TO_FUNC_H_
