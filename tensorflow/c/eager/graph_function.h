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
#ifndef TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_GRAPH_H_
#define TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_GRAPH_H_

#include "tensorflow/c/eager/abstract_function.h"
namespace tensorflow {
namespace tracing {
namespace graph {
using tensorflow::AbstractFunction;
// Thin wrapper around a FunctionDef.
class GraphFunction : public AbstractFunction {
 public:
  explicit GraphFunction(FunctionDef fdef);
  ~GraphFunction() override;

  // GraphFunction maybe stay alive for the duration of the returned
  // FunctionDef.
  Status GetFunctionDef(FunctionDef** fdef) override;

  // For LLVM style RTTI.
  static bool classof(const AbstractFunction* ptr) {
    return ptr->getKind() == kGraph;
  }

 private:
  FunctionDef fdef_;
};
}  // namespace graph
}  // namespace tracing
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_GRAPH_H_
