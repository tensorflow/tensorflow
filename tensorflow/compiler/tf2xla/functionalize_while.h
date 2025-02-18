/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_WHILE_H_
#define TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_WHILE_H_

#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Transformation that converts tf.while_loop() loops into functional While
// operators, suitable for XLA compilation. If lookup_library is provided, use
// it to make the library for control flow self-contained.
//
// If `node_filter` is defined, then only loops for whose nodes `node_filter`
// returns true are functionalized.
//
// Preconditions:
// Same as for `FunctionalizeControlFlow` (see comment there).
absl::Status FunctionalizeWhileLoop(Graph* graph,
                                    FunctionLibraryDefinition* library,
                                    const NodeFilter& node_filter = {});

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_WHILE_H_
