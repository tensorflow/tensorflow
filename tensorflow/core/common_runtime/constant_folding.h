/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_COMMON_RUNTIME_CONSTANT_FOLDING_H_
#define TENSORFLOW_COMMON_RUNTIME_CONSTANT_FOLDING_H_

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

namespace tensorflow {

// Perform constant folding optimization on "graph".
// Looks for nodes in "graph" that can be completely evaluated statically, i.e.,
// that are only dependent on constants. Evaluates those nodes on a CPU device
// and replaces those nodes with the result of the evaluation.
// Returns true if and only if "graph" has been mutated.
bool DoConstantFolding(const ConstantFoldingOptions& opts, Graph* graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_CONSTANT_FOLDING_H_
