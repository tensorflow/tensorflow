/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_VALIDATE_CONTROL_FLOW_H_
#define TENSORFLOW_COMPILER_TF2XLA_VALIDATE_CONTROL_FLOW_H_

#include <vector>

#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Populate the control flow frame info of each node in the graph. Verify that
// the graph has well-formed control flow strcuture that can be functionalized.
// If unreachable_nodes is not nullptr, append to it the names of nodes
// unreachable from the source node.
Status BuildAndValidateControlFlowInfo(
    const Graph* graph, std::vector<ControlFlowInfo>* info,
    std::vector<string>* unreachable_nodes = nullptr);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_VALIDATE_CONTROL_FLOW_H_
