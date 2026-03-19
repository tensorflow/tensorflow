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

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_REWRITE_PASS_UTILS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_REWRITE_PASS_UTILS_H_

#include "absl/status/status.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Adds a new TensorFlow graph node, with the output convention matching most TF
// code rather than the order used by Graph::AddNode().
absl::Status AddNode(const NodeDef& n_def, Node** n, Graph* graph);

// Replaces one TensorFlow graph node with another (specified by a NodeDef),
// moving all the edges.
absl::Status ReplaceNode(const NodeDef& to_def, Node* from, Node** to,
                         Graph* graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_TPU_EMBEDDING_REWRITE_PASS_UTILS_H_
