/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_FOLD_CONSTANTS_LIB_H_
#define TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_FOLD_CONSTANTS_LIB_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Finds sub-graphs that evaluate to constant expressions, and replaces them
// with Const nodes, to simplify the graph. The inputs and outputs arguments are
// the names of all the nodes that data is fed into, or read out of, when the
// graph is actually run.
Status FoldConstants(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def);

// Analyzes which nodes are used for the given set of inputs and outputs, and
// returns a copy of the graph with any that aren't used removed.
Status RemoveUnusedNodes(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def);

}  // namespace graph_transforms
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_FOLD_CONSTANTS_LIB_H_
