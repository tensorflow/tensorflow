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

#ifndef TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_
#define TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace graph_transforms {

// Used to quickly look up nodes in the graph def from a name.
void MapNamesToNodes(const GraphDef& graph_def,
                     std::map<string, const NodeDef*>* result);

// NodeDef input strings can contain other information besides the name of an
// input node. These include:
//  - Optional '^' prefix, indicating this is a control edge.
//  - The required name of the input node.
//  - Option ':<number>' suffix, showing which output of the node to use.
// This function takes a raw string, and breaks it into those component parts.
void NodeNamePartsFromInput(string input_name, string* prefix,
                            string* node_name, string* suffix);

// Convenience function to strip the optional prefix and suffix components from
// a string pulled from a NodeDef input, and return the plain node name.
string NodeNameFromInput(string input_name);

// Creates a copy of the input GraphDef, but only containing the nodes where the
// supplied selector function returned true.
void FilterGraphDef(const GraphDef& input_graph_def,
                    std::function<bool(const NodeDef&)> selector,
                    GraphDef* output_graph_def);

// Creates a copy of the input graph, with all occurences of the attributes with
// the names in the argument removed from the node defs.
void RemoveAttributes(const GraphDef& input_graph_def,
                      const std::vector<string>& attributes,
                      GraphDef* output_graph_def);

}  // namespace graph_transforms
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_GRAPH_TRANSFORMS_TRANSFORM_UTILS_H_
