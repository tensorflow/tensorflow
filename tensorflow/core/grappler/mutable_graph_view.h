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

#ifndef TENSORFLOW_CORE_GRAPPLER_MUTABLE_GRAPH_VIEW_H_
#define TENSORFLOW_CORE_GRAPPLER_MUTABLE_GRAPH_VIEW_H_

#include "tensorflow/core/grappler/graph_view.h"

namespace tensorflow {
namespace grappler {

// A utility class to simplify the traversal of a GraphDef that, unlike
// GraphView, supports updating the graph.  Note that you should not modify the
// graph separately, because the view will get out of sync.
class MutableGraphView : public GraphView {
 public:
  using GraphView::GraphView;

  GraphDef* GetGraph() { return MutableGraph(); }

  // Adds a new node to graph and updates the view.
  NodeDef* AddNode(NodeDef&& node);

  // Inserts a new node to the graph after `input` node and updates the view.
  // This adds `node` to the graph and replaces the input for the output
  // nodes of `input` with a port `output_port_id` with the new node.
  NodeDef* InsertNode(const NodeDef& input, NodeDef&& node,
                      int output_port_id = 0);

  // Replaces the input for the output nodes of 'old_input' with a port
  // `output_port_id` with 'new_input'.
  //
  // E.g: We have 2 nodes that use 'bar' node outputs as inputs:
  // foo(bar:0, bar:1),  foo2(other:0, bar:0)
  // Calling ReplaceInput(bar, new, 0) changes every occurrence of bar:0 for
  // new:0.  Result:
  // foo(new:0, bar:1),  foo2(other:0, new:0)
  void ReplaceInput(const NodeDef& old_input, const NodeDef& new_input,
                    int output_port_id = 0);

  // Deletes nodes from the graph.
  void DeleteNodes(const std::set<string>& nodes_to_delete);

 private:
  void RemoveFanouts(NodeDef* node);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_MUTABLE_GRAPH_VIEW_H_
