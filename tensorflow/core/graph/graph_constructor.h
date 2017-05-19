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

#ifndef TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_
#define TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class ShapeRefiner;

// Construct a Graph *g out of a GraphDef gdef. Returns non-OK on
// error, in which case *g is left in an incomplete state.
//
// *g is expected to be an empty graph (with no more than a source and sink
// nodes) when provided to ConvertGraphDefToGraph. To enhance an existing Graph,
// see ImportGraphDef.
struct GraphConstructorOptions {
  GraphConstructorOptions() {}

  // If true, allows internal ops in the GraphDef.
  bool allow_internal_ops = false;

  // If true, the graph def is expected to have fully specified
  // devices for all nodes. A node in the resulting graph "g" has the
  // device name set accordingly.
  //
  // TODO(zhifengc): if possible, consider removing this option.
  bool expect_device_spec = false;
};
extern Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                                     const GraphDef& gdef, Graph* g);

// Add the graph in GraphDef gdef into an existing Graph *g.
//
// On error, returns non-OK and leaves *g unmodified.
//
// "shape_refiner" can be null. It should be non-null if the caller
// intends to add additonal nodes to the graph after the import. This
// allows the caller to validate shapes of those nodes (since
// ShapeRefiner::AddNode must be called in topological order).
//
// TODO(ashankar): Push this mechanism and get rid of Session::Extend()
// as a means of enhancing an existing Graph.
struct ImportGraphDefOptions {
  ImportGraphDefOptions() {}

  // Name prefix to use for nodes imported from the GraphDef.  For example, if
  // prefix="animals" and GraphDef contains a node "bunny" then the node will be
  // named "animals/bunny" in *g.
  string prefix;

  // Maps tensors in `gdef` to existing tensors in `g`. Inputs in `gdef`
  // corresponding to `input_map` keys will be remapped to the nodes in `g`
  // corresponding to the values.
  //
  // Keys should not include `prefix`, i.e., a key TensorId's name should be the
  // name as it originally appears in `gdef`.
  //
  // If this is non-empty, ImportGraphDef must be called with the shape refiner
  // used to create the existing nodes referenced in `input_map`.
  // TODO(skyewm): can we remove this requirement? How do we access the original
  // shape refiner?
  //
  // TODO(skyewm): add functionality to retrieve unused `input_map` keys
  std::map<TensorId, TensorId> input_map;

  // The names of existing nodes in `g` that the imported graph should have
  // control dependencies on.
  //
  // Note that to avoid creating many redundant control edges, ImportGraphDef()
  // won't add control edges to nodes that will inherit the dependencies from
  // other nodes in `gdef`.
  std::vector<string> control_dependencies;

  // Tensors in `gdef` that will be returned via the `return_tensors` output
  // parameter of `ImportGraphDef()`. If this list is non-empty, the caller must
  // pass an empty vector to `ImportGraphDef()`. The vector will be populated
  // with the imported nodes in `g`.
  //
  // Entries should not include `prefix`, i.e., each TensorId's name should be
  // the name as it originally appears in `gdef`.
  //
  // If this contains a tensor that's also being remapped via `input_map`, the
  // corresponding existing tensor in `g` will be returned.
  std::vector<TensorId> return_tensors;

  // TODO(ashankar): Enable handling of GraphDefs produced by newer binaries
  // with ops that are not defined in the binary calling ImportGraphDef.
  // Similar to the producer_op_list argument to import_graph_def in the
  // python API.
};

// Each `return_tensors` entry is the requested node and output index. The index
// is included in case the returned tensor has been remapped according to
// `input_map`.
extern Status ImportGraphDef(
    const ImportGraphDefOptions& opts, const GraphDef& gdef, Graph* g,
    ShapeRefiner* refiner,
    std::vector<std::pair<Node*, int>>* return_tensors = nullptr);

// Make a copy of "src" into "*dest".
//
// REQUIRES: "*dest" is a freshly allocated graph without any nodes or edges
// other than the implicit Source/Sink nodes.
extern void CopyGraph(const Graph& src, Graph* dest);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_
