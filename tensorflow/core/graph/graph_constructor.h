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

// Same as ConvertGraphDefToGraph, but takes just nodes.  Used by function
// instantiation.
// TODO(irving): This will turn into std::vector<NodeInfoPtr> soon.
extern Status ConvertNodeDefsToGraph(const GraphConstructorOptions& opts,
                                     gtl::ArraySlice<NodeDef> nodes, Graph* g);

// Options for calling ImportGraphDef().
struct ImportGraphDefOptions {
  ImportGraphDefOptions()
      : uniquify_names(false),
        uniquify_prefix(false),
        skip_mapped_nodes(false),
        validate_shape(true) {}

  // Name prefix to use for nodes imported from the GraphDef.  For example, if
  // prefix="animals" and GraphDef contains a node "bunny" then the node will be
  // named "animals/bunny" in *g. Must not be already used as a node name or
  // prefix in the graph.
  string prefix;

  // If true, imported node names will be modified if their name already exists
  // in the graph. If false, conflicting names will be treated as an error. Note
  // that this option has no effect if `prefix` is specified, since `prefix`
  // will guarantee all node names are unique.
  bool uniquify_names;

  // If true, `prefix` will be modified if it already exists as a node name or
  // prefix in the graph. If false, a conflicting prefix will be treated as an
  // error. This option has no effect if `prefix` isn't specified.
  bool uniquify_prefix;

  // Maps tensors in `gdef` to existing tensors in `g`. Inputs in `gdef`
  // corresponding to `input_map` keys will be remapped to the nodes in `g`
  // corresponding to the values.
  //
  // Keys should not include `prefix`, i.e., a key ID's name should be the name
  // as it originally appears in `gdef`.
  //
  // If this is non-empty, ImportGraphDef must be called with the shape refiner
  // used to create the existing nodes referenced in `input_map`.
  // TODO(skyewm): can we remove this requirement? How do we access the original
  // shape refiner?
  std::map<SafeTensorId, SafeTensorId> input_map;

  // If true, nodes that will have all output edges removed because of
  // overrides in `input_map` will not be imported.
  bool skip_mapped_nodes;

  // The names of existing nodes in `g` that the imported graph should have
  // control dependencies on.
  //
  // Note that to avoid creating many redundant control edges, ImportGraphDef()
  // won't add control edges to nodes that will inherit the dependencies from
  // other nodes in `gdef`.
  std::vector<string> control_dependencies;

  // Tensors in `gdef` that will be returned via the ImportGraphDefResults
  // output parameter of `ImportGraphDef()`. If this list is non-empty, the
  // caller must pass a results object to `ImportGraphDef()`. The
  // `return_tensors` field will be populated with the imported nodes in `g`.
  //
  // Entries should not include `prefix`, i.e., each ID's name should be the
  // name as it originally appears in `gdef`.
  //
  // If this contains a tensor that's also being remapped via `input_map`, the
  // corresponding existing tensor in `g` will be returned.
  std::vector<SafeTensorId> return_tensors;

  // The names of nodes in `gdef` that will be returned via the
  // ImportGraphDefResults output parameter of `ImportGraphDef()`. If this list
  // is non-empty, the caller must pass a results object to
  // `ImportGraphDef()`. The `return_nodes` field will be populated with the
  // imported nodes in `g`.
  //
  // Entries should not include `prefix`, i.e., each node's name should be the
  // name as it originally appears in `gdef`.
  //
  // Unlike `return_tensors`, `input_map` has no effect on the nodes
  // returned. `return_nodes` must be empty if `skip_mapped_nodes` is true.
  // TODO(skyewm): make this work with `skip_mapped_nodes` if there's a need.
  std::vector<string> return_nodes;

  // If true, checks that all colocation constraints are nodes in the GraphDef.
  bool validate_colocation_constraints = true;

  // If false skips shape validation.
  bool validate_shape;

  // TODO(ashankar): Enable handling of GraphDefs produced by newer binaries
  // with ops that are not defined in the binary calling ImportGraphDef.
  // Similar to the producer_op_list argument to import_graph_def in the
  // python API.
};

// Optional results that may be returned by ImportGraphDef.
struct ImportGraphDefResults {
  // The requested tensors associated with
  // ImportGraphDefOptions::return_tensors. Note that the index may be different
  // than the requested index if the returned tensor has been remapped according
  // to `input_map`.
  typedef int Index;
  std::vector<std::pair<Node*, Index>> return_tensors;

  // The requested nodes associated with ImportGraphDefOptions::return_nodes.
  std::vector<Node*> return_nodes;

  // Keys in ImportGraphDefOptions::input_map that don't appear in `gdef` and
  // weren't used as an input to any node in `gdef`. These keys are likely due
  // to typos, and callers may wish to treat their existence as an error.
  std::vector<SafeTensorId> missing_unused_input_map_keys;
};

// Adds the graph in GraphDef `gdef` into an existing Graph `*g`.
//
// On error, returns non-OK and leaves `*g` unmodified.
//
// `refiner` can be null. It should be non-null if the caller
// intends to add additional nodes to the graph after the import. This
// allows the caller to validate shapes of those nodes (since
// ShapeRefiner::AddNode must be called in topological order).
//
// `results` must be non-null if `opts.return_tensors` or `opts.result_nodes` is
// non-empty. It can also be set to fetch the unused input map keys. If it's
// non-null, all the vector fields must be empty.
//
// TODO(ashankar): Push this mechanism and get rid of Session::Extend()
// as a means of enhancing an existing Graph.
extern Status ImportGraphDef(const ImportGraphDefOptions& opts,
                             const GraphDef& gdef, Graph* g,
                             ShapeRefiner* refiner,
                             ImportGraphDefResults* results = nullptr);

// Make a copy of "src" into "*dest".
//
// REQUIRES: "*dest" is a freshly allocated graph without any nodes or edges
// other than the implicit Source/Sink nodes.
extern void CopyGraph(const Graph& src, Graph* dest);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_
