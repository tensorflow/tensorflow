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

#ifndef TENSORFLOW_GRAPH_SUBGRAPH_H_
#define TENSORFLOW_GRAPH_SUBGRAPH_H_

#include <string>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace subgraph {

// Rewrite the graph structure of "*g" to deal with feeding node
// outputs, fetching node outputs, and only running a subset of the
// graph.  "fed_outputs" and "fetch_outputs" are both lists of
// output tensor identifiers in the form of
// "<name>[:<optional_output_index>]", and "target_nodes_str" is a
// lists of target node names in "*g" "g".
//
// In the resulting graph "*g", output edges in "fed_outputs" have
// been redirected to special "_recv" nodes introduced into the graph.
// If these fed nodes are not needed in order to compute the effects
// of the nodes in "targets_nodes" and "fetch_outputs", then these may
// be omitted from the graph.
//
// In the resulting graph "*g", additional "_send" nodes are connected
// to every output in "fetch_outputs".  These "_send" nodes are set up
// to execute on the device described by device_info.
//
// On success, returns OK, and sets "*g" to a version of "*g"
// that represents the portions of the graph necessary for producing
// the output of all nodes listed in "target_node_names" and fetching the
// specific node outputs specified in "fetch_outputs".
//
// On failure, returns the error status. Possible errors include:
//    - fed output "node:output_index" does not exist in "*g"
//    - fetch output "node:output_index" does not exist in "*g"
//    - target node "node" does not exist in "*g"
Status RewriteGraphForExecution(
    Graph* g, const gtl::ArraySlice<string>& fed_outputs,
    const gtl::ArraySlice<string>& fetch_outputs,
    const gtl::ArraySlice<string>& target_node_names,
    const DeviceAttributes& device_info);

typedef std::unordered_map<StringPiece, Node*, StringPiece::Hasher> NameIndex;

// Augment "*g" by adding special "fetch" nodes that connect to the
// tensor outputs specified in "fetch_outputs" to retrieve the output
// of the tensors.  The new nodes added are set up to execute on
// "client_device_name", and are returned in "*fetch_nodes".
//
// Return OK on success.  On error, return false and sets *error to
// an appropriate error message (and *g is left in an indeterminate
// state).
Status FetchOutputs(Graph* g, const DeviceAttributes& device_info,
                    const gtl::ArraySlice<string>& fetch_outputs,
                    NameIndex* name_index, std::vector<Node*>* fetch_nodes);

}  // namespace subgraph
}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_SUBGRAPH_H_
