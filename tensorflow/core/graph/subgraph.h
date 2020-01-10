#ifndef TENSORFLOW_GRAPH_SUBGRAPH_H_
#define TENSORFLOW_GRAPH_SUBGRAPH_H_

#include <string>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {
namespace subgraph {

// Rewrite the graph structure of "*g" to deal with feeding node
// outputs, fetching node outputs, and only running a subset of the
// graph.  "fed_outputs" and "fetch_outputs" are both lists of
// output tensor identifiers in the form of
// "<name>[:<optional_output_index>]", and "target_nodes_str" is a
// lists of of target node names in "*g" "g".
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

}  // namespace subgraph
}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_SUBGRAPH_H_
