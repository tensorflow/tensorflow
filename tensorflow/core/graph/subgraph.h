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

#ifndef TENSORFLOW_CORE_GRAPH_SUBGRAPH_H_
#define TENSORFLOW_CORE_GRAPH_SUBGRAPH_H_

#include <string>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace subgraph {

// Information about a graph rewritten by `RewriteGraphForExecution()`.
struct RewriteGraphMetadata {
  // The element type of each tensor fed to this subgraph. The order
  // of types corresponds to the order of tensor names in
  // `fed_outputs` when calling `RewriteGraphForExecution()`.
  DataTypeVector feed_types;
  // The element type of each tensor fetched from this subgraph. The
  // order of types corresponds to the order of tensor names in
  // `fetch_outputs` when calling `RewriteGraphForExecution()`.
  DataTypeVector fetch_types;
};

// Describes the action to take on a particular tensor endpoint (described by
// a "<node_name>:<output_index>" pair) when pruning the graph.
//
// The `AddNode()` method must be overridden to describe this action. The method
// will be invoked once during `RewriteGraphForExecution()` with tensor endpoint
// named by `endpoint_name`, and it may either create a single new node, or fail
// with an error if the resulting graph would be invalid.
class PruneRewrite {
 public:
  // `endpoint_name` and `device_info` must outlive this object.
  PruneRewrite(const string* endpoint_name, const DeviceAttributes* device_info)
      : endpoint_name_(endpoint_name), device_info_(device_info) {}
  virtual ~PruneRewrite() {}

  // Creates a new node whose output replaces the given `tensor` in graph `g`.
  // The node will be assigned to the device named in `device_info`.
  virtual Status AddNode(Graph* g, NodeBuilder::NodeOut tensor,
                         Node** out_node) = 0;

  // Returns the name of the tensor to which this rewrite applies.
  const string& endpoint_name() { return *endpoint_name_; }

 protected:
  // The device on which the new node will be created.
  const DeviceAttributes& device_info() { return *device_info_; }

 private:
  const string* const endpoint_name_;          // Not owned.
  const DeviceAttributes* const device_info_;  // Not owned.
};

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
// of the nodes in "target_node_names" and "fetch_outputs", then these may
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
    Graph* g, const absl::Span<const string>& fed_outputs,
    const absl::Span<const string>& fetch_outputs,
    const absl::Span<const string>& target_node_names,
    const DeviceAttributes& device_info, bool use_function_convention,
    RewriteGraphMetadata* out_metadata);

// A more general version of the above function that supports
// customizable rewriting actions for each fed and fetched tensor.
Status RewriteGraphForExecution(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
    const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
    const absl::Span<const string>& target_node_names,
    RewriteGraphMetadata* out_metadata);

/////////////////////////////////////////////////////////
// Custom rewrite actions for fed and fetched tensors. //
/////////////////////////////////////////////////////////

// A rewrite action that adds an _Arg node for a fed tensor.
class ArgFeedRewrite : public PruneRewrite {
 public:
  ArgFeedRewrite(const string* endpoint_name,
                 const DeviceAttributes* device_info, int32_t arg_index)
      : PruneRewrite(endpoint_name, device_info), arg_index_(arg_index) {}
  Status AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                 Node** out_node) override;

 private:
  const int32 arg_index_;
};

// A rewrite action that adds a client-terminated _Recv node for a fed tensor.
class RecvFeedRewrite : public PruneRewrite {
 public:
  using PruneRewrite::PruneRewrite;
  Status AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                 Node** out_node) override;
};

// A rewrite action that adds a _Retval node for a fetched tensor.
class RetvalFetchRewrite : public PruneRewrite {
 public:
  RetvalFetchRewrite(const string* endpoint_name,
                     const DeviceAttributes* device_info, int32_t retval_index)
      : PruneRewrite(endpoint_name, device_info), retval_index_(retval_index) {}
  Status AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                 Node** out_node) override;

 private:
  const int32 retval_index_;
};

// A rewrite action that adds a client-terminated _Send node for a
// fetched tensor.
class SendFetchRewrite : public PruneRewrite {
 public:
  using PruneRewrite::PruneRewrite;
  Status AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                 Node** out_node) override;
};

}  // namespace subgraph
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_SUBGRAPH_H_
