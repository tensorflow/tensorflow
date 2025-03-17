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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GEN_NODE_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GEN_NODE_H_

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

class GenNode;

// To find nodes by name.
using GenNodeMap = std::unordered_map<string, std::unique_ptr<GenNode>>;

// One node in the graph, in the form convenient for traversal and generation of
// subgraphs. It refers to the original NodeDef protobuf for most information
// and adds the extra enrichment.
//
// The graph building is 2-stage: first match a GenNode with each NodeDef and
// collect them into a map that finds them by name, then process the map,
// deep-parse the underlying NodeDefs and connect the GenNodes together.
class GenNode {
 public:
  // Will keep the pointer, so the underlying object must not be deleted while
  // GenNode is alive.
  explicit GenNode(const NodeDef* node);

  // Access wrappers.
  const string& name() const { return node_->name(); }
  const string& opcode() const { return node_->op(); }
  const NodeDef* node_def() const { return node_; }

  // Parse the inputs of this node and update the map accordingly, creating the
  // links (i.e. edges, connections between nodes) in itself and in the nodes
  // it's linked to (the map itself is unchanged, only the nodes in it are
  // updated).
  absl::Status ParseInputs(const GenNodeMap* map);

  // Does the full 2-stage build of the graph. The map should be initially
  // empty. The map keeps pointers to the nodes in source, so the source must
  // not be destroyed before the map.
  static absl::Status BuildGraphInMap(const GraphDef& source, GenNodeMap* map);

  // The enrichment that constitutes the point of this class.

  // Representation of a connection on a node.
  class Port {
   public:
    // A port may be inbound or outbound.
    // Negative ids (canonically -1) mean a control port.
    Port(bool inbound, int32_t id) : value_(id << 1) {
      if (inbound) {
        value_ |= 1;
      }
    }
    Port(const Port&) = default;
    Port& operator=(const Port&) = default;

    bool IsInbound() const { return (value_ & 0x1); }

    bool IsControl() const { return (value_ < 0); }

    int32_t Id() const {
      // Arithmetic shift preserves the sign.
      return (value_ >> 1);
    }

    // Integer type used to represent the encoded port value.
    using IntPort = int32_t;

    // Returns the encoded form of this port, so that it can be used
    // as various map indexes.
    IntPort Encoded() const { return value_; }

    static Port Decode(IntPort encoded) { return Port(encoded); }

    bool operator==(const Port& other) const { return value_ == other.value_; }
    bool operator<(const Port& other) const { return value_ < other.value_; }

    struct Hasher {
      size_t operator()(const Port& port) const noexcept {
        return hasher(port.Encoded());
      }
      std::hash<int32_t> hasher;
    };

    // Convenient for printing. I've really wanted it to be implicit but
    // ClangTidy insists on making it explicit.
    explicit operator string() const;

   private:
    explicit Port(IntPort value) : value_(value) {}

    IntPort value_;
  };

  struct LinkTarget {
    GenNode* node;  // Node where this link points.
    Port port;      // Port on the remote side of this link.

    LinkTarget(GenNode* a_node, Port a_port) : node(a_node), port(a_port) {}
  };
  // All the links that are connected to the same port of this node
  // are collected in one vector. A link is an edge of the graph that connects
  // 2 nodes. Each of the connected nodes has its own perspective on the link,
  // seeing its local port, remote port and the remote node. The direction of
  // the link is encoded in the ports, one port is always incoming and another
  // one outgoing.
  using LinkTargetVector = std::vector<LinkTarget>;
  // Both inputs and outputs are stored in the same map.
  using LinkMap = std::unordered_map<Port, LinkTargetVector, Port::Hasher>;

  // Access to the link map.
  const LinkMap& links() const { return links_; }

  // Check whether the port is an input (including the controls) with multiple
  // connections. Such inputs get handled in a special way when building the
  // subgraphs, in an "all or nothing" fashion.
  bool IsMultiInput(Port port) const;

  // When building the subgraphs, must include either all non-control inputs of
  // this node into the subgraph or none of them. This happens when at least one
  // of the inputs is a multi-input (or if the opcode is commutative, thus
  // treating all the inputs as one multi-input).
  bool AllInputsOrNone() const { return all_inputs_or_none_; }

 private:
  const NodeDef* node_;
  // Becomes valid only after ParseInputs().
  const OpDef* op_;

  // The opcode has a complicated structure of input args, with multi-input args
  // that are not commutative. This means that to make sense, the subgraphs that
  // include this node must also include either all its inputs or none of them.
  bool all_inputs_or_none_ = false;

  LinkMap links_;
};

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GEN_NODE_H_
