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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_TEST_TOOLS_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_TEST_TOOLS_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"
#include "tensorflow/core/grappler/op_types.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {

//=== Helper methods to construct the nodes.

NodeDef MakeNodeConst(const string& name);

NodeDef MakeNode2Arg(const string& name, const string& opcode,
                     const string& arg1, const string& arg2);

NodeDef MakeNode4Arg(const string& name, const string& opcode,
                     const string& arg1, const string& arg2, const string& arg3,
                     const string& arg4);

inline NodeDef MakeNodeMul(const string& name, const string& arg1,
                           const string& arg2) {
  return MakeNode2Arg(name, "Mul", arg1, arg2);
}

// Not really a 2-argument but convenient to construct.
inline NodeDef MakeNodeAddN(const string& name, const string& arg1,
                            const string& arg2) {
  return MakeNode2Arg(name, "AddN", arg1, arg2);
}

inline NodeDef MakeNodeSub(const string& name, const string& arg1,
                           const string& arg2) {
  return MakeNode2Arg(name, "Sub", arg1, arg2);
}

// Has 2 honest outputs.
inline NodeDef MakeNodeBroadcastGradientArgs(const string& name,
                                             const string& arg1,
                                             const string& arg2) {
  return MakeNode2Arg(name, "BroadcastGradientArgs", arg1, arg2);
}

NodeDef MakeNodeShapeN(const string& name, const string& arg1,
                       const string& arg2);

NodeDef MakeNodeIdentityN(const string& name, const string& arg1,
                          const string& arg2);

NodeDef MakeNodeQuantizedConcat(const string& name, const string& arg1,
                                const string& arg2, const string& arg3,
                                const string& arg4);

//=== A container of pre-constructed graphs.

class TestGraphs {
 public:
  TestGraphs();

  // Graph with 3 nodes and a control link to self (which is not valid in
  // reality but adds excitement to the tests).
  GraphDef graph_3n_self_control_;
  // Graph that has the multi-input links.
  GraphDef graph_multi_input_;
  // Graph that has the all-or-none nodes.
  GraphDef graph_all_or_none_;
  // All the nodes are connected in a circle that goes in one direction.
  GraphDef graph_circular_onedir_;
  // All the nodes are connected in a circle that goes in both directions.
  GraphDef graph_circular_bidir_;
  // The nodes are connected in a line.
  GraphDef graph_linear_;
  // The nodes are connected in a cross shape.
  GraphDef graph_cross_;
  GraphDef graph_small_cross_;
  // For testing the ordering of links at the end of signature generation,
  // a variation of a cross.
  GraphDef graph_for_link_order_;
  // Sun-shaped, a ring with "rays".
  GraphDef graph_sun_;
};

//=== Helper methods for analysing the structures.

std::vector<string> DumpLinkMap(const GenNode::LinkMap& link_map);

// Also checks for the consistency of hash values.
std::vector<string> DumpLinkHashMap(const SigNode::LinkHashMap& link_hash_map);

std::vector<string> DumpHashedPeerVector(
    const SigNode::HashedPeerVector& hashed_peers);

}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_TEST_TOOLS_H_
