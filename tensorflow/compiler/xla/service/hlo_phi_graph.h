/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PHI_GRAPH_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PHI_GRAPH_H_

#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"

namespace xla {
// Phi graph is a graph that contains and connects phi nodes build on top of
// HloValues with explicit edges, as well as non-phi nodes that are direct
// inputs to the phi nodes. The graph can be viewed as an 'overlay' on top of
// HloValues, with some edges that connect them together. After optimization,
// some phis nodes will be simplified away and the user can then ask two useful
// questions:
//
// 1. Which HloValue should a phi node being replaced with?
// 2. TODO(yunxing): What are the set of aliased HloValues that are connecting
// to the same phi (Must-aliasing).
class PhiGraph {
 public:
  // Register an hlo value into the phi node.
  void RegisterPhi(const HloValue& value,
                   absl::Span<const HloValue* const> inputs);

  HloValue::Id GetOptimizedId(const HloValue& value);

  // Returns true if the input to a hlo value is the same as `inputs`.
  bool InputsEqualTo(const HloValue& value,
                     absl::Span<const HloValue* const> inputs);

  // Given `id`, returns the new id that `id` should be replaced with. If the
  // node is not optimized, returns the same value.
  HloValue::Id FindOptimizedValue(const HloValue::Id id);

  // Optimize the entire graph.
  void Optimize();

  std::string ToString();

 private:
  struct Node {
    bool is_phi;
    // Users of this node. Non-phi node has no operands.
    std::vector<Node*> users;
    // Operands of this node.
    std::vector<Node*> operands;

    // The value that the node is originally registered with.
    HloValue::Id value_id;

    // mark_as_dead is set to true when a phi node is simplified away
    //
    // Precondition: node is a phi.
    bool mark_as_dead = false;
  };

  Node* CreateOrReuseNode(const HloValue& value);

  // Replace `node` with `replace`. Redirect all users to the `replace` and
  // all HloValues pointing to the `node` to `replace`. Also mark `node` as
  // dead.
  //
  // Precondition: node is a phi -- It's only possible to simplify away a
  // phi node.
  void ReplaceNodeWith(Node* node, Node* replace);

  // A reverse mapping of a node in the phi graph and all HloValues pointing
  // to that phi.
  absl::flat_hash_map<Node*, std::vector<HloValue::Id>> node_to_value_id_;

  // A mapping from a HloValue to node in the phi graph.
  absl::flat_hash_map<HloValue::Id, Node*> value_id_to_node_;
  std::vector<std::unique_ptr<Node>> node_storage_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PHI_GRAPH_H_
