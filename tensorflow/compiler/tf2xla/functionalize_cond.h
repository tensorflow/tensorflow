/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_COND_H_
#define TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_COND_H_

#include <deque>
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Functionalize all the switch-merge nodes of a loop-free graph into If
// nodes. That is, attempt to transform every remaining switch and merge nodes
// in the graph into If nodes.
// Precondition: All while loops have been removed from graph.
Status FunctionalizeCond(Graph* graph, FunctionLibraryDefinition* library);

// Internal functions/classes exposed for testing purposes.
namespace functionalize_cond {

// All nodes are assumed to be either in no branch, then branch, else branch,
// or both branches (such as merge nodes).
// The code below relies on Else and Then being 0 and 1 (corresponding to the
// switch outputs). Both and Neither are arbitrary.
enum class BranchType {
  kElseBranch = 0,
  kThenBranch = 1,
  kBoth = 2,
  kNeither = 3,
};

// When we keep track of which switch/merge node's feed into a node, we record
// 1) predicate for non-dead switch node,
// 2) the switch node itself for dead switch node,
// 3) the merge node itself for merge node.
// Case 1) is an optimization. With this optimization, if there are nodes from
// different switch nodes but those switch nodes have the same predicate, the
// nodes will still have same AncestorState, and they will be clustered into a
// single "If".
struct AncestorNode {
  enum class AncestorNodeType {
    kPred = 0,
    kSwitch = 1,
    kMerge = 2,
  };

  OutputTensor output_tensor;
  AncestorNodeType type;

  // Compare two AncestorNodes by (node id, index, type).
  bool operator<(const AncestorNode& other) const;
  bool operator==(const AncestorNode& other) const;

  struct Hash {
    size_t operator()(const AncestorNode&) const;
  };
};

// StateMap is responsible for mapping from each graph Node to
// * a CondState, where each CondState is a map from predicate to branch (i,e.,
//   what predicates have to hold or not hold).
// * a AncestorState, where each AncestorState is a set of switch/merge nodes
//   that are an ancestor of the node in the graph;
// For efficiency, this class interns the CondState (AncestorState), so that
// CondState (AncestorState) equality comparisons are simply pointer
// comparisons.
class StateMap {
 public:
  explicit StateMap(Graph* graph);

  // Compare two OutputTensors by (node id, index).
  struct OutputTensorLess {
    bool operator()(const OutputTensor& lhs, const OutputTensor& rhs) const;
  };

  // A node in the graph is executed when multiple conditions hold. Keep track
  // of the predicates that must hold for a node to execute.
  using CondState = std::map<OutputTensor, BranchType, OutputTensorLess>;

  // Every unique ID is mapped to a CondState.
  using CondId = const CondState*;

  // Keep track of which switch/merge node's feed into a node's values.
  using AncestorState = std::set<AncestorNode>;

  // Every unique ID is mapped to a AncestorState.
  using AncestorId = const AncestorState*;

  // Returns the CondId for a given node.
  CondId LookupCondId(const Node* node) const;

  // Returns the unique CondId for CondState.
  CondId GetCondId(const CondState& state);

  // Resets the CondId for a given node.
  void ResetCondId(const Node* node, CondId id);

  // Returns the AncestorId for a given node.
  AncestorId LookupAncestorId(const Node* node) const;

  // Returns the unique AncestorId for CondState.
  AncestorId GetAncestorId(const AncestorState& state);

  // Resets the AncestorId for a given node.
  void ResetAncestorId(const Node* node, AncestorId id);

  // Marks `node` as dead.
  void MarkDead(const Node* node);

  // Determine branch execution of CondState.
  BranchType FindBranchOf(CondId id, OutputTensor predicate) const;

  // Returns textual representation of node's CondState.
  string CondStateToString(const Node* node) const;
  string CondStateToString(CondId id) const;

  // Returns textual representation of node's AncestorState.
  string AncestorStateToString(const Node* node) const;

  // Returns whether the cond state is the dead state.
  bool IsDead(CondId id) const;

  // Returns whether the cond state is the empty state.
  bool IsEmpty(CondId id) const;

 private:
  // Hash for CondState and AncestorState.
  struct Hash {
    size_t operator()(const CondState& map) const;
    size_t operator()(const AncestorState& map) const;
  };

  // Set to keep track of unique CondStates.
  // Pointers to the entries in the unordered set are used as identifiers:
  // unordered_set guarantees that the pointers remain the same.
  std::unordered_set<CondState, Hash> condstate_set_;

  // Mapping from Node id to CondId.
  std::vector<CondId> node_to_condid_map_;

  // Track the CondId for newly inserted nodes. We use a vector to quickly map
  // from Node id in the original graph to the CondId, but there will be nodes
  // added to the original graph (such as If nodes) whose CondState needs to be
  // tracked too.
  std::unordered_map<int, CondId> added_node_condid_mapping_;

  // AncestorId variants of the CondId members.
  std::unordered_set<AncestorState, Hash> ancestorstate_set_;
  std::vector<AncestorId> node_to_ancestorid_map_;
  std::unordered_map<int, AncestorId> added_node_ancestorid_mapping_;

  // Identifier of the dead flow state. The empty flow state is represented with
  // a nullptr.
  CondId dead_id_;
};

// FunctionalizeCond groups all the state used by functionalizing conditionals
// of the given graph together.
class FunctionalizeCond {
 public:
  // Functionalize all the switch-merge nodes of a loop-free graph into If
  // nodes. That is, attempt to transform every remaining switch and merge nodes
  // in the graph into If nodes.
  // Precondition: All while loops have been removed from graph.
  static Status Functionalize(Graph* graph, FunctionLibraryDefinition* library);

  // Build identity node with the same name as the merge that will be replaced
  // in case the output is fetched/colocated.
  Status AddIdentityNode(const Node* replacee, Node* if_node, int port);

  // Add a If node to the graph defined by def that will, amongst other, replace
  // replacee in the graph.
  xla::StatusOr<Node*> AddIfNode(const NodeDef& def, const Node* replacee,
                                 const OutputTensor& predicate);

  // Propagates the state of a newly inserted node.
  Status PropagateUpdatedState(const Node* replacee);

  // Dump graph with the CondState annotated.
  void DumpGraphWithCondState(const string& name);

  // Adds `switch_id` to the list of Switch node ids.
  void AddSwitchId(int switch_id);

 private:
  FunctionalizeCond(Graph* graph, FunctionLibraryDefinition* library);

  // Performs the actual cond functionalization. Iterate over groups of merge
  // nodes (linked by common predicates & ancestor IDs), from innermost to
  // outermost, and extract into If nodes.
  Status FunctionalizeInternal();

  // Returns the forward flow state propagated along edge `e`.
  // This may modify state_map_.
  StateMap::CondId StateAlongEdge(const Edge* e);

  // Determines the CondState and AncestorState of all the nodes in the given
  // vector where the input is expected in reverse topological order.
  // This populates the state_map_.
  Status DetermineStates(std::vector<Node*> rev_topo_order);

  // Determine the CondState for a given node using the incomming edges
  // to the node. Note: it is expected that this node's CondState is only
  // determined once its input's CondState is.
  Status DetermineCondState(Node* dst) {
    if (IsMerge(dst)) return DetermineCondStateMerge(dst);
    return DetermineCondStateNonMerge(dst);
  }

  // Helper functions for DetermineCondState.
  Status DetermineCondStateNonMerge(Node* dst);
  Status DetermineCondStateMerge(Node* dst);

  // Determines the dst node's CondState by joining the src and dst's CondState
  // where either the dst node is a merge or not.
  // These may modify state_map_.
  xla::StatusOr<StateMap::CondId> JoinCondStatesMerge(Node* merge,
                                                      StateMap::CondId src,
                                                      StateMap::CondId dst);
  xla::StatusOr<StateMap::CondId> JoinCondStatesNonMerge(StateMap::CondId src,
                                                         StateMap::CondId dst);

  // Determines which switch/merge nodes are ancestors of this node.
  Status DetermineAncestorState(Node* dst);

  // Checks if a merge node is redundant and if so removes it from the graph.
  Status RemoveRedundantMerge(Node* node);

  // Checks if a switch node is redundant and if so removes it from the graph.
  Status RemoveRedundantSwitch(Node* node);

  // Sorts merge nodes (in reverse topological order) in order of increasing
  // nesting depth.
  void SortMergeNodes(std::vector<Node*>* merge_order);

  // Deletes all nodes in/consumers reachable from switch/merge nodes that were
  // extracted.
  void DeleteReachableAndDeadNodes(const std::vector<Node*>& merge_order);

  // Member used to unique the CondState to a unique CondId (AncestorState to a
  // unique AncestorId) and keep track of CondState/CondId
  // (AncestorState/AncestorId) per Node.
  StateMap state_map_;

  // Mapping from merge nodes to predicate.
  std::unordered_map<Node*, OutputTensor> merge_to_predicate_;

  // Mapping from merge nodes to corresponding If node outputs.
  std::unordered_map<Node*, OutputTensor> merge_to_replacement_;

  FunctionLibraryDefinition* library_;
  Graph* graph_;

  friend class FunctionalizeCondTest;

  std::vector<int> switch_ids_;
};

}  // namespace functionalize_cond

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_COND_H_
