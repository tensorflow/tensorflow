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

// CondStateMap is responsible for mapping from each graph Node to a CondState,
// where each CondState is the array of CondNodes (corresponding to switch,
// merge or dead states) as described below.  For efficiency, this class interns
// the CondState, so that CondState equality comparisons are simply pointer
// comparisons.
class CondStateMap {
 public:
  explicit CondStateMap(Graph* graph);

  // Represents an entry in the CondState. An entry can either be the
  // switch (along with predicate), merge, or dead:
  // * switch node indicates a node that is executed along a branch with the
  //   given predicate - a branch can be then, else or both;
  // * merge node indicates that the node is executed as output of a merge;
  // * dead indicates that this node can never be executed;
  struct CondNode {
    enum class Type { kSwitch = 1, kMerge = 2, kDead = 3 };

    CondNode(Type type, Node* switch_node = nullptr,
             BranchType branch = BranchType::kNeither);

    string ToString() const;
    bool operator==(const CondNode& other) const;
    bool operator!=(const CondNode& other) const;

    // Type of node.
    Type type;

    // Predicate and branch, only used when type is kSwitch.
    OutputTensor predicate;
    BranchType branch;
  };

  // A node in the graph is executed when multiple conditions hold. The order
  // represents the nesting of the predicates that hold and is used when
  // extracting the nested conditionals.
  using CondState = std::vector<CondNode>;

  // Every unique ID is mapped to a CondState.
  using CondId = const CondState*;

  // Returns the CondId for a given node.
  CondId LookupId(const Node* node) const;

  // Returns the unique CondId for CondState.
  CondId GetUniqueId(const CondState& state);

  // Returns the CondState for a Node.
  // REQUIRES: node has a non-empty CondState.
  const CondState& LookupState(const Node* node) const;

  // Resets the CondId for a given node.
  void ResetId(const Node* node, CondId id);

  // Marks `node` as dead.
  void MarkDead(const Node* node);

  // Determine branch execution of CondState.
  BranchType FindBranchOf(CondId id, OutputTensor predicate) const;

  // Enum to represent whether one cond flow state contains another.
  enum ContainsResult {
    kIncomparable,
    kEqual,
    kLhsContainsRhs,
    kRhsContainsLhs
  };

  // Returns whether the lhs CondState holds wherever rhs CondState hols. I.e.,
  // [(p,t)] contains [(p,t), (r,t)].
  ContainsResult LhsHoldsWhereverRhsHolds(CondId lhs, CondId rhs);

  // Returns textual representation of node's CondState.
  string CondStateToString(const Node* node) const;
  string CondStateToString(CondId id) const;

  // Returns whether the cond state is the dead state.
  bool IsDead(CondId id) const;

  // Returns whether the cond state is the empty state.
  bool IsEmpty(CondId id) const;

  // Computes the predicates that have to hold for a node to execute and returns
  // whether it was possible to determine the predicates that must hold. `scope`
  // is populated with these predicates. Scope differs from state in that it
  // does not include merge and both nodes.
  bool ScopeIn(CondId id, CondId* scope);

 private:
  // Hash for CondNode and CondState.
  struct CondHash {
    size_t operator()(const CondNode& item) const;
    size_t operator()(const CondState& vec) const;
  };

  // Set to keep track of unique CondStates.
  // Pointers to the entries in the unordered set are used as identifiers:
  // unordered_set guarantees that the pointers remain the same.
  std::unordered_set<CondState, CondHash> condstate_set_;

  // Mapping from Node id to CondId.
  std::vector<CondId> node_to_condid_map_;

  // Track the CondId for newly inserted nodes. We use a vector to quickly map
  // from Node id in the original graph to the CondId, but there will be nodes
  // added to the original graph (such as If nodes) whose CondState needs to be
  // tracked too.
  std::unordered_map<int, CondId> added_node_mapping_;

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
  xla::StatusOr<Node*> AddIfNode(const NodeDef& def, const Node* replacee);

  // Propagates the state of a newly inserted node.
  Status PropagateUpdatedState(const Node* replacee);

  // Dump graph with the CondState annotated.
  void DumpGraphWithCondState(const string& name);

 private:
  FunctionalizeCond(Graph* graph, FunctionLibraryDefinition* library);

  // Performs the actual cond functionalization. Iterate over groups of merge
  // nodes (linked by common predicate & CondIds of the incomming edges),
  // from innermost to outermost, and extract into If nodes.
  Status FunctionalizeInternal();

  // Returns the forward flow state propagated along edge `e`.
  // This may modify cond_state_map_.
  CondStateMap::CondId StateAlongEdge(const Edge* e);

  // Determines the CondState of all the nodes in the given vector where
  // the input is expected in reverse topological order.
  // This populates the cond_state_map_.
  Status DetermineCondStates(std::vector<Node*> rev_topo_order);

  // Determine the CondState for a given node using the incomming edges
  // to the node. Note: it is expected that this node's CondState is only
  // determined once its input's CondState is.
  Status DetermineCondState(Node* dst);

  // Helper functions for DetermineCondState.
  Status DetermineCondStateMerge(Node* dst);

  // Helper functions for DetermineCondStates. Determines the dst node's
  // CondState by joining the src and dst's CondState where either
  // the dst node is a merge or not.
  // These may modify cond_state_map_.
  xla::StatusOr<CondStateMap::CondId> JoinCondStatesMerge(
      CondStateMap::CondId src, CondStateMap::CondId dst);
  xla::StatusOr<CondStateMap::CondId> JoinCondStatesNonMerge(
      CondStateMap::CondId src, CondStateMap::CondId dst);

  // Checks if a merge node is redundant and if so removes it from the graph.
  Status RemoveRedundantMerge(Node* node);

  // Checks if a switch node is redundant and if so removes it from the graph.
  Status RemoveRedundantSwitch(Node* node);

  // Sorts merge nodes (in reverse topological order) in order of increasing
  // nesting depth.
  void SortMergeNodes(std::vector<Node*>* merge_order);

  // Deletes all nodes in/consumers of `delete_nodes_`.
  void DeleteReachableNodes();

  // Member used to unique the CondState to a unique CondId and keep track of
  // CondState/CondId per Node.
  CondStateMap cond_state_map_;

  // Nodes to be deleted.
  std::deque<int> delete_nodes_;

  FunctionLibraryDefinition* library_;
  Graph* graph_;

  friend class FunctionalizeCondTest;
};

}  // namespace functionalize_cond

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_COND_H_
