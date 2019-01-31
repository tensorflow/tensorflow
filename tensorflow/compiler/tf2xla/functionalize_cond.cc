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

#include "tensorflow/compiler/tf2xla/functionalize_cond.h"

#include <algorithm>
#include <deque>
#include <stack>
#include <unordered_set>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

using xla::StatusOr;

namespace tensorflow {
namespace functionalize_cond {

// TODO(jpienaar): Move to OutputTensor.
string DebugString(const OutputTensor& tensor) {
  return absl::StrCat(tensor.node->name(), ":", tensor.index);
}

string Branch_Name(BranchType b) {
  switch (b) {
    case BranchType::kElseBranch:
      return "else";
    case BranchType::kThenBranch:
      return "then";
    case BranchType::kBoth:
      return "both";
    case BranchType::kNeither:
      return "neither";
  }
}

string DebugString(StateMap::CondId cond_state) {
  if (cond_state == nullptr || cond_state->empty()) return "{}";
  using value_type = StateMap::CondState::value_type;
  return absl::StrCat(
      "{",
      absl::StrJoin(*cond_state, ", ",
                    [](string* output, const value_type& pred_branch) {
                      const OutputTensor& pred = pred_branch.first;
                      const BranchType& branch = pred_branch.second;
                      if (branch == BranchType::kNeither)
                        absl::StrAppend(output, "d");
                      else
                        absl::StrAppend(output, "s(", DebugString(pred), ",",
                                        Branch_Name(branch), ")");
                    }),
      "}");
}

// Returns the predicate of a switch.
Status GetSwitchPredicate(const Node& switch_node, OutputTensor* pred) {
  const Edge* pred_edge;
  TF_RETURN_IF_ERROR(switch_node.input_edge(1, &pred_edge));
  // The predicate can be preceded by a identity node. Look through
  // identity nodes to predicate.
  while (pred_edge->src()->IsIdentity()) {
    TF_RETURN_IF_ERROR(pred_edge->src()->input_edge(0, &pred_edge));
  }
  *pred = OutputTensor(pred_edge->src(), pred_edge->src_output());
  return Status::OK();
}

Status GetSwitchValue(const Node& switch_node, OutputTensor* val) {
  const Edge* val_edge;
  TF_RETURN_IF_ERROR(switch_node.input_edge(0, &val_edge));
  *val = OutputTensor(val_edge->src(), val_edge->src_output());
  return Status::OK();
}

bool StateMap::OutputTensorLess::operator()(const OutputTensor& lhs,
                                            const OutputTensor& rhs) const {
  return (lhs.node->id() < rhs.node->id()) ||
         (lhs.node->id() == rhs.node->id() && lhs.index < rhs.index);
}

struct CondStateLess {
  bool operator()(const StateMap::CondState::value_type& lhs,
                  const StateMap::CondState::value_type& rhs) const {
    if (StateMap::OutputTensorLess().operator()(lhs.first, rhs.first))
      return true;
    if (lhs.first.node->id() == rhs.first.node->id() &&
        lhs.first.index == rhs.first.index)
      return lhs.second < rhs.second;
    return false;
  }
};

StateMap::StateMap(Graph* graph) {
  node_to_condid_map_.resize(graph->num_node_ids());
  node_to_ancestorid_map_.resize(graph->num_node_ids());
  // Initialize the dead state (empty state is designated with a nullptr).
  dead_id_ = GetCondId(
      {std::make_pair(OutputTensor(nullptr, -1), BranchType::kNeither)});
}

bool StateMap::IsDead(StateMap::CondId id) const { return id == dead_id_; }

bool StateMap::IsEmpty(StateMap::CondId id) const { return id == nullptr; }

size_t StateMap::Hash::operator()(const StateMap::CondState& map) const {
  if (map.empty()) return 0;
  // Compute hash of the front element.
  auto it = map.begin();
  size_t h = Hash64Combine(OutputTensor::Hash()(it->first),
                           hash<BranchType>()(it->second));
  for (++it; it != map.end(); ++it) {
    // Combine the has with the different elements in the map.
    h = Hash64Combine(h, Hash64Combine(OutputTensor::Hash()(it->first),
                                       hash<BranchType>()(it->second)));
  }
  return h;
}

size_t StateMap::Hash::operator()(const StateMap::AncestorState& map) const {
  if (map.empty()) return 0;
  // Compute hash of the front element.
  auto it = map.begin();
  size_t h = hash<Node*>()(*it);
  for (++it; it != map.end(); ++it) {
    // Combine the has with the different elements in the map.
    h = Hash64Combine(h, hash<Node*>()(*it));
  }
  return h;
}

// CondArgNode represents a input to the conditional and its corresponding
// switch nodes.
struct CondArgNode {
  explicit CondArgNode(Node* src, int src_output)
      : src(src), src_output(src_output) {}

  string ToString() const {
    return absl::StrCat("src=", src->name(), ":", src_output,
                        " switches=", NodesToString(switches));
  }

  Node* src;
  int src_output;
  std::array<Node*, 2> branch_copy;
  std::vector<Node*> switches;
};
using CondArgNodes = std::vector<CondArgNode>;

string DebugString(const CondArgNodes& nodes) {
  return absl::StrCat(
      "[",
      absl::StrJoin(nodes, ", ",
                    [](string* output, const CondArgNode& node) {
                      absl::StrAppend(output, node.ToString());
                    }),
      "]");
}

StateMap::CondId StateMap::LookupCondId(const Node* node) const {
  if (node->id() < node_to_condid_map_.size())
    return node_to_condid_map_[node->id()];
  return added_node_condid_mapping_.at(node->id());
}

StateMap::CondId StateMap::GetCondId(const StateMap::CondState& state) {
  if (state.empty()) return nullptr;
  return &*condstate_set_.insert(state).first;
}

void StateMap::ResetCondId(const Node* node, StateMap::CondId id) {
  if (node->id() < node_to_condid_map_.size())
    node_to_condid_map_[node->id()] = id;
  else
    added_node_condid_mapping_[node->id()] = id;
}

StateMap::AncestorId StateMap::LookupAncestorId(const Node* node) const {
  if (node->id() < node_to_ancestorid_map_.size())
    return node_to_ancestorid_map_[node->id()];
  return added_node_ancestorid_mapping_.at(node->id());
}

StateMap::AncestorId StateMap::GetAncestorId(
    const StateMap::AncestorState& state) {
  if (state.empty()) return nullptr;
  return &*ancestorstate_set_.insert(state).first;
}

void StateMap::ResetAncestorId(const Node* node, StateMap::AncestorId id) {
  if (node->id() < node_to_ancestorid_map_.size())
    node_to_ancestorid_map_[node->id()] = id;
  else
    added_node_ancestorid_mapping_[node->id()] = id;
}

void StateMap::MarkDead(const Node* node) { ResetCondId(node, dead_id_); }

string StateMap::CondStateToString(const Node* node) const {
  return CondStateToString(LookupCondId(node));
}

string StateMap::CondStateToString(StateMap::CondId id) const {
  return DebugString(id);
}

string StateMap::AncestorStateToString(const Node* node) const {
  if (auto id = LookupAncestorId(node)) return NodesToString(*id);
  return "{}";
}

FunctionalizeCond::FunctionalizeCond(Graph* graph,
                                     FunctionLibraryDefinition* library)
    : state_map_(graph), library_(library), graph_(graph) {}

// Class representing the merge/switch nodes that will become a conditional.
class Conditional {
 public:
  Conditional(OutputTensor predicate, FunctionalizeCond* parent,
              StateMap* cond_state_map);

  // Adds merge node that is part of this conditional.
  Status AddMerge(Node* m);

  // Constructs an If node from the merge nodes.
  Status BuildAndReplace(
      Graph* graph, FunctionLibraryDefinition* library,
      std::unordered_map<Node*, OutputTensor>* merge_to_replacement);

 private:
  // Extracts the then/else bodies: creates new graphs with the nodes
  // corresponding to the nodes in the then/else branches as of this conditional
  // as function bodies.
  Status ExtractBodies(Graph* graph);

  // Builds the arguments that are the input to the If.
  Status BuildArgumentNodes();

  // Builds the If node for the extracted bodies with the given predicate.
  Status BuildIfNode(Graph* graph, FunctionLibraryDefinition* library);

  // Adds input edges to If node.
  Status AddInputEdges(
      Graph* graph,
      const std::unordered_map<Node*, OutputTensor>& merge_to_replacement);

  // Adds output edges from If node.
  // Record new output tensor for all Merge nodes in 'merge_to_replacement'.
  Status AddOutputEdges(
      Graph* graph,
      std::unordered_map<Node*, OutputTensor>* merge_to_replacement);

  // Adds switch node that is part of this conditional.
  Status AddSwitch(Node* s);

  // Adds a switch node along the edge and rewire the edge to go via the switch.
  Status AddSwitchNodeAlongEdge(const Edge* edge, BranchType branch,
                                Graph* graph);

  // Internal name of conditional. The name is based on the first merge node
  // added.
  string name() const;

  // The FunctionalizeCond instance that created this.
  FunctionalizeCond* parent_;

  // Mapping between nodes and their cond state.
  StateMap* state_map_;

  // The predicate of the conditional.
  OutputTensor predicate_;

  // The predicate of the switches of the conditional. This may be different
  // than predicate (which is initialized from the original graph) as the
  // predicate could be the output of a newly created If node.
  OutputTensor switch_predicate_;

  // Switch nodes in graph that are part of this conditional.
  std::set<Node*, NodeCmpByNameResourcesLast> switches_;

  // Merge nodes in graph that are part of this conditional.
  std::set<Node*, NodeCmpByNameResourcesLast> merges_;

  // Vector of control inputs from outside the conditional to a node inside.
  std::vector<Node*> external_control_inputs_;
  std::vector<Node*> external_control_outputs_;

  // Graphs corresponding to the then and else branch.
  std::array<std::unique_ptr<Graph>, 2> bodies_;

  // Maps from graph_ to the branch body's graph.
  std::array<std::vector<Node*>, 2> node_maps_;

  // The argument nodes created for the switches.
  CondArgNodes cond_arg_nodes_;

  // The constructed If node.
  Node* if_node_ = nullptr;

  // Whether the merge nodes of this conditional have been replaced.
  bool replaced_ = false;
};

Conditional::Conditional(OutputTensor predicate, FunctionalizeCond* parent,
                         StateMap* cond_state_map)
    : parent_(parent), state_map_(cond_state_map), predicate_(predicate) {}

Status Conditional::AddMerge(Node* m) {
  merges_.insert(m);
  return Status::OK();
}

Status Conditional::AddSwitch(Node* s) {
  VLOG(5) << "Adding switch " << s->DebugString();
  OutputTensor predicate;
  TF_RETURN_IF_ERROR(GetSwitchPredicate(*s, &predicate));
  if (switch_predicate_.node == nullptr) switch_predicate_ = predicate;
  if (!(switch_predicate_ == predicate)) {
    return errors::InvalidArgument(
        "Merge nodes ", NodesToString(merges_),
        " directly dominated by switch nodes with different predicates (",
        DebugString(switch_predicate_), " vs ", DebugString(predicate), ").");
  }
  switches_.insert(s);
  parent_->AddSwitchId(s->id());
  return Status::OK();
}

Status Conditional::BuildArgumentNodes() {
  VLOG(1) << "Build function arguments";
  struct Hash {
    size_t operator()(const std::pair<Node*, int>& item) const {
      return Hash64Combine(hash<Node*>()(item.first),
                           std::hash<int>()(item.second));
    }
  };

  std::unordered_map<std::pair<Node*, int>, int, Hash> input_index;
  for (Node* switch_node : switches_) {
    const Edge* e;
    TF_RETURN_IF_ERROR(switch_node->input_edge(0, &e));
    std::pair<Node*, int> key = std::make_pair(e->src(), e->src_output());
    if (input_index.find(key) == input_index.end()) {
      input_index[key] = cond_arg_nodes_.size();
      cond_arg_nodes_.emplace_back(key.first, key.second);
    }
    cond_arg_nodes_.at(input_index.at(key)).switches.push_back(switch_node);
  }
  VLOG(5) << "CondArg nodes created: " << DebugString(cond_arg_nodes_);

  int arg_count = 0;
  for (CondArgNode& cond_arg_node : cond_arg_nodes_) {
    DataType dtype = cond_arg_node.src->output_type(cond_arg_node.src_output);
    for (auto branch : {BranchType::kElseBranch, BranchType::kThenBranch}) {
      int branch_index = static_cast<int>(branch);
      TF_RETURN_IF_ERROR(
          NodeBuilder(absl::StrCat("_Arg", arg_count),
                      FunctionLibraryDefinition::kArgOp)
              .Attr("T", dtype)
              .Attr("index", arg_count)
              .Finalize(bodies_[branch_index].get(),
                        &cond_arg_node.branch_copy[branch_index]));
    }
    for (Node* node : cond_arg_node.switches) {
      for (const Edge* e : node->out_edges()) {
        if (e->IsControlEdge()) continue;
        int branch_index = e->src_output();
        Node* src_copy = cond_arg_node.branch_copy[branch_index];
        Node* dst_copy = node_maps_[branch_index][e->dst()->id()];

        // The graph may contain dead switch nodes,
        if (dst_copy == nullptr) continue;

        TF_RET_CHECK(dst_copy != nullptr)
            << "Unable to find copied node for " << e->dst()->DebugString()
            << " on branch " << Branch_Name(BranchType(branch_index));
        // If the input goes directly to a merge then the merge has
        // been replaced by a retval so the dst input is 0 instead of
        // dst_input.
        int dst_input = IsMerge(e->dst()) ? 0 : e->dst_input();
        bodies_[branch_index]->AddEdge(src_copy, 0, dst_copy, dst_input);
      }
    }
    ++arg_count;
  }

  // Verify that all retvals have an input.
  // TODO(jpienaar): One could add a ZerosLike in the branch that doesn't have
  // input.
  for (Node* m : merges_) {
    for (auto branch : {BranchType::kElseBranch, BranchType::kThenBranch}) {
      bool has_input = false;
      for (auto e : node_maps_[static_cast<int>(branch)][m->id()]->in_edges()) {
        if (!e->IsControlEdge()) {
          has_input = true;
          break;
        }
      }
      if (!has_input) {
        return errors::Internal(
            "Failed to functionalize control flow with merge ",
            FormatNodeForError(*m), " that doesn't have input on ",
            Branch_Name(branch), " branch.");
      }
    }
  }

  return Status::OK();
}

Status Conditional::AddSwitchNodeAlongEdge(const Edge* edge, BranchType branch,
                                           Graph* graph) {
  // Previously we had edge:
  //   src:src_output ---- edge ----> dst:dst_input
  // post this we have (in graph)
  //   src:src_output --> switch<pred> --- new_edge --> dst:dst_input

  // TODO(jpienaar): One could keep a map caching the extra switch nodes added
  // to avoid adding another switch to feed a value for which a switch was
  // already added.
  Node* switch_node;
  Node* src = edge->src();
  int src_output = edge->src_output();
  TF_RETURN_IF_ERROR(
      NodeBuilder(graph->NewName(absl::StrCat(src->name(), "_added_switch")),
                  "Switch")
          .Input(src, src_output)
          .Input(const_cast<Node*>(predicate_.node), predicate_.index)
          .Finalize(graph, &switch_node));
  state_map_->ResetCondId(switch_node, state_map_->LookupCondId(src));
  state_map_->ResetAncestorId(switch_node, state_map_->LookupAncestorId(src));

  Node* dst = edge->dst();
  int dst_input = edge->dst_input();
  graph->RemoveEdge(edge);
  graph->AddEdge(switch_node, static_cast<int>(branch), dst, dst_input);
  return AddSwitch(switch_node);
}

Status Conditional::ExtractBodies(Graph* graph) {
  VLOG(2) << "Extracting bodies for " << name();
  for (auto b : {BranchType::kElseBranch, BranchType::kThenBranch}) {
    bodies_[static_cast<int>(b)] =
        absl::make_unique<Graph>(graph->op_registry());
  }

  auto find_branch = [&](const Edge* e) {
    const auto& id = state_map_->LookupCondId(e->src());
    return IsSwitch(e->src()) ? BranchType(e->src_output())
                              : state_map_->FindBranchOf(id, predicate_);
  };

  std::array<std::vector<Node*>, 2> stacks;
  VLOG(5) << "Merges: " << NodesToString(merges_);
  for (Node* m : merges_) {
    VLOG(5) << "For merge: " << m->DebugString() << " "
            << state_map_->CondStateToString(m);
    for (auto e : m->in_edges()) {
      if (e->IsControlEdge()) continue;
      BranchType branch = find_branch(e);
      TF_RET_CHECK(branch == BranchType::kThenBranch ||
                   branch == BranchType::kElseBranch)
          << "Error: " << e->src()->name()
          << " is not on either then or else branch (" << Branch_Name(branch)
          << ") for predicate " << DebugString(predicate_) << " ["
          << DebugString(state_map_->LookupCondId(e->src())) << "].";
      Node* src = e->src();
      if (IsSwitch(src)) {
        // Switch node outputs and dependencies are handled separately.
        TF_RETURN_IF_ERROR(AddSwitch(src));
      } else {
        stacks[static_cast<int>(branch)].push_back(src);
      }
    }
  }

  for (auto branch : {BranchType::kElseBranch, BranchType::kThenBranch}) {
    int branch_index = static_cast<int>(branch);
    auto output = bodies_[branch_index].get();
    auto& stack = stacks[branch_index];
    VLOG(5) << "In branch: " << Branch_Name(branch) << " "
            << NodesToString(stack);
    std::vector<bool> visited(graph->num_node_ids(), false);
    node_maps_[branch_index].resize(graph->num_node_ids(), nullptr);
    auto& node_map = node_maps_[branch_index];

    while (!stack.empty()) {
      Node* n = stack.back();
      stack.pop_back();

      if (visited.at(n->id())) continue;
      visited[n->id()] = true;

      // Verify output edges and record control edges exitting scope.
      for (const Edge* e : n->out_edges()) {
        Node* dst = e->dst();
        if (IsMerge(dst)) continue;
        Node* src = e->src();

        auto dst_id = state_map_->LookupCondId(dst);
        auto src_id = state_map_->LookupCondId(src);
        if (dst_id != src_id) {
          if (e->IsControlEdge()) {
            external_control_outputs_.push_back(e->src());
          } else {
            // Constants are treated specially to workaround the case of
            // non-dominated constant nodes.
            if (!IsConstant(src)) {
              // TODO(b/78882471): A node that feeds into two different
              // CondState is not necessarily an error so log a warning for now
              // but revisit to improve the testing to enable making this an
              // error.
              LOG(WARNING) << errors::InvalidArgument(
                  "Graph contains node ", FormatNodeForError(*src),
                  " that feeds into node ", FormatNodeForError(*dst),
                  " but these nodes are in different control contexts (",
                  DebugString(src_id), " vs ", DebugString(dst_id),
                  " (detected during out edge testing)");
            }
          }
        }
      }

      // Copying incomming edges to dst node. Iterate over a copy of the edges
      // as they could be mutated during iteration.
      std::vector<const Edge*> in_edges(n->in_edges().begin(),
                                        n->in_edges().end());
      for (const Edge* e : in_edges) {
        Node* src = e->src();
        // Skip src/dst node.
        if (!src->IsOp()) continue;

        Node* dst = e->dst();
        if (IsSwitch(src)) {
          // Switch node outputs and dependencies are handled separately.
          TF_RETURN_IF_ERROR(AddSwitch(src));
          continue;
        }

        // Verify input is from the same context.
        auto src_id = state_map_->LookupCondId(src);
        auto dst_id = state_map_->LookupCondId(dst);
        if (IsMerge(dst) || src_id == dst_id) {
          // TODO(jpienaar): The merge case can be more strict.
          if (node_map.at(src->id()) == nullptr) {
            node_map.at(src->id()) = output->CopyNode(src);
            stack.push_back(src);
          }
        } else if (e->IsControlEdge()) {
          external_control_inputs_.push_back(src);
        } else {
          // This shouldn't happen, this means we have an external data input
          // not entering via a switch node. Work around this by for
          // * constant nodes copy them;
          // * non-constant nodes, insert a switch along the edge;
          if (IsConstant(src)) {
            node_map.at(src->id()) = output->CopyNode(src);
          } else {
            StateMap::CondState state = *dst_id;
            state.erase(predicate_);
            if (state_map_->GetCondId(state) == src_id) {
              TF_RETURN_IF_ERROR(AddSwitchNodeAlongEdge(e, branch, graph));
              continue;
            } else {
              return errors::InvalidArgument(
                  "Graph contains node ", FormatNodeForError(*src),
                  " that feeds into node ", FormatNodeForError(*dst),
                  " but these nodes are in different control contexts (",
                  DebugString(src_id), " vs ", DebugString(dst_id),
                  " (detected during in edge testing)");
            }
          }
        }

        Node* src_copy = node_map.at(e->src()->id());
        int src_output = e->src_output();
        if (node_map.at(dst->id()) == nullptr) {
          node_map.at(dst->id()) = output->CopyNode(dst);
        }
        Node* dst_copy = node_map.at(e->dst()->id());
        if (e->IsControlEdge()) {
          // Skip control inputs from external context.
          if (src_copy != nullptr) output->AddControlEdge(src_copy, dst_copy);
        } else {
          output->AddEdge(src_copy, src_output, dst_copy, e->dst_input());
        }
      }
    }
  }

  // Build return values from the merge nodes.
  int index = 0;
  for (Node* m : merges_) {
    for (auto branch : {BranchType::kElseBranch, BranchType::kThenBranch}) {
      int branch_index = static_cast<int>(branch);
      auto& node_map = node_maps_[branch_index];
      auto output = bodies_[branch_index].get();
      TF_ASSIGN_OR_RETURN(node_map[m->id()],
                          BuildRetvalNode(output, m->output_type(0), index));
    }
    ++index;

    // Connect the input to the merge_ with the retval, except if it is a
    // Swich node, which is handled separately.
    for (auto e : m->in_edges()) {
      if (e->IsControlEdge()) continue;
      int branch_index = static_cast<int>(find_branch(e));
      auto& node_map = node_maps_[branch_index];
      auto output = bodies_[branch_index].get();
      Node* in = e->src();
      if (!IsSwitch(in)) {
        if (node_map.at(in->id()) == nullptr) {
          node_map[in->id()] = output->CopyNode(in);
        }
        output->AddEdge(node_map[in->id()], e->src_output(),
                        node_map.at(m->id()), 0);
      }
    }
  }
  return Status::OK();
}

Status Conditional::BuildIfNode(Graph* graph,
                                FunctionLibraryDefinition* library) {
  VLOG(2) << "Build cond function for " << name();
  NodeDebugInfo debug_info((*merges_.begin())->def());
  NodeDefBuilder builder(name(), "If", library, &debug_info);
  const string branch_name[] = {"else_branch", "then_branch"};
  for (auto branch : {BranchType::kElseBranch, BranchType::kThenBranch}) {
    int branch_index = static_cast<int>(branch);
    static std::atomic<int64> sequence_num(0LL);
    int64 id = ++sequence_num;

    NameAttrList body_name;
    body_name.set_name(
        absl::StrCat("_functionalize_if_", branch_name[branch_index], "_", id));

    VLOG(3) << "FunctionalizeControlFlow (" << branch_name[branch_index]
            << "): "
            << dump_graph::DumpGraphToFile(
                   "functionalize_cond_body_" + branch_name[branch_index],
                   *bodies_[branch_index], nullptr);

    FunctionDef body_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*bodies_[branch_index],
                                          body_name.name(), &body_fdef));
    TF_RETURN_IF_ERROR(library->AddFunctionDef(body_fdef));
    builder.Attr(branch_name[branch_index], body_name);
  }

  VLOG(3) << "Build input type";
  std::vector<NodeDefBuilder::NodeOut> inputs;
  DataTypeVector in_arg_types;
  for (auto& kv : cond_arg_nodes_) {
    bool inserted = false;
    for (const Node* arg : kv.switches) {
      const Edge* in_edge;
      TF_RETURN_IF_ERROR(arg->input_edge(0, &in_edge));
      if (in_edge->IsControlEdge()) {
        builder.ControlInput(in_edge->src()->name());
      } else {
        if (!inserted) {
          DataType dtype = arg->input_type(0);
          inputs.emplace_back(NodeDefBuilder::NodeOut(
              in_edge->src()->name(), in_edge->src_output(), dtype));
          in_arg_types.push_back(dtype);
          inserted = true;
        }
      }
    }
  }
  builder.Attr("Tin", in_arg_types);

  DataTypeVector out_type;
  for (const Node* merge : merges_) {
    DataType dtype = merge->output_type(0);
    out_type.push_back(dtype);
  }
  builder.Attr("Tout", out_type);
  VLOG(3) << "Build output type: " << DataTypeVectorString(out_type);

  builder.Attr("Tcond", DT_BOOL);
  string outside_compilation;
  if (GetNodeAttr(predicate_.node->def(), kXlaOutsideCompilationAttrName,
                  &outside_compilation)
          .ok()) {
    builder.Attr(kXlaOutsideCompilationAttrName, outside_compilation);
  }
  builder.Device(predicate_.node->assigned_device_name());
  // Conditional should be the first input ...
  builder.Input(NodeDefBuilder::NodeOut(predicate_.node->name(),
                                        predicate_.index,
                                        predicate_.node->output_type(0)));
  // ... followed by the other inputs.
  builder.Input(inputs);

  VLOG(3) << "Build If node";
  NodeDef if_def;
  TF_RETURN_IF_ERROR(builder.Finalize(&if_def));
  TF_ASSIGN_OR_RETURN(if_node_,
                      parent_->AddIfNode(if_def, *merges_.begin(), predicate_));

  return Status::OK();
}

Status Conditional::AddInputEdges(
    Graph* graph,
    const std::unordered_map<Node*, OutputTensor>& merge_to_replacement) {
  VLOG(2) << "AddInputEdges for " << if_node_->name();
  int index = 0;
  // Add predicate input.
  if (predicate_.node->IsMerge()) {
    // If the predicate is a Merge node, we should not use Merge output as
    // predicate. Instead, we should use the corresponding If output in
    // 'merge_to_replacement'. Otherwise, this Conditional's If node is still
    // connected to the predicate Merge node; and when we call
    // DeleteReachableAndDeadNodes(), the predicate Merge node and this
    // Conditional's If node will be removed.
    auto iter = merge_to_replacement.find(predicate_.node);
    if (iter == merge_to_replacement.end()) {
      return errors::Internal("Cannot find replacement for Merge node ",
                              predicate_.node->name());
    }
    graph->AddEdge(iter->second.node, iter->second.index, if_node_, index++);
  } else {
    graph->AddEdge(const_cast<Node*>(predicate_.node), predicate_.index,
                   if_node_, index++);
  }
  // Add function body inputs.
  for (auto& arg : cond_arg_nodes_) {
    if (arg.src_output == Graph::kControlSlot) {
      graph->AddControlEdge(arg.src, if_node_);
    } else {
      graph->AddEdge(arg.src, arg.src_output, if_node_, index++);
    }
  }
  for (Node* n : external_control_inputs_) {
    graph->AddControlEdge(n, if_node_);
  }
  return Status::OK();
}

Status Conditional::AddOutputEdges(
    Graph* graph,
    std::unordered_map<Node*, OutputTensor>* merge_to_replacement) {
  VLOG(2) << "AddOutputEdges for " << if_node_->name();
  int i = 0;
  for (Node* node : merges_) {
    TF_RETURN_IF_ERROR(parent_->AddIdentityNode(node, if_node_, i));
    std::vector<const Edge*> edges(node->out_edges().begin(),
                                   node->out_edges().end());
    for (const Edge* edge : edges) {
      Node* dst = edge->dst();
      int dst_input = edge->dst_input();
      if (edge->src_output() > 0) {
        return errors::Unimplemented("Output of index (", edge->src_output(),
                                     ") of merge node ",
                                     FormatNodeForError(*node));
      }

      bool control_edge = edge->IsControlEdge();
      graph->RemoveEdge(edge);
      if (control_edge) {
        graph->AddControlEdge(if_node_, dst);
      } else {
        graph->AddEdge(if_node_, i, dst, dst_input);
      }
    }

    // Record corresponding output tensor in 'merge_to_replacement'.
    (*merge_to_replacement)[node] = OutputTensor{if_node_, i};

    ++i;
  }
  for (Node* n : external_control_outputs_) {
    graph->AddControlEdge(if_node_, n);
  }

  return Status::OK();
}

Status Conditional::BuildAndReplace(
    Graph* graph, FunctionLibraryDefinition* library,
    std::unordered_map<Node*, OutputTensor>* merge_to_replacement) {
  VLOG(1) << "Build If and replace merge nodes "
          << NodesToString(this->merges_);
  if (replaced_) return Status::OK();

  TF_RETURN_IF_ERROR(ExtractBodies(graph));
  TF_RETURN_IF_ERROR(BuildArgumentNodes());

  if (VLOG_IS_ON(3)) {
    LOG(INFO) << "Extracted bodies:";
    for (auto branch : {BranchType::kElseBranch, BranchType::kThenBranch}) {
      int branch_index = static_cast<int>(branch);
      auto output = bodies_[branch_index].get();
      LOG(INFO) << Branch_Name(branch) << ": "
                << DebugString(output->ToGraphDefDebug());
    }
  }

  TF_RETURN_IF_ERROR(BuildIfNode(graph, library));
  TF_RETURN_IF_ERROR(AddInputEdges(graph, *merge_to_replacement));
  TF_RETURN_IF_ERROR(AddOutputEdges(graph, merge_to_replacement));
  TF_RETURN_IF_ERROR(parent_->PropagateUpdatedState(if_node_));

  // Check that the if_node doesn't feed into itself.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      CheckNodeNotInCycle(if_node_, graph->num_node_ids()),
      "Converting to If failed.");

  replaced_ = true;
  return Status::OK();
}

string Conditional::name() const {
  CHECK(!merges_.empty());
  return absl::StrCat((*merges_.begin())->name(), "_if");
}

Status FunctionalizeCond::AddIdentityNode(const Node* replacee, Node* if_node,
                                          int port) {
  Node* id;
  TF_RETURN_IF_ERROR(NodeBuilder(replacee->name(), "Identity")
                         .Input(if_node, port)
                         .Finalize(graph_, &id));
  state_map_.ResetCondId(id, state_map_.LookupCondId(if_node));
  state_map_.ResetAncestorId(id, state_map_.LookupAncestorId(if_node));
  return Status::OK();
}

StatusOr<Node*> FunctionalizeCond::AddIfNode(const NodeDef& def,
                                             const Node* replacee,
                                             const OutputTensor& predicate) {
  Status status;
  Node* ret = graph_->AddNode(def, &status);
  TF_RETURN_IF_ERROR(status);
  VLOG(1) << "Adding If for " << replacee->name();
  StateMap::CondId id = state_map_.LookupCondId(replacee);
  if (id) {
    StateMap::CondState state = *id;
    state.erase(predicate);
    state_map_.ResetCondId(ret, state_map_.GetCondId(state));
  } else {
    state_map_.ResetCondId(ret, nullptr);
  }

  state_map_.ResetAncestorId(ret, state_map_.LookupAncestorId(replacee));

  return ret;
}

Status FunctionalizeCond::PropagateUpdatedState(const Node* replacee) {
  VLOG(2) << "Propagating update state for " << replacee->name() << " "
          << state_map_.CondStateToString(replacee);
  // Redo topological sort as the order could have changed.
  // TODO(jpienaar): The original topological order could also be updated
  // dynamically if needed.
  std::vector<Node*> rev_topo_order;
  GetPostOrder(*graph_, &rev_topo_order);

  // All the outputs of the new node could potentially be updated.
  std::unordered_set<Node*> changed;
  for (auto n : replacee->out_nodes())
    if (n->IsOp()) changed.insert(n);

  // Iterate through the changed/possible changed nodes in topological order.
  for (auto it = rev_topo_order.rbegin();
       it != rev_topo_order.rend() && !changed.empty(); ++it) {
    if (changed.find(*it) != changed.end()) {
      // Update the node state.
      Node* n = *it;
      StateMap::CondId old_state = state_map_.LookupCondId(n);
      state_map_.ResetCondId(n, nullptr);
      TF_RETURN_IF_ERROR(DetermineCondState(n));
      if (state_map_.LookupCondId(n) != old_state) {
        for (auto out : n->out_nodes())
          if (out->IsOp()) changed.insert(out);
      }
      changed.erase(n);
    }
  }
  return Status::OK();
}

// Returns the most restrictive branch of two branches or neither. This is the
// meet operator of the BranchType lattice.
BranchType MeetBranch(const BranchType& lhs, const BranchType& rhs) {
  if (lhs == rhs) return lhs;
  if (lhs == BranchType::kNeither) return rhs;
  if (rhs == BranchType::kNeither) return lhs;
  if (lhs == BranchType::kBoth) return rhs;
  if (rhs == BranchType::kBoth) return lhs;
  return BranchType::kNeither;
}

BranchType StateMap::FindBranchOf(CondId id, OutputTensor predicate) const {
  if (IsEmpty(id)) return BranchType::kNeither;
  const CondState& nodes = *id;
  auto it = nodes.find(predicate);
  if (it == nodes.end()) return BranchType::kNeither;
  return it->second;
}

StatusOr<StateMap::CondId> FunctionalizeCond::JoinCondStatesNonMerge(
    StateMap::CondId src, StateMap::CondId dst) {
  VLOG(5) << "Joining src=" << DebugString(src) << " [" << src
          << "] and dst=" << DebugString(dst) << " [" << dst << "]";

  if (state_map_.IsEmpty(dst) || state_map_.IsDead(src)) return src;
  if (state_map_.IsDead(dst) || state_map_.IsEmpty(src)) return dst;

  // Nothing to do if the CondState is the same.
  if (src == dst) return src;

  StateMap::CondState both = *src;
  for (const auto& kv : *dst) {
    auto it = both.find(kv.first);
    if (it == both.end()) {
      both.insert(kv);
    } else {
      if (it->second != kv.second) {
        return errors::InvalidArgument(
            "Graph contains node with inputs predicated on incompatible "
            "predicates: ",
            DebugString(src), " and ", DebugString(dst));
      }
    }
  }
  return state_map_.GetCondId(both);
}

StatusOr<StateMap::CondId> FunctionalizeCond::JoinCondStatesMerge(
    Node* merge, StateMap::CondId src, StateMap::CondId dst) {
  // Determine the flow state when joining two states for a merge
  // node. Combining the two states for a merge node is effectively performing a
  // disjunction of the states along the different input edges. For a merge that
  // can be transformed into a If the two inputs paths have to have a predicate
  // on which they differ (e.g., along one edge predicate `p` has to hold while
  // on another it should not). This function first determines this predicate
  // and then the resultant state is the common path between the two inputs
  // followed by s(p, both).
  VLOG(4) << "Joining (for merge) " << DebugString(src) << " and "
          << DebugString(dst);
  if (state_map_.IsEmpty(dst)) return src;
  if (state_map_.IsEmpty(src)) {
    return errors::Internal("Merge node ", merge->name(),
                            " has input that's not in any CondContext.");
  }

  if (state_map_.IsDead(src)) return src;
  if (state_map_.IsDead(dst)) return dst;

  std::vector<StateMap::CondState::value_type> diff;
  StateMap::CondState merged;
  std::set_symmetric_difference(src->begin(), src->end(), dst->begin(),
                                dst->end(), std::back_inserter(diff),
                                CondStateLess());
  std::set_intersection(src->begin(), src->end(), dst->begin(), dst->end(),
                        std::inserter(merged, merged.begin()), CondStateLess());

  // Update mapping from merge node to predicate.
  if (diff.size() == 2) {
    auto pred = diff[0].first;
    bool different_branches = (diff[0].second != diff[1].second) &&
                              (diff[0].second == BranchType::kThenBranch ||
                               diff[0].second == BranchType::kElseBranch) &&
                              (diff[1].second == BranchType::kThenBranch ||
                               diff[1].second == BranchType::kElseBranch);
    if (!(pred == diff[1].first) || !different_branches)
      return errors::InvalidArgument(
          "Unable to determine predicate for merge node");
    merge_to_predicate_[merge] = pred;
  } else {
    return errors::InvalidArgument(
        "Merge of two inputs that differ on more than one predicate ",
        DebugString(src), " and ", DebugString(dst));
  }

  return state_map_.GetCondId(merged);
}

StateMap::CondId FunctionalizeCond::StateAlongEdge(const Edge* e) {
  Node* src = e->src();
  StateMap::CondId id = state_map_.LookupCondId(e->src());

  // Dead nodes only propagate dead state.
  if (state_map_.IsDead(id)) return id;

  if (IsSwitch(src)) {
    StateMap::CondState state;
    if (id != nullptr) state = *id;
    OutputTensor predicate;
    TF_CHECK_OK(GetSwitchPredicate(*src, &predicate));
    if (!e->IsControlEdge()) {
      state[predicate] = BranchType(e->src_output());
    }
    return state_map_.GetCondId(state);
  }
  return id;
}

Status FunctionalizeCond::DetermineCondStateMerge(Node* dst) {
  // Only Merge nodes with two inputs are supported, but if this is a redundant
  // merge, then the dead edge may already have been removed (if due to a
  // switch) and so the input count would be incorrect.
  if (state_map_.IsDead(state_map_.LookupCondId(dst))) return Status::OK();

  int data_inputs = 0;
  for (auto e : dst->in_edges()) {
    Node* src = e->src();
    VLOG(5) << "Processing forward flow for merge: " << e->DebugString() << " "
            << state_map_.CondStateToString(src);
    if (!src->IsOp()) continue;
    if (!e->IsControlEdge()) ++data_inputs;

    StateMap::CondId prop = StateAlongEdge(e);
    auto id_or = JoinCondStatesMerge(dst, prop, state_map_.LookupCondId(dst));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(id_or.status(), "for node ",
                                    FormatNodeForError(*dst));
    state_map_.ResetCondId(dst, id_or.ValueOrDie());
  }

  // Incomplete Merge nodes are not supported.
  if (data_inputs != 2) {
    return errors::Unimplemented(
        dst->name(), " only has ", data_inputs,
        " inputs, while only merge nodes with two inputs supported.");
  }
  return Status::OK();
}

Status FunctionalizeCond::DetermineCondStateNonMerge(Node* dst) {
  // Handle non-merge join.
  for (auto e : dst->in_edges()) {
    VLOG(4) << "Processing forward flow for: " << e->DebugString() << " "
            << state_map_.CondStateToString(dst);
    Node* src = e->src();
    if (!src->IsOp()) continue;

    // Joining the state between the current and propagated state.
    StateMap::CondId prop = StateAlongEdge(e);
    auto id_or = JoinCondStatesNonMerge(prop, state_map_.LookupCondId(dst));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(id_or.status(), "for node ",
                                    FormatNodeForError(*dst));
    state_map_.ResetCondId(dst, id_or.ValueOrDie());
  }
  return Status::OK();
}

Status FunctionalizeCond::RemoveRedundantMerge(Node* node) {
  // Handle redundant merge nodes. A merge node is considered redundant if
  // one input edge is dead while the other has a value.
  if (!state_map_.IsDead(state_map_.LookupCondId(node))) return Status::OK();

  const Edge* non_dead_edge = nullptr;
  for (auto e : node->in_edges()) {
    if (e->IsControlEdge()) continue;
    Node* src = e->src();

    // Handle merge with dead state.
    const auto& src_id = state_map_.LookupCondId(src);
    if (!state_map_.IsDead(src_id)) {
      non_dead_edge = e;
      break;
    }
  }

  if (non_dead_edge == nullptr) {
    return errors::InvalidArgument("Merge node ", FormatNodeForError(*node),
                                   " has no non-dead inputs.");
  }
  state_map_.MarkDead(node);
  VLOG(5) << "removing redundant merge: " << node->name();
  while (!node->out_edges().empty()) {
    const Edge* oe = *node->out_edges().begin();
    Node* dst_node = oe->dst();
    int dst_port = oe->dst_input();
    graph_->RemoveEdge(oe);
    graph_->AddEdge(non_dead_edge->src(),
                    dst_port == Graph::kControlSlot
                        ? Graph::kControlSlot
                        : non_dead_edge->src_output(),
                    dst_node, dst_port);
  }
  return Status::OK();
}

Status FunctionalizeCond::RemoveRedundantSwitch(Node* node) {
  // Handle redundant switch nodes. A switch node is considered redundant if
  // the predicate of the switch already holds on the current branch. E.g., if
  // p is the predicate of the switch but p is already known to hold on this
  // branch, then the switch can be removed and the dead state propagated
  // along one. The checking of predicate is based on the exact predicate
  // (rather than boolean equivalence) and aimed at redundant switches as
  // currently generated by gradient code.
  StateMap::CondId dst_id = state_map_.LookupCondId(node);
  if (state_map_.IsDead(dst_id)) return Status::OK();

  BranchType b;
  OutputTensor pred;
  TF_RETURN_IF_ERROR(GetSwitchPredicate(*node, &pred));

  // Determine if we are already on a branch where the switch predicate is
  // true/false. Consider both the data and predicate to determine if the
  // node is redundant (skipping over identity node).
  b = state_map_.FindBranchOf(dst_id, pred);
  if (b != BranchType::kThenBranch && b != BranchType::kElseBranch) {
    OutputTensor val;
    const Edge* e;
    TF_RETURN_IF_ERROR(node->input_edge(0, &e));
    val = OutputTensor(e->src(), e->src_output());
    while (IsIdentity(val.node)) {
      TF_RETURN_IF_ERROR(val.node->input_edge(0, &e));
      val = OutputTensor(e->src(), e->src_output());
    }
    b = state_map_.FindBranchOf(dst_id, val);
    if (b != BranchType::kThenBranch && b != BranchType::kElseBranch)
      return Status::OK();
  }

  VLOG(5) << "Redundant switch " << node->name() << " " << Branch_Name(b) << " "
          << DebugString(dst_id);
  const Edge* value_edge;
  TF_RETURN_IF_ERROR(node->input_edge(0, &value_edge));
  Node* val_node = value_edge->src();
  int val_port = value_edge->src_output();
  while (!node->out_edges().empty()) {
    auto e = *node->out_edges().begin();
    Node* dst_node = e->dst();
    int dst_input = e->dst_input();
    int switch_branch = e->src_output();
    graph_->RemoveEdge(e);
    if (switch_branch == Graph::kControlSlot) {
      if (IsMerge(dst_node)) {
        auto id_or = JoinCondStatesMerge(dst_node, dst_id,
                                         state_map_.LookupCondId(dst_node));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(id_or.status(), "for node ",
                                        FormatNodeForError(*dst_node));
        state_map_.ResetCondId(dst_node, id_or.ValueOrDie());
      } else {
        auto id_or =
            JoinCondStatesNonMerge(dst_id, state_map_.LookupCondId(dst_node));
        TF_RETURN_IF_ERROR(id_or.status());
        state_map_.ResetCondId(dst_node, id_or.ValueOrDie());
      }
    } else if (BranchType(switch_branch) != b) {
      state_map_.MarkDead(dst_node);
      continue;
    }
    graph_->AddEdge(
        val_node,
        switch_branch == Graph::kControlSlot ? Graph::kControlSlot : val_port,
        dst_node, dst_input);
  }
  return Status::OK();
}

Status FunctionalizeCond::DetermineStates(std::vector<Node*> rev_topo_order) {
  // The state that is propagated along the given edge.
  for (auto it = rev_topo_order.rbegin(); it != rev_topo_order.rend(); ++it) {
    Node* dst = *it;
    TF_RETURN_IF_ERROR(DetermineCondState(dst));
    TF_RETURN_IF_ERROR(DetermineAncestorState(dst));
    if (IsSwitch(dst)) TF_RETURN_IF_ERROR(RemoveRedundantSwitch(dst));
    if (IsMerge(dst)) TF_RETURN_IF_ERROR(RemoveRedundantMerge(dst));

    VLOG(5) << dst->name() << " :: " << state_map_.CondStateToString(dst)
            << " @ " << state_map_.AncestorStateToString(dst);
    if (VLOG_IS_ON(10)) DumpGraphWithCondState("it");
  }
  return Status::OK();
}

Status FunctionalizeCond::DetermineAncestorState(Node* dst) {
  StateMap::AncestorId id = nullptr;
  StateMap::AncestorState state;

  auto insert = [&](StateMap::AncestorId id, Node* src) {
    auto other_id = state_map_.LookupAncestorId(src);
    if (other_id != id && other_id != nullptr) {
      state.insert(other_id->begin(), other_id->end());
    }
    if (IsSwitch(src) || IsMerge(src)) {
      state.insert(src);
    }
    return state_map_.GetAncestorId(state);
  };

  // Compute the union of all the switch/merge nodes that affects the input of
  // dst.
  for (auto e : dst->in_edges()) {
    Node* src = e->src();
    id = insert(id, src);
  }
  state_map_.ResetAncestorId(dst, id);
  return Status::OK();
}

void FunctionalizeCond::DeleteReachableAndDeadNodes(
    const std::vector<Node*>& merge_order) {
  // Delete all nodes that have been extracted or are reachable from
  // deleted/dead nodes. The input and outgoing edges should have already been
  // removed.
  std::deque<int> delete_nodes;
  std::vector<bool> deleted(graph_->num_node_ids(), false);
  // Don't try to delete source or sink nodes.
  deleted[graph_->kSourceId] = true;
  deleted[graph_->kSinkId] = true;

  // All remaining Switch nodes are not reachable from a Merge node and
  // removed. This is to account for dead Switch nodes.
  for (int s_id : switch_ids_) {
    Node* s = graph_->FindNodeId(s_id);
    if (s == nullptr) continue;
    for (const Edge* e : s->out_edges()) {
      // Control outputs of switch nodes (which are unconditionally executed if
      // the switch is) are not removed as they need not be part of a
      // conditional.
      if (!e->IsControlEdge()) delete_nodes.push_back(e->dst()->id());
    }
    deleted[s_id] = true;
    graph_->RemoveNode(s);
  }

  // All merge nodes should have been transformed at this point and we remove
  // them from the graph here.
  for (Node* m : merge_order) {
    for (const Edge* e : m->out_edges()) {
      // Similar to control outputs of switch nodes don't remove control
      // outputs of merge nodes.
      // TODO(jpienaar): Check cases where output edges still exist here vs
      // being removed in AddOutputEdges.
      if (!e->IsControlEdge()) delete_nodes.push_back(e->dst()->id());
    }
    deleted[m->id()] = true;
    graph_->RemoveNode(m);
  }

  // Enqueue all the dead nodes.
  for (Node* n : graph_->nodes()) {
    if (state_map_.IsDead(state_map_.LookupCondId(n))) {
      delete_nodes.push_back(n->id());
    }
  }

  while (!delete_nodes.empty()) {
    int d_id = delete_nodes.front();
    delete_nodes.pop_front();
    if (deleted[d_id]) continue;
    Node* d = graph_->FindNodeId(d_id);
    // Switch and Merge nodes could have been deleted already.
    if (d == nullptr) continue;
    for (const Edge* e : d->out_edges()) {
      delete_nodes.push_back(e->dst()->id());
    }
    deleted[d_id] = true;
    graph_->RemoveNode(d);
  }
}

void FunctionalizeCond::SortMergeNodes(std::vector<Node*>* merge_order) {
  // Sort merge nodes by nesting depth.
  using sort_pair = std::pair<int, Node*>;
  std::vector<sort_pair> inner_to_outer_merge_order;
  inner_to_outer_merge_order.reserve(merge_order->size());
  for (auto it = merge_order->rbegin(); it != merge_order->rend(); ++it) {
    Node* merge = *it;
    StateMap::CondId id = state_map_.LookupCondId(merge);
    int depth = id != nullptr ? id->size() : 0;
    inner_to_outer_merge_order.emplace_back(depth, merge);
  }
  std::stable_sort(
      inner_to_outer_merge_order.begin(), inner_to_outer_merge_order.end(),
      [](sort_pair lhs, sort_pair rhs) { return lhs.first > rhs.first; });
  merge_order->clear();
  for (sort_pair t : inner_to_outer_merge_order) {
    merge_order->push_back(t.second);
  }
}

Status FunctionalizeCond::FunctionalizeInternal() {
  // The general approach for converting a tf.cond (as lowered via switch/merge
  // nodes) to a functional if is as follows:
  // 1. Determine the topological order and collect all the switch and merge
  // nodes in the graph;
  // 2. Compute the predicates and dominance structure for all the nodes in the
  // graph - this includes which predicate must be true for a op to execute
  // (predicate values are considered directly rather than attempting to
  // determine deeper equivalence). We shall refer to this structure as the
  // CondState;
  // 3. Sort the merge nodes by nesting depth;
  // 4. Extract merge nodes together that have the same CondState and
  // AncestorState from the innermost to the outermost into IfOps;
  // Note: In the above only nodes that feed into a merge node will be
  // considered for functionalization.

  // Perform a DFS over the graph and
  // * Determine the reverse topological order of the nodes (there should be no
  //   cycles at this point so the post-order numbering corresponds to the
  //   reverse topological sorting);
  // * Record reverse topological for merge and switch nodes;
  std::vector<Node*> rev_topo_order;
  std::vector<Node*> merge_order;
  DFS(*graph_, nullptr, [&](Node* n) {
    if (IsSwitch(n)) {
      AddSwitchId(n->id());
    }
    if (IsMerge(n)) {
      merge_order.push_back(n);
    }
    if (n->IsOp()) {
      rev_topo_order.push_back(n);
    }
  });

  // No merges to functionalize.
  if (merge_order.empty()) {
    // No merges mean no switch values consumed (as only considering values
    // fetchable as output of merge);
    DeleteReachableAndDeadNodes(merge_order);
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(DetermineStates(std::move(rev_topo_order)));
  if (VLOG_IS_ON(4)) DumpGraphWithCondState("id");

  // Sort the merge nodes from innermost outwards.
  SortMergeNodes(&merge_order);

  // Cluster merge nodes by CondId and AncestorId in order of nesting.
  using ClusterPair = std::pair<StateMap::CondId, StateMap::AncestorId>;
  std::deque<std::vector<Node*>> merge_clusters;
  std::map<ClusterPair, int> merge_cluster_index;
  for (Node* merge : merge_order) {
    auto cond_id = state_map_.LookupCondId(merge);
    if (state_map_.IsDead(cond_id)) continue;

    ClusterPair key =
        std::make_pair(cond_id, state_map_.LookupAncestorId(merge));
    auto idx = merge_cluster_index.find(key);
    if (idx == merge_cluster_index.end()) {
      merge_cluster_index[key] = merge_clusters.size();
      merge_clusters.push_back({merge});
    } else {
      merge_clusters[idx->second].emplace_back(merge);
    }
  }

  // Extract the conditionals from inner most to outer most. Extracting from
  // innermost to outermost enables the extraction pass to stop once it
  // encounters a Switch node instead of having to keep track of Switch/Merge
  // nodes seen.
  for (const auto& cluster : merge_clusters) {
    // Construct a Conditional with the predicate of the merge.
    Conditional cond(merge_to_predicate_.at(cluster.front()), this,
                     &state_map_);
    for (Node* merge : cluster) TF_RETURN_IF_ERROR(cond.AddMerge(merge));
    TF_RETURN_IF_ERROR(
        cond.BuildAndReplace(graph_, library_, &merge_to_predicate_));

    if (VLOG_IS_ON(4)) DumpGraphWithCondState("after_extract");
  }

  DeleteReachableAndDeadNodes(merge_order);

  return Status::OK();
}

void FunctionalizeCond::DumpGraphWithCondState(const string& name) {
  const char* const kCondGroupDebugAttr = "_XlaFunctionalizeCondGroup";

  for (Node* n : graph_->nodes()) {
    n->ClearAttr(kCondGroupDebugAttr);
    n->AddAttr(kCondGroupDebugAttr,
               absl::StrCat(state_map_.CondStateToString(n), "_",
                            state_map_.AncestorStateToString(n)));
  }
  LOG(INFO) << "FunctionalizeControlFlow (" << name << "): "
            << dump_graph::DumpGraphToFile(
                   absl::StrCat("functionalize_cond_", name), *graph_,
                   library_);
}

void FunctionalizeCond::AddSwitchId(int switch_id) {
  switch_ids_.push_back(switch_id);
}

Status FunctionalizeCond::Functionalize(Graph* graph,
                                        FunctionLibraryDefinition* library) {
  VLOG(1) << "FunctionalizeCond::Functionalize";
  FunctionalizeCond fc(graph, library);
  return fc.FunctionalizeInternal();
}

}  // namespace functionalize_cond

Status FunctionalizeCond(Graph* graph, FunctionLibraryDefinition* library) {
  // FunctionalizeControlFlow is invoked for every function, so the loops's
  // bodies and conditionals that were extracted into functions will be handled
  // in successive invocations.
  return functionalize_cond::FunctionalizeCond::Functionalize(graph, library);
}

}  // namespace tensorflow
