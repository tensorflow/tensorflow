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

using xla::StatusOr;

namespace tensorflow {
namespace functionalize_cond {

string DebugString(const CondStateMap::CondNode& node) {
  return node.ToString();
}

// TODO(jpienaar): Move to OutputTensor.
string DebugString(const OutputTensor& tensor) {
  return strings::StrCat(tensor.node->name(), ":", tensor.index);
}

string DebugString(CondStateMap::CondId cond_state) {
  if (cond_state == nullptr || cond_state->empty()) return "[]";
  return strings::StrCat(
      "[",
      absl::StrJoin(*cond_state, ", ",
                    [](string* output, const CondStateMap::CondNode& node) {
                      strings::StrAppend(output, node.ToString());
                    }),
      "]");
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

CondStateMap::CondNode::CondNode(Type type, Node* switch_node,
                                 BranchType branch)
    : type(type), branch(branch) {
  if (type == Type::kSwitch) {
    TF_CHECK_OK(GetSwitchPredicate(*switch_node, &predicate));
  }
}

string CondStateMap::CondNode::ToString() const {
  switch (type) {
    case Type::kSwitch:
      return strings::StrCat("s(", DebugString(predicate), ",",
                             Branch_Name(branch), ")");
    case Type::kMerge:
      return "m";
    case Type::kDead:
      return "d";
  }
}

bool CondStateMap::CondNode::operator==(const CondNode& other) const {
  if (type != Type::kSwitch) return type == other.type;
  return type == other.type && predicate == other.predicate &&
         branch == other.branch;
}

bool CondStateMap::CondNode::operator!=(const CondNode& other) const {
  return !(*this == other);
}

CondStateMap::CondStateMap(Graph* graph) {
  node_to_condid_map_.resize(graph->num_node_ids());
  // Initialize the dead state (empty state is designated with a nullptr).
  dead_id_ = GetUniqueId({CondNode(CondStateMap::CondNode::Type::kDead)});
}

bool CondStateMap::IsDead(CondStateMap::CondId id) const {
  return id == dead_id_;
}

bool CondStateMap::IsEmpty(CondStateMap::CondId id) const {
  return id == nullptr;
}

size_t CondStateMap::CondHash::operator()(
    const CondStateMap::CondNode& item) const {
  return Hash64Combine(Hash64Combine(OutputTensor::Hash()(item.predicate),
                                     hash<BranchType>()(item.branch)),
                       hash<CondStateMap::CondNode::Type>()(item.type));
}

size_t CondStateMap::CondHash::operator()(
    const CondStateMap::CondState& vec) const {
  if (vec.empty()) return 0;
  size_t h = (*this)(vec.front());
  auto it = vec.begin();
  for (++it; it != vec.end(); ++it) {
    h = Hash64Combine(h, (*this)(*it));
  }
  return h;
}

// CondArgNode represents a input to the conditional and its corresponding
// switch nodes.
struct CondArgNode {
  explicit CondArgNode(Node* src, int src_output)
      : src(src), src_output(src_output) {}

  string ToString() const {
    return strings::StrCat("src=", src->name(), ":", src_output,
                           " switches=", NodesToString(switches));
  }

  Node* src;
  int src_output;
  std::array<Node*, 2> branch_copy;
  std::vector<Node*> switches;
};
using CondArgNodes = std::vector<CondArgNode>;

string DebugString(const CondArgNodes& nodes) {
  return strings::StrCat(
      "[",
      absl::StrJoin(nodes, ", ",
                    [](string* output, const CondArgNode& node) {
                      strings::StrAppend(output, node.ToString());
                    }),
      "]");
}

CondStateMap::CondId CondStateMap::LookupId(const Node* node) const {
  if (node->id() < node_to_condid_map_.size())
    return node_to_condid_map_[node->id()];
  return added_node_mapping_.at(node->id());
}

CondStateMap::CondId CondStateMap::GetUniqueId(
    const CondStateMap::CondState& state) {
  if (state.empty()) return nullptr;
  return &*condstate_set_.insert(state).first;
}

const CondStateMap::CondState& CondStateMap::LookupState(
    const Node* node) const {
  return *LookupId(node);
}

void CondStateMap::ResetId(const Node* node, CondStateMap::CondId id) {
  if (node->id() < node_to_condid_map_.size())
    node_to_condid_map_[node->id()] = id;
  else
    added_node_mapping_[node->id()] = id;
}

void CondStateMap::MarkDead(const Node* node) { ResetId(node, dead_id_); }

string CondStateMap::CondStateToString(const Node* node) const {
  return CondStateToString(LookupId(node));
}

string CondStateMap::CondStateToString(CondStateMap::CondId id) const {
  return DebugString(id);
}

FunctionalizeCond::FunctionalizeCond(Graph* graph,
                                     FunctionLibraryDefinition* library)
    : cond_state_map_(graph), library_(library), graph_(graph) {}

// Class representing the merge/switch nodes that will become a conditional.
class Conditional {
 public:
  Conditional(OutputTensor predicate, FunctionalizeCond* parent,
              CondStateMap* cond_state_map);

  // Adds merge node that is part of this conditional.
  Status AddMerge(Node* m);

  // Constructs an If node from the merge nodes.
  Status BuildAndReplace(Graph* graph, FunctionLibraryDefinition* library);

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
  Status AddInputEdges(Graph* graph);

  // Adds output edges from If node.
  Status AddOutputEdges(Graph* graph);

  // Adds switch node that is part of this conditional.
  Status AddSwitch(Node* s);

  // Internal name of conditional. The name is based on the first merge node
  // added.
  string name() const;

  // The FunctionalizeCond instance that created this.
  FunctionalizeCond* parent_;

  // Mapping between nodes and their cond state.
  CondStateMap* cond_state_map_;

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
                         CondStateMap* cond_state_map)
    : parent_(parent), cond_state_map_(cond_state_map), predicate_(predicate) {}

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
          NodeBuilder(strings::StrCat("_Arg", arg_count),
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

Status Conditional::ExtractBodies(Graph* graph) {
  VLOG(2) << "Extracting bodies for " << name();
  for (auto b : {BranchType::kElseBranch, BranchType::kThenBranch}) {
    bodies_[static_cast<int>(b)] =
        absl::make_unique<Graph>(graph->op_registry());
  }

  auto find_branch = [&](const Edge* e) {
    const auto& id = cond_state_map_->LookupId(e->src());
    return IsSwitch(e->src()) ? BranchType(e->src_output())
                              : cond_state_map_->FindBranchOf(id, predicate_);
  };

  std::array<std::vector<Node*>, 2> stacks;
  VLOG(5) << "Merges: " << NodesToString(merges_);
  for (Node* m : merges_) {
    VLOG(5) << "For merge: " << m->DebugString() << " "
            << cond_state_map_->CondStateToString(m);
    for (auto e : m->in_edges()) {
      if (e->IsControlEdge()) continue;
      BranchType branch = find_branch(e);
      TF_RET_CHECK(branch == BranchType::kThenBranch ||
                   branch == BranchType::kElseBranch)
          << "Error: " << e->src()->name()
          << " is not on either then or else branch (" << Branch_Name(branch)
          << ").";
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

        auto dst_id = cond_state_map_->LookupId(dst);
        auto src_id = cond_state_map_->LookupId(src);
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

      // Copying incomming edges to dst node.
      for (const Edge* e : n->in_edges()) {
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
        auto src_id = cond_state_map_->LookupId(src);
        auto dst_id = cond_state_map_->LookupId(dst);
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
          // not entering via a switch node. Work around this for constant
          // nodes as some constant nodes are inserted without the required
          // control context dominance.
          if (IsConstant(src)) {
            node_map.at(src->id()) = output->CopyNode(src);
          } else {
            return errors::InvalidArgument(
                "Graph contains node ", FormatNodeForError(*src),
                " that feeds into node ", FormatNodeForError(*dst),
                " but these nodes are in different control contexts (",
                DebugString(src_id), " vs ", DebugString(dst_id),
                " (detected during in edge testing)");
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
  NodeDefBuilder builder(name(), "If");
  const string branch_name[] = {"else_branch", "then_branch"};
  for (auto branch : {BranchType::kElseBranch, BranchType::kThenBranch}) {
    int branch_index = static_cast<int>(branch);
    static std::atomic<int64> sequence_num(0LL);
    int64 id = ++sequence_num;

    NameAttrList body_name;
    body_name.set_name(strings::StrCat("_functionalize_if_",
                                       branch_name[branch_index], "_", id));

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
  TF_ASSIGN_OR_RETURN(if_node_, parent_->AddIfNode(if_def, *merges_.begin()));

  return Status::OK();
}

Status Conditional::AddInputEdges(Graph* graph) {
  VLOG(2) << "AddInputEdges for " << if_node_->name();
  int index = 0;
  // Add predicate input.
  graph->AddEdge(const_cast<Node*>(predicate_.node), predicate_.index, if_node_,
                 index++);
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

Status Conditional::AddOutputEdges(Graph* graph) {
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
    ++i;
  }
  for (Node* n : external_control_outputs_) {
    graph->AddControlEdge(if_node_, n);
  }

  return Status::OK();
}

Status Conditional::BuildAndReplace(Graph* graph,
                                    FunctionLibraryDefinition* library) {
  VLOG(1) << "Build If and replace merge nodes " << name();
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
  TF_RETURN_IF_ERROR(AddInputEdges(graph));
  TF_RETURN_IF_ERROR(AddOutputEdges(graph));
  TF_RETURN_IF_ERROR(parent_->PropagateUpdatedState(if_node_));
  for (Node* m : merges_) cond_state_map_->MarkDead(m);

  // Check that the if_node doesn't feed into itself.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      CheckNodeNotInCycle(if_node_, graph->num_node_ids()),
      "Converting to If failed.");

  replaced_ = true;
  return Status::OK();
}

string Conditional::name() const {
  CHECK(!merges_.empty());
  return strings::StrCat((*merges_.begin())->name(), "_if");
}

bool CondStateMap::ScopeIn(CondStateMap::CondId id,
                           CondStateMap::CondId* scope) {
  if (id == nullptr) {
    *scope = nullptr;
    return true;
  }
  CondState state;
  for (const CondNode& node : *id) {
    if (node.type == CondNode::Type::kSwitch) {
      state.push_back(node);
    }
    if (node.type == CondNode::Type::kMerge) {
      if (state.empty()) {
        return false;
      }
      DCHECK(state.back().type == CondNode::Type::kSwitch &&
             state.back().branch == BranchType::kBoth);
      state.pop_back();
    }
  }
  *scope = GetUniqueId(state);
  return true;
}

Status FunctionalizeCond::AddIdentityNode(const Node* replacee, Node* if_node,
                                          int port) {
  Node* id;
  TF_RETURN_IF_ERROR(NodeBuilder(replacee->name(), "Identity")
                         .Input(if_node, port)
                         .Finalize(graph_, &id));
  cond_state_map_.ResetId(id, cond_state_map_.LookupId(if_node));
  return Status::OK();
}

StatusOr<Node*> FunctionalizeCond::AddIfNode(const NodeDef& def,
                                             const Node* replacee) {
  Status status;
  Node* ret = graph_->AddNode(def, &status);
  TF_RETURN_IF_ERROR(status);
  CondStateMap::CondState state = cond_state_map_.LookupState(replacee);
  state.pop_back();
  VLOG(1) << "Adding If for " << replacee->name();
  cond_state_map_.ResetId(ret, cond_state_map_.GetUniqueId(state));
  return ret;
}

Status FunctionalizeCond::PropagateUpdatedState(const Node* replacee) {
  VLOG(2) << "Propagating update state for " << replacee->name() << " "
          << cond_state_map_.CondStateToString(replacee);
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
      CondStateMap::CondId old_state = cond_state_map_.LookupId(n);
      cond_state_map_.ResetId(n, nullptr);
      TF_RETURN_IF_ERROR(DetermineCondState(n));
      if (cond_state_map_.LookupId(n) != old_state) {
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

CondStateMap::ContainsResult CondStateMap::LhsHoldsWhereverRhsHolds(
    CondStateMap::CondId lhs, CondStateMap::CondId rhs) {
  CondId lhs_scope;
  CondId rhs_scope;
  bool could_determine_scope = ScopeIn(lhs, &lhs_scope);
  could_determine_scope = could_determine_scope && ScopeIn(rhs, &rhs_scope);
  if (!could_determine_scope) return kIncomparable;

  // Returns whether a contains b.
  auto contains = [&](CondId a, CondId b) {
    // Handle empty states.
    if (a == nullptr && b != nullptr) return true;
    if (a == nullptr && b == nullptr) return true;
    if (a != nullptr && b == nullptr) return false;

    if (a->size() > b->size()) return false;
    auto a_it = a->begin();
    auto b_it = b->begin();
    while (a_it != a->end()) {
      if (*a_it != *b_it) {
        if (!(a_it->predicate == b_it->predicate)) return false;
        BranchType mb = MeetBranch(a_it->branch, b_it->branch);
        if (mb != b_it->branch) return false;
      }
      ++a_it;
      ++b_it;
    }
    return true;
  };

  bool lhs_contains_rhs = contains(lhs_scope, rhs_scope);
  bool rhs_contains_lhs = contains(rhs_scope, lhs_scope);
  if (lhs_contains_rhs && rhs_contains_lhs) return kEqual;
  if (lhs_contains_rhs) return kLhsContainsRhs;
  if (rhs_contains_lhs) return kRhsContainsLhs;
  return kIncomparable;
}

BranchType CondStateMap::FindBranchOf(CondId id, OutputTensor predicate) const {
  if (IsEmpty(id)) return BranchType::kNeither;
  absl::optional<BranchType> b;
  const CondState& nodes = *id;
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    if (it->type == CondStateMap::CondNode::Type::kSwitch &&
        it->predicate == predicate) {
      if (b.has_value()) {
        b = MeetBranch(*b, it->branch);
      } else {
        b = it->branch;
      }
      if (*b == BranchType::kNeither) {
        LOG(FATAL) << "Inconsistent state for node: " << DebugString(id);
      }
    }
  }
  return b.has_value() ? *b : BranchType::kNeither;
}

StatusOr<CondStateMap::CondId> FunctionalizeCond::JoinCondStatesNonMerge(
    CondStateMap::CondId src, CondStateMap::CondId dst) {
  VLOG(4) << "Joining src=" << DebugString(src) << " [" << src
          << "] and dst=" << DebugString(dst) << " [" << dst << "]";

  if (cond_state_map_.IsEmpty(dst) || cond_state_map_.IsDead(src)) return src;
  if (cond_state_map_.IsDead(dst)) return dst;

  // Nothing to do if the CondState is the same.
  if (src == dst) return src;

  CondStateMap::CondId src_scope;
  CondStateMap::CondId dst_scope;
  if (!cond_state_map_.ScopeIn(src, &src_scope))
    return errors::Unimplemented(
        "Predicates that must hold for node to execute are invalid! ",
        DebugString(src));
  if (!cond_state_map_.ScopeIn(dst, &dst_scope))
    return errors::Unimplemented(
        "Predicates that must hold for node to execute are invalid! ",
        DebugString(dst));

  auto result = cond_state_map_.LhsHoldsWhereverRhsHolds(src_scope, dst_scope);
  switch (result) {
    case CondStateMap::kIncomparable:
      return errors::InvalidArgument(
          "Graph contains node with inputs predicated on incompatible "
          "predicates: ",
          DebugString(src), " and ", DebugString(dst));
    case CondStateMap::kEqual:
      // If both respect the same predicates, propagate the longer constraint.
      if ((src != nullptr && dst == nullptr) ||
          (src != nullptr && dst != nullptr && src->size() > dst->size()))
        return src;
      else
        return dst;
    case CondStateMap::kLhsContainsRhs:
      // src contains dst, so dst is already more restrictive.
      return dst;
    case CondStateMap::kRhsContainsLhs:
      // dst contains src, so src is more restrictive.
      return src;
  }
}

StatusOr<CondStateMap::CondState::const_iterator>
FindThenElseSwitchForPredicate(const OutputTensor& pred,
                               CondStateMap::CondId id) {
  for (auto it = id->begin(); it != id->end(); ++it) {
    // Along every path one there can be only one instance of a then or else
    // switch for a given predicate, so return once found.
    if (it->type == CondStateMap::CondNode::Type::kSwitch &&
        it->predicate == pred &&
        (it->branch == BranchType::kThenBranch ||
         it->branch == BranchType::kElseBranch))
      return it;
  }
  return errors::Internal("Unable to find then/else branch with predicate ",
                          DebugString(pred), " for ", DebugString(id));
}

StatusOr<CondStateMap::CondId> FunctionalizeCond::JoinCondStatesMerge(
    CondStateMap::CondId src, CondStateMap::CondId dst) {
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
  if (cond_state_map_.IsEmpty(dst)) return src;

  if (cond_state_map_.IsDead(src)) return src;
  if (cond_state_map_.IsDead(dst)) return dst;

  CondStateMap::CondId src_scope;
  CondStateMap::CondId dst_scope;
  if (!cond_state_map_.ScopeIn(src, &src_scope))
    return errors::Unimplemented(
        "Predicates that must hold for node to execute are invalid! ",
        DebugString(src));
  if (!cond_state_map_.ScopeIn(dst, &dst_scope))
    return errors::Unimplemented(
        "Predicates that must hold for node to execute are invalid! ",
        DebugString(dst));

  TF_RET_CHECK(src_scope != nullptr && dst_scope != nullptr)
      << "Illegal merge inputs from outer scope: src=" << DebugString(src)
      << " dst=" << DebugString(dst);
  auto src_it = src_scope->begin();
  auto dst_it = dst_scope->begin();

  // Find branch divergent condition.
  OutputTensor pred;
  while (src_it != src_scope->end() && dst_it != dst_scope->end()) {
    if (*src_it != *dst_it) {
      VLOG(5) << "Diverges with: " << DebugString(*src_it) << " and "
              << DebugString(*dst_it);
      if (!(src_it->predicate == dst_it->predicate)) {
        return errors::InvalidArgument(
            "Unable to find common predicate which holds for one input "
            "but not the other of the merge node.");
      }
      pred = src_it->predicate;
      break;
    }
    ++src_it;
    ++dst_it;
  }

  if (pred.node == nullptr)
    return errors::InvalidArgument("Unable to determine predicate for merge.");

  TF_ASSIGN_OR_RETURN(auto div_src_it,
                      FindThenElseSwitchForPredicate(pred, src));
  TF_ASSIGN_OR_RETURN(auto div_dst_it,
                      FindThenElseSwitchForPredicate(pred, dst));
  TF_RET_CHECK(*div_src_it != *div_dst_it);

  CondStateMap::CondState result;
  // Populate result with the longest/most restrictive path up to the divergent
  // node. For example, if the one input is `[switch(pred:0, then)]` and the
  // other is `[switch(pred:0, both), merge, switch(pred:0, else)]` (as created
  // in gradient of cond test), then the resultant state here should be
  // `[switch(pred:0, both), merge, switch(pred:0, both)]`.
  if (std::distance(src->begin(), div_src_it) >
      std::distance(dst->begin(), div_dst_it)) {
    result.assign(src->begin(), std::next(div_src_it));
  } else {
    result.assign(dst->begin(), std::next(div_dst_it));
  }
  result.back().branch = BranchType::kBoth;
  return cond_state_map_.GetUniqueId(result);
}

CondStateMap::CondId FunctionalizeCond::StateAlongEdge(const Edge* e) {
  Node* src = e->src();
  CondStateMap::CondId id = cond_state_map_.LookupId(e->src());
  if (IsMerge(src)) {
    CondStateMap::CondState state;
    if (id != nullptr) state = *id;
    state.emplace_back(CondStateMap::CondNode::Type::kMerge);
    return cond_state_map_.GetUniqueId(state);
  }
  if (IsSwitch(src)) {
    CondStateMap::CondState state;
    if (id != nullptr) state = *id;
    if (e->IsControlEdge()) {
      state.emplace_back(CondStateMap::CondNode::Type::kSwitch, src,
                         BranchType::kBoth);
    } else {
      state.emplace_back(CondStateMap::CondNode::Type::kSwitch, src,
                         BranchType(e->src_output()));
    }
    return cond_state_map_.GetUniqueId(state);
  }
  return id;
}

Status FunctionalizeCond::DetermineCondStateMerge(Node* dst) {
  // Only Merge nodes with two inputs are supported, but if this is a redundant
  // merge, then the dead edge may already have been removed (if due to a
  // switch) and so the input count would be incorrect.
  if (cond_state_map_.IsDead(cond_state_map_.LookupId(dst)))
    return Status::OK();

  int data_inputs = 0;
  for (auto e : dst->in_edges()) {
    Node* src = e->src();
    VLOG(5) << "Processing forward flow for merge: " << e->DebugString() << " "
            << cond_state_map_.CondStateToString(src);
    if (!src->IsOp()) continue;
    if (!e->IsControlEdge()) ++data_inputs;

    CondStateMap::CondId prop = StateAlongEdge(e);
    auto id_or = JoinCondStatesMerge(prop, cond_state_map_.LookupId(dst));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(id_or.status(), "for node ",
                                    FormatNodeForError(*dst));
    cond_state_map_.ResetId(dst, id_or.ValueOrDie());
  }

  // Incomplete Merge nodes are not supported.
  if (data_inputs != 2) {
    return errors::Unimplemented(
        dst->name(), " only has ", data_inputs,
        " inputs, while only merge nodes with two inputs supported.");
  }
  return Status::OK();
}

Status FunctionalizeCond::DetermineCondState(Node* dst) {
  // The logic for the merge and non-merge case differ: for non-merge it is
  // the most restrictive CondState, while for merge nodes the
  // resultant state is less restrictive than either.
  if (IsMerge(dst)) {
    TF_RETURN_IF_ERROR(DetermineCondStateMerge(dst));
  } else {
    // Handle non-merge join.
    for (auto e : dst->in_edges()) {
      VLOG(5) << "Processing forward flow for: " << e->DebugString() << " "
              << cond_state_map_.CondStateToString(dst);
      Node* src = e->src();
      if (!src->IsOp()) continue;

      // Joining the state between the current and propagated state.
      CondStateMap::CondId prop = StateAlongEdge(e);
      auto id_or = JoinCondStatesNonMerge(prop, cond_state_map_.LookupId(dst));
      TF_RETURN_WITH_CONTEXT_IF_ERROR(id_or.status(), "for node ",
                                      FormatNodeForError(*dst));
      cond_state_map_.ResetId(dst, id_or.ValueOrDie());
    }
  }
  return Status::OK();
}

Status FunctionalizeCond::RemoveRedundantMerge(Node* node) {
  // Handle redundant merge nodes. A merge node is considered redundant if
  // one input edge is dead while the other has a value.
  if (!cond_state_map_.IsDead(cond_state_map_.LookupId(node)))
    return Status::OK();

  const Edge* non_dead_edge = nullptr;
  for (auto e : node->in_edges()) {
    if (e->IsControlEdge()) continue;
    Node* src = e->src();

    // Handle merge with dead state.
    const auto& src_id = cond_state_map_.LookupId(src);
    if (!cond_state_map_.IsDead(src_id)) {
      non_dead_edge = e;
      break;
    }
  }

  if (non_dead_edge == nullptr) {
    return errors::InvalidArgument("Merge node ", FormatNodeForError(*node),
                                   " has no non-dead inputs.");
  }
  cond_state_map_.MarkDead(node);
  delete_nodes_.push_back(node->id());
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
  OutputTensor pred;
  TF_RETURN_IF_ERROR(GetSwitchPredicate(*node, &pred));
  auto dst_id = cond_state_map_.LookupId(node);
  BranchType b = cond_state_map_.FindBranchOf(dst_id, pred);
  // Determine if we are already on a branch where the switch predicate is
  // true/false.
  if (b != BranchType::kThenBranch && b != BranchType::kElseBranch)
    return Status::OK();

  VLOG(5) << "Redundant switch " << node->name();
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
        auto id_or =
            JoinCondStatesMerge(dst_id, cond_state_map_.LookupId(dst_node));
        TF_RETURN_WITH_CONTEXT_IF_ERROR(id_or.status(), "for node ",
                                        FormatNodeForError(*dst_node));
        cond_state_map_.ResetId(dst_node, id_or.ValueOrDie());
      } else {
        auto id_or =
            JoinCondStatesNonMerge(dst_id, cond_state_map_.LookupId(dst_node));
        TF_RETURN_IF_ERROR(id_or.status());
        cond_state_map_.ResetId(dst_node, id_or.ValueOrDie());
      }
    } else if (BranchType(switch_branch) != b) {
      cond_state_map_.MarkDead(dst_node);
      delete_nodes_.push_back(dst_node->id());
      continue;
    }
    graph_->AddEdge(
        val_node,
        switch_branch == Graph::kControlSlot ? Graph::kControlSlot : val_port,
        dst_node, dst_input);
  }
  return Status::OK();
}

Status FunctionalizeCond::DetermineCondStates(
    std::vector<Node*> rev_topo_order) {
  // The state that is propagated along the given edge.
  for (auto it = rev_topo_order.rbegin(); it != rev_topo_order.rend(); ++it) {
    Node* dst = *it;
    TF_RETURN_IF_ERROR(DetermineCondState(dst));
    if (IsSwitch(dst)) TF_RETURN_IF_ERROR(RemoveRedundantSwitch(dst));
    if (IsMerge(dst)) TF_RETURN_IF_ERROR(RemoveRedundantMerge(dst));

    VLOG(5) << dst->name() << " :: " << cond_state_map_.CondStateToString(dst);
  }
  return Status::OK();
}

void FunctionalizeCond::DeleteReachableNodes() {
  // Delete all nodes that have been extracted or are reachable from
  // deleted/dead nodes. The input and outgoing edges should have already been
  // removed.
  std::vector<bool> deleted(graph_->num_node_ids(), false);
  // Don't try to delete source or sink nodes.
  deleted[graph_->kSourceId] = true;
  deleted[graph_->kSinkId] = true;
  while (!delete_nodes_.empty()) {
    int d_id = delete_nodes_.front();
    delete_nodes_.pop_front();
    if (deleted[d_id]) continue;
    Node* d = graph_->FindNodeId(d_id);
    // Switch and Merge nodes could have been deleted already.
    if (d == nullptr) continue;
    for (const Edge* e : d->out_edges()) {
      delete_nodes_.push_back(e->dst()->id());
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
    CondStateMap::CondId id = cond_state_map_.LookupId(merge);
    int depth = 0;
    for (auto cond_node_it = id->begin(); cond_node_it != id->end();
         ++cond_node_it) {
      if (cond_node_it->type == CondStateMap::CondNode::Type::kSwitch &&
          (cond_node_it->branch == BranchType::kThenBranch ||
           cond_node_it->branch == BranchType::kElseBranch)) {
        ++depth;
      }
    }
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
  // 4. Extract merge nodes together that have the same CondState and whose
  // input nodes have the same state from the innermost to the outermost into
  // IfOps; Note: In the above only nodes paths that converge to a merge node
  // will be considered for removal.

  // Perform a DFS over the graph and
  // * Determine the reverse topological order of the nodes (there should be no
  //   cycles at this point so the post-order numbering corresponds to the
  //   reverse topological sorting);
  // * Record reverse topological for merge and switch nodes;
  std::vector<Node*> rev_topo_order;
  std::vector<int> switch_ids;
  std::vector<Node*> merge_order;
  DFS(*graph_, nullptr, [&](Node* n) {
    if (IsSwitch(n)) {
      switch_ids.push_back(n->id());
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
    for (auto it = switch_ids.begin(); it != switch_ids.end(); ++it) {
      graph_->RemoveNode(graph_->FindNodeId(*it));
    }
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(DetermineCondStates(std::move(rev_topo_order)));

  if (VLOG_IS_ON(4)) DumpGraphWithCondState("cond_id");

  // Sort the merge nodes from innermost outwards.
  SortMergeNodes(&merge_order);

  // Extract from innermost out.
  for (auto it = merge_order.begin(); it != merge_order.end(); ++it) {
    Node* merge = *it;
    auto id = cond_state_map_.LookupId(merge);
    if (cond_state_map_.IsDead(id)) continue;

    // Construct a Conditional with the predicate of the merge (which is the
    // last entry of the CondState for the merge) and this as parent.
    DCHECK(id->back().predicate.node != nullptr);
    Conditional cond(id->back().predicate, this, &cond_state_map_);
    TF_RETURN_IF_ERROR(cond.AddMerge(merge));

    // Find all merge nodes with the same CondId. This is done repeatedly as
    // the CondId can change due replaced conditionals. E.g., the one branch
    // could previously have had a conditional nested in it, and so would have
    // had CondState with sub-state [switch(p,b),m] (where p is some predicate),
    // post removing the nested conditional that sub-state would no longer be
    // path of the propagated state along that path.
    auto end = merge_order.end();
    for (auto merge_candidate_it = std::next(it); merge_candidate_it != end;
         ++merge_candidate_it) {
      auto merge_candidate_it_id =
          cond_state_map_.LookupId(*merge_candidate_it);
      if (merge_candidate_it_id != id) continue;
      TF_RETURN_IF_ERROR(cond.AddMerge(*merge_candidate_it));
    }

    TF_RETURN_IF_ERROR(cond.BuildAndReplace(graph_, library_));

    if (VLOG_IS_ON(4)) DumpGraphWithCondState("after_extract");
  }

  // All remaining Switch nodes are not reachable from a Merge node and
  // removed. This is to account for dead Switch nodes.
  for (int s_id : switch_ids) delete_nodes_.push_back(s_id);
  for (Node* m : merge_order) delete_nodes_.push_back(m->id());
  DeleteReachableNodes();

  return Status::OK();
}

void FunctionalizeCond::DumpGraphWithCondState(const string& name) {
  const char* const kCondGroupDebugAttr = "_XlaFunctionalizeCondGroup";

  for (Node* n : graph_->nodes()) {
    n->ClearAttr(kCondGroupDebugAttr);
    n->AddAttr(kCondGroupDebugAttr, cond_state_map_.CondStateToString(n));
  }
  LOG(INFO) << "FunctionalizeControlFlow (" << name << "): "
            << dump_graph::DumpGraphToFile(
                   strings::StrCat("functionalize_", name), *graph_, library_);
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
