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
#include "tensorflow/core/kernels/data/hash_utils.h"

#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

// clang-format off
constexpr std::array<const char*, 3> kOpsWithSeed = {
    "AnonymousRandomSeedGenerator",
    "ShuffleDataset",
    "ShuffleAndRepeatDataset"
};
// clang-format on
constexpr char kSeedInputName[] = "seed";
constexpr char kSeed2InputName[] = "seed2";
constexpr char kSeedGeneratorInputName[] = "seed_generator";

template <std::size_t SIZE>
bool IsNodeOfType(const NodeDef& node,
                  const std::array<const char*, SIZE>& op_types) {
  for (const auto& type : op_types) {
    if (MatchesAnyVersion(type, node.op())) {
      return true;
    }
  }
  return false;
}

Status FindNode(const GraphDef& graph, const string& name,
                const NodeDef** result) {
  for (const auto& node : graph.node()) {
    if (node.name() == name) {
      *result = &node;
      return Status::OK();
    }
  }
  return errors::NotFound("Could not find node ", name, ".");
}

Status GetSink(const GraphDef& graph_def, const NodeDef** sink) {
  for (auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      *sink = &node;
      break;
    }
  }

  if (sink == nullptr) {
    return errors::Internal("Cannot find sink node for dataset graph.");
  }
  return Status::OK();
}

Status ShouldIgnoreInput(const NodeDef& node, int i, bool* result) {
  *result = false;
  if (IsNodeOfType(node, kOpsWithSeed)) {
    const OpRegistrationData* reg;
    auto status = OpRegistry::Global()->LookUp(node.op(), &reg);

    if (status.ok()) {
      if (reg->op_def.input_arg_size() > i) {
        const std::string input_arg_name = reg->op_def.input_arg(i).name();
        if (input_arg_name == kSeedInputName ||
            input_arg_name == kSeed2InputName ||
            input_arg_name == kSeedGeneratorInputName) {
          VLOG(2) << "Ignoring arg: " << input_arg_name
                  << " from node: " << node.name();
          *result = true;
          return Status::OK();
        }
      }
    } else if (errors::IsNotFound(status)) {
      LOG(WARNING) << "Cannot find " << node.op()
                   << " in global op registry, so cannot determine which "
                      "inputs are seeds.";
    } else {
      return status;
    }
  }
  return Status::OK();
}

Status ParseInputNodeName(const std::string& input_name, std::string* node_name,
                          std::string* suffix, bool* is_control_input) {
  if (input_name[0] == '^') {
    *node_name = input_name.substr(1);
    *is_control_input = true;
    return Status::OK();
  }
  std::pair<std::string, std::string> node_spec =
      absl::StrSplit(input_name, absl::MaxSplits(':', 1));
  *node_name = node_spec.first;
  *suffix = node_spec.second;
  *is_control_input = false;
  return Status::OK();
}

// Given a graph_def and a root_node, this class computes a fingerprint that
// tries to capture the structure of the graph rooted at the provided node.
// It does not at any point rely on the names of the nodes in the graph and
// just relies on the connections between different nodes. In the presence of
// multiple cycles in the graph, there is a non-zero possibility that two
// graphs with different structure might end up with the same fingerprint
// as in order to break cycles we prune away some edges (in a deterministic
// fashion though). Idea for this algorithm was borrowed from:
// https://stackoverflow.com/questions/11338746/directed-graphs-with-a-given-root-node-match-another-directed-graph-for-equali
class GraphHasher {
 public:
  // `GraphHasher` does not take ownership of `graph_def`, `root_node`, or
  // `flib_def`.
  explicit GraphHasher(const GraphDef* graph, const NodeDef* root,
                       const FunctionLibraryDefinition* flib)
      : graph_(graph), root_(root), flib_(flib) {}

  Status Init() {
    // Pre-process the graph to do a BFS and prune away cycles that might cause
    // problems.
    absl::flat_hash_set<std::string> visited;
    std::queue<const NodeDef*> bfs_queue;
    bfs_queue.push(root_);
    while (!bfs_queue.empty()) {
      const NodeDef* node = bfs_queue.front();
      bfs_queue.pop();
      if (visited.contains(node->name())) {
        continue;
      }
      visited.insert(node->name());
      NodeRep node_rep;
      for (int i = 0; i < node->input_size(); ++i) {
        DCHECK_GT(node->input(i).length(), 0);

        // We skip trying to take the hash of the seeds of any ops, as they
        // are irrelevant to the hash of the graph and may vary from run to run.
        bool should_ignore_input = false;
        TF_RETURN_IF_ERROR(ShouldIgnoreInput(*node, i, &should_ignore_input));
        if (should_ignore_input) continue;

        std::string node_name, suffix;
        bool is_control_input;
        TF_RETURN_IF_ERROR(ParseInputNodeName(node->input(i), &node_name,
                                              &suffix, &is_control_input));
        const NodeDef* input_node;
        TF_RETURN_IF_ERROR(FindNode(*graph_, node_name, &input_node));

        // If we've already seen this node before, skip it and don't add it to
        // the queue.
        if (visited.find(node_name) != visited.end()) {
          EdgeRep cycle_edge(node, input_node);
          cycle_forming_edges_.insert(cycle_edge.GetHash());
          continue;
        }
        if (is_control_input) {
          node_rep.node_control_inputs.push_back(input_node);
        } else {
          node_rep.node_inputs.push_back(std::make_pair(input_node, suffix));
          bfs_queue.push(input_node);
        }
      }
      nodes_[node] = node_rep;
    }
    return Status::OK();
  }

  Status HashRoot(uint64* hash) { return HashNode(root_, hash); }

  Status CheckEqual(GraphHasher* that) {
    return CheckNodesEqual(root_, that, that->root_);
  }

 private:
  Status HashNode(const NodeDef* node, uint64* hash) {
    auto it = cache_.find(node);
    if (it != cache_.end()) {
      *hash = it->second;
      return Status::OK();
    }

    NodeRep* node_rep = gtl::FindOrNull(nodes_, node);
    if (node_rep == nullptr) {
      return errors::InvalidArgument("Could not find node: ", node->name());
    }

    uint64 non_input_hash;
    TF_RETURN_IF_ERROR(
        HashNodeNonInput(node, /*hash_functions=*/true, &non_input_hash));

    uint64 control_inputs_hash;
    TF_RETURN_IF_ERROR(
        HashControlInputs(node_rep->node_control_inputs, &control_inputs_hash));

    // Hash regular inputs. We combine them in an ordered fashion.
    uint64 inputs_hash = 0;
    for (const auto& input : node_rep->node_inputs) {
      uint64 node_hash = 0;
      EdgeRep edge(node, input.first);
      // If the edge was pruned we get the non input node hash to avoid cycles.
      if (cycle_forming_edges_.find(edge.GetHash()) !=
          cycle_forming_edges_.end()) {
        TF_RETURN_IF_ERROR(
            HashNodeNonInput(input.first, /*hash_functions=*/true, &node_hash));
      } else {
        TF_RETURN_IF_ERROR(HashNode(input.first, &node_hash));
      }
      inputs_hash = Hash64Combine(
          inputs_hash, Hash64Combine(node_hash, Hash64(input.second)));
    }

    *hash = Hash64Combine(non_input_hash,
                          Hash64Combine(control_inputs_hash, inputs_hash));
    cache_[node] = *hash;
    return Status::OK();
  }

  Status CheckNodesEqual(const NodeDef* this_node, GraphHasher* that,
                         const NodeDef* that_node) {
    Status s = CheckNodesEqualHelper(this_node, that, that_node);
    if (!s.ok()) {
      return errors::FailedPrecondition("Nodes ", this_node->name(), " and ",
                                        that_node->name(),
                                        " are not the same:\n", s);
    }
    return s;
  }

  Status CheckNodesEqualHelper(const NodeDef* this_node, GraphHasher* that,
                               const NodeDef* that_node) {
    TF_RETURN_IF_ERROR(CheckNodesEqualNonInput(this_node, that, that_node,
                                               /*compare_functions=*/true));

    TF_RETURN_IF_ERROR(
        CheckControlInputsEqual(nodes_[this_node].node_control_inputs, that,
                                that->nodes_[that_node].node_control_inputs));

    auto& this_node_inputs = nodes_[this_node].node_inputs;
    auto& that_node_inputs = that->nodes_[that_node].node_inputs;
    if (this_node_inputs.size() != that_node_inputs.size()) {
      return errors::FailedPrecondition(
          "Nodes have different numbers of node inputs: ",
          this_node_inputs.size(), " vs ", that_node_inputs.size());
    }
    for (int i = 0; i < this_node_inputs.size(); ++i) {
      const NodeDef* this_input = this_node_inputs[i].first;
      const NodeDef* that_input = that_node_inputs[i].first;
      if (is_cycle_forming_edge(this_node, this_input)) {
        TF_RETURN_IF_ERROR(CheckNodesEqualNonInput(this_input, that, that_input,
                                                   /*compare_functions=*/true));
      } else {
        TF_RETURN_IF_ERROR(CheckNodesEqual(this_input, that, that_input));
      }
      std::string this_input_suffix = this_node_inputs[i].second;
      std::string that_input_suffix = that_node_inputs[i].second;
      if (this_input_suffix != that_input_suffix) {
        return errors::FailedPrecondition(
            "Node inputs ", this_input->name(), " and ", that_input->name(),
            " have different suffixes: ", this_input_suffix, " vs ",
            that_input_suffix);
      }
    }
    return Status::OK();
  }

  Status HashNodeNonInput(const NodeDef* node, bool hash_functions,
                          uint64* hash) {
    // Hash Attrs. We get the list of attrs from the op registry and then look
    // up their values in the NodeDef attr map. This avoids looping over
    // a map which is non-deterministic.
    uint64 attrs_hash = 0;
    const OpRegistrationData* reg;
    TF_RETURN_IF_ERROR(flib_->LookUp(node->op(), &reg));
    uint64 op_hash = 0;
    if (reg->is_function_op) {
      if (hash_functions) {
        TF_RETURN_IF_ERROR(HashFunction(node->op(), node->attr(), &op_hash));
      }
    } else {
      op_hash = Hash64(node->op());
    }

    for (const auto& attr : reg->op_def.attr()) {
      const auto& attr_key = attr.name();
      if (!node->attr().contains(attr_key)) continue;
      auto attr_value = node->attr().at(attr_key);
      if (attr_key == kColocationAttrName ||
          attr_key == kColocationGroupPrefix) {
        continue;
      }
      uint64 attr_hash = 0;
      TF_RETURN_IF_ERROR(
          HashAttr(attr_key, attr_value, hash_functions, &attr_hash));
      attrs_hash = Hash64Combine(attrs_hash, attr_hash);
    }

    // Hash Device.
    uint64 device_hash = Hash64(node->device());

    *hash = Hash64Combine(op_hash, Hash64Combine(attrs_hash, device_hash));
    return Status::OK();
  }

  Status CheckNodesEqualNonInput(const NodeDef* this_node, GraphHasher* that,
                                 const NodeDef* that_node,
                                 bool compare_functions) {
    // We get the list of attrs from the op registry and then look
    // up their values in the NodeDef attr map. This avoids looping over
    // a map which is non-deterministic.
    const OpRegistrationData* reg;
    TF_RETURN_IF_ERROR(flib_->LookUp(this_node->op(), &reg));
    if (reg->is_function_op) {
      if (compare_functions) {
        TF_RETURN_IF_ERROR(
            CheckFunctionsEqual(this_node->op(), this_node->attr(), that,
                                that_node->op(), that_node->attr()));
      }
    } else {
      if (this_node->op() != that_node->op()) {
        return errors::FailedPrecondition(
            "ops for nodes ", this_node->name(), " and ", that_node->name(),
            " are different: ", this_node->op(), " != ", that_node->op());
      }
    }

    for (const auto& attr : reg->op_def.attr()) {
      const auto& attr_key = attr.name();
      if (this_node->attr().contains(attr_key) !=
          that_node->attr().contains(attr_key)) {
        return errors::FailedPrecondition(
            "attr with key ", attr_key, " is different for nodes ",
            this_node->name(), " and ", that_node->name(),
            ". Present in former: ", this_node->attr().contains(attr_key),
            ". Present in latter: ", that_node->attr().contains(attr_key));
      }
      if (!this_node->attr().contains(attr_key)) continue;
      if (attr_key == kColocationAttrName ||
          attr_key == kColocationGroupPrefix) {
        continue;
      }
      auto this_attr = this_node->attr().at(attr_key);
      auto that_attr = that_node->attr().at(attr_key);
      TF_RETURN_IF_ERROR(CheckAttrsEqual(attr_key, this_attr, that, that_attr,
                                         compare_functions));
    }

    if (this_node->device() != that_node->device()) {
      return errors::FailedPrecondition(
          "Devices are different for nodes ", this_node->name(), " and ",
          that_node->name(), ": ", this_node->device(), " vs ",
          that_node->device());
    }
    return Status::OK();
  }

  Status HashAttr(const std::string& attr_name, const AttrValue& attr_value,
                  bool hash_functions, uint64* hash) {
    uint64 value_hash = 0;
    if (attr_value.has_func()) {
      if (hash_functions) {
        TF_RETURN_IF_ERROR(HashFunction(attr_value.func(), &value_hash));
      }
    } else if (attr_value.has_list() && attr_value.list().func_size() > 0) {
      if (hash_functions) {
        for (auto& func : attr_value.list().func()) {
          uint64 func_hash;
          TF_RETURN_IF_ERROR(HashFunction(func, &func_hash));
          value_hash = Hash64Combine(value_hash, func_hash);
        }
      }
    } else {
      value_hash = DeterministicProtoHash64(attr_value);
    }
    *hash = Hash64(absl::StrCat(attr_name, "=", value_hash));
    return Status::OK();
  }

  Status CheckAttrsEqual(const std::string& attr_name,
                         const AttrValue& this_attr, GraphHasher* that,
                         const AttrValue& that_attr, bool compare_functions) {
    if (this_attr.has_func() != that_attr.has_func()) {
      return errors::FailedPrecondition(
          "AttrValues are of different types: ", this_attr.DebugString(),
          " vs ", that_attr.DebugString());
    }
    if (this_attr.has_func()) {
      if (compare_functions) {
        TF_RETURN_IF_ERROR(
            CheckFunctionsEqual(this_attr.func(), that, that_attr.func()));
      }
      return Status::OK();
    }
    if (this_attr.has_list() != that_attr.has_list()) {
      return errors::FailedPrecondition(
          "AttrValues are of different types: ", this_attr.DebugString(),
          " vs ", that_attr.DebugString());
    }
    if (this_attr.has_list()) {
      if (this_attr.list().func_size() != that_attr.list().func_size()) {
        return errors::FailedPrecondition(
            "AttrValues have func lists of different sizes: ",
            this_attr.DebugString(), " vs ", that_attr.DebugString());
      }
      if (compare_functions) {
        for (int i = 0; i < this_attr.list().func_size(); ++i) {
          TF_RETURN_IF_ERROR(CheckFunctionsEqual(this_attr.list().func(i), that,
                                                 that_attr.list().func(i)));
        }
      }
      return Status::OK();
    }
    uint64 this_hash, that_hash;
    TF_RETURN_IF_ERROR(
        HashAttr(attr_name, this_attr, /*hash_functions=*/true, &this_hash));
    TF_RETURN_IF_ERROR(that->HashAttr(attr_name, that_attr,
                                      /*hash_functions=*/true, &that_hash));
    if (this_hash != that_hash) {
      return errors::FailedPrecondition(
          "AttrValues are different: ", this_attr.DebugString(), " vs ",
          that_attr.DebugString());
    }
    return Status::OK();
  }

  Status HashFunction(const NameAttrList& func, uint64* hash) {
    return HashFunction(func.name(), func.attr(), hash);
  }

  Status HashFunction(const std::string& name, const AttrValueMap& attrs,
                      uint64* hash) {
    const FunctionDef* fdef = flib_->Find(name);

    // Convert to a GraphDef.
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(
        FunctionDefToBodyHelper(*fdef, AttrSlice(&attrs), flib_, &fbody));
    GraphDef graph_def = fbody->graph->ToGraphDefDebug();

    // For each return node, we create a new GraphHasher to compute a hash.
    // We then combine these hashes to produce the hash ordered.
    uint64 ret_nodes_hash = 0;
    for (const auto& ret_node : fbody->ret_nodes) {
      uint64 ret_node_hash = 0;
      GraphHasher hasher(&graph_def, &ret_node->def(), flib_);
      TF_RETURN_IF_ERROR(hasher.Init());
      TF_RETURN_IF_ERROR(hasher.HashRoot(&ret_node_hash));
      ret_nodes_hash = Hash64Combine(ret_nodes_hash, ret_node_hash);
    }

    std::vector<const NodeDef*> control_rets;
    for (const auto& control_ret_node : fbody->control_ret_nodes) {
      control_rets.push_back(&control_ret_node->def());
    }
    uint64 control_ret_nodes_hash = 0;
    TF_RETURN_IF_ERROR(
        HashControlInputs(control_rets, &control_ret_nodes_hash));

    *hash = Hash64Combine(ret_nodes_hash, control_ret_nodes_hash);
    return Status::OK();
  }

  Status CheckFunctionsEqual(const NameAttrList& this_func, GraphHasher* that,
                             const NameAttrList& that_func) {
    return CheckFunctionsEqual(this_func.name(), this_func.attr(), that,
                               that_func.name(), that_func.attr());
  }
  Status CheckFunctionsEqual(const std::string& this_name,
                             const AttrValueMap& this_attrs, GraphHasher* that,
                             const std::string& that_name,
                             const AttrValueMap& that_attrs) {
    Status s = CheckFunctionsEqualHelper(this_name, this_attrs, that, that_name,
                                         that_attrs);
    if (!s.ok()) {
      return errors::FailedPrecondition("Functions ", this_name, " and ",
                                        that_name, " are not the same:\n", s);
    }
    return s;
  }

  Status CheckFunctionsEqualHelper(const std::string& this_name,
                                   const AttrValueMap& this_attrs,
                                   GraphHasher* that,
                                   const std::string& that_name,
                                   const AttrValueMap& that_attrs) {
    const FunctionDef* this_fdef = flib_->Find(this_name);
    const FunctionDef* that_fdef = that->flib_->Find(that_name);

    // Convert to GraphDefs.
    std::unique_ptr<FunctionBody> this_fbody;
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
        *this_fdef, AttrSlice(&this_attrs), flib_, &this_fbody));
    GraphDef this_graph_def = this_fbody->graph->ToGraphDefDebug();
    std::unique_ptr<FunctionBody> that_fbody;
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
        *that_fdef, AttrSlice(&that_attrs), that->flib_, &that_fbody));
    GraphDef that_graph_def = that_fbody->graph->ToGraphDefDebug();

    if (this_fbody->ret_nodes.size() != that_fbody->ret_nodes.size()) {
      return errors::FailedPrecondition(
          "Different numbers of ret nodes for functions ", this_name, " and ",
          that_name, ": ", this_fbody->ret_nodes.size(), " vs ",
          that_fbody->ret_nodes.size());
    }
    for (int i = 0; i < this_fbody->ret_nodes.size(); ++i) {
      const NodeDef* this_root = &this_fbody->ret_nodes[i]->def();
      const NodeDef* that_root = &that_fbody->ret_nodes[i]->def();
      GraphHasher this_hasher(&this_graph_def, this_root, flib_);
      TF_RETURN_IF_ERROR(this_hasher.Init());
      GraphHasher that_hasher(&that_graph_def, that_root, that->flib_);
      TF_RETURN_IF_ERROR(that_hasher.Init());
      TF_RETURN_IF_ERROR(this_hasher.CheckEqual(&that_hasher));
    }

    std::vector<const NodeDef*> this_control_rets;
    for (const auto& control_ret_node : this_fbody->control_ret_nodes) {
      this_control_rets.push_back(&control_ret_node->def());
    }
    std::vector<const NodeDef*> that_control_rets;
    for (const auto& control_ret_node : that_fbody->control_ret_nodes) {
      that_control_rets.push_back(&control_ret_node->def());
    }
    TF_RETURN_IF_ERROR(
        CheckControlInputsEqual(this_control_rets, that, that_control_rets));
    return Status::OK();
  }

  Status HashControlInputs(const std::vector<const NodeDef*>& inputs,
                           uint64* hash) {
    *hash = 0;
    for (const NodeDef* input : inputs) {
      uint64 node_hash = 0;
      TF_RETURN_IF_ERROR(
          HashNodeNonInput(input, /*hash_functions=*/false, &node_hash));
      *hash = Hash64CombineUnordered(*hash, node_hash);
    }
    return Status::OK();
  }

  Status CheckControlInputsEqual(
      const std::vector<const NodeDef*>& this_inputs, GraphHasher* that,
      const std::vector<const NodeDef*>& that_inputs) {
    absl::flat_hash_map<uint64, const NodeDef*> this_hashes;
    for (const NodeDef* input : this_inputs) {
      uint64 node_hash = 0;
      TF_RETURN_IF_ERROR(
          HashNodeNonInput(input, /*hash_functions=*/false, &node_hash));
      this_hashes[node_hash] = input;
    }
    absl::flat_hash_map<uint64, const NodeDef*> that_hashes;
    for (const NodeDef* input : that_inputs) {
      uint64 node_hash = 0;
      TF_RETURN_IF_ERROR(
          HashNodeNonInput(input, /*hash_functions=*/false, &node_hash));
      if (this_hashes.contains(node_hash)) {
        this_hashes.erase(node_hash);
      } else {
        that_hashes[node_hash] = input;
      }
    }
    if (!this_hashes.empty()) {
      std::vector<std::string> this_unmatched;
      for (const auto& it : this_hashes) {
        this_unmatched.push_back(it.second->name());
      }
      std::vector<std::string> that_unmatched;
      for (const auto& it : that_hashes) {
        that_unmatched.push_back(it.second->name());
      }
      return errors::FailedPrecondition(
          "Control dependencies are different. One node has dependencies [",
          absl::StrJoin(this_unmatched, ", "),
          "], which don't match any of the other node's dependencies [",
          absl::StrJoin(that_unmatched, ", "), "]");
    }
    return Status::OK();
  }

 private:
  bool is_cycle_forming_edge(const NodeDef* start, const NodeDef* end) {
    EdgeRep edge(start, end);
    return cycle_forming_edges_.contains(edge.GetHash());
  }

  struct NodeRep {
    std::vector<const NodeDef*> node_control_inputs;
    std::vector<std::pair<const NodeDef*, std::string>> node_inputs;
  };

  struct EdgeRep {
    const NodeDef* start_node;
    const NodeDef* end_node;

    EdgeRep(const NodeDef* start, const NodeDef* end)
        : start_node(start), end_node(end) {}

    uint64 GetHash() {
      return Hash64Combine(absl::Hash<const NodeDef*>()(start_node),
                           absl::Hash<const NodeDef*>()(end_node));
    }
  };
  const GraphDef* const graph_;                  // Not owned.
  const NodeDef* const root_;                    // Not owned.
  const FunctionLibraryDefinition* const flib_;  // Not owned.
  // Edges that need to be pruned as their presence will cause cycles.
  absl::flat_hash_set<uint64> cycle_forming_edges_;
  absl::flat_hash_map<const NodeDef*, NodeRep> nodes_;
  absl::flat_hash_map<const NodeDef*, uint64> cache_;
};

}  // anonymous namespace

Status HashTensor(const Tensor& tensor, uint64* hash) {
  const tstring* s = nullptr;
  // Hash tensor type.
  *hash = Hash64Combine(0, tensor.dtype());
  // Hash tensor shape.
  for (int i = 0; i < tensor.shape().dims(); ++i) {
    *hash = Hash64Combine(*hash, tensor.shape().dim_size(i));
  }
  // Hash tensor data.
  switch (tensor.dtype()) {
    case DT_RESOURCE:
    case DT_VARIANT:
      return errors::Unimplemented("Hashing ", DataTypeString(tensor.dtype()),
                                   " is not supported.");
    case DT_STRING:
      s = tensor.flat<tstring>().data();
      for (int i = 0; i < tensor.NumElements(); ++i, ++s) {
        *hash = Hash64Combine(*hash, Hash64(s->data(), s->size()));
      }
      break;
    default:
      *hash = Hash64(tensor.tensor_data().data(), tensor.tensor_data().size());
  }
  return Status::OK();
}

Status HashNode(const GraphDef& graph, const NodeDef& node, uint64* hash) {
  const FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                           graph.library());
  return HashNode(graph, node, flib_def, hash);
}

Status HashNode(const GraphDef& graph, const NodeDef& node,
                const FunctionLibraryDefinition& flib_def, uint64* hash) {
  GraphHasher hasher(&graph, &node, &flib_def);
  TF_RETURN_IF_ERROR(hasher.Init());
  return hasher.HashRoot(hash);
}

Status HashGraph(const GraphDef& graph_def, uint64* hash) {
  const NodeDef* sink = nullptr;
  TF_RETURN_IF_ERROR(GetSink(graph_def, &sink));
  return HashNode(graph_def, *sink, hash);
}

Status CheckGraphsEqual(const GraphDef& a, const GraphDef& b) {
  const NodeDef* sink_a;
  TF_RETURN_IF_ERROR(GetSink(a, &sink_a));
  const NodeDef* sink_b;
  TF_RETURN_IF_ERROR(GetSink(b, &sink_b));
  return CheckSubgraphsEqual(a, sink_a, b, sink_b);
}

Status CheckSubgraphsEqual(const GraphDef& a, const NodeDef* node_a,
                           const GraphDef& b, const NodeDef* node_b) {
  const FunctionLibraryDefinition flib_def_a(OpRegistry::Global(), a.library());
  GraphHasher hasher_a(&a, node_a, &flib_def_a);
  TF_RETURN_IF_ERROR(hasher_a.Init());

  const FunctionLibraryDefinition flib_def_b(OpRegistry::Global(), b.library());
  GraphHasher hasher_b(&b, node_b, &flib_def_b);
  TF_RETURN_IF_ERROR(hasher_b.Init());

  return hasher_a.CheckEqual(&hasher_b);
}

}  // namespace data
}  // namespace tensorflow
