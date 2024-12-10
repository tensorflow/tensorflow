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
#include "tensorflow/core/data/hash_utils.h"

#include <array>
#include <cstddef>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
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

absl::Status GetSink(const GraphDef& graph_def, const NodeDef** sink) {
  for (auto& node : graph_def.node()) {
    if (node.op() == kRetvalOp) {
      *sink = &node;
      break;
    }
  }

  if (sink == nullptr) {
    return errors::Internal("Cannot find sink node for dataset graph.");
  }
  return absl::OkStatus();
}

absl::Status ShouldIgnoreInput(const NodeDef& node, int i, bool* result) {
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
          return absl::OkStatus();
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
  return absl::OkStatus();
}

absl::Status ParseInputNodeName(absl::string_view input_name,
                                absl::string_view* node_name,
                                absl::string_view* suffix,
                                bool* is_control_input) {
  if (input_name[0] == '^') {
    *node_name = input_name.substr(1);
    *is_control_input = true;
    return absl::OkStatus();
  }
  std::pair<absl::string_view, absl::string_view> node_spec =
      absl::StrSplit(input_name, absl::MaxSplits(':', 1));
  *node_name = node_spec.first;
  *suffix = node_spec.second;
  *is_control_input = false;
  return absl::OkStatus();
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
  using NodeCache = absl::flat_hash_map<const NodeDef*, uint64>;
  using FunctionCache = absl::flat_hash_map<const FunctionDef*, uint64>;
  using AttrCache =
      absl::flat_hash_map<std::pair<const NodeDef*, bool>, uint64>;

 public:
  // `GraphHasher` does not take ownership of `graph_def`, `root_node`, or
  // `flib_def`.
  explicit GraphHasher(const GraphDef* graph, const NodeDef* root,
                       const FunctionLibraryDefinition* flib)
      : graph_(graph), root_(root), flib_(flib) {
    node_cache_ = std::make_shared<NodeCache>();
    function_cache_ = std::make_shared<FunctionCache>();
    attr_cache_ = std::make_shared<AttrCache>();
  }
  explicit GraphHasher(const GraphDef* graph, const NodeDef* root,
                       const FunctionLibraryDefinition* flib,
                       std::shared_ptr<NodeCache> node_cache,
                       std::shared_ptr<FunctionCache> function_cache,
                       std::shared_ptr<AttrCache> attr_cache)
      : graph_(graph),
        root_(root),
        flib_(flib),
        node_cache_(node_cache),
        function_cache_(function_cache),
        attr_cache_(attr_cache) {}

  absl::Status Init() {
    // Construct a map of name -> NodeDef to avoid repeated linear searches.
    absl::flat_hash_map<absl::string_view, const NodeDef*> node_def_by_name;
    node_def_by_name.reserve(graph_->node_size());
    for (const auto& node : graph_->node()) {
      auto result = node_def_by_name.emplace(node.name(), &node);
      if (TF_PREDICT_FALSE(!result.second)) {
        auto node_name_formatter =
            [](std::string* out,
               const decltype(node_def_by_name)::value_type& item) {
              absl::StrAppend(out, "'", item.first, "'");
            };
        return errors::Internal(
            "Encountered graph with duplicate node name '", node.name(),
            "' in [", absl::StrJoin(node_def_by_name, ",", node_name_formatter),
            "]");
      }
    }
    // Pre-process the graph to do a BFS and prune away cycles that might cause
    // problems.
    absl::flat_hash_set<absl::string_view> visited;
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

        absl::string_view node_name, suffix;
        bool is_control_input;
        TF_RETURN_IF_ERROR(ParseInputNodeName(node->input(i), &node_name,
                                              &suffix, &is_control_input));

        auto* input_node = gtl::FindPtrOrNull(node_def_by_name, node_name);
        if (input_node == nullptr) {
          return errors::Internal("Graph node [", node->name(), "] has input [",
                                  node_name, "] that doesn't exist in graph");
        }

        // If we've already seen this node before, skip it and don't add it to
        // the queue.
        if (visited.contains(node_name)) {
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
    return absl::OkStatus();
  }

  absl::Status HashRoot(uint64* hash) { return HashNode(root_, hash); }

  absl::Status CheckEqual(GraphHasher* that) {
    return CheckNodesEqual(root_, that, that->root_);
  }

 private:
  absl::Status HashNode(const NodeDef* node, uint64* hash) {
    auto it = node_cache_->find(node);
    if (it != node_cache_->end()) {
      *hash = it->second;
      return absl::OkStatus();
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
      if (cycle_forming_edges_.contains(edge.GetHash())) {
        TF_RETURN_IF_ERROR(
            HashNodeNonInput(input.first, /*hash_functions=*/true, &node_hash));
      } else {
        TF_RETURN_IF_ERROR(HashNode(input.first, &node_hash));
      }
      inputs_hash = Hash64Combine(
          inputs_hash, Hash64Combine(node_hash, Hash64(input.second.data(),
                                                       input.second.size())));
    }

    *hash = Hash64Combine(non_input_hash,
                          Hash64Combine(control_inputs_hash, inputs_hash));
    auto result = node_cache_->emplace(node, *hash);
    if (!result.second) {
      return errors::Internal(absl::StrCat("Computed the hash for node ",
                                           node->DebugString(), " twice!"));
    }
    return absl::OkStatus();
  }

  absl::Status CheckNodesEqual(const NodeDef* this_node, GraphHasher* that,
                               const NodeDef* that_node) {
    absl::Status s = CheckNodesEqualHelper(this_node, that, that_node);
    if (!s.ok()) {
      return errors::FailedPrecondition("Nodes ", this_node->name(), " and ",
                                        that_node->name(),
                                        " are not the same:\n", s);
    }
    return s;
  }

  absl::Status CheckNodesEqualHelper(const NodeDef* this_node,
                                     GraphHasher* that,
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
      absl::string_view this_input_suffix = this_node_inputs[i].second;
      absl::string_view that_input_suffix = that_node_inputs[i].second;
      if (this_input_suffix != that_input_suffix) {
        return errors::FailedPrecondition(
            "Node inputs ", this_input->name(), " and ", that_input->name(),
            " have different suffixes: ", this_input_suffix, " vs ",
            that_input_suffix);
      }
    }
    return absl::OkStatus();
  }

  absl::Status HashNodeNonInput(const NodeDef* node, bool hash_functions,
                                uint64* hash) {
    auto iter = attr_cache_->find(std::make_pair(node, hash_functions));
    if (iter != attr_cache_->end()) {
      *hash = iter->second;
      return absl::OkStatus();
    }
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
      // Ignore "metadata" attribute of tf.data operations.
      if (DatasetOpKernel::IsDatasetOp(reg->op_def) && attr_key == "metadata")
        continue;
      auto node_attr_iter = node->attr().find(attr_key);
      if (node_attr_iter == node->attr().end()) {
        continue;
      }
      const auto& attr_value = node_attr_iter->second;
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

    auto result =
        attr_cache_->emplace(std::make_pair(node, hash_functions), *hash);
    if (!result.second) {
      return errors::Internal(absl::StrCat(
          "Computed the hash for non-input node: ", node->DebugString(),
          " and hash function bool: ", hash_functions, "twice!"));
    }
    return absl::OkStatus();
  }

  absl::Status CheckNodesEqualNonInput(const NodeDef* this_node,
                                       GraphHasher* that,
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
      const bool this_has_attr = this_node->attr().contains(attr_key);
      const bool that_has_attr = that_node->attr().contains(attr_key);
      if (this_has_attr != that_has_attr) {
        return errors::FailedPrecondition(
            "attr with key ", attr_key, " is different for nodes ",
            this_node->name(), " and ", that_node->name(),
            ". Present in former: ", this_has_attr,
            ". Present in latter: ", that_has_attr);
      }
      if (!this_has_attr) {
        continue;
      }
      if (attr_key == kColocationAttrName ||
          attr_key == kColocationGroupPrefix) {
        continue;
      }
      const auto& this_attr = this_node->attr().at(attr_key);
      const auto& that_attr = that_node->attr().at(attr_key);
      TF_RETURN_IF_ERROR(CheckAttrsEqual(attr_key, this_attr, that, that_attr,
                                         compare_functions));
    }

    if (this_node->device() != that_node->device()) {
      return errors::FailedPrecondition(
          "Devices are different for nodes ", this_node->name(), " and ",
          that_node->name(), ": ", this_node->device(), " vs ",
          that_node->device());
    }
    return absl::OkStatus();
  }

  absl::Status HashAttr(const std::string& attr_name,
                        const AttrValue& attr_value, bool hash_functions,
                        uint64* hash) {
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
    *hash = Hash64Combine(Hash64(attr_name), value_hash);
    return absl::OkStatus();
  }

  absl::Status CheckAttrsEqual(const std::string& attr_name,
                               const AttrValue& this_attr, GraphHasher* that,
                               const AttrValue& that_attr,
                               bool compare_functions) {
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
      return absl::OkStatus();
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
      return absl::OkStatus();
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
    return absl::OkStatus();
  }

  absl::Status HashFunction(const NameAttrList& func, uint64* hash) {
    return HashFunction(func.name(), func.attr(), hash);
  }

  absl::Status HashFunction(const std::string& name, const AttrValueMap& attrs,
                            uint64* hash) {
    const FunctionDef* fdef = flib_->Find(name);
    auto it = function_cache_->find(fdef);
    if (it != function_cache_->end()) {
      *hash = it->second;
      return absl::OkStatus();
    }

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
      GraphHasher hasher(&graph_def, &ret_node->def(), flib_, node_cache_,
                         function_cache_, attr_cache_);
      TF_RETURN_IF_ERROR(hasher.Init());
      TF_RETURN_IF_ERROR(hasher.HashRoot(&ret_node_hash));
      ret_nodes_hash = Hash64Combine(ret_nodes_hash, ret_node_hash);
    }

    std::vector<const NodeDef*> control_rets;
    control_rets.reserve(fbody->control_ret_nodes.size());
    for (const auto& control_ret_node : fbody->control_ret_nodes) {
      control_rets.push_back(&control_ret_node->def());
    }
    uint64 control_ret_nodes_hash = 0;
    TF_RETURN_IF_ERROR(
        HashControlInputs(control_rets, &control_ret_nodes_hash));

    *hash = Hash64Combine(ret_nodes_hash, control_ret_nodes_hash);
    auto result = function_cache_->emplace(fdef, *hash);
    if (!result.second) {
      return errors::Internal(
          absl::StrCat("Computed the hash for function ", name, " twice!"));
    }
    return absl::OkStatus();
  }

  absl::Status CheckFunctionsEqual(const NameAttrList& this_func,
                                   GraphHasher* that,
                                   const NameAttrList& that_func) {
    return CheckFunctionsEqual(this_func.name(), this_func.attr(), that,
                               that_func.name(), that_func.attr());
  }
  absl::Status CheckFunctionsEqual(const std::string& this_name,
                                   const AttrValueMap& this_attrs,
                                   GraphHasher* that,
                                   const std::string& that_name,
                                   const AttrValueMap& that_attrs) {
    absl::Status s = CheckFunctionsEqualHelper(this_name, this_attrs, that,
                                               that_name, that_attrs);
    if (!s.ok()) {
      return errors::FailedPrecondition("Functions ", this_name, " and ",
                                        that_name, " are not the same:\n", s);
    }
    return s;
  }

  absl::Status CheckFunctionsEqualHelper(const std::string& this_name,
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
      GraphHasher this_hasher(&this_graph_def, this_root, flib_, node_cache_,
                              function_cache_, attr_cache_);
      TF_RETURN_IF_ERROR(this_hasher.Init());
      GraphHasher that_hasher(&that_graph_def, that_root, that->flib_,
                              node_cache_, function_cache_, attr_cache_);
      TF_RETURN_IF_ERROR(that_hasher.Init());
      TF_RETURN_IF_ERROR(this_hasher.CheckEqual(&that_hasher));
    }

    std::vector<const NodeDef*> this_control_rets;
    this_control_rets.reserve(this_fbody->control_ret_nodes.size());
    for (const auto& control_ret_node : this_fbody->control_ret_nodes) {
      this_control_rets.push_back(&control_ret_node->def());
    }
    std::vector<const NodeDef*> that_control_rets;
    that_control_rets.reserve(that_fbody->control_ret_nodes.size());
    for (const auto& control_ret_node : that_fbody->control_ret_nodes) {
      that_control_rets.push_back(&control_ret_node->def());
    }
    TF_RETURN_IF_ERROR(
        CheckControlInputsEqual(this_control_rets, that, that_control_rets));
    return absl::OkStatus();
  }

  absl::Status HashControlInputs(const std::vector<const NodeDef*>& inputs,
                                 uint64* hash) {
    *hash = 0;
    for (const NodeDef* input : inputs) {
      uint64 node_hash = 0;
      TF_RETURN_IF_ERROR(
          HashNodeNonInput(input, /*hash_functions=*/false, &node_hash));
      *hash = Hash64CombineUnordered(*hash, node_hash);
    }
    return absl::OkStatus();
  }

  absl::Status CheckControlInputsEqual(
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
      auto this_iter = this_hashes.find(node_hash);
      if (this_iter != this_hashes.end()) {
        this_hashes.erase(this_iter);
      } else {
        that_hashes[node_hash] = input;
      }
    }
    if (!this_hashes.empty()) {
      auto formatter = [](string* out,
                          const decltype(this_hashes)::value_type& item) {
        out->append(item.second->name());
      };
      return errors::FailedPrecondition(
          "Control dependencies are different. One node has dependencies [",
          absl::StrJoin(this_hashes, ", ", formatter),
          "], which don't match any of the other node's dependencies [",
          absl::StrJoin(that_hashes, ", ", formatter), "]");
    }
    return absl::OkStatus();
  }

 private:
  bool is_cycle_forming_edge(const NodeDef* start, const NodeDef* end) {
    EdgeRep edge(start, end);
    return cycle_forming_edges_.contains(edge.GetHash());
  }

  struct NodeRep {
    std::vector<const NodeDef*> node_control_inputs;
    std::vector<std::pair<const NodeDef*, absl::string_view>> node_inputs;
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
  std::shared_ptr<NodeCache> node_cache_;
  std::shared_ptr<FunctionCache> function_cache_;
  std::shared_ptr<AttrCache> attr_cache_;
};

}  // anonymous namespace

absl::Status HashTensor(const Tensor& tensor, uint64* hash) {
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
  return absl::OkStatus();
}

absl::Status HashNode(const GraphDef& graph, const NodeDef& node,
                      uint64* hash) {
  const FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                           graph.library());
  return HashNode(graph, node, flib_def, hash);
}

absl::Status HashNode(const GraphDef& graph, const NodeDef& node,
                      const FunctionLibraryDefinition& flib_def, uint64* hash) {
  GraphHasher hasher(&graph, &node, &flib_def);
  TF_RETURN_IF_ERROR(hasher.Init());
  return hasher.HashRoot(hash);
}

absl::Status HashGraph(const GraphDef& graph_def, uint64* hash) {
  const NodeDef* sink = nullptr;
  TF_RETURN_IF_ERROR(GetSink(graph_def, &sink));
  return HashNode(graph_def, *sink, hash);
}

absl::Status CheckGraphsEqual(const GraphDef& a, const GraphDef& b) {
  const NodeDef* sink_a;
  TF_RETURN_IF_ERROR(GetSink(a, &sink_a));
  const NodeDef* sink_b;
  TF_RETURN_IF_ERROR(GetSink(b, &sink_b));
  return CheckSubgraphsEqual(a, sink_a, b, sink_b);
}

absl::Status CheckSubgraphsEqual(const GraphDef& a, const NodeDef* node_a,
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
