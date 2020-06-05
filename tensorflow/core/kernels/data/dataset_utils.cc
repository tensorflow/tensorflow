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

#include "tensorflow/core/kernels/data/dataset_utils.h"

#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDelimiter[] = "@@";

// clang-format off
constexpr std::array<const char*, 3> kOpsWithSeed = {
    "AnonymousRandomSeedGenerator",
    "ShuffleDataset",
    "ShuffleAndRepeatDataset"
};
// clang-format on

constexpr char kSeedInputName[] = "seed";
constexpr char kSeed2InputName[] = "seed2";

template <std::size_t SIZE>
bool IsNodeOfType(const NodeDef& node,
                  const std::array<const char*, SIZE>& op_types) {
  for (const auto& type : op_types) {
    if (node.op() == type) return true;
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

uint64 DefaultDependencyLoopNodeHash() {
  static const uint64 hash = Hash64("DependencyLoopNode");
  return hash;
}

uint64 DefaultDependencyLoopFnHash() {
  static const uint64 hash = Hash64("DependencyLoopFn");
  return hash;
}

void ClearOpDefForHashing(OpDef* op) {
  op->clear_name();
  op->clear_description();
  op->clear_summary();
  for (auto& arg : *op->mutable_input_arg()) {
    arg.clear_name();
    arg.clear_description();
  }
  for (auto& arg : *op->mutable_output_arg()) {
    arg.clear_name();
    arg.clear_description();
  }
}

namespace {
Status ShouldIgnoreInput(const NodeDef& node, int i, bool* result) {
  *result = false;
  if (IsNodeOfType(node, kOpsWithSeed)) {
    const OpRegistrationData* reg;
    auto status = OpRegistry::Global()->LookUp(node.op(), &reg);

    if (status.ok()) {
      if (reg->op_def.input_arg_size() > i) {
        const std::string input_arg_name = reg->op_def.input_arg(i).name();
        if (input_arg_name == kSeedInputName ||
            input_arg_name == kSeed2InputName) {
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

// Returns true if its a control input.
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
}  // namespace

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
  explicit GraphHasher(const GraphDef* graph_def, const NodeDef* root_node,
                       const FunctionLibraryDefinition* flib_def)
      : graph_def_(graph_def), root_node_(root_node), flib_def_(flib_def) {}

  Status ComputeHash(uint64* hash) {
    TF_RETURN_IF_ERROR(Init());
    return ComputeNodeHash(root_node_, hash);
  }

 private:
  // Pre process the graph to do a BFS and prune away cycles that might cause
  // problems.
  Status Init() {
    absl::flat_hash_set<std::string> visited;
    std::queue<const NodeDef*> bfs_queue;
    bfs_queue.push(root_node_);
    while (!bfs_queue.empty()) {
      const NodeDef* node = bfs_queue.front();
      bfs_queue.pop();
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
        TF_RETURN_IF_ERROR(FindNode(*graph_def_, node_name, &input_node));

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
        }
        bfs_queue.push(input_node);
      }
      nodes_[node] = node_rep;
    }
    return Status::OK();
  }

  Status ComputeNodeHash(const NodeDef* node, uint64* hash) {
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
    TF_RETURN_IF_ERROR(ComputeNonInputNodeHash(*node, &non_input_hash));

    // Hash control inputs. We combine the hashes in an unordered fashion
    // because the order doesn't matter.
    uint64 control_inputs_hash = 0;
    for (const NodeDef* control_input : node_rep->node_control_inputs) {
      uint64 node_hash = 0;
      EdgeRep edge(node, control_input);
      // If the edge was pruned we get the non input node hash to avoid cycles.
      if (cycle_forming_edges_.find(edge.GetHash()) !=
          cycle_forming_edges_.end()) {
        TF_RETURN_IF_ERROR(ComputeNonInputNodeHash(*control_input, &node_hash));
      } else {
        TF_RETURN_IF_ERROR(ComputeNodeHash(control_input, &node_hash));
      }
      control_inputs_hash =
          Hash64CombineUnordered(control_inputs_hash, node_hash);
    }

    // Hash regular inputs. We combine them in an ordered fashion.
    uint64 inputs_hash = 0;
    for (const auto& input : node_rep->node_inputs) {
      uint64 node_hash = 0;
      EdgeRep edge(node, input.first);
      // If the edge was pruned we get the non input node hash to avoid cycles.
      if (cycle_forming_edges_.find(edge.GetHash()) !=
          cycle_forming_edges_.end()) {
        TF_RETURN_IF_ERROR(ComputeNonInputNodeHash(*input.first, &node_hash));
      } else {
        TF_RETURN_IF_ERROR(ComputeNodeHash(input.first, &node_hash));
      }
      inputs_hash = Hash64Combine(
          inputs_hash, Hash64Combine(node_hash, Hash64(input.second)));
    }

    *hash = Hash64Combine(non_input_hash,
                          Hash64Combine(control_inputs_hash, inputs_hash));
    cache_[node] = *hash;
    return Status::OK();
  }

  Status ComputeNonInputNodeHash(const NodeDef& node, uint64* hash) {
    // Hash Op.
    uint64 op_hash = Hash64(node.op());

    // Hash Attrs. We get the list of attrs from the op registry and then look
    // up their values in the NodeDef attr map. This avoids looping over
    // a map which is non-deterministic.
    uint64 attrs_hash = 0;
    const OpRegistrationData* reg;
    TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUp(node.op(), &reg));

    for (const auto& attr : reg->op_def.attr()) {
      auto attr_key = attr.name();
      if (!node.attr().contains(attr_key)) continue;
      auto attr_value = node.attr().at(attr_key);
      if (attr_key == kColocationAttrName ||
          attr_key == kColocationGroupPrefix) {
        continue;
      }
      uint64 attr_hash = 0;
      TF_RETURN_IF_ERROR(HashAttr(attr_key, attr_value, &attr_hash));
      attrs_hash = Hash64Combine(attrs_hash, attr_hash);
    }

    // Hash Device.
    uint64 device_hash = Hash64(node.device());

    *hash = Hash64Combine(op_hash, Hash64Combine(attrs_hash, device_hash));
    return Status::OK();
  }

  Status HashAttr(const std::string& attr_name, const AttrValue& attr_value,
                  uint64* hash) {
    uint64 value_hash = 0;
    if (attr_value.has_func()) {
      TF_RETURN_IF_ERROR(HashFunction(attr_value.func(), &value_hash));
    } else {
      value_hash = DeterministicProtoHash64(attr_value);
    }
    *hash = Hash64(absl::StrCat(attr_name, "=", value_hash));
    return Status::OK();
  }

  Status HashFunction(const NameAttrList& func, uint64* hash) {
    const FunctionDef* fdef = flib_def_->Find(func.name());

    // Convert to a GraphDef.
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                               flib_def_, &fbody));
    GraphDef graph_def = fbody->graph->ToGraphDefDebug();

    // For each return node, we create a new GraphHasher to compute a hash.
    // We then combine these hashes to produce the hash ordered.
    uint64 ret_nodes_hash = 0;
    for (const auto& ret_node : fbody->ret_nodes) {
      GraphHasher ret_node_hasher(&graph_def, &ret_node->def(), flib_def_);
      uint64 ret_node_hash = 0;
      TF_RETURN_IF_ERROR(ret_node_hasher.ComputeHash(&ret_node_hash));
      ret_nodes_hash = Hash64Combine(ret_nodes_hash, ret_node_hash);
    }

    // For the control ret nodes, we just take the non-input hash.
    uint64 control_ret_nodes_hash = 0;
    for (const auto& control_ret_node : fbody->control_ret_nodes) {
      uint64 control_ret_node_hash = 0;
      TF_RETURN_IF_ERROR(ComputeNonInputNodeHash(control_ret_node->def(),
                                                 &control_ret_node_hash));
      control_ret_nodes_hash =
          Hash64CombineUnordered(control_ret_nodes_hash, control_ret_node_hash);
    }

    *hash = Hash64Combine(ret_nodes_hash, control_ret_nodes_hash);
    return Status::OK();
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

  const GraphDef* const graph_def_;                  // Not owned.
  const NodeDef* const root_node_;                   // Not owned.
  const FunctionLibraryDefinition* const flib_def_;  // Not owned.
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
  GraphHasher graph_hasher(&graph, &node, &flib_def);
  return graph_hasher.ComputeHash(hash);
}

Status HashGraph(const GraphDef& graph_def, uint64* hash) {
  const NodeDef* sink = nullptr;
  for (auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      sink = &node;
      break;
    }
  }

  if (sink == nullptr) {
    return errors::Internal("Cannot find sink node for dataset graph.");
  }

  const FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                           graph_def.library());
  GraphHasher graph_hasher(&graph_def, sink, &flib_def);
  TF_RETURN_IF_ERROR(graph_hasher.ComputeHash(hash));
  return Status::OK();
}

std::pair<int64, int64> MaybeOverrideSeeds(std::pair<int64, int64> seeds) {
  if (seeds.first == 0 && seeds.second == 0) {
    return {random::New64(), random::New64()};
  }
  return seeds;
}

Status RegisterCancellationCallback(CancellationManager* cancellation_manager,
                                    std::function<void()> register_fn,
                                    std::function<void()>* deregister_fn) {
  if (cancellation_manager) {
    CancellationToken token = cancellation_manager->get_cancellation_token();
    if (!cancellation_manager->RegisterCallback(token,
                                                std::move(register_fn))) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [cancellation_manager, token]() {
      cancellation_manager->DeregisterCallback(token);
    };
  } else {
    VLOG(1) << "Cancellation manager is not set. Cancellation callback will "
               "not be registered.";
    *deregister_fn = []() {};
  }
  return Status::OK();
}

Status VerifyTypeMatch(const DataType& expected,
                       const DataType& received, int index) {
  if (expected != received) {
    return errors::InvalidArgument("Data type mismatch at component ", index,
                                   ": expected ", DataTypeString(expected),
                                   " but got ", DataTypeString(received),
                                   ".");
  }
  return Status::OK();
}

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    TF_RETURN_IF_ERROR(VerifyTypeMatch(expected[i], received[i], i));
  }
  return Status::OK();
}

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const std::vector<Tensor>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    TF_RETURN_IF_ERROR(VerifyTypeMatch(expected[i], received[i].dtype(), i));
  }
  return Status::OK();
}

Status VerifyShapeCompatible(const PartialTensorShape& expected,
                             const PartialTensorShape& received, int index) {
  if (!expected.IsCompatibleWith(received)) {
    return errors::InvalidArgument("Incompatible shapes at component ", index,
                                   ": expected ", expected.DebugString(),
                                   " but got ", received.DebugString(),
                                   ".");
  }
  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    TF_RETURN_IF_ERROR(VerifyShapeCompatible(expected[i], received[i], i));
  }

  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<Tensor>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    TF_RETURN_IF_ERROR(VerifyShapeCompatible(expected[i], received[i].shape(), i));
  }

  return Status::OK();
}

namespace {

// We assume that all keys are of the form <iterator_prefix>:<name>. We extract
// the iterator name by getting rid of everything post the final colon.
Status GetIteratorName(StringPiece key, string* name) {
  if (!str_util::StartsWith(key, data::kFullNameRandomHex)) {
    return errors::InvalidArgument("Save key: ", key,
                                   " not generated using full_name.");
  }
  std::vector<string> split_keys = str_util::Split(key, data::kPipe);
  if (split_keys.size() != 2) {
    return errors::InvalidArgument("Save key: ", key,
                                   " not generated using full_name.");
  }
  string real_key = split_keys[1];
  const int pos = real_key.rfind(kColon);
  *name = real_key.substr(0, pos);
  return Status::OK();
}

}  // namespace

VariantTensorDataReader::VariantTensorDataReader(
    const std::vector<const tensorflow::VariantTensorData*>& data) {
  for (const auto& d : data) {
    string metadata;
    d->get_metadata(&metadata);
    auto keys = str_util::Split(metadata, kDelimiter, str_util::SkipEmpty());
    const string name = keys[0];
    data_[name] = d;
    map_[name] = std::map<string, size_t>();
    for (size_t i = 1; i < keys.size(); ++i) {
      map_[name][keys[i]] = i - 1;
    }
  }
}

Status VariantTensorDataReader::ReadScalar(StringPiece key, int64* val) {
  return ReadScalarInternal(key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece key, tstring* val) {
  return ReadScalarInternal(key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece key, Tensor* val) {
  return ReadTensorInternal(key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece name, StringPiece key,
                                           int64* val) {
  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece name, StringPiece key,
                                           tstring* val) {
  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece name, StringPiece key,
                                           Tensor* val) {
  return ReadTensorInternal(name, key, val);
}

bool VariantTensorDataReader::Contains(StringPiece key) {
  string name;
  if (!GetIteratorName(key, &name).ok()) {
    return false;
  }
  return Contains(name, key);
}

bool VariantTensorDataReader::Contains(StringPiece n, StringPiece key) {
  string name(n);
  return map_[name].find(string(key)) != map_[name].end();
}

template <typename T>
Status VariantTensorDataReader::ReadScalarInternal(StringPiece key, T* val) {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadScalarInternal(name, key, val);
}

Status VariantTensorDataReader::ReadTensorInternal(StringPiece key,
                                                   Tensor* val) {
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return ReadTensorInternal(name, key, val);
}

template <typename T>
Status VariantTensorDataReader::ReadScalarInternal(StringPiece n,
                                                   StringPiece key, T* val) {
  string name(n);
  if (map_[name].find(string(key)) == map_[name].end()) {
    return errors::NotFound(key);
  }
  *val = data_[name]->tensors(map_[name][string(key)]).scalar<T>()();
  return Status::OK();
}

Status VariantTensorDataReader::ReadTensorInternal(StringPiece n,
                                                   StringPiece key,
                                                   Tensor* val) {
  string name(n);
  if (map_[name].find(string(key)) == map_[name].end()) {
    return errors::NotFound(key);
  }
  *val = data_[name]->tensors(map_[name][string(key)]);
  return Status::OK();
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key, const int64 val) {
  return WriteScalarInternal(key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key,
                                            const tstring& val) {
  return WriteScalarInternal(key, val);
}

Status VariantTensorDataWriter::WriteTensor(StringPiece key,
                                            const Tensor& val) {
  return WriteTensorInternal(key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                            const int64 val) {
  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece name, StringPiece key,
                                            const tstring& val) {
  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteTensor(StringPiece name, StringPiece key,
                                            const Tensor& val) {
  return WriteTensorInternal(name, key, val);
}

void VariantTensorDataWriter::MaybeFlush() {
  if (is_flushed_) return;
  for (auto& keys : keys_) {
    const string name = keys.first;
    string metadata = name;
    for (size_t i = 0; i < keys_[name].size(); ++i) {
      strings::StrAppend(&metadata, kDelimiter, keys_[name][i]);
    }
    data_[name]->set_metadata(metadata);
  }
  is_flushed_ = true;
}

void VariantTensorDataWriter::Reset() {
  is_flushed_ = false;
  data_.clear();
  keys_.clear();
}

void VariantTensorDataWriter::ReleaseData(
    std::vector<std::unique_ptr<VariantTensorData>>* variants) {
  MaybeFlush();
  for (auto& it : data_) {
    variants->push_back(std::move(it.second));
  }
  Reset();
}

void VariantTensorDataWriter::GetData(
    std::vector<const VariantTensorData*>* variants) {
  MaybeFlush();
  for (auto& it : data_) {
    variants->push_back(it.second.get());
  }
}

template <typename T>
Status VariantTensorDataWriter::WriteScalarInternal(StringPiece key,
                                                    const T& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteScalar after GetData or ReleaseData is called");
  }
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteScalarInternal(name, key, val);
}

Status VariantTensorDataWriter::WriteTensorInternal(StringPiece key,
                                                    const Tensor& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteTensor after GetData or ReleaseData is called");
  }
  string name;
  TF_RETURN_IF_ERROR(GetIteratorName(key, &name));
  return WriteTensorInternal(name, key, val);
}

template <typename T>
Status VariantTensorDataWriter::WriteScalarInternal(StringPiece name,
                                                    StringPiece key,
                                                    const T& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteScalar after GetData or ReleaseData is called");
  }
  Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
  val_t.scalar<T>()() = val;
  return WriteTensorInternal(name, key, val_t);
}

Status VariantTensorDataWriter::WriteTensorInternal(StringPiece n,
                                                    StringPiece key,
                                                    const Tensor& val) {
  if (is_flushed_) {
    return errors::FailedPrecondition(
        "Cannot call WriteTensor after GetData or ReleaseData is called");
  }
  DCHECK_EQ(key.find(kDelimiter), string::npos);
  string name(n);
  if (keys_.count(name) == 0) {
    keys_[name] = std::vector<string>();
  }
  keys_[name].push_back(string(key));
  if (data_.count(name) == 0) {
    data_[name] = absl::make_unique<VariantTensorData>();
    data_[name]->set_type_name("tensorflow::Iterator");
  }
  *(data_[name]->add_tensors()) = val;
  return Status::OK();
}

Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionLibraryDefinition& to_add) {
  for (const auto& fn : to_add.ListFunctionNames()) {
    if (auto found = base->Find(fn)) {
      if (!OpDefEqual(found->signature(), to_add.Find(fn)->signature())) {
        return errors::InvalidArgument("Cannot add function '", fn,
                                       "' because a different function with "
                                       "the same signature already exists.");
      }
      TF_RETURN_IF_ERROR(base->RemoveFunction(fn));
    }
  }
  return base->AddLibrary(to_add);
}

Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionDefLibrary& to_add) {
  for (const auto& fd : to_add.function()) {
    if (auto found = base->Find(fd.signature().name())) {
      if (!OpDefEqual(found->signature(), fd.signature())) {
        return errors::InvalidArgument("Cannot add function '",
                                       fd.signature().name(),
                                       "' because a different function with "
                                       "the same signature already exists.");
      }
      TF_RETURN_IF_ERROR(base->RemoveFunction(fd.signature().name()));
    }
  }
  return base->AddLibrary(to_add);
}

std::function<void(std::function<void()>)> RunnerWithMaxParallelism(
    std::function<void(std::function<void()>)> runner, int max_parallelism) {
  return std::bind(
      [max_parallelism](
          // Note: `runner` is a const reference to avoid copying it.
          const std::function<void(std::function<void()>)>& runner,
          std::function<void()> fn) {
        std::function<void()> scoped_fn = std::bind(
            [max_parallelism](const std::function<void()>& fn) {
              ScopedPerThreadMaxParallelism scope(max_parallelism);
              fn();
            },
            std::move(fn));
        runner(std::move(scoped_fn));
      },
      std::move(runner), std::placeholders::_1);
}

Status DeterminismPolicy::FromString(const std::string& s,
                                     DeterminismPolicy* out) {
  DeterminismPolicy::Type type;
  if (s == DeterminismPolicy::kDeterministic) {
    type = DeterminismPolicy::Type::kDeterministic;
  } else if (s == DeterminismPolicy::kNondeterministic) {
    type = DeterminismPolicy::Type::kNondeterministic;
  } else if (s == DeterminismPolicy::kDefault) {
    type = DeterminismPolicy::Type::kDefault;
  } else {
    return errors::InvalidArgument("Unrecognized determinism policy: ", s);
  }
  *out = DeterminismPolicy(type);
  return Status::OK();
}

DeterminismPolicy::DeterminismPolicy(bool is_deterministic) {
  if (is_deterministic) {
    determinism_ = DeterminismPolicy::Type::kDeterministic;
  } else {
    determinism_ = DeterminismPolicy::Type::kNondeterministic;
  }
}

std::string DeterminismPolicy::String() const {
  switch (determinism_) {
    case DeterminismPolicy::Type::kDeterministic:
      return DeterminismPolicy::kDeterministic;
    case DeterminismPolicy::Type::kNondeterministic:
      return DeterminismPolicy::kNondeterministic;
    case DeterminismPolicy::Type::kDefault:
      return DeterminismPolicy::kDefault;
    default:
      LOG(ERROR) << "Unrecognized determinism value";
      return "Unrecognized";
  }
}

}  // namespace data
}  // namespace tensorflow
