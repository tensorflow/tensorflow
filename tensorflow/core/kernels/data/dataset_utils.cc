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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
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

Status HashFunctionImpl(const FunctionDefLibrary& library,
                        const FunctionDef& func, uint64* hash,
                        std::vector<std::string>* visited,
                        absl::flat_hash_map<std::string, uint64>* cache);

// Produces a hash of a attribute from an op or a function. Since attributes
// may refer to functions present in the graph, we may need to hash the function
// referred to by the attribute, and thus we need the FunctionDefLibrary.
Status HashAttrImpl(const FunctionDefLibrary& library,
                    const std::string& attr_key, const AttrValue& attr_value,
                    uint64* hash, std::vector<std::string>* visited,
                    absl::flat_hash_map<std::string, uint64>* cache) {
  uint64 attr_hash = 0;
  if (attr_value.has_func()) {
    for (const auto& func : library.function()) {
      if (func.signature().name() == attr_value.func().name()) {
        uint64 function_hash;
        TF_RETURN_IF_ERROR(
            HashFunctionImpl(library, func, &function_hash, visited, cache));
        attr_hash = Hash64CombineUnordered(
            attr_hash, Hash64(absl::StrCat(attr_key, "=", function_hash)));
        break;
      }
    }
  } else {
    attr_hash = Hash64CombineUnordered(
        attr_hash, Hash64(absl::StrCat(attr_key, "=",
                                       DeterministicProtoHash64(attr_value))));
  }

  *hash = attr_hash;
  return Status::OK();
}

// This function hashes a subgraph (rooted at node) by traversing all possible
// dependency paths from that node.
Status HashNodeImpl(const GraphDef& graph, const NodeDef& node, uint64* hash,
                    std::vector<std::string>* visited,
                    absl::flat_hash_map<std::string, uint64>* cache) {
  uint64 input_hash = 0;
  uint64 control_dep_hash = 0;

  std::string canonical_node_name = absl::StrCat("node-", node.name());
  auto it = cache->find(canonical_node_name);
  if (it != cache->end()) {
    *hash = it->second;
    return Status::OK();
  }

  uint64 op_hash = Hash64(node.op());

  // Checks to make sure we won't get stuck in an infinite loop (especially in
  // loops with control dependencies).
  for (const std::string& visited_node_name : *visited) {
    if (visited_node_name == canonical_node_name) {
      uint64 final_hash =
          Hash64Combine(DefaultDependencyLoopNodeHash(), op_hash);
      (*cache)[canonical_node_name] = final_hash;
      *hash = final_hash;
      return Status::OK();
    }
  }
  visited->push_back(canonical_node_name);

  for (int i = 0; i < node.input_size(); ++i) {
    DCHECK_GT(node.input(i).length(), 0);
    if (node.input(i)[0] == '^') {
      // TODO(frankchn): Investigate if control dependencies are necessary
      // inputs to the hash. Control dependency node names start with '^', and
      // order of appearance for the control dependencies does not matter.
      const NodeDef* node_def;
      TF_RETURN_IF_ERROR(FindNode(graph, node.input(i).substr(1), &node_def));
      uint64 node_hash;
      TF_RETURN_IF_ERROR(
          HashNodeImpl(graph, *node_def, &node_hash, visited, cache));
      control_dep_hash = Hash64CombineUnordered(control_dep_hash, node_hash);
    } else {
      // The output port is significant and is optionally delimited by a ':'
      // for non-zero ports.
      std::pair<std::string, std::string> node_spec =
          absl::StrSplit(node.input(i), absl::MaxSplits(':', 1));
      const NodeDef* node_def;
      TF_RETURN_IF_ERROR(FindNode(graph, node_spec.first, &node_def));
      uint64 node_hash;
      TF_RETURN_IF_ERROR(
          HashNodeImpl(graph, *node_def, &node_hash, visited, cache));
      uint64 port_hash = Hash64(node_spec.second);
      input_hash =
          Hash64Combine(input_hash, Hash64Combine(node_hash, port_hash));
    }
  }

  uint64 attr_hash = 0;
  for (const auto& attr : node.attr()) {
    uint64 tmp_hash;
    TF_RETURN_IF_ERROR(HashAttrImpl(graph.library(), attr.first, attr.second,
                                    &tmp_hash, visited, cache));
    attr_hash = Hash64CombineUnordered(attr_hash, tmp_hash);
  }

  uint64 device_hash = Hash64(node.device());

  uint64 final_hash = Hash64Combine(
      Hash64Combine(attr_hash, op_hash),
      Hash64Combine(device_hash, Hash64Combine(input_hash, control_dep_hash)));

  (*cache)[canonical_node_name] = final_hash;
  visited->pop_back();

  *hash = final_hash;
  return Status::OK();
}

// This function hashes a function by traversing all possible dependency paths
// from all output nodes declared by the function in its definition.
Status HashFunctionImpl(const FunctionDefLibrary& library,
                        const FunctionDef& func, uint64* hash,
                        std::vector<std::string>* visited,
                        absl::flat_hash_map<std::string, uint64>* cache) {
  std::string canonical_function_name =
      absl::StrCat("function-", func.signature().name());

  auto it = cache->find(canonical_function_name);
  if (it != cache->end()) {
    *hash = it->second;
    return Status::OK();
  }

  OpDef op = func.signature();
  ClearOpDefForHashing(&op);
  uint64 signature_hash = OpDefHash(op);

  // Checks to make sure we won't get stuck in an infinite loop (especially when
  // functions depend on other function ops as a control dependency).
  for (const std::string& visited_node_name : *visited) {
    if (visited_node_name == canonical_function_name) {
      uint64 final_hash =
          Hash64Combine(DefaultDependencyLoopFnHash(), signature_hash);
      (*cache)[canonical_function_name] = final_hash;
      *hash = final_hash;
      return Status::OK();
    }
  }
  visited->push_back(canonical_function_name);

  uint64 attr_hash = 0;
  for (const auto& attr : func.attr()) {
    uint64 tmp_hash;
    TF_RETURN_IF_ERROR(HashAttrImpl(library, attr.first, attr.second, &tmp_hash,
                                    visited, cache));
    attr_hash = Hash64CombineUnordered(attr_hash, tmp_hash);
  }

  uint64 arg_attr_hash = 0;
  for (const auto& arg_attr : func.arg_attr()) {
    for (const auto& attr : arg_attr.second.attr()) {
      uint64 tmp_hash;
      TF_RETURN_IF_ERROR(HashAttrImpl(library, attr.first, attr.second,
                                      &tmp_hash, visited, cache));
      arg_attr_hash = Hash64CombineUnordered(
          arg_attr_hash, Hash64Combine(arg_attr.first, tmp_hash));
    }
  }

  GraphDef node_graph;
  for (const auto& node : func.node_def()) {
    NodeDef* node_graph_node = node_graph.add_node();
    *node_graph_node = node;
  }
  for (const auto& input_arg : func.signature().input_arg()) {
    // We add dummy input nodes for the inputs to the function.
    NodeDef* node_graph_node = node_graph.add_node();
    node_graph_node->set_name(input_arg.name());
    node_graph_node->set_op("_Retval");
  }
  *(node_graph.mutable_library()) = library;

  // TODO(frankchn): Investigate whether we need to hash the name of the
  // return argument / control return argument or whether we can relax it and
  // hash the index (etc...)
  uint64 ret_hash = func.ret_size();
  for (const auto& ret : func.ret()) {
    std::pair<std::string, std::string> node_spec =
        absl::StrSplit(ret.second, absl::MaxSplits(':', 1));
    // For every return value, we need to hash the output node (and the subgraph
    // rooted at the output node) to ensure that the computation graph that
    // ends at the output node has not changed.
    const NodeDef* node_def;
    TF_RETURN_IF_ERROR(FindNode(node_graph, node_spec.first, &node_def));
    uint64 node_hash;
    TF_RETURN_IF_ERROR(
        HashNodeImpl(node_graph, *node_def, &node_hash, visited, cache));
    uint64 node_port_hash = Hash64(node_spec.second);

    ret_hash = Hash64CombineUnordered(
        ret_hash, Hash64Combine(Hash64(ret.first),
                                Hash64Combine(node_hash, node_port_hash)));
  }

  uint64 control_ret_hash = func.control_ret_size();
  for (const auto& ret : func.control_ret()) {
    std::pair<std::string, std::string> node_spec =
        absl::StrSplit(ret.second, absl::MaxSplits(':', 1));

    const NodeDef* node_def;
    TF_RETURN_IF_ERROR(FindNode(node_graph, node_spec.first, &node_def));
    uint64 node_hash;
    TF_RETURN_IF_ERROR(
        HashNodeImpl(node_graph, *node_def, &node_hash, visited, cache));
    uint64 node_port_hash = Hash64(node_spec.second);

    control_ret_hash = Hash64CombineUnordered(
        control_ret_hash,
        Hash64Combine(Hash64(ret.first),
                      Hash64Combine(node_hash, node_port_hash)));
  }

  uint64 final_hash = Hash64Combine(
      Hash64Combine(Hash64Combine(signature_hash, attr_hash), arg_attr_hash),
      Hash64Combine(ret_hash, control_ret_hash));
  (*cache)[canonical_function_name] = final_hash;
  visited->pop_back();

  *hash = final_hash;
  return Status::OK();
}

}  // anonymous namespace

Status AsGraphDef(OpKernelContext* ctx, const DatasetBase* dataset,
                  SerializationContext&& serialization_ctx,
                  GraphDef* graph_def) {
  if (serialization_ctx.check_external_state()) {
    TF_RETURN_IF_ERROR(dataset->CheckExternalState());
  }
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node* output_node = nullptr;
  TF_RETURN_IF_ERROR(
      db.AddInputDataset(&serialization_ctx, dataset, &output_node));
  // Insert a purely symbolic _Retval node to indicate to consumers which node
  // represents `dataset`.
  ops::UnaryOp("_Retval", output_node,
               b.opts()
                   .WithName("dataset")
                   .WithAttr("T", DT_VARIANT)
                   .WithAttr("index", 0));
  TF_RETURN_IF_ERROR(b.ToGraphDef(graph_def));
  return Status::OK();
}

Status ConnectCancellationManagers(CancellationManager* parent,
                                   CancellationManager* child,
                                   std::function<void()>* deregister_fn) {
  if (parent) {
    CancellationToken token = parent->get_cancellation_token();
    if (!parent->RegisterCallback(token, [child]() { child->StartCancel(); })) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [parent, token]() { parent->DeregisterCallback(token); };
  } else {
    VLOG(1) << "Parent cancellation manager is not set. Cancellation will "
               "not be propagated to the child cancellation manager.";
    *deregister_fn = []() {};
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
    if (expected[i] != received[i]) {
      return errors::InvalidArgument("Data type mismatch at component ", i,
                                     ": expected ", DataTypeString(expected[i]),
                                     " but got ", DataTypeString(received[i]),
                                     ".");
    }
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
    if (!expected[i].IsCompatibleWith(received[i])) {
      return errors::InvalidArgument("Incompatible shapes at component ", i,
                                     ": expected ", expected[i].DebugString(),
                                     " but got ", received[i].DebugString(),
                                     ".");
    }
  }

  return Status::OK();
}

VariantTensorDataReader::VariantTensorDataReader(
    const tensorflow::VariantTensorData* data)
    : data_(data) {
  string metadata;
  data_->get_metadata(&metadata);
  auto keys = str_util::Split(metadata, kDelimiter, str_util::SkipEmpty());
  for (size_t i = 0; i < keys.size(); ++i) {
    map_[keys[i]] = i;
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

bool VariantTensorDataReader::Contains(StringPiece key) {
  return map_.find(string(key)) != map_.end();
}

template <typename T>
Status VariantTensorDataReader::ReadScalarInternal(StringPiece key, T* val) {
  if (map_.find(string(key)) == map_.end()) {
    return errors::NotFound(key);
  }
  *val = data_->tensors(map_[string(key)]).scalar<T>()();
  return Status::OK();
}

Status VariantTensorDataReader::ReadTensorInternal(StringPiece key,
                                                   Tensor* val) {
  if (map_.find(string(key)) == map_.end()) {
    return errors::NotFound(key);
  }
  *val = data_->tensors(map_[string(key)]);
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

Status VariantTensorDataWriter::Flush() {
  string metadata;
  for (size_t i = 0; i < keys_.size(); ++i) {
    strings::StrAppend(&metadata, kDelimiter, keys_[i]);
  }
  data_->set_metadata(metadata);
  return Status::OK();
}

template <typename T>
Status VariantTensorDataWriter::WriteScalarInternal(StringPiece key,
                                                    const T& val) {
  Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
  val_t.scalar<T>()() = val;
  return WriteTensorInternal(key, val_t);
}

Status VariantTensorDataWriter::WriteTensorInternal(StringPiece key,
                                                    const Tensor& val) {
  DCHECK_EQ(key.find(kDelimiter), string::npos);
  keys_.push_back(string(key));
  *(data_->add_tensors()) = val;
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

Status HashAttr(const FunctionDefLibrary& library, const std::string& attr_key,
                const AttrValue& attr_value, uint64* hash) {
  std::vector<std::string> visited;
  absl::flat_hash_map<std::string, uint64> cache;
  return HashAttrImpl(library, attr_key, attr_value, hash, &visited, &cache);
}

Status HashFunction(const FunctionDefLibrary& library, const FunctionDef& func,
                    uint64* hash) {
  std::vector<std::string> visited;
  absl::flat_hash_map<std::string, uint64> cache;
  return HashFunctionImpl(library, func, hash, &visited, &cache);
}

Status HashNode(const GraphDef& graph, const NodeDef& node, uint64* hash) {
  std::vector<std::string> visited;
  absl::flat_hash_map<std::string, uint64> cache;
  return HashNodeImpl(graph, node, hash, &visited, &cache);
}

Status HashTensor(const Tensor& tensor, uint64* hash) {
  const tstring* s = nullptr;
  // Hash tensor type.
  *hash = Hash64CombineUnordered(*hash, tensor.dtype());
  // Hash tensor shape.
  for (int i = 0; i < tensor.shape().dims(); ++i) {
    *hash = Hash64CombineUnordered(*hash, tensor.shape().dim_size(i));
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
        *hash = Hash64CombineUnordered(*hash, Hash64(s->data(), s->size()));
      }
      break;
    default:
      *hash = Hash64(tensor.tensor_data().data(), tensor.tensor_data().size());
  }
  return Status::OK();
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

  return HashNode(graph_def, *sink, hash);
}

}  // namespace data
}  // namespace tensorflow
