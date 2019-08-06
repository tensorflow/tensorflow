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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDelimiter[] = "@@";

void AddFakeSinks(FunctionDef* function_def) {
  int counter = 0;
  for (const auto& output : function_def->signature().output_arg()) {
    NodeDef* node = function_def->add_node_def();
    tensorflow::grappler::function_utils::SetUniqueFunctionNodeName(
        strings::StrCat("FakeSink", counter++), function_def, node);
    node->set_op("Identity");
    node->add_input(function_def->ret().at(output.name()));
    (*node->mutable_attr())["T"].set_type(output.type());

    (*function_def->mutable_ret())[output.name()] =
        strings::StrCat(node->name(), ":output:0");
  }
}

void RemoveFakeSinks(FunctionDef* function_def) {
  // Map from identity node names to their input tensor strings
  std::map<string, string> identity_map;
  for (const auto& node : function_def->node_def()) {
    if (node.op() == "Identity" && node.input_size() == 1) {
      identity_map[node.name()] = node.input(0);
    }
  }
  for (const auto& output_arg : function_def->signature().output_arg()) {
    const string& tensor = function_def->ret().at(output_arg.name());
    const string& output_node = tensor.substr(0, tensor.find(':'));
    if (identity_map.find(output_node) != identity_map.end()) {
      (*function_def->mutable_ret())[output_arg.name()] =
          identity_map.at(output_node);
    }
  }
}

Status ApplyRewrites(OpKernelContext* ctx,
                     const std::function<RewriterConfig(void)> config_factory,
                     bool optimize_function_library, GraphDef* graph_def,
                     string* output_node) {
  // Add an identity node as the fetch node, otherwise we might get 'placeholder
  // is both fed and fetched' errors in some cases when using input list with
  // placeholder dataset nodes.
  NodeDef* node = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            node);
  node->set_op("Identity");
  node->add_input(*output_node);
  (*node->mutable_attr())["T"].set_type(DT_VARIANT);
  *output_node = node->name();

  // Add fake sink node to graph and functions to allow rewriting the actual
  // sink nodes.
  //
  // TODO(b/118820916): When MetaOptimizer adds provisions for function retvals
  // to be optimizable, we will no longer need this.
  for (auto& function_def : *graph_def->mutable_library()->mutable_function()) {
    AddFakeSinks(&function_def);
  }

  // Create metagraph.
  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_graph_def()) = *graph_def;

  // Grappler determines fetch ops from collection 'train_op'.
  CollectionDef collection_def;
  auto node_list = collection_def.mutable_node_list();
  node_list->add_value(*output_node);
  (*meta_graph_def.mutable_collection_def())["train_op"] = collection_def;

  // Create Grappler item.
  tensorflow::grappler::ItemConfig item_config;
  item_config.apply_optimizations = true;
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      tensorflow::grappler::GrapplerItemFromMetaGraphDef(
          "graph", meta_graph_def, item_config);
  grappler_item->optimization_options().optimize_function_library =
      optimize_function_library;
  std::unordered_map<string, tensorflow::DeviceProperties> device_map;
  tensorflow::grappler::VirtualCluster cluster(device_map);

  // Run data optimizer using grappler's meta optimizer.
  tensorflow::ConfigProto config;
  *config.mutable_graph_options()->mutable_rewrite_options() = config_factory();
  TF_RETURN_IF_ERROR(tensorflow::grappler::RunMetaOptimizer(
      *grappler_item, config, ctx->device(), &cluster, graph_def));

  // Remove fake sinks after optimizations are done.
  //
  // TODO(b/118820916): When MetaOptimizer adds provisions for function retvals
  // to be optimizable, we will no longer need this.
  for (auto& function_def : *graph_def->mutable_library()->mutable_function()) {
    RemoveFakeSinks(&function_def);
  }

  return Status::OK();
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

// forward declaration for use in HashAttr.
uint64 HashSubgraphFunctionImpl(
    const FunctionDefLibrary& library, const FunctionDef* f,
    std::vector<std::string>* visited,
    absl::flat_hash_map<std::string, uint64>* cache);

// Produces a hash of a attribute from an op or a function. Since attributes
// may refer to functions present in the graph, we may need to hash the function
// referred to by the attribute, and thus we need the FunctionDefLibrary.
uint64 HashAttr(const FunctionDefLibrary& library, const std::string& attr_key,
                const AttrValue& attr_value, std::vector<std::string>* visited,
                absl::flat_hash_map<std::string, uint64>* cache) {
  uint64 attr_hash = 0;
  if (attr_value.has_func()) {
    for (const auto& func : library.function()) {
      if (func.signature().name() == attr_value.func().name()) {
        attr_hash = Hash64CombineUnordered(
            attr_hash,
            Hash64(absl::StrCat(
                attr_key, "=",
                HashSubgraphFunctionImpl(library, &func, visited, cache))));
        break;
      }
    }
  } else {
    attr_hash = Hash64CombineUnordered(
        attr_hash, Hash64(absl::StrCat(attr_key, "=",
                                       DeterministicProtoHash64(attr_value))));
  }

  return attr_hash;
}

// This function hashes a subgraph (rooted at node) by traversing all possible
// dependency paths from that node.
uint64 HashSubgraphImpl(const grappler::GraphView& g, const NodeDef* node,
                        std::vector<std::string>* visited,
                        absl::flat_hash_map<std::string, uint64>* cache) {
  uint64 input_hash = 0;
  uint64 control_dep_hash = 0;

  std::string canonical_node_name = absl::StrCat("node-", node->name());
  auto it = cache->find(canonical_node_name);
  if (it != cache->end()) {
    return it->second;
  }

  uint64 op_hash = Hash64(node->op());

  // Checks to make sure we won't get stuck in an infinite loop (especially in
  // loops with control dependencies).
  for (const std::string& visited_node_name : *visited) {
    if (visited_node_name == canonical_node_name) {
      uint64 final_hash =
          Hash64Combine(DefaultDependencyLoopNodeHash(), op_hash);
      (*cache)[canonical_node_name] = final_hash;
      return final_hash;
    }
  }
  visited->push_back(canonical_node_name);

  for (int i = 0; i < node->input_size(); ++i) {
    DCHECK_GT(node->input(i).length(), 0);
    if (node->input(i)[0] == '^') {
      // TODO(frankchn): Investigate if control dependencies are necessary
      // inputs to the hash.
      // Control dependency node names start with '^', and order of appearance
      // for the control dependencies does not matter.
      control_dep_hash = Hash64CombineUnordered(
          control_dep_hash,
          HashSubgraphImpl(g, g.GetNode(node->input(i).substr(1)), visited,
                           cache));
    } else {
      // The output port is significant and is optionally delimited by a ':'
      // for non-zero ports.
      std::pair<std::string, std::string> node_spec =
          absl::StrSplit(node->input(i), absl::MaxSplits(':', 1));
      uint64 child_node_hash =
          HashSubgraphImpl(g, g.GetNode(node_spec.first), visited, cache);
      uint64 child_port_hash = Hash64(node_spec.second);
      input_hash = Hash64Combine(
          input_hash, Hash64Combine(child_node_hash, child_port_hash));
    }
  }

  uint64 attr_hash = 0;
  for (const auto& attr : node->attr()) {
    attr_hash = Hash64CombineUnordered(
        attr_hash, HashAttr(g.graph()->library(), attr.first, attr.second,
                            visited, cache));
  }

  uint64 device_hash = Hash64(node->device());

  uint64 final_hash = Hash64Combine(
      Hash64Combine(attr_hash, op_hash),
      Hash64Combine(device_hash, Hash64Combine(input_hash, control_dep_hash)));

  (*cache)[canonical_node_name] = final_hash;
  visited->pop_back();

  return final_hash;
}

// This function hashes a function by traversing all possible dependency paths
// from all output nodes declared by the function in its definition.
uint64 HashSubgraphFunctionImpl(
    const FunctionDefLibrary& library, const FunctionDef* f,
    std::vector<std::string>* visited,
    absl::flat_hash_map<std::string, uint64>* cache) {
  std::string canonical_function_name =
      absl::StrCat("function-", f->signature().name());

  auto it = cache->find(canonical_function_name);
  if (it != cache->end()) {
    return it->second;
  }

  OpDef op = f->signature();
  ClearOpDefForHashing(&op);
  uint64 signature_hash = OpDefHash(op);

  // Checks to make sure we won't get stuck in an infinite loop (especially when
  // functions depend on other function ops as a control dependency).
  for (const std::string& visited_node_name : *visited) {
    if (visited_node_name == canonical_function_name) {
      uint64 final_hash =
          Hash64Combine(DefaultDependencyLoopFnHash(), signature_hash);
      (*cache)[canonical_function_name] = final_hash;
      return final_hash;
    }
  }
  visited->push_back(canonical_function_name);

  uint64 attr_hash = 0;
  for (const auto& attr : f->attr()) {
    attr_hash = Hash64CombineUnordered(
        attr_hash, HashAttr(library, attr.first, attr.second, visited, cache));
  }

  uint64 arg_attr_hash = 0;
  for (const auto& arg_attr : f->arg_attr()) {
    for (const auto& attr : arg_attr.second.attr()) {
      arg_attr_hash = Hash64CombineUnordered(
          arg_attr_hash,
          Hash64Combine(arg_attr.first, HashAttr(library, attr.first,
                                                 attr.second, visited, cache)));
    }
  }

  GraphDef node_graph;
  for (const auto& node : f->node_def()) {
    NodeDef* node_graph_node = node_graph.add_node();
    *node_graph_node = node;
  }
  for (const auto& input_arg : f->signature().input_arg()) {
    // We add dummy input nodes for the inputs to the function.
    NodeDef* node_graph_node = node_graph.add_node();
    node_graph_node->set_name(input_arg.name());
    node_graph_node->set_op("_Retval");
  }
  *(node_graph.mutable_library()) = library;

  grappler::GraphView node_gv(&node_graph);

  // TODO(frankchn): Investigate whether we need to hash the name of the
  // return argument / control return argument or whether we can relax it and
  // hash the index (etc...)
  uint64 ret_hash = f->ret_size();
  for (const auto& ret : f->ret()) {
    std::pair<std::string, std::string> node_spec =
        absl::StrSplit(ret.second, absl::MaxSplits(':', 1));
    // For every return value, we need to hash the output node (and the subgraph
    // rooted at the output node) to ensure that the computation graph that
    // ends at the output node has not changed.
    uint64 node_hash = HashSubgraphImpl(
        node_gv, node_gv.GetNode(node_spec.first), visited, cache);
    uint64 node_port_hash = Hash64(node_spec.second);

    ret_hash = Hash64CombineUnordered(
        ret_hash, Hash64Combine(Hash64(ret.first),
                                Hash64Combine(node_hash, node_port_hash)));
  }

  uint64 control_ret_hash = f->control_ret_size();
  for (const auto& ret : f->control_ret()) {
    std::pair<std::string, std::string> node_spec =
        absl::StrSplit(ret.second, absl::MaxSplits(':', 1));

    uint64 node_hash = HashSubgraphImpl(
        node_gv, node_gv.GetNode(node_spec.first), visited, cache);
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

  return final_hash;
}

}  // anonymous namespace

Status AsGraphDef(OpKernelContext* ctx, const DatasetBase* dataset,
                  SerializationContext&& serialization_ctx,
                  GraphDef* graph_def) {
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node* output_node = nullptr;
  TF_RETURN_IF_ERROR(
      db.AddInputDataset(&serialization_ctx, dataset, &output_node));
  // Insert a purely symbolic _Retval node to indicate to consumers which Tensor
  // represents this Dataset.
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

Status RewriteDataset(OpKernelContext* ctx, const DatasetBase* input,
                      std::function<RewriterConfig(void)> config_factory,
                      bool optimize_function_library,
                      DatasetBase** rewritten_input) {
  SerializationContext::Params params;
  std::vector<std::pair<string, Tensor>> input_list;
  params.input_list = &input_list;
  params.optimization_only = true;
  SerializationContext serialization_ctx(params);
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(
      AsGraphDef(ctx, input, std::move(serialization_ctx), &graph_def));

  string output_node;
  for (const auto& node : graph_def.node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  VLOG(3) << "Before graph rewrites: " << graph_def.DebugString();
  TF_RETURN_IF_ERROR(ApplyRewrites(ctx, config_factory,
                                   optimize_function_library, &graph_def,
                                   &output_node));
  VLOG(3) << "After graph rewrites: " << graph_def.DebugString();

  // Instantiate the optimized input pipeline by running the optimized graph
  // using the optimized function library.
  FunctionLibraryRuntime* flr = nullptr;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr = nullptr;
  std::unique_ptr<FunctionLibraryDefinition> lib_def = nullptr;
  TF_RETURN_IF_ERROR(
      ctx->function_library()->Clone(&lib_def, &pflr, &flr, true));

  // Some functions may have been modified without having their names
  // changed (for example, nested dataset graphs from FlatMap or
  // Interleave).
  TF_RETURN_IF_ERROR(AddToFunctionLibrary(lib_def.get(), graph_def.library()));

  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));
  std::vector<Tensor> outputs;
  GraphRunner graph_runner(flr->device());

  TF_RETURN_IF_ERROR(
      graph_runner.Run(&graph, flr, input_list, {output_node}, &outputs));
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(outputs[0], rewritten_input));
  (*rewritten_input)->Ref();
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

uint64 HashSubgraphFunction(const FunctionDefLibrary& library,
                            const FunctionDef* f) {
  std::vector<std::string> visited;
  absl::flat_hash_map<std::string, uint64> cache;
  return HashSubgraphFunctionImpl(library, f, &visited, &cache);
}

uint64 HashSubgraph(const GraphDef& g, const NodeDef* node) {
  std::vector<std::string> visited;
  absl::flat_hash_map<std::string, uint64> cache;
  return HashSubgraphImpl(grappler::GraphView(&g), node, &visited, &cache);
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

Status VariantTensorDataReader::ReadScalar(StringPiece key, string* val) {
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
                                            const string& val) {
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

}  // namespace data
}  // namespace tensorflow
