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

#include "tensorflow/core/kernels/data/rewrite_utils.h"

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/serialization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

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
                     GraphDef* graph_def, string* output_node) {
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
  // Grappler should not optimize function library of tf.data graphs. The
  // tf.data meta optimizer takes care of optimizing tf.data functions.
  grappler_item->optimization_options().optimize_function_library = false;
  std::unordered_map<string, tensorflow::DeviceProperties> device_map;
  tensorflow::grappler::VirtualCluster cluster(device_map);

  // Run data optimizer using grappler's meta optimizer.
  tensorflow::ConfigProto config;
  *config.mutable_graph_options()->mutable_rewrite_options() = config_factory();
  TF_RETURN_IF_ERROR(tensorflow::grappler::RunMetaOptimizer(
      std::move(*grappler_item), config, ctx->device(), &cluster, graph_def));

  // Remove fake sinks after optimizations are done.
  //
  // TODO(b/118820916): When MetaOptimizer adds provisions for function retvals
  // to be optimizable, we will no longer need this.
  for (auto& function_def : *graph_def->mutable_library()->mutable_function()) {
    RemoveFakeSinks(&function_def);
  }

  return Status::OK();
}

}  // anonymous namespace

Status RewriteDataset(OpKernelContext* ctx, const DatasetBase* input,
                      std::function<RewriterConfig(void)> config_factory,
                      bool record_fingerprint, DatasetBase** rewritten_input) {
  SerializationContext::Params params;
  std::vector<std::pair<string, Tensor>> input_list;
  params.input_list = &input_list;
  params.external_state_policy =
      SerializationContext::ExternalStatePolicy::kIgnore;
  params.fail_if_unimplemented = false;
  params.serialize_data_tensors = false;
  params.preserve_random_seeds = false;
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
  TF_RETURN_IF_ERROR(
      ApplyRewrites(ctx, config_factory, &graph_def, &output_node));
  VLOG(3) << "After graph rewrites: " << graph_def.DebugString();

  // Instantiate the optimized input pipeline by running the optimized graph
  // using the optimized function library.
  FunctionLibraryRuntime* flr = nullptr;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr = nullptr;
  std::unique_ptr<FunctionLibraryDefinition> lib_def = nullptr;
  TF_RETURN_IF_ERROR(
      ctx->function_library()->Clone(&lib_def, &pflr, &flr, true));

  // Some functions may have been modified without having their names changed
  // (for example, nested dataset graphs from FlatMap or Interleave).
  TF_RETURN_IF_ERROR(AddToFunctionLibrary(lib_def.get(), graph_def.library()));

  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));
  std::vector<Tensor> outputs;
  GraphRunner graph_runner(flr->device());

  TF_RETURN_IF_ERROR(
      graph_runner.Run(&graph, flr, input_list, {output_node}, &outputs));
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(outputs[0], rewritten_input));
  (*rewritten_input)->Ref();

  if (record_fingerprint) {
    (*ctx->runner())([graph_def = std::move(graph_def),
                      lib_def = lib_def.release(),
                      input_list = std::move(input_list),
                      output_node = std::move(output_node)]() {
      std::unique_ptr<FunctionLibraryDefinition> lib_def_owner(lib_def);
      const NodeDef* node_def = nullptr;
      for (const auto& node : graph_def.node()) {
        if (node.name() == output_node) {
          node_def = &node;
          break;
        }
      }
      if (node_def == nullptr) {
        VLOG(3) << "Failed to find node: " << output_node;
        return;
      }
      uint64 hash = 0;
      Status s = HashNode(graph_def, *node_def, *lib_def, &hash);
      if (!s.ok()) {
        VLOG(3) << "Failed to hash graph: " << s.ToString();
        return;
      }
      for (const auto& pair : input_list) {
        hash = Hash64CombineUnordered(hash, Hash64(pair.first));
        uint64 tensor_hash = 0;
        Status s = HashTensor(pair.second, &tensor_hash);
        if (s.ok()) {
          hash = Hash64CombineUnordered(hash, tensor_hash);
        } else {
          VLOG(3) << "Failed to hash tensor: " << s.ToString();
        }
      }
      string graph_hash =
          strings::StrCat(strings::Hex(hash, strings::kZeroPad16));
      metrics::RecordTFDataFingerprint(graph_hash);
    });
  }

  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
