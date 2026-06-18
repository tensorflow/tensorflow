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
#include "tensorflow/core/data/rewrite_utils.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/refcount.h"

// On mobile we do not provide this functionality because not all of its
// dependencies are available there.
#if !defined(IS_MOBILE_PLATFORM)

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/hash_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kOptimizerName[] = "tf_data_meta_optimizer";
constexpr char kOptimizers[] = "optimizers";
constexpr char kOptimizerConfigs[] = "optimizer_configs";

void AddFakeSinks(FunctionDef* function_def) {
  int counter = 0;
  for (const auto& output : function_def->signature().output_arg()) {
    NodeDef* node = function_def->add_node_def();
    tensorflow::grappler::function_utils::SetUniqueFunctionNodeName(
        absl::StrCat("FakeSink", counter++), function_def, node);
    node->set_op("Identity");
    node->add_input(function_def->ret().at(output.name()));
    (*node->mutable_attr())["T"].set_type(output.type());

    (*function_def->mutable_ret())[output.name()] =
        absl::StrCat(node->name(), ":output:0");
  }
}

void RemoveFakeSinks(FunctionDef* function_def) {
  // Map from identity node names to their input tensor strings
  std::map<std::string, std::string> identity_map;
  for (const auto& node : function_def->node_def()) {
    if (node.op() == "Identity" && node.input_size() == 1) {
      identity_map[node.name()] = node.input(0);
    }
  }
  for (const auto& output_arg : function_def->signature().output_arg()) {
    const std::string& tensor = function_def->ret().at(output_arg.name());
    const std::string& output_node = tensor.substr(0, tensor.find(':'));
    if (identity_map.find(output_node) != identity_map.end()) {
      (*function_def->mutable_ret())[output_arg.name()] =
          identity_map.at(output_node);
    }
  }
}

absl::Status ApplyRewrites(
    OpKernelContext* ctx,
    const std::function<RewriterConfig(void)> config_factory,
    GraphDef* graph_def, string* dataset_node) {
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      GetGrapplerItem(graph_def, dataset_node, /*add_fake_sinks=*/true);
  std::unordered_map<std::string, tensorflow::DeviceProperties> device_map;
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

  return absl::OkStatus();
}
}  // anonymous namespace

RewriterConfig CreateRewriterConfig(
    const absl::flat_hash_set<tstring>& optimizations,
    const absl::flat_hash_set<tstring>& optimizations_configs) {
  RewriterConfig rewriter_config;
  rewriter_config.add_optimizers(kOptimizerName);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
  rewriter_config.set_fail_on_optimizer_errors(true);
  auto custom_optimizer = rewriter_config.add_custom_optimizers();
  custom_optimizer->set_name(kOptimizerName);
  auto* custom_optimizations_list =
      (*custom_optimizer->mutable_parameter_map())[kOptimizers].mutable_list();
  const auto& registered_optimizers =
      grappler::CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  for (const auto& optimization : optimizations) {
    if (std::find(registered_optimizers.begin(), registered_optimizers.end(),
                  optimization) != registered_optimizers.end()) {
      custom_optimizations_list->add_s(optimization.data(),
                                       optimization.size());
    } else {
      VLOG(1) << "Optimization " << optimization << " is not registered.";
    }
  }
  auto* config_list =
      (*custom_optimizer->mutable_parameter_map())[kOptimizerConfigs]
          .mutable_list();
  for (const auto& config : optimizations_configs) {
    config_list->add_s(config.data(), config.size());
  }
  return rewriter_config;
}

absl::Status RewriteDataset(OpKernelContext* ctx, const DatasetBase* input,
                            std::function<RewriterConfig(void)> config_factory,
                            bool record_fingerprint,
                            core::RefCountPtr<DatasetBase>* rewritten_input) {
  std::vector<std::pair<string, Tensor>> input_list;
  GraphDef graph_def;
  string output_node;
  TF_RETURN_IF_ERROR(
      AsGraphDefForRewrite(ctx, input, &input_list, &graph_def, &output_node));

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
  DatasetBase* rewritten_dataset;
  TF_RETURN_IF_ERROR(
      GetDatasetFromVariantTensor(outputs[0], &rewritten_dataset));
  rewritten_dataset->Ref();
  rewritten_input->reset(rewritten_dataset);

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
      absl::Status s = HashNode(graph_def, *node_def, *lib_def, &hash);
      if (!s.ok()) {
        VLOG(3) << "Failed to hash graph: " << s;
        return;
      }
      for (const auto& pair : input_list) {
        hash = Hash64CombineUnordered(hash, Hash64(pair.first));
        uint64 tensor_hash = 0;
        absl::Status s = HashTensor(pair.second, &tensor_hash);
        if (s.ok()) {
          hash = Hash64CombineUnordered(hash, tensor_hash);
        } else {
          VLOG(3) << "Failed to hash tensor: " << s;
        }
      }
      std::string graph_hash = absl::StrCat(absl::Hex(hash, absl::kZeroPad16));
      metrics::RecordTFDataFingerprint(graph_hash);
    });
  }

  return absl::OkStatus();
}

std::unique_ptr<tensorflow::grappler::GrapplerItem> GetGrapplerItem(
    GraphDef* graph_def, std::string* dataset_node, bool add_fake_sinks,
    bool apply_optimizations) {
  // Add an identity node as the fetch node, otherwise we might get 'placeholder
  // is both fed and fetched' errors in some cases when using input list with
  // placeholder dataset nodes.
  NodeDef* node = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            node);
  node->set_op("Identity");
  node->add_input(*dataset_node);
  (*node->mutable_attr())["T"].set_type(DT_VARIANT);
  *dataset_node = node->name();

  if (add_fake_sinks) {
    // Add fake sink node to graph and functions to allow rewriting the actual
    // sink nodes.
    //
    // TODO(b/118820916): When MetaOptimizer adds provisions for function
    // retvals to be optimizable, we will no longer need this.
    for (auto& function_def :
         *graph_def->mutable_library()->mutable_function()) {
      AddFakeSinks(&function_def);
    }
  }

  // Create metagraph.
  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_graph_def()) = *graph_def;

  // Grappler determines fetch ops from collection 'train_op'.
  CollectionDef collection_def;
  auto node_list = collection_def.mutable_node_list();
  node_list->add_value(*dataset_node);
  (*meta_graph_def.mutable_collection_def())["train_op"] = collection_def;

  // Create Grappler item.
  tensorflow::grappler::ItemConfig item_config;
  item_config.apply_optimizations = apply_optimizations;
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      tensorflow::grappler::GrapplerItemFromMetaGraphDef(
          "graph", meta_graph_def, item_config);
  // Grappler should not optimize function library of tf.data graphs. The
  // tf.data meta optimizer takes care of optimizing tf.data functions.
  grappler_item->optimization_options().optimize_function_library = false;
  return grappler_item;
}

absl::flat_hash_set<tstring> SelectOptimizations(
    const absl::flat_hash_set<string>& experiments,
    const absl::flat_hash_set<tstring>& optimizations_enabled,
    const absl::flat_hash_set<tstring>& optimizations_disabled,
    const absl::flat_hash_set<tstring>& optimizations_default) {
  absl::flat_hash_set<tstring> optimizations;

  // Add the enabled optimizations.
  optimizations.insert(optimizations_enabled.begin(),
                       optimizations_enabled.end());

  // Add all default optimization that are not disabled.
  for (const auto& optimization : optimizations_default) {
    if (!optimizations_disabled.contains(optimization)) {
      optimizations.insert(optimization);
    }
  }

  // Add experiments that correspond to an optimization unless the optimization
  // is disabled.
  const auto& registered_optimizers =
      grappler::CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  for (const auto& experiment : experiments) {
    if (std::find(registered_optimizers.begin(), registered_optimizers.end(),
                  experiment) != registered_optimizers.end() &&
        !optimizations_disabled.contains(experiment)) {
      optimizations.insert(experiment);
    }
  }

  return optimizations;
}

absl::StatusOr<std::string> GetDatasetNode(const GraphDef& graph_def) {
  // Symbolic `_Retval` node indicates which node corresponds to the dataset.
  for (const auto& node : graph_def.node()) {
    if (node.op() == kRetvalOp) {
      return node.input(0);
    }
  }
  return errors::NotFound(
      absl::Substitute("Dataset node for graph is not found:\n$0",
                       graph_def.ShortDebugString()));
}

absl::StatusOr<NodeDef> GetDatasetNodeDef(const GraphDef& graph_def) {
  TF_ASSIGN_OR_RETURN(std::string dataset_node_name, GetDatasetNode(graph_def));
  for (const auto& node : graph_def.node()) {
    if (node.name() == dataset_node_name) {
      return node;
    }
  }
  return errors::NotFound(
      absl::Substitute("Dataset node for graph is not found:\n$0",
                       graph_def.ShortDebugString()));
}

}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM
