/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/meta_optimizer.h"

#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/common_subgraph_elimination.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/dependency_optimizer.h"
#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/optimizers/remapper.h"
#include "tensorflow/core/grappler/optimizers/shape_optimizer.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {

namespace {

using ConfigMap =
    std::map<string, tensorflow::RewriterConfig_CustomGraphOptimizer>;

// tf.data optimizations, in the order we want to perform them.
constexpr std::array<const char*, 15> kTFDataOptimizations = {
    "noop_elimination",
    "shuffle_and_repeat_fusion",
    "map_fusion",
    "filter_fusion",
    "filter_with_random_uniform_fusion",
    "map_and_filter_fusion",
    "hoist_random_uniform",
    "map_parallelization",
    "map_and_batch_fusion",
    "map_vectorization",
    "latency_all_edges",
    "make_sloppy",
    "parallel_batch",
    "slack",
    "inject_prefetch"};

// Standard grappler optimizations, in the order we want to perform them.
// The order matches the order in the generic meta optimizer.
constexpr std::array<const char*, 9> kGrapplerOptimizations = {
    "pruning",  "function",   "common_subgraph_elimination",
    "shape",    "arithmetic", "layout_optimizer",
    "remapper", "loop",       "dependency",
};

// Parses a list of string optimizer configurations into a map from
// optimizer name -> rewriter config for that optimizer.
Status ToConfigMap(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config,
    ConfigMap* result) {
  auto found = gtl::FindOrNull(config->parameter_map(), "optimizer_configs");
  if (!found) return Status::OK();

  auto& options = found->list().s();
  for (const auto& option_string : options) {
    // The option string has the format
    // <optimizer_name>:<config_key>:<config_value>
    std::vector<string> split = absl::StrSplit(option_string, ':');
    if (split.size() != 3) {
      return errors::Internal(
          "Wrong format for optimizer options. Expect <optimizer name>:<config "
          "key>:<config value>, received: ",
          option_string);
    }

    const string& optimizer_name = split[0];
    const string& config_key = split[1];
    const string& config_value = split[2];

    auto optimizer_config = gtl::FindOrNull(*result, optimizer_name);
    if (!optimizer_config) {
      (*result)[optimizer_name] =
          tensorflow::RewriterConfig_CustomGraphOptimizer();
      optimizer_config = gtl::FindOrNull(*result, optimizer_name);
    }
    (*optimizer_config->mutable_parameter_map())[config_key].set_s(
        config_value);
  }

  return Status::OK();
}

}  // namespace

Status TFDataMetaOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                     GraphDef* output) {
  // Stores the optimized item so far.
  GrapplerItem optimized_item = item;

  // Perform optimizations in a meaningful order.
  for (const auto& optimization : kTFDataOptimizations) {
    TF_RETURN_IF_ERROR(
        ApplyOptimization(optimization, cluster, &optimized_item));
  }

  for (const auto& optimization : kGrapplerOptimizations) {
    TF_RETURN_IF_ERROR(
        ApplyOptimization(optimization, cluster, &optimized_item));
  }

  // Store the final result of all the optimizations in `output`.
  output->Swap(&optimized_item.graph);

  // Optimize tf.data user-defined functions.
  FunctionLibraryDefinition flib =
      FunctionLibraryDefinition(OpRegistry::Global(), output->library())
          .ReachableDefinitions(*output);
  const auto producer = output->versions().producer();
  bool optimized_functions = false;
  for (const FunctionDef& func : output->library().function()) {
    // Skip non tf.data functions.
    if (!func.attr().contains(data::kTFDataFunction)) continue;
    VLOG(3) << "Optimize function: function=" << func.signature().name();
    optimized_functions = true;

    // Make a GrapplerItem from a FunctionDef.
    GrapplerFunctionItem func_item;
    TF_RETURN_IF_ERROR(
        MakeGrapplerFunctionItem(func, flib, producer, &func_item));

    GraphDef optimized_func_graph;
    TF_RETURN_IF_ERROR(Optimize(cluster, func_item, &optimized_func_graph));

    // Function body optimization might have created new functions. Add them to
    // the library.
    for (const FunctionDef& func_def :
         optimized_func_graph.library().function()) {
      if (flib.Find(func_def.signature().name()) == nullptr) {
        TF_RETURN_IF_ERROR(flib.AddFunctionDef(func_def));
      }
    }

    // Convert optimized graph back to FunctionDef.
    FunctionDef optimized_func;
    func_item.SwapFunctionBody(std::move(optimized_func_graph));
    TF_RETURN_IF_ERROR(MakeFunctionDef(func_item, flib, &optimized_func));

    // Replace optimized function with a new FunctionDef.
    TF_RETURN_IF_ERROR(
        flib.ReplaceFunction(func.signature().name(), optimized_func));
  }
  if (optimized_functions) {
    *output->mutable_library() = flib.ToProto();
  }
  return Status::OK();
}

Status TFDataMetaOptimizer::ApplyOptimization(const string& name,
                                              Cluster* cluster,
                                              GrapplerItem* item) const {
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  const auto* optimizer = gtl::FindOrNull(enabled_optimizers_, name);
  if (!optimizer) {
    return Status::OK();
  }

  GraphDef result;
  (*optimizer)->set_deadline_usec(this->deadline_usec());
  Status status = (*optimizer)->Optimize(cluster, *item, &result);
  if (status.ok()) {
    // The optimizer succeeded and wrote the optimized graph to result.
    item->graph.Swap(&result);
  } else if (errors::IsAborted(status)) {
    // A status of errors::Aborted just means that the optimizer was a no-op and
    // did not populate result. Swallow the error status and leave the original
    // graph in item.
    status = Status::OK();
  }

  return status;
}

Status TFDataMetaOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (!config) return Status::OK();

  // Initialize custom tf.data optimizers based on config.
  auto& optimizers = config->parameter_map().at("optimizers").list().s();
  ConfigMap optimizer_configs;
  TF_RETURN_IF_ERROR(ToConfigMap(config, &optimizer_configs));

  for (const auto& optimizer_name : optimizers) {
    auto optimizer =
        CustomGraphOptimizerRegistry::CreateByNameOrNull(optimizer_name);
    if (optimizer) {
      TF_RETURN_IF_ERROR(
          optimizer->Init(gtl::FindOrNull(optimizer_configs, optimizer_name)));

      enabled_optimizers_[optimizer_name] = std::move(optimizer);
    } else {
      // This should never happen.
      return errors::Internal(
          "Tried to register a dataset optimizer that doesn't exist: ",
          optimizer_name);
    }
  }

  // Enable a subset of grappler optimization that are enabled by default.
  //
  // Layout optimizations are excluded because they assume that ops without
  // explicit device assignment will be placed on GPU (if available) but that's
  // not the case for operations within tf.data functions.
  //
  // TODO(b/120437209): Re-enable constant folding.
  //
  // TODO(jsimsa): Make the set of generic Grappler optimization applied to
  // tf.data functions configurable.
  enabled_optimizers_["pruning"] = MakeUnique<ModelPruner>();
  enabled_optimizers_["shape"] = MakeUnique<ShapeOptimizer>();
  enabled_optimizers_["remapping"] = MakeUnique<Remapper>(RewriterConfig::ON);
  enabled_optimizers_["common_subgraph_elimination"] =
      MakeUnique<CommonSubgraphElimination>();
  enabled_optimizers_["arithmetic"] = MakeUnique<ArithmeticOptimizer>();
  enabled_optimizers_["dependency"] = MakeUnique<DependencyOptimizer>();
  enabled_optimizers_["loop"] = MakeUnique<LoopOptimizer>();
  enabled_optimizers_["function"] = MakeUnique<FunctionOptimizer>(
      RewriterConfig::ON, /*lower_control_flow=*/true);

  return Status::OK();
}

void TFDataMetaOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                   const GraphDef& optimize_output,
                                   double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(TFDataMetaOptimizer, "tf_data_meta_optimizer");

}  // namespace grappler
}  // namespace tensorflow
