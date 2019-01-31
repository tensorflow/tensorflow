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

#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/dependency_optimizer.h"
#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/optimizers/shape_optimizer.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {

Status TFDataMetaOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                     GraphDef* output) {
  // Stores the optimized item so far.
  GrapplerItem optimized_item = item;

  // Perform optimizations in a meaningful order.
  for (const auto& optimization :
       {"noop_elimination",
        "shuffle_and_repeat_fusion",
        "map_fusion",
        "filter_with_random_uniform_fusion",
        "filter_fusion",
        "map_and_filter_fusion",
        "hoist_random_uniform",
        "map_parallelization",
        "map_and_batch_fusion",
        "map_vectorization",
        "make_numa_aware",
        "latency_all_edges",
        "make_sloppy",
        "pruning",
        "function",
        "shape",
        "arithmetic",
        "dependency"}) {
    TF_RETURN_IF_ERROR(
        ApplyOptimization(optimization, cluster, &optimized_item));
  }

  // Store the final result of all the optimizations in `output`.
  output->Swap(&optimized_item.graph);
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
  TF_RETURN_IF_ERROR((*optimizer)->Optimize(cluster, *item, &result));
  item->graph.Swap(&result);

  return Status::OK();
}

Status TFDataMetaOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (!config) return Status::OK();

  // Initialize custom tf.data optimizers based on config.
  auto& optimizers = config->parameter_map().at("optimizers").list().s();
  for (const auto& optimizer_name : optimizers) {
    auto optimizer =
        CustomGraphOptimizerRegistry::CreateByNameOrNull(optimizer_name);
    if (optimizer) {
      // None of our data optimizers implement a meaningful Init function.
      // This returns an error in case any of them does.
      TF_RETURN_IF_ERROR(optimizer->Init());
      enabled_optimizers_[optimizer_name] = std::move(optimizer);
    } else {
      // This should never happen.
      return errors::Internal(
          "Tried to register a dataset optimizer that doesn't exist: ",
          optimizer_name);
    }
  }

  // Initialize standard grappler optimizers.
  enabled_optimizers_["pruning"] = MakeUnique<ModelPruner>();
  enabled_optimizers_["function"] =
      MakeUnique<FunctionOptimizer>(RewriterConfig::ON);
  enabled_optimizers_["shape"] = MakeUnique<ShapeOptimizer>();
  enabled_optimizers_["arithmetic"] = MakeUnique<ArithmeticOptimizer>();
  enabled_optimizers_["dependency"] = MakeUnique<DependencyOptimizer>();

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
