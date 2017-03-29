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

#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

Status MetaOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  bool already_optimized = false;
  if (!cfg_.disable_model_pruning()) {
    already_optimized = true;
    ModelPruner pruner;
    TF_RETURN_IF_ERROR(pruner.Optimize(nullptr, item, optimized_graph));
  }
  if (cfg_.optimize_tensor_layout()) {
    LayoutOptimizer layout_optimizer;
    if (!already_optimized) {
      return layout_optimizer.Optimize(nullptr, item, optimized_graph);
    } else {
      GrapplerItem optimized_item = item;
      optimized_item.graph = *optimized_graph;
      return layout_optimizer.Optimize(nullptr, optimized_item,
                                       optimized_graph);
    }
  }
  return Status::OK();
}

void MetaOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                             const GraphDef& pruned_graph, double result) {
  // Nothing to do for MetaOptimizer.
}

bool MetaOptimizerEnabled(const RewriterConfig& cfg) {
  return cfg.optimize_tensor_layout();
}

Status RunMetaOptimizer(const GrapplerItem& item, const RewriterConfig& cfg,
                        GraphDef* optimized_graph) {
  MetaOptimizer optimizer(cfg);
  return optimizer.Optimize(nullptr, item, optimized_graph);
}

}  // namespace grappler
}  // namespace tensorflow
