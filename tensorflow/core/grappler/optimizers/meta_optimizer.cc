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
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/auto_parallel.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/dependency_optimizer.h"
#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"
#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

namespace {
int64 NumEdges(const GraphDef& graph) {
  int64 num_edges = 0;
  for (const auto& node : graph.node()) {
    num_edges += node.input_size();
  }
  return num_edges;
}

string PrintSizesBeforeAfter(const GraphDef& before, const GraphDef& after) {
  return strings::StrCat("Graph size before: ", before.node_size(), " nodes, ",
                         NumEdges(before),
                         " edges. Graph size after: ", after.node_size(),
                         " nodes, ", NumEdges(after), " edges.");
}
}  // namespace

std::unique_ptr<GraphOptimizer> MetaOptimizer::NewOptimizer(
    const string& optimizer) {
  VLOG(1) << "Adding graph optimization pass: " << optimizer;
  std::unique_ptr<GraphOptimizer> graph_optimizer;
  if (optimizer == "pruning") {
    graph_optimizer.reset(new ModelPruner());
  }
  if (optimizer == "function") {
    graph_optimizer.reset(new FunctionOptimizer());
  }
  if (optimizer == "constfold") {
    graph_optimizer.reset(new ConstantFolding(cpu_device_));
  }
  if (optimizer == "layout") {
    graph_optimizer.reset(new LayoutOptimizer());
  }
  if (optimizer == "memory") {
    graph_optimizer.reset(new MemoryOptimizer(RewriterConfig::MANUAL));
  }
  if (optimizer == "arithmetic") {
    graph_optimizer.reset(
        new ArithmeticOptimizer(cfg_.arithmetic_optimization()));
  }
  if (optimizer == "autoparallel") {
    graph_optimizer.reset(
        new AutoParallel(cfg_.auto_parallel().num_replicas()));
  }
  if (optimizer == "loop") {
    graph_optimizer.reset(new LoopOptimizer(cfg_.loop_optimization()));
  }
  if (optimizer == "dependency") {
    graph_optimizer.reset(
        new DependencyOptimizer(cfg_.dependency_optimization()));
  }
  return graph_optimizer;
}

Status MetaOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  std::vector<std::unique_ptr<GraphOptimizer>> optimizers;
  if (cfg_.optimizers().empty()) {
    if (!cfg_.disable_model_pruning()) {
      optimizers.push_back(std::unique_ptr<GraphOptimizer>(new ModelPruner()));
    }
    if (cfg_.function_optimization() == RewriterConfig::ON) {
      optimizers.push_back(
          std::unique_ptr<GraphOptimizer>(new FunctionOptimizer()));
    }
    if (cfg_.constant_folding() != RewriterConfig::OFF) {
      optimizers.push_back(std::unique_ptr<GraphOptimizer>(
          new ConstantFolding(cfg_.constant_folding(), cpu_device_)));
    }
    if (cfg_.arithmetic_optimization() != RewriterConfig::OFF) {
      optimizers.push_back(std::unique_ptr<GraphOptimizer>(
          new ArithmeticOptimizer(cfg_.arithmetic_optimization())));
    }
    if (cfg_.loop_optimization() == RewriterConfig::ON) {
      optimizers.push_back(std::unique_ptr<GraphOptimizer>(
          new LoopOptimizer(cfg_.loop_optimization())));
    }
    if (cfg_.dependency_optimization() != RewriterConfig::OFF) {
      optimizers.push_back(std::unique_ptr<GraphOptimizer>(
          new DependencyOptimizer(cfg_.dependency_optimization())));
    }
    if (cfg_.layout_optimizer() != RewriterConfig::OFF) {
      optimizers.push_back(
          std::unique_ptr<GraphOptimizer>(new LayoutOptimizer()));
    }
    if (cfg_.memory_optimization() != RewriterConfig::NO_MEM_OPT) {
      if (cfg_.memory_optimizer_target_node_name_scope().empty()) {
        optimizers.push_back(std::unique_ptr<GraphOptimizer>(
            // Use the default target node name prefix "gradients/"
            new MemoryOptimizer(cfg_.memory_optimization())));
      } else {
        optimizers.push_back(
            std::unique_ptr<GraphOptimizer>(new MemoryOptimizer(
                cfg_.memory_optimization(),
                cfg_.memory_optimizer_target_node_name_scope())));
      }
    }
    if (cfg_.auto_parallel().enable()) {
      optimizers.push_back(std::unique_ptr<GraphOptimizer>(
          new AutoParallel(cfg_.auto_parallel().num_replicas())));
    }
  } else {
    const std::set<string> available_optimizers = {
        "pruning",      "function",   "constfold", "layout",    "memory",
        "autoparallel", "arithmetic", "loop",      "dependency"};
    std::vector<string> custom_optimizer_names;
    for (const auto& optimizer_name : cfg_.optimizers()) {
      if (available_optimizers.find(optimizer_name) !=
          available_optimizers.end()) {
        optimizers.push_back(NewOptimizer(optimizer_name));
      } else {
        custom_optimizer_names.push_back(optimizer_name);
      }
    }
    // Now run the custom optimizers.
    for (const auto& optimizer_name : custom_optimizer_names) {
      std::unique_ptr<CustomGraphOptimizer> opt =
          CustomGraphOptimizerRegistry::CreateByNameOrNull(optimizer_name);
      if (opt == nullptr) continue;
      TF_RETURN_IF_ERROR(opt->Init());
      optimizers.push_back(std::move(opt));
    }
  }

  if (optimizers.empty()) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  bool already_optimized = false;
  for (const auto& optimizer : optimizers) {
    if (!already_optimized) {
      Status status = optimizer->Optimize(cluster, item, optimized_graph);
      string result;
      if (!status.ok()) {
        VLOG(1) << "Not able to apply optimizer " << optimizer->name()
                << ". Return status: " << status.ToString();
        result = status.ToString();
      } else {
        already_optimized = true;
        result = strings::StrCat(
            "OK. ", PrintSizesBeforeAfter(item.graph, *optimized_graph));
      }
      result_.push_back(std::make_pair(optimizer->name(), result));
      VLOG(1) << "Optimizer " << optimizer->name()
              << " return status: " << result;
    } else {
      GrapplerItem optimized_item(item, std::move(*optimized_graph));
      Status status =
          optimizer->Optimize(cluster, optimized_item, optimized_graph);
      string result;
      if (!status.ok()) {
        VLOG(1) << "Not able to apply optimizer " << optimizer->name()
                << ". Return status: " << status.ToString();
        optimized_graph->Swap(&optimized_item.graph);
        result = status.ToString();
      } else {
        result = strings::StrCat(
            "OK. ",
            PrintSizesBeforeAfter(optimized_item.graph, *optimized_graph));
      }
      result_.push_back(std::make_pair(optimizer->name(), result));
      VLOG(1) << "Optimizer " << optimizer->name()
              << " return status: " << result;
    }
  }

  if (already_optimized) {
    TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph));
    // Make sure that the optimizers preserved the graph version and library.
    DCHECK_GE(optimized_graph->library().function_size(),
              item.graph.library().function_size());
    DCHECK_GE(optimized_graph->library().gradient_size(),
              item.graph.library().gradient_size());
    DCHECK_EQ(optimized_graph->versions().producer(),
              item.graph.versions().producer());
  } else {
    *optimized_graph = item.graph;
  }

  return Status::OK();
}

void MetaOptimizer::PrintResult() {
  for (const auto& result : result_) {
    LOG(INFO) << "Return status of optimizer " << result.first << ": "
              << result.second;
  }
}

void MetaOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                             const GraphDef& pruned_graph, double result) {
  // Nothing to do for MetaOptimizer.
}

bool MetaOptimizerEnabled(const RewriterConfig& cfg) {
  return !cfg.disable_model_pruning() ||
         cfg.layout_optimizer() != RewriterConfig::OFF ||
         cfg.function_optimization() == RewriterConfig::ON ||
         cfg.constant_folding() != RewriterConfig::OFF ||
         cfg.arithmetic_optimization() != RewriterConfig::OFF ||
         cfg.loop_optimization() == RewriterConfig::ON ||
         cfg.dependency_optimization() != RewriterConfig::OFF ||
         cfg.auto_parallel().enable() ||
         cfg.memory_optimization() != RewriterConfig::NO_MEM_OPT ||
         !cfg.optimizers().empty();
}

Status RunMetaOptimizer(const GrapplerItem& item, const RewriterConfig& cfg,
                        DeviceBase* cpu_device, Cluster* cluster,
                        GraphDef* optimized_graph) {
  MetaOptimizer optimizer(cpu_device, cfg);
  return optimizer.Optimize(cluster, item, optimized_graph);
}

}  // namespace grappler
}  // namespace tensorflow
