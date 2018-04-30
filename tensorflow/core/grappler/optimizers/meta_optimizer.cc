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

#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/auto_parallel.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/debug_stripper.h"
#include "tensorflow/core/grappler/optimizers/dependency_optimizer.h"
#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"
#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils/colocation.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr int kDefaultNumberOfIterations = 2;

int64 NumEdges(const GraphDef& graph) {
  int64 num_edges = 0;
  for (const auto& node : graph.node()) {
    num_edges += node.input_size();
  }
  return num_edges;
}

string PrintSizesBeforeAfter(const GraphDef& before, const GraphDef& after) {
  return strings::StrCat("Graph size after: ", after.node_size(), " nodes (",
                         after.node_size() - before.node_size(), "), ",
                         NumEdges(after), " edges (",
                         NumEdges(after) - NumEdges(before), ")");
}

int NumIterations(const RewriterConfig& cfg) {
  return cfg.meta_optimizer_iterations() == RewriterConfig::DEFAULT_NUM_ITERS
             ? kDefaultNumberOfIterations
             : cfg.meta_optimizer_iterations();
}

// Check if optimizer is allowed to run only once.
bool IsRunOnceOptimizer(const string& name) {
  return name == "layout" || name == "memory_optimizer" ||
         name == "arithmetic_optimizer" || name == "loop_optimizer";
}

}  // namespace

#define MK_OPT(NAME, VALUE) \
  if (optimizer == NAME) return std::unique_ptr<GraphOptimizer>(VALUE)

std::unique_ptr<GraphOptimizer> MetaOptimizer::MakeNewOptimizer(
    const string& optimizer) const {
  MK_OPT("pruning", new ModelPruner());
  MK_OPT("function", new FunctionOptimizer(cfg_.function_optimization()));
  MK_OPT("constfold", new ConstantFolding(cpu_device_));
  MK_OPT("layout", new LayoutOptimizer());
  MK_OPT("memory", new MemoryOptimizer(RewriterConfig::MANUAL));
  MK_OPT("arithmetic", new ArithmeticOptimizer(cfg_.arithmetic_optimization()));
  MK_OPT("autoparallel", new AutoParallel(cfg_.auto_parallel().num_replicas()));
  MK_OPT("loop", new LoopOptimizer(cfg_.loop_optimization()));
  MK_OPT("dependency", new DependencyOptimizer(cfg_.dependency_optimization()));
  MK_OPT("debug_stripper", new DebugStripper());

  return std::unique_ptr<GraphOptimizer>();
}

#undef MK_OPT

Status MetaOptimizer::InitializeOptimizers(
    std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const {
  if (!cfg_.disable_model_pruning()) {
    optimizers->emplace_back(new ModelPruner());
  }
  if (cfg_.function_optimization() != RewriterConfig::OFF) {
    optimizers->emplace_back(
        new FunctionOptimizer(cfg_.function_optimization()));
  }
  if (cfg_.debug_stripper() == RewriterConfig::ON) {
    optimizers->emplace_back(new DebugStripper());
  }
  if (cfg_.constant_folding() != RewriterConfig::OFF) {
    optimizers->emplace_back(
        new ConstantFolding(cfg_.constant_folding(), cpu_device_));
  }
  if (cfg_.arithmetic_optimization() != RewriterConfig::OFF) {
    optimizers->emplace_back(
        new ArithmeticOptimizer(cfg_.arithmetic_optimization()));
  }
  if (cfg_.loop_optimization() != RewriterConfig::OFF) {
    optimizers->emplace_back(new LoopOptimizer(cfg_.loop_optimization()));
  }
  if (cfg_.dependency_optimization() != RewriterConfig::OFF) {
    optimizers->emplace_back(
        new DependencyOptimizer(cfg_.dependency_optimization()));
  }
  if (cfg_.layout_optimizer() != RewriterConfig::OFF) {
    optimizers->emplace_back(new LayoutOptimizer());
  }
  if (cfg_.memory_optimization() != RewriterConfig::NO_MEM_OPT) {
    if (cfg_.memory_optimizer_target_node_name_scope().empty()) {
      optimizers->emplace_back(
          // Use the default target node name prefix "gradients/"
          new MemoryOptimizer(cfg_.memory_optimization()));
    } else {
      optimizers->emplace_back(
          new MemoryOptimizer(cfg_.memory_optimization(),
                              cfg_.memory_optimizer_target_node_name_scope()));
    }
  }
  if (cfg_.auto_parallel().enable()) {
    optimizers->emplace_back(
        new AutoParallel(cfg_.auto_parallel().num_replicas()));
  }
  return Status::OK();
}

Status MetaOptimizer::InitializeOptimizersByName(
    std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const {
  for (const string& optimizer_name : cfg_.optimizers()) {
    auto optimizer = MakeNewOptimizer(optimizer_name);
    if (optimizer) {
      VLOG(2) << "Registered default graph optimizer: " << optimizer_name;
      optimizers->push_back(std::move(optimizer));
      continue;
    }

    auto custom_optimizer =
        CustomGraphOptimizerRegistry::CreateByNameOrNull(optimizer_name);

    if (custom_optimizer) {
      VLOG(2) << "Registered custom graph optimizer: " << optimizer_name;
      TF_RETURN_IF_ERROR(custom_optimizer->Init());
      optimizers->push_back(std::move(custom_optimizer));
    } else {
      VLOG(2) << "Can't register an optimizer by name: " << optimizer_name;
    }
  }
  for (const auto& optimizer_config : cfg_.custom_optimizers()) {
    auto custom_optimizer = CustomGraphOptimizerRegistry::CreateByNameOrNull(
        optimizer_config.name());
    if (custom_optimizer) {
      VLOG(2) << "Registered custom configurable graph optimizer: "
              << optimizer_config.name();
      TF_RETURN_IF_ERROR(custom_optimizer->Init(&optimizer_config));
      optimizers->push_back(std::move(custom_optimizer));
    } else {
      VLOG(2) << "Can't register an optimizer by name: "
              << optimizer_config.name();
    }
  }
  return Status::OK();
}

Status MetaOptimizer::OptimizeGraph(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  std::vector<std::unique_ptr<GraphOptimizer>> optimizers;
  if (cfg_.optimizers().empty() && cfg_.custom_optimizers().empty()) {
    TF_RETURN_IF_ERROR(InitializeOptimizers(&optimizers));
  } else {
    TF_RETURN_IF_ERROR(InitializeOptimizersByName(&optimizers));
  }

  VLOG(2) << "Optimize GrapplerItem: item.id=" << item.id
          << " num_optimizers=" << optimizers.size();

  if (optimizers.empty()) {
    VLOG(3) << "Skip graph optimization, no optimizers registered";
    *optimized_graph = item.graph;
    return Status::OK();
  }

  // Invariant: optimized_graph contains the most recently optimized version of
  // the graph.
  GrapplerItem optimized_item = item;
  optimized_graph->Swap(&optimized_item.graph);

  bool is_optimized = false;
  GraphOptimizationResult optimization_result(item.id);

  for (int iteration = 0; iteration < NumIterations(cfg_); ++iteration) {
    VLOG(4) << "Starting optimization iteration " << iteration + 1;

    for (const auto& optimizer : optimizers) {
      // Some optimizers can run only once.
      if (iteration > 0 && IsRunOnceOptimizer(optimizer->name())) continue;

      uint64 start_us = Env::Default()->NowMicros();
      // This swaps the current optimized_graph into optimized item and
      // resets optimized_graph to an empty graph.
      optimized_graph->Swap(&optimized_item.graph);
      *optimized_graph = GraphDef();
      Status status =
          optimizer->Optimize(cluster, optimized_item, optimized_graph);
      uint64 end_us = Env::Default()->NowMicros();

      string result;
      if (!status.ok()) {
        optimized_graph->Swap(&optimized_item.graph);
        result = status.ToString();
      } else {
        is_optimized = true;
        float duration_ms = (end_us - start_us) / 1000.0f;
        result = strings::StrCat(
            PrintSizesBeforeAfter(optimized_item.graph, *optimized_graph),
            ", time = ", duration_ms, "ms.");
      }
      VLOG(4) << optimizer->name() << ": " << result;

      OptimizerResult optimizer_result{optimizer->name(), result};
      optimization_result.results.push_back(optimizer_result);
    }
  }

  // Record graph optimization result.
  optimization_results_.push_back(optimization_result);

  if (is_optimized) {
    TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph));
    ReassignColocation(optimized_graph);
    // Make sure that the optimizers preserved the graph version.
    DCHECK_EQ(optimized_graph->versions().producer(),
              item.graph.versions().producer());
  }

  return Status::OK();
}

Status MetaOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  optimization_results_.clear();

  // 1. Optimize main graph
  TF_RETURN_IF_ERROR(OptimizeGraph(cluster, item, optimized_graph));

  // 2. Optimize function library
  FunctionLibraryDefinition flib(OpRegistry::Global(),
                                 optimized_graph->library());

  // Optimize each function only once.
  std::unordered_set<string> optimized_funcs;
  bool optimize_function_library = true;

  while (optimize_function_library) {
    optimize_function_library = false;

    for (const FunctionDef& func : optimized_graph->library().function()) {
      const string& func_name = func.signature().name();

      // Skip already optimized functions.
      if (optimized_funcs.find(func_name) != optimized_funcs.end()) continue;

      // Skip parametrized functions (function type or body is defined only at
      // function call time by caller node attributes).
      if (IsParametrized(func)) continue;

      VLOG(3) << "Optimize function: function=" << func_name;

      // Function optimization might specialize nested function calls, so we
      // have to reset the flag and do at least one more pass over the library.
      optimize_function_library = true;
      optimized_funcs.insert(func_name);

      // Make a GrapplerItem from a FunctionDef.
      GrapplerFunctionItem func_item;
      TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(func, flib, &func_item));

      // Optimize function body graph.
      GraphDef optimized_func_graph;
      TF_RETURN_IF_ERROR(
          OptimizeGraph(cluster, func_item, &optimized_func_graph));

      // Function body optimization might have created new specialized
      // functions for each instantiation context. Add them to the library.
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
      TF_RETURN_IF_ERROR(flib.RemoveFunction(func_name));
      TF_RETURN_IF_ERROR(flib.AddFunctionDef(optimized_func));
    }

    // If optimized at least one function, update the graph library.
    if (optimize_function_library) {
      *optimized_graph->mutable_library() = flib.ToProto();
    }
  }

  VLOG(3) << "Optimized " << optimized_funcs.size()
          << " functions: " << str_util::Join(optimized_funcs, ", ");

  return Status::OK();
}

void MetaOptimizer::PrintResult() {
  for (const GraphOptimizationResult& graph_result : optimization_results_) {
    LOG(INFO) << "Optimization results for grappler item: " << graph_result.id;
    for (const OptimizerResult& result : graph_result.results) {
      LOG(INFO) << "  " << result.optimizer_name << ": " << result.result;
    }
  }
}

void MetaOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                             const GraphDef& pruned_graph, double result) {
  // Nothing to do for MetaOptimizer.
}

bool MetaOptimizerEnabled(const RewriterConfig& cfg) {
  return !cfg.disable_model_pruning() ||
         cfg.layout_optimizer() != RewriterConfig::OFF ||
         cfg.function_optimization() != RewriterConfig::OFF ||
         cfg.constant_folding() != RewriterConfig::OFF ||
         cfg.arithmetic_optimization() != RewriterConfig::OFF ||
         cfg.loop_optimization() != RewriterConfig::OFF ||
         cfg.dependency_optimization() != RewriterConfig::OFF ||
         cfg.auto_parallel().enable() ||
         cfg.memory_optimization() != RewriterConfig::NO_MEM_OPT ||
         cfg.debug_stripper() == RewriterConfig::ON ||
         !cfg.optimizers().empty() || !cfg.custom_optimizers().empty();
}

Status RunMetaOptimizer(const GrapplerItem& item, const RewriterConfig& cfg,
                        DeviceBase* cpu_device, Cluster* cluster,
                        GraphDef* optimized_graph) {
  MetaOptimizer optimizer(cpu_device, cfg);
  return optimizer.Optimize(cluster, item, optimized_graph);
}

}  // namespace grappler
}  // namespace tensorflow
