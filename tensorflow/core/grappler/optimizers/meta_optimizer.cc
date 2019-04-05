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
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/auto_parallel.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/debug_stripper.h"
#include "tensorflow/core/grappler/optimizers/dependency_optimizer.h"
#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include "tensorflow/core/grappler/optimizers/implementation_selector.h"
#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"
#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/optimizers/pin_to_host_optimizer.h"
#include "tensorflow/core/grappler/optimizers/remapper.h"
#include "tensorflow/core/grappler/optimizers/scoped_allocator_optimizer.h"
#include "tensorflow/core/grappler/optimizers/shape_optimizer.h"
#include "tensorflow/core/grappler/utils/colocation.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/tpu.h"
#include "tensorflow/core/grappler/verifiers/structure_verifier.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr int kDefaultNumberOfIterations = 2;
constexpr int kDefaultMinGraphNodes = 4;

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
         name == "loop_optimizer";
}

uint64 DeadlineMicroSeconds(const RewriterConfig& cfg) {
  const uint64 kFiveMinutesInUsec = 5 * 60 * 1000 * 1000;
  if (cfg.meta_optimizer_timeout_ms() < 0) {
    return 0;
  } else {
    return cfg.meta_optimizer_timeout_ms() == 0
               ? Env::Default()->NowMicros() + kFiveMinutesInUsec
               : Env::Default()->NowMicros() +
                     cfg.meta_optimizer_timeout_ms() * 1000;
  }
}

Status CompressConstants(GraphDef* graph) {
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if ((IsConstant(*node) || IsHostConstant(*node)) &&
        HasNodeAttr(*node, "value")) {
      AttrValue& attr_val = (*node->mutable_attr())["value"];
      tensor::CompressTensorProtoInPlace(attr_val.mutable_tensor());
    }
  }
  return Status::OK();
}

}  // namespace

#define MK_OPT(NAME, VALUE) \
  if (optimizer == NAME) return std::unique_ptr<GraphOptimizer>(VALUE)

std::unique_ptr<GraphOptimizer> MetaOptimizer::MakeNewOptimizer(
    const string& optimizer) const {
  MK_OPT("pruning", new ModelPruner());
  MK_OPT("function", new FunctionOptimizer(cfg_.function_optimization()));
  MK_OPT("constfold", new ConstantFolding(cpu_device_));
  MK_OPT("shape", new ShapeOptimizer());
  MK_OPT("remap", new Remapper(cfg_.remapping()));
  MK_OPT("layout", new LayoutOptimizer());
  MK_OPT("memory", new MemoryOptimizer(RewriterConfig::MANUAL));
  MK_OPT("arithmetic", new ArithmeticOptimizer(cfg_.arithmetic_optimization()));
  MK_OPT("autoparallel", new AutoParallel(cfg_.auto_parallel().num_replicas()));
  MK_OPT("loop", new LoopOptimizer(cfg_.loop_optimization(), cpu_device_));
  MK_OPT("dependency", new DependencyOptimizer(cfg_.dependency_optimization()));
  MK_OPT("debug_stripper", new DebugStripper());
  MK_OPT("scoped_allocator",
         new ScopedAllocatorOptimizer(cfg_.scoped_allocator_optimization(),
                                      cfg_.scoped_allocator_opts()));
  MK_OPT("pin_to_host",
         new PinToHostOptimizer(cfg_.pin_to_host_optimization()));

  return std::unique_ptr<GraphOptimizer>();
}

#undef MK_OPT

MetaOptimizer::MetaOptimizer(DeviceBase* cpu_device, const ConfigProto& cfg)
    : cpu_device_(cpu_device),
      config_proto_(cfg),
      cfg_(*config_proto_.mutable_graph_options()->mutable_rewrite_options()) {
  DCHECK(cpu_device_ == nullptr ||
         cpu_device_->attributes().device_type() == "CPU");
}

Status MetaOptimizer::InitializeOptimizers(
    std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const {
  if (cfg_.disable_meta_optimizer()) {
    return Status::OK();
  }
  if (!cfg_.disable_model_pruning()) {
    optimizers->push_back(MakeUnique<ModelPruner>());
  }
  if (cfg_.implementation_selector() != RewriterConfig::OFF) {
    optimizers->push_back(MakeUnique<ImplementationSelector>());
  }
  if (cfg_.function_optimization() != RewriterConfig::OFF) {
    optimizers->push_back(
        MakeUnique<FunctionOptimizer>(cfg_.function_optimization()));
  }
  if (cfg_.debug_stripper() == RewriterConfig::ON) {
    optimizers->push_back(MakeUnique<DebugStripper>());
  }
  if (cfg_.constant_folding() != RewriterConfig::OFF) {
    optimizers->push_back(
        MakeUnique<ConstantFolding>(cfg_.constant_folding(), cpu_device_));
  }
  if (cfg_.shape_optimization() != RewriterConfig::OFF) {
    optimizers->push_back(MakeUnique<ShapeOptimizer>());
  }
  if (cfg_.remapping() != RewriterConfig::OFF) {
    optimizers->push_back(MakeUnique<Remapper>(cfg_.remapping()));
  }
  if (cfg_.pin_to_host_optimization() == RewriterConfig::ON) {
    optimizers->push_back(MakeUnique<PinToHostOptimizer>());
  }
  if (cfg_.arithmetic_optimization() != RewriterConfig::OFF) {
    optimizers->push_back(
        MakeUnique<ArithmeticOptimizer>(cfg_.arithmetic_optimization()));
  }
  if (cfg_.loop_optimization() != RewriterConfig::OFF) {
    optimizers->push_back(
        MakeUnique<LoopOptimizer>(cfg_.loop_optimization(), cpu_device_));
  }
  if (cfg_.dependency_optimization() != RewriterConfig::OFF) {
    optimizers->push_back(
        MakeUnique<DependencyOptimizer>(cfg_.dependency_optimization()));
  }
  if (cfg_.layout_optimizer() != RewriterConfig::OFF) {
    optimizers->push_back(MakeUnique<LayoutOptimizer>());
  }
  if (cfg_.memory_optimization() != RewriterConfig::NO_MEM_OPT) {
    if (cfg_.memory_optimizer_target_node_name_scope().empty()) {
      optimizers->push_back(
          // Use the default target node name prefix "gradients/"
          MakeUnique<MemoryOptimizer>(cfg_.memory_optimization()));
    } else {
      optimizers->push_back(MakeUnique<MemoryOptimizer>(
          cfg_.memory_optimization(),
          cfg_.memory_optimizer_target_node_name_scope()));
    }
  }
  if (cfg_.auto_parallel().enable()) {
    optimizers->push_back(
        MakeUnique<AutoParallel>(cfg_.auto_parallel().num_replicas()));
  }
  if (cfg_.scoped_allocator_optimization()) {
    optimizers->push_back(MakeUnique<ScopedAllocatorOptimizer>(
        cfg_.scoped_allocator_optimization(), cfg_.scoped_allocator_opts()));
  }
  return InitializeCustomGraphOptimizers(std::set<string>(), optimizers);
}

Status MetaOptimizer::InitializeOptimizersByName(
    std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const {
  std::set<string> initialized_custom_optimizers;
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
      TF_RETURN_IF_ERROR(custom_optimizer->Init(
          GetCustomGraphOptimizerConfig(optimizer_name)));
      optimizers->push_back(std::move(custom_optimizer));
      initialized_custom_optimizers.insert(optimizer_name);
    } else {
      VLOG(2) << "Can't register an optimizer by name: " << optimizer_name;
    }
  }
  return InitializeCustomGraphOptimizers(initialized_custom_optimizers,
                                         optimizers);
}

Status MetaOptimizer::InitializeCustomGraphOptimizers(
    const std::set<string>& pre_initialized_optimizers,
    std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const {
  for (const auto& optimizer_config : cfg_.custom_optimizers()) {
    if (pre_initialized_optimizers.find(optimizer_config.name()) !=
        pre_initialized_optimizers.end()) {
      continue;
    }

    auto custom_optimizer = CustomGraphOptimizerRegistry::CreateByNameOrNull(
        optimizer_config.name());

    if (custom_optimizer) {
      VLOG(2) << "Registered custom configurable graph optimizer: "
              << optimizer_config.name();
      TF_RETURN_IF_ERROR(custom_optimizer->Init(&optimizer_config));
      optimizers->push_back(std::move(custom_optimizer));
    } else {
      // If there are no custom optimizers with given name, try to initalize a
      // default optimizer. This way, custom configurable optimizers can be
      // mixed with default optimizers in any order.
      auto optimizer = MakeNewOptimizer(optimizer_config.name());
      if (optimizer) {
        VLOG(2) << "Registered default graph optimizer: "
                << optimizer_config.name();
        optimizers->push_back(std::move(optimizer));
        continue;
      }
      VLOG(2) << "Can't register an optimizer by name: "
              << optimizer_config.name();
    }
  }
  return Status::OK();
}

const RewriterConfig::CustomGraphOptimizer*
MetaOptimizer::GetCustomGraphOptimizerConfig(const string& name) const {
  for (const auto& config : cfg_.custom_optimizers()) {
    if (config.name() == name) {
      return &config;
    }
  }
  return nullptr;
}

void MetaOptimizer::InitializeVerifiers(
    std::vector<std::unique_ptr<GraphVerifier>>* inter_optimizer_verifiers,
    std::vector<std::unique_ptr<GraphVerifier>>* post_optimization_verifiers)
    const {
  if (cfg_.inter_optimizer_verifier_config().structure_verifier() ==
      VerifierConfig::ON) {
    inter_optimizer_verifiers->push_back(MakeUnique<StructureVerifier>());
  }
  if (cfg_.post_optimization_verifier_config().structure_verifier() ==
      VerifierConfig::ON) {
    post_optimization_verifiers->push_back(MakeUnique<StructureVerifier>());
  }
}

Status MetaOptimizer::OptimizeGraph(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  int min_graph_nodes = cfg_.min_graph_nodes() == 0 ? kDefaultMinGraphNodes
                                                    : cfg_.min_graph_nodes();
  if (item.graph.node_size() < min_graph_nodes) {
    VLOG(3) << "Skipping optimization, graph has less than " << min_graph_nodes
            << " nodes.";
    *optimized_graph = item.graph;
    return Status::OK();
  }

  std::vector<std::unique_ptr<GraphOptimizer>> optimizers;
  if (cfg_.optimizers().empty()) {
    TF_RETURN_IF_ERROR(InitializeOptimizers(&optimizers));
  } else {
    TF_RETURN_IF_ERROR(InitializeOptimizersByName(&optimizers));
  }

  // Initialize the configured verifiers.
  std::vector<std::unique_ptr<GraphVerifier>> inter_optimizer_verifiers;
  std::vector<std::unique_ptr<GraphVerifier>> post_optimization_verifiers;
  InitializeVerifiers(&inter_optimizer_verifiers, &post_optimization_verifiers);
  if (inter_optimizer_verifiers.empty()) {
    VLOG(2) << "No inter optimizer verifiers have been configured";
  } else {
    VLOG(2) << inter_optimizer_verifiers.size()
            << " inter optimizer verifiers have been configured";
  }
  if (post_optimization_verifiers.empty()) {
    VLOG(2) << "No post optimization verifiers have been configured";
  } else {
    VLOG(2) << post_optimization_verifiers.size()
            << " post optimization verifiers have been configured";
  }

  VLOG(2) << "Optimize GrapplerItem: item.id=" << item.id
          << " num_optimizers=" << optimizers.size()
          << ", num nodes = " << item.graph.node_size();

  if (optimizers.empty()) {
    VLOG(3) << "Skipping graph optimization, no optimizers registered";
    *optimized_graph = item.graph;
    return Status::OK();
  }

  // Invariant: optimized_graph contains the most recently optimized version of
  // the graph.
  GrapplerItem optimized_item = item;
  optimized_graph->Swap(&optimized_item.graph);

  GraphOptimizationResult optimization_result(item.id);
  GraphOptimizer* fusion_optimizer = nullptr;
  GraphOptimizer* sa_optimizer = nullptr;

  for (int iteration = 0; iteration < NumIterations(cfg_); ++iteration) {
    // Don't bother optimizing further if the graph is already tiny.
    if (optimized_graph->node_size() < min_graph_nodes) {
      VLOG(3) << "Stopping after iteration " << iteration
              << ", graph is tiny (#nodes = " << optimized_graph->node_size()
              << "  < " << min_graph_nodes << ")";
      break;
    }

    VLOG(4) << "Starting optimization iteration " << iteration;
    if (VLOG_IS_ON(4)) {
      DumpGraphDefToFile(
          strings::StrCat("before_MetaOptimizer_iteration_", iteration, "_",
                          reinterpret_cast<uintptr_t>(optimized_graph)),
          *optimized_graph);
    }
    for (const auto& optimizer : optimizers) {
      GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
      // Some optimizers can run only once.
      if (iteration > 0 && IsRunOnceOptimizer(optimizer->name())) continue;
      // Some must run only on the last iteration.
      if (optimizer->name() == "scoped_allocator_optimizer") {
        if (sa_optimizer == nullptr) sa_optimizer = optimizer.get();
        continue;
      }
      if (optimizer->name() == "xla-fusion") {
        if (fusion_optimizer == nullptr) fusion_optimizer = optimizer.get();
        continue;
      }

      TF_RETURN_IF_ERROR(RunOptimizer(optimizer.get(), cluster, &optimized_item,
                                      optimized_graph, &optimization_result));

      if (VLOG_IS_ON(4)) {
        DumpGraphDefToFile(
            strings::StrCat("after_MetaOptimizer_iteration_", iteration, "_",
                            optimizer->name(), "_",
                            reinterpret_cast<uintptr_t>(optimized_graph)),
            *optimized_graph);
      }
      for (const auto& verifier : inter_optimizer_verifiers) {
        // TODO(ashwinm): Need to enforce verification_deadline.
        TF_RETURN_IF_ERROR(verifier->Verify(*optimized_graph));
      }
    }
    if (VLOG_IS_ON(4)) {
      DumpGraphDefToFile(
          strings::StrCat("after_MetaOptimizer_iteration_", iteration, "_",
                          reinterpret_cast<uintptr_t>(optimized_graph)),
          *optimized_graph);
    }
    // TODO(ashwinm): Need to enforce verification_deadline.
    for (const auto& verifier : post_optimization_verifiers) {
      TF_RETURN_IF_ERROR(verifier->Verify(*optimized_graph));
    }
  }

  // Run fusion optimizer if requested after all other optimizers since: 1) it
  // doesn't need to be called more than once. 2) we don't want subsequent
  // optimization passes to break the fusion clusters. We could potentially
  // encapsulate the fusion clusters right away, but that will prevent a lot of
  // optimizations from taking place since we don't have shape inference for
  // functions, and we can't optimize across function boundaries.
  if (fusion_optimizer != nullptr) {
    TF_RETURN_IF_ERROR(RunOptimizer(fusion_optimizer, cluster, &optimized_item,
                                    optimized_graph, &optimization_result));
  }

  // ScopedAllocatorOptimizer must run last.
  if (sa_optimizer != nullptr) {
    TF_RETURN_IF_ERROR(RunOptimizer(sa_optimizer, cluster, &optimized_item,
                                    optimized_graph, &optimization_result));
  }

  // Compress the constants in the final graph.
  TF_RETURN_IF_ERROR(CompressConstants(optimized_graph));

  bool is_optimized = std::find_if(optimization_result.results.begin(),
                                   optimization_result.results.end(),
                                   [](const OptimizerResult& result) {
                                     return result.status.ok();
                                   }) != optimization_result.results.end();

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

Status MetaOptimizer::RunOptimizer(
    GraphOptimizer* optimizer, Cluster* cluster, GrapplerItem* optimized_item,
    GraphDef* optimized_graph, GraphOptimizationResult* optimization_result) {
  uint64 start_us = Env::Default()->NowMicros();
  // This swaps the current optimized_graph into optimized item and
  // resets optimized_graph to an empty graph.
  optimized_graph->Swap(&optimized_item->graph);
  *optimized_graph = GraphDef();
  optimizer->set_deadline_usec(this->deadline_usec());
  Status status =
      optimizer->Optimize(cluster, *optimized_item, optimized_graph);
  uint64 end_us = Env::Default()->NowMicros();
  float duration_ms = (end_us - start_us) / 1000.0f;

  string message;
  if (!status.ok()) {
    optimized_graph->Swap(&optimized_item->graph);
    if (errors::IsDeadlineExceeded(status)) {
      message =
          strings::StrCat(status.ToString(), ", time = ", duration_ms, "ms.");
      LOG(WARNING) << optimizer->name() << " failed: " << message;
    } else {
      message = status.ToString();
      LOG(ERROR) << optimizer->name() << " failed: " << message;
    }
  } else {
    message = strings::StrCat(
        PrintSizesBeforeAfter(optimized_item->graph, *optimized_graph),
        ", time = ", duration_ms, "ms.");
    VLOG(1) << optimizer->name() << ": " << message;
  }

  OptimizerResult optimizer_result{optimizer->name(), message, status};
  optimization_result->results.push_back(optimizer_result);

  if (!status.ok() && cfg_.fail_on_optimizer_errors()) return status;

  return Status::OK();
}

Status MetaOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  VLOG(1) << "Starting optimization for grappler item: " << item.id;
  optimization_results_.clear();

  // Constructs a FunctionLibraryDefinition with functions that are reachable
  // from the nodes of the graph.
  const auto minimized_flib =
      [](const GraphDef& graph) -> FunctionLibraryDefinition {
    return FunctionLibraryDefinition(OpRegistry::Global(), graph.library())
        .ReachableDefinitions(graph);
  };

  // 0. Original graph might contain a huge function library, that is mostly
  // unused. This library copied over by each individual Grappler optimizer,
  // which adds a huge overhead. Before starting optimization passes we just
  // remove all the unreachable functions.
  // TODO(ezhulenev): Construct reachable function library definition directly
  // from the proto without constructing temporary FunctionLibraryDefinition.
  GraphDef trimmed_graph;  // do not copy graph with a potentially huge library
  *trimmed_graph.mutable_node() = item.graph.node();
  *trimmed_graph.mutable_versions() = item.graph.versions();
  *trimmed_graph.mutable_library() = minimized_flib(item.graph).ToProto();

  GrapplerItem trimmed_item = item.WithGraph(std::move(trimmed_graph));

  VLOG(1) << absl::Substitute(
      "Deleted $0 unreachable functions from the graph (library size = $1)",
      item.graph.library().function_size() -
          trimmed_item.graph.library().function_size(),
      trimmed_item.graph.library().function_size());

  // 1. Optimize main graph
  TF_RETURN_IF_ERROR(OptimizeGraph(cluster, trimmed_item, optimized_graph));
  VLOG(1) << "Optimized main graph.";
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  // Skip optimizing functions if this is a TPU graph. Currently, Grappler
  // passes do not handle TPU functions correctly in a variety of ways (Note
  // that due to the pre-placement TPU graph rewriting passes, the TPU-related
  // ops are encapsulated away into functions). For example, TPU graphs contain
  // TPUReplicateMetadata node that carries relevant TPU metadata and Grappler
  // passes could prune that away. Grappler passes could also cause issues
  // around shape inference. Since the desired and existing behavior is to not
  // optimize TPU functions with Grappler, this check preserves that.
  if (IsTPUGraphDef(*optimized_graph)) {
    VLOG(2) << "Skipping optimizing funcs for TPU graphs";
    if (VLOG_IS_ON(1)) {
      DumpGraphDefToFile(
          strings::StrCat("after_MetaOptimizer_",
                          reinterpret_cast<uintptr_t>(optimized_graph)),
          *optimized_graph);
    }
    return Status::OK();
  }

  // 2. Optimize functions reachable from the optimized graph.
  FunctionLibraryDefinition flib = minimized_flib(*optimized_graph);

  // Find functions for which we might need to compute a gradient at runtime.
  absl::flat_hash_set<string> differentiable_functions;
  for (const NodeDef& node : optimized_graph->node()) {
    if (IsSymbolicGradient(node)) {
      const auto* f_attr = gtl::FindOrNull(node.attr(), "f");
      if (f_attr) differentiable_functions.insert(f_attr->func().name());
    }
  }

  // Optimize each function only once.
  absl::flat_hash_set<string> optimized_funcs;
  bool optimize_function_library =
      item.optimization_options().optimize_function_library;

  while (optimize_function_library) {
    optimize_function_library = false;

    for (const FunctionDef& func : optimized_graph->library().function()) {
      GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

      const string& func_name = func.signature().name();

      // Skip functions that are not reachable from the optimized graph.
      if (!flib.Contains(func_name)) continue;

      // Skip already optimized functions.
      if (optimized_funcs.find(func_name) != optimized_funcs.end()) continue;

      // Skip parametrized functions (function type or body is defined only at
      // function call time by caller node attributes).
      // They should be specialized to their instantiation type parameters by
      // the function optimizer, before we can optimize function body.
      if (IsParametrized(func)) continue;

      VLOG(3) << "Optimize function: function=" << func_name;

      // Function optimization might specialize nested function calls, so we
      // have to reset the flag and do at least one more pass over the library.
      optimize_function_library = true;
      optimized_funcs.insert(func_name);

      // Make a GrapplerItem from a FunctionDef.
      GrapplerFunctionItem func_item;
      TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(
          func, flib, trimmed_item.graph.versions().producer(), &func_item));

      // If we need to compute the gradient of optimized function at runtime, we
      // can't perform non-differentiable rewrites.
      if (differentiable_functions.find(func_name) !=
          differentiable_functions.end()) {
        func_item.optimization_options().allow_non_differentiable_rewrites =
            false;
      }

      // Device set available to the function is defined only by the runtime,
      // when we instantiate and execute the function. We can't use all devices
      // available to the main graph, because after partitioning the function
      // call node might execute on a remote worker.
      if (!func_item.devices().empty()) {
        return errors::Internal("GrapplerFunctionItem devices must be empty.");
      }

      // We are not allowed to prune certain types of ops from the graph
      // instantiated by the function definition, because we must guarantee
      // function execution semantics wrt side effects (see
      // function_optimizer.cc).
      func_item.optimization_options().allow_pruning_stateful_and_dataset_ops =
          false;

      // TODO(b/129545186): Shape inference in GraphProperties doesn't work well
      // with _Arg nodes. Replace them with Placeholders with unknown shape.
      absl::flat_hash_set<absl::string_view> input_nodes;
      for (auto& input_arg : func_item.inputs()) {
        input_nodes.insert(input_arg.node_name);
      }
      for (NodeDef& func_node : *func_item.graph.mutable_node()) {
        if (input_nodes.contains(func_node.name())) {
          func_node.set_op("Placeholder");
          auto& attrs = *func_node.mutable_attr();
          attrs["dtype"] = attrs["T"];
          attrs.erase("index");
          attrs.erase("T");
          TensorShapeProto unknown_shape;
          unknown_shape.set_unknown_rank(true);
          *(attrs["shape"].mutable_shape()) = unknown_shape;
        }
      }

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
      TF_RETURN_IF_ERROR(flib.ReplaceFunction(func_name, optimized_func));
    }

    // If optimized at least one function, update the graph library.
    if (optimize_function_library) {
      *optimized_graph->mutable_library() = flib.ToProto();
    }
  }

  VLOG(1) << "Optimized " << optimized_funcs.size()
          << " functions: " << str_util::Join(optimized_funcs, ", ");

  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile(
        strings::StrCat("after_MetaOptimizer_",
                        reinterpret_cast<uintptr_t>(optimized_graph)),
        *optimized_graph);
  }
  return Status::OK();
}

void MetaOptimizer::PrintResult() {
  for (const GraphOptimizationResult& graph_result : optimization_results_) {
    LOG(INFO) << "Optimization results for grappler item: " << graph_result.id;
    for (const OptimizerResult& result : graph_result.results) {
      LOG(INFO) << "  " << result.optimizer_name << ": " << result.message;
    }
  }
}

void MetaOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                             const GraphDef& optimized_graph, double result) {
  // Nothing to do for MetaOptimizer.
}

bool MetaOptimizerEnabled(const ConfigProto& cfg) {
  const auto& rewrite_cfg = cfg.graph_options().rewrite_options();
  if (rewrite_cfg.disable_meta_optimizer()) {
    return false;
  }
  return !rewrite_cfg.disable_model_pruning() ||
         rewrite_cfg.layout_optimizer() != RewriterConfig::OFF ||
         rewrite_cfg.function_optimization() != RewriterConfig::OFF ||
         rewrite_cfg.constant_folding() != RewriterConfig::OFF ||
         rewrite_cfg.shape_optimization() != RewriterConfig::OFF ||
         rewrite_cfg.remapping() != RewriterConfig::OFF ||
         rewrite_cfg.arithmetic_optimization() != RewriterConfig::OFF ||
         rewrite_cfg.loop_optimization() != RewriterConfig::OFF ||
         rewrite_cfg.dependency_optimization() != RewriterConfig::OFF ||
         rewrite_cfg.auto_parallel().enable() ||
         rewrite_cfg.memory_optimization() != RewriterConfig::NO_MEM_OPT ||
         rewrite_cfg.debug_stripper() == RewriterConfig::ON ||
         rewrite_cfg.scoped_allocator_optimization() == RewriterConfig::ON ||
         rewrite_cfg.pin_to_host_optimization() == RewriterConfig::ON ||
         !rewrite_cfg.optimizers().empty() ||
         !rewrite_cfg.custom_optimizers().empty();
}

Status RunMetaOptimizer(const GrapplerItem& item, const ConfigProto& cfg,
                        DeviceBase* cpu_device, Cluster* cluster,
                        GraphDef* optimized_graph) {
  MetaOptimizer optimizer(cpu_device, cfg);
  optimizer.set_deadline_usec(
      DeadlineMicroSeconds(cfg.graph_options().rewrite_options()));
  Status status = optimizer.Optimize(cluster, item, optimized_graph);
  if (!status.ok()) {
    *optimized_graph = item.graph;
  }
  return status;
}

Status OptimizeGraph(
    std::vector<string> ret_node_names, std::vector<string> keep_node_names,
    FunctionLibraryDefinition* flib, const DeviceSet& device_set,
    Device* cpu_device, const ConfigProto& config_proto,
    const string& grappler_item_id,
    const GrapplerItem::OptimizationOptions& optimization_options,
    std::unique_ptr<tensorflow::Graph>* g) {
  if (!tensorflow::grappler::MetaOptimizerEnabled(config_proto)) {
    return Status::OK();
  }

  tensorflow::grappler::GrapplerItem item;
  item.id = grappler_item_id;
  item.optimization_options() = optimization_options;

  // Add all available devices so that inlined function can be placed.
  for (const Device* d : device_set.devices()) {
    Status added_device = item.AddDevice(d->name());
    if (!added_device.ok()) VLOG(3) << added_device.error_message();
  }

  // Add fetches so that the graph can be pruned.
  item.fetch.swap(ret_node_names);

  // Add noes that can't be removed from the graph.
  item.keep_ops = std::move(keep_node_names);

  (*g)->ToGraphDef(&item.graph);

  if (flib) {
    *item.graph.mutable_library() = flib->ToProto();
  }

  tensorflow::GraphDef out_graph;

  tensorflow::grappler::VirtualCluster cluster(&device_set);

  // TODO(nareshmodi): Consider adding and using the more generic GraphOptions
  // proto (which also contain the OptimizerOptions).
  TF_RETURN_IF_ERROR(tensorflow::grappler::RunMetaOptimizer(
      item, config_proto, cpu_device, &cluster, &out_graph));

  std::unique_ptr<tensorflow::Graph> optimized_graph(
      new tensorflow::Graph(OpRegistry::Global()));
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(GraphConstructorOptions(),
                                            out_graph, optimized_graph.get()));

  // Copy optimized functions back to the overlay lib.
  if (flib) {
    for (const FunctionDef& fdef : out_graph.library().function()) {
      const string& func_name = fdef.signature().name();
      if (flib->Contains(func_name)) {
        TF_RETURN_IF_ERROR(flib->ReplaceFunction(func_name, fdef));
      } else {
        TF_RETURN_IF_ERROR(flib->AddFunctionDef(fdef));
      }
    }
  }

  *g = std::move(optimized_graph);

  // The graph conversion sets the requested device names but not the
  // assigned device names. However, since at this point the graph is
  // placed TF expects an assigned device name for every node. Therefore
  // we copy the requested device into the assigned device field.
  for (Node* node : (*g)->nodes()) {
    if (node->IsOp() && node->assigned_device_name().empty()) {
      if (node->requested_device().empty()) {
        return errors::Internal(
            "Either placer did not place the node or Grappler did not "
            "copy the assigned device. Contact Grappler team since latter "
            "is more likely. Node=",
            node->name(), " Graph: ", (*g)->ToGraphDefDebug().DebugString());
      }
      node->set_assigned_device_name(node->requested_device());
    }
  }

  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
