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

#include <algorithm>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/auto_mixed_precision.h"
#include "tensorflow/core/grappler/optimizers/auto_parallel.h"
#include "tensorflow/core/grappler/optimizers/common_subgraph_elimination.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/debug_stripper.h"
#include "tensorflow/core/grappler/optimizers/dependency_optimizer.h"
#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer.h"
#include "tensorflow/core/grappler/optimizers/implementation_selector.h"
#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"
#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/optimizers/pin_to_host_optimizer.h"
#include "tensorflow/core/grappler/optimizers/remapper.h"
#include "tensorflow/core/grappler/optimizers/scoped_allocator_optimizer.h"
#include "tensorflow/core/grappler/optimizers/shape_optimizer.h"
#include "tensorflow/core/grappler/utils/canonicalizer.h"
#include "tensorflow/core/grappler/utils/colocation.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/tpu.h"
#include "tensorflow/core/grappler/verifiers/structure_verifier.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/xla_config_registry.h"

// #TODO(b/200087693): LLVM does not build on Fuchsia.
#if !NO_LLVM_SUPPORT
#include "tensorflow/core/grappler/optimizers/tfg_optimizer_hook.h"
#include "tensorflow/core/grappler/optimizers/tfg_passes_builder.h"
#endif

namespace tensorflow {
namespace grappler {

namespace {

constexpr int kDefaultNumberOfIterations = 2;
constexpr int kDefaultMinGraphNodes = 4;
constexpr char kGrapplerCategory[] = "Grappler";

int64_t NumEdges(const GraphDef& graph) {
  int64_t num_edges = 0;
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
         name == "loop_optimizer" ||
         absl::StartsWith(name, "auto_mixed_precision");
}

// Creates a function library stub from a real function library: copy only
// signatures and attributes of all the function defined in fdef_lib. This stub
// can be swapped with real function library in a graph, before passing it to
// optimizer, if optimizer doesn't instantiate functions.
FunctionDefLibrary GetFunctionDefLibraryStub(
    const FunctionDefLibrary& fdef_lib) {
  FunctionDefLibrary stub;
  for (const FunctionDef& fn : fdef_lib.function()) {
    FunctionDef* fn_stub = stub.mutable_function()->Add();
    *(fn_stub->mutable_signature()) = fn.signature();
    *(fn_stub->mutable_attr()) = fn.attr();
    *(fn_stub->mutable_arg_attr()) = fn.arg_attr();
    *(fn_stub->mutable_resource_arg_unique_id()) = fn.resource_arg_unique_id();
  }
  *stub.mutable_gradient() = fdef_lib.gradient();
  return stub;
}

uint64 DeadlineMicroSeconds(const RewriterConfig& cfg) {
  if (cfg.meta_optimizer_timeout_ms() <= 0) return 0;  // no deadline
  return Env::Default()->NowMicros() + cfg.meta_optimizer_timeout_ms() * 1000;
}

// A helper function to decide whether to enable the automatic mixed precision
// optimizer.
bool AutoMixedPrecisionEnabled(RewriterConfig::Toggle opt_level) {
  if (opt_level == RewriterConfig::ON ||
      opt_level == RewriterConfig::AGGRESSIVE) {
    return true;
  } else if (opt_level == RewriterConfig::EXPERIMENTAL_MLIR ||
             opt_level == RewriterConfig::EXPERIMENTAL_BOTH) {
    VLOG(2) << "auto_mixed_precision is not implemented in TFG yet";
  }
  return false;
}

bool IsXlaGlobalJitOn(
    const OptimizerOptions::GlobalJitLevel& jit_level_in_session_opts) {
  xla_config_registry::XlaGlobalJitLevel xla_global_jit_level =
      xla_config_registry::GetGlobalJitLevel(jit_level_in_session_opts);
  // Return true only if XLA JIT is ON for both single-gpu and multi-gpu
  // graphs. This is a conservative approach that turns off the memory optimizer
  // when we are sure that all graphs will be processed by XLA JIT.
  return xla_global_jit_level.single_gpu >= OptimizerOptions::ON_1 &&
         xla_global_jit_level.general >= OptimizerOptions::ON_1;
}

// A helper function to decide whether to enable the memory optimizer.
bool MemoryOptimizerEnabled(RewriterConfig::MemOptType mem_opt_type,
                            bool xla_auto_clustering_on) {
  // Disable the default memory optimizer when XLA JIT is ON as it hurts the
  // XLA JIT performance. The (current) XLA clustering can result in loss of
  // concurrency between kernel compute and memory copies. As such, it usually
  // loses the concurrency needed to hide the latencies of the inserted swap-ins
  // and swap-outs and incurs great performance overhead. Remove this check when
  // the XLA JIT can better deal with the concurrency.
  if (mem_opt_type == RewriterConfig::DEFAULT_MEM_OPT &&
      xla_auto_clustering_on) {
    return false;
  }

  return mem_opt_type != RewriterConfig::NO_MEM_OPT;
}

Status GetGraphDevice(const GraphDef& g_def, std::set<std::string>* devices) {
  for (auto& node : g_def.node()) {
    DeviceNameUtils::ParsedName parsed_name;
    if (!DeviceNameUtils::ParseFullName(node.device(), &parsed_name)) {
      return errors::InvalidArgument("Unable to parse ", node.device(),
                                     " as a device name");
    }
    devices->insert(parsed_name.type);
  }
  return absl::OkStatus();
}

}  // namespace

#define MK_OPT(NAME, CONFIG, VALUE)                                    \
  if (optimizer == NAME) {                                             \
    if (plugin_configs.toggle_config[CONFIG] != RewriterConfig::OFF) { \
      return std::unique_ptr<GraphOptimizer>(VALUE);                   \
    }                                                                  \
  }

bool MetaOptimizer::LowerControlFlow() const {
  if (config_proto_.experimental().executor_type() ==
      "SINGLE_THREADED_EXECUTOR")
    return false;

  if (config_proto_.experimental().use_tfrt()) return false;

  return true;
}

std::unique_ptr<GraphOptimizer> MetaOptimizer::MakeNewOptimizer(
    const string& optimizer, const std::set<string>& device_types) const {
  ConfigList plugin_configs = PluginGraphOptimizerRegistry::GetPluginConfigs(
      cfg_.use_plugin_optimizers() != RewriterConfig::OFF, device_types);
  if (optimizer == "pruning" && !plugin_configs.disable_model_pruning)
    return std::unique_ptr<GraphOptimizer>(new ModelPruner());
  MK_OPT("function", "function_optimization",
         new FunctionOptimizer(cfg_.function_optimization(),
                               /*lower_control_flow=*/LowerControlFlow()));
  MK_OPT("constfold", "constant_folding",
         new ConstantFolding(
             cpu_device_,
             cfg_.experimental_disable_compressed_tensor_optimization(),
             !cfg_.experimental_disable_folding_quantization_emulation()));
  MK_OPT("shape", "shape_optimization", new ShapeOptimizer());
  MK_OPT("remap", "remapping",
         new Remapper(cfg_.remapping(), cfg_.cpu_layout_conversion(),
                      xla_auto_clustering_on_));
  MK_OPT("layout", "layout_optimizer",
         new GenericLayoutOptimizer(
             /*optimization level*/ cfg_.layout_optimizer(),
             /*CPU layout conversion*/ cfg_.cpu_layout_conversion()));
  MK_OPT("auto_mixed_precision", "auto_mixed_precision",
         new AutoMixedPrecision(AutoMixedPrecisionMode::CUDA));
#ifdef INTEL_MKL
  if (IsMKLEnabled()) {
    MK_OPT("auto_mixed_precision", "auto_mixed_precision",
           new AutoMixedPrecision(AutoMixedPrecisionMode::FP16_CPU));
    MK_OPT("auto_mixed_precision_mkl", "auto_mixed_precision_mkl",
           new AutoMixedPrecision(AutoMixedPrecisionMode::BF16));
    MK_OPT("auto_mixed_precision_onednn_bfloat16",
           "auto_mixed_precision_onednn_bfloat16",
           new AutoMixedPrecision(AutoMixedPrecisionMode::BF16));
  }
#endif
  MK_OPT("auto_mixed_precision_cpu", "auto_mixed_precision_cpu",
         new AutoMixedPrecision(AutoMixedPrecisionMode::CPU));
  MK_OPT("memory", "memory_optimization",
         new MemoryOptimizer(RewriterConfig::MANUAL));
  MK_OPT("common_subgraph_elimination", "common_subgraph_elimination",
         new CommonSubgraphElimination(cfg_.common_subgraph_elimination()));
  MK_OPT("arithmetic", "arithmetic_optimization",
         new ArithmeticOptimizer(cfg_.arithmetic_optimization()));
  MK_OPT("autoparallel", "auto_parallel",
         new AutoParallel(cfg_.auto_parallel().num_replicas()));
  MK_OPT("loop", "loop_optimization",
         new LoopOptimizer(cfg_.loop_optimization(), cpu_device_));
  MK_OPT("dependency", "dependency_optimization",
         new DependencyOptimizer(cfg_.dependency_optimization()));
  MK_OPT("debug_stripper", "debug_stripper", new DebugStripper());
  MK_OPT("scoped_allocator", "scoped_allocator_optimization",
         new ScopedAllocatorOptimizer(cfg_.scoped_allocator_optimization(),
                                      cfg_.scoped_allocator_opts()));
  MK_OPT("pin_to_host", "pin_to_host_optimization",
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
  auto global_jit_level =
      cfg.graph_options().optimizer_options().global_jit_level();
  xla_auto_clustering_on_ = IsXlaGlobalJitOn(global_jit_level);
}

Status MetaOptimizer::InitializeOptimizers(
    const std::set<string>& device_types,
    std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const {
  if (cfg_.disable_meta_optimizer()) {
    return absl::OkStatus();
  }

  ConfigList plugin_configs = PluginGraphOptimizerRegistry::GetPluginConfigs(
      cfg_.use_plugin_optimizers() != RewriterConfig::OFF, device_types);
  if (!cfg_.disable_model_pruning() && !plugin_configs.disable_model_pruning) {
    optimizers->push_back(std::make_unique<ModelPruner>());
  }

  // #TODO(b/200087693): LLVM does not build on Fuchsia.
#if !NO_LLVM_SUPPORT
  if (!cfg_.disable_tfg_optimizer()) {
    // Hooks the MLIR optimizer, it won't run any optimizations right now. This
    // optimizer instance runs on functions one at a time; don't use any
    // threads.
    optimizers->push_back(std::make_unique<mlir::tfg::TFGGrapplerOptimizer>(
        mlir::tfg::DefaultGrapplerPipeline));
  }
#endif

// A set of macro utilities which check if the toggle of an optimization.
// Support both user and plugin configurations.
#define USER_IS_ON(CFG) cfg_.CFG() == RewriterConfig::ON
#define USER_IS_EXPERIMENTAL_MLIR(CFG) \
  cfg_.CFG() == RewriterConfig::EXPERIMENTAL_MLIR
#define USER_IS_EXPERIMENTAL_BOTH(CFG) \
  cfg_.CFG() == RewriterConfig::EXPERIMENTAL_BOTH
#define USER_NOT_OFF(CFG) cfg_.CFG() != RewriterConfig::OFF
#define PLUGIN_IS_ON(CFG) \
  plugin_configs.toggle_config[#CFG] == RewriterConfig::ON
#define PLUGIN_IS_EXPERIMENTAL_MLIR(CFG) \
  plugin_configs.toggle_config[#CFG] == RewriterConfig::EXPERIMENTAL_MLIR
#define PLUGIN_IS_EXPERIMENTAL_BOTH(CFG) \
  plugin_configs.toggle_config[#CFG] == RewriterConfig::EXPERIMENTAL_BOTH
#define PLUGIN_NOT_OFF(CFG) \
  plugin_configs.toggle_config[#CFG] != RewriterConfig::OFF
#define BOTH_ARE_ON(CFG) (USER_IS_ON(CFG) && PLUGIN_IS_ON(CFG))
#define BOTH_NOT_OFF(CFG) (USER_NOT_OFF(CFG) && PLUGIN_NOT_OFF(CFG))
#define BOTH_ARE_EXPERIMENTAL_MLIR(CFG) \
  (USER_IS_EXPERIMENTAL_MLIR(CFG) && PLUGIN_IS_EXPERIMENTAL_MLIR(CFG))
#define BOTH_ARE_EXPERIMENTAL_BOTH(CFG) \
  (USER_IS_EXPERIMENTAL_BOTH(CFG) && PLUGIN_IS_EXPERIMENTAL_BOTH(CFG))
  if (BOTH_NOT_OFF(implementation_selector)) {
    if (USER_IS_EXPERIMENTAL_MLIR(implementation_selector) ||
        USER_IS_EXPERIMENTAL_BOTH(implementation_selector))
      VLOG(2) << "implementation_selector is not implemented in TFG yet";
    else
      optimizers->push_back(std::make_unique<ImplementationSelector>());
  }
  if (BOTH_NOT_OFF(function_optimization)) {
    if (USER_IS_EXPERIMENTAL_MLIR(function_optimization) ||
        USER_IS_EXPERIMENTAL_BOTH(function_optimization)) {
      VLOG(2) << "function_optimization is not implemented in TFG yet";
    } else {
      optimizers->push_back(std::make_unique<FunctionOptimizer>(
          cfg_.function_optimization(),
          /*lower_control_flow=*/LowerControlFlow()));
    }
  }
  if (BOTH_NOT_OFF(common_subgraph_elimination) &&
      BOTH_NOT_OFF(arithmetic_optimization)) {
    if (USER_IS_EXPERIMENTAL_MLIR(common_subgraph_elimination) ||
        USER_IS_EXPERIMENTAL_BOTH(common_subgraph_elimination)) {
      VLOG(2) << "common_subgraph_elimination is not implemented in TFG yet";
    } else {
      optimizers->push_back(std::make_unique<CommonSubgraphElimination>(
          cfg_.common_subgraph_elimination()));
    }
  }
  if (BOTH_ARE_ON(debug_stripper))
    optimizers->push_back(std::make_unique<DebugStripper>());
  else if (BOTH_ARE_EXPERIMENTAL_MLIR(debug_stripper) ||
           BOTH_ARE_EXPERIMENTAL_BOTH(debug_stripper))
    VLOG(2) << "debug_stripper is not implemented in TFG yet";
  if (BOTH_NOT_OFF(constant_folding)) {
    if (USER_IS_EXPERIMENTAL_MLIR(constant_folding) ||
        USER_IS_EXPERIMENTAL_BOTH(constant_folding)) {
      VLOG(2) << "constant_folding is not implemented in TFG yet";
    } else {
      optimizers->push_back(std::make_unique<ConstantFolding>(
          cfg_.constant_folding(), cpu_device_,
          cfg_.experimental_disable_compressed_tensor_optimization(),
          !cfg_.experimental_disable_folding_quantization_emulation()));
    }
  }
  if (BOTH_NOT_OFF(shape_optimization)) {
    if (USER_IS_EXPERIMENTAL_MLIR(shape_optimization) ||
        USER_IS_EXPERIMENTAL_BOTH(shape_optimization))
      VLOG(2) << "shape_optimization is not implemented in TFG yet";
    else
      optimizers->push_back(std::make_unique<ShapeOptimizer>());
  }
  if (AutoMixedPrecisionEnabled(cfg_.auto_mixed_precision()) &&
      AutoMixedPrecisionEnabled(
          plugin_configs.toggle_config["auto_mixed_precision"])) {
    optimizers->push_back(
        std::make_unique<AutoMixedPrecision>(AutoMixedPrecisionMode::FP16_CPU));
    optimizers->push_back(
        std::make_unique<AutoMixedPrecision>(AutoMixedPrecisionMode::CUDA));
  }
#ifdef INTEL_MKL
  if (AutoMixedPrecisionEnabled(cfg_.auto_mixed_precision_onednn_bfloat16()) &&
      AutoMixedPrecisionEnabled(
          plugin_configs
              .toggle_config["auto_mixed_precision_onednn_bfloat16"]) &&
      IsMKLEnabled()) {
    optimizers->push_back(
        std::make_unique<AutoMixedPrecision>(AutoMixedPrecisionMode::BF16));
  }
  if (AutoMixedPrecisionEnabled(cfg_.auto_mixed_precision_mkl()) &&
      AutoMixedPrecisionEnabled(
          plugin_configs.toggle_config["auto_mixed_precision_mkl"]) &&
      IsMKLEnabled()) {
    LOG_FIRST_N(WARNING, 1)
        << "NOTE: auto_mixed_precision_mkl is deprecated."
           " Please use auto_mixed_precision_onednn_bfloat16 instead";
    optimizers->push_back(
        std::make_unique<AutoMixedPrecision>(AutoMixedPrecisionMode::BF16));
  }
#endif
  if (AutoMixedPrecisionEnabled(cfg_.auto_mixed_precision_cpu()) &&
      AutoMixedPrecisionEnabled(
          plugin_configs.toggle_config["auto_mixed_precision_cpu"])) {
    optimizers->push_back(
        std::make_unique<AutoMixedPrecision>(AutoMixedPrecisionMode::CPU));
  }
  if (BOTH_ARE_ON(pin_to_host_optimization))
    optimizers->push_back(std::make_unique<PinToHostOptimizer>());
  else if (BOTH_ARE_EXPERIMENTAL_MLIR(pin_to_host_optimization) ||
           BOTH_ARE_EXPERIMENTAL_BOTH(pin_to_host_optimization))
    VLOG(2) << "pin_to_host_optimization is not implemented in TFG yet";
  if (BOTH_NOT_OFF(arithmetic_optimization)) {
    if (USER_IS_EXPERIMENTAL_MLIR(arithmetic_optimization) ||
        USER_IS_EXPERIMENTAL_BOTH(arithmetic_optimization)) {
      VLOG(2) << "arithmetic_optimization is not implemented in TFG yet";
    } else {
      optimizers->push_back(std::make_unique<ArithmeticOptimizer>(
          cfg_.arithmetic_optimization()));
    }
  }
  if (BOTH_NOT_OFF(layout_optimizer)) {
    if (USER_IS_EXPERIMENTAL_MLIR(layout_optimizer) ||
        USER_IS_EXPERIMENTAL_BOTH(layout_optimizer)) {
      VLOG(2) << "layout_optimizer is not implemented in TFG yet";
    } else {
      optimizers->push_back(std::make_unique<GenericLayoutOptimizer>(
          /*optimization level*/ cfg_.layout_optimizer(),
          /*CPU layout conversion*/ cfg_.cpu_layout_conversion()));
    }
  }
  if (BOTH_NOT_OFF(remapping)) {
    bool enable_mlir_pass = USER_IS_EXPERIMENTAL_MLIR(remapping) ||
                            USER_IS_EXPERIMENTAL_BOTH(remapping);
    bool enable_grappler_pass =
        !enable_mlir_pass || USER_IS_EXPERIMENTAL_BOTH(remapping);
    if (enable_mlir_pass) {
// #TODO(b/200087693): LLVM does not build on Fuchsia.
#if !NO_LLVM_SUPPORT
      optimizers->push_back(std::make_unique<mlir::tfg::TFGGrapplerOptimizer>(
          mlir::tfg::RemapperPassBuilder));
#else
      VLOG(2) << "mlir Remapper pass is not supported on this platform";
#endif
    }
    if (enable_grappler_pass) {
      optimizers->push_back(std::make_unique<Remapper>(
          cfg_.remapping(), cfg_.cpu_layout_conversion(),
          xla_auto_clustering_on_));
    }
  }
  if (BOTH_NOT_OFF(loop_optimization)) {
    if (USER_IS_EXPERIMENTAL_MLIR(loop_optimization) ||
        USER_IS_EXPERIMENTAL_BOTH(loop_optimization)) {
      VLOG(2) << "loop_optimization is not implemented in TFG yet";
    } else {
      optimizers->push_back(std::make_unique<LoopOptimizer>(
          cfg_.loop_optimization(), cpu_device_));
    }
  }
  if (BOTH_NOT_OFF(dependency_optimization)) {
    if (USER_IS_EXPERIMENTAL_MLIR(dependency_optimization) ||
        USER_IS_EXPERIMENTAL_BOTH(dependency_optimization)) {
      VLOG(2) << "dependency_optimization is not implemented in TFG yet";
    } else {
      optimizers->push_back(std::make_unique<DependencyOptimizer>(
          cfg_.dependency_optimization()));
    }
  }
  if (MemoryOptimizerEnabled(cfg_.memory_optimization(),
                             xla_auto_clustering_on_) &&
      PLUGIN_NOT_OFF(memory_optimization)) {
    if (cfg_.memory_optimizer_target_node_name_scope().empty()) {
      optimizers->push_back(
          // Use the default target node name prefix "gradients/"
          std::make_unique<MemoryOptimizer>(cfg_.memory_optimization()));
    } else {
      optimizers->push_back(std::make_unique<MemoryOptimizer>(
          cfg_.memory_optimization(),
          cfg_.memory_optimizer_target_node_name_scope()));
    }
  }
  if (cfg_.auto_parallel().enable() && PLUGIN_IS_ON(auto_parallel)) {
    optimizers->push_back(
        std::make_unique<AutoParallel>(cfg_.auto_parallel().num_replicas()));
  }

#ifndef ENABLE_MKL
  if (BOTH_ARE_ON(scoped_allocator_optimization)) {
    optimizers->push_back(std::make_unique<ScopedAllocatorOptimizer>(
        cfg_.scoped_allocator_optimization(), cfg_.scoped_allocator_opts()));
  } else if (BOTH_ARE_EXPERIMENTAL_MLIR(scoped_allocator_optimization) ||
             BOTH_ARE_EXPERIMENTAL_BOTH(scoped_allocator_optimization)) {
    VLOG(2) << "scoped_allocator_optimization is not implemented in TFG yet";
  }
#endif

#undef USER_IS_ON
#undef USER_IS_EXPERIMENTAL_MLIR
#undef USER_IS_EXPERIMENTAL_BOTH
#undef USER_NOT_OFF
#undef PLUGIN_IS_ON
#undef PLUGIN_NOT_OFF
#undef BOTH_ARE_ON
#undef BOTH_NOT_OFF
  return InitializeCustomGraphOptimizers(device_types, std::set<string>(),
                                         optimizers);
}

Status MetaOptimizer::InitializeOptimizersByName(
    const std::set<string>& device_types,
    std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const {
  std::set<string> initialized_custom_optimizers;
  for (const string& optimizer_name : cfg_.optimizers()) {
    auto optimizer = MakeNewOptimizer(optimizer_name, device_types);
    if (optimizer) {
      VLOG(2) << "Registered default graph optimizer: " << optimizer_name;
      optimizers->push_back(std::move(optimizer));
      continue;
    }

    auto custom_optimizer =
        CustomGraphOptimizerRegistry::CreateByNameOrNull(optimizer_name);

    if (custom_optimizer) {
      VLOG(2) << "Registered custom graph optimizer: " << optimizer_name;
      TF_RETURN_IF_ERROR(custom_optimizer->InitWithConfig(
          config_proto_, GetCustomGraphOptimizerConfig(optimizer_name)));
      optimizers->push_back(std::move(custom_optimizer));
      initialized_custom_optimizers.insert(optimizer_name);
    } else {
      VLOG(2) << "Can't register an optimizer by name: " << optimizer_name;
    }
  }
  return InitializeCustomGraphOptimizers(
      device_types, initialized_custom_optimizers, optimizers);
}

Status MetaOptimizer::InitializeCustomGraphOptimizers(
    const std::set<string>& device_types,
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
      TF_RETURN_IF_ERROR(
          custom_optimizer->InitWithConfig(config_proto_, &optimizer_config));
      optimizers->push_back(std::move(custom_optimizer));
    } else {
      // If there are no custom optimizers with given name, try to initialize a
      // default optimizer. This way, custom configurable optimizers can be
      // mixed with default optimizers in any order.
      auto optimizer = MakeNewOptimizer(optimizer_config.name(), device_types);
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
  return InitializePluginGraphOptimizers(device_types, optimizers);
}

Status MetaOptimizer::InitializePluginGraphOptimizers(
    const std::set<string>& device_types,
    std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const {
  if (cfg_.use_plugin_optimizers() == RewriterConfig::OFF)
    return absl::OkStatus();
  auto plugin_optimizers =
      PluginGraphOptimizerRegistry::CreateOptimizers(device_types);
  for (auto& plugin_optimizer : plugin_optimizers) {
    optimizers->push_back(std::move(plugin_optimizer));
  }
  return absl::OkStatus();
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
    inter_optimizer_verifiers->push_back(std::make_unique<StructureVerifier>());
  }
  if (cfg_.post_optimization_verifier_config().structure_verifier() ==
      VerifierConfig::ON) {
    post_optimization_verifiers->push_back(
        std::make_unique<StructureVerifier>());
  }
}

void MetaOptimizer::PrintUserAndPluginConfigs(
    const std::set<string>& device_types) const {
  if (cfg_.use_plugin_optimizers() == RewriterConfig::OFF) return;
  ConfigList plugin_cfg = PluginGraphOptimizerRegistry::GetPluginConfigs(
      cfg_.use_plugin_optimizers() != RewriterConfig::OFF, device_types);
  PluginGraphOptimizerRegistry::PrintPluginConfigsIfConflict(device_types);

  ConfigList user_cfg;
  // Print user's and plugin's configs.
  if (cfg_.optimizers().empty()) {
    if (cfg_.disable_meta_optimizer()) {
      return;
    }
    user_cfg.disable_model_pruning = cfg_.disable_model_pruning();
#define PRINT_CFG(CFG) user_cfg.toggle_config[#CFG] = cfg_.CFG();
    PRINT_CFG(implementation_selector)
    PRINT_CFG(function_optimization)
    PRINT_CFG(common_subgraph_elimination)
    PRINT_CFG(arithmetic_optimization)
    PRINT_CFG(debug_stripper)
    PRINT_CFG(constant_folding)
    PRINT_CFG(shape_optimization)
    PRINT_CFG(pin_to_host_optimization)
    PRINT_CFG(layout_optimizer)
    PRINT_CFG(remapping)
    PRINT_CFG(loop_optimization)
    PRINT_CFG(dependency_optimization)
    PRINT_CFG(scoped_allocator_optimization)
#undef PRINT_CFG
    user_cfg.toggle_config["auto_mixed_precision"] =
        AutoMixedPrecisionEnabled(cfg_.auto_mixed_precision())
            ? RewriterConfig::ON
            : RewriterConfig::OFF;
    user_cfg.toggle_config["auto_mixed_precision_onednn_bfloat16"] =
        AutoMixedPrecisionEnabled(cfg_.auto_mixed_precision_onednn_bfloat16())
            ? RewriterConfig::ON
            : RewriterConfig::OFF;
    user_cfg.toggle_config["auto_mixed_precision_mkl"] =
        AutoMixedPrecisionEnabled(cfg_.auto_mixed_precision_mkl())
            ? RewriterConfig::ON
            : RewriterConfig::OFF;
    user_cfg.toggle_config["auto_mixed_precision_cpu"] =
        AutoMixedPrecisionEnabled(cfg_.auto_mixed_precision_cpu())
            ? RewriterConfig::ON
            : RewriterConfig::OFF;
    user_cfg.toggle_config["memory_optimization"] =
        MemoryOptimizerEnabled(cfg_.memory_optimization(),
                               config_proto_.graph_options()
                                   .optimizer_options()
                                   .global_jit_level())
            ? RewriterConfig::ON
            : RewriterConfig::OFF;
    user_cfg.toggle_config["auto_parallel"] = cfg_.auto_parallel().enable()
                                                  ? RewriterConfig::ON
                                                  : RewriterConfig::OFF;
  } else {
    for (const string& optimizer_name : cfg_.optimizers()) {
      if (optimizer_name == "pruning") user_cfg.disable_model_pruning = true;

#define PRINT_CFG(NAME, CONFIG) \
  if (optimizer_name == NAME)   \
    user_cfg.toggle_config[CONFIG] = RewriterConfig::ON;

      PRINT_CFG("implementation_selector", "implementation_selector")
      PRINT_CFG("function", "function_optimization")
      PRINT_CFG("common_subgraph_elimination", "common_subgraph_elimination")
      PRINT_CFG("arithmetic", "arithmetic_optimization")
      PRINT_CFG("debug_stripper", "debug_stripper")
      PRINT_CFG("constfold", "constant_folding")
      PRINT_CFG("shape", "shape_optimization")
      PRINT_CFG("auto_mixed_precision", "auto_mixed_precision")
      PRINT_CFG("auto_mixed_precision_onednn_bfloat16",
                "auto_mixed_precision_onednn_bfloat16")
      PRINT_CFG("auto_mixed_precision_mkl", "auto_mixed_precision_mkl")
      PRINT_CFG("auto_mixed_precision_cpu", "auto_mixed_precision_cpu")
      PRINT_CFG("pin_to_host", "pin_to_host_optimization")
      PRINT_CFG("layout", "layout_optimizer")
      PRINT_CFG("remap", "remapping")
      PRINT_CFG("loop", "loop_optimization")
      PRINT_CFG("dependency", "dependency_optimization")
      PRINT_CFG("memory", "memory_optimization")
      PRINT_CFG("autoparallel", "auto_parallel")
      PRINT_CFG("scoped_allocator", "scoped_allocator_optimization")
#undef PRINT_CFG
    }
  }

  // Print logs only when plugin config has conflict with user config.
  if (!PluginGraphOptimizerRegistry::IsConfigsConflict(user_cfg, plugin_cfg))
    return;

  ConfigList final_cfg = user_cfg;
  // If plugin turns on `disable_model_pruning`, then `disable_model_pruning`
  // should be true;
  if (plugin_cfg.disable_model_pruning == true)
    final_cfg.disable_model_pruning = true;
  // If plugin turns off a certain optimizer, then the optimizer should be
  // turned off;
  for (auto& pair : plugin_cfg.toggle_config) {
    if (plugin_cfg.toggle_config[pair.first] == RewriterConfig::OFF)
      final_cfg.toggle_config[pair.first] = RewriterConfig::OFF;
  }

  string logs =
      "\nConfig of optimizers\t\tUser's config\tPlugin's config\tFinal "
      "config(User & Plugin)\n";
  strings::StrAppend(&logs, "disable_model_pruning\t\t",
                     user_cfg.disable_model_pruning, "\t\t",
                     plugin_cfg.disable_model_pruning, "\t\t",
                     final_cfg.disable_model_pruning, "\n");
  for (auto& pair : user_cfg.toggle_config) {
    if (pair.first == "debug_stripper" ||
        pair.first == "auto_mixed_precision" ||
        pair.first == "auto_mixed_precision_onednn_bfloat16" ||
        pair.first == "auto_mixed_precision_mkl" ||
        pair.first == "auto_mixed_precision_cpu" ||
        pair.first == "pin_to_host_optimization" ||
        pair.first == "scoped_allocator_optimization") {
      // These optimizers are turned off by default.
      // TODO(penporn): Remove the hard-coded length and change it to max length
      // of all option strings.
      strings::StrAppend(
          &logs, pair.first, string(40 - pair.first.size(), ' '),
          (pair.second == RewriterConfig::ON), "\t\t",
          (plugin_cfg.toggle_config[pair.first] == RewriterConfig::ON), "\t\t",
          (final_cfg.toggle_config[pair.first] == RewriterConfig::ON), "\n");
    } else {
      // These optimizers are turned on by default.
      // TODO(penporn): Remove the hard-coded length and change it to max length
      // of all option strings.
      strings::StrAppend(
          &logs, pair.first, string(40 - pair.first.size(), ' '),
          (pair.second != RewriterConfig::OFF), "\t\t",
          (plugin_cfg.toggle_config[pair.first] != RewriterConfig::OFF), "\t\t",
          (final_cfg.toggle_config[pair.first] != RewriterConfig::OFF), "\n");
    }
  }
  LOG(WARNING) << "User's config has been changed based on plugin's config.";
  LOG(WARNING) << logs;
}

Status MetaOptimizer::OptimizeGraph(
    const std::vector<std::unique_ptr<GraphOptimizer>>& optimizers,
    Cluster* cluster, GrapplerItem&& item, GraphDef* optimized_graph) {
  int min_graph_nodes = cfg_.min_graph_nodes() == 0 ? kDefaultMinGraphNodes
                                                    : cfg_.min_graph_nodes();
  if (item.graph.node_size() < min_graph_nodes) {
    VLOG(3) << "Skipping optimization, graph has less than " << min_graph_nodes
            << " nodes.";
    *optimized_graph = item.graph;
    return absl::OkStatus();
  }

  tensorflow::metrics::ScopedCounter<2> timings(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {kGrapplerCategory, "OptimizeMainGraph"});

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
    return absl::OkStatus();
  }

  // Invariant: optimized_graph contains the most recently optimized version of
  // the graph.
  auto original_producer = item.graph.versions().producer();
  *optimized_graph = std::move(item.graph);

  GraphOptimizationResult optimization_result(item.id);
#ifndef ENABLE_MKL
  GraphOptimizer* sa_optimizer = nullptr;
#endif

  // Constants in the graph are normally compressed after model_pruner.
  // Do it here if model pruner is disabled.
  if (cfg_.disable_model_pruning()) {
    CompressConstants(optimized_graph);
  }

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
#ifndef ENABLE_MKL
      // Some must run only on the last iteration.
      if (optimizer->name() == "scoped_allocator_optimizer") {
        if (sa_optimizer == nullptr) sa_optimizer = optimizer.get();
        continue;
      }
#endif

      TF_RETURN_IF_ERROR(RunOptimizer(optimizer.get(), cluster, &item,
                                      optimized_graph, &optimization_result));

      if (iteration == 0 && optimizer->name() == "model_pruner") {
        CompressConstants(optimized_graph);
      }

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
#ifndef ENABLE_MKL
  // ScopedAllocatorOptimizer must run last.
  if (sa_optimizer != nullptr) {
    TF_RETURN_IF_ERROR(RunOptimizer(sa_optimizer, cluster, &item,
                                    optimized_graph, &optimization_result));
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
  }
#endif

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
    DCHECK_EQ(optimized_graph->versions().producer(), original_producer);
  }

  return absl::OkStatus();
}

Status MetaOptimizer::OptimizeGraph(Cluster* cluster, GrapplerItem&& item,
                                    GraphDef* optimized_graph) {
  std::vector<std::unique_ptr<GraphOptimizer>> optimizers;
  std::set<std::string> device_types;
  TF_RETURN_IF_ERROR(GetGraphDevice(item.graph, &device_types));
  if (cfg_.optimizers().empty()) {
    TF_RETURN_IF_ERROR(InitializeOptimizers(device_types, &optimizers));
  } else {
    TF_RETURN_IF_ERROR(InitializeOptimizersByName(device_types, &optimizers));
  }
  PrintUserAndPluginConfigs(device_types);

  return OptimizeGraph(std::move(optimizers), cluster, std::move(item),
                       optimized_graph);
}

Status MetaOptimizer::RunOptimizer(
    GraphOptimizer* optimizer, Cluster* cluster, GrapplerItem* optimized_item,
    GraphDef* optimized_graph, GraphOptimizationResult* optimization_result) {
  // If optimizer doesn't need a function library, we will replace it with a
  // stub before running optimization, and will put it back at the end.
  std::unique_ptr<FunctionDefLibrary> optimized_graph_function_library;
  const bool is_function_library_aware = optimizer->UsesFunctionLibrary();

  // Replace function library in optimized graph with a stub.
  if (!is_function_library_aware) {
    VLOG(3) << "Replace function library with a stub for " << optimizer->name();
    optimized_graph_function_library =
        absl::WrapUnique(optimized_graph->release_library());
    *optimized_graph->mutable_library() =
        GetFunctionDefLibraryStub(*optimized_graph_function_library);
  }

  // This swaps the current optimized_graph into optimized item and
  // resets optimized_graph to an empty graph.
  optimized_item->graph = std::move(*optimized_graph);
  *optimized_graph = GraphDef();
  optimizer->set_deadline_usec(this->deadline_usec());
  tensorflow::metrics::ScopedCounter<2> timings(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {kGrapplerCategory, optimizer->name()});
  Status status =
      optimizer->Optimize(cluster, *optimized_item, optimized_graph);
  auto duration_ms = timings.DurationMicroSec().value() / 1000.0f;
  timings.ReportAndStop();

  string message;
  if (!status.ok()) {
    *optimized_graph = std::move(optimized_item->graph);
    if (absl::IsAborted(status)) {
      // By convention we (ab-)use the Aborted error code to signal that the
      // optimizer returned without performing any changes to the graph.
      message = strings::StrCat(optimizer->name(),
                                " did nothing. time = ", duration_ms, "ms.");
      // Swallow the non-critical error.
      status = absl::OkStatus();
    } else if (absl::IsDeadlineExceeded(status)) {
      message =
          strings::StrCat(status.ToString(), ", time = ", duration_ms, "ms.");
      LOG_EVERY_N_SEC(WARNING, 60)
          << optimizer->name() << " failed: " << message;
    } else {
      message = status.ToString();
      LOG_EVERY_N_SEC(ERROR, 60) << optimizer->name() << " failed: " << message;
    }
  } else {
    message = strings::StrCat(
        PrintSizesBeforeAfter(optimized_item->graph, *optimized_graph),
        ", time = ", duration_ms, "ms.");
    VLOG(1) << optimizer->name() << ": " << message;
  }

  // Swap function library back into the main graph.
  if (!is_function_library_aware) {
    optimized_graph->set_allocated_library(
        optimized_graph_function_library.release());
  }

  OptimizerResult optimizer_result{optimizer->name(), message, status};
  optimization_result->results.push_back(optimizer_result);

  if (!status.ok()) {
    if (cfg_.fail_on_optimizer_errors()) return status;

    // Non-aborted failures in the TFG optimizer are always fatal.
    if (absl::StartsWith(optimizer->name(), "tfg_optimizer")) return status;
  }

  return absl::OkStatus();
}

// Propagates `_tf_data_function` attributes from functions to their callees.
void PropagateTFDataAttrs(const FunctionLibraryDefinition& flib,
                          FunctionDefLibrary& fdef_lib) {
  // Collect functions that need the attribute in this set.
  absl::flat_hash_set<std::string> tf_data_functions;
  std::function<void(const std::string&)> collect_tf_data_functions_dfs =
      [&](const std::string& func_name) -> void {
    const FunctionDef* func_def = flib.Find(func_name);
    // Skip functions that are not reachable from the optimized graph.
    if (func_def == nullptr) return;

    // Return if we already found and added this function.
    if (tf_data_functions.contains(func_name)) return;

    // We only get here if the function is (directly or indirectly) called from
    // a tf.data function, so add it to the set.
    tf_data_functions.insert(func_name);

    // Proceed with DFS for functions called from current function.
    for (const NodeDef& node : func_def->node_def()) {
      if (flib.Contains(node.op())) {
        // This is a function call node.
        collect_tf_data_functions_dfs(node.op());
      }
      // Check if there are functions in attributes.
      for (const auto& attr : node.attr()) {
        const AttrValue& attr_value = attr.second;
        if (attr_value.has_func()) {
          collect_tf_data_functions_dfs(attr_value.func().name());
        }
        if (attr_value.has_list()) {
          for (const auto& func : attr_value.list().func()) {
            collect_tf_data_functions_dfs(func.name());
          }
        }
      }
    }
  };
  // Perform DFS for all tf.data functions in `fdef_lib`.
  for (const auto& func_def : fdef_lib.function()) {
    const std::string& func_name = func_def.signature().name();
    if (data::IsTFDataFunction(func_def))
      collect_tf_data_functions_dfs(func_name);
  }
  // Set attribute for tf.data functions. We cannot do this in the DFS directly
  // because `FunctionLibraryDefinition` does not seem to provide mutable access
  // to a `FunctionDef`.
  for (FunctionDef& func_def : *fdef_lib.mutable_function()) {
    const std::string& func_name = func_def.signature().name();
    if (tf_data_functions.contains(func_name) &&
        !data::IsTFDataFunction(func_def)) {
      VLOG(2) << "Marking " << func_name << " as tf.data function";
      (*func_def.mutable_attr())[data::kTFDataFunction].set_b(true);
    }
  }
}

Status MetaOptimizer::OptimizeConsumeItem(Cluster* cluster, GrapplerItem&& item,
                                          GraphDef* optimized_graph) {
  tensorflow::metrics::ScopedCounter<2> timings(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {kGrapplerCategory, "*"});

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
  int old_library_size = item.graph.library().function_size();
  *item.graph.mutable_library() = minimized_flib(item.graph).ToProto();
  int new_library_size = item.graph.library().function_size();

  VLOG(1) << absl::Substitute(
      "Deleted $0 unreachable functions from the graph (library size = $1)",
      old_library_size - new_library_size, new_library_size);

  // Save a few small fields from item before we move it.
  bool optimize_function_library =
      item.optimization_options().optimize_function_library;
  const auto producer = item.graph.versions().producer();

  // 1. Optimize main graph
  TF_RETURN_IF_ERROR(
      OptimizeGraph(cluster, GrapplerItem(item), optimized_graph));
  VLOG(1) << "Optimized main graph.";
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  // 2. Optimize functions reachable from the optimized graph.
  FunctionLibraryDefinition flib = minimized_flib(*optimized_graph);
  using NodeDefs = protobuf::RepeatedPtrField<NodeDef>;

  // Find functions for which we might need to compute a gradient at runtime.
  absl::flat_hash_set<string> differentiable_functions;

  const auto find_differentiable_functions =
      [&](const NodeDefs& nodes) -> void {
    for (const NodeDef& node : nodes) {
      if (IsSymbolicGradient(node)) {
        const auto* f_attr = gtl::FindOrNull(node.attr(), "f");
        if (f_attr) differentiable_functions.insert(f_attr->func().name());
      }
    }
  };

  // SymbolicGradient nodes inside the main graph.
  find_differentiable_functions(optimized_graph->node());
  // SymbolicGradient nodes inside the function library.
  for (const FunctionDef& function : optimized_graph->library().function()) {
    find_differentiable_functions(function.node_def());
  }

  // Find functions that will be compiled by XLA later
  // We do it by looking for XlaLaunch ops that call functions,
  // then depth first search down those functions to find transitive functions.
  // Grappler rewrites can potentially add nodes that are
  // not supported by XLA, so we choose to skip such functions when we optimize
  // the function library.
  absl::flat_hash_set<string> xla_compiled_functions;
  std::function<void(const string&)> find_all_functions;
  find_all_functions = [&](const string& func) -> void {
    // Ignore call cycles in the graph
    if (xla_compiled_functions.contains(func)) return;
    // Find func in the flib
    const FunctionDef* func_def = flib.Find(func);
    CHECK(func_def) << "not found: " << func;
    // Mark function to be ignored by grappler
    xla_compiled_functions.insert(func);
    // Depth first search through the func for transitively called funcs
    for (const NodeDef& node : func_def->node_def()) {
      for (const auto& attr : node.attr()) {
        const AttrValue& attr_value = attr.second;
        if (attr_value.has_func()) {
          find_all_functions(attr_value.func().name());
        }
      }
    }
  };

  auto find_xla_compiled_functions = [&](const NodeDefs& nodes) -> void {
    NameAttrList function;
    for (const NodeDef& node : nodes) {
      // Look only for XlaLaunch nodes that call a function
      if (!IsXlaLaunch(node)) continue;
      if (!GetNodeAttr(node, "function", &function).ok()) continue;
      // Find all transitively called functions
      find_all_functions(function.name());
    }
  };

  // XlaLaunch ops inside the main graph ...
  find_xla_compiled_functions(optimized_graph->node());
  // ... and inside the function library.
  for (const FunctionDef& function : optimized_graph->library().function()) {
    find_xla_compiled_functions(function.node_def());
  }
  // Propagate `_tf_data_function` attributes from functions to their callees.
  PropagateTFDataAttrs(flib, *optimized_graph->mutable_library());

  // True if this is a TPU graph using the old bridge.
  bool is_tpu_graph = IsLegacyTPUBridgeGraphDef(*optimized_graph);

  // Optimize each function only once.
  absl::flat_hash_set<string> optimized_funcs;
  while (optimize_function_library) {
    optimize_function_library = false;

    int function_idx = 0;
    for (const FunctionDef& func : optimized_graph->library().function()) {
      GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

      const string& func_name = func.signature().name();

      // Skip functions that are not reachable from the optimized graph.
      if (!flib.Contains(func_name)) continue;
      // Skip already optimized functions.
      if (optimized_funcs.contains(func_name)) continue;
      // Skip functions that will be compiled by XLA.
      if (xla_compiled_functions.contains(func_name)) continue;

      // Skip parametrized functions (function type or body is defined only at
      // function call time by caller node attributes).
      // They should be specialized to their instantiation type parameters by
      // the function optimizer, before we can optimize function body.
      if (IsParametrized(func)) continue;

      // Skip tf.data functions as they are optimized by tf.data meta optimizer
      // and in function instantiation.
      if (data::IsTFDataFunction(func)) continue;

      VLOG(3) << "Optimize function: function=" << func_name << " ["
              << function_idx++ << " of "
              << optimized_graph->library().function_size() << "]";

      // Function optimization might specialize nested function calls, so we
      // have to reset the flag and do at least one more pass over the library.
      optimize_function_library = true;
      optimized_funcs.insert(func_name);

      // Make a GrapplerItem from a FunctionDef.
      GrapplerFunctionItem func_item;
      TF_RETURN_IF_ERROR(
          MakeGrapplerFunctionItem(func, flib, producer, &func_item));

      // If we need to compute the gradient of optimized function at runtime, we
      // can't perform non-differentiable rewrites.
      func_item.optimization_options().allow_non_differentiable_rewrites =
          !differentiable_functions.contains(func_name);

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

      // Optimize function body graph.
      GraphDef optimized_func_graph;
      if (is_tpu_graph) {
        // Skip optimizing functions if this is a TPU graph. Currently, Grappler
        // passes do not handle TPU functions correctly in a variety of ways
        // (Note that due to the pre-placement TPU graph rewriting passes, the
        // TPU-related ops are encapsulated away into functions). For example,
        // TPU graphs contain TPUReplicateMetadata node that carries relevant
        // TPU metadata and Grappler passes could prune that away. Grappler
        // passes could also cause issues around shape inference. Since the
        // desired and existing behavior is to not optimize TPU functions with
        // Grappler, this check preserves that. The only exception is
        // implementation selector what is required to swap in some TPU specific
        // lowering code and is verified the work correctly on TPUs.
        ImplementationSelector implementation_selector;

        // Implementation selector needs to have access to valid function
        // signature and attributes, and it doesn't need actual function body.
        std::unique_ptr<FunctionDefLibrary> func_item_function_library(
            func_item.graph.release_library());
        *func_item.graph.mutable_library() =
            GetFunctionDefLibraryStub(*func_item_function_library);

        TF_RETURN_IF_ERROR(implementation_selector.Optimize(
            cluster, func_item, &optimized_func_graph));
      } else {
        GrapplerFunctionItem func_item_copy = func_item;
        TF_RETURN_IF_ERROR(OptimizeGraph(cluster, std::move(func_item_copy),
                                         &optimized_func_graph));
      }

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

  // Run module-level TFG optimizations at the end of the meta-optimizer.
  // TODO(jeffniu): None of the TFG optimizations are meant to create new
  // opportunities for other optimizers; they could, but it's unclear whether
  // re-running all the other optimizers is worthwhile.
#if !NO_LLVM_SUPPORT
  if (!cfg_.disable_tfg_optimizer()) {
    // Create a Grappler optimization pipeline with only the TFG optimizer.
    std::vector<std::unique_ptr<GraphOptimizer>> optimizers;
    optimizers.push_back(std::make_unique<mlir::tfg::TFGGrapplerOptimizer>(
        // For module-level optimizations, use multithreading to process
        // functions in parallel.
        [&](mlir::PassManager& manager) {
          mlir::tfg::DefaultModuleGrapplerPipeline(manager, cfg_);
        },
        /*num_tfg_threads=*/4));
    // Wrap the optimized GraphDef in a new GrapplerItem with copied
    // configuration options from the provided item.
    GrapplerItem tfg_item = item.WithGraph(std::move(*optimized_graph));
    // Invoke the optimizers.
    *optimized_graph = GraphDef();
    TF_RETURN_IF_ERROR(OptimizeGraph(optimizers, cluster, std::move(tfg_item),
                                     optimized_graph));
  }
#endif

  VLOG(1) << "Optimized " << optimized_funcs.size()
          << " functions: " << absl::StrJoin(optimized_funcs, ", ");
  VLOG(3) << "Optimized graph =\n" << optimized_graph->DebugString();
  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile(
        strings::StrCat("after_MetaOptimizer_",
                        reinterpret_cast<uintptr_t>(optimized_graph)),
        *optimized_graph);
  }

  return absl::OkStatus();
}

string MetaOptimizer::GetResultString() const {
  std::string result_string;
  for (const GraphOptimizationResult& graph_result : optimization_results_) {
    absl::StrAppend(&result_string,
                    "Optimization results for grappler item: ", graph_result.id,
                    "\n");
    for (const OptimizerResult& result : graph_result.results) {
      absl::StrAppend(&result_string, "  ", result.optimizer_name, ": ",
                      result.message, "\n");
    }
  }
  return result_string;
}

void MetaOptimizer::PrintResult() { VLOG(1) << GetResultString(); }

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
         rewrite_cfg.common_subgraph_elimination() != RewriterConfig::OFF ||
         rewrite_cfg.arithmetic_optimization() != RewriterConfig::OFF ||
         rewrite_cfg.loop_optimization() != RewriterConfig::OFF ||
         rewrite_cfg.dependency_optimization() != RewriterConfig::OFF ||
         rewrite_cfg.auto_parallel().enable() ||
         rewrite_cfg.memory_optimization() != RewriterConfig::NO_MEM_OPT ||
         rewrite_cfg.debug_stripper() == RewriterConfig::ON ||
#ifndef ENABLE_MKL
         rewrite_cfg.scoped_allocator_optimization() == RewriterConfig::ON ||
#endif
         rewrite_cfg.pin_to_host_optimization() == RewriterConfig::ON ||
         AutoMixedPrecisionEnabled(rewrite_cfg.auto_mixed_precision()) ||
         AutoMixedPrecisionEnabled(
             rewrite_cfg.auto_mixed_precision_onednn_bfloat16()) ||
         AutoMixedPrecisionEnabled(rewrite_cfg.auto_mixed_precision_mkl()) ||
         AutoMixedPrecisionEnabled(rewrite_cfg.auto_mixed_precision_cpu()) ||
         !rewrite_cfg.optimizers().empty() ||
         !rewrite_cfg.custom_optimizers().empty();
}

Status RunMetaOptimizer(GrapplerItem&& item, const ConfigProto& cfg,
                        DeviceBase* cpu_device, Cluster* cluster,
                        GraphDef* optimized_graph) {
  MetaOptimizer optimizer(cpu_device, cfg);
  optimizer.set_deadline_usec(
      DeadlineMicroSeconds(cfg.graph_options().rewrite_options()));
  return optimizer.OptimizeConsumeItem(cluster, std::move(item),
                                       optimized_graph);
}

Status OptimizeGraph(
    std::vector<string> ret_node_names, std::vector<string> keep_node_names,
    FunctionLibraryDefinition* flib, const DeviceSet& device_set,
    Device* cpu_device, const ConfigProto& config_proto,
    const string& grappler_item_id,
    const GrapplerItem::OptimizationOptions& optimization_options,
    std::unique_ptr<tensorflow::Graph>* g) {
  if (!tensorflow::grappler::MetaOptimizerEnabled(config_proto)) {
    return absl::OkStatus();
  }

  tensorflow::grappler::GrapplerItem item;
  item.id = grappler_item_id;
  item.optimization_options() = optimization_options;
  if (cpu_device && std::is_same<decltype(cpu_device), LocalDevice>::value &&
      cpu_device->tensorflow_cpu_worker_threads() != nullptr) {
    // Forward to the optimisation pass number of intra threads that are used to
    // parallelise operations.
    item.optimization_options().intra_op_parallelism_threads =
        cpu_device->tensorflow_cpu_worker_threads()->num_threads;
  }

  // Add all available devices so that inlined function can be placed.
  for (const Device* d : device_set.devices()) {
    Status added_device = item.AddDevice(d->name());
    if (!added_device.ok()) VLOG(3) << added_device.message();
  }
  VLOG(3) << "Grappler available devices: "
          << absl::StrJoin(item.devices(), ", ");

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
      std::move(item), config_proto, cpu_device, &cluster, &out_graph));

  std::unique_ptr<tensorflow::Graph> optimized_graph(
      new tensorflow::Graph(OpRegistry::Global()));

  // Copy optimized functions back to the overlay lib.
  if (flib) {
    for (const FunctionDef& fdef : out_graph.library().function()) {
      const string& func_name = fdef.signature().name();
      if (flib->Contains(func_name)) {
        StackTracesMap stack_traces = *flib->GetStackTraces(func_name);
        TF_RETURN_IF_ERROR(
            flib->ReplaceFunction(func_name, fdef, stack_traces));
      } else {
        TF_RETURN_IF_ERROR(flib->AddFunctionDef(fdef));
      }
    }
  }

  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      GraphConstructorOptions(), std::move(out_graph), optimized_graph.get()));

  // The graph conversion sets the requested device names but not the
  // assigned device names. However, since at this point the graph is
  // placed TF expects an assigned device name for every node. Therefore
  // we copy the requested device into the assigned device field.
  for (Node* node : optimized_graph->nodes()) {
    if (node->IsOp() && node->assigned_device_name().empty()) {
      if (node->requested_device().empty()) {
        return errors::Internal(
            "Either placer did not place the node or Grappler did not "
            "copy the assigned device. Contact Grappler team since latter "
            "is more likely. Node=",
            node->name(),
            " Graph: ", optimized_graph->ToGraphDefDebug().DebugString());
      }
      node->set_assigned_device_name(node->requested_device());
    }
  }

  *g = std::move(optimized_graph);
  return absl::OkStatus();
}

}  // namespace grappler
}  // namespace tensorflow
