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
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_GRAPH_OPTIMIZER_REGISTRY_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_GRAPH_OPTIMIZER_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

namespace tensorflow {
namespace grappler {

struct ConfigsList {
  bool operator==(const ConfigsList& other) const {
    return (disable_model_pruning == other.disable_model_pruning) &&
           (implementation_selector == other.implementation_selector) &&
           (function_optimization == other.function_optimization) &&
           (common_subgraph_elimination == other.common_subgraph_elimination) &&
           (arithmetic_optimization == other.arithmetic_optimization) &&
           (debug_stripper == other.debug_stripper) &&
           (constant_folding == other.constant_folding) &&
           (shape_optimization == other.shape_optimization) &&
           (auto_mixed_precision == other.auto_mixed_precision) &&
           (auto_mixed_precision_mkl == other.auto_mixed_precision_mkl) &&
           (pin_to_host_optimization == other.pin_to_host_optimization) &&
           (layout_optimizer == other.layout_optimizer) &&
           (remapping == other.remapping) &&
           (loop_optimization == other.loop_optimization) &&
           (dependency_optimization == other.dependency_optimization) &&
           (auto_parallel == other.auto_parallel) &&
           (memory_optimization == other.memory_optimization) &&
           (scoped_allocator_optimization ==
            other.scoped_allocator_optimization);
  }
  bool disable_model_pruning;
  RewriterConfig_Toggle implementation_selector;
  RewriterConfig_Toggle function_optimization;
  RewriterConfig_Toggle common_subgraph_elimination;
  RewriterConfig_Toggle arithmetic_optimization;
  RewriterConfig_Toggle debug_stripper;
  RewriterConfig_Toggle constant_folding;
  RewriterConfig_Toggle shape_optimization;
  RewriterConfig_Toggle auto_mixed_precision;
  RewriterConfig_Toggle auto_mixed_precision_mkl;
  RewriterConfig_Toggle pin_to_host_optimization;
  RewriterConfig_Toggle layout_optimizer;
  RewriterConfig_Toggle remapping;
  RewriterConfig_Toggle loop_optimization;
  RewriterConfig_Toggle dependency_optimization;
  RewriterConfig_Toggle auto_parallel;
  RewriterConfig_Toggle memory_optimization;
  RewriterConfig_Toggle scoped_allocator_optimization;
};

string PrintConfigs(ConfigsList& configs);

class CustomGraphOptimizerRegistry {
 public:
  static std::unique_ptr<CustomGraphOptimizer> CreateByNameOrNull(
      const string& name);

  static std::vector<string> GetRegisteredOptimizers();

  typedef std::function<CustomGraphOptimizer*()> Creator;
  // Register graph optimizer which can be called during program initialization.
  // This class is not thread-safe.
  static void RegisterOptimizerOrDie(const Creator& optimizer_creator,
                                     const string& name);
};

class CustomGraphOptimizerRegistrar {
 public:
  explicit CustomGraphOptimizerRegistrar(
      const CustomGraphOptimizerRegistry::Creator& creator,
      const string& name) {
    CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(creator, name);
  }
};

#define REGISTER_GRAPH_OPTIMIZER_AS(MyCustomGraphOptimizerClass, name) \
  namespace {                                                          \
  static ::tensorflow::grappler::CustomGraphOptimizerRegistrar         \
      MyCustomGraphOptimizerClass##_registrar(                         \
          []() { return new MyCustomGraphOptimizerClass; }, (name));   \
  }  // namespace

#define REGISTER_GRAPH_OPTIMIZER(MyCustomGraphOptimizerClass) \
  REGISTER_GRAPH_OPTIMIZER_AS(MyCustomGraphOptimizerClass,    \
                              #MyCustomGraphOptimizerClass)

class PluginGraphOptimizerRegistry {
 public:
  static std::vector<std::unique_ptr<CustomGraphOptimizer>> CreateOptimizer(
      const std::set<string>& device_types);

  typedef std::function<CustomGraphOptimizer*()> Creator;

  // Return plugin's config. If any of the config is turned off, the returned
  // config will be turned off.
  static ConfigsList GetPluginConfigs(bool use_plugin_optimizers,
                                      const std::set<string>& device_types);

  // Register plugin graph optimizer which can be called during program
  // initialization. This class is not thread-safe.
  static void RegisterPluginOptimizerOrDie(const Creator& optimizer_creator,
                                           const std::string& device_type,
                                           ConfigsList& configs);

  // Print plugin's configs if there are some conflicts.
  static void PrintPluginConfigsIfConflict(
      const std::set<string>& device_types);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_GRAPH_OPTIMIZER_REGISTRY_H_
