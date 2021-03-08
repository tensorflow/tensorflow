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

// Contains plugin's configurations for each Grappler optimizer (on/off).
// See tensorflow/core/protobuf/rewriter_config.proto for optimizer description.
struct ConfigList {
  ConfigList() {}
  ConfigList(bool disable_model_pruning,
             std::unordered_map<string, RewriterConfig_Toggle> config)
      : disable_model_pruning(disable_model_pruning),
        toggle_config(std::move(config)) {}

  bool operator==(const ConfigList& other) const {
    return (disable_model_pruning == other.disable_model_pruning) &&
           (toggle_config == other.toggle_config);
  }
  bool disable_model_pruning;  // Don't remove unnecessary ops from the graph.
  std::unordered_map<string, RewriterConfig_Toggle> toggle_config;
};

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

// A separate registry to register all plug-in CustomGraphOptimizers.
class PluginGraphOptimizerRegistry {
 public:
  // Constructs a list of plug-in CustomGraphOptimizers from the global map
  // `registered_plugin_optimizers`.
  static std::vector<std::unique_ptr<CustomGraphOptimizer>> CreateOptimizers(
      const std::set<string>& device_types);

  typedef std::function<CustomGraphOptimizer*()> Creator;

  // Returns plugin's config. If any of the config is turned off, the returned
  // config will be turned off.
  static ConfigList GetPluginConfigs(bool use_plugin_optimizers,
                                     const std::set<string>& device_types);

  // Registers plugin graph optimizer which can be called during program
  // initialization. Dies if multiple plugins with the same `device_type` are
  // registered. This class is not thread-safe.
  static void RegisterPluginOptimizerOrDie(const Creator& optimizer_creator,
                                           const std::string& device_type,
                                           ConfigList& configs);

  // Prints plugin's configs if there are some conflicts.
  static void PrintPluginConfigsIfConflict(
      const std::set<string>& device_types);

  // Returns true when `plugin_config` conflicts with `user_config`:
  // - Plugin's `disable_model_pruning` is not equal to `user_config`'s, or
  // - At least one of plugin's `toggle_config`s is on when it is set to off in
  //   `user_config`'s.
  static bool IsConfigsConflict(ConfigList& user_config,
                                ConfigList& plugin_config);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CUSTOM_GRAPH_OPTIMIZER_REGISTRY_H_
