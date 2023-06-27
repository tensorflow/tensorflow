/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_OPTIMIZATION_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_OPTIMIZATION_REGISTRY_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"

// Classes to maintain a static registry of Graph based passes to be applied to
// a function graph.

namespace tensorflow {

// A pass to be registered with the FunctionOptimizationPassRegistry. This pass
// takes in a DeviceSet (available devices for executing the Graph), ConfigProto
// (session configuration parameters), an optional target device for XLA
// compilation, Graph (computation),
// FunctionLibraryDefinition (mapping between function names and function
// definitions of the Graph), control ret/target node names (names of nodes that
// must execute but their data outputs, if they have any, are irrelevant), and
// whether control ret nodes (via thier name) were updated. Mutations to the
// Graph and other associated arguments are performed inplace by the pass.
class FunctionOptimizationPass {
 public:
  // Grouped Options for the optimized function.
  struct FunctionOptions {
    // Specifies the compilation device type(CPU, GPU, etc)
    // that should be used for entire function.
    std::string xla_compile_device_type = "";
    // Whether soft placement and outside compilation
    // are enabled for the function.
    bool allow_soft_placement = false;
  };

  virtual ~FunctionOptimizationPass() {}
  virtual Status Run(const std::string& function_name,
                     const DeviceSet& device_set,
                     const ConfigProto& config_proto,
                     const FunctionOptions& function_options,
                     std::unique_ptr<Graph>* graph,
                     FunctionLibraryDefinition* flib_def,
                     std::vector<std::string>* control_ret_node_names,
                     bool* control_rets_updated) = 0;
};

// A global function optimization pass registry that is used to hold one
// FunctionOptimizationPass. Passes registered to this registry will run before
// passes registered in OptimizationPassRegistry.
class FunctionOptimizationPassRegistry {
 public:
  // Initializes registry with a pass. Only one pass should be set. An assertion
  // will be triggered if the registry already has a pass set and is being
  // initialized with another pass.
  void Init(std::unique_ptr<FunctionOptimizationPass> pass);

  // Runs a pass if the registry contains one.
  Status Run(const std::string& function_name, const DeviceSet& device_set,
             const ConfigProto& config_proto,
             const FunctionOptimizationPass::FunctionOptions& function_options,
             std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
             std::vector<std::string>* control_ret_node_names,
             bool* control_rets_updated);

  // Returns the global registry of function graph passes.
  static FunctionOptimizationPassRegistry& Global();

 private:
  std::unique_ptr<FunctionOptimizationPass> pass_;
};

namespace function_optimization_registration {

class FunctionOptimizationPassRegistration {
 public:
  explicit FunctionOptimizationPassRegistration(
      std::unique_ptr<FunctionOptimizationPass> pass) {
    FunctionOptimizationPassRegistry::Global().Init(std::move(pass));
  }
};

}  // namespace function_optimization_registration

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_OPTIMIZATION_REGISTRY_H_
