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

#ifndef TENSORFLOW_COMPILER_MLIR_MLIR_GRAPH_OPTIMIZATION_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_MLIR_GRAPH_OPTIMIZATION_PASS_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/mlir/tf2xla/mlir_bridge_rollout_policy.h"
#include "absl/log/check.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// -------------------------------------------------------------------------- //
// MLIR passes running on Tensorflow function graphs (Tensorflow V2).
// -------------------------------------------------------------------------- //

// Disabled - skip execution of the pass.
// Enabled - execute the pass, propagate errors to the caller if any.
// FallbackEnabled - execute the pass and commit all the changes to the MLIR
//   module in case of success. Do not commit any changes in case of failures,
//   let the rest of the pipeline run.
enum class MlirOptimizationPassState { Disabled, Enabled, FallbackEnabled };

// An API for registering MLIR ModulePass with the Tensorflow runtime. These
// passes are running only for function graphs built by Tensorflow V2 and
// instantiated by the process_function_library_runtime (see
// FunctionOptimizationPass for details).
class MlirOptimizationPass {
 public:
  virtual ~MlirOptimizationPass() = default;
  virtual llvm::StringRef name() const = 0;

  // Returns an enum value:
  //   Enabled if the pass is enabled for the given graph with specified config.
  //   Disabled if the pass is disabled.
  //   FallbackEnabled if the pass needs to be executed in fallback mode.
  //
  // When the pass is FallbackEnabled, the pass is executed and the changes it
  // makes to the MLIR module will be committed only if the pass was successful,
  // otherwise no changes are committed and the rest of the pipeline is run.
  //
  // `device_set` can be nullptr if the devices information is not
  // available or no device specific filtering is required.
  // `function_library` contains function definitions for function calls in
  // `graph` not included in the `graph` FunctionLibraryDefinition.
  virtual MlirOptimizationPassState GetPassState(
      const DeviceSet* device_set, const ConfigProto& config_proto,
      const Graph& graph,
      const FunctionLibraryDefinition& function_library) const = 0;

  virtual Status Run(const std::string& function_name,
                     const ConfigProto& config_proto, mlir::ModuleOp module,
                     const Graph& graph,
                     const FunctionLibraryDefinition& function_library) = 0;
};

class MlirOptimizationPassRegistry {
 public:
  struct PassRegistration {
    int priority;
    std::unique_ptr<MlirOptimizationPass> pass;
  };

  struct PriorityComparator {
    bool operator()(const PassRegistration& x,
                    const PassRegistration& y) const {
      return x.priority < y.priority;
    }
  };

  using Passes = std::set<PassRegistration, PriorityComparator>;

  // Returns the global registry of MLIR optimization passes.
  static MlirOptimizationPassRegistry& Global();

  // Register optimization `pass` with the given `priority`.
  void Add(int priority, std::unique_ptr<MlirOptimizationPass> pass) {
    auto inserted = passes_.insert({priority, std::move(pass)});
    CHECK(inserted.second)
        << "Pass priority must be unique. "
        << "Previously registered pass with the same priority: "
        << inserted.first->pass->name().str();
  }

  // Free the memory allocated for all passes.
  void ClearPasses() { passes_.clear(); }

  const Passes& passes() const { return passes_; }

 private:
  Passes passes_;
};

// Function optimization pass that runs all MLIR passes registered in
// MlirOptimizationPassRegistry.
class MlirFunctionOptimizationPass : public FunctionOptimizationPass {
 public:
  explicit MlirFunctionOptimizationPass(
      const MlirOptimizationPassRegistry* registry =
          &MlirOptimizationPassRegistry::Global())
      : registry_(registry) {}

  // Executes all of the underlying registered MlirOptimizationPasses.
  Status Run(const std::string& function_name, const DeviceSet& device_set,
             const ConfigProto& config_proto,
             const FunctionOptimizationPass::FunctionOptions& function_options,
             std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
             std::vector<std::string>* control_ret_node_names,
             bool* control_rets_updated) override;

 private:
  const MlirOptimizationPassRegistry* registry_;
};

// -------------------------------------------------------------------------- //
// MLIR passes running on Tensorflow V1 graphs.
// -------------------------------------------------------------------------- //

// An API for registering MLIR ModulePass with the Tensorflow runtime. These
// passes are running only for V1 graphs (legacy graphs) executed via Session
// runtime. Graph importer updates legacy graph behavior to V2 constructs (e.g.
// it raises control flow from Switch/Merge nodes to functional control flow
// with If/While operations).
class MlirV1CompatOptimizationPass {
 public:
  virtual ~MlirV1CompatOptimizationPass() = default;
  virtual llvm::StringRef name() const = 0;

  // Returns a MlirOptimizationPassState based on the given graph and
  // config. See comments on `MlirOptimizationPassState` enum for more info
  // on exact values.
  virtual MlirOptimizationPassState GetPassState(
      const DeviceSet* device_set, const ConfigProto& config_proto,
      const Graph& graph,
      const FunctionLibraryDefinition& function_library) const = 0;

  virtual Status Run(const GraphOptimizationPassOptions& options,
                     mlir::ModuleOp module) = 0;
};

class MlirV1CompatOptimizationPassRegistry {
 public:
  // Returns the global registry of MLIR optimization passes.
  static MlirV1CompatOptimizationPassRegistry& Global();

  void Add(std::unique_ptr<MlirV1CompatOptimizationPass> pass) {
    CHECK(pass_ == nullptr) << "Only a single pass can be registered";
    pass_ = std::move(pass);
  }

  MlirV1CompatOptimizationPass* pass() const {
    return pass_ ? pass_.get() : nullptr;
  }

  // Free the memory allocated for the single pass.
  // This method is used for testing mostly.
  void ClearPass() { pass_.reset(); }

 private:
  std::unique_ptr<MlirV1CompatOptimizationPass> pass_{};
};

class MlirV1CompatGraphOptimizationPass : public GraphOptimizationPass {
 public:
  explicit MlirV1CompatGraphOptimizationPass(
      const MlirV1CompatOptimizationPassRegistry* registry =
          &MlirV1CompatOptimizationPassRegistry::Global())
      : registry_(registry) {}

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  const MlirV1CompatOptimizationPassRegistry* registry_;
};

// -------------------------------------------------------------------------- //
// Helper classes for static registration of MLIR (V1 Compat) passes in the
// corresponding registry.
// -------------------------------------------------------------------------- //

namespace mlir_pass_registration {

class MlirOptimizationPassRegistration {
 public:
  explicit MlirOptimizationPassRegistration(
      int priority, std::unique_ptr<MlirOptimizationPass> pass) {
    MlirOptimizationPassRegistry::Global().Add(priority, std::move(pass));
  }
};

class MlirV1CompatOptimizationPassRegistration {
 public:
  explicit MlirV1CompatOptimizationPassRegistration(
      std::unique_ptr<MlirV1CompatOptimizationPass> pass) {
    MlirV1CompatOptimizationPassRegistry::Global().Add(std::move(pass));
  }
};

}  // namespace mlir_pass_registration

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_MLIR_GRAPH_OPTIMIZATION_PASS_H_
