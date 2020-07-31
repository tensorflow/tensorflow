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

#include "mlir/IR/Module.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// -------------------------------------------------------------------------- //
// MLIR passes running on Tensorflow function graphs (Tensorflow V2).
// -------------------------------------------------------------------------- //

// An API for registering MLIR ModulePass with the Tensorflow runtime. These
// passes are running only for function graphs built by Tensorflow V2 and
// instantiated by the process_function_library_runtime (see
// FunctionOptimizationPass for details).
class MlirOptimizationPass {
 public:
  virtual ~MlirOptimizationPass() = default;
  virtual llvm::StringRef name() const = 0;
  virtual bool IsEnabled(const ConfigProto& config_proto) const = 0;

  virtual Status Run(const ConfigProto& config_proto,
                     mlir::ModuleOp module) = 0;
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

  void Add(int priority, std::unique_ptr<MlirOptimizationPass> pass) {
    passes_.insert({priority, std::move(pass)});
  }

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

  Status Run(const DeviceSet& device_set, const ConfigProto& config_proto,
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
  virtual bool IsEnabled(const ConfigProto& config_proto) const = 0;

  virtual Status Run(const GraphOptimizationPassOptions& options,
                     mlir::ModuleOp module) = 0;
};

class MlirV1CompatOptimizationPassRegistry {
 public:
  struct PassRegistration {
    int priority;
    std::unique_ptr<MlirV1CompatOptimizationPass> pass;
  };

  struct PriorityComparator {
    bool operator()(const PassRegistration& x,
                    const PassRegistration& y) const {
      return x.priority < y.priority;
    }
  };

  using Passes = std::set<PassRegistration, PriorityComparator>;

  // Returns the global registry of MLIR optimization passes.
  static MlirV1CompatOptimizationPassRegistry& Global();

  void Add(int priority, std::unique_ptr<MlirV1CompatOptimizationPass> pass) {
    passes_.insert({priority, std::move(pass)});
  }

  const Passes& passes() const { return passes_; }

 private:
  Passes passes_;
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
      int priority, std::unique_ptr<MlirV1CompatOptimizationPass> pass) {
    MlirV1CompatOptimizationPassRegistry::Global().Add(priority,
                                                       std::move(pass));
  }
};

}  // namespace mlir_pass_registration

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_MLIR_GRAPH_OPTIMIZATION_PASS_H_
