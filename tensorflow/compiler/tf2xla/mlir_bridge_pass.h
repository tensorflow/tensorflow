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

#ifndef TENSORFLOW_COMPILER_TF2XLA_MLIR_BRIDGE_PASS_H_
#define TENSORFLOW_COMPILER_TF2XLA_MLIR_BRIDGE_PASS_H_

#include "tensorflow/compiler/mlir/tf2xla/mlir_bridge_rollout_policy.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"

namespace tensorflow {

// This pass uses MLIR to implement all the conversion steps to target XLA from
// a TensorFlow Function Graph. It is meant to expose a very limited set of
// functionalities during the bring-up of MLIR-based bridge.
class MlirBridgePass : public MlirOptimizationPass {
 public:
  llvm::StringRef name() const override { return "bridge"; }

  MlirOptimizationPassState GetPassState(
      const DeviceSet* device_set, const ConfigProto& config_proto,
      const Graph& graph,
      const FunctionLibraryDefinition& function_library) const override;

  // This should be used as a thin mapper around mlir::ModulePass::runOnModule
  // API integrated with the Tensorflow runtime.
  Status Run(const ConfigProto& config_proto, mlir::ModuleOp module,
             const Graph& graph,
             const FunctionLibraryDefinition& function_library) override;
};

// This pass uses MLIR to implement all the conversion steps to target XLA from
// a TensorFlow V1 Graph. It is meant to expose a very limited set of
// functionalities during the bring-up of MLIR-based bridge.
class MlirBridgeV1CompatPass : public MlirV1CompatOptimizationPass {
 public:
  llvm::StringRef name() const override { return "bridge"; }

  MlirOptimizationPassState GetPassState(
      const DeviceSet* device_set, const ConfigProto& config_proto,
      const Graph& graph,
      const FunctionLibraryDefinition& function_library) const override;

  // This should be used as a thin mapper around mlir::ModulePass::runOnModule
  // API integrated with the Tensorflow runtime.
  Status Run(const GraphOptimizationPassOptions& options,
             mlir::ModuleOp module) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_MLIR_BRIDGE_PASS_H_
