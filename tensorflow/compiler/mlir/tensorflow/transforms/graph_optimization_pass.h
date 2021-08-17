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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_

#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"

namespace mlir {
namespace TF {

// Bundle generic MLIR graph optimization passes (some derived from TF Grappler
// graph optimizers) into a single MLIR optimization pass.
class MlirGraphOptimizationPass : public ::tensorflow::MlirOptimizationPass {
 public:
  llvm::StringRef name() const override { return "graph_optimization"; }

  ::tensorflow::MlirOptimizationPassState GetPassState(
      const ::tensorflow::DeviceSet* device_set,
      const ::tensorflow::ConfigProto& config_proto,
      const tensorflow::Graph& graph,
      const tensorflow::FunctionLibraryDefinition& function_library)
      const override {
    return config_proto.experimental().enable_mlir_graph_optimization()
               ? tensorflow::MlirOptimizationPassState::Enabled
               : tensorflow::MlirOptimizationPassState::Disabled;
  }

  ::tensorflow::Status Run(
      const ::tensorflow::ConfigProto& config_proto, ModuleOp module,
      const ::tensorflow::Graph& graph,
      const tensorflow::FunctionLibraryDefinition& function_library) override;
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_GRAPH_OPTIMIZATION_PASS_H_
