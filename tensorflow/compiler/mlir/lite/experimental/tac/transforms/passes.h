/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_PASSES_H_

#include <functional>
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_filter.pb.h"

namespace mlir {
namespace TFL {
namespace tac {
class TacModule;

// Create an instance of the TargetAnnotationPass.
// TODO(b/177376459): Remove in favor of the one below.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTargetAnnotationPass(
    llvm::ArrayRef<std::string> device_specs);

// Create and instance of TargetAnnotationPass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTargetAnnotationPass(
    const TacModule* module);

// Create an instance of the RaiseTargetSubgraphsPass. If `skip_raise_cpu_ops`,
// we skip clustering for CPU ops for better clustering of ops running on other
// ML accelerators. When `ignore_inference_type` is set to true, the inference
// types are set to "NOT_CARE" for better clustering.
std::unique_ptr<OperationPass<ModuleOp>> CreateRaiseTargetSubgraphsPass(
    bool skip_raise_cpu_ops = false, bool ignore_inference_type = false);

// Create an instance of the AlternativeSubgraphPass.
std::unique_ptr<OperationPass<ModuleOp>> CreateAlternativeSubgraphPass(
    llvm::ArrayRef<std::string> device_specs);

// Create an instance of ComputeCostPass.
std::unique_ptr<OperationPass<ModuleOp>> CreateComputeCostPass();

// Create an instance of PickSubgraphsPass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePickSubgraphsPass();

// Create an instance of DeviceTransformGPUPass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDeviceTransformGPUPass();

// Create an instance of GetOpCostPass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateGetOpCostPass();

// Create an instance of FoldConstantsToSubgraphPass.
std::unique_ptr<OperationPass<ModuleOp>> CreateFoldConstantsToSubgraphPass(
    bool fold_all_constants);

// Create an instance of TacFilterPass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTacFilterPass(
    ::third_party::tensorflow::compiler::mlir::lite::experimental::tac::
        TacFilters* tac_filters,
    std::function<void(mlir::Operation* op,
                       const google::protobuf::Any& custom_options)>
        custom_options_callback);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_PASSES_H_
