/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_TF_XLA_CALL_MODULE_TO_STABLEHLO_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_TF_XLA_CALL_MODULE_TO_STABLEHLO_PASS_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace odml {

// Adds passes which transform TF_XlaCallModule Op to StableHLO Ops.
// Note that this pass only supports static shape tensors for now.
std::unique_ptr<mlir::OperationPass<ModuleOp>>
CreateLegalizeTFXlaCallModuleToStablehloPass();

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_TF_XLA_CALL_MODULE_TO_STABLEHLO_PASS_H_
