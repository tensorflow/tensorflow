/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_UTILITIES_ELEMENTS_ATTR_ROUNDTRIP_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_UTILITIES_ELEMENTS_ATTR_ROUNDTRIP_PASS_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {

// Pass to convert DenseElementsAttr to DenseResourceElementsAttr.
class DenseToDenseResourceElementsPass
    : public PassWrapper<DenseToDenseResourceElementsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DenseToDenseResourceElementsPass)
  StringRef getArgument() const final {
    return "tfl-dense-to-dense-resource-elements";
  }
  StringRef getDescription() const final {
    return "Converts DenseElementsAttr to DenseResourceElementsAttr.";
  }
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect,
                    tf_saved_model::TensorFlowSavedModelDialect>();
  }
};

// Pass to convert DenseResourceElementsAttr to DenseElementsAttr.
class DenseResourceToDenseElementsPass
    : public PassWrapper<DenseResourceToDenseElementsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DenseResourceToDenseElementsPass)
  StringRef getArgument() const final {
    return "tfl-dense-resource-to-dense-elements";
  }
  StringRef getDescription() const final {
    return "Converts DenseResourceElementsAttr to DenseElementsAttr.";
  }
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect,
                    tf_saved_model::TensorFlowSavedModelDialect>();
  }
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_UTILITIES_ELEMENTS_ATTR_ROUNDTRIP_PASS_H_
