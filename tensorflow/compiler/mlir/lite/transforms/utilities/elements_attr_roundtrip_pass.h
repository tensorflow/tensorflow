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

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {

struct ElementsAttrRoundtripPassOptions : public mlir::detail::PassOptions {
  mlir::detail::PassOptions::Option<bool> convert_elements_to_dense_resources{
      *this, "convert-elements-to-dense-resources",
      llvm::cl::desc("Convert elements to dense resources"),
      llvm::cl::init(true)};
};

// This pass converts elements to dense resources and vice versa.
class ElementsAttrRoundtripPass
    : public Pass<ElementsAttrRoundtripPass, ElementsAttrRoundtripPassOptions> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElementsAttrRoundtripPass)

  ElementsAttrRoundtripPass() = default;
  ElementsAttrRoundtripPass(const ElementsAttrRoundtripPass&) {};
  explicit ElementsAttrRoundtripPass(const mlir::detail::PassOptions& options)
      : Pass<ElementsAttrRoundtripPass, ElementsAttrRoundtripPassOptions>(
            options) {}

  void runOnOperation() override;
  static llvm::StringRef GetName() { return "ElementsAttrRoundtripPass"; }
  static llvm::StringRef GetArgument() { return "tfl-elements-attr-roundtrip"; }
  static llvm::StringRef GetDescription() {
    return "Pass to convert dense elements to dense resources and vice versa";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect,
                    tf_saved_model::TensorFlowSavedModelDialect>();
  }
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_UTILITIES_ELEMENTS_ATTR_ROUNDTRIP_PASS_H_
