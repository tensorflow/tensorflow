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

#include <memory>
#include <string>

#include "llvm/Support/Casting.h"
#include "mlir-c/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/Support/LogicalResult.h"    // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_dataflow.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_saved_model {

namespace {

#define GEN_PASS_DEF_CONVERTTFSAVEDMODELATTRTOEMITCATTRPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_savedmodel_passes.h.inc"

class ConvertTFSavedModelAttrToEmitCAttrPass
    : public impl::ConvertTFSavedModelAttrToEmitCAttrPassBase<
          ConvertTFSavedModelAttrToEmitCAttrPass> {
  void runOnOperation() final;
};

void ConvertTFSavedModelAttrToEmitCAttrPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(funcOp);

  auto argAttrs = funcOp.getArgAttrs();
  for (int idx = 0; idx < funcOp.getNumArguments(); ++idx) {
    mlir::DictionaryAttr dictAttr = funcOp.getArgAttrDict(idx);
    if (!argAttrs) continue;

    mlir::Attribute namedAttribute =
        dictAttr.getNamed("tf_saved_model.index_path")->getValue();
    auto arrayAttr = cast<mlir::ArrayAttr>(namedAttribute);
    StringAttr hintNameAttr;
    for (const auto attr : arrayAttr) {
      hintNameAttr = cast<mlir::StringAttr>(attr);
      funcOp.setArgAttrs(idx, builder.getDictionaryAttr(builder.getNamedAttr(
                                  "emitc.field_ref", hintNameAttr)));
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertTFSavedModelAttrToEmitCAttrPass() {
  return std::make_unique<ConvertTFSavedModelAttrToEmitCAttrPass>();
}

}  // namespace tf_saved_model
}  // namespace mlir
