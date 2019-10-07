/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/SymbolTable.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir

namespace mlir {
namespace tf_saved_model {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool IsStrArrayAttr(Attribute attr) {
  auto array = attr.dyn_cast<ArrayAttr>();
  if (!array) return false;

  return llvm::all_of(array,
                      [](Attribute attr) { return attr.isa<StringAttr>(); });
}

//===----------------------------------------------------------------------===//
// TensorFlowSavedModelDialect Op's
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc.inc"

//===----------------------------------------------------------------------===//
// TensorFlowSavedModelDialect Dialect
//===----------------------------------------------------------------------===//

TensorFlowSavedModelDialect::TensorFlowSavedModelDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf_saved_model", context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.cc.inc"
      >();
}

LogicalResult TensorFlowSavedModelDialect::verifyRegionArgAttribute(
    Operation *op, unsigned region_index, unsigned arg_index,
    NamedAttribute named_attr) {
  if (named_attr.first == "tf_saved_model.bound_input") {
    if (!named_attr.second.isa<SymbolRefAttr>()) {
      return op->emitError() << "'tf_saved_model.bound_input' attribute should "
                                "be a SymbolRefAttr";
    }
    auto symbol_name = named_attr.second.cast<SymbolRefAttr>().getValue();
    auto module = op->getParentOfType<ModuleOp>();
    auto global_tensor = module.lookupSymbol<GlobalTensorOp>(symbol_name);
    if (!global_tensor) {
      return op->emitError() << "'tf_saved_model.bound_input' attribute must "
                                "reference a valid symbol, got invalid symbol '"
                             << symbol_name << "'";
    }
    // TODO(silvasean): Check that argument type matches with the value.
    return success();
  }

  return op->emitError() << "unknown tf_saved_model dialect arg attribute '"
                         << named_attr.first << "'";
}

static LogicalResult VerifySavedModelModule(
    ModuleOp module, TensorFlowSavedModelDialect *dialect) {
  auto exported_names_ident =
      Identifier::get("tf_saved_model.exported_names", dialect->getContext());
  // Check that there are no duplicated exported_names.
  DenseMap<StringRef, Operation *> exported_name_to_op;
  for (auto &op : module) {
    auto attr = op.getAttr(exported_names_ident);
    if (!attr) continue;
    // If this verifier is called before we verify the
    // 'tf_saved_model.exported_names' attribute, then it might be invalid.
    // Forward to the dialect's verification to establish that precondition.
    if (failed(dialect->verifyOperationAttribute(
            &op, {exported_names_ident, attr}))) {
      return failure();
    }
    for (auto str : attr.cast<ArrayAttr>()) {
      auto exported_name = str.cast<StringAttr>().getValue();
      auto p = exported_name_to_op.insert({exported_name, &op});
      if (!p.second) {
        return op.emitError()
            .append("duplicate exported name '", exported_name, "'")
            .attachNote(p.first->getSecond()->getLoc())
            .append("previously seen here");
      }
    }
  }
  return success();
}

LogicalResult TensorFlowSavedModelDialect::verifyOperationAttribute(
    Operation *op, NamedAttribute named_attr) {
  if (named_attr.first == "tf_saved_model.exported_names") {
    if (!isa<FuncOp>(op) && !isa<GlobalTensorOp>(op)) {
      return op->emitError() << "'tf_saved_model.exported_names' must be on a "
                                "'func' or 'tf_saved_model.global_tensor' op";
    }
    if (!IsStrArrayAttr(named_attr.second)) {
      return op->emitError()
             << "'tf_saved_model.exported_names' must be an array of strings";
    }
    if (!op->getParentOp()->getAttr("tf_saved_model.semantics")) {
      return op->emitError()
             << "'tf_saved_model.exported_names' must be on an op "
                "whose immediate parent has attribute "
                "'tf_saved_model.semantics'";
    }
    return success();
  }
  if (named_attr.first == "tf_saved_model.semantics") {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) {
      return op->emitError() << "'tf_saved_model.semantics' must "
                                "be on a module op";
    }
    return VerifySavedModelModule(module, this);
  }

  return op->emitError() << "unknown tf_saved_model dialect attribute '"
                         << named_attr.first << "'";
}

}  // namespace tf_saved_model
}  // namespace mlir
