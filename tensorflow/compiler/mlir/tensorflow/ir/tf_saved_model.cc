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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/SymbolTable.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
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

LogicalResult VerifyTensorTypesCompatible(Type t1, Type t2) {
  if (!t1.isa<TensorType>() || !t2.isa<TensorType>()) {
    return failure();
  }
  return verifyCompatibleShape(t1.cast<TensorType>(), t2.cast<TensorType>());
}

static LogicalResult Verify(GlobalTensorOp global_tensor) {
  if (failed(VerifyTensorTypesCompatible(
          global_tensor.type(), global_tensor.value().Attribute::getType()))) {
    return global_tensor.emitError() << "'type' and 'value' attributes should "
                                        "have compatible tensor types";
  }
  return success();
}

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

static LogicalResult VerifyIndexPath(Operation *op, NamedAttribute named_attr) {
  auto attr = named_attr.second.dyn_cast<ArrayAttr>();
  if (!attr) {
    return op->emitError()
           << "'tf_saved_model.index_path' attribute should be an ArrayAttr";
  }
  for (auto element : attr) {
    if (element.isa<StringAttr>()) {
      continue;
    }
    if (auto integer = element.dyn_cast<IntegerAttr>()) {
      if (integer.getValue().getBitWidth() == 64) {
        continue;
      }
    }
    return op->emitError() << "'tf_saved_model.index_path' elements should "
                              "be strings or 64-bit integers";
  }
  return mlir::success();
}

LogicalResult TensorFlowSavedModelDialect::verifyRegionArgAttribute(
    Operation *op, unsigned region_index, unsigned arg_index,
    NamedAttribute named_attr) {
  if (named_attr.first == "tf_saved_model.bound_input") {
    if (!named_attr.second.isa<FlatSymbolRefAttr>()) {
      return op->emitError() << "'tf_saved_model.bound_input' attribute should "
                                "be a FlatSymbolRefAttr";
    }
    auto symbol_name = named_attr.second.cast<FlatSymbolRefAttr>().getValue();
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
  if (named_attr.first == "tf_saved_model.index_path") {
    return VerifyIndexPath(op, named_attr);
  }

  return op->emitError() << "unknown tf_saved_model dialect arg attribute '"
                         << named_attr.first << "'";
}

LogicalResult TensorFlowSavedModelDialect::verifyRegionResultAttribute(
    Operation *op, unsigned region_index, unsigned result_index,
    NamedAttribute named_attr) {
  if (named_attr.first == "tf_saved_model.index_path") {
    return VerifyIndexPath(op, named_attr);
  }

  return op->emitError() << "unknown tf_saved_model dialect result attribute '"
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
  SymbolTable symbol_table(module);
  auto symbol_uses = SymbolTable::getSymbolUses(module);
  if (!symbol_uses.hasValue()) {
    return module.emitError() << "modules with 'tf_saved_model.semantics' must "
                                 "have analyzable symbol uses";
  }
  for (auto symbol_use : *symbol_uses) {
    auto func = symbol_table.lookup<FuncOp>(
        symbol_use.getSymbolRef().cast<FlatSymbolRefAttr>().getValue());
    if (func && !GetExportedNames(func).empty()) {
      return symbol_use.getUser()
          ->emitError("exported function cannot be internally referenced")
          .attachNote(func.getLoc())
          .append("references this exported function");
    }
  }
  return success();
}

LogicalResult VerifyExportedFunc(FuncOp func) {
  bool reached_bound_inputs = false;
  for (int i = 0, e = func.getNumArguments(); i < e; i++) {
    if (func.getArgAttr(i, "tf_saved_model.bound_input")) {
      reached_bound_inputs = true;
      continue;
    }
    if (func.getArgAttr(i, "tf_saved_model.index_path")) {
      if (reached_bound_inputs) {
        return func.emitError()
               << "all 'tf_saved_model.index_path' arg attributes should "
                  "precede all 'tf_saved_model.bound_input' arg attributes";
      }
      continue;
    }
    return func.emitError()
           << "all arguments should have 'tf_saved_model.index_path' or "
              "'tf_saved_model.bound_input' attributes";
  }
  llvm::SmallDenseSet<StringRef, 8> unique_bound_inputs;
  for (int i = 0, e = func.getNumArguments(); i < e; i++) {
    if (auto attr = func.getArgAttrOfType<FlatSymbolRefAttr>(
            i, "tf_saved_model.bound_input")) {
      if (!unique_bound_inputs.insert(attr.getValue()).second) {
        return func.emitError()
               << "duplicate 'tf_saved_model.bound_input' binding";
      }
    }
  }

  for (int i = 0, e = func.getNumResults(); i < e; i++) {
    if (!func.getResultAttr(i, "tf_saved_model.index_path")) {
      return func.emitError() << "all results should have "
                                 "'tf_saved_model.index_path' attributes";
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
    if (auto func = dyn_cast<FuncOp>(op)) {
      if (failed(VerifyExportedFunc(func))) {
        return failure();
      }
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

SmallVector<StringRef, 2> GetExportedNames(Operation *op) {
  SmallVector<StringRef, 2> ret;
  auto exported_names =
      op->getAttrOfType<ArrayAttr>("tf_saved_model.exported_names");
  if (exported_names) {
    for (auto name : exported_names) {
      ret.push_back(name.cast<StringAttr>().getValue());
    }
  }
  return ret;
}

bool IsExported(Operation *op) { return !GetExportedNames(op).empty(); }

bool HasTfSavedModelSemantics(ModuleOp module) {
  return module.getAttr("tf_saved_model.semantics") != nullptr;
}

GlobalTensorOp LookupBoundInput(FuncOp func, int arg_index,
                                const SymbolTable &symbol_table) {
  auto attr = func.getArgAttrOfType<FlatSymbolRefAttr>(
      arg_index, "tf_saved_model.bound_input");
  if (!attr) return nullptr;
  return symbol_table.lookup<GlobalTensorOp>(attr.getValue());
}

}  // namespace tf_saved_model
}  // namespace mlir
