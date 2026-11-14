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

#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/emitc/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_dataflow.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_ADDREFLECTIONMAPPASS
#include "tensorflow/compiler/mlir/emitc/transforms/passes.h.inc"

namespace {
class AddReflectionMapPass
    : public impl::AddReflectionMapPassBase<AddReflectionMapPass> {
  void runOnOperation() final;
};

void AddReflectionMapPass::runOnOperation() {
  emitc::ClassOp classOp = getOperation();
  OpBuilder builder(classOp);

  mlir::MLIRContext* context = builder.getContext();
  emitc::OpaqueType stringViewType =
      mlir::emitc::OpaqueType::get(builder.getContext(), "std::string_view");
  emitc::OpaqueType charPtrType =
      mlir::emitc::OpaqueType::get(builder.getContext(), "char");
  emitc::OpaqueType mapType = mlir::emitc::OpaqueType::get(
      builder.getContext(), "const std::map<std::string, char*>");

  FunctionType funcType =
      builder.getFunctionType({stringViewType}, {charPtrType});
  emitc::FuncOp executeFunc =
      classOp.lookupSymbol<mlir::emitc::FuncOp>("execute");
  builder.setInsertionPoint(executeFunc);

  emitc::FuncOp getBufferFunc = builder.create<mlir::emitc::FuncOp>(
      classOp.getLoc(), "getBufferForName", funcType);

  Block* funcBody = getBufferFunc.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  // Collect all field names
  SmallVector<std::string> fieldNames;
  classOp.walk([&](mlir::emitc::FieldOp fieldOp) {
    if (mlir::Attribute attrsAttr = fieldOp->getAttrDictionary().get("attrs")) {
      if (DictionaryAttr innerDictAttr =
              dyn_cast<mlir::DictionaryAttr>(attrsAttr)) {
        auto indexPathAttr =
            innerDictAttr.getNamed("tf_saved_model.index_path");
        ArrayAttr arrayAttr =
            dyn_cast<mlir::ArrayAttr>(indexPathAttr->getValue());
        if (!arrayAttr.empty()) {
          StringAttr stringAttr = dyn_cast<mlir::StringAttr>(arrayAttr[0]);
          std::string indexPath = stringAttr.getValue().str();
          fieldNames.push_back(indexPath);
        }
        if (arrayAttr.size() > 1) {
          fieldOp.emitError() << "tf_saved_model.index_path attribute must "
                                 "contain at most one value, but found "
                              << arrayAttr.size() << " values.";
          return;
        }
      }
    }
  });

  std::string mapInitializer = "{ ";
  for (size_t i = 0; i < fieldNames.size(); ++i) {
    mapInitializer += " { \"" + fieldNames[i] + "\", " +
                      "reinterpret_cast<char*>(&" + fieldNames[i] + ")",
        mapInitializer += " }";
    if (i < fieldNames.size() - 1) mapInitializer += ", ";
  }
  mapInitializer += " }";

  emitc::OpaqueType iteratorType = mlir::emitc::OpaqueType::get(
      context, "std::map<std::string, char*>::const_iterator");

  emitc::ConstantOp bufferMap = builder.create<emitc::ConstantOp>(
      classOp.getLoc(), mapType,
      emitc::OpaqueAttr::get(context, mapInitializer));

  mlir::Value nameArg = getBufferFunc.getArgument(0);
  emitc::CallOpaqueOp it = builder.create<emitc::CallOpaqueOp>(
      classOp.getLoc(), iteratorType, builder.getStringAttr("find"),
      mlir::ValueRange{bufferMap.getResult(), nameArg});
  emitc::CallOpaqueOp endIt = builder.create<emitc::CallOpaqueOp>(
      classOp.getLoc(), iteratorType, builder.getStringAttr("end"),
      bufferMap.getResult());
  emitc::CallOpaqueOp isEnd = builder.create<emitc::CallOpaqueOp>(
      classOp.getLoc(), builder.getI1Type(),
      "operator==", mlir::ValueRange{it.getResult(0), endIt.getResult(0)});
  emitc::ConstantOp nullPtr = builder.create<emitc::ConstantOp>(
      classOp.getLoc(), charPtrType,
      emitc::OpaqueAttr::get(context, "nullptr"));
  emitc::CallOpaqueOp second = builder.create<emitc::CallOpaqueOp>(
      classOp.getLoc(), charPtrType, "second", it.getResult(0));

  emitc::ConditionalOp result = builder.create<emitc::ConditionalOp>(
      classOp.getLoc(), charPtrType, isEnd.getResult(0), nullPtr.getResult(),
      second.getResult(0));

  builder.create<emitc::ReturnOp>(classOp.getLoc(), result.getResult());
}

}  // namespace
std::unique_ptr<mlir::OperationPass<mlir::emitc::ClassOp>>
CreateAddReflectionMapPass() {
  return std::make_unique<AddReflectionMapPass>();
}

}  // namespace emitc
}  // namespace mlir
