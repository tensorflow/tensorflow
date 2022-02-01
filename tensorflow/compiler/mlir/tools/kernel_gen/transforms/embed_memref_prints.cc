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

#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/Liveness.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

using tf_framework::TFFrameworkDialect;

Operation* emitCallToPrint(Location loc, StringRef func_name, Value arg,
                           OpBuilder* b) {
  auto caller_func =
      b->getInsertionBlock()->getParent()->getParentOfType<FuncOp>();
  auto func_name_attr = b->getStringAttr(func_name);
  auto callee_func =
      SymbolTable::lookupNearestSymbolFrom<FuncOp>(caller_func, func_name_attr);
  if (!callee_func) {
    OpBuilder::InsertionGuard insertGuard(*b);

    auto module = caller_func->getParentOfType<ModuleOp>();
    b->setInsertionPointToStart(module.getBody());
    auto func_type = FunctionType::get(b->getContext(), arg.getType(),
                                       /*results=*/llvm::None);
    callee_func = b->create<FuncOp>(module.getLoc(), func_name, func_type);
    callee_func.setPrivate();
  }
  return b->create<CallOp>(loc, callee_func, arg);
}

void EmitPrint(Operation* op, Liveness& liveness, OpBuilder* b) {
  Location loc = op->getLoc();
  Value memref = op->getResult(0);
  auto memref_type = memref.getType().cast<MemRefType>();
  Type element_type = memref_type.getElementType();
  if (!element_type.isF32() && !element_type.isF64() &&
      !element_type.isIntOrIndex())
    return;

  Operation* end_op =
      liveness.getLiveness(op->getBlock())->getEndOperation(memref, op);
  b->setInsertionPoint(end_op);

  if (element_type.isIndex()) {
    element_type = b->getI64Type();
    memref_type =
        MemRefType::get(memref_type.getShape(), element_type,
                        memref_type.getLayout(), memref_type.getMemorySpace());
    memref = b->create<arith::IndexCastOp>(loc, memref, memref_type);
  }

  auto unranked_type =
      UnrankedMemRefType::get(element_type, memref_type.getMemorySpaceAsInt());
  Value unranked_memref = b->create<memref::CastOp>(loc, memref, unranked_type);

  if (element_type.isF32()) {
    emitCallToPrint(loc, "print_memref_f32", unranked_memref, b);
    return;
  }
  if (element_type.isF64()) {
    emitCallToPrint(loc, "print_memref_f64", unranked_memref, b);
    return;
  }
  if (element_type.isInteger(32)) {
    emitCallToPrint(loc, "print_memref_i32", unranked_memref, b);
    return;
  }
  if (element_type.isInteger(64) || element_type.isIndex()) {
    emitCallToPrint(loc, "print_memref_i64", unranked_memref, b);
    return;
  }
}

// The pass the memrefs allocated in a `tf-entry` function and inserts printing
// at the end of their lifetime. Printing for buffers allocated with TFAllocOp
// is currently not supported because the data is not located on host.
struct EmbedMemRefPrintsPass
    : public EmbedMemRefPrintsPassBase<EmbedMemRefPrintsPass> {
  void runOnFunction() override {
    FuncOp func = getFunction();
    if (!func->getAttrOfType<UnitAttr>(TFFrameworkDialect::kTFEntryAttrName))
      return;

    Liveness liveness(func);
    OpBuilder b(&getContext());
    func.walk([&](memref::AllocOp op) { EmitPrint(op, liveness, &b); });
    func.walk([&](memref::AllocaOp op) { EmitPrint(op, liveness, &b); });
    func.walk(
        [&](memref::ReinterpretCastOp op) { EmitPrint(op, liveness, &b); });
  }
};

}  // namespace

std::unique_ptr<FunctionPass> CreateEmbedMemRefPrintsPass() {
  return std::make_unique<EmbedMemRefPrintsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
