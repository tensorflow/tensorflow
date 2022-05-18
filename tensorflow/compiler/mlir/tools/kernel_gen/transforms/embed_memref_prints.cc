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

#include <memory>
#include <string>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/utils.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

constexpr StringRef kPrintStringFuncName = "printCString";

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

Operation* EmitMemRefPrint(Location loc, Type element_type, Value arg,
                           OpBuilder* b) {
  StringRef func_name;
  if (element_type.isF32()) {
    func_name = "printMemrefF32";
  }
  if (element_type.isF64()) {
    func_name = "printMemrefF64";
  }
  if (element_type.isInteger(32)) {
    func_name = "printMemrefI32";
  }
  if (element_type.isInteger(64) || element_type.isIndex()) {
    func_name = "printMemrefI64";
  }
  assert(!func_name.empty() &&
         "Did not find a print function for the element type");

  auto caller_func =
      b->getInsertionBlock()->getParent()->getParentOfType<func::FuncOp>();
  auto func_name_attr = b->getStringAttr(func_name);

  auto callee_func = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      caller_func, func_name_attr);
  if (!callee_func) {
    OpBuilder::InsertionGuard insertGuard(*b);

    auto module = caller_func->getParentOfType<ModuleOp>();
    b->setInsertionPointToStart(module.getBody());
    auto func_type = FunctionType::get(b->getContext(), arg.getType(),
                                       /*results=*/llvm::None);
    callee_func =
        b->create<func::FuncOp>(module.getLoc(), func_name, func_type);
    callee_func.setPrivate();
  }
  return b->create<func::CallOp>(loc, callee_func, arg);
}

bool IsElementTypePrintalble(Type element_type) {
  return element_type.isF32() || element_type.isF64() ||
         element_type.isInteger(32) || element_type.isInteger(64) ||
         element_type.isIndex();
}

void EmitMemRefPrint(Location loc, Value memref, OpBuilder* b) {
  auto memref_type = memref.getType();
  if (auto unranked_type = memref_type.dyn_cast<UnrankedMemRefType>()) {
    Type element_type = unranked_type.getElementType();
    if (!IsElementTypePrintalble(element_type)) return;

    EmitMemRefPrint(loc, element_type, memref, b);
  }
  if (auto ranked_type = memref_type.dyn_cast<MemRefType>()) {
    Type element_type = ranked_type.getElementType();
    if (!IsElementTypePrintalble(element_type)) return;

    if (element_type.isIndex()) {
      element_type = b->getI64Type();
      ranked_type = MemRefType::get(ranked_type.getShape(), element_type,
                                    ranked_type.getLayout(),
                                    ranked_type.getMemorySpace());
      memref = b->create<arith::IndexCastOp>(loc, ranked_type, memref);
    }

    auto unranked_type = UnrankedMemRefType::get(
        element_type, ranked_type.getMemorySpaceAsInt());
    Value unranked_memref =
        b->create<memref::CastOp>(loc, unranked_type, memref);
    EmitMemRefPrint(loc, element_type, unranked_memref, b);
  }
}

SmallVector<Value> ExtractValuesToPrint(Operation* op) {
  if (isa<memref::ReinterpretCastOp>(op) || isa<memref::ReshapeOp>(op) ||
      isa<memref::ExpandShapeOp>(op) || isa<memref::CollapseShapeOp>(op)) {
    return {op->getResult(0)};
  }
  if (auto linalg = dyn_cast<linalg::LinalgOp>(op)) {
    return linalg.getOutputBufferOperands();
  }
  if (auto loop = dyn_cast<gml_st::LoopOp>(op)) {
    return loop.outputs();
  }
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    return loop.getIterOperands();
  }
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    return {copy.target()};
  }
  return {};
}

void EmitOperationPrint(Operation* op, OpBuilder* b) {
  std::string debug_str = "\n\nPrint memref content after the following op\n";
  llvm::raw_string_ostream output_stream(debug_str);

  mlir::OpPrintingFlags flags;
  op->print(output_stream, flags);
  output_stream << "\n\n";

  Location loc = op->getLoc();
  Value message_constant = CreateOrFindGlobalStringConstant(
      loc, GetGlobalName("debug_op", debug_str), debug_str, b);

  // Insert function call.
  MLIRContext* ctx = op->getContext();
  auto func_type = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(op->getContext()),
      {LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8))});
  FlatSymbolRefAttr tf_func_ref =
      GetOrInsertLLVMFunction(kPrintStringFuncName, func_type, op, b);
  b->create<LLVM::CallOp>(loc, llvm::None, tf_func_ref,
                          llvm::makeArrayRef({message_constant}));
}

// The pass inserts printing on every mutation of memrefs.
struct EmbedMemRefPrintsPass
    : public EmbedMemRefPrintsPassBase<EmbedMemRefPrintsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp func) {
      if (func.isDeclaration()) return;
      Block* body = &func.getBody().front();

      // Print arguments.
      OpBuilder b(&getContext());
      b.setInsertionPointToStart(body);
      Location loc = func.getLoc();
      auto args = func.getArguments();
      if (!args.empty()) {
        EmitOperationPrint(func, &b);
      }
      for (auto arg : args) {
        EmitMemRefPrint(loc, arg, &b);
      }
      // Print buffers after every change.
      for (auto& op : func.getBody().front().getOperations()) {
        b.setInsertionPointAfter(&op);
        auto memrefs = ExtractValuesToPrint(&op);
        if (!memrefs.empty()) {
          EmitOperationPrint(&op, &b);
        }
        for (auto memref : memrefs) {
          EmitMemRefPrint(op.getLoc(), memref, &b);
        }
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateEmbedMemRefPrintsPass() {
  return std::make_unique<EmbedMemRefPrintsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
