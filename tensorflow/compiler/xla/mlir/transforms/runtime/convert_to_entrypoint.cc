/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string_view>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/passes.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/xla/mlir/transforms/runtime/passes.h.inc"

class ConvertToEntrypointPass
    : public ConvertToEntrypointBase<ConvertToEntrypointPass> {
  void runOnOperation() override;
};

static void ConvertCustomCallOperations(func::FuncOp func, Value exec_ctx) {
  MLIRContext* ctx = func->getContext();

  SymbolTable sym_table(func->getParentOfType<ModuleOp>());

  struct CustomCall {
    func::CallOp call;
    func::FuncOp callee;
    std::string_view target;
    bool direct;
  };

  // Collect function calls that have to be converted to custom calls.
  llvm::SmallVector<CustomCall> custom_calls;
  func.walk([&](func::CallOp op) {
    auto callee = dyn_cast<func::FuncOp>(sym_table.lookup(op.getCallee()));
    if (!callee) return;

    // Check if the call is an indirect custom call ...
    StringAttr target = callee->getAttrOfType<StringAttr>("rt.custom_call");
    if (target) custom_calls.push_back({op, callee, target.strref(), false});

    // ... or a direct custom call.
    target = callee->getAttrOfType<StringAttr>("rt.direct_custom_call");
    if (target) custom_calls.push_back({op, callee, target.strref(), true});
  });

  // After converting to custom call we need to clean up all declarations.
  llvm::DenseSet<func::FuncOp> erase_declarations;

  // Rewrite function calls to `rt.custom_call` operations.
  for (CustomCall custom_call : custom_calls) {
    ImplicitLocOpBuilder b(custom_call.call.getLoc(), custom_call.call);

    // Custom call intrinsic always returns the status flag.
    llvm::SmallVector<Type> results = {StatusType::get(ctx)};
    results.append(custom_call.call->getResultTypes().begin(),
                   custom_call.call->getResultTypes().end());

    // Rewrite function call with a custom call, and check the return status.
    auto call = b.create<CustomCallOp>(results, exec_ctx, custom_call.target,
                                       custom_call.direct,
                                       custom_call.call.getOperands());

    // Copy optional attributes from the custom call function declaration.
    llvm::ArrayRef<llvm::StringRef> callee_attrs =
        custom_call.callee.getAttributeNames();
    for (auto& attr : custom_call.callee->getAttrs()) {
      if (isa_and_nonnull<RuntimeDialect>(attr.getNameDialect())) continue;
      if (llvm::find(callee_attrs, attr.getName()) == callee_attrs.end())
        call->setAttr(attr.getName(), attr.getValue());
    }

    // Copy optional attributes from the call operation to the custom call.
    llvm::ArrayRef<llvm::StringRef> orig_attrs =
        custom_call.call.getAttributeNames();
    for (auto& attr : custom_call.call->getAttrs()) {
      if (llvm::find(orig_attrs, attr.getName()) == orig_attrs.end())
        call->setAttr(attr.getName(), attr.getValue());
    }

    b.create<cf::AssertOp>(
        b.create<IsOkOp>(TypeRange(b.getI1Type()), call.status()),
        b.getStringAttr("custom call '" + std::string(custom_call.target) +
                        "' failed"));

    // Forward users of the original results to custom call results.
    auto rets = llvm::zip(custom_call.call.getResults(),
                          llvm::drop_begin(call.getResults()));
    llvm::for_each(rets, [](auto ret) {
      std::get<0>(ret).replaceAllUsesWith(std::get<1>(ret));
    });

    // Keep track of custom call declaration to erase.
    erase_declarations.insert(custom_call.callee);

    // Erase the original function call operation.
    custom_call.call.erase();
  }

  // Erase all converted custom calls declarations.
  for (auto func : erase_declarations) sym_table.erase(func);
}

static void ConvertReturnOperations(func::FuncOp func, Value exec_ctx) {
  // Convert all returns to the Runtime API calls.
  func.walk([&](func::ReturnOp ret) {
    ImplicitLocOpBuilder b(ret.getLoc(), ret);

    // Return all outputs via the `rt.set_output` operation.
    for (auto& pair : llvm::enumerate(ret.getOperands())) {
      b.create<SetOutputOp>(exec_ctx, pair.index(), pair.value());
    }

    // Replace original return with an empty one.
    b.create<func::ReturnOp>();
    ret.erase();
  });

  // Update function type to the function with empty results.
  auto type = FunctionType::get(func.getContext(), func.getArgumentTypes(), {});
  func.setType(type);
}

static void ConvertAssertOperations(func::FuncOp func, Value exec_ctx) {
  // Collect all assert operations in the function body.
  llvm::SmallVector<cf::AssertOp> asserts;
  func.walk([&](cf::AssertOp op) {
    if (op->getParentOp() == func) asserts.push_back(op);
  });

  // Rewrite all asserts to the Runtime API calls.
  for (cf::AssertOp assert : asserts) {
    ImplicitLocOpBuilder b(assert.getLoc(), assert);

    // Split the block at the assert operation.
    Block* block = assert->getBlock();
    Block* ok = block->splitBlock(assert);

    // Set up block for returning error.
    Block* err = func.addBlock();
    b.setInsertionPointToStart(err);
    b.create<SetErrorOp>(exec_ctx, assert.getMsg());
    b.create<func::ReturnOp>();

    // Branch into the error block if assertion failed.
    b.setInsertionPointToEnd(block);
    b.create<cf::CondBranchOp>(assert.getArg(), ok, err);

    // Erase the original assert operation.
    assert.erase();
  }
}

static Value PrependExecutionContextArgument(func::FuncOp func) {
  Type new_type = KernelContextType::get(func.getContext());
  DictionaryAttr attr = DictionaryAttr::get(func.getContext());
  func.insertArguments({0}, {new_type}, {attr}, {func.getLoc()});
  return func.getArgument(0);
}

static void ConvertToEntrypoint(func::FuncOp func) {
  assert(func->hasAttr(kEntrypointAttrName));

  Value exec_ctx = PrependExecutionContextArgument(func);
  ConvertCustomCallOperations(func, exec_ctx);
  ConvertReturnOperations(func, exec_ctx);
  ConvertAssertOperations(func, exec_ctx);

  // After conversion !rt.execution_context is a marker of an entrypoint.
  func->removeAttr(kEntrypointAttrName);
}

void ConvertToEntrypointPass::runOnOperation() {
  llvm::SmallVector<func::FuncOp> entry_points;

  // Collect entrypoint functions.
  getOperation().walk([&](func::FuncOp op) {
    if (op->hasAttr(kEntrypointAttrName)) entry_points.push_back(op);
  });

  llvm::for_each(entry_points, ConvertToEntrypoint);
}

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertToEntrypoint() {
  return std::make_unique<ConvertToEntrypointPass>();
}

}  // namespace runtime
}  // namespace xla
