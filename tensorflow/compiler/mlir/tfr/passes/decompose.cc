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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"
#include "tensorflow/compiler/mlir/tfr/ir/tfr_types.h"
#include "tensorflow/compiler/mlir/tfr/passes/passes.h"
#include "tensorflow/compiler/mlir/tfr/utils/utils.h"
#include "tensorflow/core/lib/monitoring/counter.h"

namespace tensorflow {
namespace {

auto* tf_core_op_expansion_op_counter =
    monitoring::Counter<1>::New("/tensorflow/core/op_expansion/op_counter",
                                "The number of composite op expanded.", "name");
}

void IncreaseOpExpansionExecuteCounterByOne(const std::string& op_name) {
  tf_core_op_expansion_op_counter->GetCell(op_name)->IncrementBy(1);
}

}  // namespace tensorflow

//===----------------------------------------------------------------------===//
// The pass to decompose unregistered TF ops with the TFR compose function.
//
namespace mlir {
namespace TFR {

namespace {

// Quantize the float value based on given scale and zero point attributes.
Attribute Quantize(float value, Attribute scale_attr, Attribute zp_attr,
                   OpBuilder builder) {
  double scale = scale_attr.cast<FloatAttr>().getValueAsDouble();
  int64_t zp = zp_attr.cast<IntegerAttr>().getInt();

  int quantized = static_cast<int>(std::round(value / scale) + zp);
  quantized =
      std::min(quantized, static_cast<int>(std::numeric_limits<int8_t>::max()));
  quantized =
      std::max(quantized, static_cast<int>(std::numeric_limits<int8_t>::min()));
  return builder.getI32IntegerAttr(quantized);
}

// Decompose the TF ops with the registered composition library.
class DecomposeTFOpsPass
    : public PassWrapper<DecomposeTFOpsPass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeTFOpsPass)

  explicit DecomposeTFOpsPass(llvm::Optional<ModuleOp> external_tfr_module)
      : external_tfr_module_(external_tfr_module) {}

  StringRef getArgument() const final { return "tfr-decompose"; }

  StringRef getDescription() const final {
    return "Decompose TF ops with the registered composition library.";
  }

  void runOnOperation() override;

 private:
  // Apply canonicalization, mainly constant folding, on the function.
  void ApplyCanonicalization();

  // Rewrite unregistered TF ops to TFR func call ops. Return failure if all the
  // ops are registered or the compose function doesn't exist.
  LogicalResult RewriteUnregisteredTFOps();

  // Inline the TFR func call ops.
  LogicalResult InlineTFRFuncCalls();

  // Optional external symbol table to look up the TFR function.
  llvm::Optional<ModuleOp> external_tfr_module_;
};

#include "tensorflow/compiler/mlir/tfr/passes/generated_decompose.inc"

void DecomposeTFOpsPass::ApplyCanonicalization() {
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(&getContext());

  populateWithGenerated(patterns);
  populateCanonicalizationPatterns(func, patterns);

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

LogicalResult DecomposeTFOpsPass::RewriteUnregisteredTFOps() {
  func::FuncOp func = getOperation();
  SymbolTable table(external_tfr_module_.hasValue()
                        ? *external_tfr_module_
                        : func->getParentOfType<ModuleOp>());
  OpBuilder builder(func);
  bool changed = false;
  func.walk([&table, &builder, &changed](Operation* op) {
    // Only the un-registered ops requires decomposition. The remaining ones
    // either will be constant folded or lowered by the rules defined in the
    // bridge.
    if (op->isRegistered()) {
      return WalkResult::advance();
    }

    // Find out the compose function
    auto compose_func_name = GetComposeFuncName(op->getName().getStringRef());
    auto compose_func = table.lookup<TFRFuncOp>(compose_func_name);
    if (!compose_func || compose_func.isExternal()) {
      // There are no decomposition methods defined for this op, skip.
      return WalkResult::advance();
    }

    // Make sure all the attributes are valid. An attribute is valid when it is
    // in the signature or it is allowed explicitly.
    auto compose_func_signature =
        table.lookup<TFRFuncOp>(compose_func_name + "_");
    if (!compose_func_signature) compose_func_signature = compose_func;
    auto defined_attrs = compose_func_signature.getDefinedAttributeNames();
    if (failed(ValidateAttrs(op, defined_attrs))) {
      return WalkResult::interrupt();
    }

    tensorflow::IncreaseOpExpansionExecuteCounterByOne(
        op->getName().getStringRef().str());

    auto compose_func_type = compose_func.getFunctionType();
    builder.setInsertionPoint(op);
    TFRTensorType unconstrainted_tensor_type = builder.getType<TFRTensorType>();

    // Create the new operands. This is mapping the operands from the target
    // TF ops to the TFR function arguments. If the TFR function argument is
    // a tensor_list, a "tfr.build_list" op is used to concat the available
    // TF op operands. If the TFR function argument isn't a tensor/tensor_list,
    // a constant is created by using the attribute stored in the TF op or the
    // default value in the argument attribute.
    llvm::SmallVector<Value, 4> new_operands;
    for (auto arg : llvm::enumerate(compose_func_type.getInputs())) {
      if (auto tensor_type = arg.value().dyn_cast<TFRTensorType>()) {
        auto casted = builder.create<CastOp>(op->getLoc(), tensor_type,
                                             op->getOperand(arg.index()));
        new_operands.push_back(casted);
      } else if (auto list_type = arg.value().dyn_cast<TFRTensorListType>()) {
        llvm::SmallVector<Value, 4> variadic_operands;
        for (int i = arg.index(); i < op->getNumOperands(); i++) {
          auto casted = builder.create<CastOp>(
              op->getLoc(), unconstrainted_tensor_type, op->getOperand(i));
          variadic_operands.push_back(casted);
        }
        auto build_list_op = builder.create<BuildListOp>(
            op->getLoc(), list_type, variadic_operands);
        new_operands.push_back(build_list_op.out());
      } else {
        auto attr_name = compose_func.getArgAttrOfType<StringAttr>(
            arg.index(), kAttrArgumentNameAttr);
        auto attribute = op->getAttr(attr_name.getValue());
        if (!attribute) {
          attribute =
              compose_func.getArgAttr(arg.index(), kAttrArgumentDefaultAttr);
        }
        if (!attribute && attr_name.getValue() == "out_type") {
          auto type = op->getResult(0).getType();
          if (type.isa<TensorType>()) {
            type = type.cast<TensorType>().getElementType();
          }
          attribute = TypeAttr::get(type);
        }
        Value attr_cst;
        // Wrap these special attributes as a special TFR constant, so the SSA
        // value has a valid type to be used as TFR function argument. These
        // attributes are not expected to be manipulated by the lowering passes.
        if (attribute.isa<TypeAttr>() || attribute.isa<ArrayAttr>() ||
            attribute.isa<StringAttr>() || attribute.isa<FlatSymbolRefAttr>()) {
          TFRAttrType output_type = TFRAttrType::get(builder.getContext());
          attr_cst =
              builder.create<ConstOp>(op->getLoc(), output_type, attribute);
        } else {
          attr_cst =
              builder.create<mlir::arith::ConstantOp>(op->getLoc(), attribute);
        }
        new_operands.push_back(attr_cst);
      }
    }

    // Create the TFR call op
    auto new_op = builder.create<CallOp>(
        op->getLoc(), compose_func_type.getResults(),
        SymbolRefAttr::get(builder.getContext(), compose_func.getName()),
        new_operands);

    // Replace the use of the old op. This is mapping the results from the
    // target TF ops to the TFR function returns. If the TFR function return is
    // a tensor_list, "tfr.get_element" op is used to extract the required TF
    // op result.
    llvm::SmallVector<Value, 4> new_results;
    for (auto res : llvm::enumerate(compose_func_type.getResults())) {
      if (res.value().dyn_cast<TFRTensorType>()) {
        new_results.push_back(new_op.getResult(res.index()));
      } else if (auto list_type = res.value().dyn_cast<TFRTensorListType>()) {
        for (int i = res.index(), j = 0; i < op->getNumResults(); i++, j++) {
          auto index = builder.create<mlir::arith::ConstantOp>(
              op->getLoc(), builder.getIndexAttr(j));
          auto element_op = builder.create<GetElementOp>(
              op->getLoc(), unconstrainted_tensor_type,
              new_op.getResult(res.index()), index.getResult());
          new_results.push_back(element_op.out());
        }
      }
    }
    for (auto res : llvm::zip(op->getResults(), new_results)) {
      auto casted = builder.create<CastOp>(
          op->getLoc(), std::get<0>(res).getType(), std::get<1>(res));
      std::get<0>(res).replaceAllUsesWith(casted.out());
    }

    // Copy all the unregisted attributes to the new op.
    if (failed(CopyAllowedUnregisteredAttrs(op, new_op, defined_attrs))) {
      return WalkResult::interrupt();
    }

    op->erase();
    changed |= true;
    return WalkResult::advance();
  });

  // If `changed` is false, it is considered as a failure, so the recursive
  // rewrite will stop.
  return success(changed);
}

LogicalResult DecomposeTFOpsPass::InlineTFRFuncCalls() {
  // The Inliner will automatically use the registered dialect inliner.
  InlinerInterface inliner(&getContext());
  func::FuncOp func = getOperation();
  SymbolTable table(external_tfr_module_.hasValue()
                        ? *external_tfr_module_
                        : func->getParentOfType<ModuleOp>());

  // The inliner only inlines the TFR call op.
  bool changed = false;
  auto walk_result = func.walk([&](CallOp call_op) {
    auto callee = table.lookup<TFRFuncOp>(call_op.callee());
    if (!callee || callee.isExternal()) return WalkResult::advance();

    // Record the boundary of the inlined operations. The inlined operation will
    // be inserted between these two operations.
    Operation* inlined_point = call_op.getOperation();
    Operation* after_inlined_point =
        &*std::next(Block::iterator(call_op.getOperation()));

    // Use the inliner to replace all the uses of the call_op by its
    // composition.
    if (failed(inlineCall(inliner,
                          cast<CallOpInterface>(call_op.getOperation()),
                          cast<CallableOpInterface>(callee.getOperation()),
                          callee.getCallableRegion(),
                          /**shouldCloneInLinedRegion=*/true))) {
      // This failure is usually because the decompose function is not defined.
      // This call will be raised to TF ops.
      return WalkResult::interrupt();
    }

    // Propagate all the attributes to the inlined operations, which are defined
    // by the two boundary operations.
    PropagateAttrsToOperations(call_op, Block::iterator(inlined_point),
                               Block::iterator(after_inlined_point));

    // Remove the call_op to finish the op expansion.
    call_op.erase();
    changed |= true;
    return WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    signalPassFailure();
    return failure();
  }

  // If `changed` is false, it is considered as a failure, so the recursive
  // rewrite will stop.
  return success(changed);
}

void DecomposeTFOpsPass::runOnOperation() {
  // Set a maximum iteration threshold in case there are infinite loops in the
  // call stack.
  int max_iterators = 10;
  do {
    // canonicalization
    ApplyCanonicalization();

    // rewrite unregistered tf ops. Failed either because no ops can be
    // decomposed or the compose function isn't defined.
    auto rewrite_status = RewriteUnregisteredTFOps();
    // inline the tfr call op until there are no tfr.call op can be inlined.
    auto inline_status = InlineTFRFuncCalls();

    if (failed(rewrite_status) && failed(inline_status)) {
      break;
    }
  } while (max_iterators-- >= 0);
}

}  // namespace

// Creates an instance of the pass to decompose the TF ops.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDecomposeTFOpsPass(
    llvm::Optional<ModuleOp> tfr_module) {
  return std::make_unique<DecomposeTFOpsPass>(tfr_module);
}

static PassRegistration<DecomposeTFOpsPass> pass([] {
  return CreateDecomposeTFOpsPass();
});

}  // namespace TFR
}  // namespace mlir
