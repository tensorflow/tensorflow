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

#include <cstddef>
#include <string>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_ops.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"

namespace mlir {
namespace {

constexpr const char kTmpLoweringCastOpName[] = "tmp_lowering_cast_op";

static Type GetChainType(MLIRContext* context) {
  auto hexDialect = Identifier::get("hex", context);
  return OpaqueType::get(hexDialect, "chain", context);
}

static Type GetTfdTensorType(MLIRContext* context) {
  auto tfdDialect = Identifier::get("tfd", context);
  return OpaqueType::get(tfdDialect, "tf_tensor", context);
}

struct TfToTfdLoweringPass
    : public PassWrapper<TfToTfdLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
};

class FuncOpSignatureConversion : public OpConversionPattern<FuncOp> {
 public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FuncOp funcOp, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = funcOp.getContext();
    auto chain_type = GetChainType(ctx);
    auto tfd_tensor_type = GetTfdTensorType(ctx);
    FunctionType type = funcOp.getType();

    // Convert function return results. The lowered function is expected to
    // return a chain as the first return result. For each original TF tensor,
    // the lowered function returns a TFD tensor instead.
    llvm::SmallVector<Type, 2> converted_results;
    if (type.getNumResults() > 0) {
      // Add a chain as the first return result.
      converted_results.push_back(chain_type);

      // Convert the original TF tensor return results.
      for (unsigned i = 0, e = type.getNumResults(); i != e; ++i) {
        if (auto tensor_type = type.getResult(i).dyn_cast<TensorType>()) {
          // Each TF tensor is converted to a TFD tensor.
          converted_results.push_back(tfd_tensor_type);
        } else {
          // Only handle TF tensor conversion for now.
          return failure();
        }
      }
    }

    // Create the new function signature. The lowered function is expected to
    // take a Chain as the first argument. Then for each TF tensor argument,
    // expect a TFD tensor argument instead.
    TypeConverter::SignatureConversion new_func_sig(type.getNumInputs() + 1);
    if (type.getNumInputs() > 0) {
      // Add the first chain argument.
      new_func_sig.addInputs(chain_type);
      for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i) {
        // For each original TF tensor type, convert it to one TFD tensor type.
        if (auto tensor_type = type.getInput(i).dyn_cast<TensorType>()) {
          new_func_sig.addInputs(i, {tfd_tensor_type});
        } else {
          // Only handle TF tensor argument for now.
          return failure();
        }
      }
    }
    // Each function has a single region. In general, each region can have
    // multiple blocks. Assume that all TF-dialect functions only have a
    // single entry block.
    Block* entry = &funcOp.front();

    // Tell the rewriter to convert the region signature. After this, the
    // function region takes the new function signature, which means index
    // shifts by one.
    Block* convertedEntry =
        rewriter.applySignatureConversion(&funcOp.getBody(), new_func_sig);

    {
      // Generate the "fake" mapping ops. The insertion guard restores rewriter
      // insertion pointer when it gets out of scope.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(convertedEntry);
      // Replace block arguments. For example,
      // func @example(i64, i1) -> i64 {
      //   ^bb0(%a: i64, %cond: i1):  // replacing this.
      for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i) {
        // For each original block argument, create a fake op that takes the
        // input the input chain argument to the function, and the tfd tensor
        // argument, and returns the original TF tensor input. Note that the
        // function signature has been replaced, so entry->getArgument(0) is the
        // input chain. And we need to add 1 to index to get the original
        // argument.
        Type orig_input = type.getInput(i);
        OperationState tmp_lowering_cast_op(
            funcOp.getLoc(), kTmpLoweringCastOpName,
            {convertedEntry->getArgument(0),
             convertedEntry->getArgument(i + 1)},
            orig_input, {});
        Value repl_value =
            rewriter.createOperation(tmp_lowering_cast_op)->getResult(0);
        // Replace original uses of TF tensor block argument with the result of
        // the fake op. This sets up the lowering passes for individual ops
        // which at this point still expect TF tensors rather than TFD tensor
        // inputs.
        rewriter.replaceUsesOfBlockArgument(entry->getArgument(i), repl_value);
      }
    }

    // Create a new function op with an updated signature.
    auto new_func_op = rewriter.cloneWithoutRegions(funcOp);
    rewriter.inlineRegionBefore(funcOp.getBody(), new_func_op.getBody(),
                                new_func_op.end());
    new_func_op.setType(FunctionType::get(new_func_sig.getConvertedTypes(),
                                          converted_results, ctx));
    // Remove the old function op.
    rewriter.eraseOp(funcOp);
    return success();
  }
};

// Lower each TF op to a tfd.delegate_kernel op. For example,
//
// %1 = "tf.ReadVariableOp"(%arg) {
//     dtype = "tfdtype$DT_FLOAT"
// } : (tensor<*x!tf.resource>) -> tensor<10xf32>
//
// would be lowered to
//
// %1:2 = "tfd.delegate_kernel"(%chain_in, %arg) {
//   _name = "tf.ReadVariableOp",
//   attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT"
// } : (!hex.chain, !tfd.tf_tensor) -> (!hex.chain, !tfd.tf_tensor)
//
// Each tfd.delegate_kernel op expects a chain as the first input. This chain
// may come from the first function argument or the previous converted op
// output. The rest of inputs would be converted to a tfd tensor input.
// Each tfd.delegate_kernel op returns a chain as the first output. Each
// original output TensorType is converted a tfd tensor type.
// The TF op name becomes an _name attribute. Each TF attribute is lowered to
// two TFD attributes, one for the name, one for the type and value.
//
// Because delegate_kernel ops are threaded through chains, we lowered to a
// serial execution plan.
// TODO(zhangqiaorjc): Do analysis to allow concurrent execution.
template <typename TF_OP>
class TFOpConversion : public OpConversionPattern<TF_OP> {
 public:
  using OpConversionPattern<TF_OP>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF_OP op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter)  // NOLINT(google-runtime-references
      const override {
    auto ctx = op.getContext();
    // Handle new op operands.
    // Delegate kernel expects the first argument to be a chain, followed by
    // original arguments to the target TF op converted to TFD tensors.
    llvm::SmallVector<Value, 4> delegate_kernel_op_operands;
    int num_new_operands = op.getOperation()->getNumOperands() + 1;
    delegate_kernel_op_operands.reserve(num_new_operands);

    // Get the input chain from the previous delegate_kernel op or first block
    // argument.
    Value chain_input = nullptr;
    auto* block = op.getOperation()->getBlock();
    assert(block->isEntryBlock() && "only supports a single block");
    // Find a previous delegate_kernel op for its output chain.
    auto* prev_op = op.getOperation()->getPrevNode();
    while (prev_op != nullptr && !isa<tfd::DelegateKernelOp>(prev_op)) {
      prev_op = prev_op->getPrevNode();
    }
    if (prev_op != nullptr) {
      // There is another delegate kernel op before this op.
      auto prev_op_result_0 = prev_op->getResult(0);
      assert(prev_op_result_0.getType() == GetChainType(ctx));
      chain_input = prev_op_result_0;
    } else {
      // This op is the first delegate kernel op in a block.
      auto arg_0 = block->getArgument(0);
      assert(arg_0.getType() == GetChainType(ctx));
      chain_input = arg_0;
    }
    delegate_kernel_op_operands.push_back(chain_input);

    // Convert each TensorType operand to the corresponding TFD tensor operand.
    for (auto operand : operands) {
      auto* tmp_lowering_cast_op = operand.getDefiningOp();
      assert(tmp_lowering_cast_op->getName().getStringRef() ==
             kTmpLoweringCastOpName);
      delegate_kernel_op_operands.push_back(
          tmp_lowering_cast_op->getOperand(1));
    }

    // Handle new op results.
    llvm::SmallVector<Type, 4> delegate_kernel_op_results;
    // The first output is a chain.
    delegate_kernel_op_results.push_back(GetChainType(ctx));
    // For each original output, there is a corresponding TFD tensor output.
    for (int i = 0, e = op.getOperation()->getNumResults(); i != e; ++i) {
      delegate_kernel_op_results.push_back(GetTfdTensorType(ctx));
    }

    // Convert TF attribute to TFD attribute.
    llvm::SmallVector<NamedAttribute, 4> delegate_kernel_op_attributes;
    NamedAttribute op_name_attr(Identifier::get("_name", ctx),
                                StringAttr::get(op.getOperationName(), ctx));
    delegate_kernel_op_attributes.push_back(op_name_attr);

    int attr_idx = 0;
    for (const NamedAttribute& tf_attr : op.getAttrs()) {
      // Small std::string benefits from small string optimization in libc++.
      NamedAttribute attr_name(
          Identifier::get("attr" + std::to_string(attr_idx) + "_name", ctx),
          StringAttr::get(tf_attr.first, ctx));
      NamedAttribute attr_value(
          Identifier::get("attr" + std::to_string(attr_idx) + "_value", ctx),
          tf_attr.second);
      delegate_kernel_op_attributes.push_back(attr_name);
      delegate_kernel_op_attributes.push_back(attr_value);
      attr_idx++;
    }

    // Replace the TF op with TFD delegate kernel op.
    auto new_op = rewriter.create<tfd::DelegateKernelOp>(
        op.getLoc(), delegate_kernel_op_results, delegate_kernel_op_operands,
        delegate_kernel_op_attributes);

    // Create lowering cast ops for non-chain results.
    llvm::SmallVector<Value, 4> lowering_cast_ops_values;
    // Skip the first result. It's a chain which has no current users.
    for (int i = 1, e = new_op.getOperation()->getNumResults(); i != e; ++i) {
      Type orig_input = op.getType();
      OperationState tmp_lowering_cast_op(new_op.getLoc(),
                                          kTmpLoweringCastOpName,
                                          {new_op.getOperation()->getResult(0),
                                           new_op.getOperation()->getResult(i)},
                                          {orig_input}, {});
      Value repl_value =
          rewriter.createOperation(tmp_lowering_cast_op)->getResult(0);
      lowering_cast_ops_values.push_back(repl_value);
    }

    rewriter.replaceOp(op, lowering_cast_ops_values);
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<ReturnOp> {
 public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  // Replace std.return with hex.return. The first result is always a chain and
  // each original TF tensor result is converted to a TFD tensor.
  LogicalResult matchAndRewrite(
      ReturnOp return_op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    auto ctx = return_op.getContext();
    Value chain_output = nullptr;
    llvm::SmallVector<Value, 4> new_return_op_operands;
    new_return_op_operands.reserve(return_op.getNumOperands() + 1);
    // Convert each TF tensor operand to the corresponding TFD tensor operand.
    for (auto operand : operands) {
      auto* tmp_lowering_cast_op = operand.getDefiningOp();
      if (tmp_lowering_cast_op->getName().getStringRef() !=
          kTmpLoweringCastOpName) {
        assert(false && "unexpected producer of operand");
      }
      if (chain_output == nullptr) {
        // Get the input chain from the previous op or first block argument.
        auto* block = return_op.getOperation()->getBlock();
        if (!block->isEntryBlock()) {
          assert(false && "only supports a single block");
        }
        // Find a previous delegate_kernel op for its output chain.
        auto* prev_op = return_op.getOperation()->getPrevNode();
        while (prev_op != nullptr && !isa<tfd::DelegateKernelOp>(prev_op)) {
          prev_op = prev_op->getPrevNode();
        }
        if (prev_op != nullptr) {
          // There is another delegate kernel op before this op.
          auto prev_op_result_0 = prev_op->getResult(0);
          if (prev_op_result_0.getType() != GetChainType(ctx)) {
            assert(false &&
                   "delegate kernel must produce chain as the first result");
          }
          chain_output = prev_op_result_0;
        } else {
          // This op is the first delegate kernel op in a block.
          auto arg_0 = block->getArgument(0);
          if (arg_0.getType() != GetChainType(ctx)) {
            assert(false && "first block argument must be a chain");
          }
          chain_output = arg_0;
        }
        new_return_op_operands.push_back(chain_output);
      }
      new_return_op_operands.push_back(tmp_lowering_cast_op->getOperand(1));
    }
    // Replace the old std.return op with the new hex.return op.
    rewriter.create<tfrt::hex::ReturnOp>(return_op.getLoc(),
                                         new_return_op_operands);
    rewriter.eraseOp(return_op);

    return success();
  }
};

void TfToTfdLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());

  // Make tmp_lowering_cast_op legal for conversion. But delete them after the
  // passes.
  OperationName tmp_lowering_cast_op_name(kTmpLoweringCastOpName,
                                          &getContext());
  target.setOpAction(tmp_lowering_cast_op_name,
                     ConversionTarget::LegalizationAction::Legal);

  // target.addLegalDialect<TF::TensorFlowDialect,
  // tfd::RuntimeFallbackDialect>();
  target.addLegalDialect<tfd::RuntimeFallbackDialect>();

  target.addDynamicallyLegalOp<FuncOp>([](FuncOp function) {
    // Returns true if this function is legal, i.e. all inputs and outputs are
    // TFRT types.
    FunctionType type = function.getType();
    for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i) {
      if (type.getInput(i).isa<TensorType>()) return false;
    }
    for (unsigned i = 0, e = type.getNumResults(); i != e; ++i) {
      if (type.getResult(i).isa<TensorType>()) return false;
    }
    return true;
  });

  target.addLegalOp<mlir::ModuleTerminatorOp, mlir::ModuleOp,
                    tfrt::hex::ReturnOp>();

  OwningRewritePatternList patterns;
  patterns.insert<FuncOpSignatureConversion, TFOpConversion<TF::ReadVariableOp>,
                  TFOpConversion<TF::MatMulOp>, TFOpConversion<TF::AddV2Op>,
                  TFOpConversion<TF::ReluOp>, TFOpConversion<TF::IdentityOp>,
                  ReturnOpConversion>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();

  // Delete the tmp_lowering_cast_op's since they are illegal.
  getOperation().walk([&tmp_lowering_cast_op_name](Operation* op) {
    if (op->getName() == tmp_lowering_cast_op_name) op->erase();
  });
}

}  // namespace
}  // namespace mlir

static mlir::PassRegistration<mlir::TfToTfdLoweringPass> pass(
    "tf-to-tfd-lowering", "Lowers the TF dialect to Runtime Fallback dialect.");
