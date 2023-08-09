/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// The TF dialect uses some TF types that are illegal in the MHLO dialect and
// some generic types that are legal in MHLO. This pass legalizes TF types into
// types that are legal in MHLO. For example, TF::Qint8Type is converted to i8.
// Rewrites here should run before TF to MHLO op legalizations are run.

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/lib/monitoring/counter.h"

namespace mlir {
namespace stablehlo {
namespace {

#define GEN_PASS_DEF_CONVERTTFQUANTTYPES
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

// TODO: b/290366702 - Temporarily added metrics for debugging.
auto *mlir_tf_quant_op_count = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/tf_quant_op_count" /*metric_name*/,
    "Counts the number of ops that has qint types" /*metric description*/,
    "op_name" /*metric label*/);

bool IsIllegalElementType(Type type) {
  return type
      .isa<mlir::TF::Qint8Type, mlir::TF::Qint16Type, mlir::TF::Qint32Type,
           mlir::TF::Quint8Type, mlir::TF::Quint16Type>();
}

Type ToLegalElementType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<mlir::TF::Qint8Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 8);
      })
      .Case<mlir::TF::Qint16Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 16);
      })
      .Case<mlir::TF::Qint32Type>([&type](Type) {
        return mlir::IntegerType::get(type.getContext(), 32);
      })
      .Case<mlir::TF::Quint8Type>([&type](Type) {
        return mlir::IntegerType::get(
            type.getContext(), 8,
            mlir::IntegerType::SignednessSemantics::Unsigned);
      })
      .Case<mlir::TF::Quint16Type>([&type](Type) {
        return mlir::IntegerType::get(
            type.getContext(), 16,
            mlir::IntegerType::SignednessSemantics::Unsigned);
      })
      .Default([&type](Type) { return type; });
}

bool IsIllegalType(Type type) {
  return IsIllegalElementType(getElementTypeOrSelf(type));
}

Type ToLegalType(Type type) {
  if (IsIllegalElementType(type)) return ToLegalElementType(type);
  if (auto shaped = type.dyn_cast<ShapedType>()) {
    Type elem = shaped.getElementType();
    if (IsIllegalType(elem)) return shaped.clone(ToLegalType(elem));
  }
  return type;
}

bool IsUniformQuantizedOp(Operation *op) {
  return llvm::isa<
      // clang-format off
      // go/keep-sorted start
      TF::UniformDequantizeOp,
      TF::UniformQuantizeOp,
      TF::UniformQuantizedAddOp,
      TF::UniformQuantizedClipByValueOp,
      TF::UniformQuantizedConvolutionHybridOp,
      TF::UniformQuantizedConvolutionOp,
      TF::UniformQuantizedDotHybridOp,
      TF::UniformQuantizedDotOp,
      TF::UniformRequantizeOp
      // go/keep-sorted end
      // clang-format on
      >(op);
}

bool IsUniformQuantizedOpLegal(Operation *op) {
  // Check if an op result value is consumed by qint -> int TF Cast OP.
  auto IsQintValueQintToInCast = [](Value v) {
    if (!IsIllegalType(v.getType())) {
      return true;
    }
    if (v.getUsers().empty() || !llvm::isa<TF::CastOp>(*v.getUsers().begin())) {
      return false;
    }
    auto cast_op = llvm::dyn_cast<TF::CastOp>(*v.getUsers().begin());
    return v.getType() == cast_op.getX().getType() &&
           ToLegalType(v.getType()) == cast_op.getY().getType();
  };
  // Check if an op operand value is defined by int -> qint TF Cast OP.
  auto IsQintValueDefinedByIntToQinCast = [](Value v) {
    if (!IsIllegalType(v.getType())) {
      return true;
    }
    if (!v.getDefiningOp() || !llvm::isa<TF::CastOp>(v.getDefiningOp())) {
      return false;
    }
    auto cast_op = llvm::dyn_cast<TF::CastOp>(v.getDefiningOp());
    return v.getType() == cast_op.getY().getType() &&
           ToLegalType(v.getType()) == cast_op.getX().getType();
  };
  // UniformQuantized Ops are considered legal if its qint operands and
  // results are connected to TF CastOp.
  return op && llvm::all_of(op->getResults(), IsQintValueQintToInCast) &&
         llvm::all_of(op->getOperands(), IsQintValueDefinedByIntToQinCast);
}

bool IsCastOpLegal(TF::CastOp cast_op) {
  // Consider qint <-> qint casts illegal.
  if (IsIllegalType(cast_op.getSrcT()) && IsIllegalType(cast_op.getDstT())) {
    return false;
  }
  // Consider CastOp illegal if either of its Src/Dst type is qint and is
  // connected to a non-UQ op.
  if (IsIllegalType(cast_op.getSrcT()) &&
      !(cast_op.getX().getDefiningOp() &&
        IsUniformQuantizedOp(cast_op.getX().getDefiningOp()))) {
    return false;
  }
  if (IsIllegalType(cast_op.getDstT()) &&
      !IsUniformQuantizedOp(*cast_op.getY().getUsers().begin())) {
    return false;
  }
  return true;
}

class TFQuantTypeConverter : public TypeConverter {
 public:
  TFQuantTypeConverter() {
    addConversion([](Type type) -> Type {
      return IsIllegalType(type) ? ToLegalType(type) : type;
    });
  }
};

// An Op is illegal iff it is non-UQ op and it contains qint types.
class TFQuantTypeConversionTarget : public ConversionTarget {
 public:
  explicit TFQuantTypeConversionTarget(MLIRContext &ctx,
                                       TFQuantTypeConverter &converter)
      : ConversionTarget(ctx), converter_(converter) {
    markUnknownOpDynamicallyLegal([this](Operation *op) {
      // Consider UQ op legal if it has a CastOp next to the qint input/output.
      if (IsUniformQuantizedOp(op)) {
        return IsUniformQuantizedOpLegal(op);
      } else if (auto cast_op = llvm::dyn_cast<TF::CastOp>(op)) {
        return IsCastOpLegal(cast_op);
      }
      // The FuncOp type can contain types that the op's operand and result
      // types do not contain.
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        if (!converter_.isSignatureLegal(func.getFunctionType())) return false;
      }
      return converter_.isLegal(op);
    });
  }

 private:
  TFQuantTypeConverter &converter_;
};

class TFQuantTypePattern : public ConversionPattern {
 public:
  TFQuantTypePattern(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // This pattern only handle non-UQ ops.
    if (IsUniformQuantizedOp(op)) {
      return failure();
    }

    // Update the results.
    llvm::SmallVector<Type, 4> new_results;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                new_results)))
      return failure();

    // Update the regions. The dialect conversion framework wants new regions to
    // be created and updated, rather than updating the old op. Thus we use an
    // OperationState so we can add regions to the new up.
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_results, op->getAttrs(), op->getSuccessors());
    for (Region &region : op->getRegions()) {
      Region &new_region = *state.addRegion();
      rewriter.inlineRegionBefore(region, new_region, new_region.begin());
      if (failed(rewriter.convertRegionTypes(&new_region, *getTypeConverter())))
        return failure();
    }
    rewriter.replaceOp(op, rewriter.create(state)->getResults());

    // TODO: b/290366702 - Temporarily added metrics for debugging.
    mlir_tf_quant_op_count->GetCell(std::string(op->getName().getStringRef()))
        ->IncrementBy(1);
    return success();
  }
};

// This pattern adds qint <-> int Cast to all qint operands and results for UQ
// ops.
class TFUniformQuantizedOpsPattern : public ConversionPattern {
 public:
  TFUniformQuantizedOpsPattern(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // This pattern only handle UQ ops.
    if (!IsUniformQuantizedOp(op)) {
      return failure();
    }

    // Add CastOp int->qint before input operands if its original type is qint.
    llvm::SmallVector<Value, 4> new_operands;
    for (int i = 0; i < operands.size(); ++i) {
      Type orig_op_type = op->getOperandTypes()[i];
      if (IsIllegalType(orig_op_type)) {
        new_operands.push_back(rewriter.create<TF::CastOp>(
            op->getLoc(), orig_op_type, operands[i]));
      } else {
        new_operands.push_back(operands[i]);
      }
    }

    OperationState state(op->getLoc(), op->getName().getStringRef(),
                         new_operands, op->getResultTypes(), op->getAttrs(),
                         op->getSuccessors());
    llvm::SmallVector<Value, 4> new_results =
        rewriter.create(state)->getResults();

    // Add qint->int CastOp after output result if its original type is qint.
    for (int i = 0; i < new_results.size(); ++i) {
      Value &result = new_results[i];
      if (IsIllegalType(result.getType())) {
        result = rewriter.create<TF::CastOp>(
            op->getLoc(), getTypeConverter()->convertType(result.getType()),
            result);
      }
    }
    rewriter.replaceOp(op, new_results);
    return success();
  }
};

struct ConvertTFQuantTypes
    : public impl::ConvertTFQuantTypesBase<ConvertTFQuantTypes> {
  void runOnOperation() override;
};

void ConvertTFQuantTypes::runOnOperation() {
  TFQuantTypeConverter converter;
  RewritePatternSet patterns(&getContext());
  patterns.add<TFQuantTypePattern, TFUniformQuantizedOpsPattern>(&getContext(),
                                                                 converter);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  TFQuantTypeConversionTarget target(getContext(), converter);
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertTFQuantTypesPass() {
  return std::make_unique<ConvertTFQuantTypes>();
}

}  // namespace stablehlo
}  // namespace mlir
