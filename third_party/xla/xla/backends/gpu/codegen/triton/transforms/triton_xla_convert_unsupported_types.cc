/* Copyright 2025 The OpenXLA Authors.
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
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLACONVERTUNSUPPORTEDTYPESPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {
class UnsupportedTypesConverter : public TypeConverter {
 public:
  UnsupportedTypesConverter() : TypeConverter() {
    // Fallback to the no conversion for all other types.
    addConversion([](Type type) -> std::optional<Type> { return type; });

    // Convert F8E8M0FNUType to i8. This is a workaround for the fact that
    // Triton doesn't support F8E8M0FNUType natively.
    addConversion([](Float8E8M0FNUType type) -> std::optional<Type> {
      return IntegerType::get(type.getContext(), 8);
    });

    // Helper conversions for the nontrivial types.
    addConversion([this](Type type) -> std::optional<Type> {
      if (auto shaped_type = dyn_cast<ShapedType>(type)) {
        Type new_type = convertType(shaped_type.getElementType());
        return shaped_type.clone(new_type);
      }
      return std::nullopt;
    });
    addConversion([this](Type type) -> std::optional<Type> {
      if (auto pointer_type = dyn_cast<triton::PointerType>(type)) {
        Type new_type = convertType(pointer_type.getPointeeType());
        return triton::PointerType::get(new_type,
                                        pointer_type.getAddressSpace());
      }
      return std::nullopt;
    });
    addConversion([this](Type type) -> std::optional<Type> {
      if (auto func_type = dyn_cast<FunctionType>(type)) {
        SmallVector<Type> new_inputs = convertTypes(func_type.getInputs());
        SmallVector<Type> new_results = convertTypes(func_type.getResults());
        if (new_inputs != func_type.getInputs() ||
            new_results != func_type.getResults()) {
          return FunctionType::get(func_type.getContext(), new_inputs,
                                   new_results);
        }
      }
      return std::nullopt;
    });
  }

  // Helper method to convert a range of types.
  SmallVector<Type> convertTypes(ArrayRef<Type> types) {
    SmallVector<Type> new_types;
    for (auto type : types) {
      new_types.push_back(convertType(type));
    }
    return new_types;
  }
};

struct RewriteF8ToI8ConversionPattern final : ConversionPattern {
  RewriteF8ToI8ConversionPattern(const TypeConverter& converter,
                                 MLIRContext* ctx)
      : ConversionPattern::ConversionPattern(
            converter, Pattern::MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (getTypeConverter()->isLegal(op)) {
      return failure();
    }

    if (!isa<ExtractOp, InsertOp, arith::BitcastOp>(op)) {
      return failure();
    }

    const TypeConverter* converter = getTypeConverter();
    SmallVector<Type> result_types;
    if (failed(converter->convertTypes(op->getResultTypes(), result_types))) {
      // Note to anyone looking for this error message: this is a "can't
      // happen". If you're seeing it, there's a bug.
      return op->emitOpError("The op is not legal but type conversion failed.");
    }
    Operation* replacement = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), operands, result_types,
        op->getAttrs(), op->getSuccessors(), /*regions=*/{});
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

class TritonXLAConvertUnsupportedTypesPass
    : public impl::TritonXLAConvertUnsupportedTypesPassBase<
          TritonXLAConvertUnsupportedTypesPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    auto* ctx = &getContext();
    auto module = getOperation();
    UnsupportedTypesConverter converter;

    ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([&](Operation* op) {
      if (auto func_op = dyn_cast<func::FuncOp>(op)) {
        return converter.isLegal(func_op.getFunctionType());
      }
      return converter.isLegal(op);
    });

    RewritePatternSet patterns(ctx);
    patterns.add<RewriteF8ToI8ConversionPattern>(converter,
                                                 patterns.getContext());
    scf::populateSCFStructuralTypeConversions(converter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAConvertUnsupportedTypesPass() {
  return std::make_unique<TritonXLAConvertUnsupportedTypesPass>();
}

}  // namespace mlir::triton::xla
