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
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
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

template <typename OpType>
struct GenericOpConversionPattern final : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpType op, typename OpType::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Operation* replacement = rewriter.clone(*op);
    replacement->setOperands(adaptor.getOperands());
    const TypeConverter* converter = this->getTypeConverter();
    for (auto result : replacement->getResults()) {
      result.setType(converter->convertType(result.getType()));
    }
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
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([&](Float8E8M0FNUType type) {
      return IntegerType::get(type.getContext(), 8);
    });
    converter.addConversion([&](ShapedType type) {
      return type.clone(converter.convertType(type.getElementType()));
    });

    converter.addConversion([&](triton::PointerType type) {
      return triton::PointerType::get(
          converter.convertType(type.getPointeeType()), type.getAddressSpace());
    });
    converter.addConversion([&](FunctionType type) -> Type {
      SmallVector<Type> new_inputs, new_results;
      if (failed(converter.convertTypes(type.getInputs(), new_inputs)) ||
          failed(converter.convertTypes(type.getResults(), new_results))) {
        return nullptr;
      }
      return type.clone(new_inputs, new_results);
    });

    auto* ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) { return converter.isLegal(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<GenericOpConversionPattern<ExtractOp>,
                 GenericOpConversionPattern<InsertOp>,
                 GenericOpConversionPattern<arith::BitcastOp>>(converter, ctx);
    scf::populateSCFStructuralTypeConversions(converter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAConvertUnsupportedTypesPass() {
  return std::make_unique<TritonXLAConvertUnsupportedTypesPass>();
}

}  // namespace mlir::triton::xla
