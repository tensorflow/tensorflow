//===- ConvertStandardToSPIRV.cpp - Standard to SPIR-V dialect conversion--===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements patterns to convert Standard Ops to the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

/// Convert constant operation with IndexType return to SPIR-V constant
/// operation. Since IndexType is not used within SPIR-V dialect, this needs
/// special handling to make sure the result type and the type of the value
/// attribute are consistent.
class ConstantIndexOpConversion final : public SPIRVOpLowering<ConstantOp> {
public:
  using SPIRVOpLowering<ConstantOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(ConstantOp constIndexOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!constIndexOp.getResult()->getType().isa<IndexType>()) {
      return matchFailure();
    }
    // The attribute has index type which is not directly supported in
    // SPIR-V. Get the integer value and create a new IntegerAttr.
    auto constAttr = constIndexOp.value().dyn_cast<IntegerAttr>();
    if (!constAttr) {
      return matchFailure();
    }

    // Use the bitwidth set in the value attribute to decide the result type
    // of the SPIR-V constant operation since SPIR-V does not support index
    // types.
    auto constVal = constAttr.getValue();
    auto constValType = constAttr.getType().dyn_cast<IndexType>();
    if (!constValType) {
      return matchFailure();
    }
    auto spirvConstType =
        typeConverter.convertType(constIndexOp.getResult()->getType());
    auto spirvConstVal =
        rewriter.getIntegerAttr(spirvConstType, constAttr.getInt());
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constIndexOp, spirvConstType,
                                                   spirvConstVal);
    return matchSuccess();
  }
};

/// Convert compare operation to SPIR-V dialect.
class CmpIOpConversion final : public SPIRVOpLowering<CmpIOp> {
public:
  using SPIRVOpLowering<CmpIOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CmpIOpOperandAdaptor cmpIOpOperands(operands);

    switch (cmpIOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(                                      \
        cmpIOp, cmpIOp.getResult()->getType(), cmpIOpOperands.lhs(),           \
        cmpIOpOperands.rhs());                                                 \
    return matchSuccess();

      DISPATCH(CmpIPredicate::eq, spirv::IEqualOp);
      DISPATCH(CmpIPredicate::ne, spirv::INotEqualOp);
      DISPATCH(CmpIPredicate::slt, spirv::SLessThanOp);
      DISPATCH(CmpIPredicate::sle, spirv::SLessThanEqualOp);
      DISPATCH(CmpIPredicate::sgt, spirv::SGreaterThanOp);
      DISPATCH(CmpIPredicate::sge, spirv::SGreaterThanEqualOp);

#undef DISPATCH

    default:
      break;
    }
    return matchFailure();
  }
};

/// Convert integer binary operations to SPIR-V operations. Cannot use
/// tablegen for this. If the integer operation is on variables of IndexType,
/// the type of the return value of the replacement operation differs from
/// that of the replaced operation. This is not handled in tablegen-based
/// pattern specification.
template <typename StdOp, typename SPIRVOp>
class IntegerOpConversion final : public SPIRVOpLowering<StdOp> {
public:
  using SPIRVOpLowering<StdOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(StdOp operation, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->typeConverter.convertType(operation.getResult()->getType());
    rewriter.template replaceOpWithNewOp<SPIRVOp>(
        operation, resultType, operands, ArrayRef<NamedAttribute>());
    return this->matchSuccess();
  }
};

// If 'basePtr' is the result of lowering a value of MemRefType, and 'indices'
// are the indices used to index into the original value (for load/store),
// perform the equivalent address calculation in SPIR-V.
spirv::AccessChainOp getElementPtr(OpBuilder &builder, Location loc,
                                   Value *basePtr, ArrayRef<Value *> indices,
                                   SPIRVTypeConverter &typeConverter) {
  // MemRefType is converted to a
  // spirv::StructType<spirv::ArrayType<spirv:ArrayType...>>>
  auto ptrType = basePtr->getType().cast<spirv::PointerType>();
  (void)ptrType;
  auto structType = ptrType.getPointeeType().cast<spirv::StructType>();
  (void)structType;
  assert(structType.getNumElements() == 1);
  auto indexType = typeConverter.getIndexType(builder.getContext());

  // Need to add a '0' at the beginning of the index list for accessing into the
  // struct that wraps the nested array types.
  Value *zero = spirv::ConstantOp::getZero(indexType, loc, &builder);
  SmallVector<Value *, 4> accessIndices;
  accessIndices.reserve(1 + indices.size());
  accessIndices.push_back(zero);
  accessIndices.append(indices.begin(), indices.end());
  return builder.create<spirv::AccessChainOp>(loc, basePtr, accessIndices);
}

/// Convert load -> spv.LoadOp. The operands of the replaced operation are of
/// IndexType while that of the replacement operation are of type i32. This is
/// not supported in tablegen based pattern specification.
// TODO(ravishankarm) : These could potentially be templated on the operation
// being converted, since the same logic should work for linalg.load.
class LoadOpConversion final : public SPIRVOpLowering<LoadOp> {
public:
  using SPIRVOpLowering<LoadOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(LoadOp loadOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    LoadOpOperandAdaptor loadOperands(operands);
    auto basePtr = loadOperands.memref();
    auto loadPtr = getElementPtr(rewriter, loadOp.getLoc(), basePtr,
                                 loadOperands.indices(), typeConverter);
    rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp, loadPtr,
                                               /*memory_access =*/nullptr,
                                               /*alignment =*/nullptr);
    return matchSuccess();
  }
};

/// Convert return -> spv.Return.
class ReturnToSPIRVConversion final : public SPIRVOpLowering<ReturnOp> {
public:
  using SPIRVOpLowering<ReturnOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(ReturnOp returnOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (returnOp.getNumOperands()) {
      return matchFailure();
    }
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
    return matchSuccess();
  }
};

/// Convert select -> spv.Select
class SelectOpConversion final : public SPIRVOpLowering<SelectOp> {
public:
  using SPIRVOpLowering<SelectOp>::SPIRVOpLowering;
  PatternMatchResult
  matchAndRewrite(SelectOp op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SelectOpOperandAdaptor selectOperands(operands);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, selectOperands.condition(),
                                                 selectOperands.true_value(),
                                                 selectOperands.false_value());
    return matchSuccess();
  }
};

/// Convert store -> spv.StoreOp. The operands of the replaced operation are
/// of IndexType while that of the replacement operation are of type i32. This
/// is not supported in tablegen based pattern specification.
// TODO(ravishankarm) : These could potentially be templated on the operation
// being converted, since the same logic should work for linalg.store.
class StoreOpConversion final : public SPIRVOpLowering<StoreOp> {
public:
  using SPIRVOpLowering<StoreOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(StoreOp storeOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StoreOpOperandAdaptor storeOperands(operands);
    auto value = storeOperands.value();
    auto basePtr = storeOperands.memref();
    auto storePtr = getElementPtr(rewriter, storeOp.getLoc(), basePtr,
                                  storeOperands.indices(), typeConverter);
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, storePtr, value,
                                                /*memory_access =*/nullptr,
                                                /*alignment =*/nullptr);
    return matchSuccess();
  }
};

} // namespace

namespace {
/// Import the Standard Ops to SPIR-V Patterns.
#include "StandardToSPIRV.cpp.inc"
} // namespace

namespace mlir {
void populateStandardToSPIRVPatterns(MLIRContext *context,
                                     SPIRVTypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  populateWithGenerated(context, &patterns);
  // Add the return op conversion.
  patterns
      .insert<ConstantIndexOpConversion, CmpIOpConversion,
              IntegerOpConversion<AddIOp, spirv::IAddOp>,
              IntegerOpConversion<MulIOp, spirv::IMulOp>,
              IntegerOpConversion<DivISOp, spirv::SDivOp>,
              IntegerOpConversion<RemISOp, spirv::SModOp>,
              IntegerOpConversion<SubIOp, spirv::ISubOp>, LoadOpConversion,
              ReturnToSPIRVConversion, SelectOpConversion, StoreOpConversion>(
          context, typeConverter);
}
} // namespace mlir
