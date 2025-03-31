#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable(),
                       triton::GlobalMemory::get());
  if (getIsVolatile())
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"

// enum attribute definitions
#include "triton/Dialect/Triton/IR/OpsEnums.cpp.inc"

#include "TritonCanonicalize.inc"

namespace mlir {
namespace triton {

//-- LoadOp --
void LoadOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                   CacheModifier cache, EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, /*mask=*/{}, /*other=*/{},
                /*boundaryCheck=*/ArrayRef<int32_t>{}, /*padding=*/std::nullopt,
                cache, evict, isVolatile);
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                   ArrayRef<int32_t> boundaryCheck,
                   std::optional<PaddingOption> padding, CacheModifier cache,
                   EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, /*mask=*/{}, /*other=*/{}, boundaryCheck,
                padding, cache, evict, isVolatile);
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                   Value mask, CacheModifier cache, EvictionPolicy evict,
                   bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, /*other=*/{},
                /*boundaryCheck=*/ArrayRef<int32_t>{},
                /*padding=*/std::nullopt, cache, evict, isVolatile);
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                   Value mask, Value other, CacheModifier cache,
                   EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, other,
                /*boundaryCheck=*/ArrayRef<int32_t>{},
                /*padding=*/std::nullopt, cache, evict, isVolatile);
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                   Value mask, Value other, ArrayRef<int32_t> boundaryCheck,
                   std::optional<PaddingOption> padding, CacheModifier cache,
                   EvictionPolicy evict, bool isVolatile) {
  auto paddingAttr =
      padding.has_value()
          ? PaddingOptionAttr::get(builder.getContext(), padding.value())
          : PaddingOptionAttr();
  LoadOp::build(builder, state, ptr, mask, other,
                builder.getDenseI32ArrayAttr(boundaryCheck), paddingAttr, cache,
                evict, isVolatile);
}

// load(ptr, splat(1), ...)        -> load(ptr, ...)
// load(ptr, splat(0), other, ...) -> other
struct CanonicalizeMaskedLoadPattern : public OpRewritePattern<LoadOp> {
  CanonicalizeMaskedLoadPattern(MLIRContext *context)
      : OpRewritePattern<LoadOp>(context, 1) {}

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto mask = loadOp.getMask();
    if (!mask)
      return failure();

    auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
    if (!constantMask)
      return failure();

    auto splatMask = mlir::dyn_cast<SplatElementsAttr>(constantMask.getValue());
    if (!splatMask)
      return failure();

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(1)
      rewriter.replaceOpWithNewOp<LoadOp>(
          loadOp, loadOp.getType(), loadOp.getPtr(), Value(), Value(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    } else {
      // mask = splat(0)

      // If there's no "other", the value is "undef".  Perhaps we want to
      // optimize it in the future.x
      auto otherVal = loadOp.getOther();
      if (!otherVal)
        return failure();
      rewriter.replaceOp(loadOp, otherVal);
    }
    return success();
  }
};

void LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<CanonicalizeMaskedLoadPattern>(context);
}

//-- StoreOp --
void StoreOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                    Value value, CacheModifier cache, EvictionPolicy evict) {
  return StoreOp::build(builder, state, ptr, value, /*mask=*/{},
                        /*boundaryCheck=*/{}, cache, evict);
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                    Value value, Value mask, CacheModifier cache,
                    EvictionPolicy evict) {
  return StoreOp::build(builder, state, ptr, value, mask, /*boundaryCheck=*/{},
                        cache, evict);
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                    Value value, ArrayRef<int32_t> boundaryCheck,
                    CacheModifier cache, EvictionPolicy evict) {
  return StoreOp::build(builder, state, ptr, value, /*mask=*/{},
                        builder.getDenseI32ArrayAttr(boundaryCheck), cache,
                        evict);
}

// store(ptr, value, splat(1), ...) -> store(ptr, value, ...)
// store(ptr, value, splat(0), ...) -> [none]
struct CanonicalizeMaskedStorePattern : public OpRewritePattern<StoreOp> {
  CanonicalizeMaskedStorePattern(MLIRContext *context)
      : OpRewritePattern<StoreOp>(context, 1) {}

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto mask = storeOp.getMask();
    if (!mask)
      return failure();

    auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
    if (!constantMask)
      return failure();

    auto splatMask = mlir::dyn_cast<SplatElementsAttr>(constantMask.getValue());
    if (!splatMask)
      return failure();

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(1)
      rewriter.replaceOpWithNewOp<StoreOp>(
          storeOp, storeOp.getPtr(), storeOp.getValue(), storeOp.getCache(),
          storeOp.getEvict());
    } else {
      // mask = splat(0)
      rewriter.eraseOp(storeOp);
    }
    return success();
  }
};

void StoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<CanonicalizeMaskedStorePattern>(context);
}

//-- TransOp --
OpFoldResult TransOp::fold(FoldAdaptor adaptor) {
  // transpose(x, order=[0, 1, ...]) -> x
  if (isIota(getOrder())) {
    // If the source and result types are the same, we can return the source
    // If their layout is different (even if structurally equivalent), we need
    // to insert a convert_layout in between as otherwise ::fold complains
    // We do this in CanonicalizeConvertFromTranspose
    if (getSrc().getType() == getType()) {
      return getSrc();
    }
  }

  // transpose(transpose(x)) -> transpose(x)
  if (auto innerTrans = getSrc().getDefiningOp<TransOp>()) {
    setOrder(applyPermutation(innerTrans.getOrder(), getOrder()));
    setOperand(innerTrans.getSrc());
    return getResult();
  }

  // Eliminate splat constant transpose ops.
  if (auto attr =
          llvm::dyn_cast_if_present<SplatElementsAttr>(adaptor.getSrc()))
    return attr.reshape(getType());

  return {};
}

LogicalResult TransOp::verify() {
  auto order = getOrder();
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  if (order.size() != srcTy.getShape().size()) {
    return emitError("order must have the same size as the source tensor");
  }
  if (!isPermutationOfIota(order)) {
    return emitError("order must be a permutation of 0..n-1");
  }
  SmallVector<int64_t> retShape = applyPermutation(srcTy.getShape(), order);
  if (retShape != getType().getShape()) {
    return emitError(
        "result shape must match the permutation of the source shape");
  }
  return success();
}

LogicalResult TransOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    TransOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {

  // type is the same as the input
  auto argTy = cast<RankedTensorType>(adaptor.getSrc().getType());
  auto shape = argTy.getShape();
  auto order = adaptor.getOrder();
  SmallVector<int64_t> retShape = applyPermutation(shape, order);

  auto retEltTy = argTy.getElementType();
  Attribute argEncoding = argTy.getEncoding();
  Attribute retEncoding;
  if (argEncoding) {
    Dialect &dialect = argEncoding.getDialect();
    auto inferLayoutInterface = cast<DialectInferLayoutInterface>(&dialect);
    if (inferLayoutInterface
            ->inferTransOpEncoding(argEncoding, shape, order, retEncoding)
            .failed()) {
      return failure();
    }
  }
  inferredReturnTypes.push_back(
      RankedTensorType::get(retShape, retEltTy, retEncoding));
  return success();
}

//-- DotOp --
LogicalResult
DotOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = cast<RankedTensorType>(operands[0].getType()).getEncoding();
  auto bEnc = cast<RankedTensorType>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc && retEnc);
    Dialect &dialect = retEnc.getDialect();
    auto interface = cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return failure();
  }
  return success();
}

LogicalResult DotOp::verify() {
  auto aTy = getA().getType();
  auto bTy = getB().getType();
  if (aTy.getElementType().getIntOrFloatBitWidth() !=
      bTy.getElementType().getIntOrFloatBitWidth())
    return emitError(
        "element types of operands A and B must have same bit width");
  auto aEncoding = aTy.getEncoding();
  auto bEncoding = bTy.getEncoding();
  if (!aEncoding && !bEncoding)
    return success();
  // Verify that the encodings are valid.
  if (!aEncoding || !bEncoding)
    return emitError("mismatching encoding between A and B operands");
  auto accTy = getC().getType();
  auto retEnc = accTy.getEncoding();
  if (!retEnc)
    return emitError("miss encoding of C operand");
  Dialect &dialect = retEnc.getDialect();
  auto interface = cast<DialectInferLayoutInterface>(&dialect);
  return interface->verifyDotOpEncodingCompatibility(getOperation(), aEncoding,
                                                     bEncoding);
}

bool DotOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

//-- DotScaledOp --
bool DotScaledOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  auto aKdim = aShape[aShape.size() - 1];
  auto bKdim = bShape[aShape.size() - 2];
  if (this->getAElemType() == ScaleDotElemType::E2M1)
    aKdim *= 2;
  if (this->getBElemType() == ScaleDotElemType::E2M1)
    bKdim *= 2;

  return aKdim == bKdim;
}

//-- MakeRangeOp --
OpFoldResult MakeRangeOp::fold(FoldAdaptor adaptor) {
  // make_range(start, start + 1) -> constant(start)
  if (adaptor.getStart() + 1 == adaptor.getEnd()) {
    auto shapedType = cast<ShapedType>(getType());
    return SplatElementsAttr::get(shapedType, adaptor.getStartAttr());
  }
  return {};
}

LogicalResult MakeRangeOp::verify() {
  int64_t start = getStartAttr().getInt();
  int64_t end = getEndAttr().getInt();
  if (start > end) {
    return this->emitOpError() << "start must be less than or equal to end";
  }
  auto ty = getType();
  if (ty.getShape().size() != 1) {
    return this->emitOpError() << "return type must be a 1D tensor";
  }
  if (end - start != ty.getShape()[0]) {
    return this->emitOpError()
           << "number of elements in returned tensor, " << ty.getShape()[0]
           << ", must match size of range [" << start << ", " << end
           << "), which has " << end - start << " elements";
  }
  if (!ty.getElementType().isInteger(32)) {
    return this->emitOpError() << "returned tensor must have i32 elements";
  }
  return success();
}

//-- ReduceOp --
static LogicalResult
inferReduceReturnShape(RankedTensorType argTy, Type retEltTy, int axis,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  auto retShape = argTy.getShape().vec();
  retShape.erase(retShape.begin() + axis);
  if (retShape.empty()) {
    // 0d-tensor -> scalar
    inferredReturnTypes.push_back(retEltTy);
  } else {
    // nd-tensor where n >= 1
    // infer encoding
    Attribute argEncoding = argTy.getEncoding();
    Attribute retEncoding;
    if (argEncoding) {
      Dialect &dialect = argEncoding.getDialect();
      auto inferLayoutInterface = cast<DialectInferLayoutInterface>(&dialect);
      if (inferLayoutInterface
              ->inferReduceOpEncoding(argEncoding, axis, retEncoding)
              .failed()) {
        llvm::report_fatal_error("failed to infer layout for ReduceOp");
        return failure();
      }
    }
    // create type
    inferredReturnTypes.push_back(
        RankedTensorType::get(retShape, retEltTy, retEncoding));
  }
  return success();
}

void ReduceOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange operands, int axis) {
  SmallVector<Type> inferredReturnTypes;
  for (unsigned i = 0; i < operands.size(); ++i) {
    auto argTy = cast<RankedTensorType>(operands[i].getType());
    auto retEltTy = argTy.getElementType();
    (void)inferReduceReturnShape(argTy, retEltTy, axis, inferredReturnTypes);
  }

  ReduceOp::build(builder, state, inferredReturnTypes, operands, axis);
}

LogicalResult ReduceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  Properties *prop = properties.as<Properties *>();
  int axis = prop->axis.getInt();
  for (auto arg : operands) {
    auto argTy = cast<RankedTensorType>(arg.getType());
    auto retEltTy = argTy.getElementType();
    if (inferReduceReturnShape(argTy, retEltTy, axis, inferredReturnTypes)
            .failed()) {
      return failure();
    }
  }
  return success();
}

// Helpers for Reductions and Scans
template <class Op> LogicalResult verifyReduceScan(Op &op) {
  if (op.getOperands().empty()) {
    return op.emitOpError() << "must have at least 1 operand";
  }
  if (op.getNumOperands() != op.getNumResults()) {
    return op.emitOpError() << "must have the same number of inputs as outputs";
  }

  auto getElementType = [](Type ty) {
    if (auto tensorType = dyn_cast<RankedTensorType>(ty)) {
      return tensorType.getElementType();
    }
    return ty;
  };

  for (auto [opElemTy, resTy] :
       llvm::zip(op.getElementTypes(), op.getResultTypes())) {
    if (opElemTy != getElementType(resTy)) {
      return op.emitOpError() << "operand types and result types must agree";
    }
  }
  return success();
}

template <class ReturnOp, class Op>
static LogicalResult verifyRegionsImpl(Op &op) {
  auto argElementTypes = op.getElementTypes();
  const auto &operands = op.getOperands();
  const auto numArgs = 2 * operands.size();
  auto &block = *op.getBody();
  if (block.getNumArguments() != numArgs) {
    return op.emitOpError() << "nested block must take " << numArgs
                            << " arguments, but given block with "
                            << block.getNumArguments() << " arguments";
  }
  unsigned i = 0;
  const auto &blockArgTypes = block.getArgumentTypes();
  for (unsigned i = 0; i < numArgs; ++i) {
    const auto &blockArgTy = blockArgTypes[i];
    const auto &argElemTy = argElementTypes[i % operands.size()];
    if (blockArgTy != argElemTy) {
      return op.emitOpError()
             << "type mismatch on combine operation. Expected argument " << i
             << " to have type " << argElemTy << " but got " << blockArgTy;
    }
  }

  auto terminator = dyn_cast<ReturnOp>(block.getTerminator());
  if (!terminator) {
    return op.emitOpError()
           << "combine operation must be terminated "
           << "with a ReduceReturnOp but got " << block.getTerminator();
  }
  const auto &combineResults = terminator->getOperands();
  if (combineResults.size() != operands.size()) {
    return op.emitOpError()
           << "expected combine operation to return " << operands.size()
           << " values but got " << combineResults.size();
  }
  for (unsigned i = 0; i < combineResults.size(); ++i) {
    const auto &resultTy = combineResults[i].getType();
    const auto &argElemTy = argElementTypes[i];
    if (resultTy != argElemTy) {
      return op.emitOpError()
             << "type mismatch on combine operation. Expected argument " << i
             << " to have type " << argElemTy << " but got " << resultTy;
    }
  }
  return success();
}

static llvm::SmallVector<RankedTensorType>
getInputTypesImpl(const Operation::operand_range &operands) {
  llvm::SmallVector<RankedTensorType> srcTys;
  srcTys.reserve(operands.size());
  for (const auto &ty : operands.getTypes()) {
    srcTys.push_back(cast<RankedTensorType>(ty));
  }
  return srcTys;
}

static llvm::SmallVector<Type>
getElementTypesImpl(const Operation::operand_range &operands) {
  llvm::SmallVector<Type> srcElemTys;
  srcElemTys.reserve(operands.size());
  for (const auto &op : operands) {
    srcElemTys.push_back(cast<RankedTensorType>(op.getType()).getElementType());
  }
  return srcElemTys;
}

LogicalResult ReduceOp::verify() { return verifyReduceScan(*this); }

LogicalResult ReduceOp::verifyRegions() {
  return verifyRegionsImpl<ReduceReturnOp>(*this);
}

llvm::SmallVector<RankedTensorType> ReduceOp::getInputTypes() {
  return getInputTypesImpl(this->getOperands());
}

llvm::SmallVector<Type> ReduceOp::getElementTypes() {
  return getElementTypesImpl(this->getOperands());
}

::mlir::Operation *ReduceOp::getSingleCombiner() {
  if (getNumOperands() != 1 || getNumResults() != 1)
    return nullptr;
  Block *block = &(*getCombineOp().begin());
  Operation *yield = block->getTerminator();
  Operation *reduceOp = yield->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2 ||
      reduceOp->getNumResults() != 1)
    return nullptr;
  if (reduceOp->getOperand(0) != block->getArgument(0) ||
      reduceOp->getOperand(1) != block->getArgument(1))
    return nullptr;

  return reduceOp;
}

unsigned ReduceOp::getNumOperands() { return this->getOperands().size(); }

//-- ScanOp --
void ScanOp::build(OpBuilder &builder, OperationState &state,
                   ValueRange operands, int axis, bool reverse) {
  SmallVector<Type> inferredReturnTypes;
  for (auto arg : operands)
    inferredReturnTypes.push_back(arg.getType());
  ScanOp::build(builder, state, inferredReturnTypes, operands, axis, reverse);
}

LogicalResult
ScanOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  for (auto arg : operands)
    inferredReturnTypes.push_back(arg.getType());
  return success();
}

LogicalResult ScanOp::verify() { return verifyReduceScan(*this); }

LogicalResult ScanOp::verifyRegions() {
  return verifyRegionsImpl<ScanReturnOp>(*this);
}

llvm::SmallVector<RankedTensorType> ScanOp::getInputTypes() {
  return getInputTypesImpl(this->getOperands());
}

llvm::SmallVector<Type> ScanOp::getElementTypes() {
  return getElementTypesImpl(this->getOperands());
}

unsigned ScanOp::getNumOperands() { return this->getOperands().size(); }

//-- SplatOp --
OpFoldResult SplatOp::fold(FoldAdaptor adaptor) {
  auto value = adaptor.getSrc();
  if (!value)
    return {};
  if (!isa<FloatAttr, IntegerAttr>(value))
    return {};
  auto shapedType = cast<ShapedType>(getType());
  auto ret = SplatElementsAttr::get(shapedType, ArrayRef<Attribute>(value));
  return ret;
}

//-- ExpandDimsOp --
LogicalResult ExpandDimsOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // infer shape
  auto arg = operands[0];
  auto argTy = cast<RankedTensorType>(arg.getType());
  auto retShape = argTy.getShape().vec();
  Properties *prop = properties.as<Properties *>();
  int axis = prop->axis.getInt();
  retShape.insert(retShape.begin() + axis, 1);
  // infer encoding
  Attribute argEncoding = argTy.getEncoding();
  Attribute retEncoding;
  if (argEncoding) {
    Dialect &dialect = argEncoding.getDialect();
    auto inferLayoutInterface = cast<DialectInferLayoutInterface>(&dialect);
    if (inferLayoutInterface
            ->inferExpandDimsOpEncoding(argEncoding, axis, retEncoding, loc)
            .failed())
      return emitOptionalError(loc, "failed to infer layout for ExpandDimsOp");
  }
  // create type
  auto argEltTy = argTy.getElementType();
  inferredReturnTypes.push_back(
      RankedTensorType::get(retShape, argEltTy, retEncoding));
  return success();
}

LogicalResult ExpandDimsOp::canonicalize(ExpandDimsOp op,
                                         PatternRewriter &rewriter) {
  auto definingOp = op.getSrc().getDefiningOp();
  if (!definingOp) {
    return failure();
  }
  // expand_dims(splat) -> splat
  if (auto splat = dyn_cast<SplatOp>(definingOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), splat.getSrc());
    return success();
  }
  // expand_dims(broadcast(x)) -> broadcast(expand_dims(x))
  //
  // On its own this doesn't do much, but consider
  //    broadcast(expand_dims(broadcast))
  // -> broadcast(broadcast(expand_dims))
  // -> broadcast(expand_dims)
  if (auto broadcast = dyn_cast<BroadcastOp>(definingOp)) {
    auto src = broadcast.getSrc();
    auto srcTy = src.getType();
    SmallVector<int64_t> newExpandShape(srcTy.getShape());
    newExpandShape.insert(newExpandShape.begin() + op.getAxis(), 1);

    // Infer the encoding of the new expand op, if encodings are present.
    Attribute newExpandEnc;
    if (auto srcEnc = srcTy.getEncoding()) {
      if (cast<DialectInferLayoutInterface>(&srcEnc.getDialect())
              ->inferExpandDimsOpEncoding(srcEnc, op.getAxis(), newExpandEnc,
                                          op.getLoc())
              .failed()) {
        return emitOptionalError(op.getLoc(),
                                 "failed to infer layout for ExpandDimsOp");
      }
    }

    auto newExpandTy = RankedTensorType::get(
        newExpandShape, srcTy.getElementType(), newExpandEnc);
    auto newExpand = rewriter.create<ExpandDimsOp>(op.getLoc(), newExpandTy,
                                                   src, op.getAxis());
    auto newBroadcast = rewriter.create<BroadcastOp>(
        broadcast.getLoc(), op.getType(), newExpand.getResult());
    rewriter.replaceOp(op, {newBroadcast.getResult()});
    return success();
  }

  return failure();
}

template <typename ViewLikeOp>
static OpFoldResult foldViewLikeOp(ViewLikeOp op, Attribute value) {
  if (!value)
    return {};

  auto shapedType = cast<ShapedType>(op.getType());
  if (auto denseElemsAttr = dyn_cast<DenseElementsAttr>(value)) {
    if (denseElemsAttr.isSplat()) {
      return denseElemsAttr.resizeSplat(shapedType);
    } else {
      return denseElemsAttr.reshape(shapedType);
    }
  }
  return {};
}

OpFoldResult ExpandDimsOp::fold(FoldAdaptor adaptor) {
  return foldViewLikeOp(*this, adaptor.getSrc());
}

//-- ReshapeOp --

void ReshapeOp::build(OpBuilder &builder, OperationState &state,
                      ArrayRef<int64_t> shape,
                      TypedValue<RankedTensorType> src) {
  auto srcTy = src.getType();
  auto srcEnc = srcTy.getEncoding();
  Attribute dstEnc;
  if (srcEnc) {
    auto result = cast<DialectInferLayoutInterface>(&srcEnc.getDialect())
                      ->inferReshapeOpEncoding(srcTy.getShape(), srcEnc, shape,
                                               dstEnc, state.location);
    assert(succeeded(result));
  }
  auto dstTy = RankedTensorType::get(shape, srcTy.getElementType(), dstEnc);
  build(builder, state, dstTy, src);
}

LogicalResult ReshapeOp::canonicalize(ReshapeOp op, PatternRewriter &rewriter) {
  if (op.getEfficientLayout())
    return failure();

  auto definingOp = op.getSrc().getDefiningOp();
  if (!definingOp) {
    return failure();
  }

  // reshape(reshape) -> reshape
  if (auto parentReshape = dyn_cast<ReshapeOp>(definingOp)) {
    // Allow reorder if either reshape allowed it
    const bool allowReorder =
        (op.getAllowReorder() || parentReshape.getAllowReorder());
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(),
                                           parentReshape.getSrc(), allowReorder,
                                           op.getEfficientLayout());
    return success();
  }

  // reshape(splat) -> splat
  if (auto splat = dyn_cast<SplatOp>(definingOp)) {
    rewriter.replaceOpWithNewOp<SplatOp>(op, op.getType(), splat.getSrc());
    return success();
  }

  return failure();
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  if (getType() == getSrc().getType() && !getAllowReorder()) {
    // no-op
    return getSrc();
  }

  return foldViewLikeOp(*this, adaptor.getSrc());
}

LogicalResult ReshapeOp::verify() {
  auto dstTy = getType();
  auto srcTy = getSrc().getType();
  if (getType().getNumElements() != srcTy.getNumElements()) {
    return emitError(
        "number of src and dst elements of reshape must be the same");
  }

  Attribute srcEnc = srcTy.getEncoding();
  Attribute dstEnc = dstTy.getEncoding();
  if (!!srcEnc != !!dstEnc) {
    return emitError("Op requires that either (a) src and dst both have "
                     "encodings, or (b) neither does.");
  }

  if (!srcEnc || getAllowReorder()) {
    return success();
  }

  // Check that we can infer the dst encoding from the src encoding
  // and that the inferred dst encoding is the same as the given dst encoding
  Attribute inferredDstEnc;
  auto result =
      cast<DialectInferLayoutInterface>(&srcEnc.getDialect())
          ->inferReshapeOpEncoding(srcTy.getShape(), srcEnc, dstTy.getShape(),
                                   inferredDstEnc, getLoc());
  assert(succeeded(result));
  return cast<DialectInferLayoutInterface>(&srcEnc.getDialect())
      ->verifyLayoutsAreEqual(dstTy.getShape(), inferredDstEnc, dstEnc,
                              getLoc());
}

//-- FpToFpOp --

// Fold FpToFpOp when the input operand is a constant zero.
OpFoldResult FpToFpOp::fold(FoldAdaptor adaptor) {
  auto srcVal = getSrc();
  auto dstTy = getType();
  // Fold trivial cast
  if (srcVal.getType() == dstTy) {
    return srcVal;
  }

  auto resElemType = cast<FloatType>(getElementTypeOrSelf(getType()));
  const llvm::fltSemantics &semantic = resElemType.getFloatSemantics();

  if (matchPattern(srcVal, m_PosZeroFloat())) {
    llvm::APFloat posZero =
        llvm::APFloat::getZero(semantic, /*negative=*/false);
    if (auto tensorTy = dyn_cast<RankedTensorType>(dstTy))
      return DenseElementsAttr::get(tensorTy, posZero);
    return Builder(getContext()).getFloatAttr(resElemType, posZero);
  }

  if (matchPattern(srcVal, m_NegZeroFloat())) {
    llvm::APFloat negZero = llvm::APFloat::getZero(semantic, /*negative=*/true);
    if (auto tensorTy = dyn_cast<RankedTensorType>(dstTy))
      return DenseElementsAttr::get(tensorTy, negZero);
    return Builder(getContext()).getFloatAttr(resElemType, negZero);
  }

  return {};
}

LogicalResult FpToFpOp::verify() {
  auto dstType = getType();
  auto srcType = getSrc().getType();
  if (auto dstTensorType = dyn_cast<RankedTensorType>(dstType))
    dstType = dstTensorType.getElementType();
  if (auto srcTensorType = dyn_cast<RankedTensorType>(srcType))
    srcType = srcTensorType.getElementType();
  if ((dstType.getIntOrFloatBitWidth() < srcType.getIntOrFloatBitWidth()) &&
      (!getRounding().has_value())) {
    return emitError("Rounding mode is required for FP downcast");
  }
  return success();
}

//-- BitcastOp --
LogicalResult BitcastOp::verify() {
  // Bitcast only allows conversion between types with the same bit width.
  Type dstType = getType();
  Type srcType = getSrc().getType();
  // Strip tensor shapes; SameOperandsAndResultShape guarantees shapes match.
  if (auto dstTensorType = dyn_cast<RankedTensorType>(dstType))
    dstType = dstTensorType.getElementType();
  if (auto srcTensorType = dyn_cast<RankedTensorType>(srcType))
    srcType = srcTensorType.getElementType();
  bool dstIsPtr = isa<triton::PointerType>(dstType);
  bool srcIsPtr = isa<triton::PointerType>(srcType);
  if (dstIsPtr || srcIsPtr) {
    // Bitcast supports pointer-to-pointer conversions but not
    // pointer-to-scalar.
    if (dstIsPtr && srcIsPtr) {
      if (triton::getAddressSpace(dstType) != triton::getAddressSpace(srcType))
        return emitError(
            "Cannot bitcast pointer between different address spaces");
      return success();
    }
    return emitError("Cannot bitcast pointer to non-pointer type");
  }
  unsigned dstBits = dstType.getIntOrFloatBitWidth();
  unsigned srcBits = srcType.getIntOrFloatBitWidth();
  if (dstBits != srcBits) {
    return emitError("Cannot bitcast data-type of size ")
           << srcBits << " to data-type of size " << dstBits;
  }
  return success();
}

//-- BroadcastOp --
void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<BroadcastSplatPattern, BroadcastBroadcastPattern>(context);
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  if (getType() == getSrc().getType()) {
    // no-op
    return getSrc();
  }

  auto value = adaptor.getSrc();
  if (!value)
    return {};

  if (auto denseElemsAttr = dyn_cast<SplatElementsAttr>(value)) {
    auto shapedType = cast<ShapedType>(getType());
    return denseElemsAttr.resizeSplat(shapedType);
  }
  return {};
}

LogicalResult BroadcastOp::verify() {
  auto src = getSrc();
  auto srcTensorType = cast<RankedTensorType>(src.getType());
  auto srcShape = srcTensorType.getShape();
  auto result = getResult();
  auto resultTensorType = cast<RankedTensorType>(result.getType());
  auto resultShape = resultTensorType.getShape();
  if (srcShape.size() != resultShape.size()) {
    return emitError("rank of source must be same as rank of result");
  }
  for (int i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] != 1 && srcShape[i] != resultShape[i]) {
      return emitError("Different dimensions at index ")
             << i << " between source and result.  "
             << "Broadcast requires the source dimension to be 1.";
    }
  }
  return success();
}

//-- MakeTensorPtrOp --
void MakeTensorPtrOp::build(OpBuilder &builder, OperationState &state,
                            Value base, ValueRange shape, ValueRange strides,
                            ValueRange offsets, ArrayRef<int32_t> tensorShape,
                            ArrayRef<int32_t> order) {
  // Get pointer type from `base`
  auto pointerType = cast<PointerType>(base.getType());
  assert(pointerType != nullptr);

  // Build type `tt.ptr<tensor<tensorShape, base.pointeeType>>`
  auto tensorType = RankedTensorType::get(
      SmallVector<int64_t>(tensorShape.begin(), tensorShape.end()),
      pointerType.getPointeeType());
  auto result = PointerType::get(tensorType, pointerType.getAddressSpace());

  return build(builder, state, result, base, shape, strides, offsets,
               builder.getDenseI32ArrayAttr(order));
}

//-- AddPtrOp --
OpFoldResult AddPtrOp::fold(FoldAdaptor adaptor) {
  // addptr(ptr, 0) -> ptr
  if (matchPattern(adaptor.getOffset(), m_Zero())) {
    return getPtr();
  }
  return {};
}

//-- AdvanceOp --
OpFoldResult AdvanceOp::fold(FoldAdaptor adaptor) {
  // advance(ptr, 0, 0) -> ptr
  SmallVector<OpFoldResult> rawOffsets = getOffsets();
  auto offsets = getConstantIntValues(rawOffsets);
  if (!offsets.has_value())
    return {};
  for (int64_t offset : offsets.value())
    if (offset != 0)
      return {};
  return getPtr();
}

//-- MakeTensorDescOp --
void MakeTensorDescOp::build(OpBuilder &builder, OperationState &state,
                             Value base, ValueRange shape, ValueRange strides,
                             ArrayRef<int32_t> blockShape) {
  auto ptrTy = dyn_cast<triton::PointerType>(base.getType());
  if (!ptrTy) {
    llvm::report_fatal_error("Expected pointer type");
  }
  auto elemTy = ptrTy.getPointeeType();

  SmallVector<int64_t> blockShape64(blockShape);
  auto blockTy = RankedTensorType::get(blockShape64, elemTy);
  auto descTy = TensorDescType::get(builder.getContext(), blockTy);
  return build(builder, state, descTy, base, shape, strides);
}

// The following ops, including `call`, `func`, and `return` are copied and
// modified from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Func/IR/FuncOps.cpp
// We could revert it back once MLIR has a better inliner interface.
//-- FuncOp --
void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

// -- CallOp --
LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this).getProperties().callee;
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

// -- ReturnOp --
LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

// -- JoinOp --

void JoinOp::build(OpBuilder &builder, OperationState &state, Value lhs,
                   Value rhs) {
  auto lhsTy = cast<RankedTensorType>(lhs.getType());
  SmallVector<int64_t> retShape(lhsTy.getShape());
  retShape.push_back(2);

  Attribute srcEnc = lhsTy.getEncoding();
  Attribute retEnc;
  if (srcEnc) {
    if (cast<DialectInferLayoutInterface>(&srcEnc.getDialect())
            ->inferDefaultJoinOpEncoding(srcEnc, retEnc, lhsTy.getShape(),
                                         /*loc=*/std::nullopt)
            .failed()) {
      assert(false && "failed to infer join encoding");
    }
  }
  auto retTy = RankedTensorType::get(retShape, lhsTy.getElementType(), retEnc);
  JoinOp::build(builder, state, retTy, lhs, rhs);
}

LogicalResult JoinOp::verify() {
  RankedTensorType srcTy = getLhs().getType();
  SmallVector<int64_t> retShape(srcTy.getShape());
  retShape.push_back(2);

  RankedTensorType retTy = getType();
  if (SmallVector<int64_t>(retTy.getShape()) != retShape) {
    return emitOpError("result shape must be (")
           << retShape << "), but got " << retTy.getShape();
  }
  if (retTy.getElementType() != srcTy.getElementType()) {
    return emitOpError("result element type must match the input element type");
  }
  Attribute retEnc = retTy.getEncoding();
  if (!retEnc) {
    if (srcTy.getEncoding()) {
      return emitOpError("result encoding must be specified");
    }
    return success();
  }
  // There are multiple correct destination layout for a given source layout but
  // there is only one correct source layout for a given destination layout. So
  // we verify that the source layout match the destination layout.
  Attribute srcEnc;
  Location location = getLoc();
  if (cast<DialectInferLayoutInterface>(&retEnc.getDialect())
          ->inferSplitOpEncoding(retEnc, srcEnc, retShape, location)
          .failed()) {
    return failure();
  }

  if (cast<triton::DialectInferLayoutInterface>(&srcEnc.getDialect())
          ->verifyLayoutsAreEqual(srcTy.getShape(), srcEnc, srcTy.getEncoding(),
                                  {})
          .failed()) {
    return emitOpError("incompatible join layout");
  }
  return success();
}

// -- SplitOp --
LogicalResult SplitOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    SplitOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto srcTy = cast<RankedTensorType>(adaptor.getSrc().getType());
  auto srcShape = srcTy.getShape();

  if (srcShape.empty() || srcShape.back() != 2) {
    return emitOptionalError(location,
                             "last dimension of input tensor must be 2");
  }
  ArrayRef<int64_t> retShape(srcShape.begin(), srcShape.end() - 1);

  Attribute srcEnc = srcTy.getEncoding();
  Attribute retEnc;
  if (srcEnc) {
    if (cast<DialectInferLayoutInterface>(&srcEnc.getDialect())
            ->inferSplitOpEncoding(srcEnc, retEnc, srcTy.getShape(), location)
            .failed()) {
      return failure();
    }
  }
  auto retTy = RankedTensorType::get(retShape, srcTy.getElementType(), retEnc);
  inferredReturnTypes.push_back(retTy);
  inferredReturnTypes.push_back(retTy);
  return success();
}

// -- ElementwiseInlineAsmOp --
void ElementwiseInlineAsmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getPure())
    return;
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

Speculation::Speculatability ElementwiseInlineAsmOp::getSpeculatability() {
  if (getPure())
    return Speculation::Speculatable;
  return Speculation::NotSpeculatable;
}

LogicalResult ElementwiseInlineAsmOp::verify() {
  if (getNumOperands() >= 1) {
    auto tensorType = dyn_cast<RankedTensorType>(getOperand(0).getType());
    size_t numInputElems = tensorType ? tensorType.getNumElements() : 0;
    if (numInputElems % this->getPackedElement() != 0) {
      return emitError("number of input elements ")
             << numInputElems
             << " must be a multiple of the op's packed_element attribute, "
             << getPackedElement();
    }
  }
  return success();
}

// -- ExternElementwiseOp --
void ExternElementwiseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getPure())
    return;
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

Speculation::Speculatability ExternElementwiseOp::getSpeculatability() {
  if (getPure())
    return Speculation::Speculatable;
  return Speculation::NotSpeculatable;
}

// -- GatherOp --
LogicalResult GatherOp::verify() {
  RankedTensorType indicesTy = getIndices().getType();
  RankedTensorType srcTy = getSrc().getType();
  RankedTensorType resTy = getResult().getType();

  if (indicesTy.getShape() != resTy.getShape()) {
    return emitOpError("indices and output shapes must match");
  }
  if (indicesTy.getEncoding() != resTy.getEncoding()) {
    return emitOpError("indices and output encodings must match");
  }
  if (srcTy.getElementType() != resTy.getElementType()) {
    return emitOpError("input and output element types must match");
  }
  if (srcTy.getRank() != indicesTy.getRank()) {
    return emitOpError("input and indices ranks must match");
  }
  if (getAxis() >= srcTy.getRank()) {
    return emitOpError("gather dimension must be less than the input rank");
  }
  for (int dim = 0; dim < indicesTy.getRank(); ++dim) {
    if (dim == getAxis())
      continue;
    if (indicesTy.getShape()[dim] != srcTy.getShape()[dim]) {
      return emitOpError("indices dimension ")
             << dim << " must match the corresponding input dimension";
    }
  }

  return success();
}

LogicalResult GatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  GatherOpAdaptor adaptor(operands, attributes, properties, regions);
  auto indicesType = cast<RankedTensorType>(adaptor.getIndices().getType());
  auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());

  // Shape and encoding of the indices with the element type of the src.
  inferredReturnTypes.push_back(
      RankedTensorType::get(indicesType.getShape(), srcType.getElementType(),
                            indicesType.getEncoding()));
  return success();
}

// -- DescriptorGatherOp
LogicalResult DescriptorGatherOp::verifyResultType(Operation *op,
                                                   mlir::ShapedType type) {
  if (type.getRank() != 2)
    return op->emitOpError("result must be a 2D tensor, but got ") << type;

  // The swizzling of TMA accesses matches that of the MMAv3 shared memory
  // layouts. However, these have minimum size requirements.
  // TODO: We can support smaller gather sizes by padding the `local_alloc` this
  // lowers to to the nearest minimum tile size.
  if (unsigned rows = type.getShape()[0]; rows < 8) {
    return op->emitOpError("gather must have at least 8 rows, but got ")
           << rows;
  }

  Type dtype = type.getElementType();
  if (dtype.getIntOrFloatBitWidth() > 32)
    return op->emitOpError("TMA dtype cannot be greater than 32 bits");

  unsigned minCols = 32 / dtype.getIntOrFloatBitWidth() * 8;
  if (unsigned cols = type.getShape()[1]; cols < minCols) {
    return op->emitOpError("gather of ")
           << dtype << " must have at least " << minCols << " columns, but got "
           << cols;
  }

  return success();
}

LogicalResult DescriptorGatherOp::verify() {
  RankedTensorType blockType = getDesc().getType().getBlockType();
  // Gather from `!tt.tensordesc<tensor<1xMxdtype>>`.
  if (blockType.getRank() != 2)
    return emitOpError("block must be a 2D tensor, but got ") << blockType;
  if (blockType.getShape()[0] != 1)
    return emitOpError("block must have exactly 1 row, but got ") << blockType;

  // With x offsets `tensor<Nxinttype>`.
  RankedTensorType indicesType = getXOffsets().getType();
  if (indicesType.getRank() != 1)
    return emitOpError("x offsets must be a 1D tensor, but got ")
           << indicesType;

  // Into `tensor<NxMxdtype>`.
  RankedTensorType resultType = getType();
  if (failed(verifyResultType(*this, resultType)))
    return failure();

  if (resultType.getShape()[0] != indicesType.getShape()[0]) {
    return emitOpError("result tensor must have as many rows as indices (")
           << indicesType.getShape()[0] << "), but got " << resultType;
  }
  if (resultType.getShape()[1] != blockType.getShape()[1]) {
    return emitOpError("result tensor number of columns must match block (")
           << blockType.getShape()[1] << "), but got " << resultType;
  }
  if (resultType.getElementType() != blockType.getElementType()) {
    return emitOpError("result tensor element type must match block (")
           << blockType.getElementType() << "), but got " << resultType;
  }

  return success();
}

// -- DescriptorLoadOp --
static LogicalResult verifyDesciptorLoadStoreType(Operation *op,
                                                  TensorDescType desc,
                                                  RankedTensorType tensor) {
  RankedTensorType block = desc.getBlockType();
  ArrayRef<int64_t> blockShape = block.getShape();
  ArrayRef<int64_t> tensorShape = tensor.getShape();
  if (blockShape.size() > tensorShape.size()) {
    // Allow ranked reduced load if the leading dimensions are all 1s.
    for (int i = 0; i < blockShape.size() - tensorShape.size(); ++i) {
      if (blockShape[i] != 1)
        return op->emitOpError(
            "ranked reduce load only allowed for unit dimension leading dim.");
    }
    blockShape = blockShape.take_back(tensorShape.size());
  }

  if (blockShape == tensorShape &&
      block.getElementType() == tensor.getElementType())
    return success();
  return op->emitOpError("tensor desciptor block and tensor types must match");
}

LogicalResult DescriptorLoadOp::verify() {
  return verifyDesciptorLoadStoreType(*this, getDesc().getType(), getType());
}

// -- DescriptorStoreOp --
LogicalResult DescriptorStoreOp::verify() {
  return verifyDesciptorLoadStoreType(*this, getDesc().getType(),
                                      getSrc().getType());
}

// -- ExperimentalTensormapCreateOp --
LogicalResult ExperimentalTensormapCreateOp::verify() {
  auto rank = getBoxDim().size();
  if (getGlobalDim().size() != rank) {
    return emitError("Rank mismatch for global dim. Got ")
           << getGlobalDim().size() << " but expected " << rank;
  }
  if (getGlobalStride().size() + 1 != rank) {
    return emitError("Rank mismatch for global stride. Got ")
           << getGlobalStride().size() << " but expected " << rank - 1;
  }
  if (getElementStride().size() != rank) {
    return emitError("Rank mismatch for element stride. Got ")
           << getElementStride().size() << " but expected " << rank;
  }
  return success();
}

} // namespace triton
} // namespace mlir
