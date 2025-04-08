#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <utility>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "tritonamdgpu-canonicalize-pointers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = triton;

// -----------------------------------------------------------------------------
// Pointer canonicalizer utility class
// -----------------------------------------------------------------------------
// This class iterates through the argument of the `funcOp`, if the argument is
// a pointer, starts a walk through its transitive uses to build a in-memory
// data structure to record the current offset to that pointer. Only when the
// pointer is really loaded/stored we materialize the base pointer with the
// offset.
//
// Let's suppose that `arg0` is a pointer. The algorithm works like that:
//
// a) At the beginning the offset is a tensor initialized to zero, and we
//    associate with `%arg0` a `FatPtr{basePtr=%arg0, offset=0}`. Through the
//    algorithm `FatPtr.basePtr` represents the scalar base pointer (all the
//    uniform updates will go into that) and `FatPtr.offset` represents the
//    tensor offset (all the non-uniform updates will go into that)
//
//
// b) Follow the pointer through the IR. When we meet:
//    `%ptr = tt.addptr(%arg0, %offset)`
//
//    Isolate the uniform and the non-uniform contributions of %offset =
//    (%u_offset, %nu_offset) and update the scalar pointer and the tensor
//    offset
//    ```
//    %s_ptr = addi(%fatPoniters[ptr].basePtr, %u_offset)
//    %t_offset = addi(%fatPoniters[ptr].offset, %nu_offset)
//    %fatPointers[%ptr0] = FatPtr{base=%s_ptr, offset=%t_offset}
//    ```
// c) When we meet the `tt.load(%ptr)` or `tt.store(%ptr)` instructions,
//    replace that instruction with:
//    `%t_ptr = tt.splat(%fatPointers[%ptr].basePtr)
//    `%fat_ptr = tt.addptr(%t_ptr, %fatPointers[ptr].offset)`
//    `%data = tt.load(%fat_ptr)`
//
// Please note that `%offset` might be a 32bit or 64bit integer. If
// we can, we would like to use 32 bit integers. This can happen under
// certain conditions:
//
// a) We can determine that the offset cannot overflow. In this case, we can
//    downcast the pointer just before emitting the load
// b) We know that the underlying memory size can be expressed as a 32 bit
//    value. In this case we can simply start with a 32bit offset and downcast
//    if we ever meet 64 bit operations (because we know that the offset can be
//    contained in 32 bits)
//
namespace {

// Extend a 32bit `offset` into 64bit using a arith.extsi operation
static Value createExtend32bitOffsetTo64Bits(RewriterBase &rewriter,
                                             Location loc, Value offset) {
  if (auto tensorType = dyn_cast<RankedTensorType>(offset.getType())) {
    auto shape = tensorType.getShape();
    auto newTensorType = RankedTensorType::get(shape, rewriter.getI64Type(),
                                               tensorType.getEncoding());
    return rewriter.create<arith::ExtSIOp>(loc, newTensorType, offset);
  }
  return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offset);
}

// Narrow a 64bit `offset` into 32bit using a arith.trunci operation
static Value createNarrow64bitOffsetTo32bits(RewriterBase &rewriter,
                                             Location loc, Value offset) {
  Type elementType = getElementTypeOrSelf(offset);
  if (elementType.isInteger(32))
    return offset;

  if (auto tensorType = dyn_cast<RankedTensorType>(offset.getType())) {
    auto shape = tensorType.getShape();
    auto newTensorType = RankedTensorType::get(shape, rewriter.getI32Type(),
                                               tensorType.getEncoding());
    return rewriter.create<arith::TruncIOp>(loc, newTensorType, offset);
  }
  return rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), offset);
}

// Helper function to determine if the given `op` is a constant tensor and in
// that case return the scalar value.
std::optional<Value> maybeGetOrCreateScalarConstant(RewriterBase &rewriter,
                                                    Location loc, Value expr) {
  Operation *op = expr.getDefiningOp();

  // Check for splatness
  if (auto splatOp = dyn_cast_or_null<tt::SplatOp>(op))
    return splatOp.getSrc();

  // Check for constant
  DenseIntElementsAttr constVal;
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op)) {
    Value val = constOp.getResult();
    if (matchPattern(val, m_Constant(&constVal)) && constVal.isSplat())
      return rewriter.create<arith::ConstantOp>(
          loc, constVal.getSplatValue<IntegerAttr>());
  }

  // Check for block arguments
  if (auto blockArg = dyn_cast_or_null<BlockArgument>(expr)) {
    Type type = blockArg.getType();
    if (!isa<RankedTensorType>(type))
      return blockArg;
  }

  return {};
}

// Narrowing logic
// For now we allow to narrow down to 32 bits only in the following case:
// - `baseOffset` is 32-bits and `addOffset`(64-bits) is zero
bool canNarrowOffset(Value baseOffset, Value addOffset) {
  Type addOffsetType = getElementTypeOrSelf(addOffset);
  auto baseSplatOp = baseOffset.getDefiningOp<tt::SplatOp>();
  return baseSplatOp && addOffsetType.isInteger(32);
}

// Create a zero tensor with a given `type`
Value createTensorZero(RewriterBase &rw, Location loc, RankedTensorType type) {
  mlir::Attribute zeroAttr = rw.getZeroAttr(type.getElementType());
  auto zeroDenseAttr = DenseElementsAttr::get(type, zeroAttr);
  return rw.create<arith::ConstantOp>(loc, zeroDenseAttr);
}

} // namespace

std::pair<Value, Value> createDecomposeOffsetFromExpr(RewriterBase &rewriter,
                                                      Location loc, Value expr,
                                                      int64_t bitness);
// Offset extraction logic for an addition op:
// decompose(A+B) = {U(A)+U(B), NU(A)+NU(B)}
std::pair<Value, Value> createDecomposeOffsetFromAdd(RewriterBase &rewriter,
                                                     Location loc, Value expr,
                                                     int64_t bitness) {
  auto addOp = expr.getDefiningOp<arith::AddIOp>();
  auto [uniformOffsetL, nonUniformOffsetL] =
      createDecomposeOffsetFromExpr(rewriter, loc, addOp.getLhs(), bitness);
  auto [uniformOffsetR, nonUniformOffsetR] =
      createDecomposeOffsetFromExpr(rewriter, loc, addOp.getRhs(), bitness);
  Value uniformAdd =
      rewriter.create<arith::AddIOp>(loc, uniformOffsetL, uniformOffsetR);
  Value nonUniformAdd =
      rewriter.create<arith::AddIOp>(loc, nonUniformOffsetL, nonUniformOffsetR);
  return {uniformAdd, nonUniformAdd};
}

// Offset extraction logic for a multiplication op:
// decompose(A*B) = {U(A)*U(B), NU(A)*NU(B)+NU(B)*U(A)+U(A)*NU(B)}
std::pair<Value, Value> createDecomposeOffsetFromMul(RewriterBase &rewriter,
                                                     Location loc, Value expr,
                                                     int64_t bitness) {
  auto mulOp = expr.getDefiningOp<arith::MulIOp>();
  auto [uniformOffsetL, nonUniformOffsetL] =
      createDecomposeOffsetFromExpr(rewriter, loc, mulOp.getLhs(), bitness);
  auto [uniformOffsetR, nonUniformOffsetR] =
      createDecomposeOffsetFromExpr(rewriter, loc, mulOp.getRhs(), bitness);
  Value uniformMul =
      rewriter.create<arith::MulIOp>(loc, uniformOffsetL, uniformOffsetR);

  Value uniformOffsetLSplat = rewriter.create<tt::SplatOp>(
      loc, nonUniformOffsetL.getType(), uniformOffsetL);
  Value uniformOffsetRSplat = rewriter.create<tt::SplatOp>(
      loc, nonUniformOffsetR.getType(), uniformOffsetR);

  Value nonUNonU =
      rewriter.create<arith::MulIOp>(loc, nonUniformOffsetL, nonUniformOffsetR);
  Value nonUU = rewriter.create<arith::MulIOp>(loc, uniformOffsetLSplat,
                                               nonUniformOffsetR);
  Value uNonU = rewriter.create<arith::MulIOp>(loc, nonUniformOffsetL,
                                               uniformOffsetRSplat);

  Value tmp = rewriter.create<arith::AddIOp>(loc, nonUNonU, nonUU);
  Value nonUniformMul = rewriter.create<arith::AddIOp>(loc, tmp, uNonU);
  return {uniformMul, nonUniformMul};
}

std::pair<Value, Value> createDecomposeOffsetFromExpr(RewriterBase &rewriter,
                                                      Location loc, Value expr,
                                                      int64_t bitness) {

  // Base case 1: it is a splat. Return the scalar constant as the uniform part
  if (auto scalarConst = maybeGetOrCreateScalarConstant(rewriter, loc, expr)) {
    auto tensorZero =
        createTensorZero(rewriter, loc, cast<RankedTensorType>(expr.getType()));
    return {*scalarConst, tensorZero};
  }

  // Base case 2: block argument. Since it is not a scalar constant, it must be
  // a tensor. Note that this means we won't be able to decompose across loop
  // boundaries (TODO: giuseros).
  if (llvm::isa<BlockArgument>(expr)) {
    Value scalarZero = rewriter.create<arith::ConstantIntOp>(loc, 0, bitness);
    return {scalarZero, expr};
  }

  auto offsets =
      llvm::TypeSwitch<Operation *, std::pair<Value, Value>>(
          expr.getDefiningOp())
          .Case<tt::BroadcastOp>([&](auto broadcastOp) {
            auto [uniform, nonUniform] = createDecomposeOffsetFromExpr(
                rewriter, loc, broadcastOp.getSrc(), bitness);
            auto broadcastNonUniform = rewriter.create<tt::BroadcastOp>(
                loc, broadcastOp.getType(), nonUniform);
            return std::make_pair(uniform, broadcastNonUniform);
          })
          .Case<tt::ExpandDimsOp>([&](auto expandOp) {
            auto [uniform, nonUniform] = createDecomposeOffsetFromExpr(
                rewriter, loc, expandOp.getSrc(), bitness);
            auto expandNonUniform = rewriter.create<tt::ExpandDimsOp>(
                loc, nonUniform, expandOp.getAxis());
            return std::make_pair(uniform, expandNonUniform);
          })
          .Case<arith::AddIOp>([&](Operation *op) {
            return createDecomposeOffsetFromAdd(rewriter, loc, expr, bitness);
          })
          .Case<arith::MulIOp>([&](Operation *op) {
            return createDecomposeOffsetFromMul(rewriter, loc, expr, bitness);
          })
          .Default([&](Operation *op) {
            // Base case 3: it is not a supported operation. We assume no
            // uniform part
            Value scalarZero =
                rewriter.create<arith::ConstantIntOp>(loc, 0, bitness);
            return std::make_pair(scalarZero, expr);
          });

  return offsets;
}

static const std::string kPtrCanonPrefix = "__amdpointercanonicalize.";
static const std::string kSCFThenRewrittenAttr =
    kPtrCanonPrefix + "scf-then-rewritten__";
static const std::string kSCFElseRewrittenAttr =
    kPtrCanonPrefix + "scf-else-rewritten__";
static const std::string kSCFIfOpYieldFatPtrOffsets =
    kPtrCanonPrefix + "scf-if-yield-fatptr-offsets__";

/// This struct is basically a thin wrapper over DenseMap<fatPtr, fatPtrAttrs>
/// where fatPtr == (base, offset) and fatPtrAttrs is itself a map of (name,
/// attribute).
/// It is used to associate metadata/attributes with the canonicalized fat
/// pointers, such as `tt.pointer_range` and whether operations involving them
/// can be narrowed (`canNarrow`).
struct FatPointers {
  struct FatPtrAttrs {
    FatPtrAttrs(const FatPtrAttrs &other) = default;
    FatPtrAttrs &operator=(const FatPtrAttrs &other) = default;
    // for map default insert
    FatPtrAttrs() = default;

    friend bool operator==(const FatPtrAttrs &lhs, const FatPtrAttrs &rhs) {
      return lhs.canNarrow == rhs.canNarrow && lhs.attributes == rhs.attributes;
    }

    friend bool operator!=(const FatPtrAttrs &lhs, const FatPtrAttrs &rhs) {
      return !(lhs == rhs);
    }

    llvm::DenseMap<StringRef, Attribute> attributes;
    bool canNarrow = false;
  };

  using KeyT = std::pair<Value, Value>;
  using ValueT = FatPtrAttrs;
  using DenseMapT = DenseMap<KeyT, ValueT>;

  void collectFatPointerAttributes(const KeyT &k);
  ValueT &operator[](const KeyT &k) {
    if (!pointerAttrs.contains(k))
      collectFatPointerAttributes(k);
    return pointerAttrs[k];
  }

  ValueT &operator[](KeyT &&k) {
    if (!pointerAttrs.contains(k))
      collectFatPointerAttributes(k);
    return pointerAttrs[k];
  }

  template <typename T>
  using const_arg_type_t = typename llvm::const_pointer_or_const_ref<T>::type;
  const ValueT &at(const_arg_type_t<KeyT> k) const {
    // this is redundant - DenseMap will assert the same thing - but better to
    // have our own message
    assert(pointerAttrs.contains(k) &&
           "expected fatPtrs to contain remapped fat pointer");
    return pointerAttrs.at(k);
  }

  bool contains(const KeyT &k) { return pointerAttrs.contains(k); }

private:
  DenseMapT pointerAttrs;
};

// TODO(max): reconsider this approach, specifically how narrowing and
// attributes are propagated starting from a tt.ptr.
void FatPointers::collectFatPointerAttributes(const KeyT &k) {
  auto [base, offset] = k;
  // If it is the i-th block argument, then look if the operation defined some
  // _argi attribute and add it to the fat pointer attributes
  if (auto arg = dyn_cast<BlockArgument>(base)) {
    // If the value is a block parameter, the operation can specify
    // an attribute for the given parameter by using `tt.property_argi`
    // where `argi` refers to the arg number of the given parameter.
    // So we need to iterate through the property, find the right one
    // and push the property onto the pointers attributes.
    auto op = arg.getOwner()->getParentOp();
    for (NamedAttribute namedAttr : op->getAttrs()) {
      StringAttr attrName = namedAttr.getName();
      std::string argSuffix =
          llvm::formatv("_arg{0}", arg.getArgNumber()).str();
      if (!attrName.strref().ends_with(argSuffix))
        continue;

      auto newAttrName = attrName.strref().drop_back(argSuffix.size());
      pointerAttrs[k].attributes[newAttrName] = namedAttr.getValue();
      // Propagate the argument to the offset if it is also a block
      // argument
      if (auto offsetArg = dyn_cast<BlockArgument>(offset))
        op->setAttr(
            llvm::formatv("{0}_arg{1}", newAttrName, offsetArg.getArgNumber())
                .str(),
            namedAttr.getValue());
    }
    return;
  }

  // Otherwise add the attributes of the base to the fat pointer
  for (auto baseAttr : base.getDefiningOp()->getAttrs())
    pointerAttrs[k].attributes[baseAttr.getName()] = baseAttr.getValue();
}

Value createTensorPointer(RewriterBase &rewriter, Value basePtr, Value offset,
                          Location loc,
                          const FatPointers::FatPtrAttrs &fatPtrAttrs) {
  auto tensorType = dyn_cast<RankedTensorType>(offset.getType());

  // Scalar case: we only need to `tt.addptr %basePtr, %offset`
  if (!tensorType) {
    auto addPtrOp =
        rewriter.create<tt::AddPtrOp>(loc, basePtr.getType(), basePtr, offset);
    for (const auto &attribute : fatPtrAttrs.attributes)
      addPtrOp->setAttr(attribute.getFirst(), attribute.getSecond());
    return addPtrOp.getResult();
  }

  // Tensor case: splat the scalar pointer and add the (tensor) offset:
  // ```
  //    %tensorBasePtr = tt.splat %basePtr
  //    %tensorPtr = tt.addptr %tensorBasePtr, %offset
  // ```
  ArrayRef<int64_t> offsetShape = tensorType.getShape();
  auto tensorPtrType = RankedTensorType::get(offsetShape, basePtr.getType(),
                                             tensorType.getEncoding());
  if (fatPtrAttrs.canNarrow)
    offset = createNarrow64bitOffsetTo32bits(rewriter, loc, offset);

  tt::SplatOp tensorPtr =
      rewriter.create<tt::SplatOp>(loc, tensorPtrType, basePtr);
  tt::AddPtrOp addPtrOp =
      rewriter.create<tt::AddPtrOp>(loc, tensorPtrType, tensorPtr, offset);

  for (const auto &attribute : fatPtrAttrs.attributes)
    addPtrOp->setAttr(attribute.getFirst(), attribute.getSecond());
  return addPtrOp.getResult();
}

/// Flatten the given value ranges into a single vector of values.
static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (const ValueRange &vals : values)
    llvm::append_range(result, vals);
  return result;
}

/// Assert that the given value range contains a single value and return it.
static Value getSingleValue(ValueRange values) {
  assert(values.size() == 1 && "expected single value");
  return values.front();
}

/// This is convenience class (that is a copy-paste of some of
/// OpConversionPattern) that keeps track of (and removes from) opToRewrite
/// after successful matchAndRewrite_ calls; subclasses must define
/// matchAndRewrite_ just as that would for conventional OpConversionPatterns.
template <typename SourceOp>
struct PointerCanonicalizationPattern : ConversionPattern {
  using OpAdaptor = typename SourceOp::Adaptor;
  using OneToNOpAdaptor =
      typename SourceOp::template GenericAdaptor<ArrayRef<ValueRange>>;

  PointerCanonicalizationPattern(MLIRContext *context,
                                 llvm::SetVector<Operation *> &opsToRewrite,
                                 FatPointers &fatPtrs,
                                 PatternBenefit benefit = 1)
      : ConversionPattern(SourceOp::getOperationName(), benefit, context),
        fatPtrs(fatPtrs), opToRewrite(opsToRewrite) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceOp = cast<SourceOp>(op);
    if (failed(matchAndRewrite_(sourceOp, OneToNOpAdaptor(operands, sourceOp),
                                rewriter)))
      return failure();
    opToRewrite.remove(op);
    return success();
  }

  virtual LogicalResult
  matchAndRewrite_(SourceOp op, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const = 0;

  FatPointers &fatPtrs;
  llvm::SetVector<Operation *> &opToRewrite;
};

/// splat integer offset, keep base
class ConvertSplatOp : public PointerCanonicalizationPattern<tt::SplatOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite_(tt::SplatOp splatOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    ValueRange remappedOperands = adaptor.getSrc();
    if (remappedOperands.size() != 2) {
      // some prior op materialized the fat ptr, e.g.:
      // %3 = tt.bitcast %2
      // %4 = tt.splat %3
      return success();
    }
    Value fatPtrBase = remappedOperands[0];
    Value fatPtrOffset = remappedOperands[1];
    if (!llvm::isa<tt::PointerType>(fatPtrBase.getType()))
      return rewriter.notifyMatchFailure(splatOp,
                                         "non tt.ptr base unimplemented");
    if (!llvm::isa<IntegerType>(fatPtrOffset.getType()))
      return rewriter.notifyMatchFailure(splatOp,
                                         "non-integer offset unimplemented");

    RankedTensorType outType = splatOp.getResult().getType();
    auto newOffsetType = RankedTensorType::get(
        outType.getShape(), fatPtrOffset.getType(), outType.getEncoding());
    tt::SplatOp offset = rewriter.create<tt::SplatOp>(
        splatOp.getLoc(), newOffsetType, fatPtrOffset);
    rewriter.replaceOpWithMultiple(splatOp, {{fatPtrBase, offset}});
    fatPtrs[{fatPtrBase, offset}] = fatPtrs.at({fatPtrBase, fatPtrOffset});

    return success();
  }
};

/// Broadcast offset, keep base.
class ConvertBroadcastOp
    : public PointerCanonicalizationPattern<tt::BroadcastOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite_(tt::BroadcastOp broadcastOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    ValueRange remappedOperands = adaptor.getSrc();
    if (remappedOperands.size() != 2) {
      // some prior op materialized the fat ptr, e.g.:
      // %3 = tt.bitcast %2
      // %4 = tt.broadcast %3
      return success();
    }

    Value fatPtrBase = remappedOperands[0];
    Value fatPtrOffset = remappedOperands[1];
    if (!llvm::isa<tt::PointerType>(fatPtrBase.getType()))
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "non tt.ptr base unimplemented");
    auto offsetType = dyn_cast<RankedTensorType>(fatPtrOffset.getType());
    if (!offsetType)
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "non-tensor offset unimplemented");

    auto outType =
        dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
    auto newOffsetType = RankedTensorType::get(
        outType.getShape(), offsetType.getElementType(), outType.getEncoding());
    tt::BroadcastOp newOffset = rewriter.create<tt::BroadcastOp>(
        broadcastOp.getLoc(), newOffsetType, fatPtrOffset);
    rewriter.replaceOpWithMultiple(broadcastOp, {{fatPtrBase, newOffset}});
    fatPtrs[{fatPtrBase, newOffset}] = fatPtrs.at({fatPtrBase, fatPtrOffset});
    return success();
  }
};

/// Three cases:
/// 1. If it is a scalar pointer update -> bump only the base pointer;
/// 2. Constant tensor offset -> bump only the offset
/// 3. Non-constant tensor offset -> decompose parent(offset) into uniform and
/// non-uniform components.
class ConvertAddPtrOp : public PointerCanonicalizationPattern<tt::AddPtrOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite_(tt::AddPtrOp addPtrOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    ValueRange remappedPtr = adaptor.getPtr();
    if (remappedPtr.size() != 2) {
      // some prior op materialized the fat ptr, e.g.:
      // %3 = tt.bitcast %2
      // %4 = tt.addptr %3
      return success();
    }
    ValueRange nonRemappedOffset = adaptor.getOffset();
    if (nonRemappedOffset.size() != 1)
      return rewriter.notifyMatchFailure(
          addPtrOp, "expected AddPtrOp Offset to have not have been remapped");
    Value fatPtrBase = remappedPtr[0];
    Value fatPtrOffset = remappedPtr[1];
    Value origOffset = nonRemappedOffset[0];
    Location curLoc = addPtrOp.getLoc();

    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(addPtrOp);

    // If it is a scalar pointer update, simply bump the base pointer
    if (llvm::isa<tt::PointerType>(addPtrOp.getPtr().getType())) {
      assert(llvm::isa<IntegerType>(origOffset.getType()) &&
             "expected offset to be integer type");
      auto newAddPtrOp = rewriter.create<tt::AddPtrOp>(
          curLoc, fatPtrBase.getType(), fatPtrBase, origOffset);
      rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, fatPtrOffset}});
      fatPtrs[{newAddPtrOp, fatPtrOffset}] =
          fatPtrs.at({fatPtrBase, fatPtrOffset});
      return success();
    }

    assert(llvm::isa<RankedTensorType>(addPtrOp.getPtr().getType()) &&
           "expected Ptr to be RankedTensorType type");

    // Early exit for the case of a constant tensor
    if (auto scalarConst =
            maybeGetOrCreateScalarConstant(rewriter, curLoc, origOffset)) {
      tt::AddPtrOp newAddPtrOp = rewriter.create<tt::AddPtrOp>(
          curLoc, fatPtrBase.getType(), fatPtrBase, *scalarConst);
      rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, fatPtrOffset}});
      // If we are updating the tensor pointer with a constant value, we can
      // propagate the attributes of the tensor pointer to the fat pointer.
      fatPtrs[{newAddPtrOp, fatPtrOffset}] =
          fatPtrs.at({fatPtrBase, fatPtrOffset});
      return success();
    }

    int64_t bitness = llvm::cast<RankedTensorType>(origOffset.getType())
                          .getElementTypeBitWidth();
    auto [uniformOffset, nonUniformOffset] =
        createDecomposeOffsetFromExpr(rewriter, curLoc, origOffset, bitness);

    auto newAddPtrOp = rewriter.create<tt::AddPtrOp>(
        curLoc, fatPtrBase.getType(), fatPtrBase, uniformOffset);

    // Vector offset update (if any): bump the tensor offset
    bool canNarrow = fatPtrs.at({fatPtrBase, fatPtrOffset}).canNarrow;
    bool propagateAtrs = true;
    Value newOffset = fatPtrOffset;
    if (!isZeroConst(nonUniformOffset)) {
      Type addPtrOffsetType = getElementTypeOrSelf(nonUniformOffset);
      Type fatPtrOffsetType = getElementTypeOrSelf(fatPtrOffset);
      canNarrow = canNarrow && canNarrowOffset(fatPtrOffset, nonUniformOffset);
      // Upcast or downcast the offset accordingly
      if (addPtrOffsetType.isInteger(32) && fatPtrOffsetType.isInteger(64))
        nonUniformOffset =
            createExtend32bitOffsetTo64Bits(rewriter, curLoc, nonUniformOffset);
      else if (addPtrOffsetType.isInteger(64) && fatPtrOffsetType.isInteger(32))
        nonUniformOffset =
            createNarrow64bitOffsetTo32bits(rewriter, curLoc, nonUniformOffset);

      newOffset = rewriter.create<arith::AddIOp>(curLoc, nonUniformOffset,
                                                 fatPtrOffset);
      propagateAtrs = false;
    }

    rewriter.replaceOpWithMultiple(addPtrOp, {{newAddPtrOp, newOffset}});
    auto nextFatPtr = std::pair{newAddPtrOp.getResult(), newOffset};
    fatPtrs[nextFatPtr].canNarrow = canNarrow;
    if (propagateAtrs)
      fatPtrs[nextFatPtr].attributes =
          fatPtrs.at({fatPtrBase, fatPtrOffset}).attributes;

    return success();
  }
};

using ConversionCallbackFn =
    std::function<std::optional<LogicalResult>(Type, SmallVectorImpl<Type> &)>;

/// Rewrite init args and result type and bb args.
class ConvertSCFForOp : public PointerCanonicalizationPattern<scf::ForOp> {
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

public:
  LogicalResult
  matchAndRewrite_(scf::ForOp forOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    SmallVector<size_t> valRangeLens;
    ArrayRef<ValueRange> remappedInits = adaptor.getInitArgs();
    for (ValueRange remappedInit : remappedInits)
      valRangeLens.push_back(remappedInit.size());

    // rewrite the body bb args
    Block *oldBodyBlock = forOp.getBody();
    auto oldTypes = oldBodyBlock->getArgumentTypes();
    TypeConverter::SignatureConversion sigConversion(oldTypes.size());
    // handle the 0th arg which is the induction var
    sigConversion.addInputs(0, {oldTypes[0]});
    for (unsigned i = 1, e = oldTypes.size(); i != e; ++i) {
      SmallVector<Type> remappedInitTypes =
          llvm::to_vector(remappedInits[i - 1].getTypes());
      sigConversion.addInputs(i, remappedInitTypes);
    }
    auto newBodyBlock =
        rewriter.applySignatureConversion(oldBodyBlock, sigConversion);

    // propagate fatPtrAttrs to bb arg fatPtrs in for body bb
    // skip iv at index 0
    int offset = 1;
    for (auto operands : remappedInits) {
      if (operands.size() == 2) {
        fatPtrs[{newBodyBlock->getArgument(offset),
                 newBodyBlock->getArgument(offset + 1)}] =
            fatPtrs.at({operands[0], operands[1]});
      }
      offset += operands.size();
    }

    SmallVector<Value> initArgs = flattenValues(adaptor.getInitArgs());
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), getSingleValue(adaptor.getLowerBound()),
        getSingleValue(adaptor.getUpperBound()),
        getSingleValue(adaptor.getStep()), initArgs);

    newForOp->setAttrs(forOp->getAttrs());
    rewriter.eraseBlock(newForOp.getBody());
    rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().end());

    SmallVector<ValueRange> packedRets;
    for (unsigned i = 0, offset = 0; i < valRangeLens.size(); i++) {
      size_t len = valRangeLens[i];
      assert(offset < newForOp->getNumResults() &&
             "expected offset to be within bounds of results");
      ValueRange mappedValue = newForOp->getResults().slice(offset, len);
      // propagate fatPtrs
      if (mappedValue.size() == 2) {
        assert(remappedInits[i].size() == 2 &&
               "expected corresponding inits to be a remapped fat ptr");
        fatPtrs[{mappedValue[0], mappedValue[1]}] =
            fatPtrs.at({remappedInits[i][0], remappedInits[i][1]});
      }
      packedRets.push_back(mappedValue);
      offset += len;
    }

    rewriter.replaceOpWithMultiple(forOp, packedRets);

    return success();
  }
};

/// Rewrite with new remapped operands but also if the scf.yield is inside of
/// scf.if (possibly) annotate the scf.if.
class ConvertSCFYieldOp : public PointerCanonicalizationPattern<scf::YieldOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite_(scf::YieldOp yieldOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedYields = adaptor.getOperands();
    SmallVector<Value> newYieldedValues = flattenValues(remappedYields);
    // have to mutate here because otherwise scf.if, scf.for, and scf.while will
    // get confused about which yield is the "correct" yield (since there will
    // be two of them before the rewriter DCEs)
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp.getResultsMutable().clear();
      yieldOp.getResultsMutable().append(newYieldedValues);
    });

    // rewriting a parent op from a child op isn't a great idea but there's no
    // other to indicate to the parent IfOp that the result type can now be
    // rewritten and not before.
    if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
      rewriter.modifyOpInPlace(ifOp, [&] {
        ifOp->setDiscardableAttr(ifOp.thenBlock() == yieldOp->getBlock()
                                     ? kSCFThenRewrittenAttr
                                     : kSCFElseRewrittenAttr,
                                 rewriter.getUnitAttr());
      });
      // set indices of fatPtrs so that IfOp can propagate canNarrow to
      // result users
      int offset = 0;
      SmallVector<int64_t> fatPtrOffsets;
      for (auto operands : remappedYields) {
        if (operands.size() == 2)
          fatPtrOffsets.push_back(offset);
        offset += operands.size();
      }
      if (!fatPtrOffsets.empty())
        yieldOp->setDiscardableAttr(
            kSCFIfOpYieldFatPtrOffsets,
            rewriter.getDenseI64ArrayAttr(fatPtrOffsets));
    }

    return success();
  }
};

/// Simple here means each block arg is replaced 1-1 with the remapped operand
/// types (e.g., scf.for does not use this helper because scf.for needs to skip
/// the 0th bb arg, the induction var).
static void convertSimpleBlockSignature(Block *oldBlock,
                                        ArrayRef<ValueRange> remappedOperands,
                                        ConversionPatternRewriter &rewriter,
                                        FatPointers &fatPtrs) {
  auto oldBlockTypes = oldBlock->getArgumentTypes();
  TypeConverter::SignatureConversion blockSigConversion(oldBlockTypes.size());
  for (unsigned i = 0, e = oldBlockTypes.size(); i != e; ++i) {
    SmallVector<Type> remappedInitTypes =
        llvm::to_vector(remappedOperands[i].getTypes());
    blockSigConversion.addInputs(i, remappedInitTypes);
  }
  auto newBlock =
      rewriter.applySignatureConversion(oldBlock, blockSigConversion);

  int offset = 0;
  for (auto operands : remappedOperands) {
    if (operands.size() == 2) {
      assert(fatPtrs.contains({operands[0], operands[1]}) &&
             "expected fatPtrs to contain existing (op0, op1) fat pointer");
      fatPtrs[{newBlock->getArgument(offset),
               newBlock->getArgument(offset + 1)}] =
          fatPtrs.at({operands[0], operands[1]});
    }
    offset += operands.size();
  }
}

/// Rewrite init_args, result type, before region bb args, after region bb args.
class ConvertSCFWhileOp : public PointerCanonicalizationPattern<scf::WhileOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite_(scf::WhileOp whileOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    SmallVector<size_t> valRangeLens;
    ArrayRef<ValueRange> remappedInits = adaptor.getInits();
    for (ValueRange remappedInit : remappedInits)
      valRangeLens.push_back(remappedInit.size());

    convertSimpleBlockSignature(whileOp.getBeforeBody(), remappedInits,
                                rewriter, fatPtrs);
    convertSimpleBlockSignature(whileOp.getAfterBody(), remappedInits, rewriter,
                                fatPtrs);

    SmallVector<Value> initArgs = flattenValues(remappedInits);
    SmallVector<Type> resultTypes = llvm::map_to_vector(
        llvm::make_filter_range(
            initArgs, [](Value v) { return !v.getType().isInteger(1); }),
        [](Value v) { return v.getType(); });
    auto newWhileOp =
        rewriter.create<scf::WhileOp>(whileOp.getLoc(), resultTypes, initArgs);

    newWhileOp->setAttrs(whileOp->getAttrs());
    rewriter.inlineRegionBefore(whileOp.getBefore(), newWhileOp.getBefore(),
                                newWhileOp.getBefore().end());
    rewriter.inlineRegionBefore(whileOp.getAfter(), newWhileOp.getAfter(),
                                newWhileOp.getAfter().end());

    SmallVector<ValueRange> packedRets;
    for (unsigned i = 0, offset = 0; i < valRangeLens.size(); i++) {
      // skip %cond
      if (remappedInits[i].size() == 1 &&
          remappedInits[i].getType()[0].isInteger(1))
        continue;
      size_t len = valRangeLens[i];
      assert(offset < newWhileOp->getNumResults() &&
             "expected offset to be within bounds of results");
      ValueRange mappedValue = newWhileOp->getResults().slice(offset, len);
      // propagate fatPtrs
      if (mappedValue.size() == 2) {
        assert(remappedInits[i].size() == 2 &&
               "expected corresponding inits to be a remapped fat ptr");
        fatPtrs[{mappedValue[0], mappedValue[1]}] =
            fatPtrs.at({remappedInits[i][0], remappedInits[i][1]});
      }
      packedRets.push_back(mappedValue);
      offset += len;
    }

    rewriter.replaceOpWithMultiple(whileOp, packedRets);

    return success();
  }
};

/// Rewrite with new operands.
class ConvertSCFConditionOp
    : public PointerCanonicalizationPattern<scf::ConditionOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite_(scf::ConditionOp condOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> newArgs = flattenValues(adaptor.getArgs());
    // have to mutate here because otherwise scf.while will
    // get confused about which condition is the "correct" condition (since
    // there will be two of them before the rewriter DCEs)
    rewriter.modifyOpInPlace(condOp, [&]() {
      condOp.getArgsMutable().clear();
      condOp.getArgsMutable().append(newArgs);
    });
    return success();
  }
};

/// Rewrite operands for both true dest and false dest.
class ConvertCFCondBranch
    : public PointerCanonicalizationPattern<cf::CondBranchOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite_(cf::CondBranchOp branchOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedTrueOperands = adaptor.getTrueDestOperands();
    ArrayRef<ValueRange> remappedFalseOperands = adaptor.getFalseDestOperands();
    SmallVector<Value> trueOperands = flattenValues(remappedTrueOperands);
    SmallVector<Value> falseOperands = flattenValues(remappedFalseOperands);

    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        branchOp, branchOp.getCondition(), branchOp.getTrueDest(), trueOperands,
        branchOp.getFalseDest(), falseOperands);

    convertSimpleBlockSignature(branchOp.getTrueDest(), remappedTrueOperands,
                                rewriter, fatPtrs);
    convertSimpleBlockSignature(branchOp.getFalseDest(), remappedFalseOperands,
                                rewriter, fatPtrs);

    return success();
  }
};

/// Rewrite select(fatPtrTrue, fatPtrFalse) ->
///   (
///     select(fatPtrTrueBase, fatPtrTrueOffset),
///     select(fatPtrFalseBase, fatPtrFalseOffset)
///    )
///
/// Note, this should only be reached after both
/// operands have already been rewritten because DialectConversion walks
/// PreOrder in order ForwardDominance order: see
/// https://github.com/llvm/llvm-project/blob/58389b220a9354ed6c34bdb9310a35165579c5e3/mlir/lib/Transforms/Utils/DialectConversion.cpp#L2702
class ConvertArithSelectOp
    : public PointerCanonicalizationPattern<arith::SelectOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite_(arith::SelectOp selectOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getTrueValue().size() != 2 ||
        adaptor.getFalseValue().size() != 2) {
      assert(adaptor.getTrueValue().size() == adaptor.getFalseValue().size() &&
             "expected both true and false operands to be the same size");
      return success();
    }
    // If both have been traversed, then we can rewrite select of pointers as a
    // select of base and offset
    ValueRange fatPtrFalse = adaptor.getFalseValue();
    ValueRange fatPtrTrue = adaptor.getTrueValue();
    // Simple case of a scalar select: update the base pointer
    if (!isa<RankedTensorType>(selectOp.getType())) {
      auto newSelectOp = rewriter.create<arith::SelectOp>(
          selectOp.getLoc(), selectOp.getType(), selectOp.getCondition(),
          fatPtrTrue[0], selectOp.getFalseValue());
      rewriter.replaceOpWithMultiple(selectOp, {{newSelectOp, fatPtrTrue[1]}});
      fatPtrs[{newSelectOp, /*fatPtrOffset*/ fatPtrTrue[1]}] =
          fatPtrs.at({fatPtrTrue[0], fatPtrTrue[1]});
      return success();
    }

    // Rewrite to select(fatBaseT, fatBaseF) and select(fatOffsetT, fatOffsetF)
    auto newBase = rewriter.create<arith::SelectOp>(
        selectOp.getLoc(), selectOp.getCondition(), fatPtrTrue[0],
        fatPtrFalse[0]);
    auto newOffset = rewriter.create<arith::SelectOp>(
        selectOp.getLoc(), selectOp.getCondition(), fatPtrTrue[1],
        fatPtrFalse[1]);

    assert((fatPtrs.at({fatPtrTrue[0], fatPtrTrue[1]}) ==
            fatPtrs.at({fatPtrFalse[0], fatPtrFalse[1]})) &&
           "expected can narrow to be the same for both fatPtrT and fatPtrF");

    rewriter.replaceOpWithMultiple(selectOp, {{newBase, newOffset}});
    fatPtrs[{newBase, newOffset}] = fatPtrs.at({fatPtrTrue[0], fatPtrTrue[1]});

    return success();
  }
};

/// Rewrite result type only after both arms have been visited.
/// We contrive this to happen, even though DialectConversion does a PreOrder
/// walk, by checking for two attributes in the ConversionTarget
/// ("then_rewritten", and "else_rewritten").
class ConvertSCFIfOp : public PointerCanonicalizationPattern<scf::IfOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite_(scf::IfOp ifOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    assert(ifOp.thenYield()->hasAttr(kSCFIfOpYieldFatPtrOffsets) &&
           "expected then yield to report fat ptr indices");

    bool withElseRegion = ifOp.getNumRegions() > 1;

#ifndef NDEBUG
    if (withElseRegion) {
      assert(ifOp.thenYield().getOperandTypes() ==
                 ifOp.elseYield().getOperandTypes() &&
             "ifOp types must match in both arms");
      if (auto thenFatPtrIndxs = ifOp.thenYield()->getDiscardableAttr(
              kSCFIfOpYieldFatPtrOffsets)) {
        assert(ifOp.elseYield()->hasAttr(kSCFIfOpYieldFatPtrOffsets) &&
               "expected then yield to report fat ptr indices");
        auto elseFatPtrIndxs =
            ifOp.elseYield()->getDiscardableAttr(kSCFIfOpYieldFatPtrOffsets);
        assert(elseFatPtrIndxs &&
               "expected else fat ptr indices as well as then fat ptr indices");

        DenseI64ArrayAttr thenIdxs =
            llvm::dyn_cast<DenseI64ArrayAttr>(thenFatPtrIndxs);
        DenseI64ArrayAttr elseIdxs =
            llvm::dyn_cast<DenseI64ArrayAttr>(elseFatPtrIndxs);
        assert(bool(thenIdxs) && bool(elseIdxs) &&
               "expected else fat ptr index attrs to be DenseI64ArrayAttr");
        for (auto [i, j] :
             llvm::zip(thenIdxs.asArrayRef(), elseIdxs.asArrayRef())) {
          assert(i == j &&
                 "expected thenFatPtrIndxs and elseFatPtrIndxs to agree");
          assert(i < ifOp.thenYield().getNumOperands() &&
                 i + 1 < ifOp.thenYield().getNumOperands() &&
                 "expected idx to be within bounds of IfOp's results");
          Value thenFatPtrBase = ifOp.thenYield().getOperand(i);
          Value thenFatPtrOffset = ifOp.thenYield().getOperand(i + 1);
          Value elseFatPtrBase = ifOp.elseYield().getOperand(i);
          Value elseFatPtrOffset = ifOp.elseYield().getOperand(i + 1);
          assert((fatPtrs.at({thenFatPtrBase, thenFatPtrOffset}) ==
                  fatPtrs.at({elseFatPtrBase, elseFatPtrOffset})) &&
                 "expected then fat ptr canNarrow and else fat ptr canNarrow "
                 "to be equal");
        }
      }
    }
#endif

    auto newIfOp = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), ifOp.thenYield().getOperandTypes(), ifOp.getCondition(),
        withElseRegion);
    rewriter.inlineBlockBefore(ifOp.thenBlock(), newIfOp.thenBlock(),
                               newIfOp.thenBlock()->begin());
    if (withElseRegion)
      rewriter.inlineBlockBefore(ifOp.elseBlock(), newIfOp.elseBlock(),
                                 newIfOp.elseBlock()->begin());

    rewriter.replaceOpWithMultiple(ifOp, {newIfOp.getResults()});

    for (int64_t idx :
         llvm::cast<DenseI64ArrayAttr>(newIfOp.thenYield()->getDiscardableAttr(
                                           kSCFIfOpYieldFatPtrOffsets))
             .asArrayRef()) {
      Value thenFatPtrBase = newIfOp.thenYield().getOperand(idx);
      Value thenFatPtrOffset = newIfOp.thenYield().getOperand(idx + 1);
      fatPtrs[{newIfOp.getResult(idx), newIfOp.getResult(idx + 1)}] =
          fatPtrs.at({thenFatPtrBase, thenFatPtrOffset});
    }

    return success();
  }
};

/// Rewrite the non-cond operands and the signature of the dest bb.
class ConvertCFBranch : public PointerCanonicalizationPattern<cf::BranchOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite_(cf::BranchOp branchOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    ArrayRef<ValueRange> remappedDestOperands = adaptor.getDestOperands();
    SmallVector<Value> trueOperands = flattenValues(remappedDestOperands);

    rewriter.replaceOpWithNewOp<cf::BranchOp>(branchOp, branchOp.getDest(),
                                              trueOperands);
    convertSimpleBlockSignature(branchOp.getDest(), remappedDestOperands,
                                rewriter, fatPtrs);
    return success();
  }
};

/// Rewrite to expand(base, offset) -> base, expand(offset)
class ConvertExpandDims
    : public PointerCanonicalizationPattern<tt::ExpandDimsOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;
  LogicalResult
  matchAndRewrite_(tt::ExpandDimsOp expandOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    ValueRange remappedOperands = adaptor.getSrc();
    if (remappedOperands.size() != 2)
      return success();
    Value fatPtrBase = remappedOperands[0];
    if (!llvm::isa<tt::PointerType>(fatPtrBase.getType()))
      return rewriter.notifyMatchFailure(
          expandOp, "only scalar base currently supported");
    Value fatPtrOffset = remappedOperands[1];

    RankedTensorType result =
        llvm::cast<RankedTensorType>(expandOp->getResultTypes().front());
    if (!llvm::isa<tt::PointerType>(result.getElementType()))
      return rewriter.notifyMatchFailure(
          expandOp, "expected expand_dim result to be tensor of tt.ptr");

    RankedTensorType newResult = RankedTensorType::get(
        result.getShape(),
        llvm::cast<RankedTensorType>(fatPtrOffset.getType()).getElementType(),
        result.getEncoding());
    auto newOffset = rewriter.create<tt::ExpandDimsOp>(
        expandOp.getLoc(), newResult, fatPtrOffset, adaptor.getAxis());
    rewriter.replaceOpWithMultiple(expandOp, {{fatPtrBase, newOffset}});
    fatPtrs[{fatPtrBase, newOffset}] = fatPtrs.at({fatPtrBase, fatPtrOffset});

    return success();
  }
};

/// convert integer offset, keep base
class ConvertConvertLayoutOp
    : public PointerCanonicalizationPattern<tt::gpu::ConvertLayoutOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite_(tt::gpu::ConvertLayoutOp cvtOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    ValueRange remappedOperands = adaptor.getSrc();
    if (remappedOperands.size() != 2) {
      // some prior op materialized the fat ptr, e.g.:
      // %3 = tt.bitcast %2
      // %4 = tt.splat %3
      return success();
    }
    Value fatPtrBase = remappedOperands[0];
    Value fatPtrOffset = remappedOperands[1];
    if (!llvm::isa<tt::PointerType>(fatPtrBase.getType())) {
      return rewriter.notifyMatchFailure(cvtOp,
                                         "non tt.ptr base unimplemented");
    }
    auto offsetTensorTy = dyn_cast<RankedTensorType>(fatPtrOffset.getType());
    if (!offsetTensorTy) {
      return rewriter.notifyMatchFailure(
          cvtOp, "non RankedTensorType offset unimplemented");
    }

    RankedTensorType outType = cvtOp.getResult().getType();
    auto newOffsetType = RankedTensorType::get(outType.getShape(),
                                               offsetTensorTy.getElementType(),
                                               outType.getEncoding());
    tt::gpu::ConvertLayoutOp cvtOffset =
        rewriter.create<tt::gpu::ConvertLayoutOp>(cvtOp.getLoc(), newOffsetType,
                                                  fatPtrOffset);
    rewriter.replaceOpWithMultiple(cvtOp, {{fatPtrBase, cvtOffset}});
    fatPtrs[{fatPtrBase, cvtOffset}] = fatPtrs.at({fatPtrBase, fatPtrOffset});

    return success();
  }
};

template <typename SourceOp, int PtrLikeIdx = 0>
class MaterializeFatPointer : public PointerCanonicalizationPattern<SourceOp> {
public:
  using PointerCanonicalizationPattern<
      SourceOp>::PointerCanonicalizationPattern;

  LogicalResult matchAndRewrite_(
      SourceOp op,
      typename PointerCanonicalizationPattern<SourceOp>::OneToNOpAdaptor
          adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<tt::PointerType>(
            getElementTypeOrSelf(op->getOperandTypes()[PtrLikeIdx])))
      return rewriter.notifyMatchFailure(op,
                                         "expected operand to be pointer-like");
    ValueRange fatPtr = adaptor.getOperands()[PtrLikeIdx];
    if (fatPtr.size() != 2) {
      // some prior op materialized the fat ptr, e.g.:
      // %3 = tt.bitcast %2
      // %4 = tt.load %3
      return success();
    }

    Value fatPtrBase = fatPtr[0];
    Value fatPtrOffset = fatPtr[1];
    Location curLoc = op.getLoc();

    const FatPointers::FatPtrAttrs &fatPtrAttrs =
        this->fatPtrs.at({fatPtrBase, fatPtrOffset});
    SmallVector<Value> operands = op->getOperands();
    operands[PtrLikeIdx] = createTensorPointer(
        rewriter, fatPtrBase, fatPtrOffset, curLoc, fatPtrAttrs);

    if (op->getNumResults())
      rewriter.replaceOpWithNewOp<SourceOp>(
          op, op->getResultTypes(), ValueRange{operands}, op->getAttrs());
    else
      rewriter.replaceOpWithNewOp<SourceOp>(
          op, TypeRange{}, ValueRange{operands}, op->getAttrs());
    return success();
  }
};

template <typename SourceOp>
class MaterializeFatPointerVariadic
    : public PointerCanonicalizationPattern<SourceOp> {
public:
  using PointerCanonicalizationPattern<
      SourceOp>::PointerCanonicalizationPattern;

  LogicalResult matchAndRewrite_(
      SourceOp op,
      typename PointerCanonicalizationPattern<SourceOp>::OneToNOpAdaptor
          adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location curLoc = op.getLoc();
    SmallVector<Value> operands = op->getOperands();
    for (auto [i, maybeFatPtr] : llvm::enumerate(adaptor.getOperands())) {
      if (maybeFatPtr.size() != 2)
        continue;
      Value fatPtrBase = maybeFatPtr[0];
      Value fatPtrOffset = maybeFatPtr[1];

      const FatPointers::FatPtrAttrs &fatPtrAttrs =
          this->fatPtrs.at({fatPtrBase, fatPtrOffset});
      Value newPtr = createTensorPointer(rewriter, fatPtrBase, fatPtrOffset,
                                         curLoc, fatPtrAttrs);
      operands[i] = newPtr;
    }

    rewriter.replaceOpWithNewOp<SourceOp>(op, op->getResultTypes(),
                                          ValueRange{operands}, op->getAttrs());
    return success();
  }
};

static const std::string kInitFuncArgsRewritten =
    kPtrCanonPrefix + "init-func-ptr-args";
/// tt.func gets rewritten differently from all the other ops - the op itself is
/// not rewritten. What is rewritten are all tt.ptr args are rewritten (all
/// uses) to be %1 = unrealize_cast(%arg0: tt.ptr, c0: i32) -> tt.ptr. This
/// unrealized_cast is then (possibly) materialized in the second pass
/// (ConvertUnimplementedOpUnrealizedCasts) if it wasn't DCEd (via a user
/// extracting the tt.ptr and c0 operands).
struct InitFuncPtrArgs : OpRewritePattern<tt::FuncOp> {
  InitFuncPtrArgs(MLIRContext *context, FatPointers &fatPtrs)
      : OpRewritePattern(context, 0), fatPtrs(fatPtrs) {}

  LogicalResult matchAndRewrite(tt::FuncOp newOp,
                                PatternRewriter &rewriter) const override {
    if (newOp->hasAttr(kInitFuncArgsRewritten))
      return failure();

    int64_t bitness = 64;
    rewriter.setInsertionPointToStart(&newOp.getBody().front());
    for (auto [idx, arg] : llvm::enumerate(newOp.getArguments())) {
      // The pointer argument needs to be a scalar
      if (!isa<tt::PointerType>(arg.getType()))
        continue;
      if (auto pointerRangeAttr =
              newOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range"))
        bitness = pointerRangeAttr.getInt();
      Value zeroOffset =
          rewriter.create<arith::ConstantIntOp>(newOp.getLoc(), 0, bitness);
      auto dummyCast = rewriter.create<UnrealizedConversionCastOp>(
          arg.getLoc(), TypeRange{arg.getType()}, ValueRange{arg, zeroOffset});
      rewriter.replaceAllUsesExcept(arg, dummyCast.getResult(0), dummyCast);
      fatPtrs[{arg, zeroOffset}].canNarrow = true;
    }

    newOp->setDiscardableAttr(kInitFuncArgsRewritten, rewriter.getUnitAttr());
    return success();
  }

  FatPointers &fatPtrs;
};

/// No-op to make conversion framework happy.
class ConvertReturnOp : public PointerCanonicalizationPattern<tt::ReturnOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite_(tt::ReturnOp returnOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    auto returns = flattenValues(adaptor.getSrcs());
    rewriter.replaceOpWithNewOp<tt::ReturnOp>(returnOp, TypeRange{}, returns);
    return success();
  }
};

class ConvertFuncOpArgsUnrealizedCasts
    : public PointerCanonicalizationPattern<UnrealizedConversionCastOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite_(UnrealizedConversionCastOp castOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    if (castOp.use_empty()) {
      castOp->getParentOfType<tt::FuncOp>().emitRemark(
          "expected at least 1 use of unrealized_cast");
      return success();
    }
    // Exhaustive checking we're converting ONLY unrealized_casts inserted (by
    // the 1:N conversion) in ConvertFuncOp.
    ArrayRef<ValueRange> remappedOperands = adaptor.getOperands();
    if (remappedOperands.size() != 2 || remappedOperands[0].size() != 1 ||
        remappedOperands[1].size() != 1)
      return rewriter.notifyMatchFailure(
          castOp, "expected CastOp to have already been remapped");
    Value fatPtrBase = remappedOperands[0][0];
    if (!llvm::isa<BlockArgument>(fatPtrBase) ||
        !llvm::isa<tt::FuncOp>(fatPtrBase.getParentBlock()->getParentOp()) ||
        !llvm::isa<tt::PointerType>(fatPtrBase.getType()))
      return rewriter.notifyMatchFailure(
          castOp,
          "expected CastOp first operand to be tt.ptr block arg of tt.func");
    Value fatPtrOffset = remappedOperands[1][0];
    if (llvm::isa<BlockArgument>(fatPtrOffset) ||
        !llvm::isa<arith::ConstantOp>(fatPtrOffset.getDefiningOp()))
      return rewriter.notifyMatchFailure(
          castOp, "expected CastOp second operand to be arith.constant");
    OpFoldResult maybeScalar = getAsOpFoldResult(fatPtrOffset);
    auto maybeAttr = llvm::dyn_cast<mlir::Attribute>(maybeScalar);

    if (auto integerAttr =
            llvm::dyn_cast_or_null<mlir::IntegerAttr>(maybeAttr)) {
      if (integerAttr.getValue() == 0) {
        rewriter.replaceOpWithMultiple(castOp, {{fatPtrBase, fatPtrOffset}});
        return success();
      }
    }
    return rewriter.notifyMatchFailure(
        castOp,
        "expected CastOp second operand to be arith.constant with value 0");
  }
};

class ConvertUnimplementedOpUnrealizedCasts
    : public PointerCanonicalizationPattern<UnrealizedConversionCastOp> {
public:
  using PointerCanonicalizationPattern::PointerCanonicalizationPattern;

  LogicalResult
  matchAndRewrite_(UnrealizedConversionCastOp castOp, OneToNOpAdaptor adaptor,
                   ConversionPatternRewriter &rewriter) const override {
    if (castOp.use_empty()) {
      castOp.erase();
      return success();
    }
    ArrayRef<ValueRange> remappedOperands = adaptor.getOperands();
    if (remappedOperands.size() != 2)
      return rewriter.notifyMatchFailure(
          castOp, "expected CastOp to have already been remapped");
    Value fatPtrBase = remappedOperands[0][0];
    Value fatPtrOffset = remappedOperands[1][0];
    if (!llvm::isa<tt::PointerType>(fatPtrBase.getType()))
      return rewriter.notifyMatchFailure(castOp,
                                         "non tt.ptr base unimplemented");

    rewriter.setInsertionPointAfter(castOp);

    // shortcut if offset == 0, no need for addptr
    OpFoldResult maybeScalar = getAsOpFoldResult(fatPtrOffset);
    auto maybeAttr = llvm::dyn_cast<mlir::Attribute>(maybeScalar);
    if (auto integerAttr =
            llvm::dyn_cast_or_null<mlir::IntegerAttr>(maybeAttr)) {
      if (integerAttr.getValue() == 0) {
        rewriter.replaceAllUsesWith(castOp.getResult(0), fatPtrBase);
        rewriter.eraseOp(castOp);
        return success();
      }
    }

    const FatPointers::FatPtrAttrs &fatPtrAttrs =
        fatPtrs.at({fatPtrBase, fatPtrOffset});
    auto newPtr = createTensorPointer(rewriter, fatPtrBase, fatPtrOffset,
                                      castOp.getLoc(), fatPtrAttrs);
    rewriter.replaceAllUsesWith(newPtr, fatPtrBase);
    rewriter.eraseOp(castOp);
    return success();
  }
};

/// The pass structure/action is roughly:
///
/// 1. Perform an approximate sparse dataflow analysis to find all transitive
/// uses for `tt.func` args that are `tt.ptr`s; legalize only these ops;
/// 2. Rewrite all operations' `use`s and `result`s to be `(%baseptr,
/// %offsetptr)` using `ConversionPattern`s that takes the new
/// `OneToNOpAdaptor`, which automatically forwards both `%baseptr` and
/// `%offsetptr` through `adaptor.getOperands()`[^3];
/// 3. Clean up remaining `unrealized_casts` (currently only handling one
/// category of such remaining casts but can be extended to handle all; see
/// bullet 1 in TODOs).
class TritonAMDGPUCanonicalizePointersPass
    : public TritonAMDGPUCanonicalizePointersBase<
          TritonAMDGPUCanonicalizePointersPass> {
public:
  TritonAMDGPUCanonicalizePointersPass() = default;

  void runOnOperation() override;
};

/// Forward slice == transitive use
/// This is a port/adaptation of upstream's getForwardSliceImpl
/// that operates on values instead of ops so that we can track tt.ptr through
/// the operands/args of region ops like scf.for/scf.while.
/// It also handles scf.if in a special way beacuse scf.if does not have
/// operands.
///
/// TODO(max): this is still just a heuristic approximation to a "dataflow
/// analysis" that "understands" the relationship between each operands and
/// results for each op (i.e., whether fat ptrs are actually propagated).
static void getForwardSliceImpl(OpOperand *use, Operation *op,
                                SetVector<Operation *> *forwardSlice) {
  assert(use && op && "expected both use and op to be valid pointers");
  assert(use->getOwner() == op && "expected use's owner to be op");

  if (!llvm::isa<tt::PointerType>(getElementTypeOrSelf(use->get().getType())))
    return;

  // verbose because you can't construct <OpOperand*> from <OpOperand&>
  SmallVector<OpOperand *> nextUses;
  auto addUses = [&nextUses](const Value::use_range &uses) {
    for (auto &use : uses)
      nextUses.emplace_back(&use);
  };

  // all of this is necessary because both the LoopLikeInterface and
  // BrancOpInterface are bad...
  auto addBlockArgUses = [&use, &addUses](
                             const Block::BlockArgListType &blockArgs,
                             unsigned argOffset = 0, unsigned useOffset = 0) {
    for (auto arg : blockArgs) {
      if (arg.getArgNumber() - argOffset == use->getOperandNumber() - useOffset)
        addUses(arg.getUses());
    }
  };

  if (auto whileLoop = llvm::dyn_cast<scf::WhileOp>(op)) {
    addBlockArgUses(whileLoop.getBeforeArguments());
    addBlockArgUses(whileLoop.getAfterArguments());
  } else if (auto forLoop = llvm::dyn_cast<scf::ForOp>(op)) {
    addBlockArgUses(forLoop.getRegionIterArgs(), forLoop.getNumInductionVars(),
                    forLoop.getNumControlOperands());
  } else if (auto branchOp = llvm::dyn_cast<cf::BranchOp>(op)) {
    addBlockArgUses(branchOp.getDest()->getArguments());
  } else if (auto condBranchOp = llvm::dyn_cast<cf::CondBranchOp>(op)) {
    // the 0th operand of cf.cond_br is the condition
    addBlockArgUses(condBranchOp.getTrueDest()->getArguments(), /*argOffset*/ 0,
                    /*useOffset*/ 1);
    addBlockArgUses(condBranchOp.getFalseDest()->getArguments(),
                    /*argOffset*/ 0, /*useOffset*/ 1);
  } else if (auto yield = llvm::dyn_cast<scf::YieldOp>(op)) {
    forwardSlice->insert(yield);
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(yield->getParentOp()))
      op = ifOp;
  }

  for (auto result : op->getResults())
    addUses(result.getUses());

  for (OpOperand *nextUse : nextUses) {
    auto owner = nextUse->getOwner();
    getForwardSliceImpl(nextUse, owner, forwardSlice);
  }

  forwardSlice->insert(op);
}

void TritonAMDGPUCanonicalizePointersPass::runOnOperation() {
  LLVM_DEBUG({
    llvm::dbgs() << "before tritonamdgpu-canonicalize-pointers\n";
    getOperation()->getParentOfType<ModuleOp>()->dump();
    llvm::dbgs() << "\n";
  });

  auto func = getOperation();

  // Skip the pass as a workaround as if op with multiple results are not
  // supported yet.
  bool hasIfOpWithMultipleResults =
      func.walk([&](scf::IfOp ifOp) {
            if (ifOp.getNumResults() > 1) {
              for (auto result : ifOp.getResultTypes()) {
                if (llvm::isa<tt::PointerType>(result)) {
                  return WalkResult::interrupt();
                }
              }
            }
            return WalkResult::advance();
          })
          .wasInterrupted();
  if (hasIfOpWithMultipleResults)
    return;

  FatPointers fatPrs;
  PatternRewriter rewriter(&getContext());
  // Convert tt.func; %1 = unrealize_cast(%arg0: tt.ptr, c0: i32) -> tt.ptr
  InitFuncPtrArgs pat(&getContext(), fatPrs);
  if (failed(pat.matchAndRewrite(func, rewriter)))
    return signalPassFailure();

  llvm::SetVector<Operation *> opsToRewrite;
  for (auto arg : func.getArguments()) {
    if (llvm::isa<tt::PointerType>(arg.getType())) {
      // NB: reusing the same SetVector invalidates the topo order implied by
      // getForwardSlice
      for (auto &use : arg.getUses())
        getForwardSliceImpl(&use, use.getOwner(), &opsToRewrite);
    }
  }

  ConversionConfig config;
  config.buildMaterializations = false;
  ConversionTarget target(getContext());
  auto isLegal = [&opsToRewrite](Operation *op) {
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
      // This is the only hack in the entire pass; on first traversal,
      // `scf.if` will be walked over, but we do not want to rewrite it yet
      // because the `yields` in the then/else regions haven't been rewritten
      // yet (and those `yields` tell us the final result types of the
      // `scf.if`). Therefore, we check for these attributes and if they're
      // absent then the `scf.if` is legal. Once both `yields` have been
      // rewritten (the corresponding attributes have been added), we report the
      // `scf.if` as illegal, and it will be rewritten (the pattern will fire).
      return !(ifOp->hasAttr(kSCFThenRewrittenAttr) &&
               ifOp->hasAttr(kSCFElseRewrittenAttr));
    }
    return !opsToRewrite.contains(op);
  };

  target.addDynamicallyLegalDialect<tt::TritonDialect>(isLegal);
  target.addDynamicallyLegalDialect<triton::gpu::TritonGPUDialect>(isLegal);
  target.addDynamicallyLegalDialect<scf::SCFDialect>(isLegal);
  target.addDynamicallyLegalDialect<cf::ControlFlowDialect>(isLegal);
  target.addDynamicallyLegalDialect<arith::ArithDialect>(isLegal);

  // Rewrite the rest of the ops.
  // Note we *do not* declare unrealized_cast an illegal op here in order that
  // the whole conversion passes, even if there are tt ops that we do not
  // currently support (their operands will be handled by
  // ConvertUnimplementedOpUnrealizedCasts below). Note we *do* add
  // ConvertFuncOpArgsUnrealizedCasts because that is necessary for
  // "initializing" the chain of fat pointers starting from tt.func tt.ptr args.
  RewritePatternSet patterns(&getContext());
  patterns.add<
      ConvertFuncOpArgsUnrealizedCasts, ConvertBroadcastOp, ConvertSplatOp,
      ConvertConvertLayoutOp, ConvertAddPtrOp,
      MaterializeFatPointer<tt::AtomicCASOp>,
      MaterializeFatPointer<tt::AtomicRMWOp>,
      MaterializeFatPointer<tt::BitcastOp>, MaterializeFatPointer<tt::LoadOp>,
      MaterializeFatPointer<triton::gpu::AsyncCopyGlobalToLocalOp>,
      MaterializeFatPointer<tt::PtrToIntOp>, MaterializeFatPointer<tt::StoreOp>,
      MaterializeFatPointerVariadic<tt::CallOp>,
      MaterializeFatPointerVariadic<tt::PrintOp>, ConvertSCFForOp,
      ConvertExpandDims, ConvertSCFYieldOp, ConvertSCFIfOp,
      ConvertSCFConditionOp, ConvertSCFWhileOp, ConvertCFCondBranch,
      ConvertCFBranch, ConvertArithSelectOp, ConvertReturnOp>(
      patterns.getContext(), opsToRewrite, fatPrs);
  if (failed(applyPartialConversion(func, target, std::move(patterns), config)))
    return signalPassFailure();

  // Rewrite any lingering unrealized_casts that *should* only be the result of
  // unsupported ops.
  target.addIllegalOp<UnrealizedConversionCastOp>();
  patterns.clear();
  patterns.add<ConvertUnimplementedOpUnrealizedCasts>(patterns.getContext(),
                                                      opsToRewrite, fatPrs);
  if (failed(applyPartialConversion(func, target, std::move(patterns), config)))
    return signalPassFailure();

  func->walk<WalkOrder::PreOrder>([](Operation *op) {
    for (auto attr : op->getDiscardableAttrs()) {
      if (attr.getName().strref().starts_with(kPtrCanonPrefix))
        op->removeDiscardableAttr(attr.getName());
    }
  });
}

std::unique_ptr<Pass> mlir::createTritonAMDGPUCanonicalizePointersPass() {
  return std::make_unique<TritonAMDGPUCanonicalizePointersPass>();
}
