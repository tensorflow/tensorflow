#ifndef TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {

namespace gpu {

Type getElementType(Value value);

class MultipleOperandsRange
    : public iterator_range<SmallVector<SmallVector<Value>>::iterator> {
  using ContainerT = SmallVector<SmallVector<Value>>;

public:
  using iterator_range<ContainerT::iterator>::iterator_range;
  ContainerT::reference operator[](ContainerT::size_type idx) {
    return begin()[idx];
  }
  ContainerT::const_reference operator[](ContainerT::size_type idx) const {
    return begin()[idx];
  }
  ContainerT::size_type size() const { return end() - begin(); }
};

// Base pattern for elementwise conversion using ConcreteT. Unpacks individual
// elements from a `!llvm.struct` via `llvm.extactvalue`, calls
// ConcreteT::createDestOps on each element, and packs them back into an
// `!llvm.struct` using `llvm.insertvalue`.
//
// Also supports processing the inputs in a vectorized form by consuming and
// producing multiple operand sets in ConcreteT::createDestOps.
template <typename SourceOp, typename ConcreteT>
class ElementwiseOpConversionBase : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpConversionBase(
      LLVMTypeConverter &typeConverter,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        axisAnalysisPass(axisAnalysisPass) {}

  // Try to deduplicate the resultVals based on the
  // constancy properties of the result discovered by
  // the axis analysis pass. If possible, redundant
  // computation is eliminated.
  SmallVector<Value> maybeDeduplicate(SourceOp op,
                                      SmallVector<Value> resultVals) const {
    if (!isMemoryEffectFree(op))
      // the op has side effects: can't dedup
      return resultVals;
    SmallVector<Value> results = op->getResults();
    if (results.size() == 0 || results.size() > 1)
      // there must be exactly 1 result
      return resultVals;
    Value result = results[0];
    Type type = result.getType();
    if (!type)
      return resultVals;
    RankedTensorType rtType = dyn_cast<RankedTensorType>(type);
    if (!rtType)
      // the result must be a tensor
      return resultVals;
    Attribute encoding = rtType.getEncoding();
    if (!encoding)
      // encoding not available
      return resultVals;
    Attribute baseEncoding = encoding;
    if (isa<AMDMfmaEncodingAttr>(baseEncoding) ||
        isa<AMDWmmaEncodingAttr>(baseEncoding))
      // TODO: this logic seems incorrect for mfma and wmma layout. Skip for
      // now. We saw mismatches for some flash-attention and dot tests on AMD
      // backend. Note that this logic works for sliced layout whose parent is
      // mfma layout. Therefore, this is not combined with the following check.
      return resultVals;
    while (auto sliced = dyn_cast<SliceEncodingAttr>(baseEncoding))
      baseEncoding = sliced.getParent();
    if (isa<LinearEncodingAttr, DotOperandEncodingAttr>(baseEncoding)) {
      // TODO: this logic seems incorrect for mma layout. Skip for now.
      // The following test crashes and some other miscompile:
      // test_core::test_fp8_dot_acc
      return resultVals;
    }

    SmallVector<unsigned> elemsPerThread = getElemsPerThread(rtType);
    int rank = elemsPerThread.size();
    if (product<unsigned>(elemsPerThread) != resultVals.size())
      return resultVals;
    AxisInfo *axisInfo = axisAnalysisPass.getAxisInfo(result);
    if (!axisInfo)
      // axis info (e.g., constancy) not available
      return resultVals;
    SmallVector<unsigned> contigPerThread = getContigPerThread(rtType);
    if (rank != contigPerThread.size())
      return resultVals;

    SmallVector<int64_t> constancy = axisInfo->getConstancy();
    if (rank != constancy.size())
      return resultVals;
    bool hasConstancy = false;
    for (int i = 0; i < rank; ++i) {
      if (constancy[i] > contigPerThread[i]) {
        if (constancy[i] % contigPerThread[i] != 0)
          // constancy is not evenly covered by contigPerThread
          return resultVals;
        // can't move the values across different
        // "contigPerThread"-sized blocks
        constancy[i] = contigPerThread[i];
      }
      if (elemsPerThread[i] < 1 || constancy[i] < 1)
        return resultVals;
      if (!(elemsPerThread[i] % constancy[i] == 0 ||
            constancy[i] % elemsPerThread[i] == 0))
        // either the constancy along each dimension must fit
        // into the elemsPerThread or the other way around
        return resultVals;
      if (constancy[i] > 1)
        hasConstancy = true;
    }
    if (!hasConstancy)
      // nothing to deduplicate
      return resultVals;

    if (rank > 1) {
      // reorder the shape and constancy vectors by the axis order:
      // from the fastest-changing to the smallest-changing axis
      SmallVector<unsigned> order = getOrder(rtType);
      if (rank != order.size())
        return resultVals;
      elemsPerThread = applyPermutation(elemsPerThread, order);
      constancy = applyPermutation(constancy, order);
    }

    SmallVector<unsigned> strides(rank, 1);
    for (int i = 1; i < rank; ++i) {
      strides[i] = strides[i - 1] * elemsPerThread[i - 1];
    }
    SmallVector<Value> dedupResultVals;
    dedupResultVals.reserve(resultVals.size());
    for (int i = 0; i < resultVals.size(); ++i) {
      // each coordinate of the orig_idx is "coarsened" using the
      // constancy along this dimension: the resulting dedup_idx
      // points to the reused value in the original resultsVal
      int orig_idx = i;
      int dedup_idx = 0;
      for (int j = 0; j < rank; ++j) {
        int coord_j = orig_idx % elemsPerThread[j];
        dedup_idx += (coord_j / constancy[j] * constancy[j]) * strides[j];
        orig_idx /= elemsPerThread[j];
      }
      dedupResultVals.push_back(resultVals[dedup_idx]);
    }

    return dedupResultVals;
  }
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();
    // element type
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<SmallVector<Value>> allOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      allOperands.resize(subOperands.size());
      for (auto v : llvm::enumerate(subOperands))
        allOperands[v.index()].push_back(v.value());
    }
    if (allOperands.size() == 0)
      allOperands.push_back({});

    SmallVector<Value> resultVals;
    for (auto it = allOperands.begin(), end = allOperands.end(); it != end;) {
      auto curr = static_cast<const ConcreteT *>(this)->createDestOps(
          op, adaptor, rewriter, elemTy, MultipleOperandsRange(it, end), loc);
      if (curr.size() == 0)
        return failure();
      for (auto v : curr) {
        if (!static_cast<bool>(v))
          return failure();
        resultVals.push_back(v);
      }
      it += curr.size();
    }
    resultVals = maybeDeduplicate(op, resultVals);
    Value view = packLLElements(loc, this->getTypeConverter(), resultVals,
                                rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

// Trivial case where we map elementwise to an existing LLVM operator
template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
};

} // namespace gpu

} // namespace mlir::triton
#endif
