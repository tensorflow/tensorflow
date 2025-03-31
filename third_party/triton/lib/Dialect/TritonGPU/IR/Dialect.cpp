#include "triton/Dialect/Triton/IR/Dialect.h"

#include <cstdint>
#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"

// Include TableGen'erated code
#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"
#include "triton/Dialect/TritonGPU/IR/TypeInterfaces.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Utility
namespace mlir {
namespace triton {
namespace gpu {

LinearEncodingAttr TritonGPUDialect::toLinearEncoding(ArrayRef<int64_t> shape,
                                                      Attribute layout) {
  CacheKey key{std::vector<int64_t>(shape.begin(), shape.end()), layout};
  if (auto result = leCache.get(key)) {
    return *result;
  }
  auto linearLayout = toLinearLayout(shape, layout);
  auto linearEncoding =
      LinearEncodingAttr::get(layout.getContext(), std::move(linearLayout));
  leCache.set(key, linearEncoding);
  return linearEncoding;
}

LinearEncodingAttr toLinearEncoding(RankedTensorType type) {
  return toLinearEncoding(type.getEncoding(), type.getShape());
}

LinearEncodingAttr toLinearEncoding(Attribute layout, ArrayRef<int64_t> shape) {
  auto *ctx = layout.getContext();
  return ctx->getLoadedDialect<TritonGPUDialect>()->toLinearEncoding(shape,
                                                                     layout);
}

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape) {
  auto distributedEncoding = mlir::cast<DistributedEncodingTrait>(layout);
  return distributedEncoding.getTotalElemsPerThread(shape);
}

SmallVector<unsigned> getElemsPerThread(Attribute layout,
                                        ArrayRef<int64_t> shape) {
  auto distributedEncoding = mlir::cast<DistributedEncodingTrait>(layout);
  return distributedEncoding.getElemsPerThread(shape);
}

SmallVector<unsigned> getElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<triton::PointerType>(type))
    return SmallVector<unsigned>(1, 1);
  auto tensorType = cast<RankedTensorType>(type);
  return getElemsPerThread(tensorType.getEncoding(), tensorType.getShape());
}

unsigned getTotalElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<triton::PointerType>(type))
    return 1;
  auto tensorType = cast<RankedTensorType>(type);
  return getTotalElemsPerThread(tensorType.getEncoding(),
                                tensorType.getShape());
}

SmallVector<unsigned> getThreadsPerWarp(Attribute layout,
                                        ArrayRef<int64_t> shape) {
  return toLinearEncoding(layout, shape).getThreadsPerWarp();
}

SmallVector<unsigned> getWarpsPerCTA(Attribute layout,
                                     ArrayRef<int64_t> shape) {
  return toLinearEncoding(layout, shape).getWarpsPerCTA();
}

SmallVector<unsigned> getContigPerThread(RankedTensorType type) {
  return toLinearEncoding(type).getContigPerThread();
}

SmallVector<unsigned> getShapePerCTATile(RankedTensorType type) {
  return toLinearEncoding(type).getShapePerCTATile();
}

bool isExpensiveView(Type srcType, Type dstType) {
  auto tensorSrcType = cast<RankedTensorType>(srcType);
  auto tensorDstType = cast<RankedTensorType>(dstType);
  auto llSrc =
      toLinearLayout(tensorSrcType.getShape(), tensorSrcType.getEncoding());
  auto llDst =
      toLinearLayout(tensorDstType.getShape(), tensorDstType.getEncoding());
  // In case there are replicated value we need to make sure the new and old
  // layout have matching masks.
  for (auto [srcMask, dstMask] :
       llvm::zip(llSrc.getFreeVariableMasks(), llDst.getFreeVariableMasks())) {
    assert(srcMask.first == dstMask.first);
    if (srcMask.second != dstMask.second)
      return true;
  }
  return getTotalElemsPerThread(srcType) != getTotalElemsPerThread(dstType);
}

/* Utility function used by get.*Order methods of SliceEncodingAttr.
 * Erase dim and decrease all values larger than dim by 1.
 * Example:    order = [0, 2, 4, 3, 1], dim = 2
 *          resOrder = [0,    3, 2, 1]
 */
static SmallVector<unsigned> eraseOrder(ArrayRef<unsigned> order,
                                        unsigned dim) {
  unsigned rank = order.size();
  assert(dim < rank && "Invalid dim to erase");
  SmallVector<unsigned> resOrder;
  for (unsigned i : order)
    if (i < dim)
      resOrder.push_back(i);
    else if (i > dim)
      resOrder.push_back(i - 1);
  return resOrder;
}

SmallVector<unsigned> getMatrixOrder(unsigned rank, bool rowMajor) {
  // Return the order that represents that the batch is in row-major or
  // column-major order for a batch of matrices of shape [*, m, n] with
  // len(shape) == rank.
  assert(rank >= 2);
  SmallVector<unsigned> order(rank);
  std::iota(order.rbegin(), order.rend(), 0);
  if (!rowMajor) {
    std::swap(order[0], order[1]);
  }
  return order;
}

SmallVector<unsigned> getOrderForDotOperand(unsigned opIdx, unsigned rank,
                                            bool kContig) {
  // kContig: if true, the matrix is fastest-running on k,
  //         otherwise it is on m (resp. n)
  // opIdx=0: [batch, m, k] if rank == 3 else [m, k]
  // opIdx=1: [batch, k, n] if rank == 3 else [k, n]
  // batch (if rank == 3) is always the slowest running dimension
  assert(rank == 2 || rank == 3);
  assert(opIdx == 0 || opIdx == 1);
  auto rowMajor = bool(opIdx) != kContig;
  return getMatrixOrder(rank, rowMajor);
}

SmallVector<unsigned> getRepOrder(RankedTensorType type) {
  auto layout = type.getEncoding();
  if (auto distributedLayout = mlir::dyn_cast<DistributedEncodingTrait>(layout))
    return distributedLayout.getRepOrder();
  else
    llvm::report_fatal_error("Unimplemented usage of getRepOrder");
  return {};
}

// Legacy impl for now
// This one's not terribly bad as we don't broadcast ShareEncodings
SmallVector<unsigned> getOrder(SharedEncodingTrait layout,
                               ArrayRef<int64_t> shape) {
  if (auto swizzledLayout =
          mlir::dyn_cast<SwizzledSharedEncodingAttr>(layout)) {
    return llvm::to_vector(swizzledLayout.getOrder());
  }
  if (auto sharedLayout = mlir::dyn_cast<NVMMASharedEncodingAttr>(layout)) {
    return sharedLayout.getOrder();
  }
  if (auto sharedLayout =
          mlir::dyn_cast<AMDRotatingSharedEncodingAttr>(layout)) {
    return llvm::to_vector(sharedLayout.getOrder());
  }
  llvm::report_fatal_error("Unimplemented usage of getOrder for MemDescType");
  return {};
}

SmallVector<unsigned> getOrder(DistributedEncodingTrait layout,
                               ArrayRef<int64_t> shape) {
  return toLinearEncoding(layout, shape).getOrder();
}

SmallVector<unsigned> getOrderForMemory(DistributedEncodingTrait layout,
                                        ArrayRef<int64_t> shape) {
  auto linear = toLinearEncoding(layout, shape);
  auto order = linear.getOrder();
  auto threadOrder = linear.getThreadOrder();
  if (order == threadOrder) {
    return order;
  }
  // Heuristic:
  // If the element contiguity does not align with the thread order
  // because the thread order dimension has contiguity of 1---meaning that
  // the order position of this dimension is irrelevant---we prefer
  // to use the thread order for the memory layout
  auto contig = linear.getElemsPerThread(shape);
  if (contig[threadOrder[0]] == 1) {
    return threadOrder;
  }
  return order;
}

SmallVector<unsigned> getThreadOrder(DistributedEncodingTrait layout,
                                     ArrayRef<int64_t> shape) {
  return toLinearEncoding(layout, shape).getThreadOrder();
}

SmallVector<unsigned> getWarpOrder(DistributedEncodingTrait layout,
                                   ArrayRef<int64_t> shape) {
  return toLinearEncoding(layout, shape).getWarpOrder();
}

CTALayoutAttr getCTALayout(Attribute layout) {
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout)) {
    return CTALayoutAttr::get(layout.getContext(), getCTAsPerCGA(ttgLayout),
                              getCTASplitNum(ttgLayout),
                              getCTAOrder(ttgLayout));
  }
  llvm::report_fatal_error("Unimplemented usage of getCTALayout");
  return {};
}

SmallVector<unsigned> getCTAsPerCGA(Attribute layout) {
  ArrayRef<unsigned> ref;
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout))
    return ttgLayout.getCTAsPerCGA();
  else
    llvm::report_fatal_error("Unimplemented usage of getCTAsPerCGA");
  return SmallVector<unsigned>(ref.begin(), ref.end());
}

SmallVector<unsigned> getCTASplitNum(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout)) {
    return ttgLayout.getCTASplitNum();
  } else if (auto tmemLayout =
                 mlir::dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
                     layout)) {
    res.resize(2);
    res[0] = tmemLayout.getCTASplitM();
    res[1] = tmemLayout.getCTASplitN();
  } else if (auto tmemScaleLayout = mlir::dyn_cast<
                 triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(layout)) {
    res.resize(2);
    res[0] = tmemScaleLayout.getCTASplitM();
    res[1] = tmemScaleLayout.getCTASplitN();
  } else {
    assert(false && "Unimplemented usage of getCTASplitNum");
  }
  return res;
}

SmallVector<unsigned> getCTAOrder(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout)) {
    res = ttgLayout.getCTAOrder();
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTAOrder");
  }
  return res;
}

SmallVector<int64_t> getShapePerCTA(ArrayRef<unsigned> CTASplitNum,
                                    ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  SmallVector<int64_t> shapePerCTA(rank);
  for (unsigned i = 0; i < rank; ++i) {
    unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
    shapePerCTA[i] = shape[i] / splitNum;
  }
  return shapePerCTA;
}

SmallVector<int64_t> getShapePerCTA(Attribute layout, ArrayRef<int64_t> shape) {
  if (mlir::isa<SharedEncodingTrait>(layout)) {
    // Special logic for pipeline pass, where shape is 3D and CTALayout is 2D.
    // The first dim of shape is numStages. This is a work around, otherwise
    // too many places would have to be modified in pipeline pass. Maybe we
    // need to refactor this logic in the future.
    auto CTASplitNum = cast<LayoutEncodingTrait>(layout).getCTASplitNum();
    if (shape.size() == CTASplitNum.size() + 1) {
      auto res = getShapePerCTA(CTASplitNum, shape.drop_front());
      res.insert(res.begin(), shape.front());
      return res;
    }
  }
  SmallVector<unsigned> splitNum = getCTASplitNum(layout);
  if (auto tmem = dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(layout)) {
    if (shape.size() > splitNum.size()) {
      splitNum.insert(splitNum.begin(), shape.size() - splitNum.size(), 1);
    }
  }
  return getShapePerCTA(splitNum, shape);
}

SmallVector<int64_t> getAllocationShapePerCTA(Attribute layout,
                                              ArrayRef<int64_t> shapeLogical) {
  SmallVector<int64_t> shape(shapeLogical);
  if (auto sharedMMALayout = mlir::dyn_cast<NVMMASharedEncodingAttr>(layout)) {
    if (sharedMMALayout.getFp4Padded()) {
      auto packedAxis = getOrder(sharedMMALayout, shapeLogical)[0];
      if (shape.size() == 3) {
        // Take into account multi buffering
        shape[1 + packedAxis] *= 2;
      } else {
        shape[packedAxis] *= 2;
      }
    }
  }
  return getShapePerCTA(layout, shape);
}

SmallVector<int64_t> getShapePerCTA(Type type) {
  auto tensorType = cast<TensorOrMemDesc>(type);
  return getShapePerCTA(tensorType.getEncoding(), tensorType.getShape());
}

SmallVector<int64_t> getAllocationShapePerCTA(Type type) {
  auto tensorType = cast<TensorOrMemDesc>(type);
  return getAllocationShapePerCTA(tensorType.getEncoding(),
                                  tensorType.getShape());
}

unsigned getNumCTAs(Attribute layout) {
  return product<unsigned>(getCTAsPerCGA(layout));
}

bool isExpensiveCat(CatOp cat, Attribute targetEncoding) {
  // If the new elements per thread is less than the old one, we will need to
  // do convert encoding that goes through shared memory anyway. So we
  // consider it as expensive.
  RankedTensorType tensorTy = cat.getType();
  auto totalElemsPerThread = gpu::getTotalElemsPerThread(tensorTy);
  auto shape = tensorTy.getShape();
  auto newTotalElemsPerThread =
      gpu::getTotalElemsPerThread(targetEncoding, shape);
  return newTotalElemsPerThread < totalElemsPerThread;
}

LogicalResult CTALayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<unsigned> CTAsPerCGA,
    ArrayRef<unsigned> CTASplitNum, ArrayRef<unsigned> CTAOrder) {
  if (CTAsPerCGA.size() != CTASplitNum.size() ||
      CTASplitNum.size() != CTAOrder.size()) {
    return emitError() << "CTAsPerCGA, CTASplitNum, and CTAOrder must all have "
                          "the same rank.";
  }

  if (!isPermutationOfIota(CTAOrder)) {
    return emitError()
           << "CTAOrder must be a permutation of 0..(rank-1), but was ["
           << CTAOrder << "]";
  }

  if (llvm::any_of(CTAsPerCGA, [](unsigned x) { return x == 0; })) {
    return emitError() << "Every element in CTAsPerCGA must be greater than 0.";
  }

  if (llvm::any_of(CTASplitNum, [](unsigned x) { return x == 0; })) {
    return emitError()
           << "Every element in CTASplitNum must be greater than 0.";
  }

  return success();
}

LogicalResult
BlockedEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            ArrayRef<unsigned> sizePerThread,
                            ArrayRef<unsigned> threadsPerWarp,
                            ArrayRef<unsigned> warpsPerCTA,
                            ArrayRef<unsigned> order, CTALayoutAttr CTALayout) {
  if (sizePerThread.size() != threadsPerWarp.size() ||
      threadsPerWarp.size() != warpsPerCTA.size() ||
      warpsPerCTA.size() != order.size()) {
    return emitError() << "sizePerThread, threadsPerWarp, warpsPerCTA, and "
                          "order must all have the same rank.";
  }

  // Empty CTALayout is allowed, but if it's present its rank must match the
  // BlockedEncodingAttr's rank.
  if (CTALayout.getCTASplitNum().size() != 0 &&
      sizePerThread.size() != CTALayout.getCTASplitNum().size()) {
    return emitError() << "BlockedEncodingAttr and CTALayout's fields must "
                          "have the same rank.";
  }
  if (!isPermutationOfIota(order)) {
    return emitError()
           << "order must be a permutation of 0..(rank-1), but was [" << order
           << "]";
  }
  return success();
}

// 1 element per thread
// order = reverse(arange(rank))
triton::gpu::BlockedEncodingAttr
getDefaultBlockedEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          int numWarps, int threadsPerWarp, int numCTAs) {
  int rank = shape.size();
  llvm::SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  llvm::SmallVector<unsigned> sizePerThread(rank, 1);
  triton::gpu::BlockedEncodingAttr encoding =
      triton::gpu::BlockedEncodingAttr::get(context, shape, sizePerThread,
                                            order, numWarps, threadsPerWarp,
                                            numCTAs);
  return encoding;
}

LogicalResult tryJoinOnAxis(MLIRContext *ctx, const LinearLayout &inLl,
                            LinearLayout &outLl, bool fwdInference, int axis,
                            std::optional<Location> loc) {
  auto kRegister = StringAttr::get(ctx, "register");
  auto outDims = llvm::to_vector(inLl.getOutDimNames());
  if (fwdInference) {
    auto split = LinearLayout::identity1D(2, kRegister, outDims[axis]);
    outLl = split * inLl;
  } else {
    // TODO This requires a division algorithm!
    // Implement manually ll.divideLeft(split)
    auto contiguousElems =
        LinearEncodingAttr::get(ctx, inLl).getContigPerThread();
    if (contiguousElems[axis] > 1) {
      LinearLayout::BasesT newBases;
      for (const auto &basesDim : inLl.getBases()) {
        std::vector<std::vector<int32_t>> newBasesDim;
        for (auto base : basesDim.second) {
          if (base[axis] == 1) {
            continue;
          }
          base[axis] /= 2;
          newBasesDim.push_back(std::move(base));
        }
        newBases.insert({basesDim.first, std::move(newBasesDim)});
      }
      outLl = LinearLayout(std::move(newBases), std::move(outDims));
    } else {
      return emitOptionalError(loc,
                               "Fp4ToFpOp/SplitOp requires at least 2 elements "
                               "per thread in the axis/last dimension");
    }
  }
  return success();
}

} // namespace gpu
} // namespace triton
} // namespace mlir

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

static LogicalResult parseBoolAttrValue(AsmParser &parser, Attribute attr,
                                        bool &value, StringRef desc) {
  auto boolAttr = mlir::dyn_cast<BoolAttr>(attr);
  if (!boolAttr) {
    parser.emitError(parser.getNameLoc(), "expected an bool type in ") << desc;
    return failure();
  }
  value = boolAttr.getValue();
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

static LogicalResult parseUInt(AsmParser &parser, const NamedAttribute &attr,
                               unsigned &value, StringRef desc) {
  return parseIntAttrValue(parser, attr.getValue(), value, desc);
};

static LogicalResult parseBool(AsmParser &parser, const NamedAttribute &attr,
                               bool &value, StringRef desc) {
  return parseBoolAttrValue(parser, attr.getValue(), value, desc);
};

// Print the CTALayout if it's not equal to the default.
static void maybePrintCTALayout(mlir::MLIRContext *context,
                                mlir::AsmPrinter &printer, CTALayoutAttr layout,
                                unsigned rank) {
  if (layout != CTALayoutAttr::getDefault(context, rank)) {
    printer << ", CTAsPerCGA = [" << ArrayRef(layout.getCTAsPerCGA()) << "]"
            << ", CTASplitNum = [" << ArrayRef(layout.getCTASplitNum()) << "]"
            << ", CTAOrder = [" << ArrayRef(layout.getCTAOrder()) << "]";
  }
}

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonGPU/IR/AttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/AttrDefs.cpp.inc"

// If we only had BlockedEncodingAttr, we could simply return ArrayRefs here.
// But we need to have a consistent interface with e.g. SliceEncodingAttr, which
// computes some of these fields.
SmallVector<unsigned> BlockedEncodingAttr::getRepOrder() const {
  return SmallVector<unsigned>(getOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> BlockedEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}

template <class T>
SmallVector<T> SliceEncodingAttr::paddedShape(ArrayRef<T> shape) const {
  size_t rank = shape.size();
  unsigned dim = getDim();
  SmallVector<T> retShape(rank + 1);
  for (unsigned d = 0; d < rank + 1; ++d) {
    if (d < dim)
      retShape[d] = shape[d];
    else if (d == dim)
      retShape[d] = 1;
    else
      retShape[d] = shape[d - 1];
  }
  return retShape;
}
template SmallVector<unsigned>
SliceEncodingAttr::paddedShape<unsigned>(ArrayRef<unsigned> shape) const;
template SmallVector<int64_t>
SliceEncodingAttr::paddedShape<int64_t>(ArrayRef<int64_t> shape) const;
SmallVector<unsigned> SliceEncodingAttr::getRepOrder() const {
  auto parentRepOrder = getParent().getRepOrder();
  return eraseOrder(parentRepOrder, getDim());
}
SmallVector<unsigned> SliceEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res = ::getCTASplitNum(getParent());
  res.erase(res.begin() + getDim());
  return res;
}
SmallVector<unsigned> SliceEncodingAttr::getCTAOrder() const {
  auto parentCTAOrder = ::getCTAOrder(getParent());
  return eraseOrder(parentCTAOrder, getDim());
}
SmallVector<unsigned> SliceEncodingAttr::getCTAsPerCGA() const {
  auto parentCTAsPerCGA = ::getCTAsPerCGA(getParent());
  if (parentCTAsPerCGA[getDim()] == 1) {
    parentCTAsPerCGA.erase(parentCTAsPerCGA.begin() + getDim());
    return parentCTAsPerCGA;
  }
  /* For getCTAsPerCGA of a slice layout, we have two choices:
   * (1) Return CTAsPerCGA of its parent. This is not a perfect solution
   * because the rank of the returned CTAsPerCGA does not match the rank of
   * tensorShape.
   * (2) Get CTAsPerCGA of its parent and erase the sliced dim. This is not a
   * perfect solution because the product of the returned CTAsPerCGA might not
   * match numCTAs.
   * To avoid introducing inconsistencies to the shape and
   * layout system, the usage of directly getting CTAsPerCGA of a slice layout
   * in which the sliced dim is not 1 is banned. You should always consider
   * slice layout as a special case and use getCTAsPerCGA(layout.getParent())
   * in the branch where layout is an instance of SliceEncodingAttr. This is
   * inconvenient but safe.
   */
  llvm::report_fatal_error(
      "getCTAsPerCGA for SliceEncodingAttr is not well-defined");
}

// Wmma encoding

int32_t SwizzledSharedEncodingAttr::getAlignment() const { return 16; }

SmallVector<unsigned> SwizzledSharedEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> SwizzledSharedEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> SwizzledSharedEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}

int32_t NVMMASharedEncodingAttr::getAlignment() const { return 1024; }

SmallVector<unsigned> NVMMASharedEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> NVMMASharedEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> NVMMASharedEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}

int32_t AMDRotatingSharedEncodingAttr::getAlignment() const { return 16; }

SmallVector<unsigned> AMDRotatingSharedEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> AMDRotatingSharedEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> AMDRotatingSharedEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}

SmallVector<unsigned> DotOperandEncodingAttr::getCTAsPerCGA() const {
  return ::getCTAsPerCGA(getParent());
}
SmallVector<unsigned> DotOperandEncodingAttr::getCTAOrder() const {
  return ::getCTAOrder(getParent());
}
SmallVector<unsigned> DotOperandEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res = ::getCTASplitNum(getParent());
  auto rank = res.size();
  assert(rank == 2 || rank == 3 && "Invalid dotLayout");

  // Do not split CTA in K dimension
  auto kDim = getOpIdx() == 0 ? rank - 1 : rank - 2;
  res[kDim] = 1;
  return res;
}

LogicalResult DotOperandEncodingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    unsigned opIdx, Attribute parent, unsigned kWidth) {
  if (opIdx != 0 && opIdx != 1) {
    return emitError() << "ttg.dot_op opIdx parameter can be 0 or 1, got: "
                       << opIdx;
  }
  if (!parent) {
    return emitError() << "ttg.dot_op parent parameter cannot be null";
  }
  if (auto parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent)) {
    if (kWidth != 0 && !(parentAttr.isAmpere() || parentAttr.isHopper()))
      return emitError() << "ttg.dot_op kWidth parameter can only be "
                            "non-zero for Ampere or Hopper MMA parent";
    if (kWidth == 0 && (parentAttr.isAmpere() || parentAttr.isHopper()))
      return emitError() << "ttg.dot_op kWidth parameter is mandatory for "
                            "Ampere or Hopper MMA parent";
    if (opIdx != 0 && parentAttr.isHopper())
      return emitError()
             << "ttg.dot_op opIdx parameter must be 0 for "
                "Hopper MMA parent, since Hopper WGMMA only allows first "
                "operand to be in registers";
    return success();
  }

  if (auto parentAttr = mlir::dyn_cast<AMDWmmaEncodingAttr>(parent)) {
    if (kWidth != 16 && parentAttr.getVersion() == 1 ||
        kWidth != 8 && kWidth != 16 && parentAttr.getVersion() == 2)
      return emitError() << "ttg.dot_op kWidth parameter must be 16 for "
                            "gfx11 and 8/16 for gfx12";
    return success();
  }

  if (auto parentAttr = mlir::dyn_cast<AMDMfmaEncodingAttr>(parent)) {
    if (kWidth == 0)
      return emitError() << "ttg.dot_op kWidth parameter is mandatory for "
                            "MFMA parent";
    return success();
  }

  if (auto parentAttr = mlir::dyn_cast<BlockedEncodingAttr>(parent)) {
    if (kWidth != 0)
      return emitError() << "ttg.dot_op kWidth parameter is not supported "
                            "when the parent is a blocked layout";
    return success();
  }

  return emitError() << "ttg.dot_op unexpected parent layout: " << parent;
}

//===----------------------------------------------------------------------===//
// Blocked Encoding
//===----------------------------------------------------------------------===//

static std::optional<CTALayoutAttr> getCTALayoutOrError(
    AsmParser &parser, std::optional<SmallVector<unsigned>> CTAsPerCGA,
    std::optional<SmallVector<unsigned>> CTASplitNum,
    std::optional<SmallVector<unsigned>> CTAOrder, unsigned rank) {
  if (CTAsPerCGA && CTASplitNum && CTAOrder) {
    return CTALayoutAttr::get(parser.getContext(), *CTAsPerCGA, *CTASplitNum,
                              *CTAOrder);
  }
  if (!CTAsPerCGA && !CTASplitNum && !CTAOrder) {
    return CTALayoutAttr::getDefault(parser.getContext(), rank);
  }
  parser.emitError(parser.getNameLoc(), "CTAsPerCGA, CTASplitNum, and CTAOrder "
                                        "must all be present or all be absent");
  return std::nullopt;
}

Attribute BlockedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> sizePerThread;
  SmallVector<unsigned> threadsPerWarp;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> order;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "sizePerThread") {
      if (parseIntArrayAttr(parser, attr, sizePerThread,
                            "number of elements per thread")
              .failed())
        return {};
    } else if (attr.getName() == "threadsPerWarp") {
      if (parseIntArrayAttr(parser, attr, threadsPerWarp,
                            "number of threads per warp")
              .failed())
        return {};
    } else if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA,
                            "number of warps per CTA")
              .failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/sizePerThread.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<BlockedEncodingAttr>(parser.getContext(),
                                                sizePerThread, threadsPerWarp,
                                                warpsPerCTA, order, *CTALayout);
}

void BlockedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerThread = [" << ArrayRef(getSizePerThread()) << "]"
          << ", threadsPerWarp = [" << ArrayRef(getThreadsPerWarp()) << "]"
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]"
          << ", order = [" << getOrder() << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getSizePerThread().size());

  printer << "}>";
}

// FIXME Can we take the LinearLayout by const&?
LogicalResult
LinearEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           LinearLayout linearLayout) {
  // Example of LinearEncodingAttr
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  // The input dims must be {register, lane, warp, block}
  // The output dims of the linear layout should be dim0..dim[rank-1]

  static const auto expectedInDims =
      SmallVector<std::string>({"register", "lane", "warp", "block"});
  for (const auto &[i, dims] : llvm::enumerate(
           llvm::zip(linearLayout.getInDimNames(), expectedInDims))) {
    const auto &[dim, expectedDimStr] = dims;
    if (dim.str() != expectedDimStr) {
      return emitError() << "Expected input dimension " << i << " to be '"
                         << expectedDimStr << "'. Got " << dim;
    }
  }

  // outDims are ['dim0', 'dim1', ...]
  for (auto [i, dim] : llvm::enumerate(linearLayout.getOutDimNames())) {
    if (dim.str() != ("dim" + llvm::Twine(i)).str()) {
      return emitError()
             << "Expected output dimensions to be ['dim0', 'dim1', ...]. Got "
             << dim << " at position " << i;
    }
  }

  const auto &bases = linearLayout.getBases();
  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &dimBases : llvm::make_second_range(bases)) {
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return std::count_if(basis.begin(), basis.end(), nonZero) <= 1;
        })) {
      return emitError()
             << "In a distributed layout, each base must move in at most one "
                "dimension.";
    }
  }

  return success();
}

void LinearEncodingAttr::print(mlir::AsmPrinter &printer) const {
  // We don't use the default implementation as it's a bit too verbose
  // This prints in the following format that is shape agnostic, in the sense
  // that we don't print explicitly the outShape of the LL
  // We always assume LLs to be surjective
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  auto ll = getLinearLayout();
  printer << "<{" << join(ll.getBases(), ", ", [](const auto &base) {
    return base.first.str() + " = " + "[" +
           join(base.second, ", ",
                [](const std::vector<int32_t> &vec) {
                  return "[" + join(vec, ", ") + "]";
                }) +
           "]";
  }) << "}>";
}

Attribute LinearEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};

  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};

  if (parser.parseGreater().failed())
    return {};

  LinearLayout::BasesT bases;

  // Parse the basis names in order (the order is relevant)
  std::vector<std::string> inDimNames = {"register", "lane", "warp", "block"};

  for (const auto &inDimNameStr : inDimNames) {
    auto inDimName = StringAttr::get(parser.getContext(), inDimNameStr);
    Attribute value = dict.get(inDimName);

    // Expecting an array of arrays
    auto arrayOfArraysAttr = mlir::dyn_cast<ArrayAttr>(value);
    if (!arrayOfArraysAttr) {
      parser.emitError(parser.getCurrentLocation(),
                       "Expected array of arrays for basis of '")
          << inDimName.getValue() << "'";
      return {};
    }

    std::vector<std::vector<int32_t>> inDimBases;
    for (Attribute arrayAttr : arrayOfArraysAttr) {
      auto intArrayAttr = mlir::dyn_cast<ArrayAttr>(arrayAttr);
      if (!intArrayAttr) {
        parser.emitError(parser.getCurrentLocation(),
                         "Expected array of integers in basis for '")
            << inDimName.getValue() << "'";
        return {};
      }
      std::vector<int32_t> basis;
      for (Attribute intAttr : intArrayAttr) {
        auto intValueAttr = mlir::dyn_cast<IntegerAttr>(intAttr);
        if (!intValueAttr) {
          parser.emitError(parser.getCurrentLocation(),
                           "Expected integer in basis for '")
              << inDimName.getValue() << "'";
          return {};
        }
        basis.push_back(intValueAttr.getInt());
      }
      inDimBases.push_back(std::move(basis));
    }
    bases[inDimName] = std::move(inDimBases);
  }
  size_t rank = 0;
  for (const auto &basesDim : llvm::make_second_range(bases)) {
    if (!basesDim.empty()) {
      rank = basesDim[0].size();
      break;
    }
  }

  // To implement this we'd need to serialise the rank as well.
  // We can do this if we ever need it
  if (rank == 0) {
    parser.emitError(parser.getCurrentLocation(), "Empty Layout not supported");
    return {};
  }

  // Generate standared outDimNames (dim0, dim1, ...)
  SmallVector<StringAttr> outDimNames;
  for (int i = 0; i < rank; ++i) {
    outDimNames.push_back(
        StringAttr::get(parser.getContext(), "dim" + llvm::Twine(i)));
  }

  // Create LinearLayout
  LinearLayout linearLayout(std::move(bases), std::move(outDimNames));

  // Create and return the LinearEncodingAttr
  return parser.getChecked<LinearEncodingAttr>(parser.getContext(),
                                               std::move(linearLayout));
}

SmallVector<unsigned> basesPerDimImpl(const LinearLayout::BasesT &namedBases,
                                      StringAttr dimName, size_t rank,
                                      bool skipBroadcast = true) {
  const auto &bases = namedBases.find(dimName)->second;

  if (bases.empty()) {
    return SmallVector<unsigned>(rank, 1);
  }

  SmallVector<unsigned> ret(rank, 1);
  auto nonZero = [](auto val) { return val != 0; };
  int nonZeroIdx = 0;
  for (const auto &basis : bases) {
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    // Bases can have one or zero non-zero elements
    // Skip a basis if it's broadcasting (all zeros)
    // e.g. warps for DotOperandEncodingAttr (see ampereDotToLinearLayout)
    if (it != basis.end()) {
      nonZeroIdx = it - basis.begin();
      ret[nonZeroIdx] *= 2;
    } else if (!skipBroadcast) {
      // If we've seen a non-zero basis, we double the size of the previous dim
      // This is just needed to count the CTAsPerCGA
      ret[nonZeroIdx] *= 2;
    }
  }
  return ret;
}

SmallVector<unsigned>
LinearEncodingAttr::basesPerDim(StringAttr dimName, bool skipBroadcast) const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), dimName, rank, skipBroadcast);
}

SmallVector<unsigned>
LinearEncodingAttr::orderPerDim(StringAttr dimName,
                                ArrayRef<unsigned> defaultOrder) const {
  auto ll = getLinearLayout();
  const auto &bases = ll.getBases().find(dimName)->second;
  llvm::SetVector<unsigned> order;
  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &basis : bases) {
    // Bases can have one or zero non-zero elements
    // Skip a basis if it's broadcasting (all zeros)
    // e.g. warps for DotOperandEncodingAttr (see ampereDotToLinearLayout)
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    if (it != basis.end()) {
      auto i = it - basis.begin();
      order.insert(i);
    }
  }
  // If any dim is missing, we add them in the defaultOrder
  for (auto i : defaultOrder) {
    order.insert(i);
  }
  return SmallVector<unsigned>(order.begin(), order.end());
}

// [Note. Divergence of methods wrt. legacy layouts]
// For smaller shapes where the CTATile is larger than the output
// tensor, some methods return different values than the legacy layouts. I think
// this is benign tho. An example: what is the vector of `warpsPerCTA` if
// all the warps hold the same data? I think it should be [1, 1], even if we
// have 4 warps. But perhaps for this we have to add some masking in some
// places... We'll see
SmallVector<unsigned> LinearEncodingAttr::getRepOrder() const {
  // This is not correct, but:
  // - It happens to agree in most places with the legacy layout
  // - getRepOrder does not make sense for LinearEncodingAttr as it already has
  //   the same shape as the tensor that uses it
  return getOrder();
}
SmallVector<unsigned> LinearEncodingAttr::getCTAsPerCGA() const {
  // CTAs are split into an identity part (SplitNum) and a broadcast part
  return basesPerDim(StringAttr::get(getContext(), "block"),
                     /*skipBroadcast=*/false);
}
SmallVector<unsigned> LinearEncodingAttr::getCTAOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "block"), getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getCTASplitNum() const {
  return basesPerDim(StringAttr::get(getContext(), "block"));
}
SmallVector<unsigned> LinearEncodingAttr::getWarpsPerCTA() const {
  return basesPerDim(StringAttr::get(getContext(), "warp"));
}
SmallVector<unsigned> LinearEncodingAttr::getWarpOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "warp"), getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getThreadsPerWarp() const {
  return basesPerDim(StringAttr::get(getContext(), "lane"));
}
SmallVector<unsigned> LinearEncodingAttr::getThreadOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "lane"), getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getSizePerThread() const {
  auto rank = getOrder().size();
  auto ll = getLinearLayout();
  auto ctx = getContext();
  auto kRegister = StringAttr::get(ctx, "register");

  // We canonicalize on the spot, as if we use CGAs the regs are not in
  // canonical form The order is [reg, lane, warp, rep, block], so we first
  // remove the blocks
  llvm::SmallVector<unsigned> ctaShape;
  for (auto [shape, cgaNum] :
       llvm::zip(ll.getOutDimSizes(), getCTASplitNum())) {
    ctaShape.push_back(shape / cgaNum);
  }
  LinearLayout::BasesT bases = ll.getBases();

  llvm::SetVector<unsigned> reverseRepOrder;
  auto nonZero = [](auto val) { return val != 0; };
  auto &registers = bases[kRegister];
  while (!registers.empty()) {
    auto &basis = registers.back();
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    // If there's broadcasting (base == zeros) there are no more reps
    if (it == basis.end()) {
      break;
    }
    auto dim = it - basis.begin();
    reverseRepOrder.insert(dim);
    // As soon as we stop finding reps, we stop
    if (dim != reverseRepOrder.back() || 2 * basis[dim] != ctaShape[dim]) {
      break;
    }
    ctaShape[dim] /= 2;
    registers.pop_back();
  }
  return basesPerDimImpl(bases, kRegister, rank);
}

SmallVector<unsigned> LinearEncodingAttr::getShapePerCTATile() const {
  auto sizePerThread = getSizePerThread();
  auto threadsPerWarp = getThreadsPerWarp();
  auto warpsPerCTA = getWarpsPerCTA();
  SmallVector<unsigned> shape;
  for (auto [size, thread, warp] :
       llvm::zip(sizePerThread, threadsPerWarp, warpsPerCTA)) {
    shape.push_back(size * thread * warp);
  }
  return shape;
}

SmallVector<unsigned> LinearEncodingAttr::getOrder() const {
  auto rank = getLinearLayout().getNumOutDims();
  SmallVector<unsigned> order(rank);
  // Choose [rank-1, rank-2, ... 0] as the default order in case
  // there are dims that do not move in the register
  // This order is as good as any really
  std::iota(order.rbegin(), order.rend(), 0);

  return orderPerDim(StringAttr::get(getContext(), "register"), order);
}

LinearLayout LinearEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ll = getLinearLayout();
  auto canonicalDims = llvm::to_vector(ll.getOutDimNames());
  llvm::SmallDenseMap<StringAttr, int64_t> namedShape;
  llvm::SmallVector<StringAttr> permutedDims;
  for (auto dim : getRepOrder()) {
    permutedDims.push_back(canonicalDims[dim]);
    namedShape[canonicalDims[dim]] = shape[dim];
  }
  ll = ll.transposeOuts(permutedDims);
  ll = ensureLayoutNotSmallerThan(ll, namedShape);
  ll = ensureLayoutNotLargerThan(ll, namedShape, /*broadcastRegisters=*/false);
  ll = ll.transposeOuts(canonicalDims);
  return ll;
}

SmallVector<unsigned>
LinearEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
  // When broadcasting the layout the shape changes, otherwise the shape is
  // the same as the shape of the tensor
  // We can either have BroadcastOp with SameOperandsAndResultEncoding, or keep
  // the invariant that the shape of the LL is that of the tensor
  // We choose the former for BC
  auto scaledLayout = get(getContext(), toLinearLayout(shape));
  auto kRegister = StringAttr::get(getContext(), "register");
  return scaledLayout.basesPerDim(kRegister, /*skipBroadcast=*/false);
}

SmallVector<unsigned>
LinearEncodingAttr::getContig(const char *inDim,
                              SmallVector<unsigned int> lowerContig) const {
  auto ll = getLinearLayout();
  const auto &bases =
      ll.getBases().find(StringAttr::get(getContext(), inDim))->second;
  auto order = getOrder();
  auto rank = order.size();

  SmallVector<unsigned> contig(lowerContig);
  auto basisIt = bases.begin();
  for (unsigned dim : order) {
    std::vector<int32_t> basis(rank, 0);
    basis[dim] = contig[dim];

    while (basisIt != bases.end() && *basisIt == basis) {
      contig[dim] *= 2;
      basis[dim] *= 2;
      ++basisIt;
    }
  }
  return contig;
}

SmallVector<unsigned> LinearEncodingAttr::getContigPerThread() const {
  SmallVector<unsigned> contig(getOrder().size(), 1);
  return getContig("register", contig);
}

SmallVector<unsigned> LinearEncodingAttr::getContigPerWarp() const {
  return getContig("lane", getContigPerThread());
}

unsigned
LinearEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape) const {
  return product(getElemsPerThread(shape));
}

//===----------------------------------------------------------------------===//
// MMA encoding
//===----------------------------------------------------------------------===//

Attribute NvidiaMmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned> warpsPerCTA;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  SmallVector<unsigned> instrShape;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "versionMajor") {
      if (parseUInt(parser, attr, versionMajor, "versionMajor").failed())
        return {};
    }
    if (attr.getName() == "versionMinor") {
      if (parseUInt(parser, attr, versionMinor, "versionMinor").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
    if (attr.getName() == "instrShape") {
      if (parseIntArrayAttr(parser, attr, instrShape, "instrShape").failed()) {
        return {};
      }
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<NvidiaMmaEncodingAttr>(
      parser.getContext(), versionMajor, versionMinor, warpsPerCTA, *CTALayout,
      instrShape);
}

void NvidiaMmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "versionMajor = " << getVersionMajor()
          << ", versionMinor = " << getVersionMinor() //
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getRank());

  printer << ", instrShape = [" << getInstrShape() << "]}>";
}

//===----------------------------------------------------------------------===//
// MFMA encoding
//===----------------------------------------------------------------------===//

Attribute AMDMfmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> instrShape;
  bool isTransposed;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "versionMajor") {
      if (parseUInt(parser, attr, versionMajor, "versionMajor").failed())
        return {};
    }
    if (attr.getName() == "versionMinor") {
      if (parseUInt(parser, attr, versionMinor, "versionMinor").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "instrShape") {
      if (parseIntArrayAttr(parser, attr, instrShape, "instrShape").failed())
        return {};
    }

    if (attr.getName() == "isTransposed") {
      if (parseBool(parser, attr, isTransposed, "isTransposed").failed())
        return {};
    }
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<AMDMfmaEncodingAttr>(
      parser.getContext(), versionMajor, versionMinor, warpsPerCTA,
      instrShape[0], instrShape[1], isTransposed, *CTALayout);
}

void AMDMfmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "versionMajor = " << getVersionMajor()                      //
          << ", versionMinor = " << getVersionMinor()                    //
          << ", warpsPerCTA = [" << getWarpsPerCTA() << "]"              //
          << ", instrShape = [" << ArrayRef{getMDim(), getNDim()} << "]" //
          << ", isTransposed = " << getIsTransposed();
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getRank());
  printer << "}>";
}

LogicalResult
AMDMfmaEncodingAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                            unsigned versionMajor, unsigned versionMinor,
                            llvm::ArrayRef<unsigned int> warpsPerCTA,
                            unsigned mDim, unsigned nDim, bool isTransposed,
                            mlir::triton::gpu::CTALayoutAttr) {
  if (!(versionMajor >= 0 && versionMajor <= 4)) {
    return emitError() << "major version must be in the [0, 4] range";
  }
  if (versionMinor != 0) {
    return emitError() << "minor version must be 0";
  }
  if (!((mDim == 32 && nDim == 32) || (mDim == 16 && nDim == 16))) {
    return emitError()
           << "(M, N) cases other than (32, 32) or (16, 16) unimplemented";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// WMMA encoding
//===----------------------------------------------------------------------===//

Attribute AMDWmmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned version = 0;
  bool isTransposed = false;
  SmallVector<unsigned> warpsPerCTA;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "version") {
      if (parseUInt(parser, attr, version, "version").failed())
        return {};
    }
    if (attr.getName() == "isTranspose") {
      if (parseBool(parser, attr, isTransposed, "isTranspose").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<AMDWmmaEncodingAttr>(
      parser.getContext(), version, isTransposed, warpsPerCTA, *CTALayout);
}

void AMDWmmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "version = " << getVersion()
          << ", isTranspose = " << getIsTransposed() << ", warpsPerCTA = ["
          << ArrayRef(getWarpsPerCTA()) << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getWarpsPerCTA().size());
  printer << "}>";
}

LogicalResult
AMDWmmaEncodingAttr::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                            unsigned version, bool isTransposed,
                            llvm::ArrayRef<unsigned int> warpsPerCTA,
                            mlir::triton::gpu::CTALayoutAttr) {
  if (version != 1 && version != 2) {
    return emitError() << "WMMA version must be in the [1, 2] range";
  }
  // Transposed layout is needed for bypassing LDS between multiple dots.
  // Version 1 tt.dot results and tt.dot operand layouts are different,
  // therefore we test and support transposed only for version 2.
  if (version != 2 && isTransposed) {
    return emitError() << "Transposed WMMA is supported only for version 2";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Sliced Encoding
//===----------------------------------------------------------------------===//

Attribute SliceEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};
  unsigned dim = mlir::cast<IntegerAttr>(attrs.get("dim")).getInt();
  auto parent = mlir::dyn_cast<DistributedEncodingTrait>(attrs.get("parent"));
  if (!parent) {
    parser.emitError(parser.getNameLoc(),
                     "expected a distributed encoding trait");
    return {};
  }
  return parser.getChecked<SliceEncodingAttr>(parser.getContext(), dim, parent);
}

void SliceEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "dim = " << getDim() << ", "
          << "parent = " << getParent() << "}>";
}

//===----------------------------------------------------------------------===//
// Helper shared encoding functions
//===----------------------------------------------------------------------===//

template <typename SpecificEncoding>
Attribute parseSwizzledEncoding(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned vec = 0;
  unsigned perPhase = 0;
  unsigned maxPhase = 0;
  SmallVector<unsigned> order;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "vec") {
      if (parseUInt(parser, attr, vec, "vec").failed())
        return {};
    } else if (attr.getName() == "perPhase") {
      if (parseUInt(parser, attr, perPhase, "perPhase").failed())
        return {};
    } else if (attr.getName() == "maxPhase") {
      if (parseUInt(parser, attr, maxPhase, "maxPhase").failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/order.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<SpecificEncoding>(parser.getContext(), vec, perPhase,
                                             maxPhase, order, *CTALayout);
}

//===----------------------------------------------------------------------===//
// SwizzledShared encoding
//===----------------------------------------------------------------------===//

Attribute SwizzledSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  return parseSwizzledEncoding<SwizzledSharedEncodingAttr>(parser, type);
}

void SwizzledSharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() //
          << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() //
          << ", order = [" << getOrder() << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getOrder().size());
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// NVMMAShared encoding
//===----------------------------------------------------------------------===//

Attribute NVMMASharedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned swizzlingByteWidth;
  bool transposed = false;
  bool fp4Padded = false;
  unsigned elementBitWidth;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "swizzlingByteWidth") {
      if (parseUInt(parser, attr, swizzlingByteWidth, "swizzlingByteWidth")
              .failed())
        return {};
    } else if (attr.getName() == "transposed") {
      if (parseBool(parser, attr, transposed, "transposed").failed())
        return {};
    } else if (attr.getName() == "elementBitWidth") {
      if (parseUInt(parser, attr, elementBitWidth, "elementBitWidth").failed())
        return {};
    } else if (attr.getName() == "fp4Padded") {
      if (parseBool(parser, attr, fp4Padded, "fp4Padded").failed())
        return {};
    } else if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/2);
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<NVMMASharedEncodingAttr>(
      parser.getContext(), swizzlingByteWidth, transposed, elementBitWidth,
      fp4Padded, *CTALayout);
}

void NVMMASharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "swizzlingByteWidth = " << getSwizzlingByteWidth() //
          << ", transposed = " << getTransposed()               //
          << ", elementBitWidth = " << getElementBitWidth();
  if (getFp4Padded()) {
    // Print only in this case to reduce the noise for the more common case.
    printer << ", fp4Padded = true";
  }
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/2);
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// AMDRotatingShared encoding
//===----------------------------------------------------------------------===//

Attribute AMDRotatingSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  return parseSwizzledEncoding<AMDRotatingSharedEncodingAttr>(parser, type);
}

void AMDRotatingSharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() //
          << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() //
          << ", order = [" << getOrder() << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getOrder().size());
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// Mfma encoding
//===----------------------------------------------------------------------===//
// TODO: there is a lot of common code with MmaEncoding here

SmallVector<unsigned> AMDMfmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}

SmallVector<int64_t>
AMDMfmaEncodingAttr::getInstrShapeForOperand(int kWidth, int opIdx) const {
  unsigned mDim = getMDim();
  unsigned nDim = getNDim();
  assert((mDim == nDim) && (mDim == 32 || mDim == 16 || mDim == 4) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  constexpr int warpSize = 64; // MFMA is always based on the 64-wide warps.
  int kGroups = -1;
  if (mDim == nDim)
    kGroups = warpSize / mDim;
  if (mDim == 64 && nDim == 4 || mDim == 4 && nDim == 64)
    kGroups = 1;
  int64_t kDim = kWidth * kGroups;
  if (opIdx == 0)
    return {mDim, kDim};
  else
    assert(opIdx == 1);
  return {kDim, nDim};
}

SmallVector<unsigned> AMDMfmaEncodingAttr::getRepOrder() const {
  return getMatrixOrder(getRank(), /*rowMajor*/ true);
}

SmallVector<unsigned>
AMDMfmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getOrderForDotOperand(opIdx, getRank(), /*kContig*/ true);
}

SmallVector<int64_t>
AMDMfmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> operandShape,
                                      int kWidth, int opIdx) const {
  auto operandTileShape = getInstrShapeForOperand(kWidth, opIdx);
  auto rank = operandShape.size();
  auto warpsPerCTA = getWarpsPerCTA();
  int numRepBatch =
      rank == 3 ? std::max<int64_t>(1, operandShape[0] / warpsPerCTA[0]) : 1;
  if (opIdx == 0)
    return {
        numRepBatch,
        std::max<int64_t>(1, operandShape[rank - 2] /
                                 (operandTileShape[0] * warpsPerCTA[rank - 2])),
        std::max<int64_t>(1, operandShape[rank - 1] / operandTileShape[1])};
  else {
    assert(opIdx == 1);
    return {
        numRepBatch,
        std::max<int64_t>(1, operandShape[rank - 2] / operandTileShape[0]),
        std::max<int64_t>(1, operandShape[rank - 1] / (operandTileShape[1] *
                                                       warpsPerCTA[rank - 1]))};
  }
}

SwizzledSharedEncodingAttr AMDMfmaEncodingAttr::composeSharedLayoutForOperand(
    CTALayoutAttr ctaLayout, int operandIdx, ArrayRef<int64_t> operandShape,
    ArrayRef<unsigned> sharedOrder, unsigned vectorSize, unsigned elemBitWidth,
    bool needTrans) const {
  int kDimIndex = operandIdx == 0 ? 1 : 0;
  if (needTrans)
    kDimIndex = 1 - kDimIndex;

  bool isKContig = sharedOrder[0] == kDimIndex;
  // GFX950 supports LDS transpose load instructions, so we need swizzling even
  // when K dimension is not the contiguous dimension.
  bool isGFX950 = getVersionMajor() == 4;
  bool swizzleNonKContig =
      isGFX950 && (elemBitWidth == 8 || elemBitWidth == 16);

  if (!isKContig && !swizzleNonKContig) {
    // Do not swizzle. In this case accesses will go in different banks even
    // without swizzling.
    return SwizzledSharedEncodingAttr::get(getContext(), 1, 1, 1, sharedOrder,
                                           ctaLayout);
  }

  const unsigned numBanks = isGFX950 ? 64 : 32;
  const unsigned bankBitWidth = 32;
  const unsigned simdWidth = 16;

  // Number of inner dimension rows per one pattern repeat
  int innerDimLength = operandShape[sharedOrder[0]];
  int elemsPerOneBanksRow = (numBanks * bankBitWidth) / elemBitWidth;

  int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
  int maxPhase =
      std::max(std::min(simdWidth / perPhase, innerDimLength / vectorSize), 1u);

  // TODO (zhanglx): figure out better parameters for mfma4
  if (getMDim() == 4)
    maxPhase = 4;

  return SwizzledSharedEncodingAttr::get(getContext(), vectorSize, perPhase,
                                         maxPhase, sharedOrder, ctaLayout);
}

//===----------------------------------------------------------------------===//
// Wmma encoding
//===----------------------------------------------------------------------===//

SmallVector<unsigned> AMDWmmaEncodingAttr::getRepOrder() const {
  return getMatrixOrder(getRank(), /*rowMajor*/ true);
}

SmallVector<unsigned>
AMDWmmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getOrderForDotOperand(opIdx, getRank(), /*kContig*/ true);
}

SmallVector<unsigned> AMDWmmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}

SmallVector<int64_t> AMDWmmaEncodingAttr::getElemsPerInstrForOperands() const {
  return {16, 16};
}

SmallVector<int64_t>
AMDWmmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> operandShape,
                                      Type elemType, int kWidth,
                                      int opIdx) const {
  auto operandTileShape = getElemsPerInstrForOperands();
  assert(operandTileShape.size() == 2);
  auto warpsPerCTA = getWarpsPerCTA();
  auto rank = operandShape.size();
  assert(rank == 2 || rank == 3);
  int numRepBatch =
      rank == 3 ? std::max<int64_t>(1, operandShape[0] / warpsPerCTA[0]) : 1;
  if (opIdx == 0)
    return {
        numRepBatch,
        std::max<int64_t>(1, operandShape[rank - 2] /
                                 (operandTileShape[0] * warpsPerCTA[rank - 2])),
        std::max<int64_t>(1, operandShape[rank - 1] / operandTileShape[1])};
  else {
    assert(opIdx == 1);
    return {
        numRepBatch,
        std::max<int64_t>(1, operandShape[rank - 2] / operandTileShape[0]),
        std::max<int64_t>(1, operandShape[rank - 1] / (operandTileShape[1] *
                                                       warpsPerCTA[rank - 1]))};
  }
}

SmallVector<unsigned> AMDWmmaEncodingAttr::getMNKDimPerInstr() {
  // TODO: move magic numbers out of the code
  return {16, 16, 16};
}

unsigned AMDWmmaEncodingAttr::getKWidthForOperands() const {
  SmallVector<unsigned> sizePerThread(getRank(), 1);
  auto numReplicated = getVersion() == 1 ? 2 : 1;
  auto elemsPerInstr =
      numReplicated * product(getElemsPerInstrForOperands()) / 32;
  return elemsPerInstr;
}

//===----------------------------------------------------------------------===//
// Mma encoding
//===----------------------------------------------------------------------===//

bool NvidiaMmaEncodingAttr::isVolta() const { return getVersionMajor() == 1; }

bool NvidiaMmaEncodingAttr::isTuring() const {
  return getVersionMajor() == 2 && getVersionMinor() == 1;
}

bool NvidiaMmaEncodingAttr::isAmpere() const { return getVersionMajor() == 2; }

bool NvidiaMmaEncodingAttr::isHopper() const { return getVersionMajor() == 3; }

SmallVector<unsigned> NvidiaMmaEncodingAttr::getRepOrder() const {
  return getMatrixOrder(getRank(), /*rowMajor*/ true);
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}

SmallVector<unsigned>
NvidiaMmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getOrderForDotOperand(opIdx, getRank(), /*kContig*/ true);
}

SmallVector<int64_t>
NvidiaMmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> shape, int bitwidth,
                                        int kWidth, int opIdx) const {
  assert(
      kWidth >= 32 / bitwidth &&
      "kWidth must be >= 32 / bitwidth for this function to be well-defined");
  auto rank = shape.size();
  // Broadcast long K
  auto warpsPerCTA = to_vector(getWarpsPerCTA());
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;
  warpsPerCTA[kDim] = 1;

  SmallVector<int> tileSize;
  if (rank == 3) {
    tileSize.push_back(1);
  }
  if (opIdx == 0) {
    // m x k
    tileSize.push_back(16);
    tileSize.push_back(4 * 64 / bitwidth);
  } else {
    // k x n
    // Hopper path never uses the n value, since this method is only invoked
    // for in-RF (dotOpEnc) operands, but WGMMA only supports in A to be in RF
    // so it's fine if the n is incorrect here
    tileSize.push_back(4 * 64 / bitwidth);
    tileSize.push_back(8);
  }

  SmallVector<int64_t> numRep;
  // Lezcano: This is odd. Why do we always return a vector of size 3?
  if (rank != 3) {
    numRep.push_back(1);
  }
  for (auto [s, size, warp] : llvm::zip(shape, tileSize, warpsPerCTA)) {
    numRep.push_back(std::max<int64_t>(1, s / (size * warp)));
  }
  return numRep;
}

//===----------------------------------------------------------------------===//
// DotOperand Encoding
//===----------------------------------------------------------------------===//
SmallVector<unsigned> DotOperandEncodingAttr::getRepOrder() const {
  if (auto mma = mlir::dyn_cast<MmaEncodingTrait>(getParent())) {
    return mma.getRepOrderForOperand(getOpIdx());
  }
  llvm::report_fatal_error(
      "getRepOrder not implemented for DotOperandEncodingAttr");
  return {};
}

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//

class TritonGPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    // Encoding attributes
    if (auto mmaAttr = mlir::dyn_cast<MmaEncodingTrait>(attr)) {
      os << "mma";
      return AliasResult::FinalAlias;
    } else if (auto sharedAttr = mlir::dyn_cast<SharedEncodingTrait>(attr)) {
      os << "shared";
      return AliasResult::FinalAlias;
    } else if (auto blockedAttr = mlir::dyn_cast<BlockedEncodingAttr>(attr)) {
      os << "blocked";
      return AliasResult::FinalAlias;
    } else if (auto linearAttr = mlir::dyn_cast<LinearEncodingAttr>(attr)) {
      os << "linear";
      return AliasResult::FinalAlias;
    } /* else if (auto sliceAttr = dyn_cast<SliceEncodingAttr>(attr)) {
      os << "slice";
      return AliasResult::FinalAlias;
    } */
    // Memory space attributes
    if (auto smem = mlir::dyn_cast<SharedMemorySpaceAttr>(attr)) {
      os << "smem";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

struct TritonGPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding) const override {
    resultEncoding =
        SliceEncodingAttr::get(getDialect()->getContext(), axis,
                               cast<DistributedEncodingTrait>(operandEncoding));
    return success();
  }

  // Infer the encoding of a tt.trans(x) given the encoding of x.
  //
  // Our goal is to choose an encoding so that the trans is a "nop".  For
  // example, in a blocked encoding, the same GPU threads hold the same
  // elements, they're just "renamed" -- what was element [i,j] of the tensor is
  // now element [j,i], but that element is held by the same GPU thread.
  //
  // For most properties of the encoding, we let
  //   outputEnc.prop = inputEnc.prop * trans.order,
  // where `x * y` means we apply permutation y to x.
  //
  // This works because prop[i] tells you something about the i'th dimension of
  // the tensor. (For example, sizePerThread[2] == 4 means that one GPU thread
  // contains 4 elements along dim 2 of the tensor.) The transpose reorders the
  // dimensions according to the perm trans.order, so we achieve our goal of
  // having a "nop" transpose by reordering the values in the prop the same way.
  //
  // The big exception to this is the encoding's `order`.
  //
  // An encoding's order is a list of dimensions, from fastest moving (most
  // minor) to slowest moving.  Thus enc.order[i] does not tell you something
  // about the i'th dimension of the tensor, and it would be disasterously
  // incorrect to do enc.order * trans.order.
  //
  // But!  If we invert enc.order, it *does* meet this criterion.  For example,
  // if enc.order = [2,0,1], inverse(enc.order) = [1,2,0].  If you stare at it,
  // you'll see that inverse(enc.order)[i] == j means that dimension i is the
  // j'th most minor.  Therefore we can safely permute *this* by trans.order.
  //
  // Thus we have
  //
  //   outputEnc.order = inverse(inverse(inputEnc.order) * trans.order)
  //                   = inverse(trans.order) * inputEnc.order.
  //
  LogicalResult inferTransOpEncoding(Attribute operandEncoding,
                                     ArrayRef<int64_t> shape,
                                     ArrayRef<int32_t> order, // trans order
                                     Attribute &resultEncoding) const override {
    // Note: inferFooOpEncoding should not crash if given invalid inputs, which
    // happens when someone creates invalid IR.  If we return failure() on
    // error, then MLIR will generate a helpful error message.

    auto *ctx = getDialect()->getContext();
    auto invOrder = inversePermutation(order);
    SmallVector<unsigned> invOrderUnsigned(invOrder.begin(), invOrder.end());

    auto permuteCTALayout =
        [&](const CTALayoutAttr &layout) -> FailureOr<CTALayoutAttr> {
      auto n = order.size();
      if (layout.getCTAsPerCGA().size() != n ||
          layout.getCTASplitNum().size() != n ||
          layout.getCTAOrder().size() != n) {
        return failure();
      }

      return CTALayoutAttr::get(
          ctx, applyPermutation(layout.getCTAsPerCGA(), order),
          applyPermutation(layout.getCTASplitNum(), order),
          applyPermutation(invOrderUnsigned, layout.getCTAOrder()));
    };

    if (auto enc =
            mlir::dyn_cast<SwizzledSharedEncodingAttr>(operandEncoding)) {
      if (enc.getOrder().size() != order.size()) {
        return failure();
      }
      FailureOr<CTALayoutAttr> ctaLayout = permuteCTALayout(enc.getCTALayout());
      if (failed(ctaLayout)) {
        return failure();
      }
      resultEncoding = SwizzledSharedEncodingAttr::get(
          ctx, enc.getVec(), enc.getPerPhase(), enc.getMaxPhase(),
          applyPermutation(invOrderUnsigned, enc.getOrder()), *ctaLayout);
      return success();
    }

    if (auto enc = mlir::dyn_cast<NVMMASharedEncodingAttr>(operandEncoding)) {
      if (order != ArrayRef<int32_t>({1, 0})) {
        return failure();
      }
      FailureOr<CTALayoutAttr> ctaLayout = permuteCTALayout(enc.getCTALayout());
      if (failed(ctaLayout)) {
        return failure();
      }
      resultEncoding = NVMMASharedEncodingAttr::get(
          ctx, enc.getSwizzlingByteWidth(), !enc.getTransposed(),
          enc.getElementBitWidth(), enc.getFp4Padded(), *ctaLayout);
      return success();
    }

    if (auto enc = mlir::dyn_cast<BlockedEncodingAttr>(operandEncoding)) {
      auto n = order.size();
      if (enc.getSizePerThread().size() != n ||
          enc.getThreadsPerWarp().size() != n ||
          enc.getWarpsPerCTA().size() != n || enc.getOrder().size() != n) {
        return failure();
      }
      FailureOr<CTALayoutAttr> ctaLayout = permuteCTALayout(enc.getCTALayout());
      if (failed(ctaLayout)) {
        return failure();
      }
      resultEncoding = BlockedEncodingAttr::get(
          ctx, applyPermutation(enc.getSizePerThread(), order),
          applyPermutation(enc.getThreadsPerWarp(), order),
          applyPermutation(enc.getWarpsPerCTA(), order),
          applyPermutation(invOrderUnsigned, enc.getOrder()), *ctaLayout);
      return success();
    }
    auto ll = toLinearLayout(shape, operandEncoding);
    auto namedBases = ll.getBases();
    for (auto &bases : llvm::make_second_range(namedBases)) {
      for (auto &b : bases) {
        std::vector<int32_t> newB;
        for (auto i : order) {
          newB.push_back(b[i]);
        }
        b = std::move(newB);
      }
    }
    auto retLl = LinearLayout(std::move(namedBases),
                              llvm::to_vector(ll.getOutDimNames()));
    resultEncoding = LinearEncodingAttr::get(ctx, std::move(retLl));
    return success();
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    auto sliceEncoding = mlir::dyn_cast<SliceEncodingAttr>(operandEncoding);
    if (!sliceEncoding)
      return emitOptionalError(
          location, "ExpandDimsOp operand encoding must be SliceEncodingAttr");
    if (sliceEncoding.getDim() != axis)
      return emitOptionalError(
          location, "Incompatible slice dimension for ExpandDimsOp operand");
    resultEncoding = sliceEncoding.getParent();
    return success();
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> location) const override {
    auto mmaRetEncoding = mlir::dyn_cast<NvidiaMmaEncodingAttr>(retEncoding);
    if (mmaRetEncoding && mmaRetEncoding.isHopper()) {
      auto dotOpEnc = mlir::dyn_cast<DotOperandEncodingAttr>(operandEncoding);
      if (!mlir::isa<NVMMASharedEncodingAttr>(operandEncoding) &&
          !(opIdx == 0 && dotOpEnc && dotOpEnc.getOpIdx() == 0 &&
            mlir::isa<NvidiaMmaEncodingAttr>(dotOpEnc.getParent()))) {
        return emitOptionalError(
            location, "unexpected operand layout for NvidiaMmaEncodingAttr v3");
      }
    } else if (auto dotOpEnc =
                   mlir::dyn_cast<DotOperandEncodingAttr>(operandEncoding)) {
      if (opIdx != dotOpEnc.getOpIdx())
        return emitOptionalError(location, "Wrong opIdx");
      if (retEncoding != dotOpEnc.getParent())
        return emitOptionalError(location, "Incompatible parent encoding");
    } else
      return emitOptionalError(
          location, "Dot's a/b's encoding should be of DotOperandEncodingAttr");
    return success();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    auto aEncoding =
        mlir::dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandEncodingA);
    auto bEncoding =
        mlir::dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandEncodingB);
    if (!aEncoding && !bEncoding)
      return mlir::success();
    auto mmaAEncoding =
        mlir::dyn_cast_or_null<NvidiaMmaEncodingAttr>(aEncoding.getParent());
    if (mmaAEncoding && mmaAEncoding.isHopper())
      return success();
    // Verify that the encodings are valid.
    if (!aEncoding || !bEncoding)
      return op->emitError("mismatching encoding between A and B operands");
    if (aEncoding.getKWidth() != bEncoding.getKWidth())
      return op->emitError("mismatching kWidth between A and B operands");
    return success();
  }

  // Given a src shape + encoding and a dst shape, our goal is to compute a dst
  // encoding that makes the reshape a "nop".  That is, if GPU thread [x,y,z]
  // contains elements [a,b,c,d] before the reshape, it contains those same
  // elements after the reshape, they're just "renamed".
  //
  // Using legacy layouts, a dst encoding that satisfies this property may not
  // exist.  Here are some positive and negative examples.
  //
  //   - NOT OK: 4x4 order=[0,1] -> 16.  Reshape merges elements so
  //     dim 1 is the fastest-changing in the dst, but the src has the opposite
  //     order.
  //   - OK: 2x2x32 order=[1,0,2] -> 4x32.  We choose dst order [0,1].
  //     What's important is that the 2x2 dimensions appear in major-to-minor
  //     order.
  //   - NOT OK: 32x32 sizePerThread=[2,2] -> 1024.  Thread 0 in the src
  //     contains elements [(0,0), (0,1), (1,0), and (1,1)].  We cannot express
  //     this with an encoding based on the dst shape.
  //   - OK: 32x4 sizePerThread=[4,4] -> 128.  dst with sizePerThread=[16] will
  //     contain the same elements as before.
  //
  // With linear layouts, we can always find a dst encoding that satisfies
  // this property. See inferReshapeOpEncoding.
  //
  // Users of this function require that it is symmetrical: if
  // (srcShape,srcEnc,dstShape) => dstEnc, then (dstShape,dstEnc,srcShape) =>
  // srcEnc.
  LogicalResult inferReshapeOpLegacyEncoding(ArrayRef<int64_t> srcShape,
                                             Attribute srcEnc,
                                             ArrayRef<int64_t> dstShape,
                                             Attribute &dstEnc) const {
    auto src = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc);
    if (!src) {
      return failure();
    }

    // Nop reshape; we can always infer an encoding.
    if (srcShape == dstShape) {
      dstEnc = srcEnc;
      return success();
    }

    // default -> default encoding is always a nop.
    auto context = srcEnc.getContext();
    int32_t numWarps = product(src.getWarpsPerCTA());
    int32_t threadsPerWarp = product(src.getThreadsPerWarp());
    int32_t numCTAs = product(src.getCTALayout().getCTAsPerCGA());
    if (srcEnc == getDefaultBlockedEncoding(context, srcShape, numWarps,
                                            threadsPerWarp, numCTAs)) {
      dstEnc = getDefaultBlockedEncoding(context, dstShape, numWarps,
                                         threadsPerWarp, numCTAs);
      return success();
    }

    // Feature flag to disable this routine while it's relatively new.
    // TODO(jlebar): Remove this once we're confident in the code.
    if (triton::tools::getBoolEnv(
            "TRITON_DISABLE_RESHAPE_ENCODING_INFERENCE")) {
      return failure();
    }

    // Cowardly refuse to handle encodings with multiple CTAs.  CTAsPerCGA
    // should be like the other fields in blocked encoding, but I'm not sure how
    // to handle CTASplitNum.
    if (!all_of(src.getCTAsPerCGA(), [](int32_t x) { return x == 1; }) ||
        !all_of(src.getCTASplitNum(), [](int32_t x) { return x == 1; })) {
      return failure();
    }

    // Cowardly refuse to handle encodings where shape[dim] is not divisible by
    // sizePerThread[dim], threadsPerWarp[dim], and warpsPerCTA[dim].  (We make
    // an exception if the block is larger than the shape.)
    auto checkDivisibility = [&](StringRef name, ArrayRef<unsigned> subblock) {
      for (int dim = 0; dim < srcShape.size(); dim++) {
        if (srcShape[dim] >= subblock[dim] &&
            srcShape[dim] % subblock[dim] != 0) {
          return failure();
        }
      }
      return success();
    };
    if (!succeeded(
            checkDivisibility("sizePerThread", src.getSizePerThread())) ||
        !succeeded(
            checkDivisibility("threadsPerWarp", src.getThreadsPerWarp())) ||
        !succeeded(checkDivisibility("warpsPerCTA", src.getWarpsPerCTA()))) {
      return failure();
    }

    SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> decomp =
        getReshapeDecomposition(srcShape, dstShape);

    // enc.order[i] == j means that dimension j is the enc.order[i]'th most
    // minor. But what we usually want is the inverse: inverse(enc.order)[i] = j
    // means that dimension i is the j'th most minor (larger means more major).
    auto srcInvOrder = inversePermutation(src.getOrder());

    // If src dims [a,b,c] are to be merged, then they must be consecutive in
    // physical order, with `a` being the most major.
    for (const auto &[srcDims, dstDims] : decomp) {
      if (!isConsecutive(to_vector(reverse(gather(srcInvOrder, srcDims))))) {
        return failure();
      }
    }

    // If src dims [a,b,c] are to be merged, then `c` must fill up sizePerThread
    // / threadsPerWarp / blocksPerCTA before `b` can have any non-1 values.
    // Examples:
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[1,2,2].
    //    The total sizePerThread for dim 2 is 2, which is less than dim 2's
    //    size of 4.  Therefore dim 1 cannot have non-1 sizePerThread.
    //
    //  - OK: shape=[4,4,4], sizePerThread=[1,2,4].
    //    Dim 2's sizePerThread covers its whole size, so dim 1 is allowed to
    //    have non-1 sizePerThread.
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[2,1,4].
    //    Dim 1's sizePerThread does not cover its whole size, so dim 0 is not
    //    allowed to have non-1 sizePerThread.
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[1,1,2],
    //            threadsPerWarp=[1,2,1].
    //    Dim 2 has 2 elems per thread and 1 thread per warp.  2*1 is less than
    //    dim 2's size.  Therefore dim 1 must have threadsPerWarp=1.
    //
    // In addition, the encoding's block can be larger than the shape, but only
    // in the most-major dimension of each decomposed chunk, and only after
    // we've "used up" the more minor dims.  Examples:
    //
    //  - OK: shape=[4,4,4], sizePerThread=[1,2,4], threadsPerWarp=[16,2,1],
    //        warpsPerCTA=[4,1,1].
    //    The whole size of dims 0 and 1 are covered by sizePerThread *
    //    threadsPerWarp.  Therefore dim 2 is allowed to have threadsPerWarp and
    //    warpsPerCTA larger than its size.
    for (const auto &[srcDims, dstDims] : decomp) {
      auto shapeRemaining = gather(srcShape, srcDims);
      auto checkSubblock = [&, srcDims = srcDims](ArrayRef<unsigned> subblock) {
        // Iterate minor-to-major (i==0 is most major).
        for (int i = srcDims.size() - 1; i >= 0; i--) {
          int dim = srcDims[i];
          if (subblock[dim] == 1) {
            continue;
          }

          // Check that more-minor dims all have 1 in shapeRemaining.
          for (int j = i + 1; j < srcDims.size(); j++) {
            if (shapeRemaining[j] != 1) {
              return failure();
            }
          }

          if (shapeRemaining[i] >= subblock[dim]) {
            assert(shapeRemaining[i] % subblock[dim] == 0); // checked earlier
            shapeRemaining[i] /= subblock[dim];
          } else {
            shapeRemaining[i] = 0;
          }

          // Is the block larger than the shape in this dimension?  This is OK
          // only if we're the most-major dimension of the chunk and in all
          // future chunks, only this most-major dim has a non-1 size.
          if (shapeRemaining[i] == 0 && i != 0) {
            return failure();
          }
        }
        return success();
      };
      if (!succeeded(checkSubblock(src.getSizePerThread())) ||
          !succeeded(checkSubblock(src.getThreadsPerWarp())) ||
          !succeeded(checkSubblock(src.getWarpsPerCTA()))) {
        return failure();
      }
    }

    // Given e.g. src.getSizePerThread(), computeSubblockSize computes e.g.
    // dst.getSizePerThread().  This should be called for each of sizePerThread,
    // threadsPerWarp, and warpsPerCTA, in that order.
    SmallVector<int64_t> dstShapeRemaining(dstShape);
    auto computeSubblockSize = [&](ArrayRef<unsigned> srcSubblock,
                                   SmallVector<unsigned> &dstSubblock,
                                   StringRef fieldName) -> LogicalResult {
      // The dst subblock is "filled up" greedily starting with the most minor
      // dim.  When we're done, we are left with a smaller shape, of size
      // dstShape / dstSubblock, which we store in dstShapeRemaining and use for
      // the next call to computeSubblockSize.
      dstSubblock.resize(dstShape.size());
      for (const auto &[srcDims, dstDims] : decomp) {
        int64_t subblockRemaining = product(gather(srcSubblock, srcDims));
        for (int i = dstDims.size() - 1; i >= 0; i--) {
          auto &val = dstSubblock[dstDims[i]];
          auto &shapeRemaining = dstShapeRemaining[dstDims[i]];
          val = std::min(subblockRemaining, shapeRemaining);

          assert(shapeRemaining % val == 0); // Checked earlier.
          subblockRemaining /= val;
          shapeRemaining /= val;
        }

        // If there are any elems remaining in the subblock, it must be because
        // the block is larger than the shape.  This excess goes into the
        // most-major dim of the subblock.
        dstSubblock[dstDims[0]] *= subblockRemaining;
      }
      return success();
    };

    SmallVector<unsigned> dstSizePerThread;
    SmallVector<unsigned> dstThreadsPerWarp;
    SmallVector<unsigned> dstWarpsPerCTA;
    if (!succeeded(computeSubblockSize(src.getSizePerThread(), dstSizePerThread,
                                       "sizePerThread")) ||
        !succeeded(computeSubblockSize(src.getThreadsPerWarp(),
                                       dstThreadsPerWarp, "threadsPerWarp")) ||
        !succeeded(computeSubblockSize(src.getWarpsPerCTA(), dstWarpsPerCTA,
                                       "warpsPerCTA"))) {
      return failure();
    }

    // Since we know that each set of srcDims is consecutive, we can
    // meaningfully sort decomp by the physical order of the src dimensions,
    // major-to-minor.  This will also be the order of the dst dimensions.
    llvm::sort(decomp, [&](const auto &a, const auto &b) {
      const auto &[srcDimsA, dstDimsA] = a;
      const auto &[srcDimsB, dstDimsB] = b;
      return srcInvOrder[srcDimsA.front()] < srcInvOrder[srcDimsB.front()];
    });

    // Compute the dst order.  Make the dimensions appear in the same order as
    // their corresponding src dimensions.
    SmallVector<unsigned> dstInvOrder(dstShape.size());
    int i = 0;
    for (const auto &[srcDims, dstDims] : decomp) {
      for (auto dim : reverse(dstDims)) {
        dstInvOrder[dim] = i++;
      }
    }
    auto dstOrder = inversePermutation(dstInvOrder);

    // CTALayout can be all 1's because we bailed on multi-CTA layouts above.
    auto CTALayout = CTALayoutAttr::get(
        src.getContext(),
        /*CTAsPerCGA=*/SmallVector<unsigned>(dstShape.size(), 1),
        /*CTASplitNum=*/SmallVector<unsigned>(dstShape.size(), 1),
        /*CTAOrder=*/llvm::to_vector(llvm::seq<unsigned>(dstShape.size())));

    dstEnc = BlockedEncodingAttr::get(src.getContext(), dstSizePerThread,
                                      dstThreadsPerWarp, dstWarpsPerCTA,
                                      dstOrder, CTALayout);

    return success();
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    if (expected == got) {
      return success();
    }
    if (!expected || !got)
      return failure();
    // Check whether the encodings are structurally the same.
    auto expectedLL = triton::gpu::toLinearLayout(shape, expected);
    auto gotLL = triton::gpu::toLinearLayout(shape, got);
    if (expectedLL != gotLL) {
      return emitOptionalError(loc, "Expected result encoding ", expected,
                               " but was ", got);
    }
    return success();
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    auto result =
        inferReshapeOpLegacyEncoding(srcShape, srcEnc, dstShape, dstEnc);
    if (succeeded(result)) {
      return result;
    }

    // If the legacy encoding failed use LinearLayouts.
    // Once LinearLayouts are more widely used, we can remove
    // inferReshapeOpLegacyEncoding and simply use LLs.
    auto *ctx = getContext();
    auto src = toLinearLayout(srcShape, srcEnc);

    if (product(srcShape) != product(dstShape)) {
      return emitOptionalError(loc, "numel of dst shape does not match "
                                    "numel of src shape");
    }

    auto newRank = dstShape.size();

    auto newOutDims = standardOutDimPairs(ctx, dstShape);

    // reshapeOp assumes minor-to-major, so we need to transpose the out dims
    // before the reshape
    auto srcOutDims = to_vector(src.getOutDimNames());
    std::reverse(srcOutDims.begin(), srcOutDims.end());
    std::reverse(newOutDims.begin(), newOutDims.end());
    auto dst = src.transposeOuts(srcOutDims)
                   .reshapeOuts(newOutDims)
                   .transposeOuts(standardOutDimNames(ctx, newRank));
    dstEnc = LinearEncodingAttr::get(ctx, dst);
    return success();
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    if (auto enc = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc)) {
      // JoinOp takes two tensors of shape AxBxC and generates a tensor of shape
      // AxBxCx2.  The encoding is the same as the input, but with 2 elems per
      // thread in the new dimension.  The new dimension is most-minor.
      auto append = [](ArrayRef<unsigned> vals, int val) {
        SmallVector<unsigned> ret(vals);
        ret.push_back(val);
        return ret;
      };
      auto appendMinorDim = [](ArrayRef<unsigned> order) {
        SmallVector<unsigned> ret(order);
        ret.insert(ret.begin(), ret.size());
        return ret;
      };
      dstEnc = BlockedEncodingAttr::get(
          enc.getContext(), append(enc.getSizePerThread(), 2),
          append(enc.getThreadsPerWarp(), 1), append(enc.getWarpsPerCTA(), 1),
          appendMinorDim(enc.getOrder()),
          CTALayoutAttr::get(enc.getContext(), append(enc.getCTAsPerCGA(), 1),
                             append(enc.getCTASplitNum(), 1),
                             appendMinorDim(enc.getCTAOrder())));
      return success();
    }

    auto ctx = getContext();

    // Append dim to shape
    auto ll = toLinearLayout(shape, srcEnc);
    SmallVector<int64_t> dstShape(shape.begin(), shape.end());
    dstShape.push_back(1);
    ll = ll.reshapeOuts(standardOutDimPairs(ctx, dstShape));

    // Try join on last dim
    auto axis = dstShape.size() - 1;
    auto newLl = LinearLayout::empty();
    auto result =
        tryJoinOnAxis(ctx, ll, newLl, /*fwdInference=*/true, axis, loc);

    assert(result.succeeded());
    dstEnc = LinearEncodingAttr::get(ctx, newLl);
    return success();
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    auto enc = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc);
    if (enc) {
      // SplitOp takes a tensor of shape AxBxCx2 and generates two tensors of
      // shape AxBxC.  The input must have 2 elements per thread in the last
      // dimension, which must be most-minor.  The result encoding is the same
      // as the input, but with the last dimension removed.
      if (enc.getSizePerThread().back() != 2) {
        return emitOptionalError(
            loc, "SplitOp requires 2 elements per thread in the "
                 "last dimension of the input");
      }
      if (enc.getThreadsPerWarp().back() != 1 ||
          enc.getWarpsPerCTA().back() != 1 || enc.getCTAsPerCGA().back() != 1) {
        return emitOptionalError(
            loc, "SplitOp requires threadsPerWarp, warpsPerCTA, "
                 "and CTAsPerCGA = 1 for the last dimension of the input");
      }
      if (enc.getCTALayout().getCTAsPerCGA().back() != 1) {
        return emitOptionalError(
            loc,
            "SplitOp requires the last dimension to be most-minor in CTAOrder");
      }
      SmallVector<unsigned> newOrder(enc.getOrder());
      int splitDim = newOrder.size() - 1;
      // Remove splitDim from order.
      newOrder.erase(std::remove(newOrder.begin(), newOrder.end(), splitDim),
                     newOrder.end());
      dstEnc = BlockedEncodingAttr::get(
          enc.getContext(), //
          ArrayRef(enc.getSizePerThread()).drop_back(1),
          ArrayRef(enc.getThreadsPerWarp()).drop_back(1),
          ArrayRef(enc.getWarpsPerCTA()).drop_back(1), ArrayRef(newOrder),
          CTALayoutAttr::get(enc.getContext(), //
                             ArrayRef(enc.getCTAsPerCGA()).drop_back(1),
                             ArrayRef(enc.getCTASplitNum()).drop_back(1),
                             ArrayRef(enc.getCTAOrder()).drop_front(1)));
      return success();
    }

    auto axis = shape.size() - 1;
    assert(shape[axis] == 2 &&
           "SplitOp input shape should have 2 in the last dim");

    auto ctx = getContext();

    // Split on last dim
    auto ll = toLinearLayout(shape, srcEnc);
    auto newLl = LinearLayout::empty();
    auto result =
        tryJoinOnAxis(ctx, ll, newLl, /*fwdInference=*/false, axis, loc);
    if (!result.succeeded()) {
      return failure();
    }

    // Remove last dim from newLl (which should be 1)
    SmallVector<int64_t> dstShape(shape.begin(), shape.end());
    dstShape.pop_back();
    newLl = newLl.reshapeOuts(standardOutDimPairs(ctx, dstShape));
    dstEnc = LinearEncodingAttr::get(ctx, newLl);
    return success();
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    // We implement two legacy layout propagations
    // Once we fully migrate to LinearLayouts, we can remove these.
    auto *ctx = getContext();
    // The output encoding will only be a legacy encoding if the axis is the
    // fastest running dimension.
    // FIXME: We should make sure that there are enough elements along the axis
    // axis whenever fwdInference is false
    if (getOrder(cast<DistributedEncodingTrait>(inEnc), shape)[axis] == 0) {
      // Dot operand: double kWidth if kDim == axis.
      if (auto dotEnc = mlir::dyn_cast<DotOperandEncodingAttr>(inEnc)) {
        auto kWidth = dotEnc.getKWidth();
        if (fwdInference) {
          kWidth *= 2;
        } else {
          if (kWidth > 1) {
            // bwd inference
            kWidth /= 2;
          } else {
            return emitOptionalError(loc,
                                     "Fp4ToFpOp requires at least 2 elements "
                                     "per thread in the axis dimension");
          }
        }
        outEnc = DotOperandEncodingAttr::get(ctx, dotEnc.getOpIdx(),
                                             dotEnc.getParent(), kWidth);
        return success();
      }

      // Blocked layout: double elemsPerThread[axis].
      if (auto blockedEnc = mlir::dyn_cast<BlockedEncodingAttr>(inEnc)) {
        auto sizePerThread = llvm::to_vector(blockedEnc.getSizePerThread());
        if (fwdInference) {
          sizePerThread[axis] *= 2;
        } else {
          if (sizePerThread[axis] > 1) {
            sizePerThread[axis] /= 2;
          } else {
            return emitOptionalError(
                loc, "Fp4ToFpOp requires at least 2 elements per "
                     "thread in the axis dimension");
          }
        }
        outEnc = BlockedEncodingAttr::get(
            ctx, sizePerThread, blockedEnc.getThreadsPerWarp(),
            blockedEnc.getWarpsPerCTA(), blockedEnc.getOrder(),
            blockedEnc.getCTALayout());
        return success();
      }
    }

    auto ll = toLinearLayout(shape, inEnc);
    auto newLl = LinearLayout::empty();
    auto result = tryJoinOnAxis(ctx, ll, newLl, fwdInference, axis, loc);
    if (!result.succeeded())
      return result;
    outEnc = LinearEncodingAttr::get(ctx, newLl);
    return success();
  }
};

struct TritonGPUVerifyTensorLayoutInterface
    : public triton::DialectVerifyTensorLayoutInterface {
  using DialectVerifyTensorLayoutInterface::DialectVerifyTensorLayoutInterface;

  LogicalResult verifyTensorLayout(
      Attribute layout, RankedTensorType rankedTy, Operation *op,
      function_ref<InFlightDiagnostic()> makeErr) const override {
    if (isa<triton::gpu::SharedEncodingTrait>(layout))
      return makeErr() << "Shared layout is not allowed on tensor type.";
    // TODO(jlebar): Currently this only checks blocked layouts, but other
    // layouts also have invariants!

    // TODO(jlebar): Handle the case when the encoding is nested within tt.ptr.
    if (auto blocked = dyn_cast<BlockedEncodingAttr>(layout)) {
      ModuleOp module = op->getParentOfType<ModuleOp>();

      // A different verifier should have checked that the layout itself is
      // valid, including that threads-per-warp has the same rank as
      // warps-per-block etc.
      if (blocked.getRank() != rankedTy.getRank()) {
        return makeErr() << layout << ".\nLayout has rank " << blocked.getRank()
                         << ", but the tensor it's attached to has rank "
                         << rankedTy.getRank() << ".";
      }

      int moduleThreadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
      int64_t layoutThreadsPerWarp = product(blocked.getThreadsPerWarp());
      if (layoutThreadsPerWarp != moduleThreadsPerWarp) {
        return makeErr() << layout << ".\nLayout has a total of "
                         << layoutThreadsPerWarp
                         << " threads per warp, but the module specifies "
                         << moduleThreadsPerWarp << " threads per warp.";
      }

      std::optional<int> moduleWarpsPerCTA = maybeLookupNumWarps(op);
      if (!moduleWarpsPerCTA) {
        return makeErr()
               << "Could not determine the number of warps per CTA. Operation "
                  "is not in a context with `ttg.num-warps`.";
      }
      int64_t layoutWarpsPerCTA = product(blocked.getWarpsPerCTA());
      if (layoutWarpsPerCTA != *moduleWarpsPerCTA) {
        return makeErr() << layout << ".\nLayout has a total of "
                         << layoutWarpsPerCTA
                         << " warps per CTA, but the context requires "
                         << *moduleWarpsPerCTA << " warps per CTA.";
      }

      if (blocked.getCTALayout().getCTAsPerCGA().size() > 0) {
        int moduleCTAsPerCGA = TritonGPUDialect::getNumCTAs(module);
        int64_t layoutCTAsPerCGA =
            product(blocked.getCTALayout().getCTAsPerCGA());
        if (layoutCTAsPerCGA != moduleCTAsPerCGA) {
          return makeErr() << layout << ".\nLayout has a total of "
                           << layoutCTAsPerCGA
                           << " CTAs per CGA, but the module specifies "
                           << moduleCTAsPerCGA << " CTAs per CGA.";
        }
      }
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Layout debug printing
//===----------------------------------------------------------------------===//

// Return N-D delinearized indices from a linear index.
static SmallVector<int64_t> delinearizeIndex(int64_t idx,
                                             ArrayRef<int64_t> shape) {
  SmallVector<int64_t> ret(shape.size());
  for (int i = shape.size() - 1; i >= 0; i--) {
    ret[i] = idx % shape[i];
    idx /= shape[i];
  }
  return ret;
}

// Returns how many padding characters are needed for the string representation
// of value to be the same as max.
static int numCharacterPadding(int value, int max) {
  return std::to_string(max).size() - std::to_string(value).size();
}

// return the string padded to have the same length as max.
static std::string paddedString(int value, int max) {
  int nbChar = numCharacterPadding(value, max);
  std::string str;
  for (int i = 0; i < nbChar; i++)
    str += " ";
  str += std::to_string(value);
  return str;
}

std::string getSharedLayoutStr(RankedTensorType tensorType,
                               bool useHWPointOfView) {
  auto layout = tensorType.getEncoding();
  if (!layout)
    return "";

  LinearLayout ll = triton::gpu::toLinearLayout(tensorType.getShape(), layout);

  StringAttr kOffset = StringAttr::get(tensorType.getContext(), "offset");
  StringAttr kBlock = StringAttr::get(tensorType.getContext(), "block");
  int64_t tensorSize = product(tensorType.getShape());
  unsigned numBlocks = getNumCTAs(layout);
  int32_t blockSize = tensorSize / numBlocks;

  // elementMapping is for the non-hw layout, offsetMapping for hw-layout
  std::vector<std::string> elementMapping(tensorSize);
  std::vector<std::string> offsetMapping;

  // Shared layouts are a mapping of (block, offset) --> (...)

  // We can just use a single int to index into elementMapping because
  // the 'swizzle' operation rearranges the indicies---and we want to keep it
  // that way
  int32_t idx = 0;
  // Enumerate all the offsets for each block
  for (int32_t block = 0; block < numBlocks; block++) {
    for (int32_t offset = 0; offset < blockSize; offset++) {
      SmallVector<std::pair<StringAttr, int32_t>> inputs = {
          {kBlock, block},
          {kOffset, offset},
      };

      SmallVector<std::pair<StringAttr, int32_t>> outputs = ll.apply(inputs);

      std::string sharedInfo = "(";
      std::string &value = elementMapping[idx];

      if (!value.empty())
        value += "|";

      value += "(";
      // We can build up both strings (for hw/non-hw layouts) concurrently
      for (int i = 0; i < outputs.size(); i++) {
        // Based on the formatting from LinearLayout::toString, the format for
        // the hw layout is slightly different. HW layouts use "," vs ":".
        if (i > 0) {
          sharedInfo += ",";
          value += ":";
        }
        auto index = paddedString(outputs[i].second, tensorType.getDimSize(i));
        sharedInfo += index;
        value += index;
      }
      value += ")";
      sharedInfo += ")";

      offsetMapping.push_back(sharedInfo);

      idx++;
    }
  }

  std::string layoutStr;

  if (!useHWPointOfView) {
    int rank = tensorType.getRank();
    bool newLine = true;
    for (int i = 0; i < tensorSize; i++) {
      auto indices = delinearizeIndex(i, tensorType.getShape());
      int numOpenBracket = 0;
      for (int j = rank - 1; j >= 0; j--) {
        if (indices[j] % tensorType.getDimSize(j) != 0)
          break;
        layoutStr += "[";
        numOpenBracket++;
      }
      if (newLine) {
        for (int j = 0; j < rank - numOpenBracket; j++)
          layoutStr += " ";
        newLine = false;
      }

      layoutStr += elementMapping[i];
      auto nextIndices = delinearizeIndex(i + 1, tensorType.getShape());
      for (int j = rank - 1; j >= 0; j--) {
        if (nextIndices[j] % tensorType.getDimSize(j) != 0)
          break;
        layoutStr += "]";
      }
      if (nextIndices.back() % tensorType.getShape().back() == 0) {
        layoutStr += "\n";
        newLine = true;
      } else {
        layoutStr += ",";
      }
    }
  } else {
    // For the HW view here, print the (block, offset) --> (r,c) mapping
    uint32_t idx = 0;
    for (int32_t block = 0; block < numBlocks; block++) {
      layoutStr += "Block: " + std::to_string(block) + ":\n";
      for (int32_t offset = 0; offset < (tensorSize / numBlocks); offset++) {
        layoutStr += "Offset: " + std::to_string(offset) + " -> ";
        layoutStr += offsetMapping[idx];
        layoutStr += "\n";
        idx++;
      }
    }
  }

  return layoutStr;
}

std::string getDistributedLayoutStr(RankedTensorType tensorType,
                                    bool useHWPointOfView) {
  auto layout = tensorType.getEncoding();
  if (!layout)
    return "";

  StringAttr kRegister = StringAttr::get(tensorType.getContext(), "register");
  StringAttr kLane = StringAttr::get(tensorType.getContext(), "lane");
  StringAttr kWarp = StringAttr::get(tensorType.getContext(), "warp");
  StringAttr kBlock = StringAttr::get(tensorType.getContext(), "block");

  LinearLayout ll = triton::gpu::toLinearLayout(tensorType.getShape(), layout);
  int64_t tensorSize = product(tensorType.getShape());
  std::vector<std::string> elementMapping(tensorSize);
  std::vector<std::string> threadMapping;
  unsigned threadsPerWarp = ll.getInDimSize(kLane);
  unsigned numWarpsPerCTA = ll.getInDimSize(kWarp);
  unsigned numBlocks = ll.getInDimSize(kBlock);
  int numElementsPerThreads = ll.getInDimSize(kRegister);
  for (int blockId = 0; blockId < numBlocks; ++blockId) {
    for (int warpId = 0; warpId < numWarpsPerCTA; warpId++) {
      for (int tid = 0; tid < threadsPerWarp; ++tid) {
        for (int idx = 0; idx < numElementsPerThreads; ++idx) {
          SmallVector<std::pair<StringAttr, int32_t>> inputs = {
              {kBlock, blockId},
              {kWarp, warpId},
              {kLane, tid},
              {kRegister, idx}};
          SmallVector<std::pair<StringAttr, int32_t>> outputs =
              ll.apply(inputs);
          int32_t linearizedIdx = 0;
          int stride = 1;
          for (int i = outputs.size() - 1; i >= 0; i--) {
            linearizedIdx += outputs[i].second * stride;
            stride *= tensorType.getDimSize(i);
          }
          std::string &value = elementMapping[linearizedIdx];
          if (!value.empty())
            value += "|";
          int padding = numCharacterPadding(blockId, numBlocks) +
                        numCharacterPadding(tid + warpId * threadsPerWarp,
                                            numWarpsPerCTA * threadsPerWarp) +
                        numCharacterPadding(idx, numElementsPerThreads);
          for (int i = 0; i < padding; i++)
            value += " ";
          if (numBlocks > 1)
            value += "B" + std::to_string(blockId) + ":";
          value += "T" + std::to_string(tid + warpId * threadsPerWarp) + ":" +
                   std::to_string(idx);
          // Now also compute the thread mapping.
          std::string threadInfo = "(";
          for (int i = 0; i < outputs.size(); i++) {
            if (i > 0)
              threadInfo += ",";
            threadInfo +=
                paddedString(outputs[i].second, tensorType.getDimSize(i));
          }
          threadInfo += ")";
          threadMapping.push_back(threadInfo);
        }
      }
    }
  }
  std::string layoutStr;
  if (!useHWPointOfView) {
    // Printing the threads containing each elements of the tensor.
    int rank = tensorType.getRank();
    bool newLine = true;
    for (int i = 0; i < tensorSize; i++) {
      auto indices = delinearizeIndex(i, tensorType.getShape());
      int numOpenBracket = 0;
      for (int j = rank - 1; j >= 0; j--) {
        if (indices[j] % tensorType.getDimSize(j) != 0)
          break;
        layoutStr += "[";
        numOpenBracket++;
      }
      if (newLine) {
        for (int j = 0; j < rank - numOpenBracket; j++)
          layoutStr += " ";
        newLine = false;
      }

      layoutStr += elementMapping[i];
      auto nextIndices = delinearizeIndex(i + 1, tensorType.getShape());
      for (int j = rank - 1; j >= 0; j--) {
        if (nextIndices[j] % tensorType.getDimSize(j) != 0)
          break;
        layoutStr += "]";
      }
      if (nextIndices.back() % tensorType.getShape().back() == 0) {
        layoutStr += "\n";
        newLine = true;
      } else {
        layoutStr += ", ";
      }
    }
  } else {
    // Printing the elements in each physical reg/warps/threads.
    for (int blockId = 0; blockId < numBlocks; blockId++) {
      if (numBlocks > 1)
        layoutStr += "Block" + std::to_string(blockId) + ":\n";
      for (int warpId = 0; warpId < numWarpsPerCTA; warpId++) {
        layoutStr += "Warp" + std::to_string(warpId) + ":\n";
        for (int idx = 0; idx < numElementsPerThreads; ++idx) {
          for (int tid = 0; tid < threadsPerWarp; ++tid) {
            int linearizedIdx =
                blockId * numWarpsPerCTA * threadsPerWarp *
                    numElementsPerThreads +
                warpId * threadsPerWarp * numElementsPerThreads +
                tid * numElementsPerThreads + idx;
            layoutStr += threadMapping[linearizedIdx];
            if (tid < threadsPerWarp - 1)
              layoutStr += ", ";
          }
          layoutStr += "\n";
        }
      }
    }
  }
  return layoutStr;
}

template <typename T>
llvm::SmallVector<T>
mlir::triton::gpu::expandMatrixShapeWithBatch(llvm::ArrayRef<T> s) {
  auto rank = s.size();
  assert(rank == 2 || rank == 3);
  if (rank == 3)
    return llvm::SmallVector<T>(s);
  return {1, s[0], s[1]};
}

template llvm::SmallVector<int64_t>
mlir::triton::gpu::expandMatrixShapeWithBatch<int64_t>(
    llvm::ArrayRef<int64_t> s);

template llvm::SmallVector<unsigned>
mlir::triton::gpu::expandMatrixShapeWithBatch<unsigned>(
    llvm::ArrayRef<unsigned> s);

llvm::SmallVector<unsigned>
mlir::triton::gpu::expandMatrixOrderWithBatch(llvm::ArrayRef<unsigned> o) {
  int rank = o.size();
  assert(rank == 2 || rank == 3);
  if (rank == 3)
    return llvm::SmallVector<unsigned>(o);
  llvm::SmallVector<unsigned> expanded(3, 0);
  for (int i = 0; i < rank; ++i)
    expanded[i] += o[i] + 1;
  return expanded;
}

std::string mlir::triton::gpu::getLayoutStr(RankedTensorType tensorType,
                                            bool useHWPointOfView) {
  auto layout = tensorType.getEncoding();

  // tensorType is needed later on (e.g., getDimSize(j)), so we still have to
  // pass it as a param
  if (mlir::isa<SharedEncodingTrait>(layout)) {
    return getSharedLayoutStr(tensorType, useHWPointOfView);
  } else if (auto distributedLayout =
                 mlir::dyn_cast<DistributedEncodingTrait>(layout)) {
    return getDistributedLayoutStr(tensorType, useHWPointOfView);
  }

  // else unimplemented, return error
  llvm::report_fatal_error("Unimplemented usage of getLayoutStr");
  return "";
}

void mlir::triton::gpu::dumpLayout(RankedTensorType tensorType) {
  llvm::errs() << getLayoutStr(tensorType, /*useHWPointOfView=*/false);
}

void mlir::triton::gpu::dumpHWLayout(RankedTensorType tensorType) {
  llvm::errs() << getLayoutStr(tensorType, /*useHWPointOfView=*/true);
}

namespace {
struct TensorModel
    : public triton::gpu::TensorOrMemDesc::ExternalModel<TensorModel,
                                                         RankedTensorType> {
  Type getElementType(Type pointer) const {
    return cast<RankedTensorType>(pointer).getElementType();
  }
  Attribute getEncoding(Type pointer) const {
    return cast<RankedTensorType>(pointer).getEncoding();
  }
  ArrayRef<int64_t> getShape(Type pointer) const {
    return cast<RankedTensorType>(pointer).getShape();
  }
  int64_t getRank(Type pointer) const {
    return cast<RankedTensorType>(pointer).getRank();
  }
  int64_t getElementTypeBitWidth(Type pointer) const {
    return cast<RankedTensorType>(pointer).getElementTypeBitWidth();
  }
};

struct MemDescModel
    : public triton::gpu::TensorOrMemDesc::ExternalModel<MemDescModel,
                                                         MemDescType> {
  Type getElementType(Type pointer) const {
    return cast<MemDescType>(pointer).getElementType();
  }
  Attribute getEncoding(Type pointer) const {
    return cast<MemDescType>(pointer).getEncoding();
  }
  ArrayRef<int64_t> getShape(Type pointer) const {
    return cast<MemDescType>(pointer).getShape();
  }
  int64_t getRank(Type pointer) const {
    return cast<MemDescType>(pointer).getShape().size();
  }
  int64_t getElementTypeBitWidth(Type pointer) const {
    return cast<MemDescType>(pointer).getElementType().getIntOrFloatBitWidth();
  }
};
} // namespace

void TritonGPUDialect::initialize() {
  registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonGPU/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"
#include "triton/Dialect/TritonGPU/IR/OpsEnums.cpp.inc"
      >();
  addInterfaces<TritonGPUOpAsmInterface>();
  addInterfaces<TritonGPUInferLayoutInterface>();
  addInterfaces<TritonGPUVerifyTensorLayoutInterface>();

  RankedTensorType::attachInterface<TensorModel>(*getContext());
  MemDescType::attachInterface<MemDescModel>(*getContext());
}

LogicalResult TritonGPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // Verify that dialect attributes are attached to the right ops.
  if (llvm::is_contained(
          {AttrNumCTAsName, AttrTargetName, AttrNumThreadsPerWarp},
          attr.getName()) &&
      !isa<ModuleOp>(op)) {
    return op->emitOpError("has unexpected attribute ")
           << attr.getName() << " which is expected only on `module` ops";
  }
  if (attr.getName() == AttrNumWarpsName && !isa<ModuleOp, FuncOp>(op)) {
    return op->emitOpError("has unexpected attribute ")
           << attr.getName()
           << " which is expected only on `module` or `tt.func` ops";
  }

  return success();
}

int TritonGPUDialect::getNumCTAs(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(AttrNumCTAsName))
    return attr.getInt();
  return 1;
}

int TritonGPUDialect::getThreadsPerWarp(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(AttrNumThreadsPerWarp))
    return attr.getInt();
  return 32;
}

std::optional<int> triton::gpu::maybeLookupNumWarps(Operation *op) {
  if (isa<ModuleOp, FuncOp>(op)) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(AttrNumWarpsName))
      return attr.getInt();
  } else if (auto partitions =
                 dyn_cast<WarpSpecializePartitionsOp>(op->getParentOp())) {
    unsigned idx = op->getParentRegion()->getRegionNumber();
    return partitions.getParentOp().getPartitionNumWarps()[idx];
  }
  if (Operation *parent = op->getParentOp())
    return maybeLookupNumWarps(parent);
  return {};
}

int triton::gpu::lookupNumWarps(Operation *op) {
  std::optional<int> numWarps = maybeLookupNumWarps(op);
  if (!numWarps) {
    op->emitOpError(
        "is not contained within a context that specifies the number of warps");
    llvm::report_fatal_error("failed to lookup the number of warps, the "
                             "surrounding module should contain a " +
                             Twine(AttrNumWarpsName) + " attribute");
  }
  return *numWarps;
}

int triton::gpu::lookupThreadsPerWarp(OpBuilder &rewriter) {
  assert(rewriter.getInsertionBlock() && "expected an insertion point");
  Operation *op = rewriter.getInsertionBlock()->getParentOp();
  while (op && !isa<ModuleOp>(op))
    op = op->getParentOp();
  assert(op && "cannot create thread ID outside of module");
  return triton::gpu::TritonGPUDialect::getThreadsPerWarp(cast<ModuleOp>(op));
}
