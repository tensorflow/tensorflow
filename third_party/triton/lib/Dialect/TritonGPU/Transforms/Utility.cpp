#include "triton/Analysis/Utility.h"

#include <fstream>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "ttg-utility"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace mlir {

using namespace triton;

SmallVector<unsigned, 3> mmaVersionToInstrShape(int version,
                                                const ArrayRef<int64_t> &shape,
                                                Type eltType, int numWarps) {
  if (version == 1)
    return {16, 16};
  else if (version == 2) {
    auto rank = shape.size();
    SmallVector<unsigned, 3> ret(rank, 1);
    ret[rank - 1] = 8;
    ret[rank - 2] = 16;
    return ret;
  } else if (version == 3) {
    unsigned k = 256 / eltType.getIntOrFloatBitWidth();
    if (shape[0] % 64 != 0 || shape[1] % 8 != 0) {
      assert(false && "type not supported");
      return {0, 0, 0};
    }
    SmallVector<unsigned> validN;

    // MMAv3 with larger instruction shape is preferred.
    if (llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E4M3FNUZType>(
            eltType) ||
        eltType.isF16() || eltType.isBF16() || eltType.isF32()) {
      validN.assign({256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176,
                     168, 160, 152, 144, 136, 128, 120, 112, 104, 96,  88,
                     80,  72,  64,  56,  48,  40,  32,  24,  16,  8});
    }

    if (eltType.isInteger(8)) {
      validN.assign({224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32,
                     24, 16, 8});
    }

    unsigned m = 16;
    unsigned mWarps = std::max<unsigned>(shape[0] / m, 1);
    unsigned nWarps = std::max<unsigned>(numWarps / mWarps, 1);
    unsigned maxN = std::max<unsigned>(shape[1] / nWarps, 8);
    for (auto n : validN) {
      if (shape[1] % n == 0 && n <= maxN) {
        return {m, n, k};
      }
    }

    assert(false && "type not supported");
    return {0, 0, 0};
  } else if (version == 5) {
    unsigned m = shape[0] >= 128 ? 128 : 64;
    // Right now default to distributing along N. TODO: For cases where we have
    // dot followed by reduction we need to be able to distribute along M.
    //    if (numWarps > 4)
    //      m = 64;
    unsigned n = shape[1] >= 256 ? 256 : shape[1];
    unsigned k = 256 / eltType.getIntOrFloatBitWidth();
    return {m, n, k};
  } else {
    assert(false && "version not supported");
    return {0, 0};
  }
}

bool isLoadFromTensorPtr(triton::LoadOp op) {
  return mlir::triton::isTensorPointerType(op.getPtr().getType());
}

SmallVector<unsigned, 4> argSort(const SmallVector<int64_t> &arr) {
  SmallVector<unsigned, 4> ret(arr.size());
  std::iota(ret.begin(), ret.end(), 0);
  std::stable_sort(ret.begin(), ret.end(),
                   [&](unsigned x, unsigned y) { return arr[x] > arr[y]; });
  return ret;
}

Value getMemAccessPtr(Operation *op) {
  if (auto ld = dyn_cast<triton::LoadOp>(op))
    return ld.getPtr();
  if (auto atomic = dyn_cast<triton::AtomicRMWOp>(op))
    return atomic.getPtr();
  if (auto atomic = dyn_cast<triton::AtomicCASOp>(op))
    return atomic.getPtr();
  if (auto copy = dyn_cast<triton::gpu::AsyncCopyGlobalToLocalOp>(op))
    return copy.getSrc();
  if (auto store = dyn_cast<triton::StoreOp>(op))
    return store.getPtr();
  return nullptr;
}

unsigned getElementBitWidth(RankedTensorType type) {
  auto typeForMem =
      isa<PointerType>(type.getElementType())
          ? cast<PointerType>(type.getElementType()).getPointeeType()
          : type.getElementType();
  return typeForMem.getIntOrFloatBitWidth();
}

unsigned getNumElementsPerThread(Operation *op, SmallVector<unsigned> order,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  Value val = getMemAccessPtr(op);
  auto ty = cast<RankedTensorType>(val.getType());
  auto shapePerCTA = triton::gpu::getShapePerCTA(ty);
  AxisInfo &valInfo = *axisInfoAnalysis.getAxisInfo(val);
  unsigned elemNumBits = getElementBitWidth(ty);
  unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
  unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
  unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
  unsigned maxContig =
      std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
  LDBG("elemNumBytes: " << elemNumBytes
                        << ", divisibility: " << maxMultipleBytes
                        << ", contig: " << valInfo.getContiguity(order[0])
                        << ", alignment: " << alignment);
  return currPerThread;
}

bool isView(Operation *op) {
  return isa<ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp>(op);
}

//===----------------------------------------------------------------------===//
// GraphDumper
//===----------------------------------------------------------------------===//

GraphDumper::NodeInfo GraphDumper::onValue(Value value) const {
  return {{"shape", "box"}, {"style", "filled"}, {"fillcolor", "white"}};
}

GraphDumper::NodeInfo GraphDumper::onOperation(Operation *op) const {
  return {{"shape", "ellipse"}, {"style", "filled"}, {"fillcolor", "white"}};
}

std::string GraphDumper::dump(triton::FuncOp func) const {
  llvm::SetVector<Value> values;
  llvm::SetVector<Operation *> operations;

  func.walk([&](Operation *op) {
    operations.insert(op);
    for (Value operand : op->getOperands())
      values.insert(operand);
    for (Value result : op->getResults())
      values.insert(result);
  });

  std::ostringstream oss;
  oss << "// Generated by Triton GraphDumper\n"
      << "\n"
      << "digraph {\n";

  oss << "    // Value Nodes\n";
  for (Value value : values)
    oss << "    " << emitValueNode(value) << "\n";
  oss << "\n";

  oss << "    // Operation Nodes\n";
  for (Operation *op : operations)
    oss << "    " << emitOperationNode(op) << "\n";
  oss << "\n";

  oss << "    // Edges\n";
  for (Operation *op : operations) {
    for (Value operand : op->getOperands())
      oss << "    " << emitEdge(getUniqueId(operand), getUniqueId(op)) << "\n";
    for (Value result : op->getResults())
      oss << "    " << emitEdge(getUniqueId(op), getUniqueId(result)) << "\n";
  }

  oss << "}\n";
  return oss.str();
}

void GraphDumper::dumpToFile(triton::FuncOp func,
                             const std::string &filename) const {
  std::ofstream ofs(filename);
  ofs << dump(func);
}

std::string GraphDumper::getShapeStr(const Type &type) const {
  std::ostringstream oss;
  oss << "[";
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    auto shape = tensorTy.getShape();
    for (unsigned i = 0; i < shape.size(); ++i) {
      if (i > 0)
        oss << ", ";
      oss << shape[i];
    }
  }
  oss << "]";
  return oss.str();
}

std::string GraphDumper::getUniqueId(Value value) const {
  std::ostringstream oss;
  oss << value.getImpl();
  return oss.str();
}

std::string GraphDumper::getUniqueId(Operation *op) const {
  std::ostringstream oss;
  oss << op;
  return oss.str();
}

std::string GraphDumper::emitNode(const std::string &id,
                                  const GraphDumper::NodeInfo info) const {
  std::ostringstream oss;
  oss << "\"" << id << "\" [";
  for (auto it = info.begin(); it != info.end(); ++it) {
    if (it != info.begin())
      oss << ", ";
    oss << it->first << " = \"" << it->second << "\"";
  }
  oss << "];";
  return oss.str();
}

std::string GraphDumper::emitEdge(const std::string &srcId,
                                  const std::string &destId) const {
  std::ostringstream oss;
  oss << "\"" << srcId << "\" -> \"" << destId << "\";";
  return oss.str();
}

std::string GraphDumper::emitValueNode(Value value) const {
  NodeInfo info = onValue(value);
  if (info.find("label") == info.end()) {
    std::string shapeStr = getShapeStr(value.getType());
    if (auto arg = mlir::dyn_cast<BlockArgument>(value))
      info["label"] =
          "BlockArg" + std::to_string(arg.getArgNumber()) + " " + shapeStr;
    else
      info["label"] = shapeStr;
  }
  return emitNode(getUniqueId(value), info);
}

std::string GraphDumper::emitOperationNode(Operation *op) const {
  NodeInfo info = onOperation(op);
  if (info.find("label") == info.end())
    info["label"] = op->getName().getStringRef().str();
  return emitNode(getUniqueId(op), info);
}

//===----------------------------------------------------------------------===//
// GraphLayoutMarker
//===----------------------------------------------------------------------===//

GraphDumper::NodeInfo GraphLayoutMarker::onValue(Value value) const {
  std::string color = getColor(value.getType());
  return {{"shape", "box"}, {"style", "filled"}, {"fillcolor", color}};
}

std::string GraphLayoutMarker::getColor(const Type &type) const {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    auto layout = tensorTy.getEncoding();
    if (isa<triton::gpu::BlockedEncodingAttr>(layout))
      return "green";
    else if (isa<triton::gpu::SliceEncodingAttr>(layout))
      return "yellow";
    else if (isa<triton::gpu::NvidiaMmaEncodingAttr>(layout))
      return "lightslateblue";
    else if (isa<triton::gpu::DotOperandEncodingAttr>(layout))
      return "orange";
    else if (isa<triton::gpu::SharedEncodingTrait>(layout))
      return "orangered";
    else {
      llvm::report_fatal_error("Unrecognized layout");
      return "unknown";
    }
  } else {
    return "white";
  }
}
// -------------------------------------------------------------------------- //

static Attribute inferDstEncoding(triton::ReduceOp op, Attribute encoding) {
  return triton::gpu::SliceEncodingAttr::get(
      op->getContext(), op.getAxis(),
      cast<ttg::DistributedEncodingTrait>(encoding));
}

static Attribute inferDstEncoding(triton::ExpandDimsOp op, Attribute encoding) {
  auto sliceEncoding = mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(encoding);
  if (!sliceEncoding)
    return {};
  if (op.getAxis() != sliceEncoding.getDim())
    return {};
  return sliceEncoding.getParent();
}

static Attribute inferDstEncoding(JoinOp op, Attribute srcEnc) {
  Attribute dstEnc;
  auto shape = op.getLhs().getType().getShape();
  if (srcEnc.getDialect()
          .getRegisteredInterface<DialectInferLayoutInterface>()
          ->inferDefaultJoinOpEncoding(srcEnc, dstEnc, shape,
                                       /*loc=*/std::nullopt)
          .succeeded()) {
    return dstEnc;
  }
  return {};
}

static Attribute inferDstEncoding(SplitOp op, Attribute srcEnc) {
  Attribute dstEnc;
  auto shape = op.getSrc().getType().getShape();
  if (srcEnc.getDialect()
          .getRegisteredInterface<DialectInferLayoutInterface>()
          ->inferSplitOpEncoding(srcEnc, dstEnc, shape,
                                 /*loc=*/std::nullopt)
          .succeeded()) {
    return dstEnc;
  }
  return {};
}

static Attribute inferSrcEncoding(triton::ReduceOp op, Attribute encoding) {
  auto sliceEncoding = mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(encoding);
  if (!sliceEncoding)
    return {};
  if (op.getAxis() != sliceEncoding.getDim())
    return {};
  return sliceEncoding.getParent();
}

static Attribute inferSrcEncoding(triton::ExpandDimsOp op, Attribute encoding) {
  return triton::gpu::SliceEncodingAttr::get(
      op->getContext(), op.getAxis(),
      cast<ttg::DistributedEncodingTrait>(encoding));
}

static Attribute inferSrcEncoding(JoinOp op, Attribute dstEnc) {
  // Split is the inverse of join.
  auto shape = op.getResult().getType().getShape();
  Attribute srcEnc;
  if (dstEnc.getDialect()
          .getRegisteredInterface<DialectInferLayoutInterface>()
          ->inferSplitOpEncoding(dstEnc, srcEnc, shape, /*loc=*/std::nullopt)
          .succeeded()) {
    return srcEnc;
  }
  return {};
}

static Attribute inferSrcEncoding(SplitOp op, Attribute dstEnc) {
  // Join is the inverse of split.
  Attribute srcEnc;
  auto shape = op.getOutLHS().getType().getShape();
  if (dstEnc.getDialect()
          .getRegisteredInterface<DialectInferLayoutInterface>()
          ->inferDefaultJoinOpEncoding(dstEnc, srcEnc, shape,
                                       /*loc=*/std::nullopt)
          .succeeded()) {
    return srcEnc;
  }
  return {};
}

static Attribute inferSrcEncoding(GatherOp op, Attribute dstEnc) {
  // The index encoding is the same as the output encoding.
  return dstEnc;
}

static Attribute inferTransOpDstEncoding(Attribute srcEnc,
                                         ArrayRef<int64_t> shape,
                                         ArrayRef<int32_t> order) {
  // Simply forward to the existing inferTransOpEncoding function.
  Attribute retEncoding;
  if (succeeded(
          srcEnc.getDialect()
              .getRegisteredInterface<triton::DialectInferLayoutInterface>()
              ->inferTransOpEncoding(srcEnc, shape, order, retEncoding))) {
    return retEncoding;
  }
  return {};
}

static Attribute inferDstEncoding(triton::gpu::Fp4ToFpOp op, Attribute srcEnc) {
  Attribute dstEnc;
  auto shape = op.getSrc().getType().getShape();
  auto result =
      srcEnc.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>()
          ->inferFp4ToFpOpEncoding(shape, op.getAxis(), srcEnc, dstEnc,
                                   /*fwdInference*/ true, std::nullopt);
  assert(succeeded(result));
  return dstEnc;
}

static Attribute inferSrcEncoding(triton::gpu::Fp4ToFpOp op, Attribute dstEnc) {
  Attribute srcEnc;
  auto shape = op.getSrc().getType().getShape();
  if (succeeded(
          dstEnc.getDialect()
              .getRegisteredInterface<triton::DialectInferLayoutInterface>()
              ->inferFp4ToFpOpEncoding(shape, op.getAxis(), dstEnc, srcEnc,
                                       /*fwdInference*/ false, std::nullopt))) {
    return srcEnc;
  }
  return {};
}

static Attribute inferDstEncoding(triton::TransposeOpInterface op,
                                  Attribute encoding) {
  return inferTransOpDstEncoding(
      encoding, cast<RankedTensorType>(op.getSrc().getType()).getShape(),
      op.getOrder());
}

static Attribute inferSrcEncoding(triton::TransposeOpInterface op,
                                  Attribute encoding) {
  // We want to solve for srcEnc in
  //   transpose(srcEnc, order) -> dstEnc.
  // Given the identity
  //   transpose(transpose(x, order), inverse(order)) == x,
  // we can see this is equivalent to
  //   transpose(dstEnc, inverse(order)) -> srcEnc.
  auto shape = cast<RankedTensorType>(op->getResult(0).getType()).getShape();
  return inferTransOpDstEncoding(encoding, shape,
                                 triton::inversePermutation(op.getOrder()));
}

static Attribute inferReshapeOpDstEncoding(ArrayRef<int64_t> srcShape,
                                           Attribute srcEnc,
                                           ArrayRef<int64_t> dstShape,
                                           bool allowReorder) {
  // We don't do anything smart to allow-reorder reshapes here.  They are
  // handled in OptimizeThreadLocality.
  if (allowReorder)
    return {};

  Attribute dstEnc;
  auto result =
      srcEnc.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>()
          ->inferReshapeOpEncoding(srcShape, srcEnc, dstShape, dstEnc,
                                   /*loc=*/std::nullopt);
  assert(succeeded(result));
  return dstEnc;
}

static Attribute inferDstEncoding(triton::ReshapeOp op, Attribute encoding) {
  return inferReshapeOpDstEncoding(op.getSrc().getType().getShape(), encoding,
                                   op.getType().getShape(),
                                   op.getAllowReorder());
}

static Attribute inferDstEncoding(GatherOp op, Attribute encoding) {
  // The output encoding is the same as the index encoding.
  // FIXME: This assumes `encoding` is the index encoding, which can be
  // different than the source encoding.
  return encoding;
}

static Attribute inferSrcEncoding(triton::ReshapeOp op, Attribute encoding) {
  // The encoding of x given the encoding of y in `reshape(x) -> y` is the same
  // as the encoding of x given the encoding of y in `reshape(y) -> x`.  It's an
  // invariant of inferReshapeOpNoReorderEncoding that it's symmetric in this
  // way.
  return inferReshapeOpDstEncoding(op.getType().getShape(), encoding,
                                   op.getSrc().getType().getShape(),
                                   op.getAllowReorder());
}

static bool isSingleValue(Value value) {
  // Don't consider load as expensive if it is loading a scalar.
  if (auto tensorTy = dyn_cast<RankedTensorType>(value.getType()))
    return tensorTy.getNumElements() == 1;
  // TODO: Handle other cases.
  // For example, when ptr is a tensor of single value.
  // It means that ptr is a resultant of broadcast or generated through
  // a chain of broadcast and other operations.
  // Rematerialize it without considering contiguous memory access pattern is
  // fine.
  return true;
}

Attribute inferSrcEncoding(Operation *op, Attribute encoding) {
  if (isa<triton::ScanOp>(op)) {
    // Scan only supports blocked encoding at the moment.
    if (!isa<triton::gpu::BlockedEncodingAttr>(encoding))
      return {};
  }
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::SameLoadStoreOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>() ||
      isa<scf::WhileOp, scf::YieldOp, scf::ConditionOp,
          nvidia_gpu::WarpGroupDotWaitOp>(op)) {
    return encoding;
  }

  if (auto reduceOp = dyn_cast<triton::ReduceOp>(op))
    return inferSrcEncoding(reduceOp, encoding);
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op))
    return inferSrcEncoding(expand, encoding);
  if (auto join = dyn_cast<triton::JoinOp>(op))
    return inferSrcEncoding(join, encoding);
  if (auto split = dyn_cast<triton::SplitOp>(op))
    return inferSrcEncoding(split, encoding);
  if (auto trans = dyn_cast<triton::TransposeOpInterface>(op))
    return inferSrcEncoding(trans, encoding);
  if (auto reshape = dyn_cast<triton::ReshapeOp>(op))
    return inferSrcEncoding(reshape, encoding);
  if (auto gather = dyn_cast<triton::GatherOp>(op))
    return inferSrcEncoding(gather, encoding);
  if (auto fp4ToFp = dyn_cast<triton::gpu::Fp4ToFpOp>(op))
    return inferSrcEncoding(fp4ToFp, encoding);

  return {};
}

Attribute inferDstEncoding(Operation *op, Attribute encoding) {
  if (isa<triton::ScanOp>(op)) {
    if (!isa<triton::gpu::BlockedEncodingAttr>(encoding))
      return {};
  }
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::SameLoadStoreOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>() ||
      isa<scf::WhileOp, scf::ForOp, scf::YieldOp, scf::ConditionOp,
          nvidia_gpu::WarpGroupDotWaitOp>(op))
    return encoding;
  if (auto reduceOp = dyn_cast<triton::ReduceOp>(op))
    return inferDstEncoding(reduceOp, encoding);
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op))
    return inferDstEncoding(expand, encoding);
  if (auto join = dyn_cast<triton::JoinOp>(op))
    return inferDstEncoding(join, encoding);
  if (auto split = dyn_cast<triton::SplitOp>(op))
    return inferDstEncoding(split, encoding);
  if (auto trans = dyn_cast<triton::TransposeOpInterface>(op))
    return inferDstEncoding(trans, encoding);
  if (auto reshape = dyn_cast<triton::ReshapeOp>(op))
    return inferDstEncoding(reshape, encoding);
  if (auto gather = dyn_cast<triton::GatherOp>(op))
    return inferDstEncoding(gather, encoding);
  if (auto fp4ToFp = dyn_cast<triton::gpu::Fp4ToFpOp>(op))
    return inferDstEncoding(fp4ToFp, encoding);

  return {};
}

bool isExpensiveLoadOrStore(Operation *op) {
  // Case 1: Pointer of tensor is always expensive
  auto operandType = op->getOperand(0).getType();
  if (triton::isTensorPointerType(operandType))
    return true;
  // Case 2a: A size 1 tensor is not expensive since all threads will load the
  // same
  if (isSingleValue(op->getOperand(0)))
    return false;
  // Case 2b: Tensor of pointers has more threads than elements
  // we can presume a high hit-rate that makes it cheap to load
  auto ptrType = cast<RankedTensorType>(op->getOperand(0).getType());
  auto mod = op->getParentOfType<ModuleOp>();
  int numWarps = triton::gpu::lookupNumWarps(op);
  int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  if (ptrType.getNumElements() < numWarps * threadsPerWarp)
    return false;
  return true;
}

bool isExpensiveToRemat(Operation *op, Attribute &targetEncoding) {
  if (!op)
    return true;
  if (isa<triton::LoadOp, triton::StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  if (isa<triton::CatOp>(op))
    return triton::gpu::isExpensiveCat(cast<triton::CatOp>(op), targetEncoding);
  if (isa<triton::gpu::AsyncCopyGlobalToLocalOp, triton::AtomicRMWOp,
          triton::AtomicCASOp, triton::DotOp>(op))
    return true;
  if (isa<scf::YieldOp, scf::ForOp, scf::IfOp, scf::WhileOp, scf::ConditionOp>(
          op))
    return true;
  return false;
}

bool canFoldIntoConversion(Operation *op, Attribute targetEncoding) {
  if (isa<triton::CatOp>(op))
    return !triton::gpu::isExpensiveCat(cast<triton::CatOp>(op),
                                        targetEncoding);
  if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    if (mlir::isa<triton::gpu::NvidiaMmaEncodingAttr>(targetEncoding)) {
      auto srcEncoding = convert.getSrc().getType().getEncoding();
      if (targetEncoding != srcEncoding)
        return false;
    }
    return true;
  }

  if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
    auto reshapeDstType = reshape.getType();
    RankedTensorType newDstType =
        RankedTensorType::get(reshapeDstType.getShape(),
                              reshapeDstType.getElementType(), targetEncoding);
    return reshape.getAllowReorder() && !reshape.getEfficientLayout() &&
           !triton::gpu::isExpensiveView(reshape.getSrc().getType(),
                                         newDstType);
  }
  return isa<triton::gpu::ConvertLayoutOp, arith::ConstantOp,
             triton::MakeRangeOp, triton::SplatOp, triton::HistogramOp,
             triton::gpu::LocalAllocOp, triton::gpu::LocalLoadOp,
             triton::gpu::LocalStoreOp>(op);
}

scf::ForOp replaceForOpWithNewSignature(
    OpBuilder &rewriter, scf::ForOp loop, ValueRange newIterOperands,
    SmallVectorImpl<std::tuple<Value, Value>> &replacements) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loop);

  // Create a new loop before the existing one, with the extra operands.
  auto operands = llvm::to_vector<4>(loop.getInitArgs());
  operands.append(newIterOperands.begin(), newIterOperands.end());
  scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      operands);
  newLoop->setAttrs(loop->getAttrs());
  newLoop.getBody()->erase();
  newLoop.getRegion().getBlocks().splice(
      newLoop.getRegion().getBlocks().begin(), loop.getRegion().getBlocks());
  for (Value operand : newIterOperands)
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());

  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    replacements.push_back(it);
  return newLoop;
}

scf::ForOp replaceForOpWithNewSignature(OpBuilder &rewriter, scf::ForOp loop,
                                        ValueRange newIterOperands) {
  SmallVector<std::tuple<Value, Value>> replacements;
  auto newForOp = replaceForOpWithNewSignature(rewriter, loop, newIterOperands,
                                               replacements);
  for (auto [result, value] : replacements) {
    result.replaceAllUsesWith(value);
  }
  return newForOp;
}

scf::WhileOp replaceWhileOpWithNewSignature(
    OpBuilder &rewriter, scf::WhileOp loop, ValueRange newIterOperands,
    TypeRange newResultTypes,
    SmallVectorImpl<std::tuple<Value, Value>> &replacements) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loop);

  // Create a new loop before the existing one, with the extra operands.
  auto operands = llvm::to_vector<4>(loop.getInits());
  operands.append(newIterOperands.begin(), newIterOperands.end());

  // Result and operand types
  SmallVector<Type> resultTypes;
  SmallVector<Type> argsTypesBefore;
  for (auto res : loop.getResults())
    resultTypes.push_back(res.getType());
  for (auto type : newResultTypes)
    resultTypes.push_back(type);
  for (Value operand : operands)
    argsTypesBefore.push_back(operand.getType());
  scf::WhileOp newLoop =
      rewriter.create<scf::WhileOp>(loop.getLoc(), resultTypes, operands);
  newLoop->setAttrs(loop->getAttrs());

  SmallVector<Location> bbArgLocsBefore(argsTypesBefore.size(), loop.getLoc());
  SmallVector<Location> bbArgLocsAfter(resultTypes.size(), loop.getLoc());
  rewriter.createBlock(&newLoop.getBefore(), {}, argsTypesBefore,
                       bbArgLocsBefore);
  rewriter.createBlock(&newLoop.getAfter(), {}, resultTypes, bbArgLocsAfter);

  // Copy regions
  for (int i = 0; i < loop.getNumRegions(); ++i)
    newLoop->getRegion(i).front().getOperations().splice(
        newLoop->getRegion(i).front().getOperations().begin(),
        loop->getRegion(i).front().getOperations());

  // Remap arguments
  for (auto [oldArg, newArg] : llvm::zip(
           loop.getBeforeArguments(), newLoop.getBeforeArguments().take_front(
                                          loop.getBeforeArguments().size())))
    oldArg.replaceAllUsesWith(newArg);
  for (auto [oldArg, newArg] : llvm::zip(loop.getAfterArguments(),
                                         newLoop.getAfterArguments().take_front(
                                             loop.getAfterArguments().size())))
    oldArg.replaceAllUsesWith(newArg);

  // Stack the new results
  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    replacements.push_back(it);

  return newLoop;
}

scf::WhileOp replaceWhileOpWithNewSignature(OpBuilder &rewriter,
                                            scf::WhileOp loop,
                                            ValueRange newIterOperands,
                                            TypeRange newResultTypes) {
  SmallVector<std::tuple<Value, Value>> replacements;
  auto newWhileOp = replaceWhileOpWithNewSignature(
      rewriter, loop, newIterOperands, newResultTypes, replacements);
  for (auto &kv : replacements) {
    std::get<0>(kv).replaceAllUsesWith(std::get<1>(kv));
  }
  return newWhileOp;
}

scf::IfOp replaceIfOpWithNewSignature(
    OpBuilder &rewriter, scf::IfOp ifOp, TypeRange newResultTypes,
    SmallVectorImpl<std::tuple<Value, Value>> &replacements) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(ifOp);

  // Create a new loop before the existing one, with the extra operands.
  auto resultTypes = llvm::to_vector<4>(ifOp.getResults().getTypes());
  resultTypes.append(newResultTypes.begin(), newResultTypes.end());
  scf::IfOp newIf = rewriter.create<scf::IfOp>(ifOp.getLoc(), resultTypes,
                                               ifOp.getCondition());
  newIf->setAttrs(ifOp->getAttrs());

  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  newIf.getElseRegion().takeBody(ifOp.getElseRegion());
  scf::IfOp::ensureTerminator(newIf.getThenRegion(), rewriter, ifOp.getLoc());
  scf::IfOp::ensureTerminator(newIf.getElseRegion(), rewriter, ifOp.getLoc());

  for (auto it : llvm::zip(ifOp.getResults(),
                           newIf.getResults().take_front(ifOp.getNumResults())))
    replacements.push_back(it);
  return newIf;
}

void appendToForOpYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands());
  operands.append(newOperands.begin(), newOperands.end());

  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

scf::IfOp replaceIfOpWithNewSignature(OpBuilder &rewriter, scf::IfOp ifOp,
                                      TypeRange newResultTypes) {
  SmallVector<std::tuple<Value, Value>> replacements;
  auto newIfOp =
      replaceIfOpWithNewSignature(rewriter, ifOp, newResultTypes, replacements);
  for (auto &kv : replacements)
    std::get<0>(kv).replaceAllUsesWith(std::get<1>(kv));
  return newIfOp;
}

Operation *cloneWithInferType(mlir::OpBuilder &rewriter, Operation *op,
                              IRMapping &mapping) {
  Operation *newOp = rewriter.clone(*op, mapping);
  // if input types haven't changed, we're done
  bool preserveTypes =
      std::all_of(op->operand_begin(), op->operand_end(), [&](Value v) {
        return !mapping.contains(v) ||
               v.getType() == mapping.lookup(v).getType();
      });
  if (preserveTypes)
    return newOp;

  if (newOp->getNumResults() == 0)
    return newOp;
  auto origType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto argType = dyn_cast<RankedTensorType>(newOp->getOperand(0).getType());
  if (!origType || !argType)
    return newOp;
  auto newType = RankedTensorType::get(
      origType.getShape(), origType.getElementType(), argType.getEncoding());
  newOp->getResult(0).setType(newType);
  auto typeInfer = dyn_cast<InferTypeOpInterface>(newOp);
  if (typeInfer) {
    SmallVector<Type, 1> newTypes;
    auto success = typeInfer.inferReturnTypes(
        newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
        newOp->getAttrDictionary(), newOp->getPropertiesStorage(),
        newOp->getRegions(), newTypes);
    if (succeeded(success)) {
      for (size_t i = 0; i < newTypes.size(); i++)
        newOp->getResult(i).setType(newTypes[i]);
    }
  }
  return newOp;
}

// Check if the convert will be performed by reordering registers.
static bool isFreeConvert(Operation *op) {
  auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!convertOp)
    return false;
  return cvtReordersRegisters(convertOp.getSrc().getType(),
                              convertOp.getType());
}

LogicalResult getConvertBackwardSlice(
    OpOperand &root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation,
    std::function<Value(OpOperand &, Attribute)> getExistingConversion) {
  DenseSet<std::pair<OpOperand *, Attribute>> seen;
  SmallVector<std::pair<OpOperand *, Attribute>> queue;

  auto enqueue = [&](OpOperand &operand, Attribute encoding) {
    auto x = std::make_pair(&operand, encoding);
    if (!seen.insert(x).second) {
      return; // Already enqueued, skip
    }
    queue.push_back(x);
  };
  enqueue(root, rootEncoding);

  auto updateLayout = [&](Value value, Attribute encoding) {
    assert((isa<RankedTensorType>(value.getType())));
    slice.insert(value);
    Attribute &existing = layout[value];
    if (existing && existing != encoding)
      return failure();
    existing = encoding;
    return success();
  };

  while (!queue.empty()) {
    auto [currentValueUse, encoding] = queue.back();
    Value currentValue = currentValueUse->get();
    queue.pop_back();
    if (!isa<RankedTensorType>(currentValue.getType()))
      continue;
    // Skip propagating through for op results for now.
    // TODO: enable this based on needs.
    if (currentValue.getDefiningOp<scf::ForOp>())
      return failure();
    if (failed(updateLayout(currentValue, encoding)))
      return failure();

    Value existing;
    if (getExistingConversion &&
        (existing = getExistingConversion(*currentValueUse, encoding))) {
      if (failed(updateLayout(existing, encoding)))
        return failure();
      currentValue = existing;
    }

    if (auto ifOp = currentValue.getDefiningOp<scf::IfOp>()) {
      if (stopPropagation && stopPropagation(ifOp))
        continue;
      unsigned argIdx = mlir::cast<OpResult>(currentValue).getResultNumber();

      OpOperand &thenValue = ifOp.thenYield()->getOpOperand(argIdx);
      OpOperand &elseValue = ifOp.elseYield()->getOpOperand(argIdx);

      enqueue(thenValue, encoding);
      enqueue(elseValue, encoding);

      continue;
    }
    if (auto *definingOp = currentValue.getDefiningOp()) {
      // If the op has multiple results we need to update all results layout.
      for (Value result : definingOp->getResults()) {
        if (result == currentValue || !isa<RankedTensorType>(result.getType()))
          continue;
        if (failed(updateLayout(result, encoding)))
          return failure();
      }
      if (isFreeConvert(definingOp)) {
        enqueue(definingOp->getOpOperand(0), encoding);
        continue;
      }
      if (canFoldIntoConversion(definingOp, encoding))
        continue;
      if (stopPropagation && stopPropagation(definingOp))
        continue;
      if (isa<triton::CatOp>(definingOp))
        return failure();
      if (auto gather = dyn_cast<GatherOp>(definingOp)) {
        // Specially handle gather since its transfer function only applies
        // between its index operand and result.
        auto srcEncoding = inferSrcEncoding(gather, encoding);
        if (!srcEncoding)
          return failure();
        enqueue(gather.getIndicesMutable(), srcEncoding);
        continue;
      }
      for (auto [i, operand] : llvm::enumerate(definingOp->getOpOperands())) {
        auto srcEncoding = inferSrcEncoding(definingOp, encoding);
        if (!srcEncoding)
          return failure();
        enqueue(operand, srcEncoding);
      }
      continue;
    }
    auto blockArg = cast<BlockArgument>(currentValue);
    Block *block = blockArg.getOwner();
    Operation *parentOp = block->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      OpOperand *initOperand = forOp.getTiedLoopInit(blockArg);
      OpOperand &yieldOperand = forOp.getBody()->getTerminator()->getOpOperand(
          blockArg.getArgNumber() - forOp.getNumInductionVars());
      enqueue(*initOperand, encoding);
      enqueue(yieldOperand, encoding);
      continue;
    }
    // TODO: add support for WhileOp and other region types.
    return failure();
  }
  return success();
}

// TODO(thomas): this is duplicated with what is in GPUToLLVM
//  Convert an \param index to a multi-dim coordinate given \param shape and
//  \param order.
SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = triton::applyPermutation(shape, order);
  auto reorderedMultiDim = delinearize(b, loc, linear, reordered);
  SmallVector<Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  if (rank == 1) {
    multiDim[0] = linear;
  } else {
    Value remained = linear;
    for (auto &&en : llvm::enumerate(shape.drop_back())) {
      auto dimSize = b.create<arith::ConstantIntOp>(loc, en.value(), 32);
      multiDim[en.index()] = b.create<arith::RemSIOp>(loc, remained, dimSize);
      remained = b.create<arith::DivSIOp>(loc, remained, dimSize);
    }
    multiDim[rank - 1] = remained;
  }
  return multiDim;
}

Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order) {
  return linearize(b, loc, triton::applyPermutation(multiDim, order),
                   triton::applyPermutation(shape, order));
}

Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape) {
  auto rank = multiDim.size();
  Value linear = b.create<arith::ConstantIntOp>(loc, 0, 32);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      Value dimSize = b.create<arith::ConstantIntOp>(loc, dimShape, 32);
      linear = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, linear, dimSize), dim);
    }
  }
  return linear;
}

bool isPureUnaryInlineAsm(Operation *op) {
  auto inlineAsmOp = dyn_cast<ElementwiseInlineAsmOp>(op);
  if (!inlineAsmOp)
    return false;
  return op->getNumOperands() == 1 && op->getNumResults() == 1 &&
         inlineAsmOp.getPure();
}

int getNVIDIAComputeCapability(Operation *module) {
  StringAttr targetAttr =
      module->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
  assert(targetAttr && "Expected a target attribute on the module operation");

  StringRef ref = targetAttr.strref();
  assert(ref.starts_with("cuda:") &&
         "expected target attribute to be prefixed with \"cuda:\"");

  StringRef capabilityStr = ref.drop_front(5); // drop the "cuda:"
  int computeCapability;
  bool parseError = capabilityStr.getAsInteger(10, computeCapability);
  assert(!parseError &&
         "invalid compute capability string in target attribute");

  return computeCapability;
}

StringRef getAMDArch(Operation *module) {
  StringAttr targetAttr =
      module->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
  assert(targetAttr && "Expected a target attribute on the module operation");

  StringRef ref = targetAttr.strref();
  assert(ref.starts_with("hip:") &&
         "expected target attribute to be prefixed with \"hip:\"");

  return ref.drop_front(4); // drop the "hip:"
}

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return the shared encoding that needs to be
// used to be compatible with users' layouts. If there are incompatible shared
// encodings, set incompatible to true.
std::optional<ttg::SwizzledSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value val, bool &incompatible) {
  ttg::SwizzledSharedEncodingAttr attr;
  incompatible = false;
  for (Operation *user : val.getUsers()) {
    ttg::SwizzledSharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return std::nullopt;
    if (auto memDesc =
            dyn_cast<triton::gpu::MemDescType>(user->getResult(0).getType())) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr =
          dyn_cast<ttg::SwizzledSharedEncodingAttr>(memDesc.getEncoding());
      if (!tempAttr)
        return std::nullopt;
      if (!getSharedEncIfAllUsersAreDotEnc(user->getResult(0), incompatible)
               .has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;
      auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
          cast<triton::gpu::TensorOrMemDesc>(user->getResult(0).getType())
              .getEncoding());
      if (!dotOpEnc)
        return std::nullopt;
      auto srcTy = cast<triton::gpu::TensorOrMemDesc>(val.getType());
      auto CTALayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = getOrderForMemory(srcTy);
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      tempAttr = ttg::SwizzledSharedEncodingAttr::get(
          val.getContext(), dotOpEnc, srcTy.getShape(), order, CTALayout,
          bitWidth, /*needTrans=*/false);
    }
    // Check that the shared encodings needed by the users are compatible.
    if (attr != nullptr && attr != tempAttr) {
      incompatible = true;
      return std::nullopt;
    }
    attr = tempAttr;
  }
  return attr;
}

namespace {

/// Detect dead arguments in scf.for op by assuming all the values are dead and
/// propagate liveness property.
struct ForOpDeadArgElimination : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    Block &block = *forOp.getBody();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
    // Assume that nothing is live at the beginning and mark values as live
    // based on uses.
    DenseSet<Value> aliveValues;
    SmallVector<Value> queue;
    // Helper to mark values as live and add them to the queue of value to
    // propagate if it is the first time we detect the value as live.
    auto markLive = [&](Value val) {
      if (!forOp->isAncestor(val.getParentRegion()->getParentOp()))
        return;
      if (aliveValues.insert(val).second)
        queue.push_back(val);
    };
    // Mark all yield operands as live if the associated forOp result has any
    // use.
    for (auto result : llvm::enumerate(forOp.getResults())) {
      if (!result.value().use_empty())
        markLive(yieldOp.getOperand(result.index()));
    }
    if (aliveValues.size() == forOp.getNumResults())
      return failure();
    // Operations with side-effects are always live. Mark all theirs operands as
    // live.
    block.walk([&](Operation *op) {
      if (!isa<scf::YieldOp, scf::ForOp>(op) && !wouldOpBeTriviallyDead(op)) {
        for (Value operand : op->getOperands())
          markLive(operand);
      }
    });
    // Propagate live property until reaching a fixed point.
    while (!queue.empty()) {
      Value value = queue.pop_back_val();
      if (auto nestedFor = value.getDefiningOp<scf::ForOp>()) {
        auto result = mlir::cast<OpResult>(value);
        OpOperand &forOperand = *nestedFor.getTiedLoopInit(result);
        markLive(forOperand.get());
        auto nestedYieldOp =
            cast<scf::YieldOp>(nestedFor.getBody()->getTerminator());
        Value nestedYieldOperand =
            nestedYieldOp.getOperand(result.getResultNumber());
        markLive(nestedYieldOperand);
        continue;
      }
      if (auto nestedIf = value.getDefiningOp<scf::IfOp>()) {
        auto result = mlir::cast<OpResult>(value);
        // mark condition as live.
        markLive(nestedIf.getCondition());
        for (scf::YieldOp nestedYieldOp :
             {nestedIf.thenYield(), nestedIf.elseYield()}) {
          Value nestedYieldOperand =
              nestedYieldOp.getOperand(result.getResultNumber());
          markLive(nestedYieldOperand);
        }
        continue;
      }
      if (Operation *def = value.getDefiningOp()) {
        // TODO: support while ops.
        if (isa<scf::WhileOp>(def))
          return failure();
        for (Value operand : def->getOperands())
          markLive(operand);
        continue;
      }
      // If an argument block is live then the associated yield operand and
      // forOp operand are live.
      auto arg = mlir::cast<BlockArgument>(value);
      if (auto forOwner = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        if (arg.getArgNumber() < forOwner.getNumInductionVars())
          continue;
        unsigned iterIdx = arg.getArgNumber() - forOwner.getNumInductionVars();
        Value yieldOperand =
            forOwner.getBody()->getTerminator()->getOperand(iterIdx);
        markLive(yieldOperand);
        markLive(forOwner.getInitArgs()[iterIdx]);
      }
    }
    SmallVector<unsigned> deadArg;
    for (auto yieldOperand : llvm::enumerate(yieldOp->getOperands())) {
      if (aliveValues.contains(yieldOperand.value()))
        continue;
      if (yieldOperand.value() == block.getArgument(yieldOperand.index() + 1))
        continue;

      // The yield operand might live outside the loop, e.g.
      //   %init = ...
      //   %x = ...
      //   %y = for iter_args(%unused = %init) {
      //     yield %x
      //   }
      //
      // In this case, the loop returns %x if it runs 1 or more times, and
      // otherwise it returns %init.  We cowardly refuse to remove this operand
      // from the yield.  (We could, but we'd need to prove that the loop runs 0
      // or >=1 times.)
      //
      // As a special case, if it doesn't matter whether the loop runs 0 or >=1
      // times (because the loop returns the same value in both cases) then we
      // can still mark the operand as dead. This occurs in the above example
      // when %init is the same as %x.
      if (!forOp->isAncestor(
              yieldOperand.value().getParentRegion()->getParentOp()) &&
          yieldOperand.value() != forOp.getInitArgs()[yieldOperand.index()])
        continue;

      deadArg.push_back(yieldOperand.index());
    }
    if (deadArg.empty())
      return failure();
    rewriter.modifyOpInPlace(forOp, [&]() {
      // For simplicity we just change the dead yield operand to use the
      // associated argument and leave the operations and argument removal to
      // dead code elimination.
      for (unsigned deadArgIdx : deadArg) {
        BlockArgument arg = block.getArgument(deadArgIdx + 1);
        yieldOp.setOperand(deadArgIdx, arg);
      }
    });
    return success();
  }
};

} // namespace

void populateForOpDeadArgumentElimination(RewritePatternSet &patterns) {
  patterns.add<ForOpDeadArgElimination>(patterns.getContext());
}

ttg::LocalAllocOp findShmemAlloc(Value operand) {
  // If it's a shmem operand, it must either be defined outside the loop, or
  // come from an MemDescSubview op. Only ConvertLayout and Trans ops are
  // allowed in between.
  Value transitiveOperand = operand;
  while (
      isa_and_nonnull<ttg::ConvertLayoutOp, tt::TransOp, ttg::MemDescTransOp>(
          transitiveOperand.getDefiningOp()) ||
      isa<BlockArgument>(transitiveOperand)) {
    if (auto blockArg = dyn_cast<BlockArgument>(transitiveOperand)) {
      assert(isa<scf::ForOp>(blockArg.getOwner()->getParentOp()) &&
             "Block argument must come from a for loop");
      transitiveOperand =
          cast<scf::YieldOp>(blockArg.getOwner()->getTerminator())
              .getOperand(blockArg.getArgNumber() - 1);
    } else {
      transitiveOperand = transitiveOperand.getDefiningOp()->getOperand(0);
    }
  }
  if (auto subView = dyn_cast_or_null<ttg::MemDescSubviewOp>(
          transitiveOperand.getDefiningOp())) {
    // Multi-buffered operand
    return dyn_cast_or_null<ttg::LocalAllocOp>(
        subView.getSrc().getDefiningOp());
  } else {
    // Single bufferred operand that does not require a subview (not loaded in
    // the loop)
    return dyn_cast_or_null<ttg::LocalAllocOp>(
        transitiveOperand.getDefiningOp());
  }
  return nullptr;
}

SmallVector<Operation *>
getMMAsWithMultiBufferredOperands(scf::ForOp forOp,
                                  SmallVector<Operation *> &mmaOps) {
  // The A and B operands of the mmaOp should be multi-buffered
  SmallVector<Operation *> eligible;
  for (auto mmaOp : mmaOps) {
    auto a = findShmemAlloc(mmaOp->getOperand(0));
    auto b = findShmemAlloc(mmaOp->getOperand(1));
    if (a && forOp.isDefinedOutsideOfLoop(a) && b &&
        forOp.isDefinedOutsideOfLoop(b)) {
      eligible.push_back(mmaOp);
    }
  }

  return eligible;
}

} // namespace mlir
