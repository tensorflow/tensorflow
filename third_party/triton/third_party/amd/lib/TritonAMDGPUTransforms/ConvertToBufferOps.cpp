#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using ::mlir::LLVM::AMD::getVectorSize;
using mlir::triton::AMD::ISAFamily;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {

bool verifyNonSmallerByAssumption(
    Value expr, const DenseSet<Value> &assumptions,
    const std::function<bool(Value)> &matchesOther) {
  for (Value assume : assumptions) {
    if (auto cmpOp = assume.getDefiningOp<arith::CmpIOp>()) {
      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::sgt: {
        if (cmpOp.getLhs() == expr && matchesOther(cmpOp.getRhs())) {
          LDBG("  " << expr << " non-neg by assumption " << cmpOp);
          return true;
        }
        break;
      }
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::slt: {
        if (cmpOp.getRhs() == expr && matchesOther(cmpOp.getLhs())) {
          LDBG("  " << expr << " non-neg by assumption " << cmpOp);
          return true;
        }
        break;
      }
      default:
        break;
      }
    }
  }
  return false;
}

bool verifyNonNegativeByAssumption(Value expr,
                                   const DenseSet<Value> &assumptions) {
  return verifyNonSmallerByAssumption(expr, assumptions, [](auto otherExpr) {
    APInt cst;
    return matchPattern(otherExpr, m_ConstantInt(&cst)) && cst.isNonNegative();
  });
}

bool verifyNonSmallerByAssumption(Value expr,
                                  const DenseSet<Value> &assumptions,
                                  Value other) {
  return verifyNonSmallerByAssumption(
      expr, assumptions, [&](auto otherAssum) { return otherAssum == other; });
}

bool verifyNonNegativeExpr(Value expr, const DenseSet<Value> &assumptions,
                           std::shared_ptr<DataFlowSolver> solver) {
  LDBG("Determing if non-negative: " << expr);

  if (!llvm::isa<mlir::BlockArgument>(expr) &&
      succeeded(dataflow::staticallyNonNegative(*solver, expr))) {
    return true;
  }

  // Check if the expression is contained in any assumption
  if (verifyNonNegativeByAssumption(expr, assumptions)) {
    return true;
  }

  // Recurse if the operation is defined
  Operation *op = expr.getDefiningOp();
  if (!op) {
    LDBG("  No defining op, assuming possibly negative");
    return false;
  }

  bool nonNegative =
      llvm::TypeSwitch<Operation *, bool>(expr.getDefiningOp())
          // Various unary triton ops that don't change the sign of the operand
          .Case<triton::TransOp, triton::SplitOp, triton::BroadcastOp,
                triton::ExpandDimsOp, triton::SplatOp, triton::ReshapeOp,
                triton::gpu::ConvertLayoutOp>([&](auto unaryOp) {
            return verifyNonNegativeExpr(unaryOp.getOperand(), assumptions,
                                         solver);
          })
          .Case<triton::GatherOp>([&](auto gatherOp) {
            return verifyNonNegativeExpr(gatherOp.getSrc(), assumptions,
                                         solver);
          })
          // Joining two non-negative tensors is still non-negative
          .Case<triton::JoinOp, triton::CatOp>([&](auto joinOp) {
            return verifyNonNegativeExpr(joinOp.getLhs(), assumptions,
                                         solver) &&
                   verifyNonNegativeExpr(joinOp.getRhs(), assumptions, solver);
          })
          // Returns a tensor representing histogram: historgrams only contain
          // buckets of non-negative values.
          .Case<triton::HistogramOp>([&](auto) { return true; })
          .Case<triton::MakeRangeOp>([&](auto makeRangeOp) {
            // See the warning in TritonOps.td: getStart/getEnd return unsigned,
            // so we need to look through get*Attr.
            return makeRangeOp.getStartAttr().getInt() >= 0 &&
                   makeRangeOp.getEndAttr().getInt() >= 0;
          })
          .Case<arith::ConstantIntOp>(
              [&](auto constIntOp) { return constIntOp.value() >= 0; })
          .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
            Value val = constOp.getResult();
            DenseIntElementsAttr constVal;
            if (matchPattern(val, m_Constant(&constVal)) && constVal.isSplat())
              return constVal.getSplatValue<APInt>().isNonNegative();
            return false;
          })
          .Case<triton::GetNumProgramsOp, triton::GetProgramIdOp>([&](auto) {
            // These are defined as signless, but are actually unsigned
            return true;
          })
          .Case<arith::MaxSIOp>([&](auto maxOp) {
            // max(a,b) >= 0 iff a>=0 || b>=0
            return verifyNonNegativeExpr(maxOp.getLhs(), assumptions, solver) ||
                   verifyNonNegativeExpr(maxOp.getRhs(), assumptions, solver);
          })
          .Case<arith::RemSIOp>([&](auto remsiOp) {
            // a % b >= 0 iff a>=0
            return verifyNonNegativeExpr(remsiOp.getLhs(), assumptions, solver);
          })
          .Case<arith::TruncIOp, arith::ExtSIOp>([&](Operation *unaryOp) {
            // a = OP b >= 0 iff b >= 0
            return verifyNonNegativeExpr(unaryOp->getOperand(0), assumptions,
                                         solver);
          })
          // Casting from arbitrary data does *not* guarantee the offset is in
          // range (even if pointer, or the data is non-negative when
          // interpreted as the src's type).
          .Case<triton::PtrToIntOp, triton::BitcastOp>(
              [&](auto) { return false; })
          .Case<arith::CeilDivUIOp, arith::DivUIOp, arith::ExtUIOp,
                arith::FPToUIOp, arith::MaxUIOp, arith::MinUIOp, arith::RemUIOp,
                arith::ShRUIOp>(
              // These OPs also return unsigned values.
              // TODO: We can also sniff whether a Value is unsigned by looking
              //       for whether or not it's used as an argument to one of
              //       these OPs.
              [&](auto uOp) { return true; })
          .Case<arith::AddIOp, arith::MinSIOp, arith::MulIOp, arith::DivSIOp>(
              // Generally speaking, a OP b >= 0  iff  a >= 0 && b >= 0 when
              // OP != sub
              [&](Operation *binOp) {
                return verifyNonNegativeExpr(binOp->getOperand(0), assumptions,
                                             solver) &&
                       verifyNonNegativeExpr(binOp->getOperand(1), assumptions,
                                             solver);
              })
          // TODO: more scf
          .Case<scf::IfOp>([&](auto ifOp) {
            auto results = ifOp.getResults();
            auto it = std::find(results.begin(), results.end(), expr);
            assert(it != results.end() && "expr should be the result of ifOp");
            auto resultIdx = it - results.begin();

            // If we're here then we must have both then/else regions
            // (each with 1 block) and each region must terminate with an
            // `scf.yield` expression.
            auto thenYield = cast<scf::YieldOp>(ifOp.thenYield());
            auto elseYield = cast<scf::YieldOp>(ifOp.elseYield());
            return verifyNonNegativeExpr(thenYield->getOperand(resultIdx),
                                         assumptions, solver) &&
                   verifyNonNegativeExpr(elseYield->getOperand(resultIdx),
                                         assumptions, solver);
          })
          .Case<arith::SubIOp>([&](auto op) {
            // If a user annotates tl.assume(a >= b) then we know a - b >= 0
            return verifyNonSmallerByAssumption(op.getLhs(), assumptions,
                                                op.getRhs());
          })
          .Default([&](Operation *) {
            // Conservatively assume that the expression is negative
            LDBG("  Unhandled op, cannot assume non-negative");
            return false;
          });
  return nonNegative;
}

// Quick analysis on the Triton IR to decide if we can safely use
// buffer operations
bool canUseBufferOps(Value ptr, const DenseSet<Value> &assumptions,
                     std::shared_ptr<DataFlowSolver> solver) {
  // 1. Check if the pointer is uniform: i.e., if it comes from a uniform
  // pointer(splatted) and non-uniform offset addition

  LDBG("Buffer op checks for: " << ptr);
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return false;

  auto maybeSplatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
  if (!maybeSplatOp)
    return false;
  LDBG("Pattern matched");

  // 2. Check if the offset is a 32-bit tensor
  Value offset = addPtrOp.getOffset();
  if (cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth() != 32)
    return false;
  LDBG("32 bit offset");

  return verifyNonNegativeExpr(offset, assumptions, std::move(solver));
}

// Extract stride of the blocked offset of LD/ST ops.
Value getBlockStride(Location loc, Value offset, PatternRewriter &rewriter) {
  // canonicalize pointer pass sets block stride via
  // `offset:add-broadcast-muli-splat`, backtrace that pattern to reach the
  // stride.
  if (auto maybeAdd = offset.getDefiningOp<arith::AddIOp>())
    for (auto addOpr : maybeAdd.getOperands())
      if (auto maybeBC = addOpr.getDefiningOp<tt::BroadcastOp>()) {
        auto bcSrc = maybeBC.getSrc();
        if (auto maybeMul = bcSrc.getDefiningOp<arith::MulIOp>())
          for (auto mulOpr : maybeMul.getOperands())
            if (auto maybeSplat = mulOpr.getDefiningOp<tt::SplatOp>())
              return maybeSplat.getSrc();
      }
  return nullptr;
}

} // namespace

struct ConvertTritonAtomicRMWOpToBufferAtomicRMW
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicRMWOpToBufferAtomicRMW(
      mlir::MLIRContext *context, DenseSet<Value> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto atomicRmwOp = op.getAtomicRmwOp();
    auto sem = op.getSem();
    auto scope = op.getScope();

    // In addition to the `canUserBufferOps` check, we should ensure that
    // 1. Perform the canUserBufferOps check
    if (!canUseBufferOps(ptr, assumptions, solver)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    // 2. Check the scope. We support GPU and CTA for now (SYSTEM scope is not
    // supported yet)
    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "RMW with unsupported scope");
    }
    LDBG("RMW supported scope");

    // 3. Check the memory ordering.
    //    TODO: support monotonic
    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "RMW with unsupported memory ordering");
    }

    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value tensorOffset = addPtrOp.getOffset();
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    // 4. Buffer atomic RMW does not support FP8 ops
    //    easier to just check what we support
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isF16() || checkType.isBF16() ||
                           checkType.isF32() || checkType.isF64() ||
                           checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "RMW with unsupported type");
    }
    LDBG("RMW supported type");

    auto vecSize = getVectorSize(ptr, axisAnalysisPass);
    // f16/bf16 dtypes could only be efficiently calculated using instructions
    // that pack 2 elements (e.g. @llvm.amdgcn.raw.buffer.atomic.fadd.v2f16)
    if (vecSize % 2 != 0 && (checkType.isF16() || checkType.isBF16())) {
      return rewriter.notifyMatchFailure(
          op, "RMW float 16 dtypes must be aligned by 2");
    }
    LDBG("RMW passed alignment check");

    // 5. Check if the RMWOp is supported
    switch (atomicRmwOp) {
    case RMWOp::AND:
    case RMWOp::OR:
    case RMWOp::XOR:
    case RMWOp::ADD:
    case RMWOp::FADD:
    case RMWOp::MAX:
    case RMWOp::MIN:
    case RMWOp::UMAX:
    case RMWOp::UMIN:
    case RMWOp::XCHG:
      break;
    default:
      auto rmwOpStr = stringifyRMWOp(atomicRmwOp).str();
      return rewriter.notifyMatchFailure(op, "RMW with unsupported op: " +
                                                 rmwOpStr);
    }
    LDBG("RMW supported Op");

    // 6. Buffer atomics support 32 and 64-bit operations, so inputs must be at
    //    least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      // We can't just compute the opBitWidth using the numElements *
      // elemBitWidth here. In cases such as tensor<2xf16...>, if the elements
      // are contiguous we can emit the buffer op. Otherwise, the buffer ops
      // lowering will try to emit individual (unsupported) f16/bf16 ops.
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth = vecSize * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(op, "RMW requires opBitWidth >= 32");
    }

    Value maybeMask{};
    if (op.getMask() && !isZeroConst(op.getMask()))
      maybeMask = op.getMask();
    Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicRMWOp>(
        op, op.getVal().getType(), atomicRmwOp, basePtr, tensorOffset,
        op.getVal(), blockStride, sem, scope, maybeMask);

    return success();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
};

// Workaround to allow static_assert(false) on older compilers as it was
// ill-formed before defect report CWG2518
// (https://cplusplus.github.io/CWG/issues/2518.html)
template <typename T> struct always_false : std::false_type {};

template <typename SourceOp>
struct ConvertTritonLoadToBufferLoad : public mlir::OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(mlir::MLIRContext *context,
                                DenseSet<Value> &assumptions,
                                std::shared_ptr<DataFlowSolver> solver)
      : mlir::OpRewritePattern<SourceOp>(context), assumptions(assumptions),
        solver(std::move(solver)) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getOperand(0);

    if (canUseBufferOps(ptr, assumptions, solver)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther{};
      if (op.getOther() && !isZeroConst(op.getOther()))
        maybeOther = op.getOther();
      Value maybeMask{};
      if (op.getMask() && !isZeroConst(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);

      auto bufferLoadOp = [&]() {
        if constexpr (std::is_same_v<SourceOp, triton::LoadOp>) {
          return rewriter.create<triton::amdgpu::BufferLoadOp>(
              op->getLoc(), op.getType(), basePtr, tensorOffset, blockStride,
              op.getCache(), maybeMask, maybeOther);
        } else if constexpr (std::is_same_v<
                                 SourceOp,
                                 triton::gpu::AsyncCopyGlobalToLocalOp>) {
          return rewriter.create<triton::amdgpu::BufferLoadToLocalOp>(
              op->getLoc(), op.getType(), op.getResult(), basePtr, tensorOffset,
              maybeMask, maybeOther, blockStride, op.getCache());
        } else {
          static_assert(always_false<SourceOp>::value,
                        "Unsupported type in ConvertTritonLoadToBufferLoad");
        }
      }();

      assert(bufferLoadOp);

      // Propagate `OpIdxAttr` if the currently processed `tt.LoadOp` was
      // labeled it. The attribute needs to be preserved for custom instruction
      // scheduling.
      if (auto opIdxAttr =
              op->template getAttrOfType<triton::amdgpu::OpIdxAttr>(
                  triton::amdgpu::OpIdxAttr::getMnemonic())) {
        bufferLoadOp->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(),
                              opIdxAttr);
      }
      rewriter.replaceOp(op, bufferLoadOp);
      return success();
    }

    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert LoadOp");
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
  std::shared_ptr<DataFlowSolver> solver;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(mlir::MLIRContext *context,
                                  DenseSet<Value> &assumptions,
                                  std::shared_ptr<DataFlowSolver> solver)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions), solver(std::move(solver)) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (canUseBufferOps(ptr, assumptions, solver)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask{};
      if (op.getMask() && !isZeroConst(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, blockStride, op.getCache(),
          maybeMask);
      return success();
    }
    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert StoreOp");
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
  std::shared_ptr<DataFlowSolver> solver;
};

class TritonAMDGPUConvertToBufferOpsPass
    : public TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {

public:
  TritonAMDGPUConvertToBufferOpsPass() = default;
  TritonAMDGPUConvertToBufferOpsPass(StringRef archGen) {
    this->archGenerationName = archGen.data();
  };
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    // Collect assumptions in the function
    DenseSet<Value> assumptions;
    mod.walk([&](LLVM::AssumeOp op) {
      auto oper = op->getOperand(0);
      if (oper.getDefiningOp<arith::CmpIOp>())
        assumptions.insert(oper);
    });
    LLVM_DEBUG({
      DBGS() << "Number of assumptions found: " << assumptions.size() << "\n";
      for (Value assume : assumptions) {
        DBGS() << "Assumption:" << assume << "\n";
      }
    });

    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<AMD::TritonIntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    patterns.add<ConvertTritonLoadToBufferLoad<tt::LoadOp>,
                 ConvertTritonLoadToBufferLoad<ttg::AsyncCopyGlobalToLocalOp>,
                 ConvertTritonStoreToBufferStore>(context, assumptions, solver);

    // Gate buffer atomics behind CDNA3 for now
    // GFX942-specific assumptions regarding cache coherence are made when
    // lowering to LLVM
    if (ISAFamily::CDNA3 == triton::AMD::deduceISAFamily(archGenerationName))
      patterns.add<ConvertTritonAtomicRMWOpToBufferAtomicRMW>(
          context, assumptions, axisInfoAnalysis, solver);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUConvertToBufferOpsPass(std::string archGen) {
  return std::make_unique<TritonAMDGPUConvertToBufferOpsPass>(archGen);
}
