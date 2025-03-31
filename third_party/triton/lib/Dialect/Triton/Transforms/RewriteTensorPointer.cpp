#include <memory>
#include <stack>

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

/// An additional struct to record the meta information of operations
/// with tensor pointers
struct RewritedInfo {
private:
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  ArrayRef<int64_t> tensorShape;

  // A cache to avoid generating the same offset with range
  DenseMap<unsigned, Value> cachedOffsetWithRange;

public:
  RewritedInfo() = default;

  RewritedInfo(const RewritedInfo &other) = default;

  RewritedInfo(Value base, const SmallVector<Value> &shape,
               const SmallVector<Value> &strides,
               const SmallVector<Value> &offsets,
               const ArrayRef<int64_t> &tensorShape)
      : base(base), shape(shape), strides(strides), offsets(offsets),
        tensorShape(tensorShape) {
    assert(shape.size() == strides.size() && shape.size() == offsets.size() &&
           shape.size() == tensorShape.size());
  }

  unsigned int length() const { return shape.size(); }

  Value getOffset(unsigned i) { return offsets[i]; }

  SmallVector<Value> getOffsets() { return offsets; }

  void setOffset(unsigned i, Value newOffset) {
    offsets[i] = newOffset;
    cachedOffsetWithRange.clear();
  }

  void setOffsets(const SmallVector<Value> &newOffsets) {
    offsets = newOffsets;
    cachedOffsetWithRange.clear();
  }

  Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                   unsigned i) {
    if (cachedOffsetWithRange.count(i))
      return cachedOffsetWithRange[i];

    // Add range
    auto indexI32RowType =
        RankedTensorType::get({tensorShape[i]}, builder.getI32Type());
    auto indexRowType =
        RankedTensorType::get({tensorShape[i]}, builder.getI64Type());
    Value splatOffset =
        builder.create<triton::SplatOp>(loc, indexRowType, offsets[i]);
    Value range = builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0,
                                                      tensorShape[i]);
    Value i64Range = builder.create<arith::ExtSIOp>(loc, indexRowType, range);

    // Expand dimensions
    Value expandedResult =
        builder.create<arith::AddIOp>(loc, splatOffset, i64Range);
    for (int j = 0; j < tensorShape.size(); ++j) {
      if (j == i)
        continue;
      expandedResult =
          builder.create<triton::ExpandDimsOp>(loc, expandedResult, j);
    }

    return cachedOffsetWithRange[i] = expandedResult;
  }

  Value generatePtr(OpBuilder &builder, const Location &loc) {
    assert(tensorShape.size() == offsets.size() &&
           tensorShape.size() == strides.size());
    auto indexTensorType =
        RankedTensorType::get(tensorShape, builder.getI64Type());
    auto ptrType = cast<triton::PointerType>(base.getType());
    auto ptrTensorType = RankedTensorType::get(tensorShape, ptrType);

    // Generate offsets per dimension
    Value ptr = builder.create<triton::SplatOp>(loc, ptrTensorType, base);
    for (unsigned i = 0; i < tensorShape.size(); ++i) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

      // We must splat strides into the expanded shape not a row for retaining
      // the divisibility information given by strides
      Value splatStride = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), strides[i]);
      Value offsetWithStride =
          builder.create<arith::MulIOp>(loc, offsetWithRange, splatStride);
      Value broadcasted = builder.create<triton::BroadcastOp>(
          loc, indexTensorType, offsetWithStride);

      // Add to the pointer
      ptr = builder.create<triton::AddPtrOp>(loc, ptrTensorType, ptr,
                                             broadcasted);
    }

    return ptr;
  }

  Value generateMask(OpBuilder &builder, const Location &loc,
                     const std::optional<ArrayRef<int32_t>> &boundaryCheck) {
    if (!boundaryCheck.has_value())
      return {};

    // Generate mask per dimension
    auto maskTensorType =
        RankedTensorType::get(tensorShape, builder.getI1Type());
    Value mask;
    for (auto i : boundaryCheck.value()) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

      // Compare with lower bound
      Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
          loc, 0, builder.getI64Type());
      Value splatLowerBound = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), lowerBound);
      Value cmpLower = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, offsetWithRange, splatLowerBound);

      // Compare with upper bound
      Value splatUpperBound = builder.create<triton::SplatOp>(
          loc, offsetWithRange.getType(), shape[i]);
      Value cmpUpper = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, offsetWithRange, splatUpperBound);

      // And and broadcast
      Value andResult = builder.create<arith::AndIOp>(loc, cmpLower, cmpUpper);
      Value broadcasted =
          builder.create<triton::BroadcastOp>(loc, maskTensorType, andResult);

      // And up all results
      if (!mask) {
        mask = broadcasted;
      } else {
        mask = builder.create<arith::AndIOp>(loc, mask, broadcasted);
      }
    }

    return mask;
  }

  Value generateOther(OpBuilder &builder, const Location &loc,
                      const std::optional<triton::PaddingOption> &padding) {
    if (!padding.has_value())
      return Value();

    // Create element attribute
    auto elementType =
        cast<triton::PointerType>(base.getType()).getPointeeType();
    auto otherTensorType = RankedTensorType::get(tensorShape, elementType);

    // Set zero padding value
    TypedAttr attr = builder.getZeroAttr(elementType);

    // Float NaN padding case
    if (padding.value() == triton::PaddingOption::PAD_NAN) {
      assert(!elementType.isIntOrIndex());
      auto apNaN = llvm::APFloat::getNaN(
          cast<FloatAttr>(attr).getValue().getSemantics());
      attr = builder.getFloatAttr(elementType, apNaN);
    }

    // Create tensor
    Value constant = builder.create<arith::ConstantOp>(loc, attr);
    return builder.create<triton::SplatOp>(loc, otherTensorType, constant);
  }
};

} // namespace

// TODO: this pass relies on assumptions of how block pointers are created and
// on pattern matches that walks the SSA links to find the base/strides. This is
// very fragile and to solve we should expose convert Ptr of tensor to a
// structure containins all values and not only offsets.
class RewriteTensorPointerPass
    : public TritonRewriteTensorPointerBase<RewriteTensorPointerPass> {
private:
  DenseMap<Value, RewritedInfo> rewritedInfo;

public:
  static bool needRewrite(Operation *op) {
    return std::any_of(op->getOperands().begin(), op->getOperands().end(),
                       [](Value operand) {
                         return triton::isTensorPointerType(operand.getType());
                       });
  }

  static void generateNewOperands(SmallVector<Value> &oldOperands,
                                  unsigned index, ArrayRef<Value> newValues) {
    size_t size = oldOperands.size();
    assert(index < size);
    SmallVector<Value> operands = oldOperands;
    oldOperands.reserve(size - 1 + newValues.size());
    oldOperands.clear();
    if (index != 0) {
      oldOperands.append(operands.begin(), operands.begin() + index);
    }
    oldOperands.append(newValues.begin(), newValues.end());
    if (index != size - 1) {
      oldOperands.append(operands.begin() + index + 1, operands.end());
    }
  }

  Operation *rewriteMakeTensorPtrOp(OpBuilder &builder,
                                    triton::MakeTensorPtrOp op,
                                    std::stack<Operation *> &eraser) {
    // Save info for later use
    auto ptrType = cast<triton::PointerType>(op.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());

    // Cast I32 offsets into I64
    SmallVector<Value> i64Offsets;
    for (auto offset : op.getOffsets()) {
      auto i64Offset = builder.create<arith::ExtSIOp>(
          op.getLoc(), builder.getI64Type(), offset);
      i64Offsets.push_back(i64Offset);
    }

    // Save information
    rewritedInfo[op.getResult()] =
        RewritedInfo(op.getBase(), op.getShape(), op.getStrides(), i64Offsets,
                     tensorType.getShape());

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteAdvanceOp(OpBuilder &builder, triton::AdvanceOp op,
                              std::stack<Operation *> &eraser) {
    // Get info from previous results
    assert(rewritedInfo.count(op.getPtr()));
    auto info = rewritedInfo[op.getPtr()];

    // Calculate new offsets
    assert(info.length() == op.getOffsets().size());
    SmallVector<Value> newOffsets;
    for (int i = 0; i < info.length(); ++i) {
      Value i64Offset = builder.create<arith::ExtSIOp>(
          op.getLoc(), builder.getI64Type(), op.getOffsets()[i]);
      Value newOffset = builder.create<arith::AddIOp>(
          op.getLoc(), info.getOffset(i), i64Offset);
      newOffsets.push_back(newOffset);
    }

    // Save info for later use
    info.setOffsets(newOffsets);
    rewritedInfo[op.getResult()] = info;

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteLoadStoreOp(OpBuilder &builder, Operation *op,
                                std::stack<Operation *> &eraser) {
    assert(isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op));

    // We only have to rewrite load/stores with tensor pointers
    auto ptr = op->getOperand(0);
    if (!triton::isTensorPointerType(ptr.getType()))
      return nullptr;

    // Get info from previous results
    assert(rewritedInfo.count(ptr));
    auto info = rewritedInfo[ptr];

    // Load/store with tensor pointers implicitly will check the bound while
    // accessing memory, so we should set `mask` and `other` (according to the
    // padding). Also note that load with tensor pointers do not have `mask` and
    // `other` while building IR from Python AST
    std::optional<ArrayRef<int>> boundaryCheck;
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      assert(!loadOp.getMask() && !loadOp.getOther());
      boundaryCheck = loadOp.getBoundaryCheck();
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      assert(!storeOp.getMask());
      boundaryCheck = storeOp.getBoundaryCheck();
    }

    // Generate new `ptr`, `mask` and `other`
    auto newPtr = info.generatePtr(builder, op->getLoc());
    auto newMask = info.generateMask(builder, op->getLoc(), boundaryCheck);
    Value newOther;
    if (auto loadOp = dyn_cast<triton::LoadOp>(op))
      newOther = info.generateOther(builder, op->getLoc(), loadOp.getPadding());

    // Create a new operation
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      auto newResult = builder.create<triton::LoadOp>(
          loadOp.getLoc(), newPtr, newMask, newOther, loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile());
      op->getResult(0).replaceAllUsesWith(newResult);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      builder.create<triton::StoreOp>(storeOp.getLoc(), newPtr,
                                      storeOp.getValue(), newMask,
                                      storeOp.getCache(), storeOp.getEvict());
    }

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteIfOp(OpBuilder &builder, scf::IfOp op,
                         std::stack<Operation *> &eraser) {
    auto thenYieldOp = op.thenYield();
    assert(op.getNumResults() == thenYieldOp.getNumOperands());
    SmallVector<Value> results = thenYieldOp.getOperands();

    // get new result types
    SmallVector<Type> newRetTypes;
    bool needRewrite = false;
    for (unsigned i = 0; i < results.size(); ++i) {
      if (!triton::isTensorPointerType(results[i].getType())) {
        newRetTypes.push_back(results[i].getType());
        continue;
      }
      needRewrite = true;
      auto makeTensorPtrOp = getMakeTensorPtrOp(results[i]);
      assert(rewritedInfo.count(makeTensorPtrOp.getResult()));
      const auto &info = rewritedInfo[makeTensorPtrOp.getResult()];
      for (unsigned j = 0; j < info.length(); ++j) {
        newRetTypes.push_back(builder.getI64Type());
      }
    }
    if (!needRewrite)
      return op;
    // create and clone new IfOp
    bool hasElse = !op.getElseRegion().empty();
    scf::IfOp newOp = builder.create<scf::IfOp>(op.getLoc(), newRetTypes,
                                                op.getCondition(), hasElse);
    IRMapping mapping;
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      mapping.map(op->getOperand(i), newOp->getOperand(i));
    }
    auto rematerialize = [&](Block *block) {
      for (Operation &opInIf : block->getOperations()) {
        builder.clone(opInIf, mapping);
      }
    };
    builder.setInsertionPointToStart(newOp.thenBlock());
    rematerialize(op.thenBlock());
    if (hasElse) {
      builder.setInsertionPointToStart(newOp.elseBlock());
      rematerialize(op.elseBlock());
    }

    // update rewritedInfo
    auto opResults = op.getResults();
    unsigned oldResIdx = 0, newResIdx = 0;
    while (oldResIdx < results.size()) {
      if (!triton::isTensorPointerType(results[oldResIdx].getType())) {
        opResults[oldResIdx].replaceAllUsesWith(newOp.getResult(newResIdx));
        oldResIdx++;
        newResIdx++;
      } else {
        auto makeTensorPtrOp = getMakeTensorPtrOp(results[oldResIdx]);
        assert(rewritedInfo.count(makeTensorPtrOp.getResult()));
        auto info = rewritedInfo[makeTensorPtrOp.getResult()];
        for (unsigned j = 0; j < info.length(); ++j) {
          info.setOffset(j, newOp->getResult(newResIdx++));
        }
        rewritedInfo[op.getResult(oldResIdx)] = info;
        oldResIdx++;
      }
    }

    eraser.push(op);
    return newOp;
  }

  Operation *rewriteForOp(OpBuilder &builder, scf::ForOp op,
                          std::stack<Operation *> &eraser) {
    // Generate new iteration operands and set rewritten information
    SmallVector<Value> oldIterOperands = llvm::to_vector(op.getInitArgs());
    SmallVector<Value> newIterOperands = llvm::to_vector(op.getInitArgs());
    for (unsigned i = 0, oldI = 0, size = op.getInitArgs().size(); i < size;
         ++i, ++oldI) {
      if (!triton::isTensorPointerType(newIterOperands[i].getType()))
        continue;

      // Expand the tensor pointer into offsets
      assert(rewritedInfo.count(newIterOperands[i]));
      auto info = rewritedInfo[newIterOperands[i]];
      generateNewOperands(newIterOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }

    // Rebuild the loop type
    auto newForOp = builder.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                               op.getUpperBound(), op.getStep(),
                                               newIterOperands);
    newForOp->setAttrs(op->getAttrs());

    // Create value mapping. Note that for tensor pointers, we use identity
    // mapping. It may refer to a value in the old loop, but we will rewrite it
    // later
    IRMapping mapping;
    for (unsigned i = 0, oldI = 0, sz = op.getInitArgs().size(); oldI < sz;
         ++i, ++oldI) {
      auto oldRegionIterArg = op.getRegionIterArg(oldI);
      if (triton::isTensorPointerType(oldRegionIterArg.getType())) {
        // Pass rewritten info inside
        assert(rewritedInfo.count(oldIterOperands[oldI]));
        auto info = rewritedInfo[oldIterOperands[oldI]];
        mapping.map(oldRegionIterArg, oldRegionIterArg);
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getRegionIterArg(i + j));
        rewritedInfo[oldRegionIterArg] = info;
        i += info.length() - 1;
      } else {
        mapping.map(oldRegionIterArg, newForOp.getRegionIterArg(i));
      }
    }
    mapping.map(op.getInductionVar(), newForOp.getInductionVar());

    // Clone body
    builder.setInsertionPointToStart(newForOp.getBody());
    for (auto &opInFor : *op.getBody()) {
      builder.clone(opInFor, mapping);
    }

    // Replace later usages
    assert(op.getNumResults() == op.getInitArgs().size());
    for (unsigned i = 0, oldI = 0; oldI < op.getNumResults(); ++i, ++oldI) {
      auto oldResult = op.getResult(oldI);
      if (triton::isTensorPointerType(oldResult.getType())) {
        // Pack new offsets into rewritten info
        assert(rewritedInfo.count(oldIterOperands[oldI]));
        auto info = rewritedInfo[oldIterOperands[oldI]];
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getResult(i + j));
        i += info.length() - 1;
        rewritedInfo[oldResult] = info;
      } else {
        oldResult.replaceAllUsesWith(newForOp.getResult(i));
      }
    }

    // Erase later
    eraser.push(op);
    return newForOp;
  }

  Operation *rewriteYieldOp(OpBuilder &builder, scf::YieldOp op,
                            std::stack<Operation *> &eraser) {
    // Replace tensor pointers with offsets
    SmallVector<Value> newOperands = op->getOperands();
    for (unsigned i = 0, size = op.getNumOperands(); i < size; ++i) {
      if (!triton::isTensorPointerType(newOperands[i].getType()))
        continue;

      assert(rewritedInfo.count(newOperands[i]));
      auto info = rewritedInfo[newOperands[i]];
      generateNewOperands(newOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }
    op->setOperands(newOperands);

    // No need to erase
    return nullptr;
  }

  Operation *rewriteOp(Operation *op, std::stack<Operation *> &eraser) {
    OpBuilder builder(op);

    // Rewrite `make_tensor_ptr` and `advance` and make a tensor of pointers
    // Rewriting functions return the next operation to visit, if there is no
    // next one, simply return `nullptr`
    if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
      return rewriteMakeTensorPtrOp(builder, makeTensorPtrOp, eraser);
    } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
      return rewriteAdvanceOp(builder, advanceOp, eraser);
    } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
      return rewriteLoadStoreOp(builder, op, eraser);
    } else if (isa<scf::SCFDialect, cf::ControlFlowDialect>(op->getDialect())) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        return rewriteIfOp(builder, ifOp, eraser);
      }
      if (!needRewrite(op))
        return op;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return rewriteForOp(builder, forOp, eraser);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        return rewriteYieldOp(builder, yieldOp, eraser);
      } else {
        llvm_unreachable("Currently we only support tensor pointer usages "
                         "inside a `scf::ForOp` or `scf::IfOp`, others such as "
                         "`scf::WhileOp`, `cf::BranchOp` or `cf::CondBranchOp` "
                         "are not supported yet");
      }
    }

    // Otherwise return the original one
    return op;
  }

  void visitOperation(Operation *op, std::stack<Operation *> &eraser) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &nestedOp : llvm::make_early_inc_range(block)) {
          if (auto newOp = rewriteOp(&nestedOp, eraser)) {
            visitOperation(newOp, eraser);
          }
        }
      }
    }
  }

  void runOnOperation() override {
    // NOTES(Chenggang): we don't use `ConversionPatternRewriter`, because
    // MLIR does not support one-multiple value mapping. For example, if we use
    // `ConversionPatternRewriter`, we can not make a type converter, which
    // converts `ptr<tensor>` into multiple types `ptr<>, int64, int64, ...`
    // (containing the base/offsets/strides...). What we can do is to convert
    // `ptr<tensor>` into a single type `Tuple<ptr<>, int64, int64, ...>`. But
    // in this way, we also have to define `PackTuple` and `UnpackTuple`
    // operations and make a canonicalization pass to optimize, which is much
    // So here we recursively build the IR, to be specific, we have to rewrite
    // `tt.make_tensor_ptr`, `tt.advance`, `tt.load`, `tt.store`,
    // `scf.for` (tensor pointer usages may be in a loop fashion)
    std::stack<Operation *> eraser;
    visitOperation(getOperation(), eraser);

    // The operation could not be erased during visit, because they may have
    // later usages, so we erase after visit
    rewritedInfo.clear();
    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }
  }
};

std::unique_ptr<Pass> triton::createRewriteTensorPointerPass() {
  return std::make_unique<RewriteTensorPointerPass>();
}
