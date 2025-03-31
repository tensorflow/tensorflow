#include <memory>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>
#include <memory>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUREMOVELAYOUTCONVERSIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-remove-layout-conversions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// The current algorithm works by analyzing the IR and doing a one-shot rewrite
// based on the analysis. The algorithm is as follows.
//
// 1. Find all the anchor ops. These are ops that have a layout we want to
//    preserve.
//
// 2. For each anchor, propagate its layout to all its descendants.
//    An op can have multiple ancestors that are anchors, so at this stage an op
//    may have multiple layouts associated with it.
//
// 3. Resolve conflicts by deciding which of the multiple layouts the op should
//    keep, inserting convert-layout ops to resolve conflicts.  After this
//    stage, each value has only one layout associated with it.
//
// 4. Rewrite the IR by walking the function in dominance order. Since we
//    assume the IR is structured we just need to process the regions in the
//    correct order. For each op, rewrite it using the layout decided by the
//    analysis phase.
class LayoutPropagation {
public:
  // Structure to keep track of the layout associated to a value.
  struct LayoutInfo {
    LayoutInfo(Attribute encoding) { encodings.insert(encoding); }
    LayoutInfo() {}
    llvm::SmallSetVector<Attribute, 8> encodings;
  };
  LayoutPropagation(FuncOp F) : funcOp(F) {}
  // Find the anchor ops and set their layout in the data structure.
  void initAnchorLayout();
  // Recursively Propagate the layout to all the users of the anchor ops until
  // we reach a fix point.
  void propagateLayout();
  // Add layouts given in `Info` to the uses of `value`.
  SmallVector<Value> propagateToUsers(Value value, LayoutInfo &info);
  // Set the encoding to all the values and fill out the values with new layout
  // in `changed`.
  void setEncoding(ValueRange values, LayoutInfo &info,
                   SmallVector<Value> &changed, Operation *op);
  // Resolve cases where a value has multiple layouts associated to it.
  void resolveConflicts();
  // Rewrite the IR for the full module.
  void rewrite();
  // Rewrite the IR for a region.
  void rewriteRegion(Region &R);
  // Rewrite an op based on the layout picked by the analysis.
  Operation *rewriteOp(Operation *op);
  // Rewrite a for op based on the layout picked by the analysis.
  Operation *rewriteForOp(scf::ForOp forOp);
  Operation *rewriteWhileOp(scf::WhileOp whileOp);
  Operation *rewriteIfOp(scf::IfOp ifOp);
  void rewriteYieldOp(scf::YieldOp yieldOp);
  void rewriteConditionOp(scf::ConditionOp conditionOp);
  void rewriteReduceToScalar(Operation *reduceOp);
  void rewriteAssertOp(AssertOp assertOp);
  Operation *cloneElementwise(OpBuilder &rewriter, Operation *op,
                              Attribute encoding);
  // Map the original value to the rewritten one.
  void map(Value old, Value newV);
  // Return the mapped value in the given encoding. This will insert a convert
  // if the encoding is different than the encoding decided at resolve time.
  Value getValueAs(Value value, Attribute encoding);
  // Dump the current stage of layout information.
  void dump();

private:
  // map from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layouts;
  // map of the values rewrite based on their encoding.
  DenseMap<std::pair<Value, Attribute>, Value> rewriteMapping;
  SetVector<Operation *> opToDelete;
  FuncOp funcOp;
};

class LayoutRematerialization {
public:
  LayoutRematerialization(FuncOp F) : funcOp(F) {}

  // Map the original value to the remat'ed one.
  void addRematValue(Value old, Attribute encoding, Value newV);
  // Get the remat'ed value in the given encoding, if one already exists and
  // is different then the layout conversion root.
  Value getRematValue(Value value, Attribute encoding) const {
    return rematMapping.lookup({value, encoding});
  }

  void cleanup();
  void backwardRematerialization();
  void backwardRematerialization(ConvertLayoutOp convertOp);
  // TODO: Merge the three hoistConvert*(); functions as they are duplicate code
  void hoistConvertDotOperand();
  void hoistConvertDotOperand(ConvertLayoutOp convertOp);
  void hoistConvertOnTopOfExtOrBroadcast();
  void hoistConvertOnTopOfExtOrBroadcast(ConvertLayoutOp convertOp);
  void hoistConvertIntoConditionals();
  void hoistConvertIntoConditionals(ConvertLayoutOp convertOp);
  void rewriteSlice(SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
                    ConvertLayoutOp convertOp, IRMapping &mapping);
  void rewriteSlice(SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
                    ConvertLayoutOp convertOp);

  LogicalResult
  getConvertBackwardSlice(OpOperand &root, Attribute rootEncoding,
                          SetVector<Value> &slice,
                          DenseMap<Value, Attribute> &layout,
                          std::function<bool(Operation *)> stopPropagation);

  LogicalResult getRematerializableSlice(
      OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
      DenseMap<Value, Attribute> &layout,
      std::function<bool(Operation *)> stopPropagation = nullptr);

private:
  void updateRematMapping(SmallVector<std::tuple<Value, Value>> &values);
  // Existing tuples of (value, layout) that needs to be updated when recreating
  // scf ops. This prevents keeping track of Values that have been delete when
  // rewriting slices.
  DenseMap<Value, Attribute> mappedValues;
  // map of the values remat based on encoding.
  DenseMap<std::pair<Value, Attribute>, Value> rematMapping;
  // DenseMap<std::pair<Operation*, Attribute>, Operation*>
  SetVector<Operation *> opToDelete;
  FuncOp funcOp;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
};

void LayoutRematerialization::addRematValue(Value old, Attribute encoding,
                                            Value newV) {
  LDBG("addRematValue " << old << " encoding " << encoding << " " << newV);
  rematMapping[{old, encoding}] = newV;
  mappedValues[old] = encoding;
}

// Remove unneeded values now that we are done with the rematMapping.
void LayoutRematerialization::cleanup() {
  for (Operation *op : llvm::reverse(opToDelete))
    op->erase();
}

// Return true if the op is an op with a layout we don't want to change. We will
// propagate the layout starting from anchor ops.
bool isLayoutAnchor(Operation *op) {
  if (isa<LoadOp, StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  if (isa<DotOp, DotScaledOp, nvidia_gpu::WarpGroupDotOp, AtomicRMWOp,
          AtomicCASOp, triton::nvidia_gpu::TMEMLoadOp>(op))
    return true;
  if (auto gatherOp = dyn_cast<GatherOp>(op))
    return gatherOp.getEfficientLayout();

  // Heuristic: Mark permuting reshape as a layout anchor.  Its dst can be
  // anything, so it stops forward-propagation of layouts.  We rely on the
  // backwards pass to fix it up if necessary.  (If we didn't do this, then
  // anything following the reshape won't be covered by the forward pass at
  // all.)
  if (auto reshape = dyn_cast<ReshapeOp>(op))
    return reshape.getAllowReorder();

  return false;
}

void LayoutPropagation::initAnchorLayout() {
  auto addAnchor = [&](Value v) {
    if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
      layouts.insert({v, LayoutInfo(tensorType.getEncoding())});
    }
  };

  // Consider function args as anchors.  This makes it easier to write tests --
  // you can pass a tensor with an encoding as an arg, instead of explicitly
  // calling tt.load.
  for (auto arg : funcOp.getArguments()) {
    addAnchor(arg);
  }

  funcOp.walk([&](Operation *op) {
    if (isLayoutAnchor(op)) {
      for (auto result : op->getResults()) {
        addAnchor(result);
      }
    }
  });
}

void LayoutPropagation::setEncoding(ValueRange values, LayoutInfo &info,
                                    SmallVector<Value> &changed,
                                    Operation *op) {
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool hasChanged = false;
    for (auto encoding : info.encodings) {
      Attribute dstEncoding;
      if (isa<ConvertLayoutOp>(op)) {
        // Try to remove the convert by making the dst encoding match the source
        // encoding.
        dstEncoding = encoding;
      } else {
        dstEncoding = inferDstEncoding(op, encoding);
      }
      if (dstEncoding)
        hasChanged |= layouts[value].encodings.insert(dstEncoding);
    }
    if (hasChanged)
      changed.push_back(value);
  }
}

SmallVector<Value> LayoutPropagation::propagateToUsers(Value value,
                                                       LayoutInfo &info) {
  SmallVector<Value> changed;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      setEncoding({arg, result}, info, changed, user);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      Value arg = whileOp.getBeforeArguments()[use.getOperandNumber()];
      setEncoding({arg}, info, changed, user);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto parent = yieldOp->getParentOp();
      SmallVector<Value> valuesToPropagate;
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        valuesToPropagate.push_back(parent->getResult(use.getOperandNumber()));
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        valuesToPropagate.push_back(
            forOp.getRegionIterArg(use.getOperandNumber()));
      if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
        valuesToPropagate.push_back(
            whileOp.getBeforeArguments()[use.getOperandNumber()]);
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        setEncoding(valuesToPropagate, info, changed, user);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      // Skip arg 0 as it is the condition.
      unsigned argIndex = use.getOperandNumber() - 1;
      Value afterArg = whileOp.getAfterArguments()[argIndex];
      Value result = whileOp->getResult(argIndex);
      setEncoding({afterArg, result}, info, changed, user);
      continue;
    }
    if (auto dotWaitOp = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(user)) {
      unsigned opIndex = use.getOperandNumber();
      Value result = dotWaitOp->getResult(opIndex);
      setEncoding(result, info, changed, user);
      continue;
    }
    if (auto gatherOp = dyn_cast<GatherOp>(user)) {
      // Propagate the layout through the indices only, and if the layout does
      // not have an efficient layout set.
      if (!gatherOp.getEfficientLayout() &&
          &use == &gatherOp.getIndicesMutable()) {
        setEncoding(gatherOp.getResult(), info, changed, user);
        continue;
      }
    }
    if (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
        user->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
            ConvertLayoutOp>(user)) {
      setEncoding(user->getResults(), info, changed, user);
      continue;
    }
  }
  return changed;
}

void LayoutPropagation::propagateLayout() {
  SmallVector<Value> queue;
  for (auto it : layouts) {
    queue.push_back(it.first);
  }
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo info = layouts[currentValue];
    queue.pop_back();
    SmallVector<Value> changed = propagateToUsers(currentValue, info);

    LLVM_DEBUG({
      DBGS() << "propagateLayout considering " << currentValue << ", which has "
             << info.encodings.size() << " candidate encoding(s):\n";
      for (Attribute encoding : info.encodings)
        DBGS() << "  " << encoding << "\n";
    });

    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

void LayoutPropagation::resolveConflicts() {
  for (auto &it : layouts) {
    Operation *op = it.first.getDefiningOp();
    LayoutInfo &info = it.second;
    if (info.encodings.size() <= 1)
      continue;
    // Hacky resolve, prefer block encoding.
    // TODO: add a proper heuristic.
    Attribute encoding = *info.encodings.begin();
    bool isLoadOrStore =
        op && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(op);
    for (Attribute e : info.encodings) {
      if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
          (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
        encoding = e;
        break;
      }
    }
    info.encodings.clear();
    info.encodings.insert(encoding);
  }
}

void LayoutPropagation::dump() {
  for (auto it : layouts) {
    llvm::errs() << "Value: ";
    OpPrintingFlags flags;
    flags.skipRegions();
    it.first.print(llvm::errs(), flags);
    llvm::errs() << " \n encoding:\n";
    for (auto encoding : it.second.encodings) {
      encoding.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "--\n";
  }
}

void LayoutPropagation::rewrite() { rewriteRegion(funcOp->getRegion(0)); }

bool reduceToScalar(Operation *op) {
  // For reductions returning a scalar we can change the src encoding without
  // affecting the output.
  return isa<ReduceOp>(op) && !isa<RankedTensorType>(op->getResultTypes()[0]);
}

void LayoutPropagation::rewriteRegion(Region &region) {
  std::deque<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.front();
    queue.pop_front();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = false;
      SmallVector<Value> results = op.getResults();
      for (Value result : results) {
        auto it = layouts.find(result);
        // If we haven't mapped this value skip.
        if (it == layouts.end())
          continue;
        LayoutInfo &info = it->second;
        assert(info.encodings.size() == 1 &&
               "we should have resolved to a single encoding");
        auto encoding = cast<RankedTensorType>(result.getType()).getEncoding();
        // If the encoding is already what we want skip.
        if (encoding == *info.encodings.begin())
          continue;
        needRewrite = true;
      }
      if (needRewrite) {
        Operation *newOp = rewriteOp(&op);
        for (Region &R : newOp->getRegions())
          queue.push_back(&R);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        rewriteYieldOp(yieldOp);
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
        rewriteConditionOp(conditionOp);
      } else if (reduceToScalar(&op)) {
        rewriteReduceToScalar(&op);
      } else if (auto assertOp = dyn_cast<AssertOp>(&op)) {
        rewriteAssertOp(assertOp);
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
        for (OpOperand &operand : op.getOpOperands()) {
          auto it = layouts.find(operand.get());
          if (it == layouts.end())
            continue;
          Attribute encoding =
              cast<RankedTensorType>(operand.get().getType()).getEncoding();
          Value newOperand = getValueAs(operand.get(), encoding);
          op.setOperand(operand.getOperandNumber(), newOperand);
        }
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      }
    }
  }
  for (Operation *op : llvm::reverse(opToDelete))
    op->erase();
}

void LayoutPropagation::map(Value old, Value newV) {
  rewriteMapping[{old, cast<RankedTensorType>(newV.getType()).getEncoding()}] =
      newV;
}

Value LayoutPropagation::getValueAs(Value value, Attribute encoding) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    Value rewrittenValue;
    auto layoutIt = layouts.find(value);
    if (layoutIt == layouts.end()) {
      rewrittenValue = value;
    } else {
      assert(layoutIt->second.encodings.size() == 1 &&
             "we should have resolved to a single encoding");
      Attribute encodingPicked = *(layoutIt->second.encodings.begin());
      if (encodingPicked == tensorType.getEncoding())
        rewrittenValue = value;
      else
        rewrittenValue = rewriteMapping[{value, encodingPicked}];
    }
    assert(rewrittenValue);
    if (cast<RankedTensorType>(rewrittenValue.getType()).getEncoding() ==
        encoding)
      return rewrittenValue;
    OpBuilder rewriter(value.getContext());
    rewriter.setInsertionPointAfterValue(rewrittenValue);
    auto tmpType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), encoding);
    Value converted = rewriter.create<ConvertLayoutOp>(value.getLoc(), tmpType,
                                                       rewrittenValue);
    // TODO: we could cache the conversion.
    return converted;
  }
  return value;
}

Operation *LayoutPropagation::cloneElementwise(OpBuilder &rewriter,
                                               Operation *op,
                                               Attribute encoding) {
  Operation *newOp = rewriter.clone(*op);

  Attribute operandEnc;
  if (op->getNumOperands() > 0) {
    operandEnc = inferSrcEncoding(op, encoding);
    assert(operandEnc);
  }

  for (OpOperand &operand : op->getOpOperands()) {
    newOp->setOperand(operand.getOperandNumber(),
                      getValueAs(operand.get(), operandEnc));
  }

  for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
    auto origType = dyn_cast<RankedTensorType>(op->getResult(i).getType());
    if (!origType)
      continue;
    auto newType = RankedTensorType::get(origType.getShape(),
                                         origType.getElementType(), encoding);
    newOp->getResult(i).setType(newType);
  }
  return newOp;
}

Operation *LayoutPropagation::rewriteForOp(scf::ForOp forOp) {
  SmallVector<Value> operands;
  OpBuilder rewriter(forOp);
  for (auto [operand, result] :
       llvm::zip(forOp.getInitArgs(), forOp.getResults())) {
    Value convertedOperand = operand;
    if (layouts.count(result))
      convertedOperand =
          getValueAs(operand, *layouts[result].encodings.begin());
    operands.push_back(convertedOperand);
  }
  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), operands);
  newForOp->setAttrs(forOp->getAttrs());
  newForOp.getBody()->getOperations().splice(
      newForOp.getBody()->getOperations().begin(),
      forOp.getBody()->getOperations());

  for (auto [oldResult, newResult] :
       llvm::zip(forOp.getResults(), newForOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    map(oldResult, newResult);
  }

  for (auto [oldArg, newArg] : llvm::zip(forOp.getBody()->getArguments(),
                                         newForOp.getBody()->getArguments())) {
    if (oldArg.getType() == newArg.getType()) {
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }
    map(oldArg, newArg);
  }
  return newForOp.getOperation();
}

Operation *LayoutPropagation::rewriteWhileOp(scf::WhileOp whileOp) {
  SmallVector<Value> operands;
  SmallVector<Type> returnTypes;
  OpBuilder rewriter(whileOp);
  for (auto [operand, arg] :
       llvm::zip(whileOp->getOperands(), whileOp.getBeforeArguments())) {
    Value convertedOperand = operand;
    if (layouts.count(arg))
      convertedOperand = getValueAs(operand, *layouts[arg].encodings.begin());
    operands.push_back(convertedOperand);
  }
  for (Value ret : whileOp.getResults()) {
    auto it = layouts.find(ret);
    if (it == layouts.end()) {
      returnTypes.push_back(ret.getType());
      continue;
    }
    auto origType = dyn_cast<RankedTensorType>(ret.getType());
    auto newType =
        RankedTensorType::get(origType.getShape(), origType.getElementType(),
                              it->second.encodings[0]);
    returnTypes.push_back(newType);
  }

  auto newWhileOp =
      rewriter.create<scf::WhileOp>(whileOp.getLoc(), returnTypes, operands);
  SmallVector<Type> argsTypesBefore;
  for (Value operand : operands)
    argsTypesBefore.push_back(operand.getType());
  SmallVector<Location> bbArgLocsBefore(argsTypesBefore.size(),
                                        whileOp.getLoc());
  SmallVector<Location> bbArgLocsAfter(returnTypes.size(), whileOp.getLoc());
  rewriter.createBlock(&newWhileOp.getBefore(), {}, argsTypesBefore,
                       bbArgLocsBefore);
  rewriter.createBlock(&newWhileOp.getAfter(), {}, returnTypes, bbArgLocsAfter);

  for (int i = 0; i < whileOp.getNumRegions(); ++i) {
    newWhileOp->getRegion(i).front().getOperations().splice(
        newWhileOp->getRegion(i).front().getOperations().begin(),
        whileOp->getRegion(i).front().getOperations());
  }

  auto remapArg = [&](Value oldVal, Value newVal) {
    if (oldVal.getType() == newVal.getType())
      oldVal.replaceAllUsesWith(newVal);
    else
      map(oldVal, newVal);
  };
  for (auto [oldResult, newResult] :
       llvm::zip(whileOp.getResults(), newWhileOp.getResults()))
    remapArg(oldResult, newResult);
  for (auto [oldArg, newArg] :
       llvm::zip(whileOp.getBeforeArguments(), newWhileOp.getBeforeArguments()))
    remapArg(oldArg, newArg);
  for (auto [oldArg, newArg] :
       llvm::zip(whileOp.getAfterArguments(), newWhileOp.getAfterArguments()))
    remapArg(oldArg, newArg);
  return newWhileOp.getOperation();
}

Operation *LayoutPropagation::rewriteIfOp(scf::IfOp ifOp) {
  SmallVector<Value> operands;
  OpBuilder rewriter(ifOp);
  SmallVector<Type> newResultTypes(ifOp->getResultTypes());
  for (unsigned i = 0, e = ifOp->getNumResults(); i < e; ++i) {
    auto it = layouts.find(ifOp->getResult(i));
    if (it == layouts.end())
      continue;
    auto origType = cast<RankedTensorType>(ifOp->getResult(i).getType());
    Attribute encoding = *(it->second.encodings.begin());
    newResultTypes[i] = RankedTensorType::get(
        origType.getShape(), origType.getElementType(), encoding);
  }
  auto newIfOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                            ifOp.getCondition(), true, true);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  for (auto [oldResult, newResult] :
       llvm::zip(ifOp.getResults(), newIfOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    map(oldResult, newResult);
  }
  return newIfOp.getOperation();
}

void LayoutPropagation::rewriteYieldOp(scf::YieldOp yieldOp) {
  Operation *parentOp = yieldOp->getParentOp();
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    Type yieldType = operand.get().getType();
    if (isa<scf::ForOp, scf::IfOp>(parentOp))
      yieldType = parentOp->getResult(operand.getOperandNumber()).getType();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
      yieldType =
          whileOp.getBeforeArguments()[operand.getOperandNumber()].getType();
    auto tensorType = dyn_cast<RankedTensorType>(yieldType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    yieldOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteConditionOp(scf::ConditionOp conditionOp) {
  scf::WhileOp whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
  for (unsigned i = 1; i < conditionOp->getNumOperands(); ++i) {
    OpOperand &operand = conditionOp->getOpOperand(i);
    Type argType = whileOp->getResult(operand.getOperandNumber() - 1).getType();
    auto tensorType = dyn_cast<RankedTensorType>(argType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    conditionOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteReduceToScalar(Operation *reduceOp) {
  OpBuilder rewriter(reduceOp);
  Attribute srcEncoding;
  // Since all the operands need to have the same encoding pick the first one
  // and use it for all the operands.
  for (Value operand : reduceOp->getOperands()) {
    auto it = layouts.find(operand);
    if (it != layouts.end()) {
      srcEncoding = it->second.encodings[0];
      break;
    }
  }
  if (!srcEncoding)
    return;
  for (OpOperand &operand : reduceOp->getOpOperands()) {
    Value newOperand = getValueAs(operand.get(), srcEncoding);
    reduceOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteAssertOp(AssertOp assertOp) {
  Attribute srcEncoding;
  // Only need to deal with the first operand which is the condition tensor.
  Value operand = assertOp->getOperand(0);
  auto it = layouts.find(operand);
  if (it == layouts.end())
    return;
  srcEncoding = it->second.encodings[0];
  Value newOperand = getValueAs(operand, srcEncoding);
  assertOp->setOperand(0, newOperand);
}

Operation *LayoutPropagation::rewriteOp(Operation *op) {
  opToDelete.insert(op);
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewriteForOp(forOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewriteWhileOp(whileOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp);
  OpBuilder rewriter(op);
  Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
  if (auto convertOp = dyn_cast<ConvertLayoutOp>(op)) {
    Attribute srcEncoding = convertOp.getSrc().getType().getEncoding();
    auto it = layouts.find(convertOp.getSrc());
    if (it != layouts.end())
      srcEncoding = *(it->second.encodings.begin());
    Value src = getValueAs(convertOp.getSrc(), srcEncoding);
    auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    auto newType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), encoding);
    auto cvt = rewriter.create<ConvertLayoutOp>(op->getLoc(), newType, src);
    map(op->getResult(0), cvt.getResult());
    return cvt.getOperation();
  }
  if (canFoldIntoConversion(op, encoding)) {
    Operation *newOp = rewriter.clone(*op);
    auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
    auto newType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), encoding);
    auto cvt = rewriter.create<ConvertLayoutOp>(op->getLoc(), newType,
                                                newOp->getResult(0));
    map(op->getResult(0), cvt.getResult());
    return cvt.getOperation();
  }
  if (op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp, GatherOp,
          ConvertLayoutOp, nvidia_gpu::WarpGroupDotWaitOp>(op)) {
    Operation *newOp = cloneElementwise(rewriter, op, encoding);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      if (oldResult.getType() == newResult.getType()) {
        oldResult.replaceAllUsesWith(newResult);
        continue;
      }
      map(oldResult, newResult);
    }
    return newOp;
  }
  llvm::report_fatal_error("unexpected op in rewrite");
  return nullptr;
}

bool canBeRemat(Operation *op) {
  if (isa<LoadOp, StoreOp>(op))
    return !isExpensiveLoadOrStore(op);
  if (isa<AtomicRMWOp, AtomicCASOp, DotOp>(op))
    return false;
  if (auto gather = dyn_cast<GatherOp>(op))
    return !gather.getEfficientLayout();

  if (isa<scf::WhileOp, scf::ConditionOp>(op))
    return false;

  return true;
}

void LayoutRematerialization::updateRematMapping(
    SmallVector<std::tuple<Value, Value>> &values) {
  for (auto [old, newV] : values) {
    auto it = mappedValues.find(old);
    if (it != mappedValues.end()) {
      Attribute encoding = it->second;
      auto rematIt = rematMapping.find({old, it->second});
      assert(rematIt != rematMapping.end());
      Value replacedValue = rematIt->second;
      rematMapping.erase(rematIt);
      mappedValues.erase(it);
      // Loop through the replacement value to find the new version of remat
      // value. This should be okay as the number of values should be small.
      for (auto [before, after] : values) {
        if (before == replacedValue) {
          replacedValue = after;
          break;
        }
      }
      rematMapping[{newV, encoding}] = replacedValue;
      mappedValues[newV] = encoding;
    }
  }
}

void LayoutRematerialization::rewriteSlice(SetVector<Value> &slice,
                                           DenseMap<Value, Attribute> &layout,
                                           ConvertLayoutOp convertOp,
                                           IRMapping &mapping) {
  SetVector<Operation *> opsToRewrite;
  // Keep track of yield operands that need to be duplicated.
  DenseMap<Operation *, SmallVector<int>> yieldOperandsMap;
  // Keep these around to remove them from the slice after our collection pass
  // This ensures we don't duplicate them during an for rewrite or causing the
  // for/yield to fall out of sync
  SetVector<Value> valuesWithExistingRemat;
  for (Value v : slice) {
    auto layoutIt = layout.find(v);
    assert(layoutIt != layout.end());
    // If we already have a remat value for this value, use it.
    if (Value remat = getRematValue(v, layoutIt->second)) {
      mapping.map(v, remat);
      valuesWithExistingRemat.insert(v);
      continue;
    }
    if (v.getDefiningOp()) {
      opsToRewrite.insert(v.getDefiningOp());
      if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
        unsigned operandIdx = cast<OpResult>(v).getResultNumber();
        opsToRewrite.insert(ifOp.thenYield().getOperation());
        yieldOperandsMap[ifOp.thenYield()].push_back(operandIdx);
        opsToRewrite.insert(ifOp.elseYield().getOperation());
        yieldOperandsMap[ifOp.elseYield()].push_back(operandIdx);
      }
    } else {
      BlockArgument blockArg = cast<BlockArgument>(v);
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (auto loopOp = cast<LoopLikeOpInterface>(parentOp)) {
        opsToRewrite.insert(loopOp.getOperation());
        OpOperand *operand = loopOp.getTiedLoopYieldedValue(blockArg);
        auto yieldOp = blockArg.getOwner()->getTerminator();
        yieldOperandsMap[yieldOp].push_back(operand->getOperandNumber());
        opsToRewrite.insert(yieldOp);
      }
    }
  }
  slice.set_subtract(valuesWithExistingRemat);
  opsToRewrite = multiRootTopologicalSort(opsToRewrite);

  // replaceAllUsesWith calls delayed until after initial rewrite.
  // This is required for slice.count(value) to work mid rewrite.
  SmallVector<std::tuple<Value, Value>> replacements;

  SmallVector<Operation *> deadOps;
  IRRewriter builder(slice.begin()->getContext());
  for (Operation *op : opsToRewrite) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Keep a mapping of the operands index to the new operands index.
      SmallVector<std::pair<size_t, size_t>> argMapping;
      SmallVector<Value> newOperands;
      for (auto arg : forOp.getRegionIterArgs()) {
        if (slice.count(arg)) {
          OpOperand &initVal = *forOp.getTiedLoopInit(arg);
          argMapping.push_back(std::make_pair(
              forOp.getTiedLoopResult(&initVal).getResultNumber(),
              forOp.getInitArgs().size() + newOperands.size()));
          newOperands.push_back(mapping.lookup(initVal.get()));
        }
      }
      // Create a new for loop with the new operands.
      scf::ForOp newForOp = replaceForOpWithNewSignature(
          builder, forOp, newOperands, replacements);
      deadOps.push_back(forOp.getOperation());
      Block &loopBody = *newForOp.getBody();
      for (auto m : argMapping) {
        mapping.map(forOp.getResult(m.first), newForOp.getResult(m.second));
        int numIndVars = newForOp.getNumInductionVars();
        mapping.map(loopBody.getArgument(m.first + numIndVars),
                    loopBody.getArgument(m.second + numIndVars));
        LLVM_DEBUG({
          DBGS() << "mapping forOp "
                 << loopBody.getArgument(m.first + numIndVars) << " to "
                 << loopBody.getArgument(m.second + numIndVars) << '\n';
        });
        // The result is not in the layout/slice, the argument is.
        Value oldArg = loopBody.getArgument(m.first + numIndVars);
        addRematValue(newForOp.getResult(m.first), layout[oldArg],
                      newForOp.getResult(m.second));
        addRematValue(oldArg, layout[oldArg],
                      loopBody.getArgument(m.second + numIndVars));
      }
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Type> newTypes;
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          auto it = layout.find(res);
          assert(it != layout.end());

          auto oldType = cast<RankedTensorType>(res.getType());
          auto newType = RankedTensorType::get(
              oldType.getShape(), oldType.getElementType(), it->second);
          newTypes.push_back(newType);
        }
      }
      scf::IfOp newIfOp =
          replaceIfOpWithNewSignature(builder, ifOp, newTypes, replacements);
      unsigned oldIdx = 0;
      unsigned newIdx = ifOp.getNumResults();
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          // Why can't we use res instead of ifOp.getResult(oldIdx)?
          mapping.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
          addRematValue(ifOp.getResult(oldIdx), layout[res],
                        newIfOp.getResult(newIdx));
          ++newIdx;
        }
        ++oldIdx;
      }
      deadOps.push_back(ifOp.getOperation());
      continue;
    }
    builder.setInsertionPoint(op);
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto yieldOperands = llvm::to_vector(yieldOp.getOperands());
      SmallVector<int> operandsToRewrite = yieldOperandsMap[op];
      // Sort so that operands are added in the same order as the new scf
      // results/arguments.
      std::sort(operandsToRewrite.begin(), operandsToRewrite.end());
      for (int operandIdx : operandsToRewrite) {
        yieldOperands.push_back(mapping.lookup(yieldOp.getOperand(operandIdx)));
      }
      builder.create<scf::YieldOp>(op->getLoc(), yieldOperands);
      op->erase();
      continue;
    }
    if (isa<arith::ConstantOp>(op)) {
      Operation *newOp = builder.clone(*op);
      auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
      auto newType = RankedTensorType::get(tensorType.getShape(),
                                           tensorType.getElementType(),
                                           layout[op->getResult(0)]);
      auto cvt = builder.create<ConvertLayoutOp>(op->getLoc(), newType,
                                                 newOp->getResult(0));
      mapping.map(op->getResult(0), cvt.getResult());
      addRematValue(op->getResult(0), layout[op->getResult(0)],
                    cvt.getResult());
      continue;
    }
    Operation *newOp = builder.clone(*op, mapping);
    for (auto [old, newV] : llvm::zip(op->getResults(), newOp->getResults())) {
      auto it = layout.find(old);
      if (it == layout.end())
        continue;
      auto newType = RankedTensorType::get(
          cast<RankedTensorType>(old.getType()).getShape(),
          cast<RankedTensorType>(old.getType()).getElementType(), it->second);
      newV.setType(newType);
      addRematValue(old, it->second, newV);
    }
  }
  // Check mapping and see if there are existing convertOps on the old Argument
  convertOp.replaceAllUsesWith(mapping.lookup(convertOp.getSrc()));
  opToDelete.insert(convertOp);

  updateRematMapping(replacements);
  for (auto &kv : replacements) {
    builder.replaceAllUsesWith(std::get<0>(kv), std::get<1>(kv));
  }

  for (Operation *op : deadOps)
    opToDelete.insert(op);
}

void LayoutRematerialization::rewriteSlice(SetVector<Value> &slice,
                                           DenseMap<Value, Attribute> &layout,
                                           ConvertLayoutOp convertOp) {
  IRMapping mapping;
  rewriteSlice(slice, layout, convertOp, mapping);
}

LogicalResult LayoutRematerialization::getConvertBackwardSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation) {
  // Allow re-using existing conversions for a value. Check dominance of any
  // reusable materializations against the root value. This is sufficient
  // because the conversions are processed in post-order.
  auto getExistingConversion = [&](OpOperand &value, Attribute encoding) {
    Value remat = getRematValue(value.get(), encoding);
    if (!remat)
      return Value();
    // `value` can be replaced with an existing rematerialization if it
    // dominates the current use of value.
    Operation *user = value.getOwner();
    if (domInfo.properlyDominates(remat, user)) {
      return remat;
    }
    // FIXME: If the current user is a conversion, then we know it will become
    // a no-op when its operand is replaced with `remat`, but we need to check
    // that its users are all dominated by `remat` so the IR is valid.
    // if (isa<ConvertLayoutOp>(user) && remat.getDefiningOp() &&
    //     domInfo.properlyDominates(user, remat.getDefiningOp())) {
    //   for (Operation *op : user->getUsers()) {
    //     if (!domInfo.dominates(remat, op))
    //       return Value();
    //   }
    //   return remat;
    // }
    return Value();
  };

  return mlir::getConvertBackwardSlice(root, slice, rootEncoding, layout,
                                       stopPropagation, getExistingConversion);
}

LogicalResult LayoutRematerialization::getRematerializableSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation) {
  LogicalResult result = getConvertBackwardSlice(root, rootEncoding, slice,
                                                 layout, stopPropagation);
  if (result.failed() || slice.empty())
    return failure();

  // Check if all the operations in the slice can be rematerialized.
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      if (!canBeRemat(op))
        return failure();
    }
  }
  return success();
}

void LayoutRematerialization::backwardRematerialization() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    backwardRematerialization(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertOnTopOfExtOrBroadcast(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertIntoConditionals() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertIntoConditionals(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::backwardRematerialization(
    ConvertLayoutOp convertOp) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return;
  Value oldV = convertOp.getSrc();
  LDBG("check backward remat with source " << oldV << " encoding "
                                           << targetType.getEncoding());
  // Check to see if there are existing remat'ed values for the pair of oldValue
  // and encoding. Make sure it dominates the current conversion.
  Value newV = getRematValue(oldV, targetType.getEncoding());
  if (newV && domInfo.properlyDominates(newV, convertOp)) {
    // Replace it with the remat'ed value.
    convertOp.replaceAllUsesWith(newV);
    opToDelete.insert(convertOp);
    LDBG("found remat'ed value" << newV);
    return;
  }

  // 1. Take a backward slice of all the tensor dependencies that can be
  // rematerialized.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout);
  if (result.failed()) {
    LDBG("  getRematerializableSlice failed");
    return;
  }

  LLVM_DEBUG({
    DBGS() << "  remat convert op " << convertOp << '\n';
    for (Value v : slice)
      DBGS() << "    " << v << '\n';
  });
  // 2. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp);
}

void LayoutRematerialization::hoistConvertDotOperand() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertDotOperand(convertOp);
    if (!opToDelete.contains(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertDotOperand(
    ConvertLayoutOp convertOp) {
  auto targetType = convertOp.getType();
  // The pass is targeted to Nvidia mma/wgmma dot operands

  auto canBePipelined = [&](ConvertLayoutOp convertOp) {
    // FIXME: Check that the parent is a for loop
    auto parent = convertOp->getParentOp();
    if (!parent)
      return false;

    // Find all the dot-like ops in the for loop that have a nvidia dot operand
    // encoding on the lhs and check if any of them post-dominates the load +
    // cvt
    SmallVector<Operation *> dotLikeOps;
    parent->walk([&](Operation *op) {
      if (!isa<mlir::triton::DotOpInterface>(op))
        return;
      auto opType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!opType)
        return;
      auto dotEnc = dyn_cast<DotOperandEncodingAttr>(opType.getEncoding());
      if (!dotEnc)
        return;
      if (isa<NvidiaMmaEncodingAttr>(dotEnc.getParent()))
        dotLikeOps.push_back(op);
    });
    if (dotLikeOps.empty())
      return false;
    return llvm::any_of(dotLikeOps, [&](Operation *dot) {
      return postDomInfo.postDominates(dot, convertOp);
    });
  };

  // We move convert #dot_operand next to their loads. This is done
  // so that it's then easy to pipeline these loads
  if (!canBePipelined(convertOp))
    return;

  // We hoist over any operation that can be done without data movement between
  // threads We do views and elementwise pure ops for now
  auto noDataMovement = [](Operation *op) {
    return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
           isa<BroadcastOp, Fp4ToFpOp, ConvertLayoutOp>(op) || isView(op);
  };
  // Stop the slice as soon as we find an operation that cannot be done without
  // data movement between threads
  auto stop = std::not_fn(noDataMovement);

  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  // Set-up the conversion "cache"
  LogicalResult result = getConvertBackwardSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout, stop);
  if (result.failed())
    return;

  IRMapping mapping;
  OpBuilder builder(convertOp.getContext());
  SetVector<Value> innerSlice;
  for (Value v : slice) {
    if (!v.getDefiningOp()) {
      LLVM_DEBUG(
          { DBGS() << "  Block arguments not supported. Got " << v << "\n"; });
      return;
    }
    auto loadOp = dyn_cast<LoadOp>(v.getDefiningOp());
    // We expect the leaves of the slice to be Load or arith::Constant
    // This could be generalised if necessary
    if (!loadOp) {
      auto op = v.getDefiningOp();
      if (isa<arith::ConstantOp>(op) || noDataMovement(op)) {
        innerSlice.insert(v);
        continue;
      } else {
        LLVM_DEBUG({
          DBGS() << "  Leaves must be Load or Constant. Got " << v << "\n";
        });
        return;
      }
    }
    builder.setInsertionPointAfter(loadOp);
    auto type = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!type)
      continue;
    auto newType = RankedTensorType::get(type.getShape(), type.getElementType(),
                                         layout[loadOp]);
    auto newConvertOp = builder.create<ConvertLayoutOp>(
        convertOp.getLoc(), newType, loadOp.getResult());
    mapping.map(loadOp.getResult(), newConvertOp.getResult());
  }

  if (innerSlice.empty()) {
    return;
  }

  LLVM_DEBUG({
    DBGS() << "  Hoisting " << convertOp << '\n';
    for (Value v : innerSlice)
      DBGS() << "    " << v << '\n';
  });

  rewriteSlice(innerSlice, layout, convertOp, mapping);
}

// For convert left we try to hoist them above type extension to reduce the cost
// of the convert.
void LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast(
    ConvertLayoutOp convertOp) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return;

  auto isExtOrBroadcastOp = [](Operation *op) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, BroadcastOp,
            ExpandDimsOp>(op)) {
      return true;
    }
    if (auto fpToFpOp = dyn_cast<FpToFpOp>(op)) {
      auto srcType = cast<RankedTensorType>(fpToFpOp.getOperand().getType());
      return getElementBitWidth(srcType) <
             getElementBitWidth(cast<RankedTensorType>(fpToFpOp.getType()));
    }
    return false;
  };
  // 1. Take a backward slice of all the tensor dependencies.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      isExtOrBroadcastOp);
  if (result.failed())
    return;

  Operation *extOrBroadcatOp = nullptr;
  unsigned sliceSize = slice.size();
  for (unsigned i = 0; i < sliceSize; i++) {
    Value v = slice[i];
    Operation *op = v.getDefiningOp();
    if (!op)
      continue;
    if (isExtOrBroadcastOp(op)) {
      SetVector<Value> tempSlice;
      DenseMap<Value, Attribute> tempLayout;
      Attribute srcEncoding = inferSrcEncoding(op, layout[v]);
      if (!srcEncoding)
        return;
      LogicalResult result = getRematerializableSlice(
          op->getOpOperand(0), srcEncoding, tempSlice, tempLayout);
      // If we can rematerialize the rest of the ext slice we can ignore this
      // ext as it won't need a convert.
      if (result.succeeded()) {
        slice.insert(tempSlice.begin(), tempSlice.end());
        layout.insert(tempLayout.begin(), tempLayout.end());
        continue;
      }
      // Only apply it if there is a single ext op otherwise we would have to
      // duplicate the convert.
      if (extOrBroadcatOp != nullptr)
        return;
      extOrBroadcatOp = op;
    }
  }

  if (extOrBroadcatOp == nullptr)
    return;
  Attribute dstEncoding = layout[extOrBroadcatOp->getResult(0)];
  Attribute srcEncoding = inferSrcEncoding(extOrBroadcatOp, dstEncoding);
  if (!srcEncoding)
    return;
  // Move the convert before the ext op and rewrite the slice.
  OpBuilder builder(extOrBroadcatOp);
  auto tensorType =
      cast<RankedTensorType>(extOrBroadcatOp->getOperand(0).getType());
  auto newType = RankedTensorType::get(
      tensorType.getShape(), tensorType.getElementType(), srcEncoding);
  auto newConvertOp = builder.create<ConvertLayoutOp>(
      convertOp.getLoc(), newType, extOrBroadcatOp->getOperand(0));
  Operation *newExtOrBroadcast = builder.clone(*extOrBroadcatOp);
  newExtOrBroadcast->setOperand(0, newConvertOp.getResult());
  auto oldExtOrBroadcastType =
      cast<RankedTensorType>(extOrBroadcatOp->getResult(0).getType());
  Type newExtOrBroadcasrType = RankedTensorType::get(
      oldExtOrBroadcastType.getShape(), oldExtOrBroadcastType.getElementType(),
      dstEncoding);
  newExtOrBroadcast->getResult(0).setType(newExtOrBroadcasrType);
  IRMapping mapping;
  mapping.map(extOrBroadcatOp->getResult(0), newExtOrBroadcast->getResult(0));
  slice.remove(extOrBroadcatOp->getResult(0));
  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp, mapping);
}

void LayoutRematerialization::hoistConvertIntoConditionals(
    ConvertLayoutOp convertOp) {
  // Take the backward slice of tensor dependencies rooted at the conversion,
  // stopping at conditionals. This subslice is used to initialize the analysis.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  auto isIfOp = [](Operation *op) { return isa<scf::IfOp>(op); };
  if (failed(getRematerializableSlice(convertOp.getSrcMutable(),
                                      convertOp.getType().getEncoding(), slice,
                                      layout, isIfOp)))
    return;

  // These are the conditional edges above which conversions should be hoisted.
  // The value represents the `scf.if` op result and the operand represents the
  // edge into one of the branches.
  SmallVector<std::pair<Value, OpOperand *>> hoistAbove;

  // The list of `scf.if` op results in the slice that are not rematerializable.
  // Hoisting is terminated at these values.
  SmallVector<OpResult> terminals;

  // This loop recurses through the subslices of the backwards dependencies, so
  // re-query the size of `slice`.
  for (unsigned i = 0; i != slice.size(); ++i) {
    Value v = slice[i];
    auto ifOp = v.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      continue;

    Attribute rootLayout = layout.at(v);
    unsigned resIdx = cast<OpResult>(v).getResultNumber();

    // Take the backward slice along each branch.
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    OpOperand &thenRes = thenYield.getResultsMutable()[resIdx];
    OpOperand &elseRes = elseYield.getResultsMutable()[resIdx];

    SetVector<Value> thenSlice, elseSlice;
    DenseMap<Value, Attribute> thenLayout, elseLayout;

    LogicalResult thenResult = getRematerializableSlice(
        thenRes, rootLayout, thenSlice, thenLayout, isIfOp);
    LogicalResult elseResult = getRematerializableSlice(
        elseRes, rootLayout, elseSlice, elseLayout, isIfOp);

    // If propagation across both edges of this conditional succeeded, then we
    // don't need to hoist across it. Merge into the current slice.
    if (succeeded(thenResult) && succeeded(elseResult)) {
      slice.insert(thenSlice.begin(), thenSlice.end());
      slice.insert(elseSlice.begin(), elseSlice.end());
      layout.insert(thenLayout.begin(), thenLayout.end());
      layout.insert(elseLayout.begin(), elseLayout.end());
      continue;
    }

    // If propagation across both edges failed, then this conditional
    // terminates backwards rematerialization.
    if (failed(thenResult) && failed(elseResult)) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    // Only hoist into conditionals inside loops. The assumption is that an if
    // inside a loop executes fewer than the total number of loop iterations,
    // making this hoist profitable.
    if (!isa<scf::ForOp>(ifOp->getParentOp())) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    // The layout conversion can be rematerialized along one edge but not the
    // other. We can hoist the conversion into the other branch. Push this
    // into the subslice list for analysis.
    if (succeeded(thenResult)) {
      hoistAbove.emplace_back(v, &elseRes);
      slice.insert(thenSlice.begin(), thenSlice.end());
      layout.insert(thenLayout.begin(), thenLayout.end());
    } else {
      hoistAbove.emplace_back(v, &thenRes);
      slice.insert(elseSlice.begin(), elseSlice.end());
      layout.insert(elseLayout.begin(), elseLayout.end());
    }
  }

  // Exit early if there is nothing to do.
  if (hoistAbove.empty())
    return;

  // Rematerialize failed hoists right before the condtional, and hoist those
  // that succeeded into the branch and then rewrite the slice.
  IRMapping mapping;
  auto hoistRemat = [&](OpBuilder &b, Value v, Attribute encoding) {
    auto tensorType = cast<RankedTensorType>(v.getType());
    auto newType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), encoding);
    Value newCvt = b.create<ConvertLayoutOp>(convertOp.getLoc(), newType, v);

    mapping.map(v, newCvt);
    slice.remove(v);
  };
  for (Value v : terminals) {
    OpBuilder b(v.getContext());
    b.setInsertionPointAfter(v.getDefiningOp());
    hoistRemat(b, v, layout.at(v));
  }
  for (auto [result, edge] : hoistAbove) {
    OpBuilder b(edge->getOwner());
    hoistRemat(b, edge->get(), layout.at(result));
  }
  rewriteSlice(slice, layout, convertOp, mapping);
}

void backwardRematerialization(ModuleOp module) {
  module.walk([](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp);
    layoutRemat.backwardRematerialization();
    layoutRemat.cleanup();
  });
}

void hoistConvert(ModuleOp module) {
  SmallVector<ConvertLayoutOp> convertOps;
  module.walk([](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp);
    layoutRemat.hoistConvertOnTopOfExtOrBroadcast();
    layoutRemat.cleanup();

    layoutRemat = LayoutRematerialization(funcOp);
    layoutRemat.hoistConvertIntoConditionals();
    layoutRemat.cleanup();

    layoutRemat = LayoutRematerialization(funcOp);
    layoutRemat.hoistConvertDotOperand();
    layoutRemat.cleanup();
  });
}
} // namespace

class TritonGPURemoveLayoutConversionsPass
    : public impl::TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
  // Cleanup convert ops.
  void cleanupConvertOps() {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet cleanUpPatterns(context);
    ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns, context);
    if (applyPatternsGreedily(m, std::move(cleanUpPatterns)).failed()) {
      signalPassFailure();
    }

    LLVM_DEBUG({
      DBGS() << "Module after canonicalizing:\n";
      m.dump();
    });
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // 1. Propagate layout forward starting from "anchor" ops.
    m.walk([](FuncOp funcOp) {
      LayoutPropagation layoutPropagation(funcOp);
      layoutPropagation.initAnchorLayout();
      layoutPropagation.propagateLayout();
      layoutPropagation.resolveConflicts();
      layoutPropagation.rewrite();
    });

    LLVM_DEBUG({
      DBGS() << "Module after propagating layouts forward:\n";
      m.dump();
    });

    cleanupConvertOps();

    // 2. For remaining convert ops, try to rematerialize the slice of producer
    // operation to avoid having to convert.
    backwardRematerialization(m);
    LLVM_DEBUG({
      DBGS() << "Module after backward remat:\n";
      m.dump();
    });

    // Cleanup dummy converts created during backward remat.
    cleanupConvertOps();

    // 3. For remaining converts, try to hoist them above cast generating larger
    // size types in order to reduce the cost of the convert op.
    hoistConvert(m);
    LLVM_DEBUG({
      DBGS() << "Module after hoisting converts:\n";
      m.dump();
    });

    // 4. Apply clean up patterns to remove remove dead convert and dead code
    // generated by the previous transformations.
    RewritePatternSet cleanUpPatterns2(context);
    populateForOpDeadArgumentElimination(cleanUpPatterns2);
    scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    scf::IfOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    if (applyPatternsGreedily(m, std::move(cleanUpPatterns2)).failed()) {
      signalPassFailure();
    }
    LLVM_DEBUG({
      DBGS() << "Module after final cleanups:\n";
      m.dump();
    });
  }
};

} // namespace mlir::triton::gpu
