//===- AffineOps.cpp - MLIR Affine Operations -----------------------------===//
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

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/AffineStructures.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"
using namespace mlir;
using llvm::dbgs;

#define DEBUG_TYPE "affine-analysis"

//===----------------------------------------------------------------------===//
// AffineOpsDialect
//===----------------------------------------------------------------------===//

AffineOpsDialect::AffineOpsDialect(MLIRContext *context)
    : Dialect(/*namePrefix=*/"", context) {
  addOperations<AffineApplyOp, AffineForOp, AffineIfOp>();
}

// Value can be used as a dimension id if it is valid as a symbol, or
// it is an induction variable, or it is a result of affine apply operation
// with dimension id arguments.
bool mlir::isValidDim(const Value *value) {
  if (auto *inst = value->getDefiningInst()) {
    // Top level instruction or constant operation is ok.
    if (inst->getParentInst() == nullptr || inst->isa<ConstantOp>())
      return true;
    // Affine apply operation is ok if all of its operands are ok.
    if (auto op = inst->dyn_cast<AffineApplyOp>())
      return op->isValidDim();
    return false;
  }
  // This value is a block argument.
  return true;
}

// Value can be used as a symbol if it is a constant, or it is defined at
// the top level, or it is a result of affine apply operation with symbol
// arguments.
bool mlir::isValidSymbol(const Value *value) {
  if (auto *inst = value->getDefiningInst()) {
    // Top level instruction or constant operation is ok.
    if (inst->getParentInst() == nullptr || inst->isa<ConstantOp>())
      return true;
    // Affine apply operation is ok if all of its operands are ok.
    if (auto op = inst->dyn_cast<AffineApplyOp>())
      return op->isValidSymbol();
    return false;
  }
  // Otherwise, the only valid symbol is a non induction variable block
  // argument.
  return !isForInductionVar(value);
}

//===----------------------------------------------------------------------===//
// AffineApplyOp
//===----------------------------------------------------------------------===//

void AffineApplyOp::build(Builder *builder, OperationState *result,
                          AffineMap map, ArrayRef<Value *> operands) {
  result->addOperands(operands);
  result->types.append(map.getNumResults(), builder->getIndexType());
  result->addAttribute("map", builder->getAffineMapAttr(map));
}

bool AffineApplyOp::parse(OpAsmParser *parser, OperationState *result) {
  auto &builder = parser->getBuilder();
  auto affineIntTy = builder.getIndexType();

  AffineMapAttr mapAttr;
  unsigned numDims;
  if (parser->parseAttribute(mapAttr, "map", result->attributes) ||
      parseDimAndSymbolList(parser, result->operands, numDims) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;
  auto map = mapAttr.getValue();

  if (map.getNumDims() != numDims ||
      numDims + map.getNumSymbols() != result->operands.size()) {
    return parser->emitError(parser->getNameLoc(),
                             "dimension or symbol index mismatch");
  }

  result->types.append(map.getNumResults(), affineIntTy);
  return false;
}

void AffineApplyOp::print(OpAsmPrinter *p) const {
  auto map = getAffineMap();
  *p << "affine_apply " << map;
  printDimAndSymbolList(operand_begin(), operand_end(), map.getNumDims(), p);
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"map");
}

bool AffineApplyOp::verify() const {
  // Check that affine map attribute was specified.
  auto affineMapAttr = getAttrOfType<AffineMapAttr>("map");
  if (!affineMapAttr)
    return emitOpError("requires an affine map");

  // Check input and output dimensions match.
  auto map = affineMapAttr.getValue();

  // Verify that operand count matches affine map dimension and symbol count.
  if (getNumOperands() != map.getNumDims() + map.getNumSymbols())
    return emitOpError(
        "operand count and affine map dimension and symbol count must match");

  // Verify that result count matches affine map result count.
  if (map.getNumResults() != 1)
    return emitOpError("mapping must produce one value");

  return false;
}

/// Returns an AffineValueMap representing this affine apply.
AffineValueMap AffineApplyOp::getAsAffineValueMap() {
  SmallVector<Value *, 8> operands(getOperands());
  return AffineValueMap(getAffineMap(), operands, getResult());
}

// The result of the affine apply operation can be used as a dimension id if it
// is a CFG value or if it is an Value, and all the operands are valid
// dimension ids.
bool AffineApplyOp::isValidDim() const {
  return llvm::all_of(getOperands(),
                      [](const Value *op) { return mlir::isValidDim(op); });
}

// The result of the affine apply operation can be used as a symbol if it is
// a CFG value or if it is an Value, and all the operands are symbols.
bool AffineApplyOp::isValidSymbol() const {
  return llvm::all_of(getOperands(),
                      [](const Value *op) { return mlir::isValidSymbol(op); });
}

Attribute AffineApplyOp::constantFold(ArrayRef<Attribute> operands,
                                      MLIRContext *context) const {
  auto map = getAffineMap();
  SmallVector<Attribute, 1> result;
  if (map.constantFold(operands, result))
    return Attribute();
  return result[0];
}

namespace {
/// SimplifyAffineApply operations.
///
struct SimplifyAffineApply : public RewritePattern {
  SimplifyAffineApply(MLIRContext *context)
      : RewritePattern(AffineApplyOp::getOperationName(), 1, context) {}

  PatternMatchResult match(Instruction *op) const override;
  void rewrite(Instruction *op, std::unique_ptr<PatternState> state,
               PatternRewriter &rewriter) const override;
};
} // end anonymous namespace.

namespace {
/// An `AffineApplyNormalizer` is a helper class that is not visible to the user
/// and supports renumbering operands of AffineApplyOp. This acts as a
/// reindexing map of Value* to positional dims or symbols and allows
/// simplifications such as:
///
/// ```mlir
///    %1 = affine_apply (d0, d1) -> (d0 - d1) (%0, %0)
/// ```
///
/// into:
///
/// ```mlir
///    %1 = affine_apply () -> (0)
/// ```
struct AffineApplyNormalizer {
  AffineApplyNormalizer(AffineMap map, ArrayRef<Value *> operands);

  /// Returns the AffineMap resulting from normalization.
  AffineMap getAffineMap() { return affineMap; }

  SmallVector<Value *, 8> getOperands() {
    SmallVector<Value *, 8> res(reorderedDims);
    res.append(concatenatedSymbols.begin(), concatenatedSymbols.end());
    return res;
  }

private:
  /// Helper function to insert `v` into the coordinate system of the current
  /// AffineApplyNormalizer. Returns the AffineDimExpr with the corresponding
  /// renumbered position.
  AffineDimExpr applyOneDim(Value *v);

  /// Given an `other` normalizer, this rewrites `other.affineMap` in the
  /// coordinate system of the current AffineApplyNormalizer.
  /// Returns the rewritten AffineMap and updates the dims and symbols of
  /// `this`.
  AffineMap renumber(const AffineApplyNormalizer &other);

  /// Given an `app`, rewrites `app.getAffineMap()` in the coordinate system of
  /// the current AffineApplyNormalizer.
  /// Returns the rewritten AffineMap and updates the dims and symbols of
  /// `this`.
  AffineMap renumber(const AffineApplyOp &app);

  /// Maps of Value* to position in `affineMap`.
  DenseMap<Value *, unsigned> dimValueToPosition;

  /// Ordered dims and symbols matching positional dims and symbols in
  /// `affineMap`.
  SmallVector<Value *, 8> reorderedDims;
  SmallVector<Value *, 8> concatenatedSymbols;

  AffineMap affineMap;

  /// Used with RAII to control the depth at which AffineApply are composed
  /// recursively. Only accepts depth 1 for now.
  /// Note that if one wishes to compose all AffineApply in the program and
  /// follows program order, maxdepth 1 is sufficient. This is as much as this
  /// abstraction is willing to support for now.
  static unsigned &affineApplyDepth() {
    static thread_local unsigned depth = 0;
    return depth;
  }
  static constexpr unsigned kMaxAffineApplyDepth = 1;

  AffineApplyNormalizer() { affineApplyDepth()++; }

public:
  ~AffineApplyNormalizer() { affineApplyDepth()--; }
};

/// FIXME: this is massive overkill for simple obviously always matching
/// canonicalizations.  Fix the pattern rewriter to make this easy.
struct SimplifyAffineApplyState : public PatternState {
  AffineMap map;
  SmallVector<Value *, 8> operands;

  SimplifyAffineApplyState(AffineMap map,
                           const SmallVector<Value *, 8> &operands)
      : map(map), operands(operands) {}
};

} // end anonymous namespace.

AffineDimExpr AffineApplyNormalizer::applyOneDim(Value *v) {
  DenseMap<Value *, unsigned>::iterator iterPos;
  bool inserted = false;
  std::tie(iterPos, inserted) =
      dimValueToPosition.insert(std::make_pair(v, dimValueToPosition.size()));
  if (inserted) {
    reorderedDims.push_back(v);
  }
  return getAffineDimExpr(iterPos->second, v->getFunction()->getContext())
      .cast<AffineDimExpr>();
}

AffineMap AffineApplyNormalizer::renumber(const AffineApplyNormalizer &other) {
  SmallVector<AffineExpr, 8> dimRemapping;
  for (auto *v : other.reorderedDims) {
    auto kvp = other.dimValueToPosition.find(v);
    if (dimRemapping.size() <= kvp->second)
      dimRemapping.resize(kvp->second + 1);
    dimRemapping[kvp->second] = applyOneDim(kvp->first);
  }
  unsigned numSymbols = concatenatedSymbols.size();
  unsigned numOtherSymbols = other.concatenatedSymbols.size();
  SmallVector<AffineExpr, 8> symRemapping(numOtherSymbols);
  for (unsigned idx = 0; idx < numOtherSymbols; ++idx) {
    symRemapping[idx] =
        getAffineSymbolExpr(idx + numSymbols, other.affineMap.getContext());
  }
  concatenatedSymbols.insert(concatenatedSymbols.end(),
                             other.concatenatedSymbols.begin(),
                             other.concatenatedSymbols.end());
  auto map = other.affineMap;
  return map.replaceDimsAndSymbols(dimRemapping, symRemapping,
                                   dimRemapping.size(), symRemapping.size());
}

AffineMap AffineApplyNormalizer::renumber(const AffineApplyOp &app) {
  assert(app.getAffineMap().getRangeSizes().empty() && "Non-empty range sizes");

  // Create the AffineApplyNormalizer for the operands of this
  // AffineApplyOp and combine it with the current AffineApplyNormalizer.
  SmallVector<Value *, 8> operands(
      const_cast<AffineApplyOp &>(app).getOperands().begin(),
      const_cast<AffineApplyOp &>(app).getOperands().end());
  AffineApplyNormalizer normalizer(app.getAffineMap(), operands);
  return renumber(normalizer);
}

AffineApplyNormalizer::AffineApplyNormalizer(AffineMap map,
                                             ArrayRef<Value *> operands)
    : AffineApplyNormalizer() {
  assert(map.getRangeSizes().empty() && "Unbounded map expected");
  assert(map.getNumInputs() == operands.size() &&
         "number of operands does not match the number of map inputs");

  SmallVector<AffineExpr, 8> exprs;
  for (auto en : llvm::enumerate(operands)) {
    auto *t = en.value();
    assert(t->getType().isIndex());
    bool operandNotFromAffineApply =
        !t->getDefiningInst() || !t->getDefiningInst()->isa<AffineApplyOp>();
    if (operandNotFromAffineApply ||
        affineApplyDepth() > kMaxAffineApplyDepth) {
      if (en.index() < map.getNumDims()) {
        exprs.push_back(applyOneDim(t));
      } else {
        // Composition of mathematical symbols must occur by concatenation.
        // A subsequent canonicalization will drop duplicates. Duplicates are
        // not dropped here because it would just amount to code duplication.
        concatenatedSymbols.push_back(t);
      }
    } else {
      auto *inst = t->getDefiningInst();
      auto app = inst->dyn_cast<AffineApplyOp>();
      auto tmpMap = renumber(*app);
      exprs.push_back(tmpMap.getResult(0));
    }
  }

  // Map is already composed.
  if (exprs.empty()) {
    affineMap = map;
    return;
  }

  auto numDims = dimValueToPosition.size();
  auto numSymbols = concatenatedSymbols.size() - map.getNumSymbols();
  auto exprsMap = AffineMap::get(numDims, numSymbols, exprs, {});
  LLVM_DEBUG(map.print(dbgs() << "\nCompose map: "));
  LLVM_DEBUG(exprsMap.print(dbgs() << "\nWith map: "));
  LLVM_DEBUG(map.compose(exprsMap).print(dbgs() << "\nResult: "));

  affineMap = simplifyAffineMap(map.compose(exprsMap));
  LLVM_DEBUG(affineMap.print(dbgs() << "\nSimplified result: "));
  LLVM_DEBUG(dbgs() << "\n");
}

/// Implements `map` and `operands` composition and simplification to support
/// `makeComposedAffineApply`. This can be called to achieve the same effects
/// on `map` and `operands` without creating an AffineApplyOp that needs to be
/// immediately deleted.
static void composeAffineMapAndOperands(AffineMap *map,
                                        SmallVectorImpl<Value *> *operands) {
  AffineApplyNormalizer normalizer(*map, *operands);
  auto normalizedMap = normalizer.getAffineMap();
  auto normalizedOperands = normalizer.getOperands();
  canonicalizeMapAndOperands(&normalizedMap, &normalizedOperands);
  *map = normalizedMap;
  *operands = normalizedOperands;
  assert(*map);
}

void mlir::fullyComposeAffineMapAndOperands(
    AffineMap *map, SmallVectorImpl<Value *> *operands) {
  while (llvm::any_of(*operands, [](Value *v) {
    return v->getDefiningInst() && v->getDefiningInst()->isa<AffineApplyOp>();
  })) {
    composeAffineMapAndOperands(map, operands);
  }
}

OpPointer<AffineApplyOp>
mlir::makeComposedAffineApply(FuncBuilder *b, Location loc, AffineMap map,
                              ArrayRef<Value *> operands) {
  AffineMap normalizedMap = map;
  SmallVector<Value *, 8> normalizedOperands(operands.begin(), operands.end());
  composeAffineMapAndOperands(&normalizedMap, &normalizedOperands);
  assert(normalizedMap);
  return b->create<AffineApplyOp>(loc, normalizedMap, normalizedOperands);
}

void mlir::canonicalizeMapAndOperands(
    AffineMap *map, llvm::SmallVectorImpl<Value *> *operands) {
  if (!map || operands->empty())
    return;

  assert(map->getNumInputs() == operands->size() &&
         "map inputs must match number of operands");

  // Check to see what dims are used.
  llvm::SmallBitVector usedDims(map->getNumDims());
  llvm::SmallBitVector usedSyms(map->getNumSymbols());
  map->walkExprs([&](AffineExpr expr) {
    if (auto dimExpr = expr.dyn_cast<AffineDimExpr>())
      usedDims[dimExpr.getPosition()] = true;
    else if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>())
      usedSyms[symExpr.getPosition()] = true;
  });

  auto *context = map->getContext();

  SmallVector<Value *, 8> resultOperands;
  resultOperands.reserve(operands->size());

  llvm::SmallDenseMap<Value *, AffineExpr, 8> seenDims;
  SmallVector<AffineExpr, 8> dimRemapping(map->getNumDims());
  unsigned nextDim = 0;
  for (unsigned i = 0, e = map->getNumDims(); i != e; ++i) {
    if (usedDims[i]) {
      auto it = seenDims.find((*operands)[i]);
      if (it == seenDims.end()) {
        dimRemapping[i] = getAffineDimExpr(nextDim++, context);
        resultOperands.push_back((*operands)[i]);
        seenDims.insert(std::make_pair((*operands)[i], dimRemapping[i]));
      } else {
        dimRemapping[i] = it->second;
      }
    }
  }
  llvm::SmallDenseMap<Value *, AffineExpr, 8> seenSymbols;
  SmallVector<AffineExpr, 8> symRemapping(map->getNumSymbols());
  unsigned nextSym = 0;
  for (unsigned i = 0, e = map->getNumSymbols(); i != e; ++i) {
    if (usedSyms[i]) {
      auto it = seenSymbols.find((*operands)[i + map->getNumDims()]);
      if (it == seenSymbols.end()) {
        symRemapping[i] = getAffineSymbolExpr(nextSym++, context);
        resultOperands.push_back((*operands)[i + map->getNumDims()]);
        seenSymbols.insert(std::make_pair((*operands)[i + map->getNumDims()],
                                          symRemapping[i]));
      } else {
        symRemapping[i] = it->second;
      }
    }
  }
  *map =
      map->replaceDimsAndSymbols(dimRemapping, symRemapping, nextDim, nextSym);
  *operands = resultOperands;
}

PatternMatchResult SimplifyAffineApply::match(Instruction *op) const {
  auto apply = op->cast<AffineApplyOp>();
  auto map = apply->getAffineMap();

  AffineMap oldMap = map;
  SmallVector<Value *, 8> resultOperands(apply->getOperands());
  composeAffineMapAndOperands(&map, &resultOperands);
  if (map != oldMap)
    return matchSuccess(
        std::make_unique<SimplifyAffineApplyState>(map, resultOperands));

  return matchFailure();
}

void SimplifyAffineApply::rewrite(Instruction *op,
                                  std::unique_ptr<PatternState> state,
                                  PatternRewriter &rewriter) const {
  auto *applyState = static_cast<SimplifyAffineApplyState *>(state.get());
  rewriter.replaceOpWithNewOp<AffineApplyOp>(op, applyState->map,
                                             applyState->operands);
}

void AffineApplyOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyAffineApply>(context));
}

//===----------------------------------------------------------------------===//
// AffineForOp
//===----------------------------------------------------------------------===//

void AffineForOp::build(Builder *builder, OperationState *result,
                        ArrayRef<Value *> lbOperands, AffineMap lbMap,
                        ArrayRef<Value *> ubOperands, AffineMap ubMap,
                        int64_t step) {
  assert((!lbMap && lbOperands.empty()) ||
         lbOperands.size() == lbMap.getNumInputs() &&
             "lower bound operand count does not match the affine map");
  assert((!ubMap && ubOperands.empty()) ||
         ubOperands.size() == ubMap.getNumInputs() &&
             "upper bound operand count does not match the affine map");
  assert(step > 0 && "step has to be a positive integer constant");

  // Add an attribute for the step.
  result->addAttribute(getStepAttrName(),
                       builder->getIntegerAttr(builder->getIndexType(), step));

  // Add the lower bound.
  result->addAttribute(getLowerBoundAttrName(),
                       builder->getAffineMapAttr(lbMap));
  result->addOperands(lbOperands);

  // Add the upper bound.
  result->addAttribute(getUpperBoundAttrName(),
                       builder->getAffineMapAttr(ubMap));
  result->addOperands(ubOperands);

  // Reserve a block list for the body.
  result->reserveBlockLists(/*numReserved=*/1);

  // Set the operands list as resizable so that we can freely modify the bounds.
  result->setOperandListToResizable();
}

void AffineForOp::build(Builder *builder, OperationState *result, int64_t lb,
                        int64_t ub, int64_t step) {
  auto lbMap = AffineMap::getConstantMap(lb, builder->getContext());
  auto ubMap = AffineMap::getConstantMap(ub, builder->getContext());
  return build(builder, result, {}, lbMap, {}, ubMap, step);
}

bool AffineForOp::verify() const {
  const auto &bodyBlockList = getInstruction()->getBlockList(0);

  // The body block list must contain a single basic block.
  if (bodyBlockList.empty() ||
      std::next(bodyBlockList.begin()) != bodyBlockList.end())
    return emitOpError("expected body block list to have a single block");

  // Check that the body defines as single block argument for the induction
  // variable.
  const auto *body = getBody();
  if (body->getNumArguments() != 1 ||
      !body->getArgument(0)->getType().isIndex())
    return emitOpError("expected body to have a single index argument for the "
                       "induction variable");

  // TODO: check that loop bounds are properly formed.
  return false;
}

/// Parse a for operation loop bounds.
static bool parseBound(bool isLower, OperationState *result, OpAsmParser *p) {
  // 'min' / 'max' prefixes are generally syntactic sugar, but are required if
  // the map has multiple results.
  bool failedToParsedMinMax = p->parseOptionalKeyword(isLower ? "max" : "min");

  auto &builder = p->getBuilder();
  auto boundAttrName = isLower ? AffineForOp::getLowerBoundAttrName()
                               : AffineForOp::getUpperBoundAttrName();

  // Parse ssa-id as identity map.
  SmallVector<OpAsmParser::OperandType, 1> boundOpInfos;
  if (p->parseOperandList(boundOpInfos))
    return true;

  if (!boundOpInfos.empty()) {
    // Check that only one operand was parsed.
    if (boundOpInfos.size() > 1)
      return p->emitError(p->getNameLoc(),
                          "expected only one loop bound operand");

    // TODO: improve error message when SSA value is not an affine integer.
    // Currently it is 'use of value ... expects different type than prior uses'
    if (p->resolveOperand(boundOpInfos.front(), builder.getIndexType(),
                          result->operands))
      return true;

    // Create an identity map using symbol id. This representation is optimized
    // for storage. Analysis passes may expand it into a multi-dimensional map
    // if desired.
    AffineMap map = builder.getSymbolIdentityMap();
    result->addAttribute(boundAttrName, builder.getAffineMapAttr(map));
    return false;
  }

  // Get the attribute location.
  llvm::SMLoc attrLoc;
  p->getCurrentLocation(&attrLoc);

  Attribute boundAttr;
  if (p->parseAttribute(boundAttr, builder.getIndexType(), boundAttrName.data(),
                        result->attributes))
    return true;

  // Parse full form - affine map followed by dim and symbol list.
  if (auto affineMapAttr = boundAttr.dyn_cast<AffineMapAttr>()) {
    unsigned currentNumOperands = result->operands.size();
    unsigned numDims;
    if (parseDimAndSymbolList(p, result->operands, numDims))
      return true;

    auto map = affineMapAttr.getValue();
    if (map.getNumDims() != numDims)
      return p->emitError(
          p->getNameLoc(),
          "dim operand count and integer set dim count must match");

    unsigned numDimAndSymbolOperands =
        result->operands.size() - currentNumOperands;
    if (numDims + map.getNumSymbols() != numDimAndSymbolOperands)
      return p->emitError(
          p->getNameLoc(),
          "symbol operand count and integer set symbol count must match");

    // If the map has multiple results, make sure that we parsed the min/max
    // prefix.
    if (map.getNumResults() > 1 && failedToParsedMinMax) {
      if (isLower) {
        return p->emitError(attrLoc, "lower loop bound affine map with "
                                     "multiple results requires 'max' prefix");
      }
      return p->emitError(attrLoc, "upper loop bound affine map with multiple "
                                   "results requires 'min' prefix");
    }
    return false;
  }

  // Parse custom assembly form.
  if (auto integerAttr = boundAttr.dyn_cast<IntegerAttr>()) {
    result->attributes.pop_back();
    result->addAttribute(
        boundAttrName, builder.getAffineMapAttr(
                           builder.getConstantAffineMap(integerAttr.getInt())));
    return false;
  }

  return p->emitError(
      p->getNameLoc(),
      "expected valid affine map representation for loop bounds");
}

bool AffineForOp::parse(OpAsmParser *parser, OperationState *result) {
  auto &builder = parser->getBuilder();
  // Parse the induction variable followed by '='.
  if (parser->parseBlockListEntryBlockArgument(builder.getIndexType()) ||
      parser->parseEqual())
    return true;

  // Parse loop bounds.
  if (parseBound(/*isLower=*/true, result, parser) ||
      parser->parseKeyword("to", " between bounds") ||
      parseBound(/*isLower=*/false, result, parser))
    return true;

  // Parse the optional loop step, we default to 1 if one is not present.
  if (parser->parseOptionalKeyword("step")) {
    result->addAttribute(
        getStepAttrName(),
        builder.getIntegerAttr(builder.getIndexType(), /*value=*/1));
  } else {
    llvm::SMLoc stepLoc;
    IntegerAttr stepAttr;
    if (parser->getCurrentLocation(&stepLoc) ||
        parser->parseAttribute(stepAttr, builder.getIndexType(),
                               getStepAttrName().data(), result->attributes))
      return true;

    if (stepAttr.getValue().getSExtValue() < 0)
      return parser->emitError(
          stepLoc,
          "expected step to be representable as a positive signed integer");
  }

  // Parse the body block list.
  result->reserveBlockLists(/*numReserved=*/1);
  if (parser->parseBlockList())
    return true;

  // Set the operands list as resizable so that we can freely modify the bounds.
  result->setOperandListToResizable();
  return false;
}

static void printBound(AffineBound bound, const char *prefix, OpAsmPrinter *p) {
  AffineMap map = bound.getMap();

  // Check if this bound should be printed using custom assembly form.
  // The decision to restrict printing custom assembly form to trivial cases
  // comes from the will to roundtrip MLIR binary -> text -> binary in a
  // lossless way.
  // Therefore, custom assembly form parsing and printing is only supported for
  // zero-operand constant maps and single symbol operand identity maps.
  if (map.getNumResults() == 1) {
    AffineExpr expr = map.getResult(0);

    // Print constant bound.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 0) {
      if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
        *p << constExpr.getValue();
        return;
      }
    }

    // Print bound that consists of a single SSA symbol if the map is over a
    // single symbol.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 1) {
      if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
        p->printOperand(bound.getOperand(0));
        return;
      }
    }
  } else {
    // Map has multiple results. Print 'min' or 'max' prefix.
    *p << prefix << ' ';
  }

  // Print the map and its operands.
  p->printAffineMap(map);
  printDimAndSymbolList(bound.operand_begin(), bound.operand_end(),
                        map.getNumDims(), p);
}

void AffineForOp::print(OpAsmPrinter *p) const {
  *p << "for ";
  p->printOperand(getBody()->getArgument(0));
  *p << " = ";
  printBound(getLowerBound(), "max", p);
  *p << " to ";
  printBound(getUpperBound(), "min", p);

  if (getStep() != 1)
    *p << " step " << getStep();
  p->printBlockList(getInstruction()->getBlockList(0),
                    /*printEntryBlockArgs=*/false);
}

namespace {
/// This is a pattern to fold constant loop bounds.
struct AffineForLoopBoundFolder : public RewritePattern {
  /// The rootOpName is the name of the root operation to match against.
  AffineForLoopBoundFolder(MLIRContext *context)
      : RewritePattern(AffineForOp::getOperationName(), 1, context) {}

  PatternMatchResult match(Instruction *op) const override {
    auto forOp = op->cast<AffineForOp>();

    // If the loop has non-constant bounds, it may be foldable.
    if (!forOp->hasConstantBounds())
      return matchSuccess();

    return matchFailure();
  }

  void rewrite(Instruction *op, PatternRewriter &rewriter) const override {
    auto forOp = op->cast<AffineForOp>();
    auto foldLowerOrUpperBound = [&forOp](bool lower) {
      // Check to see if each of the operands is the result of a constant.  If
      // so, get the value.  If not, ignore it.
      SmallVector<Attribute, 8> operandConstants;
      auto boundOperands = lower ? forOp->getLowerBoundOperands()
                                 : forOp->getUpperBoundOperands();
      for (const auto *operand : boundOperands) {
        Attribute operandCst;
        if (auto *operandOp = operand->getDefiningInst())
          if (auto operandConstantOp = operandOp->dyn_cast<ConstantOp>())
            operandCst = operandConstantOp->getValue();
        operandConstants.push_back(operandCst);
      }

      AffineMap boundMap =
          lower ? forOp->getLowerBoundMap() : forOp->getUpperBoundMap();
      assert(boundMap.getNumResults() >= 1 &&
             "bound maps should have at least one result");
      SmallVector<Attribute, 4> foldedResults;
      if (boundMap.constantFold(operandConstants, foldedResults))
        return;

      // Compute the max or min as applicable over the results.
      assert(!foldedResults.empty() &&
             "bounds should have at least one result");
      auto maxOrMin = foldedResults[0].cast<IntegerAttr>().getValue();
      for (unsigned i = 1, e = foldedResults.size(); i < e; i++) {
        auto foldedResult = foldedResults[i].cast<IntegerAttr>().getValue();
        maxOrMin = lower ? llvm::APIntOps::smax(maxOrMin, foldedResult)
                         : llvm::APIntOps::smin(maxOrMin, foldedResult);
      }
      lower ? forOp->setConstantLowerBound(maxOrMin.getSExtValue())
            : forOp->setConstantUpperBound(maxOrMin.getSExtValue());
    };

    // Try to fold the lower bound.
    if (!forOp->hasConstantLowerBound())
      foldLowerOrUpperBound(/*lower=*/true);

    // Try to fold the upper bound.
    if (!forOp->hasConstantUpperBound())
      foldLowerOrUpperBound(/*lower=*/false);

    rewriter.updatedRootInPlace(op);
  }
};
} // end anonymous namespace

void AffineForOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.push_back(std::make_unique<AffineForLoopBoundFolder>(context));
}

Block *AffineForOp::createBody() {
  auto &bodyBlockList = getBlockList();
  assert(bodyBlockList.empty() && "expected no existing body blocks");

  // Create a new block for the body, and add an argument for the induction
  // variable.
  Block *body = new Block();
  body->addArgument(IndexType::get(getInstruction()->getContext()));
  bodyBlockList.push_back(body);
  return body;
}

const AffineBound AffineForOp::getLowerBound() const {
  auto lbMap = getLowerBoundMap();
  return AffineBound(ConstOpPointer<AffineForOp>(*this), 0,
                     lbMap.getNumInputs(), lbMap);
}

const AffineBound AffineForOp::getUpperBound() const {
  auto lbMap = getLowerBoundMap();
  auto ubMap = getUpperBoundMap();
  return AffineBound(ConstOpPointer<AffineForOp>(*this), lbMap.getNumInputs(),
                     getNumOperands(), ubMap);
}

void AffineForOp::setLowerBound(ArrayRef<Value *> lbOperands, AffineMap map) {
  assert(lbOperands.size() == map.getNumInputs());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");

  SmallVector<Value *, 4> newOperands(lbOperands.begin(), lbOperands.end());

  auto ubOperands = getUpperBoundOperands();
  newOperands.append(ubOperands.begin(), ubOperands.end());
  getInstruction()->setOperands(newOperands);

  setAttr(Identifier::get(getLowerBoundAttrName(), map.getContext()),
          AffineMapAttr::get(map));
}

void AffineForOp::setUpperBound(ArrayRef<Value *> ubOperands, AffineMap map) {
  assert(ubOperands.size() == map.getNumInputs());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");

  SmallVector<Value *, 4> newOperands(getLowerBoundOperands());
  newOperands.append(ubOperands.begin(), ubOperands.end());
  getInstruction()->setOperands(newOperands);

  setAttr(Identifier::get(getUpperBoundAttrName(), map.getContext()),
          AffineMapAttr::get(map));
}

void AffineForOp::setLowerBoundMap(AffineMap map) {
  auto lbMap = getLowerBoundMap();
  assert(lbMap.getNumDims() == map.getNumDims() &&
         lbMap.getNumSymbols() == map.getNumSymbols());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");
  (void)lbMap;
  setAttr(Identifier::get(getLowerBoundAttrName(), map.getContext()),
          AffineMapAttr::get(map));
}

void AffineForOp::setUpperBoundMap(AffineMap map) {
  auto ubMap = getUpperBoundMap();
  assert(ubMap.getNumDims() == map.getNumDims() &&
         ubMap.getNumSymbols() == map.getNumSymbols());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");
  (void)ubMap;
  setAttr(Identifier::get(getUpperBoundAttrName(), map.getContext()),
          AffineMapAttr::get(map));
}

bool AffineForOp::hasConstantLowerBound() const {
  return getLowerBoundMap().isSingleConstant();
}

bool AffineForOp::hasConstantUpperBound() const {
  return getUpperBoundMap().isSingleConstant();
}

int64_t AffineForOp::getConstantLowerBound() const {
  return getLowerBoundMap().getSingleConstantResult();
}

int64_t AffineForOp::getConstantUpperBound() const {
  return getUpperBoundMap().getSingleConstantResult();
}

void AffineForOp::setConstantLowerBound(int64_t value) {
  setLowerBound(
      {}, AffineMap::getConstantMap(value, getInstruction()->getContext()));
}

void AffineForOp::setConstantUpperBound(int64_t value) {
  setUpperBound(
      {}, AffineMap::getConstantMap(value, getInstruction()->getContext()));
}

AffineForOp::operand_range AffineForOp::getLowerBoundOperands() {
  return {operand_begin(), operand_begin() + getLowerBoundMap().getNumInputs()};
}

AffineForOp::const_operand_range AffineForOp::getLowerBoundOperands() const {
  return {operand_begin(), operand_begin() + getLowerBoundMap().getNumInputs()};
}

AffineForOp::operand_range AffineForOp::getUpperBoundOperands() {
  return {operand_begin() + getLowerBoundMap().getNumInputs(), operand_end()};
}

AffineForOp::const_operand_range AffineForOp::getUpperBoundOperands() const {
  return {operand_begin() + getLowerBoundMap().getNumInputs(), operand_end()};
}

bool AffineForOp::matchingBoundOperandList() const {
  auto lbMap = getLowerBoundMap();
  auto ubMap = getUpperBoundMap();
  if (lbMap.getNumDims() != ubMap.getNumDims() ||
      lbMap.getNumSymbols() != ubMap.getNumSymbols())
    return false;

  unsigned numOperands = lbMap.getNumInputs();
  for (unsigned i = 0, e = lbMap.getNumInputs(); i < e; i++) {
    // Compare Value *'s.
    if (getOperand(i) != getOperand(numOperands + i))
      return false;
  }
  return true;
}

/// Returns the induction variable for this loop.
Value *AffineForOp::getInductionVar() { return getBody()->getArgument(0); }

/// Returns if the provided value is the induction variable of a AffineForOp.
bool mlir::isForInductionVar(const Value *val) {
  return getForInductionVarOwner(val) != nullptr;
}

/// Returns the loop parent of an induction variable. If the provided value is
/// not an induction variable, then return nullptr.
OpPointer<AffineForOp> mlir::getForInductionVarOwner(Value *val) {
  const BlockArgument *ivArg = dyn_cast<BlockArgument>(val);
  if (!ivArg || !ivArg->getOwner())
    return OpPointer<AffineForOp>();
  auto *containingInst = ivArg->getOwner()->getParent()->getContainingInst();
  if (!containingInst)
    return OpPointer<AffineForOp>();
  return containingInst->dyn_cast<AffineForOp>();
}
ConstOpPointer<AffineForOp> mlir::getForInductionVarOwner(const Value *val) {
  auto nonConstOwner = getForInductionVarOwner(const_cast<Value *>(val));
  return ConstOpPointer<AffineForOp>(nonConstOwner);
}

/// Extracts the induction variables from a list of AffineForOps and returns
/// them.
void mlir::extractForInductionVars(ArrayRef<OpPointer<AffineForOp>> forInsts,
                                   SmallVectorImpl<Value *> *ivs) {
  ivs->reserve(forInsts.size());
  for (auto forInst : forInsts)
    ivs->push_back(forInst->getInductionVar());
}

bool mlir::addAffineForOpDomain(ConstOpPointer<AffineForOp> forOp,
                                FlatAffineConstraints *constraints) {
  unsigned pos;
  // Pre-condition for this method.
  if (!constraints->findId(*forOp->getInductionVar(), &pos)) {
    assert(0 && "Value not found");
    return false;
  }

  if (forOp->getStep() != 1)
    LLVM_DEBUG(llvm::dbgs()
               << "Domain conservative: non-unit stride not handled\n");

  // Adds a lower or upper bound when the bounds aren't constant.
  auto addLowerOrUpperBound = [&](bool lower) -> bool {
    auto operands =
        lower ? forOp->getLowerBoundOperands() : forOp->getUpperBoundOperands();
    for (const auto &operand : operands) {
      unsigned loc;
      if (!constraints->findId(*operand, &loc)) {
        if (isValidSymbol(operand)) {
          constraints->addSymbolId(constraints->getNumSymbolIds(),
                                   const_cast<Value *>(operand));
          loc =
              constraints->getNumDimIds() + constraints->getNumSymbolIds() - 1;
          // Check if the symbol is a constant.
          if (auto *opInst = operand->getDefiningInst()) {
            if (auto constOp = opInst->dyn_cast<ConstantIndexOp>()) {
              constraints->setIdToConstant(*operand, constOp->getValue());
            }
          }
        } else {
          constraints->addDimId(constraints->getNumDimIds(),
                                const_cast<Value *>(operand));
          loc = constraints->getNumDimIds() - 1;
        }
      }
    }
    // Record positions of the operands in the constraint system.
    SmallVector<unsigned, 8> positions;
    for (const auto &operand : operands) {
      unsigned loc;
      if (!constraints->findId(*operand, &loc))
        assert(0 && "expected to be found");
      positions.push_back(loc);
    }

    auto boundMap =
        lower ? forOp->getLowerBoundMap() : forOp->getUpperBoundMap();

    FlatAffineConstraints localVarCst;
    std::vector<SmallVector<int64_t, 8>> flatExprs;
    if (!getFlattenedAffineExprs(boundMap, &flatExprs, &localVarCst)) {
      LLVM_DEBUG(llvm::dbgs() << "semi-affine expressions not yet supported\n");
      return false;
    }
    if (localVarCst.getNumLocalIds() > 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "loop bounds with mod/floordiv expr's not yet supported\n");
      return false;
    }

    for (const auto &flatExpr : flatExprs) {
      SmallVector<int64_t, 4> ineq(constraints->getNumCols(), 0);
      ineq[pos] = lower ? 1 : -1;
      for (unsigned j = 0, e = boundMap.getNumInputs(); j < e; j++) {
        ineq[positions[j]] = lower ? -flatExpr[j] : flatExpr[j];
      }
      // Constant term.
      ineq[constraints->getNumCols() - 1] =
          lower ? -flatExpr[flatExpr.size() - 1]
                // Upper bound in flattenedExpr is an exclusive one.
                : flatExpr[flatExpr.size() - 1] - 1;
      constraints->addInequality(ineq);
    }
    return true;
  };

  if (forOp->hasConstantLowerBound()) {
    constraints->addConstantLowerBound(pos, forOp->getConstantLowerBound());
  } else {
    // Non-constant lower bound case.
    if (!addLowerOrUpperBound(/*lower=*/true))
      return false;
  }

  if (forOp->hasConstantUpperBound()) {
    constraints->addConstantUpperBound(pos, forOp->getConstantUpperBound() - 1);
    return true;
  }
  // Non-constant upper bound case.
  return addLowerOrUpperBound(/*lower=*/false);
}

/// Returns an AffineValueMap representing this bound.
AffineValueMap AffineBound::getAsAffineValueMap() {
  SmallVector<Value *, 8> operands(getOperands());
  return AffineValueMap(getMap(), operands);
}

//===----------------------------------------------------------------------===//
// AffineIfOp
//===----------------------------------------------------------------------===//

void AffineIfOp::build(Builder *builder, OperationState *result,
                       IntegerSet condition,
                       ArrayRef<Value *> conditionOperands) {
  result->addAttribute(getConditionAttrName(), IntegerSetAttr::get(condition));
  result->addOperands(conditionOperands);

  // Reserve 2 block lists, one for the 'then' and one for the 'else' regions.
  result->reserveBlockLists(2);
}

bool AffineIfOp::verify() const {
  // Verify that we have a condition attribute.
  auto conditionAttr = getAttrOfType<IntegerSetAttr>(getConditionAttrName());
  if (!conditionAttr)
    return emitOpError("requires an integer set attribute named 'condition'");

  // Verify that the operands are valid dimension/symbols.
  IntegerSet condition = conditionAttr.getValue();
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    const Value *operand = getOperand(i);
    if (i < condition.getNumDims() && !isValidDim(operand))
      return emitOpError("operand cannot be used as a dimension id");
    if (i >= condition.getNumDims() && !isValidSymbol(operand))
      return emitOpError("operand cannot be used as a symbol");
  }

  // Verify that the entry of each child blocklist does not have arguments.
  for (const auto &blockList : getInstruction()->getBlockLists()) {
    if (blockList.empty())
      continue;

    // TODO(riverriddle) We currently do not allow multiple blocks in child
    // block lists.
    if (std::next(blockList.begin()) != blockList.end())
      return emitOpError(
          "expects only one block per 'if' or 'else' block list");
    if (blockList.front().getTerminator())
      return emitOpError("expects region block to not have a terminator");

    for (const auto &b : blockList)
      if (b.getNumArguments() != 0)
        return emitOpError(
            "requires that child entry blocks have no arguments");
  }
  return false;
}

bool AffineIfOp::parse(OpAsmParser *parser, OperationState *result) {
  // Parse the condition attribute set.
  IntegerSetAttr conditionAttr;
  unsigned numDims;
  if (parser->parseAttribute(conditionAttr, getConditionAttrName().data(),
                             result->attributes) ||
      parseDimAndSymbolList(parser, result->operands, numDims))
    return true;

  // Verify the condition operands.
  auto set = conditionAttr.getValue();
  if (set.getNumDims() != numDims)
    return parser->emitError(
        parser->getNameLoc(),
        "dim operand count and integer set dim count must match");
  if (numDims + set.getNumSymbols() != result->operands.size())
    return parser->emitError(
        parser->getNameLoc(),
        "symbol operand count and integer set symbol count must match");

  // Parse the 'then' block list.
  if (parser->parseBlockList())
    return true;

  // If we find an 'else' keyword then parse the else block list.
  if (!parser->parseOptionalKeyword("else")) {
    if (parser->parseBlockList())
      return true;
  }

  // Reserve 2 block lists, one for the 'then' and one for the 'else' regions.
  result->reserveBlockLists(2);
  return false;
}

void AffineIfOp::print(OpAsmPrinter *p) const {
  auto conditionAttr = getAttrOfType<IntegerSetAttr>(getConditionAttrName());
  *p << "if " << conditionAttr;
  printDimAndSymbolList(operand_begin(), operand_end(),
                        conditionAttr.getValue().getNumDims(), p);
  p->printBlockList(getInstruction()->getBlockList(0));

  // Print the 'else' block list if it has any blocks.
  const auto &elseBlockList = getInstruction()->getBlockList(1);
  if (!elseBlockList.empty()) {
    *p << " else";
    p->printBlockList(elseBlockList);
  }
}

IntegerSet AffineIfOp::getIntegerSet() const {
  return getAttrOfType<IntegerSetAttr>(getConditionAttrName()).getValue();
}
void AffineIfOp::setIntegerSet(IntegerSet newSet) {
  setAttr(
      Identifier::get(getConditionAttrName(), getInstruction()->getContext()),
      IntegerSetAttr::get(newSet));
}

/// Returns the list of 'then' blocks.
BlockList &AffineIfOp::getThenBlocks() {
  return getInstruction()->getBlockList(0);
}

/// Returns the list of 'else' blocks.
BlockList &AffineIfOp::getElseBlocks() {
  return getInstruction()->getBlockList(1);
}
