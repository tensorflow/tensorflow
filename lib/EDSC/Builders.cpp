//===- Builders.cpp - MLIR Declarative Builder Classes ----------*- C++ -*-===//
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

#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/StandardOps/Ops.h"

#include "llvm/ADT/Optional.h"

using namespace mlir;
using namespace mlir::edsc;

mlir::edsc::ScopedContext::ScopedContext(Function *fun, Location *loc)
    : builder(fun), location(loc ? *loc : fun->getLoc()),
      enclosingScopedContext(ScopedContext::getCurrentScopedContext()),
      nestedBuilder(nullptr) {
  getCurrentScopedContext() = this;
}

mlir::edsc::ScopedContext::ScopedContext(FuncBuilder builder, Location location)
    : builder(builder), location(location),
      enclosingScopedContext(ScopedContext::getCurrentScopedContext()),
      nestedBuilder(nullptr) {
  getCurrentScopedContext() = this;
}

mlir::edsc::ScopedContext::~ScopedContext() {
  assert(!nestedBuilder &&
         "Active NestedBuilder must have been exited at this point!");
  getCurrentScopedContext() = enclosingScopedContext;
}

ScopedContext *&mlir::edsc::ScopedContext::getCurrentScopedContext() {
  thread_local ScopedContext *context = nullptr;
  return context;
}

FuncBuilder *mlir::edsc::ScopedContext::getBuilder() {
  assert(ScopedContext::getCurrentScopedContext() &&
         "Unexpected Null ScopedContext");
  return &ScopedContext::getCurrentScopedContext()->builder;
}

Location mlir::edsc::ScopedContext::getLocation() {
  assert(ScopedContext::getCurrentScopedContext() &&
         "Unexpected Null ScopedContext");
  return ScopedContext::getCurrentScopedContext()->location;
}

MLIRContext *mlir::edsc::ScopedContext::getContext() {
  assert(getBuilder() && "Unexpected null builder");
  return getBuilder()->getContext();
}

mlir::edsc::ValueHandle::ValueHandle(index_t cst) {
  auto *b = ScopedContext::getBuilder();
  auto loc = ScopedContext::getLocation();
  v = b->create<ConstantIndexOp>(loc, cst.v).getResult();
  t = v->getType();
}

ValueHandle &mlir::edsc::ValueHandle::operator=(const ValueHandle &other) {
  assert(t == other.t && "Wrong type capture");
  assert(!v && "ValueHandle has already been captured, use a new name!");
  v = other.v;
  return *this;
}

ValueHandle
mlir::edsc::ValueHandle::createComposedAffineApply(AffineMap map,
                                                   ArrayRef<Value *> operands) {
  assert(ScopedContext::getBuilder() && "Unexpected null builder");
  Operation *op =
      makeComposedAffineApply(ScopedContext::getBuilder(),
                              ScopedContext::getLocation(), map, operands)
          .getOperation();
  assert(op->getNumResults() == 1 && "Not a single result AffineApply");
  return ValueHandle(op->getResult(0));
}

ValueHandle ValueHandle::create(StringRef name, ArrayRef<ValueHandle> operands,
                                ArrayRef<Type> resultTypes,
                                ArrayRef<NamedAttribute> attributes) {
  Operation *op =
      OperationHandle::create(name, operands, resultTypes, attributes);
  if (op->getNumResults() == 1) {
    return ValueHandle(op->getResult(0));
  }
  if (auto f = dyn_cast<AffineForOp>(op)) {
    return ValueHandle(f.getInductionVar());
  }
  llvm_unreachable("unsupported operation, use an OperationHandle instead");
}

OperationHandle OperationHandle::create(StringRef name,
                                        ArrayRef<ValueHandle> operands,
                                        ArrayRef<Type> resultTypes,
                                        ArrayRef<NamedAttribute> attributes) {
  OperationState state(ScopedContext::getContext(),
                       ScopedContext::getLocation(), name);
  SmallVector<Value *, 4> ops(operands.begin(), operands.end());
  state.addOperands(ops);
  state.addTypes(resultTypes);
  for (const auto &attr : attributes) {
    state.addAttribute(attr.first, attr.second);
  }
  return OperationHandle(ScopedContext::getBuilder()->createOperation(state));
}

BlockHandle mlir::edsc::BlockHandle::create(ArrayRef<Type> argTypes) {
  auto *currentB = ScopedContext::getBuilder();
  auto *ib = currentB->getInsertionBlock();
  auto ip = currentB->getInsertionPoint();
  BlockHandle res;
  res.block = ScopedContext::getBuilder()->createBlock();
  // createBlock sets the insertion point inside the block.
  // We do not want this behavior when using declarative builders with nesting.
  currentB->setInsertionPoint(ib, ip);
  for (auto t : argTypes) {
    res.block->addArgument(t);
  }
  return res;
}

static llvm::Optional<ValueHandle> emitStaticFor(ArrayRef<ValueHandle> lbs,
                                                 ArrayRef<ValueHandle> ubs,
                                                 int64_t step) {
  if (lbs.size() != 1 || ubs.size() != 1)
    return llvm::Optional<ValueHandle>();

  auto *lbDef = lbs.front().getValue()->getDefiningOp();
  auto *ubDef = ubs.front().getValue()->getDefiningOp();
  if (!lbDef || !ubDef)
    return llvm::Optional<ValueHandle>();

  auto lbConst = dyn_cast<ConstantIndexOp>(lbDef);
  auto ubConst = dyn_cast<ConstantIndexOp>(ubDef);
  if (!lbConst || !ubConst)
    return llvm::Optional<ValueHandle>();

  return ValueHandle::create<AffineForOp>(lbConst.getValue(),
                                          ubConst.getValue(), step);
}

mlir::edsc::LoopBuilder::LoopBuilder(ValueHandle *iv,
                                     ArrayRef<ValueHandle> lbHandles,
                                     ArrayRef<ValueHandle> ubHandles,
                                     int64_t step) {
  if (auto res = emitStaticFor(lbHandles, ubHandles, step)) {
    *iv = res.getValue();
  } else {
    SmallVector<Value *, 4> lbs(lbHandles.begin(), lbHandles.end());
    SmallVector<Value *, 4> ubs(ubHandles.begin(), ubHandles.end());
    *iv = ValueHandle::create<AffineForOp>(
        lbs, ScopedContext::getBuilder()->getMultiDimIdentityMap(lbs.size()),
        ubs, ScopedContext::getBuilder()->getMultiDimIdentityMap(ubs.size()),
        step);
  }
  auto *body = getForInductionVarOwner(iv->getValue()).getBody();
  enter(body, /*prev=*/1);
}

ValueHandle
mlir::edsc::LoopBuilder::operator()(ArrayRef<CapturableHandle> stmts) {
  // Call to `exit` must be explicit and asymmetric (cannot happen in the
  // destructor) because of ordering wrt comma operator.
  /// The particular use case concerns nested blocks:
  ///
  /// ```c++
  ///    For (&i, lb, ub, 1)({
  ///      /--- destructor for this `For` is not always called before ...
  ///      V
  ///      For (&j1, lb, ub, 1)({
  ///        some_op_1,
  ///      }),
  ///      /--- ... this scope is entered, resulting in improperly nested IR.
  ///      V
  ///      For (&j2, lb, ub, 1)({
  ///        some_op_2,
  ///      }),
  ///    });
  /// ```
  exit();
  return ValueHandle::null();
}

mlir::edsc::LoopNestBuilder::LoopNestBuilder(ArrayRef<ValueHandle *> ivs,
                                             ArrayRef<ValueHandle> lbs,
                                             ArrayRef<ValueHandle> ubs,
                                             ArrayRef<int64_t> steps) {
  assert(ivs.size() == lbs.size() && "Mismatch in number of arguments");
  assert(ivs.size() == ubs.size() && "Mismatch in number of arguments");
  assert(ivs.size() == steps.size() && "Mismatch in number of arguments");
  for (auto it : llvm::zip(ivs, lbs, ubs, steps)) {
    loops.emplace_back(std::get<0>(it), std::get<1>(it), std::get<2>(it),
                       std::get<3>(it));
  }
}

ValueHandle
mlir::edsc::LoopNestBuilder::operator()(ArrayRef<CapturableHandle> stmts) {
  // Iterate on the calling operator() on all the loops in the nest.
  // The iteration order is from innermost to outermost because enter/exit needs
  // to be asymmetric (i.e. enter() occurs on LoopBuilder construction, exit()
  // occurs on calling operator()). The asymmetry is required for properly
  // nesting imperfectly nested regions (see LoopBuilder::operator()).
  for (auto lit = loops.rbegin(), eit = loops.rend(); lit != eit; ++lit) {
    (*lit)({});
  }
  return ValueHandle::null();
}

mlir::edsc::BlockBuilder::BlockBuilder(BlockHandle bh, Append) {
  assert(bh && "Expected already captured BlockHandle");
  enter(bh.getBlock());
}

mlir::edsc::BlockBuilder::BlockBuilder(BlockHandle *bh,
                                       ArrayRef<ValueHandle *> args) {
  assert(!*bh && "BlockHandle already captures a block, use "
                 "the explicit BockBuilder(bh, Append())({}) syntax instead.");
  llvm::SmallVector<Type, 8> types;
  for (auto *a : args) {
    assert(!a->hasValue() &&
           "Expected delayed ValueHandle that has not yet captured.");
    types.push_back(a->getType());
  }
  *bh = BlockHandle::create(types);
  for (auto it : llvm::zip(args, bh->getBlock()->getArguments())) {
    *(std::get<0>(it)) = ValueHandle(std::get<1>(it));
  }
  enter(bh->getBlock());
}

/// Only serves as an ordering point between entering nested block and creating
/// stmts.
void mlir::edsc::BlockBuilder::operator()(ArrayRef<CapturableHandle> stmts) {
  // Call to `exit` must be explicit and asymmetric (cannot happen in the
  // destructor) because of ordering wrt comma operator.
  exit();
}

template <typename Op>
static ValueHandle createBinaryHandle(ValueHandle lhs, ValueHandle rhs) {
  return ValueHandle::create<Op>(lhs.getValue(), rhs.getValue());
}

static std::pair<AffineExpr, Value *>
categorizeValueByAffineType(MLIRContext *context, Value *val, unsigned &numDims,
                            unsigned &numSymbols) {
  AffineExpr d;
  Value *resultVal = nullptr;
  if (auto constant = dyn_cast_or_null<ConstantIndexOp>(val->getDefiningOp())) {
    d = getAffineConstantExpr(constant.getValue(), context);
  } else if (isValidSymbol(val) && !isValidDim(val)) {
    d = getAffineSymbolExpr(numSymbols++, context);
    resultVal = val;
  } else {
    assert(isValidDim(val) && "Must be a valid Dim");
    d = getAffineDimExpr(numDims++, context);
    resultVal = val;
  }
  return std::make_pair(d, resultVal);
}

static ValueHandle createBinaryIndexHandle(
    ValueHandle lhs, ValueHandle rhs,
    llvm::function_ref<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  MLIRContext *context = ScopedContext::getContext();
  unsigned numDims = 0, numSymbols = 0;
  AffineExpr d0, d1;
  Value *v0, *v1;
  std::tie(d0, v0) =
      categorizeValueByAffineType(context, lhs.getValue(), numDims, numSymbols);
  std::tie(d1, v1) =
      categorizeValueByAffineType(context, rhs.getValue(), numDims, numSymbols);
  SmallVector<Value *, 2> operands;
  if (v0) {
    operands.push_back(v0);
  }
  if (v1) {
    operands.push_back(v1);
  }
  auto map = AffineMap::get(numDims, numSymbols, {affCombiner(d0, d1)}, {});
  // TODO: createOrFold when available.
  return ValueHandle::createComposedAffineApply(map, operands);
}

template <typename IOp, typename FOp>
static ValueHandle createBinaryHandle(
    ValueHandle lhs, ValueHandle rhs,
    llvm::function_ref<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  auto thisType = lhs.getValue()->getType();
  auto thatType = rhs.getValue()->getType();
  assert(thisType == thatType && "cannot mix types in operators");
  (void)thisType;
  (void)thatType;
  if (thisType.isIndex()) {
    return createBinaryIndexHandle(lhs, rhs, affCombiner);
  } else if (thisType.isa<IntegerType>()) {
    return createBinaryHandle<IOp>(lhs, rhs);
  } else if (thisType.isa<FloatType>()) {
    return createBinaryHandle<FOp>(lhs, rhs);
  } else if (auto aggregateType = thisType.dyn_cast<VectorOrTensorType>()) {
    if (aggregateType.getElementType().isa<IntegerType>())
      return createBinaryHandle<IOp>(lhs, rhs);
    else if (aggregateType.getElementType().isa<FloatType>())
      return createBinaryHandle<FOp>(lhs, rhs);
  }
  llvm_unreachable("failed to create a ValueHandle");
}

ValueHandle mlir::edsc::op::operator+(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<AddIOp, AddFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 + d1; });
}

ValueHandle mlir::edsc::op::operator-(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<SubIOp, SubFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 - d1; });
}

ValueHandle mlir::edsc::op::operator*(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<MulIOp, MulFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 * d1; });
}

ValueHandle mlir::edsc::op::operator/(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<DivISOp, DivFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) -> AffineExpr {
        llvm_unreachable("only exprs of non-index type support operator/");
      });
}

ValueHandle mlir::edsc::op::operator%(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryHandle<RemISOp, RemFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 % d1; });
}

ValueHandle mlir::edsc::op::floorDiv(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryIndexHandle(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0.floorDiv(d1); });
}

ValueHandle mlir::edsc::op::ceilDiv(ValueHandle lhs, ValueHandle rhs) {
  return createBinaryIndexHandle(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0.ceilDiv(d1); });
}

ValueHandle mlir::edsc::op::operator!(ValueHandle value) {
  assert(value.getType().isInteger(1) && "expected boolean expression");
  return ValueHandle::create<ConstantIntOp>(1, 1) - value;
}

ValueHandle mlir::edsc::op::operator&&(ValueHandle lhs, ValueHandle rhs) {
  assert(lhs.getType().isInteger(1) && "expected boolean expression on LHS");
  assert(rhs.getType().isInteger(1) && "expected boolean expression on RHS");
  return lhs * rhs;
}

ValueHandle mlir::edsc::op::operator||(ValueHandle lhs, ValueHandle rhs) {
  return !(!lhs && !rhs);
}

static ValueHandle createComparisonExpr(CmpIPredicate predicate,
                                        ValueHandle lhs, ValueHandle rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  (void)lhsType;
  (void)rhsType;
  assert(lhsType == rhsType && "cannot mix types in operators");
  assert((lhsType.isa<IndexType>() || lhsType.isa<IntegerType>()) &&
         "only integer comparisons are supported");

  auto op = ScopedContext::getBuilder()->create<CmpIOp>(
      ScopedContext::getLocation(), predicate, lhs.getValue(), rhs.getValue());
  return ValueHandle(op.getResult());
}

ValueHandle mlir::edsc::op::operator==(ValueHandle lhs, ValueHandle rhs) {
  return createComparisonExpr(CmpIPredicate::EQ, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator!=(ValueHandle lhs, ValueHandle rhs) {
  return createComparisonExpr(CmpIPredicate::NE, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator<(ValueHandle lhs, ValueHandle rhs) {
  // TODO(ntv,zinenko): signed by default, how about unsigned?
  return createComparisonExpr(CmpIPredicate::SLT, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator<=(ValueHandle lhs, ValueHandle rhs) {
  return createComparisonExpr(CmpIPredicate::SLE, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator>(ValueHandle lhs, ValueHandle rhs) {
  return createComparisonExpr(CmpIPredicate::SGT, lhs, rhs);
}
ValueHandle mlir::edsc::op::operator>=(ValueHandle lhs, ValueHandle rhs) {
  return createComparisonExpr(CmpIPredicate::SGE, lhs, rhs);
}
