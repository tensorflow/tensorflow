//===- Builders.cpp - MLIR Declarative Linalg Builders --------------------===//
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

#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/Functional.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::edsc::ops;

static void getMaxDimIndex(ArrayRef<StructuredIndexed> structuredIndices,
                           unsigned &pos) {
  for (auto sidx : structuredIndices) {
    for (auto expr : sidx.getExprs()) {
      expr.walk([&pos](AffineExpr e) {
        if (auto d = e.dyn_cast<AffineDimExpr>())
          pos = std::max(pos, d.getPosition());
      });
    }
  }
}

Operation *mlir::edsc::makeLinalgGenericOp(
    ArrayRef<IterType> iteratorTypes, ArrayRef<StructuredIndexed> inputs,
    ArrayRef<StructuredIndexed> outputs,
    function_ref<void(ArrayRef<BlockArgument *>)> regionBuilder,
    ArrayRef<Value *> otherValues, ArrayRef<Attribute> otherAttributes) {
  auto &builder = edsc::ScopedContext::getBuilder();
  auto *ctx = builder.getContext();
  unsigned nInputs = inputs.size();
  unsigned nOutputs = outputs.size();
  unsigned maxPos = 0;
  getMaxDimIndex(inputs, maxPos);
  getMaxDimIndex(outputs, maxPos);
  // maxPos is 0 indexed, need to turn this into a count (i.e. +1)
  unsigned nDims = maxPos + 1;

  SmallVector<AffineMap, 4> maps;
  maps.reserve(nInputs + nOutputs);
  for (auto in : inputs)
    maps.push_back(
        AffineMap::get(/*dimCount=*/nDims, /*symbolCount=*/0, in.getExprs()));
  for (auto out : outputs)
    maps.push_back(
        AffineMap::get(/*dimCount=*/nDims, /*symbolCount=*/0, out.getExprs()));

  unsigned nViews = nInputs + nOutputs;
  SmallVector<Value *, 4> values;
  values.reserve(nViews);
  values.append(inputs.begin(), inputs.end());
  values.append(outputs.begin(), outputs.end());

  auto iteratorStrTypes = functional::map(toString, iteratorTypes);
  // clang-format off
  auto *op =
      edsc::ScopedContext::getBuilder()
          .create<linalg::GenericOp>(
              edsc::ScopedContext::getLocation(),
              values,
              IntegerAttr::get(IntegerType::get(64, ctx), nInputs),
              IntegerAttr::get(IntegerType::get(64, ctx), nOutputs),
              builder.getAffineMapArrayAttr(maps),
              builder.getStrArrayAttr(iteratorStrTypes),
              StringAttr() /*doc*/,
              FlatSymbolRefAttr() /*fun*/,
              StringAttr() /*library_call*/
              /* TODO: other attributes in op */
              )
          .getOperation();
  // clang-format on

  using namespace edsc;
  SmallVector<Type, 4> blockTypes;
  blockTypes.reserve(values.size());
  for (auto it : llvm::enumerate(values))
    blockTypes.push_back((it.index() < nViews)
                             ? getElementTypeOrSelf(it.value())
                             : it.value()->getType());

  assert(op->getRegions().front().empty());
  op->getRegions().front().push_front(new Block);
  OpBuilder bb(op->getRegions().front());
  ScopedContext scope(bb, op->getLoc());
  BlockHandle b;
  auto handles = makeValueHandles(blockTypes);
  BlockBuilder(&b, makeHandlePointers(MutableArrayRef<ValueHandle>(handles)))(
      [&] { regionBuilder(b.getBlock()->getArguments()); });
  return op;
}

void mlir::edsc::ops::macRegionBuilder(ArrayRef<BlockArgument *> args) {
  using edsc::op::operator+;
  using edsc::op::operator*;
  assert(args.size() == 3 && "expected 3 block arguments");
  ValueHandle a(args[0]), b(args[1]), c(args[2]);
  linalg_yield((c + a * b).getValue());
}

Operation *mlir::edsc::ops::linalg_pointwise(UnaryPointwiseOpBuilder unaryOp,
                                             StructuredIndexed I,
                                             StructuredIndexed O) {
  SmallVector<edsc::IterType, 4> iterTypes(O.getExprs().size(),
                                           edsc::IterType::Parallel);
  auto fun = [&unaryOp](ArrayRef<BlockArgument *> args) {
    assert(args.size() == 2 && "expected 2 block arguments");
    ValueHandle a(args[0]);
    linalg_yield(unaryOp(a));
  };
  return makeLinalgGenericOp(iterTypes, {I}, {O}, fun);
}

Operation *mlir::edsc::ops::linalg_pointwise_tanh(StructuredIndexed I,
                                                  StructuredIndexed O) {
  ;
  using edsc::intrinsics::tanh;
  UnaryPointwiseOpBuilder unOp(
      [](ValueHandle a) -> Value * { return tanh(a); });
  return linalg_pointwise(unOp, I, O);
}

/// Binary pointwise operation (with broadcast) entry point.
Operation *mlir::edsc::ops::linalg_pointwise(BinaryPointwiseOpBuilder binaryOp,
                                             StructuredIndexed I1,
                                             StructuredIndexed I2,
                                             StructuredIndexed O) {
  SmallVector<edsc::IterType, 4> iterTypes(O.getExprs().size(),
                                           edsc::IterType::Parallel);
  auto fun = [&binaryOp](ArrayRef<BlockArgument *> args) {
    assert(args.size() == 3 && "expected 3 block arguments");
    ValueHandle a(args[0]), b(args[1]);
    linalg_yield(binaryOp(a, b));
  };
  return makeLinalgGenericOp(iterTypes, {I1, I2}, {O}, fun);
}

Operation *mlir::edsc::ops::linalg_pointwise_add(StructuredIndexed I1,
                                                 StructuredIndexed I2,
                                                 StructuredIndexed O) {
  using edsc::op::operator+;
  BinaryPointwiseOpBuilder binOp(
      [](ValueHandle a, ValueHandle b) -> Value * { return a + b; });
  return linalg_pointwise(binOp, I1, I2, O);
}

Operation *mlir::edsc::ops::linalg_pointwise_max(StructuredIndexed I1,
                                                 StructuredIndexed I2,
                                                 StructuredIndexed O) {
  BinaryPointwiseOpBuilder binOp([](ValueHandle a, ValueHandle b) -> Value * {
    using edsc::intrinsics::select;
    using edsc::op::operator>;
    return select(a > b, a, b).getValue();
  });
  return linalg_pointwise(binOp, I1, I2, O);
}

Operation *mlir::edsc::ops::linalg_matmul(ValueHandle vA, ValueHandle vB,
                                          ValueHandle vC) {
  // clang-format off
  AffineExpr m, n, k;
  bindDims(ScopedContext::getContext(), m, n, k);
  StructuredIndexed A(vA), B(vB), C(vC);
  return makeLinalgGenericOp(
    {IterType::Parallel, IterType::Parallel, IterType::Reduction},
    {A({m, k}), B({k, n})},
    {C({m, n})},
    macRegionBuilder);
  // clang-format on
}

Operation *mlir::edsc::ops::linalg_conv_nhwc(ValueHandle vI, ValueHandle vW,
                                             ValueHandle vO,
                                             ArrayRef<int> strides,
                                             ArrayRef<int> dilations) {
  MLIRContext *ctx = ScopedContext::getContext();
  // TODO(ntv) some template magic to make everything rank-polymorphic.
  assert((dilations.empty() || dilations.size() == 2) && "only 2-D conv atm");
  assert((strides.empty() || strides.size() == 2) && "only 2-D conv atm");

  // Some short names.
  auto par = IterType::Parallel;
  auto red = IterType::Reduction;
  auto s = strides;
  auto d = dilations;

  AffineExpr b, f, h, w, kh, kw, c;
  bindDims(ctx, b, f, h, w, kh, kw, c);
  unsigned numDims = c.cast<AffineDimExpr>().getPosition() + 1;
  StructuredIndexed I(vI), W(vW), O(vO);
  // clang-format off
  return makeLinalgGenericOp(
    {par, par, par, par, red, red, red}, {
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, f})}, {
      O({b, h, w, f})},
    macRegionBuilder);
  // clang-format on
}

Operation *mlir::edsc::ops::linalg_dilated_conv_nhwc(
    ValueHandle vI, ValueHandle vW, ValueHandle vO, int depth_multiplier,
    ArrayRef<int> strides, ArrayRef<int> dilations) {
  MLIRContext *ctx = ScopedContext::getContext();
  // TODO(ntv) some template magic to make everything rank-polymorphic.
  assert((dilations.empty() || dilations.size() == 2) && "only 2-D conv atm");
  assert((strides.empty() || strides.size() == 2) && "only 2-D conv atm");

  // Some short names.
  auto par = IterType::Parallel;
  auto red = IterType::Reduction;
  auto s = strides;
  auto d = dilations;

  // clang-format off
  AffineExpr b, dm, c, h, w, kh, kw;
  bindDims(ctx, b, dm, c, h, w, kh, kw);
  unsigned numDims = kw.cast<AffineDimExpr>().getPosition() + 1;
  StructuredIndexed I(vI), W(vW), O(vO);
  return makeLinalgGenericOp(
    {par, par, par, par, par, red, red}, {
      I({b,
         // Roundtrip to flattened form to serve as canonicalization and ensure
         // consistent ordering of subexpressions.
         simplifyAffineExpr(s[0] * h + d[0] * kh, numDims, 0),
         simplifyAffineExpr(s[1] * w + d[1] * kw, numDims, 0),
         c}),
      W({kh, kw, c, dm})}, {
      O({b, h, w, simplifyAffineExpr(c * depth_multiplier + dm, numDims, 0)})},
    macRegionBuilder);
  // clang-format on
}
