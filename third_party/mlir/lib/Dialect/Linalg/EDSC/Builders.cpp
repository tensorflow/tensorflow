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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/Functional.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

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
    decltype(defaultRegionBuilder) regionBuilder, ArrayRef<Value *> otherValues,
    ArrayRef<Attribute> otherAttributes) {
  auto &builder = edsc::ScopedContext::getBuilder();
  auto *ctx = builder.getContext();
  unsigned nInputs = inputs.size();
  unsigned nOutputs = outputs.size();
  unsigned rank = 0;
  getMaxDimIndex(inputs, rank);
  getMaxDimIndex(outputs, rank);

  SmallVector<AffineMap, 4> maps;
  maps.reserve(nInputs + nOutputs);
  for (auto in : inputs)
    maps.push_back(
        AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, in.getExprs()));
  for (auto out : outputs)
    maps.push_back(
        AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, out.getExprs()));

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

using linalg_yield = OperationBuilder<linalg::YieldOp>;

Operation *mlir::edsc::linalg_matmul(ValueHandle vA, ValueHandle vB,
                                     ValueHandle vC) {
  // clang-format off
  AffineExpr m, n, k;
  bindDims(ScopedContext::getContext(), m, n, k);
  StructuredIndexed A(vA), B(vB), C(vC);
  return makeLinalgGenericOp(
    {IterType::Parallel, IterType::Parallel, IterType::Reduction},
    {A({m, n}), B({k, n})},
    {C({m, n})},
    [](ArrayRef<BlockArgument *> args) {
      using edsc::op::operator*;
      using edsc::op::operator+;
      ValueHandle a(args[0]), b(args[1]), c(args[2]);
      linalg_yield((c + a * b).getValue());
  });
  // clang-format on
}
