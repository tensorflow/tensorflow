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

#include "mlir/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::edsc;

Operation *mlir::edsc::makeLinalgGenericOp(
    ArrayRef<AffineExpr> indices, ArrayRef<ArrayRef<AffineExpr>> mapExpressions,
    ArrayRef<Value *> inputViews, ArrayRef<Value *> outputViews,
    ArrayRef<StringRef> iteratorTypes,
    decltype(defaultRegionBuilder) regionBuilder) {
  auto &builder = edsc::ScopedContext::getBuilder();
  auto *ctx = builder.getContext();

  SmallVector<AffineMap, 4> maps;
  maps.reserve(mapExpressions.size());
  for (auto exprs : mapExpressions)
    maps.push_back(AffineMap::get(indices.size(), 0, exprs));

  SmallVector<Value *, 4> views;
  views.reserve(inputViews.size() + outputViews.size());
  views.append(inputViews.begin(), inputViews.end());
  views.append(outputViews.begin(), outputViews.end());

  auto *op =
      edsc::ScopedContext::getBuilder()
          .create<linalg::GenericOp>(
              edsc::ScopedContext::getLocation(), views,
              IntegerAttr::get(IntegerType::get(64, ctx), inputViews.size()),
              IntegerAttr::get(IntegerType::get(64, ctx), outputViews.size()),
              builder.getAffineMapArrayAttr(maps),
              builder.getStrArrayAttr(iteratorTypes), StringAttr() /*doc*/,
              FlatSymbolRefAttr() /*fun*/, StringAttr() /*library_call*/
              )
          .getOperation();

  using namespace edsc;
  SmallVector<Type, 4> blockTypes;
  blockTypes.reserve(views.size());
  for (auto *v : views)
    blockTypes.push_back(getElementTypeOrSelf(v));

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
