/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <optional>
#include <sstream>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {

using llvm::ParseResult;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::ArrayRef;
using mlir::AsmParser;
using mlir::AsmPrinter;
using mlir::failure;
using mlir::success;

mlir::Attribute IndexingMapAttr::parse(mlir::AsmParser& parser, mlir::Type) {
  if (parser.parseLess()) {
    return {};
  }
  auto indexing_map = parseChainOfStringsAsIndexingMap(parser);
  if (!indexing_map.has_value() || parser.parseGreater()) {
    return {};
  }
  return IndexingMapAttr::get(parser.getContext(), *indexing_map);
}

void IndexingMapAttr::print(mlir::AsmPrinter& printer) const {
  gpu::SymbolicExprContext symbolic_expr_context(getContext());
  printer << "<\"" << ToString(getIndexingMap(&symbolic_expr_context)) << "\">";
}

IndexingMapAttr IndexingMapAttr::get(mlir::MLIRContext* context,
                                     const IndexingMap& indexing_map) {
  llvm::SmallVector<std::pair<AffineExpr, Interval>> constraints;
  for (auto& constraint : indexing_map.GetConstraints()) {
    constraints.push_back({constraint.first, constraint.second});
  }
  return get(context, indexing_map.GetAffineMap(), indexing_map.GetDimVars(),
             indexing_map.GetRangeVars(), constraints);
}

mlir::LogicalResult IndexingMapAttr::verify(
    mlir::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::AffineMap map, ArrayRef<IndexingMap::Variable> dim_vars,
    ArrayRef<IndexingMap::Variable> range_vars,
    ArrayRef<std::pair<AffineExpr, Interval>> constraints) {
  // Create the context locally from the AffineMap's context.
  gpu::SymbolicExprContext symbolic_expr_context(map.getContext());
  auto indexing_map =
      IndexingMap(map, &symbolic_expr_context, dim_vars, range_vars,
                  /*rt_vars=*/{}, constraints);
  std::stringstream ss;
  if (!indexing_map.Verify(ss)) {
    return emitError() << ss.str();
  }

  return success();
}

IndexingMap IndexingMapAttr::getIndexingMap(
    gpu::SymbolicExprContext* symbolic_expr_context) const {
  return IndexingMap(getMap(), symbolic_expr_context, getDimVars(),
                     getRangeVars(), /*rt_vars=*/{}, getConstraints());
}

int64_t IndexingMapAttr::getNumResults() const {
  return getMap().getNumResults();
}

}  // namespace xla
