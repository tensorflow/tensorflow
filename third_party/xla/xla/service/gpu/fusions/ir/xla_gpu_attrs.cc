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
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
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
#include "xla/service/gpu/fusions/ir/xla_gpu_ops.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

using llvm::ParseResult;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::ArrayRef;
using mlir::AsmParser;
using mlir::AsmPrinter;
using mlir::failure;
using mlir::success;

constexpr llvm::StringRef kIsSimplifiedKeyword = "is_simplified";

ParseResult ParseInterval(AsmParser& parser, Interval& interval) {
  // ParseResult converts to `true` if parsing failed.
  return failure(parser.parseLSquare() || parser.parseInteger(interval.lower) ||
                 parser.parseComma() || parser.parseInteger(interval.upper) ||
                 parser.parseRSquare());
}

ParseResult parseBool(AsmParser& parser, bool* result) {
  if (succeeded(parser.parseOptionalKeyword("true"))) {
    *result = true;
    return success();
  }
  if (succeeded(parser.parseOptionalKeyword("false"))) {
    *result = false;
    return success();
  }
  return failure();
}

void PrintDimVars(AsmPrinter& p, ArrayRef<DimVar> dim_vars) {
  for (const auto [index, dim_var] : llvm::enumerate(dim_vars)) {
    p << "d" << index << " in " << dim_var.bounds << ", ";
  }
}

ParseResult ParseDimVars(AsmParser& parser, ArrayRef<std::string> dim_names,
                         SmallVector<DimVar>& dim_vars) {
  dim_vars.reserve(dim_names.size());
  for (const auto& [index, dim_name] : llvm::enumerate(dim_names)) {
    if (parser.parseKeyword(dim_name) || parser.parseKeyword("in") ||
        ParseInterval(parser, dim_vars.emplace_back().bounds) ||
        parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

void PrintRangeVars(AsmPrinter& p, ArrayRef<RangeVar> range_vars) {
  for (const auto [index, range_var] : llvm::enumerate(range_vars)) {
    p << "s" << index << " in " << range_var.range << ", ";
  }
}

ParseResult ParseRangeVars(AsmParser& parser,
                           ArrayRef<std::string> range_symbol_names,
                           SmallVector<RangeVar>& range_vars) {
  range_vars.reserve(range_symbol_names.size());
  for (const auto& [index, range_symbol_name] :
       llvm::enumerate(range_symbol_names)) {
    if (parser.parseKeyword(range_symbol_name) || parser.parseKeyword("in") ||
        ParseInterval(parser, range_vars.emplace_back().range) ||
        parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

void PrintConstraints(AsmPrinter& p,
                      ArrayRef<std::pair<AffineExpr, Interval>> constraints) {
  for (const auto& [expr, interval] : constraints) {
    p << expr << " in " << interval << ", ";
  }
}

mlir::Attribute IndexingMapAttr::parse(mlir::AsmParser& parser, mlir::Type) {
  mlir::AffineMap map;
  if (parser.parseLess() || parser.parseAffineMap(map)) {
    return {};
  }

  // Store real strings to back up StringRef throughout ParseConstraints.
  SmallVector<std::string> dim_strings(map.getNumDims());
  SmallVector<std::string> symbol_strings(map.getNumSymbols());
  SmallVector<std::pair<llvm::StringRef, AffineExpr>> symbolSet;
  symbolSet.reserve(map.getNumDims() + map.getNumSymbols());
  for (int i = 0; i < map.getNumDims(); ++i) {
    dim_strings[i] = absl::StrFormat("d%d", i);
    symbolSet.push_back(
        {dim_strings[i], mlir::getAffineDimExpr(i, parser.getContext())});
  }
  for (int i = 0; i < map.getNumSymbols(); ++i) {
    symbol_strings[i] = absl::StrFormat("s%d", i);
    symbolSet.push_back(
        {symbol_strings[i], mlir::getAffineSymbolExpr(i, parser.getContext())});
  }
  if (map.getNumDims() + map.getNumSymbols() == 0) {
    if (parser.parseGreater()) return {};
    return IndexingMapAttr::get(parser.getContext(), map, /*dim_vars=*/{},
                                /*range_vars=*/{},
                                /*constraints=*/{}, /*is_simplified=*/true);
  }
  if (parser.parseComma() || parser.parseKeyword("domain") ||
      parser.parseColon()) {
    return {};
  }

  SmallVector<DimVar> dim_vars;
  if (ParseDimVars(parser, dim_strings, dim_vars)) {
    return {};
  }
  SmallVector<RangeVar> range_vars;
  if (ParseRangeVars(parser, symbol_strings, range_vars)) {
    return {};
  }

  SmallVector<std::pair<AffineExpr, Interval>> constraints;
  while (failed(parser.parseOptionalKeyword(kIsSimplifiedKeyword))) {
    auto& constraint = constraints.emplace_back();
    if (parser.parseAffineExpr(symbolSet, constraint.first) ||
        parser.parseKeyword("in") || ParseInterval(parser, constraint.second) ||
        parser.parseComma()) {
      return {};
    }
    constraints.push_back(constraint);
  }

  bool is_simplified = false;
  if (parser.parseColon() || parseBool(parser, &is_simplified) ||
      parser.parseGreater()) {
    return {};
  }
  return IndexingMapAttr::get(parser.getContext(), map, dim_vars, range_vars,
                              constraints, is_simplified);
}

void IndexingMapAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<" << getIndexingMap().ToString() << ">";
}

IndexingMapAttr IndexingMapAttr::get(mlir::MLIRContext* context,
                                     const IndexingMap& indexing_map) {
  llvm::SmallVector<std::pair<AffineExpr, Interval>> constraints;
  for (auto& constraint : indexing_map.GetConstraints()) {
    constraints.push_back({constraint.first, constraint.second});
  }
  return get(context, indexing_map.GetAffineMap(), indexing_map.GetDimVars(),
             indexing_map.GetRangeVars(), constraints,
             indexing_map.IsSimplified());
}

mlir::LogicalResult IndexingMapAttr::verify(
    mlir::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::AffineMap map, ArrayRef<DimVar> dim_vars,
    ArrayRef<RangeVar> range_vars,
    ArrayRef<std::pair<AffineExpr, Interval>> constraints, bool is_simplified) {
  if (map.getNumDims() != dim_vars.size()) {
    return emitError() << "dim size must match the number of dimensions in "
                          "the affine map";
  }
  if (map.getNumSymbols() != range_vars.size()) {
    return emitError()
           << "range size must match the number of symbols in the affine map";
  }
  return mlir::success();
}

IndexingMap IndexingMapAttr::getIndexingMap() const {
  return IndexingMap(getMap(), getDimVars(), getRangeVars(), /*rt_vars=*/{},
                     getConstraints(), getIsSimplified());
}

int64_t IndexingMapAttr::getNumResults() const {
  return getMap().getNumResults();
}

}  // namespace gpu
}  // namespace xla
