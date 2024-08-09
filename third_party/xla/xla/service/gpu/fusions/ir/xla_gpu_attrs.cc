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

#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/model/indexing_map.h"

#define GET_ATTRDEF_LIST
#define GET_ATTRDEF_CLASSES
#include "xla/service/gpu/fusions/ir/xla_gpu_attrs.h.inc"

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

ParseResult ParseInterval(AsmParser& parser, Interval& interval) {
  // ParseResult converts to `true` if parsing failed.
  return failure(parser.parseLSquare() || parser.parseInteger(interval.lower) ||
                 parser.parseComma() || parser.parseInteger(interval.upper) ||
                 parser.parseRSquare());
}

void PrintDimVars(AsmPrinter& p, ArrayRef<DimVar> dim_vars) {
  int index = 0;
  llvm::interleaveComma(dim_vars, p, [&](const DimVar& dim_var) {
    p << "d" << index++ << " in " << dim_var.bounds;
  });
}

ParseResult ParseDimVars(AsmParser& parser, ArrayRef<std::string> dim_names,
                         SmallVector<DimVar>& dim_vars) {
  dim_vars.reserve(dim_names.size());
  for (const auto& [index, dim_name] : llvm::enumerate(dim_names)) {
    if (parser.parseKeyword(dim_name) || parser.parseKeyword("in") ||
        ParseInterval(parser, dim_vars.emplace_back().bounds)) {
      return failure();
    }
    if (index < dim_names.size() - 1 && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

void PrintRangeVars(AsmPrinter& p, ArrayRef<RangeVar> range_vars) {
  int index = 0;
  llvm::interleaveComma(range_vars, p, [&](const RangeVar& range_var) {
    p << "s" << index++ << " in " << range_var.range;
  });
}

ParseResult ParseRangeVars(AsmParser& parser,
                           ArrayRef<std::string> range_symbol_names,
                           SmallVector<RangeVar>& range_vars) {
  range_vars.reserve(range_symbol_names.size());
  for (const auto& [index, range_symbol_name] :
       llvm::enumerate(range_symbol_names)) {
    if (parser.parseKeyword(range_symbol_name) || parser.parseKeyword("in") ||
        ParseInterval(parser, range_vars.emplace_back().range)) {
      return failure();
    }
    if (index < range_symbol_names.size() - 1 && parser.parseComma()) {
      return failure();
    }
  }
  return success();
}

void PrintConstraints(AsmPrinter& p,
                      ArrayRef<std::pair<AffineExpr, Interval>> constraints) {
  llvm::interleaveComma(constraints, p, [&](const auto& constraint) {
    p << constraint.first << " in " << constraint.second;
  });
}

ParseResult ParseConstraints(
    AsmParser& parser,
    ArrayRef<std::pair<llvm::StringRef, AffineExpr>> symbolSet,
    SmallVector<std::pair<AffineExpr, Interval>>& constraints) {
  // In order for there to be any constraints, there must be at least 1 symbol
  // or dimension meaning there will be commas for as long as there are
  // constraints left.
  while (succeeded(parser.parseOptionalComma())) {
    auto& constraint = constraints.emplace_back();
    if (parser.parseAffineExpr(symbolSet, constraint.first) ||
        parser.parseKeyword("in") || ParseInterval(parser, constraint.second)) {
      return failure();
    }
  }
  return success();
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
  if (map.getNumDims() + map.getNumSymbols() > 0) {
    if (parser.parseComma() || parser.parseKeyword("domain") ||
        parser.parseColon()) {
      return {};
    }
  }

  SmallVector<DimVar> dim_vars;
  if (map.getNumDims() > 0) {
    if (ParseDimVars(parser, dim_strings, dim_vars)) {
      return {};
    }
  }

  SmallVector<RangeVar> range_vars;
  if (map.getNumSymbols() > 0) {
    if (!dim_vars.empty() && parser.parseComma()) {
      return {};
    }
    if (ParseRangeVars(parser, symbol_strings, range_vars)) {
      return {};
    }
  }

  SmallVector<std::pair<AffineExpr, Interval>> constraints;
  if (ParseConstraints(parser, symbolSet, constraints) ||
      parser.parseGreater()) {
    return {};
  }
  return IndexingMapAttr::get(parser.getContext(), map, dim_vars, range_vars,
                              constraints);
}

void IndexingMapAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getMap());
  if (getDimVars().size() + getRangeVars().size() + getConstraints().size() >
      0) {
    printer << ", domain: ";
  }
  PrintDimVars(printer, getDimVars());
  if (!getDimVars().empty() &&
      getRangeVars().size() + getConstraints().size() > 0) {
    printer << ", ";
  }
  PrintRangeVars(printer, getRangeVars());
  if (!getRangeVars().empty() && !getConstraints().empty()) {
    printer << ", ";
  }
  PrintConstraints(printer, getConstraints());
  printer << ">";
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
    mlir::AffineMap map, ArrayRef<DimVar> dim_vars,
    ArrayRef<RangeVar> range_vars,
    ArrayRef<std::pair<AffineExpr, Interval>> constraints) {
  if (map.getNumDims() != dim_vars.size()) {
    return emitError()
           << "dim size must match the number of dimensions in the affine map";
  }
  if (map.getNumSymbols() != range_vars.size()) {
    return emitError()
           << "range size must match the number of symbols in the affine map";
  }
  return mlir::success();
}

IndexingMap IndexingMapAttr::getIndexingMap() {
  return IndexingMap(getMap(), getDimVars(), getRangeVars(), /*rt_vars=*/{},
                     getConstraints());
}

}  // namespace gpu
}  // namespace xla
