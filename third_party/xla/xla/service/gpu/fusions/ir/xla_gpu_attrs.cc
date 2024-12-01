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
#include <string>
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
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/service/gpu/fusions/ir/xla_gpu_ops.h"

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

// Parses a chain of string attributes into an indexing map.
// Example:
// "()[s0, s1] -> (1 + s0 + s1 mod 3 - s1, s0 mod 2),"
//   " domain: s0 in [-10, 10], s1 in [0, 2]"
// will be parsed as 3 StringAttrs, concatenated into a single string, and then
// parsed into an IndexingMap.
std::optional<IndexingMap> parseChainOfStringsAsIndexingMap(
    mlir::AsmParser& parser) {
  mlir::StringAttr indexing_map_attr;
  std::string indexing_map_str;
  while (parser.parseOptionalAttribute(indexing_map_attr).has_value()) {
    indexing_map_str.append(indexing_map_attr.getValue());
  }
  return ParseIndexingMap(indexing_map_str, parser.getContext());
}

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
  printer << "<\"" << ToString(getIndexingMap()) << "\">";
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
  auto indexing_map =
      IndexingMap(map, dim_vars, range_vars, /*rt_vars=*/{}, constraints);
  std::stringstream ss;
  if (!indexing_map.Verify(ss)) {
    return emitError() << ss.str();
  }
  return success();
}

IndexingMap IndexingMapAttr::getIndexingMap() const {
  return IndexingMap(getMap(), getDimVars(), getRangeVars(), /*rt_vars=*/{},
                     getConstraints());
}

int64_t IndexingMapAttr::getNumResults() const {
  return getMap().getNumResults();
}

mlir::Attribute LayoutAttr::parse(mlir::AsmParser& parser, mlir::Type) {
  mlir::StringAttr memory_space_str;
  if (parser.parseLess() || parser.parseAttribute(memory_space_str) ||
      parser.parseComma()) {
    return {};
  }
  std::optional<MemorySpace> memspace =
      symbolizeMemorySpace(memory_space_str.getValue());
  if (!memspace.has_value()) {
    return {};
  }
  std::optional<IndexingMap> indexing_map =
      parseChainOfStringsAsIndexingMap(parser);
  if (!indexing_map.has_value() || parser.parseGreater()) {
    return {};
  }
  auto* context = parser.getContext();
  return LayoutAttr::get(context, MemorySpaceAttr::get(context, *memspace),
                         IndexingMapAttr::get(context, *indexing_map));
}

void LayoutAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<\"" << stringifyMemorySpace(getMemorySpace().getValue())
          << "\", \"" << ToString(getThreadMap().getIndexingMap()) << "\">";
}

}  // namespace gpu
}  // namespace xla
