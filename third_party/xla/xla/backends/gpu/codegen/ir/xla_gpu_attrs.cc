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
#include "xla/backends/gpu/codegen/ir/xla_gpu_ops.h"
#include "xla/codegen/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"

namespace xla {
namespace gpu {

using mlir::AsmPrinter;

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
  context->getOrLoadDialect<xla::XlaDialect>();
  return LayoutAttr::get(context, MemorySpaceAttr::get(context, *memspace),
                         IndexingMapAttr::get(context, *indexing_map));
}

void LayoutAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<\"" << stringifyMemorySpace(getMemorySpace().getValue())
          << "\", \"" << ToString(getThreadMap().getIndexingMap()) << "\">";
}

}  // namespace gpu
}  // namespace xla
