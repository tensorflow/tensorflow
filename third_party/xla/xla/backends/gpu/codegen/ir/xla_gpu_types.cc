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

#include "mlir/IR/Attributes.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/IR/Dialect.h"  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/ir/xla_gpu_ops.h"
#include "xla/hlo/analysis/indexing_map.h"  // IWYU pragma: keep

namespace xla {
namespace gpu {

mlir::Type IndexedVectorType::parse(mlir::AsmParser& parser) {
  mlir::SmallVector<int64_t, 4> shape;
  mlir::Type type;
  IndexingMapAttr indexing_map_attr;
  if (parser.parseLess() ||
      parser.parseDimensionList(shape, /*allowDynamic=*/false) ||
      parser.parseType(type) || parser.parseComma() ||
      parser.parseAttribute(indexing_map_attr) || parser.parseGreater()) {
    return {};
  }
  return IndexedVectorType::get(parser.getContext(), shape, type,
                                indexing_map_attr);
}

void IndexedVectorType::print(mlir::AsmPrinter& printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << "x" << getElementType() << ", " << getIndexingMapAttr() << ">";
}

}  // namespace gpu
}  // namespace xla
