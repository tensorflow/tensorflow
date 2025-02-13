/* Copyright 2025 The OpenXLA Authors.

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

#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/Types.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"

namespace mlir::triton::xla {

mlir::Type TiledTensorType::parse(mlir::AsmParser &parser) {
  mlir::SmallVector<int64_t, 4> shape;
  mlir::Type type;
  if (parser.parseLess() ||
      parser.parseDimensionList(shape, /*allowDynamic=*/false) ||
      parser.parseType(type) || parser.parseGreater()) {
    return {};
  }
  return TiledTensorType::get(parser.getContext(), shape, type);
}

void TiledTensorType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << "x" << getElementType() << ">";
}

}  // namespace mlir::triton::xla
