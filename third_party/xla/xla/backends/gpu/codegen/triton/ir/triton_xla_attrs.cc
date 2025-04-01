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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::xla {

static mlir::ParseResult parseI64ArrayAttr(mlir::AsmParser& parser,
                                           mlir::DenseI64ArrayAttr& array) {
  array = mlir::dyn_cast_or_null<mlir::DenseI64ArrayAttr>(
      mlir::DenseI64ArrayAttr::parse(parser, mlir::Type{}));
  if (!array) return mlir::failure();
  return mlir::success();
}

Attribute TmaDescriptorAttr::parse(mlir::AsmParser& parser, mlir::Type) {
  int element_byte_size;
  DenseI64ArrayAttr global_shape, block_shape;

  if (parser.parseLess() || parser.parseKeyword("global_shape") ||
      parser.parseEqual() || parseI64ArrayAttr(parser, global_shape) ||
      parser.parseComma() || parser.parseKeyword("block_shape") ||
      parser.parseEqual() || parseI64ArrayAttr(parser, block_shape) ||
      parser.parseComma() || parser.parseKeyword("element_byte_size") ||
      parser.parseEqual() || parser.parseInteger(element_byte_size) ||
      parser.parseGreater()) {
    return {};
  }
  return TmaDescriptorAttr::get(parser.getContext(), global_shape.asArrayRef(),
                                block_shape.asArrayRef(), element_byte_size);
}

void TmaDescriptorAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<global_shape = [";
  llvm::interleaveComma(getGlobalShape(), printer);
  printer << "], block_shape = [";
  llvm::interleaveComma(getBlockShape(), printer);
  printer << "], element_byte_size = " << getElementByteSize() << ">";
}

}  // namespace mlir::triton::xla
