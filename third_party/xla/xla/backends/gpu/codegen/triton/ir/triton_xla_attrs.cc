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

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"

namespace mlir::triton::xla {

static mlir::ParseResult parseI64ArrayAttr(mlir::AsmParser& parser,
                                           mlir::DenseI64ArrayAttr& array) {
  array = mlir::dyn_cast_or_null<mlir::DenseI64ArrayAttr>(
      mlir::DenseI64ArrayAttr::parse(parser, mlir::Type{}));
  if (!array) return mlir::failure();
  return mlir::success();
}

ParseResult ParseSwizzleMode(mlir::AsmParser& parser,
                             SwizzleMode& swizzle_mode) {
  StringAttr swizzle_mode_str;
  if (parser.parseKeyword("swizzle_mode") || parser.parseEqual() ||
      parser.parseAttribute(swizzle_mode_str)) {
    return mlir::failure();
  }
  auto maybe_swizzle_mode = symbolizeSwizzleMode(swizzle_mode_str);
  if (!maybe_swizzle_mode.has_value()) {
    return mlir::failure();
  }
  swizzle_mode = maybe_swizzle_mode.value();
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
      parser.parseEqual() || parser.parseInteger(element_byte_size)) {
    return {};
  }

  SwizzleMode swizzle_mode = SwizzleMode::kUnset;
  if (succeeded(parser.parseOptionalComma()) &&
      ParseSwizzleMode(parser, swizzle_mode)) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  return TmaDescriptorAttr::get(
      parser.getContext(), global_shape.asArrayRef(), block_shape.asArrayRef(),
      element_byte_size,
      SwizzleModeAttr::get(parser.getContext(), swizzle_mode));
}

void TmaDescriptorAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<global_shape = [";
  llvm::interleaveComma(getGlobalShape(), printer);
  printer << "], block_shape = [";
  llvm::interleaveComma(getBlockShape(), printer);
  printer << "], element_byte_size = " << getElementByteSize();
  if (getSwizzleMode().getValue() != SwizzleMode::kUnset) {
    printer << ", swizzle_mode = \""
            << stringifySwizzleMode(getSwizzleMode().getValue()) << "\"";
  }
  printer << ">";
}

}  // namespace mlir::triton::xla
