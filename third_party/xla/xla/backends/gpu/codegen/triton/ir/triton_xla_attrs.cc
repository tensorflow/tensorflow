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

ParseResult ParseOptionalSwizzleMode(mlir::AsmParser& parser,
                                     SwizzleModeAttr& swizzle_mode) {
  if (parser.parseOptionalComma()) {
    // If there is no comma, we don't have a swizzle mode, but it's still valid.
    swizzle_mode = nullptr;
    return mlir::success();
  }
  StringAttr swizzle_mode_str;
  if (parser.parseKeyword("swizzle_mode") || parser.parseEqual() ||
      parser.parseAttribute(swizzle_mode_str)) {
    return mlir::failure();
  }
  auto maybe_swizzle_mode = symbolizeSwizzleMode(swizzle_mode_str);
  if (!maybe_swizzle_mode.has_value()) {
    return mlir::failure();
  }
  swizzle_mode =
      SwizzleModeAttr::get(parser.getContext(), maybe_swizzle_mode.value());
  return mlir::success();
}

Attribute TmaDescriptorAttr::parse(mlir::AsmParser& parser, mlir::Type) {
  int element_byte_size;
  DenseI64ArrayAttr global_shape, tile_shape, tile_strides, layout;
  SwizzleModeAttr swizzle_mode = nullptr;

  if (parser.parseLess() || parser.parseKeyword("global_shape") ||
      parser.parseEqual() || parseI64ArrayAttr(parser, global_shape) ||
      parser.parseComma() || parser.parseKeyword("tile_shape") ||
      parser.parseEqual() || parseI64ArrayAttr(parser, tile_shape) ||
      parser.parseComma() || parser.parseKeyword("tile_strides") ||
      parser.parseEqual() || parseI64ArrayAttr(parser, tile_strides) ||
      parser.parseComma() || parser.parseKeyword("layout") ||
      parser.parseEqual() || parseI64ArrayAttr(parser, layout) ||
      parser.parseComma() || parser.parseKeyword("element_byte_size") ||
      parser.parseEqual() || parser.parseInteger(element_byte_size) ||
      ParseOptionalSwizzleMode(parser, swizzle_mode) || parser.parseGreater()) {
    return {};
  }

  return TmaDescriptorAttr::get(parser.getContext(), global_shape.asArrayRef(),
                                tile_shape.asArrayRef(),
                                tile_strides.asArrayRef(), layout.asArrayRef(),
                                element_byte_size, swizzle_mode);
}

void TmaDescriptorAttr::print(mlir::AsmPrinter& printer) const {
  printer << "<global_shape = [";
  llvm::interleaveComma(getGlobalShape(), printer);
  printer << "], tile_shape = [";
  llvm::interleaveComma(getTileShape(), printer);
  printer << "], tile_strides = [";
  llvm::interleaveComma(getTileStrides(), printer);
  printer << "], layout = [";
  llvm::interleaveComma(getLayout(), printer);
  printer << "], element_byte_size = " << getElementByteSize();
  if (getSwizzleMode()) {
    printer << ", swizzle_mode = \""
            << stringifySwizzleMode(getSwizzleMode().getValue()) << "\"";
  }
  printer << ">";
}

}  // namespace mlir::triton::xla
