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

#include "xla/codegen/xtile/ir/xtile_ops.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"

#define GET_OP_CLASSES
#include "xla/codegen/xtile/ir/xtile_ops.cc.inc"

namespace xla::xtile {

// This is lifted from the func::FuncOp builder, modified to make the tile
// index implicit.
void EntryFuncOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                        mlir::StringRef name,
                        mlir::ArrayRef<mlir::Type> memref_arg_types,
                        mlir::ArrayRef<mlir::NamedAttribute> attrs,
                        mlir::ArrayRef<mlir::DictionaryAttr> memref_arg_attrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  mlir::SmallVector<mlir::Type> arg_types(memref_arg_types.begin(),
                                          memref_arg_types.end());
  // Append the tile id index type.
  arg_types.push_back(builder.getIndexType());
  mlir::FunctionType function_type = builder.getFunctionType(arg_types,
                                                             /*results=*/{});
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     mlir::TypeAttr::get(function_type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (memref_arg_attrs.empty()) {
    return;
  }

  assert(memref_arg_types.size() == memref_arg_attrs.size());
  // As the arg attrs passed relate to the memref arg types we need to also
  // append a tile id attr.
  llvm::SmallVector<mlir::DictionaryAttr> arg_attrs_with_tile_id(
      memref_arg_attrs.begin(), memref_arg_attrs.end());
  arg_attrs_with_tile_id.push_back(
      mlir::DictionaryAttr::get(builder.getContext(), {}));
  mlir::call_interface_impl::addArgAndResultAttrs(
      builder, state, arg_attrs_with_tile_id, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

mlir::ParseResult EntryFuncOp::parse(mlir::OpAsmParser& parser,
                                     mlir::OperationState& result) {
  auto buildFuncType =
      [](mlir::Builder& builder, mlir::ArrayRef<mlir::Type> argTypes,
         mlir::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string&) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void EntryFuncOp::print(mlir::OpAsmPrinter& printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

mlir::LogicalResult EntryFuncOp::verify() {
  if (!getResultTypes().empty()) {
    return emitOpError() << "entry function should not have any return values";
  }

  if (getArgumentTypes().empty()) {
    return emitOpError()
           << "entry function must have at least the workgroup id";
  }

  // Deciphering the exact user error is non-trivial as they may have the
  // arguments in the wrong order or the incorrect number of workgroup ids etc,
  // so we just give a generic error message.
  constexpr absl::string_view argument_error =
      "entry function arguments should be of the form (arg: memref..., "
      "tile_id: index)";

  for (mlir::Type arg_types : getArgumentTypes().drop_back()) {
    if (!mlir::isa<mlir::MemRefType>(arg_types)) {
      return emitOpError() << argument_error;
    }
  }

  if (!mlir::isa<mlir::IndexType>(getArgumentTypes().back())) {
    return emitOpError() << argument_error;
  }

  return mlir::success();
}

llvm::SmallDenseSet<unsigned> ExtractTileOp::getReducedDimensions() {
  std::optional<llvm::SmallDenseSet<unsigned>> mask =
      mlir::computeRankReductionMask(getFullTileShape(), getType().getShape());
  // This should have already been verified.
  CHECK(mask.has_value());
  return *mask;
}

// This is the function ODS expects you to implement
mlir::LogicalResult ExtractTileOp::verify() {
  mlir::MemRefType source_type = getSource().getType();
  int64_t source_rank = source_type.getRank();
  mlir::Type source_element_type = source_type.getElementType();

  if (getFullTileShape().size() != source_rank) {
    return emitOpError() << "full tile shape size: "
                         << getFullTileShape().size()
                         << " does not match rank of source: " << source_rank;
  }

  size_t offset_count = getOffsets().size();
  if (offset_count != source_rank) {
    return emitOpError() << "expected " << source_rank
                         << " offset operands, got " << offset_count;
  }

  mlir::RankedTensorType result_type = getType();
  if (!mlir::computeRankReductionMask(getFullTileShape(),
                                      result_type.getShape())) {
    return emitOpError() << "full tile shape: [" << getFullTileShape()
                         << "] does not reduce to result shape: ["
                         << result_type.getShape() << "]";
  }

  if (result_type.getElementType() != source_element_type) {
    return emitOpError() << "result element type: "
                         << result_type.getElementType()
                         << " does not match element type of source: "
                         << source_element_type;
  }

  return mlir::success();
}

llvm::SmallDenseSet<unsigned> InsertTileOp::getReducedDimensions() {
  std::optional<llvm::SmallDenseSet<unsigned>> mask =
      mlir::computeRankReductionMask(getFullTileShape(),
                                     getSource().getType().getShape());
  // This should have already been verified.
  CHECK(mask.has_value());
  return *mask;
}

mlir::LogicalResult InsertTileOp::verify() {
  mlir::MemRefType destination_type = getDestination().getType();
  int64_t destination_rank = destination_type.getRank();

  if (getFullTileShape().size() != destination_rank) {
    return emitOpError() << "full tile shape size: "
                         << getFullTileShape().size()
                         << " does not match rank of destination: "
                         << destination_rank;
  }

  size_t offset_count = getOffsets().size();
  if (offset_count != destination_rank) {
    return emitOpError() << "expected " << destination_rank
                         << " offset operands, got " << offset_count;
  }

  mlir::RankedTensorType source_type = getSource().getType();
  if (!mlir::computeRankReductionMask(getFullTileShape(),
                                      source_type.getShape())) {
    return emitOpError() << "full tile shape: [" << getFullTileShape()
                         << "] does not reduce to source shape: ["
                         << source_type.getShape() << "]";
  }

  mlir::Type destination_element_type = destination_type.getElementType();
  mlir::Type source_element_type = source_type.getElementType();
  if (destination_element_type != source_element_type) {
    return emitOpError() << "destination element type: "
                         << destination_element_type
                         << " does not match element type of source: "
                         << source_element_type;
  }

  return mlir::success();
}

}  // namespace xla::xtile
