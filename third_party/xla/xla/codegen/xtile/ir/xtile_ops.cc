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
#include "xla/codegen/xtile/ir/xtile_interface_ops.cc.inc"
#include "xla/codegen/xtile/ir/xtile_ops.cc.inc"

namespace xla::xtile {

llvm::SmallDenseSet<unsigned> TiledBufferInterface::getReducedDimensions() {
  std::optional<llvm::SmallDenseSet<unsigned>> mask =
      mlir::computeRankReductionMask(getFullTileShape(),
                                     getTile().getType().getShape());
  // This should have already been verified.
  CHECK(mask.has_value());
  return *mask;
}

static mlir::LogicalResult VerifyBufferOp(TiledBufferInterface op) {
  mlir::MemRefType buffer_type = op.getBuffer().getType();
  int64_t buffer_rank = buffer_type.getRank();

  if (op.getFullTileShape().size() != buffer_rank) {
    return op.emitOpError()
           << "full tile shape size: " << op.getFullTileShape().size()
           << " does not match rank of buffer: " << buffer_rank;
  }

  size_t offset_count = op.getOffsets().size();
  if (offset_count != buffer_rank) {
    return op.emitOpError() << "expected " << buffer_rank
                            << " offset operands, got " << offset_count;
  }

  mlir::RankedTensorType tile_type = op.getTile().getType();
  if (!mlir::computeRankReductionMask(op.getFullTileShape(),
                                      tile_type.getShape())) {
    return op.emitOpError() << "full tile shape: [" << op.getFullTileShape()
                            << "] does not reduce to tile shape: ["
                            << tile_type.getShape() << "]";
  }

  mlir::Type buffer_element_type = buffer_type.getElementType();
  mlir::Type tile_element_type = tile_type.getElementType();
  if (buffer_element_type != tile_element_type) {
    return op.emitOpError()
           << "buffer element type: " << buffer_element_type
           << " does not match element type of tile: " << tile_element_type;
  }

  return mlir::success();
}

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

  // + 1 for the tile id.
  const int64_t num_opaque_args = getNumOpaqueArgs() + 1;
  for (mlir::Type arg_types : getArgumentTypes().drop_back(num_opaque_args)) {
    if (!mlir::isa<mlir::MemRefType>(arg_types)) {
      return emitOpError() << argument_error;
    }
  }

  if (!mlir::isa<mlir::IndexType>(getArgumentTypes().back())) {
    return emitOpError() << argument_error;
  }

  return mlir::success();
}

mlir::TypedValue<mlir::MemRefType> ExtractTileOp::getBuffer() {
  return getSource();
}

mlir::TypedValue<mlir::RankedTensorType> ExtractTileOp::getTile() {
  return getResult();
}

// This is the function ODS expects you to implement
mlir::LogicalResult ExtractTileOp::verify() { return VerifyBufferOp(*this); }

mlir::TypedValue<mlir::MemRefType> InsertTileOp::getBuffer() {
  return getDestination();
}

mlir::TypedValue<mlir::RankedTensorType> InsertTileOp::getTile() {
  return getSource();
}

mlir::LogicalResult InsertTileOp::verify() { return VerifyBufferOp(*this); }

mlir::LogicalResult ToScalarOp::inferReturnTypes(
    mlir::MLIRContext* context, ::std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
  if (operands.size() != 1) {
    return mlir::failure();
  }

  auto tensor_type =
      mlir::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
  if (!tensor_type) {
    return mlir::failure();
  }

  if (tensor_type.getRank() != 0) {
    return mlir::failure();
  }

  inferredReturnTypes.push_back(tensor_type.getElementType());
  return mlir::success();
}

mlir::OpFoldResult ToScalarOp::fold(FoldAdaptor adaptor) {
  if (auto to_tensor = getOperand().getDefiningOp<ToTensorOp>()) {
    // to_scalar(to_tensor(x)) -> x
    return to_tensor.getOperand();
  }

  return {};
}

mlir::LogicalResult ToTensorOp::inferReturnTypes(
    mlir::MLIRContext* context, ::std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
  if (operands.size() != 1) {
    return mlir::failure();
  }
  inferredReturnTypes.push_back(
      mlir::RankedTensorType::get({}, operands[0].getType()));
  return mlir::success();
}

mlir::OpFoldResult ToTensorOp::fold(FoldAdaptor adaptor) {
  if (auto to_scalar = getOperand().getDefiningOp<ToScalarOp>()) {
    // to_tensor(to_scalar(x)) -> x
    return to_scalar.getOperand();
  }

  return {};
}

}  // namespace xla::xtile
