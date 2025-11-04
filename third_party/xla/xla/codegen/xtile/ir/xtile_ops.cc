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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/xtile/ir/xtile_attrs.h"

#define GET_OP_CLASSES
#include "xla/codegen/xtile/ir/xtile_interface_ops.cc.inc"
#include "xla/codegen/xtile/ir/xtile_ops.cc.inc"

namespace xla::xtile {

// Deciphering the exact user error is non-trivial as they may have the
// arguments in the wrong order or the incorrect number of workgroup ids etc,
// so we just give a generic error message.
constexpr absl::string_view argument_error =
    "entry function arguments should be of the form (arg: memref..., "
    "opaque: types..., tile_id: index)";

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

void EntryFuncOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                        mlir::StringRef name,
                        mlir::ArrayRef<mlir::Type> memref_arg_types,
                        mlir::ArrayRef<mlir::Type> opaque_arg_types,
                        TilingInfoAttr tile_info,
                        mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  mlir::SmallVector<mlir::Type> arg_types(memref_arg_types.begin(),
                                          memref_arg_types.end());
  // Append the tile id index type.
  arg_types.push_back(builder.getIndexType());

  state.addAttribute(getMemrefArgTypesAttrName(state.name),
                     builder.getTypeArrayAttr(memref_arg_types));
  state.addAttribute(getOpaqueArgTypesAttrName(state.name),
                     builder.getTypeArrayAttr(opaque_arg_types));
  if (tile_info) {
    state.addAttribute(getTileInfoAttrName(state.name), tile_info);
  }

  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

mlir::ParseResult EntryFuncOp::parse(mlir::OpAsmParser& parser,
                                     mlir::OperationState& result) {
  mlir::SmallVector<mlir::OpAsmParser::Argument> entry_args;
  auto symbol_attr_name = mlir::SymbolTable::getSymbolAttrName();
  auto memref_types_attr_name = getMemrefArgTypesAttrName(result.name);
  auto opaque_types_attr_name = getOpaqueArgTypesAttrName(result.name);
  auto tile_info_attr_name = getTileInfoAttrName(result.name);

  auto& builder = parser.getBuilder();

  mlir::StringAttr name_attr;
  if (parser.parseSymbolName(name_attr, symbol_attr_name, result.attributes)) {
    return mlir::failure();
  }

  mlir::SMLoc signature_loc = parser.getCurrentLocation();
  bool is_variadic = false;
  mlir::SmallVector<mlir::Type> result_types;
  mlir::SmallVector<mlir::DictionaryAttr> result_attrs;
  if (mlir::function_interface_impl::parseFunctionSignatureWithArguments(
          parser, false, entry_args, is_variadic, result_types, result_attrs)) {
    return mlir::failure();
  }

  if (!result_types.empty()) {
    return parser.emitError(signature_loc)
           << "entry function should not have any return values";
  }

  if (entry_args.empty() ||
      !mlir::isa<mlir::IndexType>(entry_args.back().type)) {
    return parser.emitError(signature_loc) << argument_error;
  }

  auto non_tile_id_args = llvm::drop_end(entry_args);
  auto arg_itr = non_tile_id_args.begin();

  mlir::SmallVector<mlir::Attribute> memref_arg_types;
  while (arg_itr != non_tile_id_args.end() &&
         mlir::isa<mlir::MemRefType>(arg_itr->type)) {
    memref_arg_types.push_back(mlir::TypeAttr::get(arg_itr->type));
    ++arg_itr;
  }
  result.addAttribute(memref_types_attr_name,
                      builder.getArrayAttr(memref_arg_types));

  mlir::SmallVector<mlir::Attribute> opaque_arg_types;
  for (; arg_itr != non_tile_id_args.end(); ++arg_itr) {
    opaque_arg_types.push_back(mlir::TypeAttr::get(arg_itr->type));
  }
  result.addAttribute(opaque_types_attr_name,
                      builder.getArrayAttr(opaque_arg_types));

  if (!parser.parseOptionalKeyword("tiling")) {
    TilingInfoAttr tile_info;
    if (parser.parseAttribute(tile_info)) {
      return mlir::failure();
    }
    result.addAttribute(tile_info_attr_name, tile_info);
  }

  mlir::NamedAttrList parsed_attributes;
  mlir::SMLoc attribute_dict_location = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsed_attributes)) {
    return mlir::failure();
  }

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (llvm::StringRef disallowed :
       {symbol_attr_name, memref_types_attr_name.getValue(),
        opaque_types_attr_name.getValue(), tile_info_attr_name.getValue()}) {
    if (parsed_attributes.get(disallowed)) {
      return parser.emitError(attribute_dict_location, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
    }
  }
  result.attributes.append(parsed_attributes);

  for (const auto& arg : entry_args) {
    if (!arg.attrs.empty()) {
      return parser.emitError(attribute_dict_location)
             << "argument attributes are not supported for entry function";
    }
  }

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  mlir::Region* body = result.addRegion();
  mlir::SMLoc loc = parser.getCurrentLocation();
  mlir::OptionalParseResult body_parse_result =
      parser.parseOptionalRegion(*body, entry_args,
                                 /*enableNameShadowing=*/false);
  if (body_parse_result.has_value()) {
    if (mlir::failed(*body_parse_result)) {
      return mlir::failure();
    }
    // Function body was parsed, make sure its not empty.
    if (body->empty()) {
      return parser.emitError(loc, "expected non-empty function body");
    }
  }
  return mlir::success();
}

void EntryFuncOp::print(mlir::OpAsmPrinter& printer) {
  printer << ' ';

  printer.printSymbolName(getSymName());

  llvm::ArrayRef<mlir::Type> arg_types = getArgumentTypes();

  // Entry function has no return values.
  llvm::ArrayRef<mlir::Type> result_types;
  auto return_attrs = mlir::ArrayAttr::get(getContext(), {});

  mlir::Region& body = getRegion();
  mlir::call_interface_impl::printFunctionSignature(
      printer, arg_types, getArgAttrsAttr(), false, result_types, return_attrs,
      &body,
      /*printEmptyResult=*/false);

  if (auto tile_info_attr = getTileInfoAttr()) {
    printer << " tiling ";
    printer.printAttribute(tile_info_attr);
  }

  mlir::function_interface_impl::printFunctionAttributes(
      printer, *this,
      {getMemrefArgTypesAttrName(), getOpaqueArgTypesAttrName(),
       getTileInfoAttrName()});
  // Print the body if this is not an external function.
  if (!body.empty()) {
    printer << ' ';
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

mlir::LogicalResult EntryFuncOp::verify() {
  if (!getResultTypes().empty()) {
    return emitOpError() << "entry function should not have any return values";
  }

  if (getArgumentTypes().empty()) {
    return emitOpError()
           << "entry function must have at least the workgroup id";
  }

  for (mlir::Type buffer_type : getBufferArgs().getTypes()) {
    if (!mlir::isa<mlir::MemRefType>(buffer_type)) {
      return emitOpError() << argument_error;
    }
  }

  for (mlir::Type opaque_types : getOpaqueArgs().getTypes()) {
    if (mlir::isa<mlir::MemRefType>(opaque_types)) {
      return emitOpError() << "opaque arguments cannot be memrefs";
    }
  }

  if (!mlir::isa<mlir::IndexType>(getArgumentTypes().back())) {
    return emitOpError() << argument_error;
  }

  return mlir::success();
}

mlir::FunctionType EntryFuncOp::getFunctionType() {
  ::mlir::ArrayAttr memref_arg_types = getMemrefArgTypes();
  llvm::SmallVector<mlir::Type> arg_types;
  for (mlir::Attribute type : memref_arg_types) {
    arg_types.push_back(mlir::cast<mlir::TypeAttr>(type).getValue());
  }
  ::mlir::ArrayAttr opaque_arg_types = getOpaqueArgTypes();
  for (mlir::Attribute type : opaque_arg_types) {
    arg_types.push_back(mlir::cast<mlir::TypeAttr>(type).getValue());
  }
  mlir::MLIRContext* context = getContext();
  arg_types.push_back(mlir::IndexType::get(context));
  return mlir::FunctionType::get(context, arg_types, /*results=*/{});
}

void EntryFuncOp::setFunctionTypeAttr(mlir::TypeAttr type) {
  auto func_type = mlir::dyn_cast<mlir::FunctionType>(type.getValue());
  if (!func_type) {
    return;
  }

  llvm::ArrayRef<mlir::Type> arg_types = func_type.getInputs();
  if (arg_types.empty() || !mlir::isa<mlir::IndexType>(arg_types.back())) {
    emitOpError() << argument_error;
    return;
  }

  mlir::MLIRContext* context = getContext();

  auto non_tile_id_args = llvm::drop_end(arg_types);
  auto arg_itr = non_tile_id_args.begin();

  mlir::SmallVector<mlir::Attribute> memref_arg_types;
  while (arg_itr != non_tile_id_args.end() &&
         mlir::isa<mlir::MemRefType>(*arg_itr)) {
    memref_arg_types.push_back(mlir::TypeAttr::get(*arg_itr));
    ++arg_itr;
  }
  setMemrefArgTypesAttr(mlir::ArrayAttr::get(context, memref_arg_types));

  mlir::SmallVector<mlir::Attribute> opaque_arg_types;
  for (; arg_itr != non_tile_id_args.end(); ++arg_itr) {
    opaque_arg_types.push_back(mlir::TypeAttr::get(*arg_itr));
  }
  setOpaqueArgTypesAttr(mlir::ArrayAttr::get(context, opaque_arg_types));
}

// Entry func currently does not support argument attributes.
mlir::ArrayAttr EntryFuncOp::getArgAttrsAttr() { return {}; }
void EntryFuncOp::setArgAttrsAttr(mlir::ArrayAttr) {}
mlir::Attribute EntryFuncOp::removeArgAttrsAttr() { return {}; }

// Entry func has no return values.
mlir::ArrayAttr EntryFuncOp::getResAttrsAttr() {
  return mlir::ArrayAttr::get(getContext(), {});
}
void EntryFuncOp::setResAttrsAttr(mlir::ArrayAttr) {}
mlir::Attribute EntryFuncOp::removeResAttrsAttr() { return {}; }

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

llvm::SmallVector<int64_t> MaskOp::getMaskedDimensions() {
  llvm::SmallVector<int64_t> masked_dimensions;

  int64_t idx = 0;
  for (const auto [bound_size, tensor_size] :
       llvm::zip(getBounds(), getType().getShape())) {
    if (bound_size < tensor_size) {
      masked_dimensions.push_back(idx);
    }
    ++idx;
  }

  return masked_dimensions;
}

mlir::LogicalResult MaskOp::verify() {
  mlir::ArrayRef<int64_t> tensor_shape = getType().getShape();
  mlir::ArrayRef<int64_t> bounds = getBounds();

  if (tensor_shape.size() != bounds.size()) {
    return emitOpError() << "tensor rank: " << tensor_shape.size()
                         << " does not match mask bounds rank: "
                         << bounds.size();
  }

  for (const auto [bound_size, tensor_size] : llvm::zip(bounds, tensor_shape)) {
    if (bound_size > tensor_size) {
      return emitOpError()
             << "mask bound not less than or equal to the tensor size";
    }
  }

  return mlir::success();
}

mlir::OpFoldResult MaskOp::fold(FoldAdaptor) {
  if (getMaskedDimensions().empty()) {
    // If none of the dimensions are masked then the op is a nop.
    return getSource();
  }

  return {};
}

}  // namespace xla::xtile
