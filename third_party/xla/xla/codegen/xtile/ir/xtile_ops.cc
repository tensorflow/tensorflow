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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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

void ExtractTileOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                          mlir::Type result, mlir::Value source,
                          mlir::ValueRange offsets,
                          mlir::ArrayRef<int64_t> full_tile_shape,
                          mlir::ArrayRef<int64_t> strides,
                          std::optional<bool> should_allocate) {
  mlir::Operation* context_op = nullptr;
  if (auto* block = builder.getInsertionBlock()) {
    context_op = block->getParentOp();
  }
  bool should_allocate_val =
      should_allocate.value_or(ShouldAllocateForLayoutFixup(
          mlir::cast<mlir::RankedTensorType>(result),
          mlir::cast<mlir::MemRefType>(source.getType()), offsets, strides,
          context_op));

  state.addOperands(source);
  state.addOperands(offsets);
  state.getOrAddProperties<Properties>().full_tile_shape =
      builder.getDenseI64ArrayAttr(full_tile_shape);
  state.getOrAddProperties<Properties>().strides =
      builder.getDenseI64ArrayAttr(strides);
  state.getOrAddProperties<Properties>().should_allocate =
      builder.getBoolAttr(should_allocate_val);
  state.addTypes(result);
}

mlir::TypedValue<mlir::MemRefType> ExtractTileOp::getBuffer() {
  return getSource();
}

mlir::TypedValue<mlir::RankedTensorType> ExtractTileOp::getTile() {
  return getResult();
}

// This is the function ODS expects you to implement
mlir::LogicalResult ExtractTileOp::verify() {
  if (auto result = VerifyBufferOp(*this); result.failed()) {
    return result;
  }
  return mlir::success();
}

void InsertTileOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                         mlir::Value source, mlir::Value destination,
                         mlir::ValueRange offsets,
                         mlir::ArrayRef<int64_t> full_tile_shape,
                         mlir::ArrayRef<int64_t> strides,
                         std::optional<bool> should_allocate) {
  mlir::Operation* context_op = nullptr;
  if (auto* block = builder.getInsertionBlock()) {
    context_op = block->getParentOp();
  }
  bool should_allocate_val =
      should_allocate.value_or(ShouldAllocateForLayoutFixup(
          mlir::cast<mlir::RankedTensorType>(source.getType()),
          mlir::cast<mlir::MemRefType>(destination.getType()), offsets, strides,
          context_op));

  state.addOperands(source);
  state.addOperands(destination);
  state.addOperands(offsets);
  state.getOrAddProperties<Properties>().full_tile_shape =
      builder.getDenseI64ArrayAttr(full_tile_shape);
  state.getOrAddProperties<Properties>().strides =
      builder.getDenseI64ArrayAttr(strides);
  state.getOrAddProperties<Properties>().should_allocate =
      builder.getBoolAttr(should_allocate_val);
}

mlir::TypedValue<mlir::MemRefType> InsertTileOp::getBuffer() {
  return getDestination();
}

mlir::TypedValue<mlir::RankedTensorType> InsertTileOp::getTile() {
  return getSource();
}

mlir::LogicalResult InsertTileOp::verify() { return VerifyBufferOp(*this); }

constexpr int64_t kXlaCpuDefaultCacheLineAlignment = 64;

bool IsAlwaysFullSize(mlir::RankedTensorType tile_type,
                      mlir::MemRefType buffer_type, mlir::ValueRange offsets,
                      llvm::ArrayRef<int64_t> strides) {
  if (tile_type.getRank() != buffer_type.getRank()) {
    return false;
  }
  llvm::ArrayRef<int64_t> tile_shape = tile_type.getShape();
  llvm::ArrayRef<int64_t> buffer_shape = buffer_type.getShape();
  if (offsets.size() != tile_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    mlir::Value offset = offsets[i];
    int64_t offset_val = 0;
    mlir::APInt const_offset;
    if (mlir::matchPattern(offset, mlir::m_ConstantInt(&const_offset))) {
      offset_val = const_offset.getSExtValue();
    } else if (!mlir::matchPattern(offset, mlir::m_Zero())) {
      return false;
    }
    int64_t tile_size = tile_shape[i];
    int64_t stride = strides[i];
    int64_t buffer_size = buffer_shape[i];
    if (mlir::ShapedType::isDynamic(buffer_size) ||
        mlir::ShapedType::isDynamic(tile_size)) {
      return false;
    }
    if (offset_val < 0 ||
        offset_val + (tile_size - 1) * stride >= buffer_size) {
      return false;
    }
  }
  return true;
}

bool CanBeFullSize(mlir::RankedTensorType tile_type,
                   mlir::MemRefType buffer_type) {
  if (tile_type.getRank() != buffer_type.getRank()) {
    return false;
  }
  for (auto [tile_size, buffer_size] :
       llvm::zip(tile_type.getShape(), buffer_type.getShape())) {
    if (mlir::ShapedType::isDynamic(tile_size) ||
        mlir::ShapedType::isDynamic(buffer_size)) {
      continue;
    }
    if (tile_size > buffer_size) {
      return false;
    }
  }
  return true;
}

static bool IsContiguousSlice(llvm::ArrayRef<int64_t> tile_shape,
                              llvm::ArrayRef<int64_t> buffer_shape) {
  int rank = tile_shape.size();
  int k = rank - 1;
  while (k >= 0 && tile_shape[k] == buffer_shape[k]) {
    k--;
  }
  if (k < 0) {
    return true;
  }

  for (int i = 0; i < k; i++) {
    if (tile_shape[i] != 1) {
      return false;
    }
  }
  return true;
}

bool ShouldAllocateForLayoutFixup(mlir::RankedTensorType tile_type,
                                  mlir::MemRefType buffer_type,
                                  mlir::ValueRange offsets,
                                  llvm::ArrayRef<int64_t> strides,
                                  mlir::Operation* op) {
  if (!IsAlwaysFullSize(tile_type, buffer_type, offsets, strides)) {
    return true;
  }

  if (tile_type.getRank() != buffer_type.getRank()) {
    return true;
  }

  if (!IsContiguousSlice(tile_type.getShape(), buffer_type.getShape())) {
    return true;
  }
  if (!buffer_type.getLayout().isIdentity()) {
    return true;
  }

  int64_t alignment = kXlaCpuDefaultCacheLineAlignment;
  if (op && tile_type.getRank() > 0) {
    int64_t minor_dim = tile_type.getRank() - 1;
    int64_t minor_tile_size = tile_type.getDimSize(minor_dim);
    if (!mlir::ShapedType::isDynamic(minor_tile_size)) {
      auto element_type = tile_type.getElementType();
      auto vector_type = mlir::VectorType::get({minor_tile_size}, element_type);
      mlir::DataLayout layout = mlir::DataLayout::closest(op);
      alignment = layout.getTypePreferredAlignment(vector_type);
    }
  } else if (tile_type.getRank() > 0) {
    int64_t minor_dim = tile_type.getRank() - 1;
    int64_t minor_tile_size = tile_type.getDimSize(minor_dim);

    auto element_type = tile_type.getElementType();
    int64_t element_alignment =
        std::max<int64_t>(1, element_type.getIntOrFloatBitWidth() / 8);
    alignment = llvm::PowerOf2Ceil(element_alignment * minor_tile_size);
  }

  if (tile_type.getRank() > 0) {
    int64_t minor_dim = tile_type.getRank() - 1;
    int64_t minor_tile_size = tile_type.getDimSize(minor_dim);
    int64_t bit_width = tile_type.getElementType().getIntOrFloatBitWidth();
    int64_t element_size = bit_width / 8;
    if (element_size > 0 && (minor_tile_size * element_size) % alignment != 0) {
      return true;
    }
  }

  int64_t bit_width = tile_type.getElementType().getIntOrFloatBitWidth();
  int64_t element_size = bit_width / 8;

  for (mlir::Value offset : offsets) {
    mlir::APInt const_offset;
    if (mlir::matchPattern(offset, mlir::m_ConstantInt(&const_offset))) {
      if (element_size > 0) {
        int64_t offset_in_bytes = const_offset.getSExtValue() * element_size;
        if (offset_in_bytes % alignment != 0) {
          return true;
        }
      } else {
        return true;
      }
    } else {
      // If the offset is not a constant, we must assume allocation is needed.
      return true;
    }
  }
  for (int64_t stride : strides) {
    if (stride != 1) {
      return true;
    }
  }
  return false;
}

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

static bool IsSupportedScaleElementType(mlir::Type type) {
  return mlir::isa<mlir::Float8E8M0FNUType, mlir::Float8E4M3FNType,
                   mlir::Float8E5M2Type>(type) ||
         (mlir::isa<mlir::IntegerType>(type) &&
          mlir::cast<mlir::IntegerType>(type).getWidth() == 8);
}

static bool IsSupportedOperandElementType(mlir::Type type) {
  if (mlir::isa<mlir::FloatType>(type)) {
    return true;
  }
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return int_type.getWidth() == 8 || int_type.getWidth() == 4;
  }
  return false;
}

mlir::LogicalResult DotScaledOp::verify() {
  mlir::Type lhs_storage_type =
      mlir::cast<mlir::ShapedType>(getLhs().getType()).getElementType();
  if (!IsSupportedOperandElementType(lhs_storage_type)) {
    return emitOpError() << "LHS tensor element type " << lhs_storage_type
                         << " is not supported. Supported LHS element "
                            "types are float or int8/uint8";
  }

  mlir::Type rhs_storage_type =
      mlir::cast<mlir::ShapedType>(getRhs().getType()).getElementType();
  if (!IsSupportedOperandElementType(rhs_storage_type)) {
    return emitOpError() << "RHS tensor element type " << rhs_storage_type
                         << " is not supported. Supported RHS element "
                            "types are float or int8/uint8";
  }

  if (!IsSupportedOperandElementType(getLhsElemType())) {
    return emitOpError() << "LHS logical element type " << getLhsElemType()
                         << " is not supported. Supported LHS logical "
                            "element types are float or int8/int4";
  }

  if (!IsSupportedOperandElementType(getRhsElemType())) {
    return emitOpError() << "RHS logical element type " << getRhsElemType()
                         << " is not supported. Supported RHS logical "
                            "element types are float or int8/int4";
  }

  if (mlir::Value lhs_scale = getLhsScale()) {
    mlir::Type scale_type =
        mlir::cast<mlir::ShapedType>(lhs_scale.getType()).getElementType();
    if (!IsSupportedScaleElementType(scale_type)) {
      return emitOpError() << "LHS scale element type " << scale_type
                           << " is not supported. Supported scale element "
                              "types are: f8E8M0FNU, f8E4M3FN, f8E5M2, i8/s8";
    }
  }
  if (mlir::Value rhs_scale = getRhsScale()) {
    mlir::Type scale_type =
        mlir::cast<mlir::ShapedType>(rhs_scale.getType()).getElementType();
    if (!IsSupportedScaleElementType(scale_type)) {
      return emitOpError() << "RHS scale element type " << scale_type
                           << " is not supported. Supported scale element "
                              "types are: f8E8M0FNU, f8E4M3FN, f8E5M2, i8/s8";
    }
  }
  return mlir::success();
}

}  // namespace xla::xtile
