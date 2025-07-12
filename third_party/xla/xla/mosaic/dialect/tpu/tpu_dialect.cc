/* Copyright 2023 The JAX Authors.

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

#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

#include <cstdint>
#include <optional>

#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep.
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.cc.inc"
#include "xla/mosaic/dialect/tpu/tpu_enums.cc.inc"

// This is a bit unclean, but we need to squat the xla namespace to make sure
// that this overload is found via argument-dependent lookup.
namespace xla {

llvm::hash_code hash_value(const ::xla::Tile &p) { return absl::HashOf(p); }

}  // namespace xla

#define GET_ATTRDEF_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_attr_defs.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "xla/mosaic/dialect/tpu/tpu_type_defs.cc.inc"

namespace mlir::tpu {

void TPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/mosaic/dialect/tpu/tpu_attr_defs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/mosaic/dialect/tpu/tpu_type_defs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "xla/mosaic/dialect/tpu/tpu_ops.cc.inc"
      >();
}

Operation *TPUDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

/* static */ std::optional<CoreType> TPUDialect::GetCoreTypeAttr(
    Operation *op) {
  Attribute attr = op->getAttr(GetCoreTypeKey());
  if (attr == nullptr) {
    return std::nullopt;
  }
  if (!mlir::isa<CoreTypeAttr>(attr)) {
    return std::nullopt;
  }
  return mlir::cast<CoreTypeAttr>(attr).getValue();
}

FailureOr<CoreType> GetCoreTypeOfParentFunc(Operation &op) {
  mlir::Operation *func_op = op.getParentOfType<mlir::func::FuncOp>();
  if (func_op == nullptr) {
    return op.emitError() << "Operation " << op.getName()
                          << " is not inside a func.func";
  }
  return TPUDialect::GetCoreTypeAttr(func_op).value_or(CoreType::kTc);
}

void VectorLayoutAttr::print(AsmPrinter &printer) const {
  printer << '<';
  printer << getLayout();
  printer << '>';
}

Attribute VectorLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  if (auto layout = parseLayout(parser);
      layout.has_value() && succeeded(parser.parseGreater())) {
    return get(parser.getContext(), *layout);
  }
  return {};
}

void TiledLayoutAttr::print(AsmPrinter &printer) const {
  printer << '<';
  for (const xla::Tile &tile : getTiles()) {
    printer << tile.ToString();
  }
  printer << ",[";
  for (int i = 0; i < getTileStrides().size(); ++i) {
    if (i > 0) {
      printer << ',';
    }
    printer << getTileStrides()[i];
  }
  printer << "]>";
}

Attribute TiledLayoutAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  SmallVector<xla::Tile, 2> tiles;
  int64_t size;
  while (succeeded(parser.parseOptionalLParen())) {
    xla::Tile &tile = tiles.emplace_back();
    bool first = true;
    while (!succeeded(parser.parseOptionalRParen())) {
      if (!first) {
        if (failed(parser.parseComma())) {
          return {};
        }
      }
      first = false;
      if (failed(parser.parseInteger(size))) {
        return {};
      }
      tile.add_dimensions(size);
    }
  }
  SmallVector<int64_t, 2> tile_strides;
  int64_t stride;
  if (failed(parser.parseComma())) {
    return {};
  }
  if (succeeded(parser.parseOptionalLSquare())) {
    bool first = true;
    while (!succeeded(parser.parseOptionalRSquare())) {
      if (!first) {
        if (failed(parser.parseComma())) {
          return {};
        }
      }
      first = false;
      if (failed(parser.parseInteger(stride))) {
        return {};
      }
      tile_strides.push_back(stride);
    }
  } else {
    return {};
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return get(parser.getContext(), tiles, tile_strides);
}

AffineMap TiledLayoutAttr::getAffineMap() const {
  AffineMap map =
      AffineMap::getMultiDimIdentityMap(getTileStrides().size(), getContext());
  SmallVector<AffineExpr, 8> exprs;
  for (const xla::Tile &tile : getTiles()) {
    exprs.clear();
    auto dimensions = tile.dimensions();
    int64_t untiled_dims = map.getNumResults() - dimensions.size();
    if (untiled_dims < 0) {
      LOG(FATAL) << "Invalid TiledLayoutAttr: Number of dims must be larger "
                    "or equal to the rank of the tile";
    }
    for (int64_t i = 0; i < untiled_dims; ++i) {
      exprs.push_back(getAffineDimExpr(i, getContext()));
    }
    for (int i = 0; i < dimensions.size(); ++i) {
      exprs.push_back(getAffineDimExpr(untiled_dims + i, getContext())
                          .floorDiv(dimensions[i]));
    }
    for (int i = 0; i < dimensions.size(); ++i) {
      exprs.push_back(getAffineDimExpr(untiled_dims + i, getContext()) %
                      dimensions[i]);
    }
    auto tile_map = AffineMap::get(map.getNumResults(), 0, exprs, getContext());
    map = tile_map.compose(map);
  }
  return map;
}

MemRefType getMemRefType(Value value) {
  if (auto erase_op = value.getDefiningOp<tpu::EraseLayoutOp>()) {
    value = erase_op.getOperand();
  }
  return cast<MemRefType>(value.getType());
}

bool isGuaranteedDivisible(Value value, int64_t divisor, int64_t fuel) {
  if (fuel <= 0) {
    return false;
  }
  if (divisor == 1) {
    return true;
  }
  if (auto assume_op = value.getDefiningOp<tpu::AssumeMultipleOp>()) {
    return assume_op.getMultiple() % divisor == 0;
  }
  if (auto mul_op = value.getDefiningOp<arith::MulIOp>()) {
    // We check RHS first, because MLIR canonicalizes constants to the right.
    return isGuaranteedDivisible(mul_op.getRhs(), divisor, fuel / 2) ||
           isGuaranteedDivisible(mul_op.getLhs(), divisor, (fuel + 1) / 2);
  }
  if (auto cst_op = value.getDefiningOp<arith::ConstantOp>()) {
    auto int_attr = dyn_cast<IntegerAttr>(cst_op.getValue());
    return int_attr && int_attr.getInt() % divisor == 0;
  }
  if (auto cast_op = value.getDefiningOp<arith::IndexCastOp>()) {
    return isGuaranteedDivisible(cast_op.getOperand(), divisor, fuel - 1);
  }
  return false;
}

DotDimensionNumbersAttr defaultDimensionNumbers(Builder &builder,
                                                bool transpose_lhs,
                                                bool transpose_rhs) {
  return tpu::DotDimensionNumbersAttr::get(
      builder.getContext(),
      /*lhs_contracting_dims=*/{transpose_lhs ? 0 : 1},
      /*rhs_contracting_dims=*/{transpose_rhs ? 1 : 0},
      /*lhs_non_contracting_dims=*/{transpose_lhs ? 1 : 0},
      /*rhs_non_contracting_dims=*/{transpose_rhs ? 0 : 1},
      /*output_dim_order=*/{0, transpose_lhs ? 1 : 0, 1, transpose_rhs ? 0 : 1},
      /*lhs_batch_dims=*/{},
      /*rhs_batch_dims=*/{});
}

}  // namespace mlir::tpu
