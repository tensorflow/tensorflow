/* Copyright 2026 The OpenXLA Authors.

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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep.
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep.
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.cc.inc"
#include "xla/mosaic/dialect/tpu/tpu_enums.cc.inc"
#include "xla/mosaic/dialect/tpu/util.h"
#include "xla/layout.h"

// This is a bit unclean, but we need to squat the xla namespace to make sure
// that this overload is found via argument-dependent lookup.
namespace xla {

llvm::hash_code hash_value(const ::xla::Tile& p) { return absl::HashOf(p); }

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

Operation* TPUDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                           Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

/* static */ std::optional<CoreType> TPUDialect::GetCoreTypeAttr(
    Operation* op) {
  Attribute attr = op->getAttr(GetCoreTypeKey());
  if (attr == nullptr) {
    // For backwards compatibility we assume that the "main" function without
    // an explicit core type belongs to TensorCore.
    // TODO(b/505757864): Remove this in 6 months.
    if (auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(op);
        func_op && func_op.getName() == "main") {
      return CoreType::kTc;
    }
    return std::nullopt;
  }
  if (!mlir::isa<CoreTypeAttr>(attr)) {
    return std::nullopt;
  }
  return mlir::cast<CoreTypeAttr>(attr).getValue();
}

// Rewrites
//
//     memref.dim(tpu.memref_slice(..., dynamic_sizes), i)
//
// to
//
//     dynamic_sizes[dynamicDimIndex(i)]
//
// if i is a constant and refers to a dynamic dimension.
struct MemRefDimOfSlice : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dim_op,
                                PatternRewriter& rewriter) const override {
    auto slice_op = dim_op.getSource().getDefiningOp<MemRefSliceOp>();
    if (!slice_op) {
      return failure();
    }
    const std::optional<int64_t> maybe_dim =
        getConstantIntValue(dim_op.getDimension());
    if (!maybe_dim) {
      return failure();
    }
    const int64_t dim = *maybe_dim;
    MemRefType result_type = slice_op.getType();
    if (dim < 0 || result_type.getRank() <= dim) {
      return dim_op.emitWarning("Dimension index is out of bounds");
    }
    if (result_type.getDimSize(dim) != ShapedType::kDynamic) {
      return failure();
    }
    const unsigned dynamic_dim_idx = result_type.getDynamicDimIndex(dim);
    ValueRange dynamic_sizes = slice_op.getDynamicSizes();
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
        dim_op, dim_op.getType(), dynamic_sizes[dynamic_dim_idx]);
    return success();
  }
};

// Rewrites memref.dim(tpu.memref_squeeze(x)) to memref.dim(x) with the
// dimension index adjusted to account for squeezed dimensions.
struct MemRefDimOfSqueeze : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dim_op,
                                PatternRewriter& rewriter) const override {
    auto squeeze_op = dim_op.getSource().getDefiningOp<MemRefSqueezeOp>();
    if (!squeeze_op) {
      return failure();
    }
    const std::optional<int64_t> maybe_dim =
        getConstantIntValue(dim_op.getDimension());
    if (!maybe_dim) {
      return failure();
    }
    const int64_t dim = *maybe_dim;
    MemRefType result_type = squeeze_op.getType();
    if (dim < 0 || result_type.getRank() <= dim) {
      return dim_op.emitWarning("Dimension index is out of bounds");
    }
    if (result_type.getDimSize(dim) != ShapedType::kDynamic) {
      return failure();
    }
    MemRefType source_type = squeeze_op.getInput().getType();
    FAILUREOR_ASSIGN_OR_RETURN(
        SmallVector<int> squeezed,
        computeSqueezedDimsChecked(squeeze_op, source_type.getShape(),
                                   result_type.getShape()));
    int64_t source_dim = dim;
    for (int squeezed_dim : squeezed) {
      if (squeezed_dim <= source_dim) {
        ++source_dim;
      }
    }
    rewriter.replaceOpWithNewOp<memref::DimOp>(dim_op, squeeze_op.getInput(),
                                               source_dim);
    return success();
  }
};

void TPUDialect::getCanonicalizationPatterns(RewritePatternSet& results) const
/*override*/ {
  results.add<MemRefDimOfSlice, MemRefDimOfSqueeze>(getContext());
}

Operation* GetParentOpWithCoreType(Operation& op) {
  Operation* parent = &op;
  while ((parent = parent->getParentOp())) {
    if (auto core_type = TPUDialect::GetCoreTypeAttr(parent);
        core_type.has_value()) {
      return parent;
    }
  }
  return nullptr;
}

CoreType GetCoreTypeOfParentOp(Operation& op) {
  Operation* parent = GetParentOpWithCoreType(op);
  return parent ? *TPUDialect::GetCoreTypeAttr(parent) : CoreType::kTc;
}

absl::StatusOr<func::FuncOp> GetFuncWithCoreType(ModuleOp module,
                                                 CoreType core_type) {
  func::FuncOp result = nullptr;
  for (func::FuncOp func_op : module.getOps<func::FuncOp>()) {
    if (TPUDialect::GetCoreTypeAttr(func_op) != core_type) {
      continue;
    }
    if (result != nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Multiple functions with tpu.core_type = %v found", core_type));
    }
    result = func_op;
  }
  if (result != nullptr) {
    return result;
  }
  // We have to maintain backwards compatibility with TC kernels which do not
  // set the tpu.core_type attribute.
  bool fallback_to_main = core_type == CoreType::kTc;
  if (!fallback_to_main) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "No function with tpu.core_type = %v found", core_type));
  }
  if (auto main_func = module.lookupSymbol<func::FuncOp>("main")) {
    return main_func;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "No function with tpu.core_type = %v nor a main function found",
      core_type));
}

void VectorLayoutAttr::print(AsmPrinter& printer) const {
  printer << '<';
  printer << getLayout();
  printer << '>';
}

Attribute VectorLayoutAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  if (auto layout = parseLayout(parser);
      layout.has_value() && succeeded(parser.parseGreater())) {
    return get(parser.getContext(), *layout);
  }
  return {};
}

void TiledLayoutAttr::print(AsmPrinter& printer) const {
  printer << '<';
  for (const xla::Tile& tile : getTiles()) {
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

Attribute TiledLayoutAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) {
    return {};
  }
  SmallVector<xla::Tile, 2> tiles;
  int64_t size;
  while (succeeded(parser.parseOptionalLParen())) {
    xla::Tile& tile = tiles.emplace_back();
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
  SmallVector<AffineExpr, 8> exprs;
  for (int64_t i = 0; i < getRank(); ++i) {
    exprs.push_back(getAffineDimExpr(i, getContext()));
  }
  for (const xla::Tile& tile : getTiles()) {
    SmallVector<AffineExpr, 8> new_exprs;
    auto dimensions = tile.dimensions();
    int64_t untiled_rank = exprs.size() - dimensions.size();
    assert(untiled_rank >= 0);
    for (int64_t i = 0; i < untiled_rank; ++i) {
      new_exprs.push_back(exprs[i]);
    }
    for (int64_t i = 0; i < dimensions.size(); ++i) {
      new_exprs.push_back(exprs[untiled_rank + i].floorDiv(dimensions[i]));
    }
    for (int64_t i = 0; i < dimensions.size(); ++i) {
      new_exprs.push_back(exprs[untiled_rank + i] % dimensions[i]);
    }
    exprs = std::move(new_exprs);
  }
  int64_t num_symbols = 0;
  AffineExpr result = getAffineConstantExpr(0, getContext());
  SmallVector<int64_t> strides = getExpandedStrides();
  assert(strides.size() == exprs.size());
  for (int64_t i = 0; i < exprs.size(); ++i) {
    AffineExpr stride_expr =
        ShapedType::isDynamic(strides[i])
            ? getAffineSymbolExpr(num_symbols++, getContext())
            : getAffineConstantExpr(strides[i], getContext());
    result = result + exprs[i] * stride_expr;
  }
  return AffineMap::get(getRank(), num_symbols, result);
}

namespace {
int64_t getUntiledRank(ArrayRef<xla::Tile> tiles, const int64_t rank) {
  // Note: This implementation does not assume there is no nested tiling across
  // the first level of tiling, though this is enforced by the verifier.
  int64_t untiled_rank = rank;
  int64_t tiled_rank = rank;
  for (const xla::Tile& tile : tiles) {
    const int64_t tile_ndims = tile.dimensions().size();
    untiled_rank = std::min(untiled_rank, tiled_rank - tile_ndims);
    tiled_rank += tile_ndims;
  }
  return untiled_rank;
}
}  // namespace

int64_t TiledLayoutAttr::getUntiledRank() const {
  return mlir::tpu::getUntiledRank(getTiles(), getRank());
}

namespace {
FailureOr<SmallVector<int64_t>> getExpandedShape(
    const ArrayRef<int64_t> untiled_shape, const ArrayRef<xla::Tile> tiles,
    const bool require_alignment) {
  SmallVector<int64_t> shape(untiled_shape);
  for (const xla::Tile& tile : tiles) {
    const int64_t tile_ndims = tile.dimensions().size();
    const llvm::ArrayRef<int64_t> tiled_shape =
        llvm::ArrayRef(shape).take_back(tile_ndims);
    llvm::SmallVector<int64_t> new_tiled_shape(2 * tile_ndims);
    for (int64_t i = 0; i < tile_ndims; ++i) {
      if (require_alignment && (ShapedType::isDynamic(tiled_shape[i]) ||
                                tiled_shape[i] % tile.dimension(i) != 0)) {
        return failure();
      }
      if (ShapedType::isDynamic(tiled_shape[i])) {
        new_tiled_shape[i] = ShapedType::kDynamic;
      } else {
        new_tiled_shape[i] =
            llvm::divideCeil(tiled_shape[i], tile.dimension(i));
      }
      new_tiled_shape[tile_ndims + i] = tile.dimension(i);
    }
    shape.pop_back_n(tile_ndims);
    shape.append(new_tiled_shape);
  }
  return shape;
}
}  // namespace

SmallVector<int64_t> TiledLayoutAttr::getDefaultTileStrides(
    const ArrayRef<xla::Tile> tiles, const ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size());
  int64_t stride = 1;
  const xla::Tile* const first_tile = tiles.empty() ? nullptr : &tiles.front();
  const int64_t first_tile_rank =
      first_tile == nullptr ? 0 : first_tile->dimensions().size();
  for (int64_t d = shape.size() - 1; d >= 0; --d) {
    assert(!ShapedType::isDynamic(shape[d]));
    strides[d] = stride;
    if (d >= shape.size() - first_tile_rank) {
      assert(first_tile != nullptr);
      const int64_t tile_d = d - (shape.size() - first_tile_rank);
      stride *= llvm::divideCeil(shape[d], first_tile->dimension(tile_d));
    } else {
      stride *= shape[d];
    }
  }
  return strides;
}

TiledLayoutAttr TiledLayoutAttr::getContiguous(MLIRContext* context,
                                               ArrayRef<xla::Tile> tiles,
                                               ArrayRef<int64_t> shape) {
  return TiledLayoutAttr::get(
      context, tiles, TiledLayoutAttr::getDefaultTileStrides(tiles, shape));
}

int64_t TiledLayoutAttr::getNumTrailingDimsWithContiguousTiles(
    const ArrayRef<int64_t> shape) const {
  const ArrayRef<xla::Tile> tiles = getTiles();
  const ArrayRef<int64_t> tile_strides = getTileStrides();
  int64_t stride = 1;
  const xla::Tile* const first_tile = tiles.empty() ? nullptr : &tiles.front();
  const int64_t first_tile_rank =
      first_tile == nullptr ? 0 : first_tile->dimensions().size();
  int64_t d = shape.size() - 1;
  for (; d >= 0; --d) {
    int64_t size_tiles;
    if (d >= shape.size() - first_tile_rank &&
        shape[d] != ShapedType::kDynamic) {
      assert(first_tile != nullptr);
      const int64_t tile_d = d - (shape.size() - first_tile_rank);
      size_tiles = llvm::divideCeil(shape[d], first_tile->dimension(tile_d));
    } else {
      size_tiles = shape[d];
    }
    assert(tile_strides[d] != ShapedType::kDynamic);
    // Dimensions with only one element/tile can have any stride.
    if (stride != tile_strides[d] && size_tiles != 1) {
      break;
    }
    if (stride == ShapedType::kDynamic || size_tiles == ShapedType::kDynamic) {
      stride = ShapedType::kDynamic;
    } else {
      stride *= size_tiles;
    }
  }
  return shape.size() - 1 - d;
}

SmallVector<int64_t> TiledLayoutAttr::getExpandedShape(
    ArrayRef<int64_t> untiled_shape) const {
  // getExpandedShape should never fail without require_alignment
  return *mlir::tpu::getExpandedShape(untiled_shape, getTiles(),
                                      /*require_alignment=*/false);
}

SmallVector<int64_t> TiledLayoutAttr::getExpandedStrides() const {
  if (getTiles().empty()) {
    return SmallVector<int64_t>(getTileStrides());
  }
  SmallVector<int64_t> strides(getTileStrides());
  // Expand front tile
  const xla::Tile& first_tile = getTiles().front();
  const FailureOr<SmallVector<int64_t>> failure_or_expanded_tile =
      mlir::tpu::getExpandedShape(first_tile.dimensions(),
                                  getTiles().drop_front(),
                                  /*require_alignment=*/true);
  // Verification should ensure this:
  assert(succeeded(failure_or_expanded_tile));
  const SmallVector<int64_t>& expanded_tile = *failure_or_expanded_tile;
  strides.resize_for_overwrite(getRank() + expanded_tile.size());
  int64_t first_tile_size = llvm::product_of(first_tile.dimensions());
  int64_t tile_size = 1;
  for (int64_t d = strides.size() - 1; d >= 0; --d) {
    if (d >= getRank()) {
      const int64_t new_stride = tile_size;
      tile_size *= expanded_tile[d - getRank()];
      strides[d] = new_stride;
    } else {
      strides[d] *= first_tile_size;
    }
  }
  return strides;
}

LogicalResult TiledLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    const llvm::ArrayRef<xla::Tile> tiles,
    const llvm::ArrayRef<int64_t> tile_strides) {
  if (llvm::any_of(tile_strides, ShapedType::isDynamic)) {
    return emitError() << "Not implemented: Dynamic tile strides";
  }
  if (tiles.empty()) {
    return success();
  }
  const int64_t rank = tile_strides.size();
  const xla::Tile& first_tile = tiles.front();
  const int64_t first_tile_rank = first_tile.dimensions().size();
  // The interpretation of tile strides is unclear if there is nested tiling
  // across first tiles (e.g. T(8, 128)(2, 4, 64)), and this has no applications
  // anyway.
  if (mlir::tpu::getUntiledRank(tiles, rank) != rank - first_tile_rank) {
    return emitError() << "Not implemented: Nested tiling across first tiles";
  }
  // Check that nested tiles evenly divide previous tiles (so they don't add any
  // padding or change the tile size)
  if (failed(mlir::tpu::getExpandedShape(first_tile.dimensions(),
                                         tiles.drop_front(),
                                         /*require_alignment=*/true))) {
    return emitError() << "Not implemented: Nested tiles must evenly divide "
                       << "the first tile " << first_tile.ToString()
                       << " but they do not (would add padding)";
  }
  return success();
}

LogicalResult TiledLayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (getRank() != shape.size()) {
    return emitError() << "Layout rank does not match shape rank";
  }
  return success();
}

MemRefType getMemRefType(Value value) {
  if (auto erase_op = value.getDefiningOp<tpu::EraseLayoutOp>()) {
    value = erase_op.getOperand();
  }
  return cast<MemRefType>(value.getType());
}

std::optional<bool> isDivisible(Value value, int64_t divisor, int64_t fuel);

namespace {
// Returns true if divisibilities of both lhs and rhs can be proven.
// Returns false if divisibilities of both lhs and rhs can be disproven.
// Returns nullopt if any of the two divisibilities is not known.
std::optional<bool> areAllDivisible(Value lhs, Value rhs, int64_t divisor,
                                    int64_t fuel) {
  std::optional<bool> lhs_divisible = isDivisible(lhs, divisor, fuel / 2);
  // If either is known to be false, the result is false.
  if (lhs_divisible.has_value() && !*lhs_divisible) {
    return false;
  }
  std::optional<bool> rhs_divisible = isDivisible(rhs, divisor, (fuel + 1) / 2);
  // If either is known to be false, the result is false.
  if (rhs_divisible.has_value() && !*rhs_divisible) {
    return false;
  }

  // If both are known to be true, the result is true.
  if (lhs_divisible.has_value() && *lhs_divisible &&
      rhs_divisible.has_value() && *rhs_divisible) {
    return true;
  }

  // Otherwise, the result is unknown.
  return std::nullopt;
}

// Returns true if we can prove that at least one of lhs or rhs is divisible.
// Returns false if we can prove that neither lhs nor rhs is divisible.
// Returns nullopt if we can't prove that at least one is divisible.
std::optional<bool> areAnyDivisible(Value lhs, Value rhs, int64_t divisor,
                                    int64_t fuel) {
  std::optional<bool> lhs_divisible = isDivisible(lhs, divisor, fuel / 2);
  // If either is known to be true, the result is true.
  if (lhs_divisible.has_value() && *lhs_divisible) {
    return true;
  }
  std::optional<bool> rhs_divisible = isDivisible(rhs, divisor, (fuel + 1) / 2);

  // If either is known to be true, the result is true.
  if (rhs_divisible.has_value() && *rhs_divisible) {
    return true;
  }

  // If both are known to be false, the result is false.
  if (lhs_divisible.has_value() && !*lhs_divisible &&
      rhs_divisible.has_value() && !*rhs_divisible) {
    return false;
  }

  // Otherwise, the result is unknown.
  return std::nullopt;
}
}  // namespace

std::optional<int64_t> getRemainder(Value value, int64_t divisor,
                                    int64_t fuel) {
  if (auto cst_op = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto int_attr = dyn_cast<IntegerAttr>(cst_op.getValue())) {
      return int_attr.getInt() % divisor;
    }
  }
  if (isGuaranteedDivisible(value, divisor, fuel)) {
    return 0;
  }
  return std::nullopt;
}

std::optional<bool> isDivisible(Value value, int64_t divisor, int64_t fuel) {
  if (fuel <= 0) {
    return std::nullopt;
  }
  if (divisor == 1) {
    return true;
  }
  if (auto block_arg = dyn_cast<BlockArgument>(value)) {
    if (auto for_op =
            dyn_cast<scf::ForOp>(block_arg.getOwner()->getParentOp())) {
      if (for_op.getInductionVar() == value) {
        return areAllDivisible(for_op.getLowerBound(), for_op.getStep(),
                               divisor, fuel);
      }
    }
  }
  if (auto assume_op = value.getDefiningOp<tpu::AssumeMultipleOp>()) {
    if (assume_op.getMultiple() % divisor == 0) {
      return true;
    }
    return isDivisible(assume_op.getOperand(), divisor, fuel - 1);
  }
  if (auto mul_op = value.getDefiningOp<arith::MulIOp>()) {
    // We check RHS first, because MLIR canonicalizes constants to the right.
    if (auto rhs_cst = mlir::getConstantIntValue(mul_op.getRhs())) {
      int64_t gcd = std::gcd(*rhs_cst, divisor);
      if (gcd > 1) {
        return isDivisible(mul_op.getLhs(), divisor / gcd, fuel - 1);
      }
    }
    if (auto lhs_cst = mlir::getConstantIntValue(mul_op.getLhs())) {
      int64_t gcd = std::gcd(*lhs_cst, divisor);
      if (gcd > 1) {
        return isDivisible(mul_op.getRhs(), divisor / gcd, fuel - 1);
      }
    }
    return areAnyDivisible(mul_op.getRhs(), mul_op.getLhs(), divisor, fuel);
  }
  if (auto cst_op = value.getDefiningOp<arith::ConstantOp>()) {
    auto int_attr = dyn_cast<IntegerAttr>(cst_op.getValue());
    if (int_attr == nullptr) {
      // Floating point divisibility check is not supported.
      return std::nullopt;
    }
    return int_attr.getInt() % divisor == 0;
  }
  if (auto cast_op = value.getDefiningOp<arith::IndexCastOp>()) {
    return isDivisible(cast_op.getOperand(), divisor, fuel - 1);
  }
  if (auto div_op = value.getDefiningOp<arith::DivUIOp>()) {
    if (auto rhs_cst = mlir::getConstantIntValue(div_op.getRhs())) {
      return isDivisible(div_op.getLhs(), divisor * *rhs_cst, fuel - 1);
    }
  }
  if (auto div_op = value.getDefiningOp<arith::DivSIOp>()) {
    if (auto rhs_cst = mlir::getConstantIntValue(div_op.getRhs())) {
      return isDivisible(div_op.getLhs(), divisor * *rhs_cst, fuel - 1);
    }
  }
  if (auto add_op = value.getDefiningOp<arith::AddIOp>()) {
    return areAllDivisible(add_op.getLhs(), add_op.getRhs(), divisor, fuel);
  }
  if (auto sub_op = value.getDefiningOp<arith::SubIOp>()) {
    return areAllDivisible(sub_op.getLhs(), sub_op.getRhs(), divisor, fuel);
  }
  if (auto rem_op = value.getDefiningOp<arith::RemSIOp>()) {
    return areAllDivisible(rem_op.getLhs(), rem_op.getRhs(), divisor, fuel);
  }
  if (auto min_op = value.getDefiningOp<arith::MinSIOp>()) {
    return areAllDivisible(min_op.getLhs(), min_op.getRhs(), divisor, fuel);
  }
  if (auto select_op = value.getDefiningOp<arith::SelectOp>()) {
    auto true_val_divisible =
        isDivisible(select_op.getTrueValue(), divisor, fuel / 2);
    // If divisibility of either branch is unknown, the select op result
    // divisibility is unknown.
    if (!true_val_divisible.has_value()) {
      return std::nullopt;
    }
    auto false_val_divisible =
        isDivisible(select_op.getFalseValue(), divisor, (fuel + 1) / 2);
    if (!false_val_divisible.has_value()) {
      return std::nullopt;
    }
    // Divisibilities of both branches are known.
    // If both branches are divisible, the select op result is divisible.
    // If both branches are not divisible, the select op result is not
    // divisible.
    if (*true_val_divisible == *false_val_divisible) {
      return *true_val_divisible;
    }
    // If divisibilities of the two branches are known and differ, the select op
    // result divisibility is unknown.
    return std::nullopt;
  }
  return std::nullopt;
}

bool isGuaranteedDivisible(Value value, int64_t divisor, int64_t fuel) {
  return isDivisible(value, divisor, fuel).value_or(false);
}

DotDimensionNumbersAttr defaultDimensionNumbers(Builder& builder,
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

const ::llvm::fltSemantics& Float8EXMYType::getFloatSemantics() const {
  if (mlir::isa<Float6E3M2FNType>(getUnderlyingType())) {
    return llvm::APFloat::Float6E3M2FN();
  } else if (mlir::isa<Float6E2M3FNType>(getUnderlyingType())) {
    return llvm::APFloat::Float6E2M3FN();
  }
  return cast<FloatType>(getUnderlyingType()).getFloatSemantics();
}

namespace {

struct CommsAnalysisState {
  bool has_communication = false;
  bool has_custom_barrier = false;

  explicit operator bool() { return has_communication && has_custom_barrier; }
};

void analyzeCrossChipCommunication(mlir::Operation* op,
                                   CommsAnalysisState* state) {
  if (auto dma = dyn_cast<tpu::EnqueueDMAOp>(op)) {
    state->has_communication |= dma.getDeviceId() != nullptr;
  } else if (auto signal = dyn_cast<tpu::SemaphoreSignalOp>(op)) {
    state->has_communication |= signal.getDeviceId() != nullptr;
  } else if (auto barrier = dyn_cast<tpu::GetBarrierSemaphoreOp>(op)) {
    state->has_custom_barrier = true;
  }
  for (Region& region : op->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block.getOperations()) {
        analyzeCrossChipCommunication(&op, state);
        if (*state) {
          return;
        }
      }
    }
  }
}

}  // namespace

std::pair<bool, bool> mightCommunicateBetweenChips(mlir::Operation* op) {
  CommsAnalysisState state;
  analyzeCrossChipCommunication(op, &state);
  return std::make_pair(state.has_communication, state.has_custom_barrier);
}

LogicalResult verifyGather(Operation* op, ArrayRef<int64_t> operand_shape,
                           ArrayRef<int64_t> offsets_shape,
                           ArrayRef<int64_t> result_shape) {
  // Expected shapes:
  //   Slice shape   : [s0, ..., sm]
  //
  //   1D offsets:
  //     Operand shape : [z, s0, ..., sm]
  //     Offsets shape : [o]
  //     Result shape  : [o, s0, ..., sm]
  //
  //   2D offsets:
  //     Operand shape : [1, z, s0, ..., sm]
  //     Offsets shape : [1, o]
  //     Result shape  : [1, o, s0, ..., sm]

  uint64_t offsets_rank = offsets_shape.size();
  uint64_t slice_rank = result_shape.size() - offsets_rank;
  if (operand_shape.size() <= slice_rank) {
    return op->emitOpError(
               "Source (gather operand) rank must be > slice rank, ")
           << "got source rank: " << operand_shape.size()
           << ", slice rank: " << slice_rank;
  }
  uint64_t operand_sample_rank = operand_shape.size() - slice_rank;
  ArrayRef<int64_t> result_offset_dims = result_shape.take_front(offsets_rank);
  ArrayRef<int64_t> result_slice_dims = result_shape.take_back(slice_rank);
  ArrayRef<int64_t> operand_slice_dims = operand_shape.take_back(slice_rank);
  ArrayRef<int64_t> operand_sample_dims =
      operand_shape.take_front(operand_sample_rank);

  // We require offsets shape and operand sample shape to be 1D or (1, N), and
  // their ranks must match.
  // Offsets shape : [o] or [1, o]
  // Operand sample shape : [z] or [1, z]
  if (offsets_rank > 2 || (offsets_rank == 2 && offsets_shape[0] != 1)) {
    return op->emitOpError("Offsets shape must be 1D or (1, N), got (")
           << absl::StrJoin(offsets_shape, ", ") << ")";
  }
  if (operand_sample_rank > 2 ||
      (operand_sample_rank == 2 && operand_sample_dims[0] != 1)) {
    return op->emitOpError("Source (gather operand) sample shape must be ")
           << "1D or (1, N), got (" << absl::StrJoin(operand_sample_dims, ", ")
           << ")";
  }
  if (operand_sample_rank != offsets_rank) {
    return op->emitOpError("Source (gather operand) sample rank must match ")
           << "offsets rank, got " << operand_sample_rank << " vs "
           << offsets_rank;
  }

  const std::string result_shape_str = absl::StrJoin(result_shape, ", ");

  // Make sure that there is one output slice per offset.
  // Offsets shape : [o] or [1, o]
  // Result shape  : [o'0, .., o'p, s0, .., sm]
  // [o] or [1, o] == [o'0, .., o'p]
  if (!absl::c_equal(offsets_shape, result_offset_dims)) {
    return op->emitOpError("Offsets shape (")
           << absl::StrJoin(offsets_shape, ", ")
           << ") must match the majormost dimensions of the target (gather "
              "result) shape ("
           << result_shape_str << ")";
  }

  // At each offset, we are copying an ND slice of data. Make sure that the
  // slice shape is the same in the operand and the output for the gather.
  // Operand shape : [z, s0, .., sm] or [1, z, s0, .., sm]
  // Result shape :  [o, s'0, .., s'm] or [1, o, s'0, .., s'm]
  // [s0, .., sm] == [s'0, .., s'm]
  if (!absl::c_equal(operand_slice_dims, result_slice_dims)) {
    const std::string plural = slice_rank == 1 ? "" : "s";
    return op->emitOpError(absl::StrFormat(
        "%d minormost dimension%s of the source (gather operand) shape (%s) "
        "must match the minormost dimension%s of the target (gather result) "
        "shape (%s)",
        slice_rank, plural, absl::StrJoin(operand_shape, ", "), plural,
        result_shape_str));
  }
  return success();
}

LogicalResult verifyScatter(Operation* op, ArrayRef<int64_t> updates_shape,
                            ArrayRef<int64_t> offsets_shape,
                            ArrayRef<int64_t> operand_shape) {
  // Expected shapes:
  //   Slice shape   : [s0, ..., sm]
  //
  //   1D offsets:
  //     Operand shape : [z, s0, ..., sm]
  //     Offsets shape : [o]
  //     Updates shape : [o, s0, ..., sm]
  //
  //   2D offsets:
  //     Operand shape : [1, z, s0, ..., sm]
  //     Offsets shape : [1, o]
  //     Updates shape : [1, o, s0, ..., sm]

  uint64_t offsets_rank = offsets_shape.size();
  uint64_t slice_rank = updates_shape.size() - offsets_rank;
  if (operand_shape.size() <= slice_rank) {
    return op->emitOpError(
               "Target (scatter operand) rank must be > slice rank, ")
           << "got target rank: " << operand_shape.size()
           << ", slice rank: " << slice_rank;
  }
  uint64_t operand_sample_rank = operand_shape.size() - slice_rank;
  ArrayRef<int64_t> updates_offset_dims =
      updates_shape.take_front(offsets_rank);
  ArrayRef<int64_t> updates_slice_dims = updates_shape.take_back(slice_rank);
  ArrayRef<int64_t> operand_slice_dims = operand_shape.take_back(slice_rank);
  ArrayRef<int64_t> operand_sample_dims =
      operand_shape.take_front(operand_sample_rank);

  // We require offsets shape and operand sample shape to be 1D or (1, N), and
  // their ranks must match.
  // Offsets shape : [o] or [1, o]
  // Operand sample shape : [z] or [1, z]
  if (offsets_rank > 2 || (offsets_rank == 2 && offsets_shape[0] != 1)) {
    return op->emitOpError("Offsets shape must be 1D or (1, N), got (")
           << absl::StrJoin(offsets_shape, ", ") << ")";
  }
  if (operand_sample_rank > 2 ||
      (operand_sample_rank == 2 && operand_sample_dims[0] != 1)) {
    return op->emitOpError("Target (scatter operand) sample shape must be ")
           << "1D or (1, N), got (" << absl::StrJoin(operand_sample_dims, ", ")
           << ")";
  }
  if (operand_sample_rank != offsets_rank) {
    return op->emitOpError("Target (scatter operand) sample rank must match ")
           << "offsets rank, got " << operand_sample_rank << " vs "
           << offsets_rank;
  }

  const std::string updates_shape_str = absl::StrJoin(updates_shape, ", ");

  // Make sure that there is one slice of updates per offset.
  // Offsets shape : [o] or [1, o]
  // Updates shape : [o'0, .., o'p, s0, .., sm]
  // [o] or [1, o] == [o'0, .., o'p]
  if (!absl::c_equal(offsets_shape, updates_offset_dims)) {
    return op->emitOpError("Offsets shape (")
           << absl::StrJoin(offsets_shape, ", ")
           << ") must match the majormost dimensions of the source "
              "(scatter updates) shape ("
           << updates_shape_str << ")";
  }

  // At each offset, we are copying an ND slice of data. Make sure that the
  // slice shape is the same in the updates and the operand for the scatter.
  // Updates shape : [o, s0, .., sm] or [1, o, s0, .., sm]
  // Operand shape : [z, s'0, .., s'm] or [1, z, s'0, .., s'm]
  // [s0, .., sm] == [s'0, .., s'm]
  if (!absl::c_equal(operand_slice_dims, updates_slice_dims)) {
    const std::string plural = slice_rank == 1 ? "" : "s";
    return op->emitOpError(absl::StrFormat(
        "%d minormost dimension%s of the source (scatter updates) shape (%s) "
        "must match the minormost dimension%s of the target (scatter operand) "
        "shape (%s)",
        slice_rank, plural, updates_shape_str, plural,
        absl::StrJoin(operand_shape, ", ")));
  }
  return success();
}

namespace {
bool hasSharedMemorySpace(MemorySpace memory_space,
                          std::optional<CoreType> core_type) {
  return memory_space == MemorySpace::kHbm ||
         (memory_space == MemorySpace::kVmem && core_type.has_value() &&
          *core_type == CoreType::kTc) ||
         memory_space == MemorySpace::kVmemShared;
}
}  // namespace

FailureOr<bool> isGather(Operation& op, MemorySpace source_memory_space,
                         std::optional<CoreType> source_core_type,
                         MemorySpace target_memory_space,
                         std::optional<CoreType> target_core_type) {
  if (hasSharedMemorySpace(source_memory_space, source_core_type) &&
      target_memory_space == MemorySpace::kVmem) {
    return true;
  }
  if (source_memory_space == MemorySpace::kVmem &&
      hasSharedMemorySpace(target_memory_space, target_core_type)) {
    return false;
  }
  return op.emitOpError(
      "The transfer must be between HBM and VMEM, VMEM_SHARED and VMEM, or TC "
      "VMEM and VMEM");
}

}  // namespace mlir::tpu
