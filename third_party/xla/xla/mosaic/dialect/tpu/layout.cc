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

#include "xla/mosaic/dialect/tpu/layout.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {
namespace {

mlir::ParseResult parseOffset(StringRef* data, std::optional<int64_t>* result) {
  int64_t int_result;
  if (data->consume_front("*")) {
    *result = std::nullopt;
    return success();
  }
  if (!data->consumeInteger(10, int_result)) {
    *result = int_result;
    return success();
  }
  return failure();
}

std::array<int64_t, 2> nativeTiling(const int8_t bitwidth,
                                    const std::array<int64_t, 2> target_shape) {
  const int packing = 32 / bitwidth;
  return {target_shape[0] * packing, target_shape[1]};
}
}  // namespace

bool TiledRectangularVregBounds::usesAllTiles(
    const std::array<int64_t, 2> target_shape) const {
  return start_offsets_[1] / layout_.tiling()[1] == 0 &&
         llvm::divideCeil(end_offsets_[1], layout_.tiling()[1]) ==
             layout_.tilesPerVreg(target_shape);
}

// See base class.
bool TiledRectangularVregBounds::maskVariesAlong(
    const Direction direction, const std::array<int64_t, 2> target_shape) const
/*override*/ {
  switch (direction) {
    case Direction::kSublanes:
      return !usesAllTiles(target_shape) || start_offsets_[0] != 0 ||
             end_offsets_[0] != layout_.tiling()[0];
    case Direction::kLanes:
      return start_offsets_[1] % layout_.tiling()[1] != 0 ||
             end_offsets_[1] % layout_.tiling()[1] != 0;
    case Direction::kSubelements:
      return start_offsets_[0] % layout_.packing() != 0 ||
             end_offsets_[0] % layout_.packing() != 0;
  }
}

// See base class.
FailureOr<TypedValue<VectorType>> TiledRectangularVregBounds::getVectorMask(
    OpBuilder& builder, const Location loc, const int generation,
    const std::array<int64_t, 2> target_shape) const /*override*/ {
  const int8_t bitwidth = layout_.bitwidth();
  const int packing = layout_.packing();
  const int max_subelems = generation < 4 ? 1 : generation < 5 ? 2 : 4;
  const IntegerType i1 = builder.getI1Type();
  const VectorType mask_vreg_ty = [&]() {
    if (maskVariesAlong(Direction::kSubelements, target_shape)) {
      // When CreateSubelementMask isn't supported, we virtualize masking.
      if (packing > max_subelems) {
        return VectorType::get(target_shape, i1);
      } else {
        return VectorType::get({target_shape[0], target_shape[1], packing}, i1);
      }
    }
    return VectorType::get(target_shape, i1);
  }();
  if (isComplete(target_shape)) {
    return cast<TypedValue<VectorType>>(
        arith::ConstantOp::create(
            builder, loc, mask_vreg_ty,
            DenseElementsAttr::get(mask_vreg_ty, builder.getBoolAttr(true)))
            .getResult());
  }
  Value mask = nullptr;
  const int64_t start_sub = start_offsets_[0] / packing;
  const int64_t end_sub = llvm::divideCeil(end_offsets_[0], packing);
  CHECK_LE(0, start_sub);
  CHECK_LT(start_sub, end_sub);
  CHECK_LE(end_sub, target_shape[0]);
  const int64_t sublanes_per_tile = layout_.sublanesPerTile(target_shape);
  const int64_t start_tile = start_offsets_[1] / layout_.tiling()[1];
  const int64_t end_tile =
      llvm::divideCeil(end_offsets_[1], layout_.tiling()[1]);
  for (int64_t tile = start_tile; tile < end_tile; ++tile) {
    const int64_t sublane_offset = sublanes_per_tile * tile;
    const int64_t row_offset = sublane_offset * layout_.packing();
    const int64_t start_lane =
        tile == start_tile ? start_offsets_[1] % layout_.tiling()[1] : 0;
    const int64_t end_lane =
        tile == end_tile - 1 ? positiveMod(end_offsets_[1], layout_.tiling()[1])
                             : target_shape[1];
    CHECK_LE(0, start_lane);
    CHECK_LT(start_lane, end_lane);
    CHECK_LE(end_lane, target_shape[1]);
    auto boundIdxConst =
        std::bind(IdxConst, std::placeholders::_1, builder, loc);
    // TODO(apaszke): For loads/stores whole sublanes are covered by the sublane
    // mask, so we can focus only on lanes and partial sublanes.
    Value tile_mask = CreateMaskOp::create(
        builder, loc, mask_vreg_ty,
        ValueRange{boundIdxConst(sublane_offset + start_sub),
                   boundIdxConst(start_lane)},
        ValueRange{boundIdxConst(sublane_offset + end_sub),
                   boundIdxConst(end_lane)});
    if (maskVariesAlong(Direction::kSubelements, target_shape)) {
      int64_t start_row = start_offsets_[0] + row_offset;
      int64_t end_row = end_offsets_[0] + row_offset;
      if (packing <= max_subelems) {
        // Only use non-trivial start/end if they don't fall on sublane
        // boundary. Otherwise CreateMaskOp already does the right thing. This
        // lets us use cheaper instruction sequences on TPUv4.
        if (start_offsets_[0] % packing == 0) {
          start_row = 0;
        }
        if (end_offsets_[0] % packing == 0) {
          end_row = target_shape[0] * packing;
        }
        auto submask = tpu::CreateSubelementMaskOp::create(
            builder, loc, mask_vreg_ty, start_row, end_row);
        tile_mask = arith::AndIOp::create(builder, loc, tile_mask, submask);
      } else {  // packing > max_subelems
        const auto getMaskCst = [&](const uint64_t v) {
          const auto int_mask_ty =
              VectorType::get(target_shape, builder.getI32Type());
          return arith::ConstantOp::create(
              builder, loc, int_mask_ty,
              DenseElementsAttr::get(
                  int_mask_ty,
                  builder.getIntegerAttr(builder.getI32Type(), APInt(32, v))));
        };
        tile_mask = arith::SelectOp::create(
            builder, loc, tile_mask, getMaskCst(0xFFFFFFFF), getMaskCst(0));
        if (const int64_t row_in_sublane = start_row % packing;
            row_in_sublane != 0) {
          auto row_mask = tpu::CreateMaskOp::create(
              builder, loc, mask_vreg_ty,
              ValueRange{boundIdxConst(start_row / packing), boundIdxConst(0)},
              ValueRange{boundIdxConst(start_row / packing + 1),
                         boundIdxConst(target_shape[1])});
          auto row_bitmask = arith::SelectOp::create(
              builder, loc, row_mask,
              getMaskCst(0xFFFFFFFF << row_in_sublane * bitwidth),
              getMaskCst(0xFFFFFFFF));
          tile_mask =
              arith::AndIOp::create(builder, loc, tile_mask, row_bitmask);
        }
        if (const int64_t row_in_sublane = end_row % packing;
            row_in_sublane != 0) {
          auto row_mask = tpu::CreateMaskOp::create(
              builder, loc, mask_vreg_ty,
              ValueRange{boundIdxConst(end_row / packing), boundIdxConst(0)},
              ValueRange{boundIdxConst(end_row / packing + 1),
                         boundIdxConst(target_shape[1])});
          auto row_bitmask = arith::SelectOp::create(
              builder, loc, row_mask,
              getMaskCst(0xFFFFFFFFu >> (packing - row_in_sublane) * bitwidth),
              getMaskCst(0xFFFFFFFF));
          tile_mask =
              arith::AndIOp::create(builder, loc, tile_mask, row_bitmask);
        }
      }
    }
    mask = mask == nullptr
               ? tile_mask
               : arith::OrIOp::create(builder, loc, tile_mask, mask);
  }
  CHECK(mask != nullptr);
  return cast<TypedValue<VectorType>>(mask);
}

// See base class
DenseBoolArrayAttr TiledRectangularVregBounds::getSublaneMask(
    MLIRContext* mlir_ctx, const std::array<int64_t, 2> target_shape) const
/*override*/ {
  SmallVector<bool> mask(target_shape[0], false);
  const int64_t start = start_offsets_[0] / layout_.packing();
  const int64_t end = llvm::divideCeil(end_offsets_[0], layout_.packing());
  const int64_t sublanes_per_tile = layout_.sublanesPerTile(target_shape);
  const int64_t start_tile = start_offsets_[1] / layout_.tiling()[1];
  const int64_t end_tile =
      llvm::divideCeil(end_offsets_[1], layout_.tiling()[1]);
  for (int64_t tile = start_tile; tile < end_tile; ++tile) {
    for (int64_t i = start; i < end; ++i) {
      CHECK(!mask[tile * sublanes_per_tile + i]);
      mask[tile * sublanes_per_tile + i] = true;
    }
  }
  return DenseBoolArrayAttr::get(mlir_ctx, mask);
}

// Total number of entries contained in a vreg.
int64_t SingleRowVRegBounds::getEntriesPerVreg(
    const std::array<int64_t, 2> target_shape) const {
  return target_shape[0] * target_shape[1] * layout_.packing();
}

// See base class.
bool SingleRowVRegBounds::maskVariesAlong(
    const Direction direction, const std::array<int64_t, 2> target_shape) const
/*override*/ {
  if (start_offset_ == 0 && stop_offset_ == getEntriesPerVreg(target_shape)) {
    return false;
  }
  const int64_t entries_per_vreg = getEntriesPerVreg(target_shape);
  switch (direction) {
    case Direction::kSublanes:
      return start_offset_ >= target_shape[1] ||
             stop_offset_ < entries_per_vreg - target_shape[1];
    case Direction::kLanes:
      return true;
    // This is very different from the definition in the 2d tiling case
    // `TiledRectangularVregBounds` where we only need to check whether the
    // offsets are divisible by packing. Here in the 1d tiling case, the data
    // is still vertically packed, so as long as the start and stop offsets
    // are not aligned to sublanes, subelement masking is required. For
    // example, consider a 16-bit vector with `start_offset_ = 0` and
    // `stop_offset_ = 128`, the mask should be true for the lower 16 bits and
    // false for the upper 16 bits across all lanes.
    case Direction::kSubelements:
      return layout_.bitwidth() != 32 &&
             (start_offset_ % (target_shape[1] * layout_.packing()) != 0 ||
              stop_offset_ % (target_shape[1] * layout_.packing()) != 0);
  }
}

// See base class.
FailureOr<TypedValue<VectorType>> SingleRowVRegBounds::getVectorMask(
    OpBuilder& builder, const Location loc, const int generation,
    const std::array<int64_t, 2> target_shape) const /*override*/ {
  // Only packed types may require subelement masking.
  if (maskVariesAlong(Direction::kSubelements, target_shape)) {
    if (layout_.bitwidth() != 16 && layout_.bitwidth() != 8) {
      return emitError(loc,
                       "Only 16-bit and 8-bit subelement masking is currently "
                       "implemented in SingleRowVRegBounds::getVectorMask.");
    }

    const auto i16_vreg = VectorType::get({target_shape[0], target_shape[1], 2},
                                          builder.getI16Type());
    const auto getI16VregConstant = [&](const int32_t v) {
      return arith::ConstantOp::create(
          builder, loc, i16_vreg,
          DenseElementsAttr::get(i16_vreg, builder.getI16IntegerAttr(v)));
    };
    const Value start = getI16VregConstant(start_offset_);
    const Value end = getI16VregConstant(stop_offset_);
    const Value i16_iota =
        tpu::IotaOp::create(builder, loc, i16_vreg, ArrayRef<int32_t>{0, 2, 1});

    auto generate_mask = [&](Value iota) {
      return cast<TypedValue<VectorType>>(
          arith::AndIOp::create(
              builder, loc,
              arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                    iota, start),
              arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                    iota, end))
              .getResult());
    };

    // Handle 16-bit.
    if (layout_.bitwidth() == 16) {
      if (generation < 4) {
        return emitError(
            loc, "16-bit subelement masking is only supported for TPU v4+.");
      }
      return generate_mask(i16_iota);
    }

    // Handle 8-bit by packing two 16-bit masks together.
    if (generation < 5) {
      return emitError(
          loc, "8-bit subelement masking is only supported for TPU v5+.");
    }
    int32_t elements_per_vreg_16_bit = target_shape[0] * target_shape[1] * 2;
    CHECK(llvm::isPowerOf2_64(elements_per_vreg_16_bit));
    // `i16_iota` generates data from 0 to `elements_per_vreg_16_bit - 1`, so
    // `i16_iota_shifted` should generate data from `elements_per_vreg_16_bit`
    // to `2 * elements_per_vreg_16_bit - 1`. Here, we use a trick to avoid
    // using i16 addition which only works for TPU v6+, a simple OR operation
    // will suffice.
    Value i16_iota_shifted = arith::OrIOp::create(
        builder, loc, i16_iota, getI16VregConstant(elements_per_vreg_16_bit));
    Value mask1 = generate_mask(i16_iota);
    Value mask2 = generate_mask(i16_iota_shifted);

    auto mask_vreg_ty =
        VectorType::get({target_shape[0], target_shape[1], layout_.packing()},
                        builder.getI1Type());
    SmallVector<Value> masks_to_pack = {mask1, mask2};
    Value final_mask =
        PackMaskOp::create(builder, loc, mask_vreg_ty, masks_to_pack);

    return cast<TypedValue<VectorType>>(final_mask);
  }

  // Handle 32-bit as well as packed types with sublane-aligned offsets.
  const auto i32_vreg = VectorType::get(target_shape, builder.getI32Type());
  const auto getI32VregConstant = [&](const int32_t v) {
    return arith::ConstantOp::create(builder, loc, i32_vreg,
                                     DenseElementsAttr::get(i32_vreg, v));
  };
  const Value start = getI32VregConstant(start_offset_ / layout_.packing());
  const Value end = getI32VregConstant(stop_offset_ / layout_.packing());
  const Value iota =
      tpu::IotaOp::create(builder, loc, i32_vreg, ArrayRef<int32_t>{0, 1});
  return cast<TypedValue<VectorType>>(
      arith::AndIOp::create(
          builder, loc,
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge, iota,
                                start),
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, iota,
                                end))
          .getResult());
}

// See base class.
DenseBoolArrayAttr SingleRowVRegBounds::getSublaneMask(
    MLIRContext* mlir_ctx, const std::array<int64_t, 2> target_shape) const
/*override*/ {
  const int64_t start_sublane =
      start_offset_ / layout_.packing() / target_shape[1];
  const int64_t end_sublane = llvm::divideCeil(
      llvm::divideCeil(stop_offset_, layout_.packing()), target_shape[1]);

  SmallVector<bool> sublane_mask(target_shape[0], false);
  for (int64_t i = start_sublane; i < end_sublane; ++i) {
    sublane_mask[i] = true;
  }
  return DenseBoolArrayAttr::get(mlir_ctx, sublane_mask);
}

std::tuple<std::optional<int64_t>, std::optional<int64_t>, int64_t, int64_t,
           int8_t, VectorLayout::ImplicitDim>
VectorLayout::as_tuple() const {
  return std::make_tuple(offsets_[0], offsets_[1], tiling_[0], tiling_[1],
                         bitwidth_, implicit_dim_);
}

bool VectorLayout::operator==(const VectorLayout& other) const {
  return as_tuple() == other.as_tuple();
}

bool VectorLayout::hasNativeTiling(
    const std::array<int64_t, 2> target_shape) const {
  return tiling_ == nativeTiling(bitwidth_, target_shape);
}

SmallVector<int64_t> VectorLayout::implicitShape(
    ArrayRef<int64_t> shape) const {
  SmallVector<int64_t> implicit_shape(shape);
  implicit_shape.reserve(shape.size() + num_implicit_dims());
  insertImplicit<int64_t>(implicit_shape, 1);
  return implicit_shape;
}

SmallVector<int64_t> VectorLayout::tileArrayShape(
    const bool src_is_implicit, const bool res_is_implicit,
    SmallVector<int64_t>&& src_shape,
    const std::array<int64_t, 2> target_shape) const {
  const std::array<int64_t, 2> vreg_slice = vregSlice(target_shape);
  if (!src_is_implicit) {
    CHECK_GE(src_shape.size(), layout_rank());
    insertImplicit<int64_t>(src_shape, 1);
  }
  int64_t& second_minor = *(src_shape.end() - 2);
  int64_t& minor = *(src_shape.end() - 1);
  second_minor =
      offsets_[0].has_value()
          ? llvm::divideCeil(*offsets_[0] + second_minor, vreg_slice[0])
          : 1;
  minor = offsets_[1].has_value()
              ? llvm::divideCeil(*offsets_[1] + minor, vreg_slice[1])
              : 1;
  if (!res_is_implicit) {
    CHECK_GE(src_shape.size(), 2);
    eraseImplicit(src_shape);
  }
  return std::move(src_shape);
}

std::unique_ptr<VRegDataBounds> VectorLayout::tileDataBounds(
    MLIRContext* mlir_ctx, const ArrayRef<int64_t> full_shape,
    const ArrayRef<int64_t> idxs, const std::array<int64_t, 2> target_shape,
    const std::array<bool, 2> allow_replicated /*= {false, false}*/) const {
  // TODO(apaszke): allow_replicated could have been generalized to specify
  // what action should be taken when a REPLICATED offset is encountered.
  // Right now it either disallows replication, or selects the whole dimension.
  for (const int i : {0, 1}) {
    if (!allow_replicated[i] && !offsets_[i].has_value()) {
      emitError(UnknownLoc::get(mlir_ctx), "Unexpected replicated offset");
      return nullptr;
    }
  }
  const std::array<int64_t, 2> vreg_slice = vregSlice(target_shape);
  const std::array<int64_t, 2> tiled_idxs = getImplicitTiledDims(idxs, 0);
  const int64_t s = tiled_idxs[0];
  const int64_t l = tiled_idxs[1];
  const SmallVector<int64_t> tiles_implicit_shape =
      tileArrayImplicitShape(full_shape, target_shape);
  const int64_t ns = *(tiles_implicit_shape.end() - 2);
  const int64_t nl = *(tiles_implicit_shape.end() - 1);
  const std::array<int64_t, 2> tiled_ishape =
      getImplicitTiledDims(full_shape, 1);
  // The starts and ends of the data within the vreg slice:
  const std::array<int64_t, 2> starts = {
      offsets_[0] && s == 0 ? *offsets_[0] : 0,
      offsets_[1] && l == 0 ? *offsets_[1] : 0};
  const std::array<int64_t, 2> ends = {
      offsets_[0] && s == ns - 1
          ? positiveMod(*offsets_[0] + tiled_ishape[0], vreg_slice[0])
          : vreg_slice[0],
      offsets_[1] && l == nl - 1
          ? positiveMod(*offsets_[1] + tiled_ishape[1], vreg_slice[1])
          : vreg_slice[1]};

  if (tiling_[0] == 1 && tiling_[1] % target_shape[1] == 0) {
    return std::make_unique<SingleRowVRegBounds>(*this, starts[1], ends[1],
                                                 target_shape);
  }
  if (tiling_[1] != target_shape[1]) {
    emitError(UnknownLoc::get(mlir_ctx),
              "Not implemented: Unaligned tiling on minormost dimension");
    return nullptr;
  }
  return std::make_unique<TiledRectangularVregBounds>(*this, starts, ends,
                                                      target_shape);
}

bool VectorLayout::generalizes(
    const VectorLayout& other, ArrayRef<int64_t> shape,
    const std::array<int64_t, 2> target_shape) const {
  if (bitwidth_ != other.bitwidth_) {
    return false;
  }
  for (auto [s, o] : llvm::zip(offsets_, other.offsets_)) {
    if (s.has_value() && s != o) {
      return false;
    }
  }
  if (implicit_dim_ != other.implicit_dim_) {
    // Don't fail yet!
    if (tiling_[0] == 1 && other.tiling_[0] == 1 &&
        ((implicit_dim_ == ImplicitDim::kSecondMinor &&
          other.implicit_dim_ == ImplicitDim::kNone) ||
         (implicit_dim_ == ImplicitDim::kNone &&
          other.implicit_dim_ == ImplicitDim::kSecondMinor))) {
      // If the tiling is (1, n), we can always squeeze an implicit 2nd minor
      // dimension without having to combine vregs.
    } else {
      if (shape.data() == nullptr) {
        return false;
      }
      // Since we do not reorder axes, if the shapes resulting from inserting
      // implicit dimensions are the same in the 2 minormost dimensions for both
      // layouts, then the elements must be laid out the same way (before
      // tiling).
      if (getImplicitTiledDims(shape, 1) !=
          other.getImplicitTiledDims(shape, 1)) {
        return false;
      }
    }
  }
  if (tiling_ != other.tiling_) {
    // Don't fail yet!
    // If there is only one tile in both tilings, then they are equivalent.
    if (shape.data() == nullptr) {
      return false;
    }

    // We can assume the implicit shape is the same for both layouts. They are
    // only allowed to be different when both tilings are equal to (1, n) (and
    // each other), and we've checked that tilings are different above.
    const std::array<int64_t, 2> ishape_tiled_dims =
        getImplicitTiledDims(shape, 1);
    if (!(tiling_[1] == other.tiling_[1] && tiling_[1] == target_shape[1] &&
          offsets_[1].value_or(0) + ishape_tiled_dims[1] <= target_shape[1] &&
          offsets_[0].value_or(0) + ishape_tiled_dims[0] <=
              std::min(tiling_[0], other.tiling_[0]))) {
      return false;
    }
  }
  return true;
}

template <typename Stream>
Stream& printImplicitDim(Stream& os, VectorLayout::ImplicitDim dim) {
  switch (dim) {
    case VectorLayout::ImplicitDim::kNone:
      os << "none";
      break;
    case VectorLayout::ImplicitDim::kMinor:
      os << "-1";
      break;
    case VectorLayout::ImplicitDim::kSecondMinor:
      os << "-2";
      break;
    case VectorLayout::ImplicitDim::kMinorAndSecondMinor:
      os << "-2,-1";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, VectorLayout::ImplicitDim dim) {
  return printImplicitDim(os, dim);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              VectorLayout::ImplicitDim dim) {
  return printImplicitDim(os, dim);
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag,
                             VectorLayout::ImplicitDim dim) {
  return printImplicitDim(diag, dim);
}

template <typename Stream>
static void printVectorLayout(Stream& os, const int32_t bitwidth,
                              const VectorLayout::ImplicitDim implicit_dim,
                              const LayoutOffsets offsets,
                              const std::array<int64_t, 2>& tiling) {
  os << static_cast<int32_t>(bitwidth) << ",{";
  bool first = true;
  for (auto o : offsets) {
    if (first) {
      first = false;
    } else {
      os << ',';
    }
    if (!o) {
      os << '*';
    } else {
      os << *o;
    }
  }
  os << "},(" << tiling[0] << ',' << tiling[1] << ")";
  if (implicit_dim != VectorLayout::ImplicitDim::kNone) {
    os << "," << implicit_dim;
  }
}

void VectorLayout::print(llvm::raw_ostream& os) const {
  printVectorLayout(os, bitwidth_, implicit_dim_, offsets_, tiling_);
}

void VectorLayout::print(std::ostream& os) const {
  printVectorLayout(os, bitwidth_, implicit_dim_, offsets_, tiling_);
}

void VectorLayout::print(mlir::Diagnostic& diag) const {
  printVectorLayout(diag, bitwidth_, implicit_dim_, offsets_, tiling_);
}

std::optional<VectorLayout> VectorLayout::join(const VectorLayout& l,
                                               const VectorLayout& r,
                                               ArrayRef<int64_t> shape) {
  auto is_fully_replicated = [&](const VectorLayout& layout) {
    const LayoutOffsets& offsets = layout.getCanonicalOffsets(shape);
    return !offsets[0] && !offsets[1];
  };
  if (is_fully_replicated(l) && l.layout_rank() >= r.layout_rank()) {
    return r;
  }
  if (is_fully_replicated(r) && r.layout_rank() >= l.layout_rank()) {
    return l;
  }
  if (l.bitwidth_ != r.bitwidth_ || l.tiling_ != r.tiling_) {
    return std::nullopt;
  }
  if (l.getImplicitTiledDims(shape, 1) != r.getImplicitTiledDims(shape, 1)) {
    return std::nullopt;
  }
  LayoutOffsets offsets;
  for (int i = 0; i < 2; ++i) {
    auto lo = l.offsets()[i];
    auto ro = r.offsets()[i];
    if (lo && ro && lo != ro) {
      return std::nullopt;
    }
    offsets[i] = lo.has_value() ? lo : ro;
  }
  return VectorLayout(l.bitwidth_, offsets, l.tiling_, l.implicit_dim_);
}

std::optional<VectorLayout> VectorLayout::parse(StringRef* data) {
  StringRef local(*data);
  int8_t bitwidth;
  LayoutOffsets offsets;
  std::array<int64_t, 2> tiling;
  ImplicitDim implicit_dim = ImplicitDim::kNone;
  if (local.consumeInteger(10, bitwidth) || !local.consume_front(",{") ||
      parseOffset(&local, &offsets[0]) || !local.consume_front(",") ||
      parseOffset(&local, &offsets[1]) || !local.consume_front("},(") ||
      local.consumeInteger(10, tiling[0]) || !local.consume_front(",") ||
      local.consumeInteger(10, tiling[1]) || !local.consume_front(")")) {
    return std::nullopt;
  }
  if (local.consume_front(",-2,-1")) {
    implicit_dim = ImplicitDim::kMinorAndSecondMinor;
  } else if (local.consume_front(",-1")) {
    implicit_dim = ImplicitDim::kMinor;
  } else if (local.consume_front(",-2")) {
    implicit_dim = ImplicitDim::kSecondMinor;
  }
  *data = local;
  return VectorLayout(bitwidth, offsets, tiling, implicit_dim);
}

namespace {
template <class>
inline constexpr bool false_v = false;

template <typename Stream>
Stream& printLayout(Stream& os, const Layout& v) {
  os << '"';
  if (v.has_value()) {
    v->print(os);
  } else {
    os << "none";
  }
  os << '"';
  return os;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Layout& v) {
  return printLayout<std::ostream>(os, v);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Layout& v) {
  return printLayout<llvm::raw_ostream>(os, v);
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Layout& v) {
  return printLayout<mlir::Diagnostic>(diag, v);
}

llvm::hash_code hash_value(const VectorLayout& layout) {
  return llvm::hash_value(layout.as_tuple());
}

std::optional<Layout> parseLayout(mlir::AsmParser& parser) {
  std::string layout_str;
  if (failed(parser.parseString(&layout_str))) {
    return std::nullopt;
  }
  if (layout_str == "none") {
    return kNoLayout;
  }
  StringRef ref(layout_str);
  if (auto layout = VectorLayout::parse(&ref); ref.empty()) {
    return *layout;
  }
  return std::nullopt;
}

const Layout kNoLayout = std::nullopt;

}  // namespace mlir::tpu
