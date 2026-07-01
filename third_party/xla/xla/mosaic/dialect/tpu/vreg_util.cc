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

#include "xla/mosaic/dialect/tpu/vreg_util.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"
#include "xla/array.h"

namespace mlir::tpu {

namespace {

VectorType getNativeVregOrVmaskTypeImpl(Type elem_ty, const int8_t bitwidth,
                                        ArrayRef<int64_t> target_shape) {
  if (bitwidth == 32) {
    return VectorType::get(target_shape, elem_ty);
  }
  SmallVector<int64_t> shape(target_shape);
  shape.push_back(32 / bitwidth);
  return VectorType::get(shape, elem_ty);
}

}  // namespace

VectorType getNativeVregOrVmaskType(Type elem_ty, const int8_t layout_bitwidth,
                                    ArrayRef<int64_t> target_shape) {
  int8_t bitwidth = getTypeBitwidth(elem_ty);
  if (bitwidth == 1) {
    bitwidth = layout_bitwidth;
  } else {
    CHECK_EQ(bitwidth, layout_bitwidth);
  }
  return getNativeVregOrVmaskTypeImpl(elem_ty, bitwidth, target_shape);
}

VectorType getNativeVregType(Type elem_ty, ArrayRef<int64_t> target_shape) {
  return getNativeVregOrVmaskTypeImpl(elem_ty, getTypeBitwidth(elem_ty),
                                      target_shape);
}

TypedValue<VectorType> getFullVector(ImplicitLocOpBuilder& builder,
                                     VectorType vty, Attribute value) {
  return cast<TypedValue<VectorType>>(
      arith::ConstantOp::create(builder, DenseElementsAttr::get(vty, value))
          .getResult());
}

TypedValue<VectorType> getFullLikeVector(ImplicitLocOpBuilder& builder,
                                         TypedValue<VectorType> vec,
                                         Attribute value) {
  return getFullVector(builder, vec.getType(), value);
}

TypedValue<VectorType> getFullVector(OpBuilder& builder, Location loc,
                                     VectorType vty, Attribute value) {
  return cast<TypedValue<VectorType>>(
      arith::ConstantOp::create(builder, loc,
                                DenseElementsAttr::get(vty, value))
          .getResult());
}

TypedValue<VectorType> getFullLikeVector(OpBuilder& builder, Location loc,
                                         TypedValue<VectorType> vec,
                                         Attribute value) {
  return getFullVector(builder, loc, vec.getType(), value);
}

TypedValue<VectorType> getZerosVector(ImplicitLocOpBuilder& builder,
                                      VectorType vty) {
  return getFullVector(builder, vty, builder.getZeroAttr(vty.getElementType()));
}

TypedValue<VectorType> getZerosLikeVector(ImplicitLocOpBuilder& builder,
                                          TypedValue<VectorType> vec) {
  return getZerosVector(builder, vec.getType());
}

LogicalResult maskTiledVregs(ImplicitLocOpBuilder& builder,
                             xla::Array<Value>& vregs,
                             std::array<int64_t, 2> target_shape,
                             std::array<int64_t, 2> tiling,
                             int64_t padding_bottom, int64_t padding_right,
                             int generation) {
  auto vreg_ty = dyn_cast<VectorType>(vregs.begin()->getType());
  if (!vreg_ty) {
    return builder.emitError() << "Expected a vector type";
  }
  if (vregs.num_dimensions() < 2) {
    return builder.emitError() << "Vregs must have at least two dimensions";
  }

  const llvm::SmallVector<int64_t> orig_vregs_dims(vregs.dimensions().begin(),
                                                   vregs.dimensions().end());
  int64_t num_dims = orig_vregs_dims.size();
  // Flatten untiled dimensions.
  vregs.Reshape(
      {llvm::product_of(
           ArrayRef<int64_t>(orig_vregs_dims).take_front(num_dims - 2)),
       orig_vregs_dims[num_dims - 2], orig_vregs_dims[num_dims - 1]});

  VectorLayout layout(getElementTypeBitwidth(vreg_ty), LayoutOffsets{0, 0},
                      tiling);
  std::array<int64_t, 2> vreg_slice = layout.vregSlice(target_shape);
  bool is_1d_tiling = tiling[0] == 1;

  auto make_data_bounds =
      [&](int64_t dim,
          int64_t padding) -> FailureOr<std::unique_ptr<VRegDataBounds>> {
    if (vreg_slice[dim] <= padding) {
      return builder.emitError()
             << "Padding must be less than the vreg slice. Padding: " << padding
             << ", vreg_slice[" << dim << "]: " << vreg_slice[dim];
    }

    std::unique_ptr<VRegDataBounds> bounds;
    if (is_1d_tiling) {
      int64_t end_offset = vreg_slice[dim] - padding;
      bounds = std::make_unique<SingleRowVRegBounds>(layout, /*start_offset=*/0,
                                                     /*end_offset=*/end_offset,
                                                     target_shape);
    } else {
      const std::array<int64_t, 2> start_offsets = {0, 0};
      std::array<int64_t, 2> end_offsets = vreg_slice;
      end_offsets[dim] -= padding;
      bounds = std::make_unique<TiledRectangularVregBounds>(
          layout, start_offsets, end_offsets, target_shape);
    }
    return std::move(bounds);
  };

  DCHECK_EQ(vregs.num_dimensions(), 3);
  TypedValue<VectorType> zeros_vreg = getZerosVector(builder, vreg_ty);
  // Mask out the bottom.
  if (padding_bottom > 0) {
    FAILUREOR_ASSIGN_OR_RETURN(auto bounds,
                               make_data_bounds(/*dim=*/0, padding_bottom));
    for (int64_t untiled_idx = 0; untiled_idx < vregs.dim(0); ++untiled_idx) {
      for (int64_t i = 0; i < vregs.dim(2); ++i) {
        Value& vreg = vregs({untiled_idx, vregs.dim(1) - 1, i});
        FAILUREOR_ASSIGN_OR_RETURN(
            vreg, selectWithBounds(builder, *bounds,
                                   cast<TypedValue<VectorType>>(vreg),
                                   zeros_vreg, target_shape, generation));
      }
    }
  }
  // Mask out the right.
  if (padding_right > 0) {
    FAILUREOR_ASSIGN_OR_RETURN(auto bounds,
                               make_data_bounds(/*dim=*/1, padding_right));
    for (int64_t untiled_idx = 0; untiled_idx < vregs.dim(0); ++untiled_idx) {
      for (int64_t i = 0; i < vregs.dim(1); ++i) {
        Value& vreg = vregs({untiled_idx, i, vregs.dim(2) - 1});
        FAILUREOR_ASSIGN_OR_RETURN(
            vreg, selectWithBounds(builder, *bounds,
                                   cast<TypedValue<VectorType>>(vreg),
                                   zeros_vreg, target_shape, generation));
      }
    }
  }

  // Reshape back to original dimensions.
  vregs.Reshape(orig_vregs_dims);
  return success();
}

FailureOr<TypedValue<VectorType>> selectWithBounds(
    ImplicitLocOpBuilder& builder, const VRegDataBounds& bounds,
    TypedValue<VectorType> in_bounds_vreg,
    TypedValue<VectorType> out_of_bounds_vreg,
    std::array<int64_t, 2> target_shape, int generation) {
  auto native_vreg_ty = getNativeVregType(
      in_bounds_vreg.getType().getElementType(), target_shape);
  TPU_ASSERT_LOC(builder.getLoc(), in_bounds_vreg.getType() == native_vreg_ty);
  TPU_ASSERT_LOC(builder.getLoc(),
                 out_of_bounds_vreg.getType() == native_vreg_ty);
  if (bounds.isComplete(target_shape)) {
    return in_bounds_vreg;
  }
  FAILUREOR_ASSIGN_OR_RETURN(TypedValue<VectorType> vmask,
                             bounds.getVectorMask(builder, builder.getLoc(),
                                                  generation, target_shape));
  VectorType vmask_ty = vmask.getType();
  if (vmask_ty.getElementType() == builder.getI32Type()) {
    // Handle bitmask.
    VectorType i32_vreg_ty =
        getNativeVregType(builder.getI32Type(), target_shape);
    Value i32_in_bounds_vreg =
        tpu::BitcastVregOp::create(builder, i32_vreg_ty, in_bounds_vreg);
    Value i32_out_of_bounds_vreg =
        tpu::BitcastVregOp::create(builder, i32_vreg_ty, out_of_bounds_vreg);
    // See https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
    //
    // r = a if mask = 0; or b if mask = 1
    //   = a ^ ((a ^ b) & mask)
    //
    // Note that when a is 0, compilers fold the expression to b & mask.
    Value masked_i32_vreg = arith::XOrIOp::create(
        builder, builder.getLoc(), i32_out_of_bounds_vreg,
        arith::AndIOp::create(
            builder, builder.getLoc(),
            arith::XOrIOp::create(builder, builder.getLoc(),
                                  i32_out_of_bounds_vreg, i32_in_bounds_vreg),
            vmask));
    return tpu::BitcastVregOp::create(builder, native_vreg_ty, masked_i32_vreg)
        .getResult();
  }

  DCHECK_EQ(vmask_ty.getElementType(), builder.getI1Type());
  const bool needs_bitcast = vmask_ty.getShape() != native_vreg_ty.getShape();
  if (needs_bitcast) {
    DCHECK_GE(vmask_ty.getRank(), 2);
    const int mask_packing =
        vmask_ty.getRank() == 2 ? 1 : vmask_ty.getDimSize(2);
    const int mask_bitwidth = 32 / mask_packing;
    VectorType bitcast_ty = VectorType::get(
        vmask_ty.getShape(), builder.getIntegerType(mask_bitwidth));
    in_bounds_vreg =
        tpu::BitcastVregOp::create(builder, bitcast_ty, in_bounds_vreg);
    out_of_bounds_vreg =
        tpu::BitcastVregOp::create(builder, bitcast_ty, out_of_bounds_vreg);
  }
  Value result = arith::SelectOp::create(builder, vmask, in_bounds_vreg,
                                         out_of_bounds_vreg);
  if (needs_bitcast) {
    result = tpu::BitcastVregOp::create(builder, native_vreg_ty, result);
  }
  return cast<TypedValue<VectorType>>(result);
}

FailureOr<TypedValue<VectorType>> broadcastSubelements(
    ImplicitLocOpBuilder& builder, TypedValue<VectorType> vec,
    int subelement_idx, ArrayRef<int64_t> target_shape) {
  int bitwidth = getElementTypeBitwidth(vec.getType());
  int packing = 32 / bitwidth;
  if (subelement_idx < 0 || subelement_idx >= packing) {
    return builder.emitError()
           << "subelement_idx must be in [0, packing). subelement_idx: "
           << subelement_idx << ", packing: " << packing;
  }
  if (packing == 1) {
    return vec;
  }
  VectorType vreg_native_int_ty =
      getNativeVregType(builder.getIntegerType(32), target_shape);
  VectorType vreg_packed_int_ty =
      getNativeVregType(builder.getIntegerType(bitwidth), target_shape);
  // The chosen subelements must be in the low bits. High bits are unspecified.
  Value src_vreg_int =
      tpu::BitcastVregOp::create(builder, vreg_native_int_ty, vec);
  Value vreg_subelement_low = arith::ShRUIOp::create(
      builder, src_vreg_int,
      getFullVector(builder, vreg_native_int_ty,
                    builder.getI32IntegerAttr(subelement_idx * bitwidth)));
  SmallVector<Value> packed_vregs(packing, vreg_subelement_low);
  Value vreg_result_int = tpu::PackSubelementsOp::create(
      builder, vreg_packed_int_ty, packed_vregs, tpu::PackFormat::kInterleaved);
  return cast<TypedValue<VectorType>>(
      tpu::BitcastVregOp::create(builder, vec.getType(), vreg_result_int)
          .getResult());
}

}  // namespace mlir::tpu
