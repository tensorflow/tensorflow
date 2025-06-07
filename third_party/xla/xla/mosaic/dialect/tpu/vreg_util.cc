/* Copyright 2024 The JAX Authors.

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

#include "absl/log/check.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/array.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

namespace {

VectorType getNativeVregOrVmaskTypeImpl(
    Type elem_ty, const int8_t bitwidth,
    const std::array<int64_t, 2> target_shape) {
  if (bitwidth == 32) {
    return VectorType::get(target_shape, elem_ty);
  }
  return VectorType::get({target_shape[0], target_shape[1], 32 / bitwidth},
                         elem_ty);
}

}  // namespace

VectorType getNativeVregOrVmaskType(Type elem_ty, const int8_t layout_bitwidth,
                                    const std::array<int64_t, 2> target_shape) {
  int8_t bitwidth = elem_ty.getIntOrFloatBitWidth();
  if (bitwidth == 1) {
    bitwidth = layout_bitwidth;
  } else {
    CHECK_EQ(bitwidth, layout_bitwidth);
  }
  return getNativeVregOrVmaskTypeImpl(elem_ty, bitwidth, target_shape);
}

VectorType getNativeVregType(Type elem_ty,
                             const std::array<int64_t, 2> target_shape) {
  return getNativeVregOrVmaskTypeImpl(elem_ty, elem_ty.getIntOrFloatBitWidth(),
                                      target_shape);
}

TypedValue<VectorType> getFullVector(ImplicitLocOpBuilder &builder,
                                     VectorType vty, Attribute value) {
  return cast<TypedValue<VectorType>>(
      builder.create<arith::ConstantOp>(DenseElementsAttr::get(vty, value))
          .getResult());
}

TypedValue<VectorType> getFullLikeVector(ImplicitLocOpBuilder &builder,
                                         TypedValue<VectorType> vec,
                                         Attribute value) {
  return getFullVector(builder, vec.getType(), value);
}

TypedValue<VectorType> getFullVector(OpBuilder &builder, Location loc,
                                     VectorType vty, Attribute value) {
  return cast<TypedValue<VectorType>>(
      builder.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vty, value))
          .getResult());
}

TypedValue<VectorType> getFullLikeVector(OpBuilder &builder, Location loc,
                                         TypedValue<VectorType> vec,
                                         Attribute value) {
  return getFullVector(builder, loc, vec.getType(), value);
}

TypedValue<VectorType> getZerosVector(ImplicitLocOpBuilder &builder,
                                      VectorType vty) {
  return getFullVector(builder, vty, builder.getZeroAttr(vty.getElementType()));
}

TypedValue<VectorType> getZerosLikeVector(ImplicitLocOpBuilder &builder,
                                          TypedValue<VectorType> vec) {
  return getZerosVector(builder, vec.getType());
}

FailureOr<TypedValue<VectorType>> getX32VmaskByPaddingEnd(
    ImplicitLocOpBuilder &builder, int64_t padding,
    const std::array<int64_t, 2> target_shape, int64_t dim) {
  if (dim != 0 && dim != 1) {
    return builder.emitError()
           << "Expected a 2D vector for getX32VmaskByPaddingEnd";
  }

  if (padding < 0 || padding > target_shape[dim]) {
    return builder.emitError()
           << "Padding must be in [0, target_shape[dim]]. Padding: " << padding
           << ", target_shape[dim]: " << target_shape[dim];
  }

  auto idx_const = [&builder](int64_t idx) {
    return IdxConst(idx, builder, builder.getLoc());
  };

  tpu::CreateMaskOp mask_op;
  const VectorType vmask_ty = getNativeVregOrVmaskType(
      builder.getI1Type(), /*layout_bitwidth=*/32, target_shape);
  if (dim == 0) {
    mask_op = builder.create<tpu::CreateMaskOp>(
        vmask_ty, ValueRange{idx_const(0), idx_const(0)},
        ValueRange{idx_const(target_shape[0] - padding),
                   idx_const(target_shape[1])});
  } else {
    mask_op = builder.create<tpu::CreateMaskOp>(
        vmask_ty, ValueRange{idx_const(0), idx_const(0)},
        ValueRange{idx_const(target_shape[0]),
                   idx_const(target_shape[1] - padding)});
  }
  return cast<TypedValue<VectorType>>(mask_op.getResult());
}

LogicalResult maskNativeTilingVregs(ImplicitLocOpBuilder &builder,
                                    xla::Array<Value> &vregs,
                                    std::array<int64_t, 2> target_shape,
                                    int64_t padding_bottom,
                                    int64_t padding_right) {
  auto vreg_ty = dyn_cast<VectorType>(vregs.begin()->getType());
  if (!vreg_ty) {
    return builder.emitError() << "Expected a vector type";
  }

  VectorType i32_vreg_ty =
      getNativeVregType(builder.getI32Type(), target_shape);
  Value i32_zeros_vreg = getZerosVector(builder, i32_vreg_ty);
  Value i32_max_vreg = getFullVector(builder, i32_vreg_ty,
                                     builder.getI32IntegerAttr(0xffffffff));

  int packing = vreg_ty.getRank() > 2 ? vreg_ty.getShape()[2] : 1;
  // Mask out the bottom.
  if (padding_bottom > 0) {
    // The function is only called when the vreg has native tiling. Therefore,
    // it is safe to bitcast to x32 vreg for masking.
    int sub_padding = padding_bottom % packing;
    int x32_padding_bottom = padding_bottom / packing;
    FAILUREOR_ASSIGN_OR_RETURN(
        Value mask_top, getX32VmaskByPaddingEnd(builder, x32_padding_bottom + 1,
                                                target_shape, /*dim=*/0));
    FAILUREOR_ASSIGN_OR_RETURN(
        Value mask_bottom,
        getX32VmaskByPaddingEnd(builder, x32_padding_bottom, target_shape,
                                /*dim=*/0));
    // Create an int32 vreg which contains subelement masking and then
    // logical_and with target vreg to mask out the unaligned paddings.
    // Eg. if padding_bottom = 5, packing = 2, and assume the vreg shape is
    // [8, 128], then the mask will be:
    //
    // sublane 0: [0xffffffff, 0xffffffff, ..., 0xffffffff]
    // sublane 1: [0xffffffff, 0xffffffff, ..., 0xffffffff]
    // sublane 2: [0xffffffff, 0xffffffff, ..., 0xffffffff]
    // sublane 3: [0xffffffff, 0xffffffff, ..., 0xffffffff]
    // sublane 4: [0xffffffff, 0xffffffff, ..., 0xffffffff]
    // sublane 5: [0x0000ffff, 0x0000ffff, ..., 0x0000ffff]
    // sublane 6: [0         , 0         , ..., 0         ]
    // sublane 7: [0         , 0         , ..., 0         ]
    //
    // Through this way, in order to mask sub-elements, each target vreg only
    // needs to apply 1 op (logical_and) instead of 3 ops (unpacking + select
    // + packing).
    Value partial_sublane_mask = getFullVector(
        builder, i32_vreg_ty,
        builder.getI32IntegerAttr(
            0xffffffff >> (sub_padding * vreg_ty.getElementTypeBitWidth())));
    // Insert 0xffffffff above the blended sublane.
    Value sublane_mask = builder.create<arith::SelectOp>(mask_top, i32_max_vreg,
                                                         partial_sublane_mask);
    // Insert 0 below the blended sublane.
    sublane_mask = builder.create<arith::SelectOp>(mask_bottom, sublane_mask,
                                                   i32_zeros_vreg);
    for (int64_t i = 0; i < vregs.dim(1); ++i) {
      Value &vreg = vregs({vregs.dim(0) - 1, i});
      Value i32_vreg = builder.create<tpu::BitcastVregOp>(i32_vreg_ty, vreg);
      if (sub_padding > 0) {
        i32_vreg = builder.create<arith::AndIOp>(i32_vreg, sublane_mask);
      } else {
        i32_vreg = builder.create<arith::SelectOp>(mask_bottom, i32_vreg,
                                                   i32_zeros_vreg);
      }
      vreg = builder.create<tpu::BitcastVregOp>(vreg_ty, i32_vreg);
    }
  }
  // Mask out the right.
  if (padding_right > 0) {
    FAILUREOR_ASSIGN_OR_RETURN(
        Value mask_right, getX32VmaskByPaddingEnd(builder, padding_right,
                                                  target_shape, /*dim=*/1));
    for (int64_t i = 0; i < vregs.dim(0); ++i) {
      Value &vreg = vregs({i, vregs.dim(1) - 1});
      Value i32_vreg = builder.create<tpu::BitcastVregOp>(i32_vreg_ty, vreg);
      i32_vreg =
          builder.create<arith::SelectOp>(mask_right, i32_vreg, i32_zeros_vreg);
      vreg = builder.create<tpu::BitcastVregOp>(vreg_ty, i32_vreg);
    }
  }
  return success();
}

FailureOr<TypedValue<VectorType>> broadcastSubelements(
    ImplicitLocOpBuilder &builder, TypedValue<VectorType> vec,
    int subelement_idx, std::array<int64_t, 2> target_shape) {
  int bitwidth = vec.getType().getElementTypeBitWidth();
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
      builder.create<tpu::BitcastVregOp>(vreg_native_int_ty, vec);
  Value vreg_subelement_low = builder.create<arith::ShRUIOp>(
      src_vreg_int,
      getFullVector(builder, vreg_native_int_ty,
                    builder.getI32IntegerAttr(subelement_idx * bitwidth)));
  SmallVector<Value> packed_vregs(packing, vreg_subelement_low);
  Value vreg_result_int = builder.create<tpu::PackSubelementsOp>(
      vreg_packed_int_ty, packed_vregs, tpu::PackFormat::kInterleaved);
  return cast<TypedValue<VectorType>>(
      builder.create<tpu::BitcastVregOp>(vec.getType(), vreg_result_int)
          .getResult());
}

}  // namespace mlir::tpu
