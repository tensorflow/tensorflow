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

#include "xla/backends/gpu/codegen/triton/transforms/tdm_lowering_util.h"

#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/lowering_util.h"
#include "xla/service/decision.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

namespace xtriton = ::xla::gpu::triton;
namespace arith = ::mlir::arith;

SmallVector<int64_t> GetSingletonTileDims(
    ArrayRef<int64_t> tile_sizes, ArrayRef<int64_t> minor_to_major_layout) {
  SmallVector<int64_t> singletons;
  for (int64_t dim : minor_to_major_layout) {
    if (tile_sizes[dim] == 1) {
      singletons.push_back(dim);
    }
  }
  return singletons;
}

::xla::Decision CanUseTdm(bool allow_tdm, ArrayRef<int64_t> original_shape,
                          ArrayRef<int64_t> tile_sizes,
                          ArrayRef<int64_t> tile_strides,
                          ArrayRef<int64_t> minor_to_major_layout,
                          int64_t element_bit_width) {
  if (!allow_tdm) {
    return ::xla::Decision::Forbid("TDM is disabled.");
  }
  // Dynamic sizes or strides would feed sentinel values into the i32/i64
  // descriptor constants and silently produce a malformed descriptor.
  if (mlir::ShapedType::isDynamicShape(tile_sizes) ||
      mlir::ShapedType::isDynamicShape(tile_strides)) {
    return ::xla::Decision::Forbid("dynamic tile sizes or strides.");
  }
  // TDM descriptors describe contiguous boxes; non-unit (or zero) tile strides
  // cannot be expressed and would silently produce a contiguous load.
  for (int64_t s : tile_strides) {
    if (s != 1) {
      return ::xla::Decision::Forbid("non-unit tile stride.");
    }
  }
  // The minor-most surviving (non-singleton) tile dim, after folding all
  // singleton dims into the base pointer, must have global stride 1.
  std::optional<int64_t> minor_survivor;
  for (int64_t dim : minor_to_major_layout) {
    if (tile_sizes[dim] != 1) {
      minor_survivor = dim;
      break;
    }
  }
  if (!minor_survivor.has_value()) {
    return ::xla::Decision::Forbid("all tile dims are singletons.");
  }
  SmallVector<int64_t> original_strides =
      xtriton::ComputeStrides(original_shape, minor_to_major_layout);
  if (original_strides[*minor_survivor] != 1) {
    return ::xla::Decision::Forbid(
        "surviving minor-most dim has global stride != 1.");
  }
  constexpr int64_t kDwordBitWidth = 32;
  constexpr int64_t kMinPadIntervalBits = 2 * kDwordBitWidth;
  if (tile_sizes[*minor_survivor] * element_bit_width < kMinPadIntervalBits) {
    return ::xla::Decision::Forbid(
        "surviving minor-most dim spans fewer than 2 dwords.");
  }
  return ::xla::Decision::Allow();
}

bool TdmAllowed(const ::xla::Decision& decision) {
  if (!decision) {
    VLOG(1) << "Can't use TDM: " << decision.Explain();
  }
  return decision.IsAllowed();
}

TdmDescriptorOperands::TdmDescriptorOperands(Value pointer,
                                             ArrayRef<int64_t> shape,
                                             ArrayRef<int64_t> layout,
                                             ArrayRef<int64_t> sizes,
                                             ValueRange offsets)
    : pointer(pointer),
      shape(shape),
      layout(layout),
      strides(xtriton::ComputeStrides(shape, layout)),
      sizes(sizes),
      offsets(offsets) {}

TdmDescriptorOperands TdmDescriptorOperands::DropSingletonTileDims(
    ImplicitLocOpBuilder& builder, ArrayRef<int64_t> dims_to_drop) const {
  if (dims_to_drop.empty()) {
    return *this;
  }
  int64_t rank = shape.size();

  // Fold each dropped dim's fixed offset into the base pointer, and map each
  // surviving dim to its index in the reduced rank (dropped dims map to -1):
  //   new_ptr = ptr + sum(offset[d] * stride[d]) for d in dims_to_drop.
  SmallVector<int64_t> old_to_new(rank, -1);
  Value element_offset;
  for (int64_t dim = 0, new_index = 0; dim < rank; ++dim) {
    if (!llvm::is_contained(dims_to_drop, dim)) {
      old_to_new[dim] = new_index++;
      continue;
    }
    Value offset_i64 =
        xtriton::IndexCast(builder, builder.getI64Type(), offsets[dim])[0];
    Value stride_i64 = arith::ConstantOp::create(
        builder, builder.getI64IntegerAttr(strides[dim]));
    Value contribution = arith::MulIOp::create(builder, offset_i64, stride_i64);
    element_offset = element_offset ? arith::AddIOp::create(
                                          builder, element_offset, contribution)
                                    : contribution;
  }

  TdmDescriptorOperands result;
  result.pointer =
      AddPtrOp::create(builder, pointer.getType(), pointer, element_offset);
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (old_to_new[dim] < 0) {
      continue;
    }
    result.shape.push_back(shape[dim]);
    result.strides.push_back(strides[dim]);
    result.sizes.push_back(sizes[dim]);
    result.offsets.push_back(offsets[dim]);
  }
  for (int64_t dim : layout) {
    if (old_to_new[dim] >= 0) {
      result.layout.push_back(old_to_new[dim]);
    }
  }
  return result;
}

MakeTensorDescOp BuildTensorDescriptor(ImplicitLocOpBuilder& builder,
                                       const TdmDescriptorOperands& operands) {
  // Global shape as i32 SSA values, in major-to-minor order.
  auto ordered_shape = xtriton::GetMajorToMinorOrder(
      ArrayRef<int64_t>(operands.shape), operands.layout);
  SmallVector<Value> shape_values;
  for (int64_t dim : ordered_shape) {
    shape_values.push_back(
        arith::ConstantOp::create(builder, builder.getI32IntegerAttr(dim)));
  }

  // Global strides as i64 SSA values, in major-to-minor order.
  auto ordered_strides = xtriton::GetMajorToMinorOrder(
      ArrayRef<int64_t>(operands.strides), operands.layout);
  SmallVector<Value> stride_values;
  for (int64_t s : ordered_strides) {
    stride_values.push_back(
        arith::ConstantOp::create(builder, builder.getI64IntegerAttr(s)));
  }

  // Block shape in major-to-minor order as i32.
  auto ordered_sizes = xtriton::GetMajorToMinorOrder(
      ArrayRef<int64_t>(operands.sizes), operands.layout);
  SmallVector<int32_t> block_shape;
  for (int64_t s : ordered_sizes) {
    CHECK_LE(s, INT32_MAX) << "tile dim " << s << " exceeds i32 range";
    block_shape.push_back(static_cast<int32_t>(s));
  }

  auto element_type =
      cast<PointerType>(operands.pointer.getType()).getPointeeType();
  bool is_signed_integer =
      mlir::isa<IntegerType>(element_type) && !element_type.isUnsignedInteger();

  return MakeTensorDescOp::create(builder, operands.pointer, shape_values,
                                  stride_values, block_shape, is_signed_integer,
                                  PaddingOption::PAD_ZERO);
}

}  // namespace mlir::triton::xla
