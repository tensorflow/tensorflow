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

#include "xla/mosaic/dialect/tpu/util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

std::ostream &operator<<(std::ostream &os, Print p) {
  std::string s;
  llvm::raw_string_ostream tmp_os(s);
  p.payload_->print(tmp_os);
  os << tmp_os.str();
  return os;
}

SmallVector<int64_t> ComputeTileStrides(absl::Span<const int64_t> shape,
                                        absl::Span<const int64_t> tiling) {
  SmallVector<int64_t> tile_strides(shape.size());
  int64_t stride = 1;
  for (int64_t i = 0; i < shape.size(); ++i) {
    int64_t idx = shape.size() - 1 - i;
    int64_t tiling_idx = tiling.size() - 1 - i;
    tile_strides[idx] = stride;
    if (tiling_idx >= 0) {
      stride *= llvm::divideCeil(shape[idx], tiling[tiling_idx]);
    } else {
      stride *= shape[idx];
    }
  }
  return tile_strides;
}

FailureOr<SmallVector<int>> computeSqueezedDimsChecked(
    Operation *op, ArrayRef<int64_t> source_shape,
    ArrayRef<int64_t> target_shape) {
  SmallVector<int> squeezed;
  int source_index = source_shape.size() - 1;
  int target_index = target_shape.size() - 1;

  while (source_index >= 0 || target_index >= 0) {
    int64_t target_dim = (target_index >= 0) ? target_shape[target_index] : -1;
    if (source_index < 0) {
      op->emitError() << llvm::formatv(
          "Target shape is not valid. Source: {0}, Target: {1}.",
          shapeToString(source_shape), shapeToString(target_shape));
      return failure();
    }
    int64_t source_dim = source_shape[source_index];
    if (source_dim == target_dim) {
      source_index--;
      target_index--;
    } else {
      if (source_dim != 1) {
        op->emitError() << llvm::formatv(
            "Target shape is not valid. Source: {0}, Target: {1}.",
            shapeToString(source_shape), shapeToString(target_shape));
        return failure();
      }
      squeezed.push_back(source_index);
      source_index--;
    }
  }

  if (source_index != -1 || target_index != -1) {
    op->emitError() << "Shape mismatch after traversal. Source shape: "
                    << shapeToString(source_shape)
                    << ", target shape: " << shapeToString(target_shape);
    return failure();
  }
  std::reverse(squeezed.begin(), squeezed.end());
  return squeezed;
}

std::optional<std::pair<bool, bool>> isTransposedMatmul(
    DotDimensionNumbersAttr dim_numbers) {
  auto lhs_contracting_dims = dim_numbers.getLhsContractingDims();
  auto rhs_contracting_dims = dim_numbers.getRhsContractingDims();
  auto lhs_non_contracting_dims = dim_numbers.getLhsNonContractingDims();
  auto rhs_non_contracting_dims = dim_numbers.getRhsNonContractingDims();

  if (lhs_contracting_dims.size() != 1 || rhs_contracting_dims.size() != 1 ||
      lhs_non_contracting_dims.size() != 1 ||
      rhs_non_contracting_dims.size() != 1) {
    return std::nullopt;
  }

  int64_t lhs_non_contracting_dim = lhs_non_contracting_dims[0];
  int64_t lhs_contracting_dim = lhs_contracting_dims[0];
  int64_t rhs_non_contracting_dim = rhs_non_contracting_dims[0];
  int64_t rhs_contracting_dim = rhs_contracting_dims[0];

  bool lhs_transposed = lhs_non_contracting_dim > lhs_contracting_dim;

  bool rhs_transposed = rhs_contracting_dim > rhs_non_contracting_dim;

  return std::pair<bool, bool>{lhs_transposed, rhs_transposed};
}

bool canReinterpretToUntiledMemref(TypedValue<MemRefType> tiled_memref,
                                   const std::array<int64_t, 2> &target_shape,
                                   bool allow_minormost_padding) {
  MemRefType tiled_memref_ty = tiled_memref.getType();
  auto tiled_layout =
      dyn_cast<tpu::TiledLayoutAttr>(tiled_memref_ty.getLayout());
  ValueRange dynamic_sizes = {};
  if (!tiled_layout) {
    if (auto erase_op = tiled_memref.getDefiningOp<tpu::EraseLayoutOp>()) {
      tiled_memref = erase_op.getOperand();
      tiled_memref_ty = tiled_memref.getType();
      tiled_layout =
          dyn_cast<tpu::TiledLayoutAttr>(tiled_memref_ty.getLayout());
      // TODO(b/375641258): Currently we rely on the pattern `slice ->
      // (squeeze)* -> eraseLayout` to get the dynamic sizes, but other patterns
      // may not work here: eg., slice -> eraseLayout -> reshape ->
      // eraseLayout`. We should fix this! For now, if we can not get the
      // expected dynamic sizes, we consider the memref cannot be reinterpreted
      // to untiled.
      Value ref = tiled_memref;
      while (auto squeeze_op = ref.getDefiningOp<tpu::MemRefSqueezeOp>()) {
        ref = squeeze_op.getInput();
      }
      if (auto slice_op = ref.getDefiningOp<tpu::MemRefSliceOp>()) {
        dynamic_sizes = slice_op.getDynamicSizes();
      }
    }
  }
  if (!tiled_layout) {
    // We expect the tiled memref to have a tiled layout.
    return false;
  }
  if (tiled_memref_ty.getNumDynamicDims() != dynamic_sizes.size()) {
    return false;
  }
  if (tiled_layout.getTiles().empty() ||
      tiled_layout.getTiles().front().dimensions().size() != 2 ||
      tiled_memref_ty.getRank() < 2) {
    // TODO(b/375642202): Currently we only support >= 2D memref, we might
    // need to handle 1D memref if we find a use case.
    return false;
  }
  auto rank = tiled_memref_ty.getRank();
  auto packing = 32 / tiled_memref_ty.getElementTypeBitWidth();
  if (tiled_memref_ty.isDynamicDim(rank - 1)) {
    // TODO(jevinjiang): we can still allow the minormost padding if we know the
    // max bound of the dynamic size is not larger than the target_shape[1].
    if (!isGuaranteedDivisible(dynamic_sizes.back(), target_shape[1])) {
      return false;
    }
    dynamic_sizes = dynamic_sizes.drop_back();
  } else {
    if (!allow_minormost_padding &&
        tiled_memref_ty.getShape()[rank - 1] != target_shape[1]) {
      return false;
    }
  }
  if (tiled_memref_ty.isDynamicDim(rank - 2)) {
    if (!isGuaranteedDivisible(dynamic_sizes.back(), packing)) {
      return false;
    }
  } else {
    if (tiled_memref_ty.getShape()[rank - 2] % packing != 0) {
      return false;
    }
  }
  // Check if the minormost dim has a single tile.
  return *(tiled_layout.getTileStrides().end() - 1) == 1 &&
         *(tiled_layout.getTileStrides().end() - 2) == 1;
}

bool isContiguousMemref(TypedValue<MemRefType> memref) {
  auto memref_ty = getMemRefType(memref);
  if (auto tiled_layout =
          dyn_cast<tpu::TiledLayoutAttr>(memref_ty.getLayout())) {
    auto contiguous_tile_strides = ComputeTileStrides(
        memref_ty, tiled_layout.getTiles().front().dimensions());
    return contiguous_tile_strides == tiled_layout.getTileStrides();
  }
  return true;
}

bool HasMemorySpace(MemRefType ty, tpu::MemorySpace space) {
  auto memory_space =
      dyn_cast_or_null<tpu::MemorySpaceAttr>(ty.getMemorySpace());
  return memory_space && memory_space.getValue() == space;
}

bool layoutIsValidForValue(const Layout &l, const Value v,
                           const std::array<int64_t, 2> target_shape) {
  // l must be non-null iff v is of vector type
  if (const auto vty = dyn_cast<VectorType>(v.getType())) {
    if (!l.has_value()) {
      return false;
    }

    // Vector type should have the same bitwidth as the layout, except for the
    // i1 special case, used for vmasks (see comment for VectorLayout class).
    if (!vty.getElementType().isIntOrFloat()) {
      return false;
    }
    const int8_t bitwidth = vty.getElementTypeBitWidth();
    if (bitwidth != l->bitwidth() && bitwidth != 1) {
      return false;
    }

    return l->isValid(target_shape) && l->layout_rank() <= vty.getRank();
  }
  return !l.has_value();
}

FailureOr<SmallVector<Layout>> getLayoutArrayFromAttr(const Attribute attr) {
  if (const auto array_attr = dyn_cast_if_present<ArrayAttr>(attr)) {
    SmallVector<Layout> out_layouts;
    out_layouts.reserve(array_attr.size());
    for (const Attribute a : array_attr) {
      if (auto layout_attr = dyn_cast_if_present<VectorLayoutAttr>(a)) {
        out_layouts.push_back(layout_attr.getLayout());
      } else {
        return failure();
      }
    }
    return out_layouts;
  }
  return SmallVector<Layout>{};
}

// TODO(tlongeri, jevinjiang): Unify with infer_vector_layout.cc's getOutLayout.
FailureOr<SmallVector<Layout>> getOutLayouts(
    Operation &op, const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> out_layouts,
                             getLayoutArrayFromAttr(op.getAttr("out_layout")));
  if (out_layouts.size() != op.getNumResults()) {
    return op.emitOpError("out_layout size does not match number of results")
           << " results: " << op.getNumResults()
           << " vs layout size: " << out_layouts.size() << " for "
           << op.getName();
  }
  for (const auto [l, res] : llvm::zip_equal(out_layouts, op.getResults())) {
    if (!layoutIsValidForValue(l, res, target_shape)) {
      return op.emitOpError("Invalid output layout");
    }
  }
  return out_layouts;
}

FailureOr<SmallVector<Layout>> getInLayouts(
    Operation &op, const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> in_layouts,
                             getLayoutArrayFromAttr(op.getAttr("in_layout")));
  if (in_layouts.size() != op.getNumOperands()) {
    return op.emitOpError("in_layout size does not match number of operands");
  }
  for (const auto [l, operand] :
       llvm::zip_equal(in_layouts, op.getOperands())) {
    if (!layoutIsValidForValue(l, operand, target_shape)) {
      return op.emitOpError("Invalid input layout");
    }
  }
  return in_layouts;
}

void setInLayout(Operation *op, ArrayRef<Layout> in) {
  CHECK_EQ(in.size(), op->getNumOperands()) << Print(op);
  SmallVector<Attribute, 4> in_attrs;
  in_attrs.reserve(in.size());
  for (const Layout &p : in) {
    in_attrs.push_back(VectorLayoutAttr::get(op->getContext(), p));
  }
  op->setAttr("in_layout", ArrayAttr::get(op->getContext(), in_attrs));
}

void setOutLayout(Operation *op, Layout out) {
  setOutLayout(op, ArrayRef<Layout>(out));
}

void setOutLayout(Operation *op, ArrayRef<Layout> out) {
  SmallVector<Attribute, 4> out_attrs;
  out_attrs.reserve(out.size());
  for (const Layout &p : out) {
    out_attrs.push_back(VectorLayoutAttr::get(op->getContext(), p));
  }
  op->setAttr("out_layout", ArrayAttr::get(op->getContext(), out_attrs));
}

void setLayout(Operation *op, Layout in, Layout out) {
  setLayout(op, ArrayRef<Layout>(in), ArrayRef<Layout>(out));
}

void setLayout(Operation *op, ArrayRef<Layout> in, Layout out) {
  setLayout(op, in, ArrayRef<Layout>(out));
}

void setLayout(Operation *op, Layout in, ArrayRef<Layout> out) {
  setLayout(op, ArrayRef<Layout>(in), out);
}

void setLayout(Operation *op, ArrayRef<Layout> in, ArrayRef<Layout> out) {
  setInLayout(op, in);
  setOutLayout(op, out);
}

std::optional<int64_t> getIntConst(Value v) {
  if (auto const_op = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto cst_attr = dyn_cast<IntegerAttr>(const_op.getValue())) {
      return cst_attr.getValue().getSExtValue();
    }
  }
  return std::nullopt;
}

bool canFoldMinorDimsToSize(ArrayRef<int64_t> shape, int64_t target_size) {
  CHECK_GE(shape.size(), 2);
  int64_t product = shape.back();
  for (int i = shape.size() - 2; i >= 1; --i) {
    product *= shape[i];
    if (product >= target_size) {
      break;
    }
  }
  return product == target_size;
}

SmallVector<Operation *> getNontrivialTransitiveUsers(Value v) {
  auto isUnaryElementwise = [](Operation *op) {
    if (!op->hasTrait<mlir::OpTrait::Elementwise>()) {
      return false;
    }
    return op->getNumOperands() == 1 && op->getNumResults() == 1;
  };
  SmallVector<Operation *> users;
  SmallVector<Value> candidates;
  candidates.push_back(v);
  while (!candidates.empty()) {
    Value candidate = candidates.back();
    candidates.pop_back();
    for (const auto &user : candidate.getUsers()) {
      if (isa<tpu::BitcastOp>(user) || isUnaryElementwise(user))
        candidates.push_back(user->getResult(0));
      else
        users.push_back(user);
    }
  }
  return users;
}

}  // namespace mlir::tpu
