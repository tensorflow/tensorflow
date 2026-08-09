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

#include "xla/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/array.h"
#include "xla/mosaic/dialect/tpu/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/apply_vector_layout.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"

// TODO(tlongeri): null pointer checks?

namespace {
DEFINE_C_API_PTR_METHODS(MlirTpuVectorLayout, mlir::tpu::VectorLayout);
DEFINE_C_API_PTR_METHODS(MlirTpuVregDataBounds, mlir::tpu::VRegDataBounds);

MlirTpuImplicitDim wrap(mlir::tpu::VectorLayout::ImplicitDim implicit_dim) {
  switch (implicit_dim) {
    case mlir::tpu::VectorLayout::ImplicitDim::kNone:
      return MlirTpuImplicitDimNone;
    case mlir::tpu::VectorLayout::ImplicitDim::kMinor:
      return MlirTpuImplicitDimMinor;
    case mlir::tpu::VectorLayout::ImplicitDim::kSecondMinor:
      return MlirTpuImplicitDimSecondMinor;
  }
  LOG(FATAL) << "Invalid implicit dim (C++)";
}
mlir::tpu::VectorLayout::ImplicitDim unwrap(MlirTpuImplicitDim implicit_dim) {
  switch (implicit_dim) {
    case MlirTpuImplicitDimNone:
      return mlir::tpu::VectorLayout::ImplicitDim::kNone;
    case MlirTpuImplicitDimMinor:
      return mlir::tpu::VectorLayout::ImplicitDim::kMinor;
    case MlirTpuImplicitDimSecondMinor:
      return mlir::tpu::VectorLayout::ImplicitDim::kSecondMinor;
  }
  LOG(FATAL) << "Invalid implicit dim (C)";
}
mlir::tpu::Direction unwrap(MlirTpuDirection direction) {
  switch (direction) {
    case MlirTpuDirectionSublanes:
      return mlir::tpu::Direction::kSublanes;
    case MlirTpuImplicitDimMinor:
      return mlir::tpu::Direction::kLanes;
    case MlirTpuImplicitDimSecondMinor:
      return mlir::tpu::Direction::kSubelements;
  }
  LOG(FATAL) << "Invalid direction (C)";
}
MlirTpuLayoutOffsets wrap(mlir::tpu::LayoutOffsets offsets) {
  return {offsets[0].value_or(-1), offsets[1].value_or(-1)};
}
mlir::tpu::LayoutOffsets unwrap(MlirTpuLayoutOffsets offsets) {
  auto translateOffset = [](int64_t offset) {
    CHECK_GE(offset, -1);
    return offset == -1 ? std::nullopt : mlir::tpu::LayoutOffset{offset};
  };
  return {translateOffset(offsets.sublane), translateOffset(offsets.lane)};
}
std::array<bool, 2> unwrap(MlirTpuBoolTargetTuple arr) {
  return {arr.sublane, arr.lane};
}
std::array<int64_t, 2> unwrap(MlirTpuI64TargetTuple arr) {
  return {arr.sublane, arr.lane};
}
MlirTpuI64TargetTuple wrap(std::array<int64_t, 2> arr) {
  return {arr[0], arr[1]};
}
mlir::tpu::ApplyVectorLayoutContext unwrap(
    MlirTpuApplyVectorLayoutContext ctx) {
  return mlir::tpu::ApplyVectorLayoutContext{
      .hardware_generation = ctx.hardware_generation,
      .target_shape = unwrap(ctx.target_shape),
      .mxu_shape = {ctx.mxu_shape.contracting_size,
                    ctx.mxu_shape.non_contracting_size},
      .max_sublanes_in_scratch = ctx.max_sublanes_in_scratch};
}

mlir::OpBuilder mlirTpuInsertionPointToOpBuilder(
    MlirTpuInsertionPoint insertion_point) {
  mlir::Operation *ref_operation = unwrap(insertion_point.ref_operation);
  return ref_operation == nullptr
             ? mlir::OpBuilder::atBlockEnd(unwrap(insertion_point.block))
             : mlir::OpBuilder(ref_operation);
}

// We do not use the names wrap/unwrap for MlirTpuI64ArrayRef because whether
// they should refer to SmallVector or ArrayRef is ambiguous
MlirTpuI64ArrayRef mlirTpuI64ArrayRefFromLlvmSmallVector(
    const mlir::SmallVector<int64_t> &vec) {
  // TODO(tlongeri): It would be good to steal the buffer from implicit_shape,
  // but there are no public member functions for this.
  int64_t *ptr =
      static_cast<int64_t *>(llvm::safe_malloc(vec.size() * sizeof(int64_t)));
  memcpy(ptr, vec.data(), vec.size() * sizeof(int64_t));
  return {ptr, vec.size()};
}
llvm::ArrayRef<int64_t> mlirTpuI64ArrayRefToLlvmArrayRef(
    MlirTpuI64ArrayRef tpu_array_ref) {
  return {tpu_array_ref.ptr, tpu_array_ref.size};
}

// We do not use the names wrap/unwrap for MlirTpuValueArray because it
// allocates memory (i.e. they have side effects)
xla::Array<mlir::Value> MlirTpuValueArrayToXlaArray(MlirTpuValueArray arr) {
  llvm::ArrayRef<int64_t> shape = mlirTpuI64ArrayRefToLlvmArrayRef(arr.shape);
  xla::Array<mlir::Value> res(shape);
  int64_t n = res.num_elements();
  for (int64_t i = 0; i < n; ++i) {
    res.data()[i] = unwrap(arr.vals[i]);
  }
  return res;
}
MlirTpuValueArray MlirTpuValueArrayFromXlaArray(
    const xla::Array<mlir::Value> &vals) {
  int64_t nd = vals.num_dimensions();
  int64_t *shape =
      static_cast<int64_t *>(llvm::safe_malloc(nd * sizeof(int64_t)));
  memcpy(shape, vals.dimensions().data(), nd * sizeof(int64_t));
  int64_t n = vals.num_elements();
  MlirValue *elements =
      static_cast<MlirValue *>(llvm::safe_malloc(n * sizeof(MlirValue)));
  memcpy(elements, vals.data(), n * sizeof(MlirValue));
  return {{shape, static_cast<size_t>(nd)}, elements};
}

}  // namespace

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TPU, tpu, mlir::tpu::TPUDialect);

bool mlirTPUAttributeIsATiledLayoutAttr(MlirAttribute attr) {
  return llvm::isa<mlir::tpu::TiledLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirTPUTiledLayoutAttrGetTiles(MlirAttribute attr) {
  auto layout_attr = llvm::cast<mlir::tpu::TiledLayoutAttr>(unwrap(attr));
  std::vector<mlir::Attribute> tile_attrs;
  tile_attrs.reserve(layout_attr.getTiles().size());
  mlir::MLIRContext *ctx = layout_attr.getContext();
  for (auto &tile : layout_attr.getTiles()) {
    auto d = tile.dimensions();
    tile_attrs.push_back(mlir::DenseI64ArrayAttr::get(
        ctx, llvm::ArrayRef<int64_t>(d.begin(), d.end())));
  }
  return wrap(mlir::ArrayAttr::get(ctx, tile_attrs));
}

void mlirTPUAnalyzePotentialCommunication(MlirOperation op,
                                          bool *has_communication,
                                          bool *has_custom_barrier) {
  auto result = mlir::tpu::mightCommunicateBetweenChips(unwrap(op));
  *has_communication = result.first;
  *has_custom_barrier = result.second;
}

MlirTpuVectorLayout mlirTpuVectorLayoutCreate(int bitwidth,
                                              MlirTpuLayoutOffsets offsets,
                                              MlirTpuI64TargetTuple tiling,
                                              MlirTpuImplicitDim implicit_dim) {
  return wrap(new mlir::tpu::VectorLayout(
      bitwidth, unwrap(offsets), unwrap(tiling), unwrap(implicit_dim)));
}

void mlirTpuVectorLayoutDestroy(MlirTpuVectorLayout layout) {
  delete unwrap(layout);
}

int mlirTpuVectorLayoutGetBitwidth(MlirTpuVectorLayout layout) {
  return unwrap(layout)->bitwidth();
}

MlirTpuLayoutOffsets mlirTpuVectorLayoutGetOffsets(MlirTpuVectorLayout layout) {
  return wrap(unwrap(layout)->offsets());
}

MlirTpuI64TargetTuple mlirTpuVectorLayoutGetTiling(MlirTpuVectorLayout layout) {
  return wrap(unwrap(layout)->tiling());
}

MlirTpuImplicitDim mlirTpuVectorLayoutGetImplicitDim(
    MlirTpuVectorLayout layout) {
  return wrap(unwrap(layout)->implicit_dim());
}

int mlirTpuVectorLayoutGetPacking(MlirTpuVectorLayout layout) {
  return unwrap(layout)->packing();
}

int mlirTpuVectorLayoutGetLayoutRank(MlirTpuVectorLayout layout) {
  return unwrap(layout)->layout_rank();
}

bool mlirTpuVectorLayoutEquals(MlirTpuVectorLayout lhs,
                               MlirTpuVectorLayout rhs) {
  return *unwrap(lhs) == *unwrap(rhs);
}

int64_t mlirTpuVectorLayoutTilesPerVreg(MlirTpuVectorLayout layout,
                                        MlirTpuI64TargetTuple target_shape) {
  return unwrap(layout)->tilesPerVreg(unwrap(target_shape));
}

int64_t mlirTpuVectorLayoutSublanesPerTile(MlirTpuVectorLayout layout,
                                           MlirTpuI64TargetTuple target_shape) {
  return unwrap(layout)->sublanesPerTile(unwrap(target_shape));
}

MlirTpuI64TargetTuple mlirTpuVectorLayoutVregSlice(
    MlirTpuVectorLayout layout, MlirTpuI64TargetTuple target_shape) {
  return wrap(unwrap(layout)->vregSlice(unwrap(target_shape)));
}

MlirTpuI64ArrayRef mlirTpuVectorLayoutImplicitShape(MlirTpuVectorLayout layout,
                                                    MlirTpuI64ArrayRef shape) {
  mlir::SmallVector<int64_t> implicit_shape =
      unwrap(layout)->implicitShape(mlirTpuI64ArrayRefToLlvmArrayRef(shape));
  return mlirTpuI64ArrayRefFromLlvmSmallVector(implicit_shape);
}

MlirTpuI64ArrayRef mlirTpuVectorLayoutTileArrayShape(
    MlirTpuVectorLayout layout, MlirTpuI64ArrayRef shape,
    MlirTpuI64TargetTuple target_shape) {
  mlir::SmallVector<int64_t> tile_array_shape = unwrap(layout)->tileArrayShape(
      mlirTpuI64ArrayRefToLlvmArrayRef(shape), unwrap(target_shape));
  return mlirTpuI64ArrayRefFromLlvmSmallVector(tile_array_shape);
}

MlirTpuVregDataBounds mlirTpuVectorLayoutTileDataBounds(
    MlirTpuVectorLayout layout, MlirContext ctx, int64_t *full_shape,
    int64_t *idxs, size_t size, MlirTpuI64TargetTuple target_shape,
    MlirTpuBoolTargetTuple allow_replicated) {
  std::unique_ptr<mlir::tpu::VRegDataBounds> ptr =
      unwrap(layout)->tileDataBounds(
          unwrap(ctx), llvm::ArrayRef<int64_t>{full_shape, size},
          llvm::ArrayRef<int64_t>{idxs, size}, unwrap(target_shape),
          unwrap(allow_replicated));
  return wrap(ptr.release());
}

bool mlirTpuVectorLayoutHasNaturalTopology(MlirTpuVectorLayout layout,
                                           MlirTpuI64TargetTuple target_shape) {
  return unwrap(layout)->hasNaturalTopology(unwrap(target_shape));
}

bool mlirTpuVectorLayoutHasNativeTiling(MlirTpuVectorLayout layout,
                                        MlirTpuI64TargetTuple target_shape) {
  return unwrap(layout)->hasNativeTiling(unwrap(target_shape));
}

bool mlirTpuVectorLayoutGeneralizes(MlirTpuVectorLayout layout,
                                    MlirTpuVectorLayout other,
                                    MlirTpuI64ArrayRef shape,
                                    MlirTpuI64TargetTuple target_shape) {
  return unwrap(layout)->generalizes(*unwrap(other),
                                     mlirTpuI64ArrayRefToLlvmArrayRef(shape),
                                     unwrap(target_shape));
}

bool mlirTpuVectorLayoutEquivalentTo(MlirTpuVectorLayout layout,
                                     MlirTpuVectorLayout other,
                                     MlirTpuI64ArrayRef shape,
                                     MlirTpuI64TargetTuple target_shape) {
  return unwrap(layout)->equivalentTo(*unwrap(other),
                                      mlirTpuI64ArrayRefToLlvmArrayRef(shape),
                                      unwrap(target_shape));
}

void mlirTpuVectorLayoutPrint(MlirTpuVectorLayout layout,
                              MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  unwrap(layout)->print<llvm::raw_ostream>(stream);
}

bool mlirTpuVectorLayoutIsValid(MlirTpuVectorLayout layout,
                                MlirTpuI64TargetTuple target_shape) {
  return unwrap(layout)->isValid(unwrap(target_shape));
}

void mlirTpuVregDataBoundsDestroy(MlirTpuVregDataBounds data_bounds) {
  delete unwrap(data_bounds);
}

bool mlirTpuVregDataBoundsMaskVariesAlong(MlirTpuVregDataBounds data_bounds,
                                          MlirTpuDirection direction,
                                          MlirTpuI64TargetTuple target_shape) {
  return unwrap(data_bounds)
      ->maskVariesAlong(unwrap(direction), unwrap(target_shape));
}

bool mlirTpuVregDataBoundsIsComplete(MlirTpuVregDataBounds data_bounds,
                                     MlirTpuI64TargetTuple target_shape) {
  return unwrap(data_bounds)->isComplete(unwrap(target_shape));
}

MlirValue mlirTpuVregDataBoundsGetVectorMask(
    MlirTpuVregDataBounds data_bounds, MlirTpuInsertionPoint insertion_point,
    MlirLocation location, int generation, MlirTpuI64TargetTuple target_shape) {
  mlir::OpBuilder builder = mlirTpuInsertionPointToOpBuilder(insertion_point);
  auto failure_or_mask = unwrap(data_bounds)
                             ->getVectorMask(builder, unwrap(location),
                                             generation, unwrap(target_shape));
  if (failed(failure_or_mask)) {
    return wrap(mlir::Value());
  } else {
    return wrap(failure_or_mask.value());
  }
}

MlirAttribute mlirTpuVregDataBoundsGetSublaneMask(
    MlirTpuVregDataBounds data_bounds, MlirContext ctx,
    MlirTpuI64TargetTuple target_shape) {
  return wrap(
      unwrap(data_bounds)->getSublaneMask(unwrap(ctx), unwrap(target_shape)));
}

MlirOperation mlirTpuAssemble(MlirTpuInsertionPoint insertion_point,
                              MlirType vector_type, MlirTpuVectorLayout layout,
                              MlirTpuValueArray vals,
                              MlirTpuI64TargetTuple target_shape) {
  mlir::OpBuilder builder = mlirTpuInsertionPointToOpBuilder(insertion_point);
  // This cast will fail and assert if the caller passed a non-vector type
  auto vty = mlir::cast<mlir::VectorType>(unwrap(vector_type));
  return wrap(mlir::tpu::assemble(builder, vty, *unwrap(layout),
                                  MlirTpuValueArrayToXlaArray(vals),
                                  unwrap(target_shape))
                  .getOperation());
}

MlirTpuValueArray mlirTpuDisassemble(MlirTpuInsertionPoint insertion_point,
                                     MlirTpuVectorLayout layout, MlirValue val,
                                     MlirTpuI64TargetTuple target_shape) {
  mlir::OpBuilder builder = mlirTpuInsertionPointToOpBuilder(insertion_point);
  // This cast will fail and assert if the caller passed a non-vector
  auto vector_val = mlir::cast<mlir::TypedValue<mlir::VectorType>>(unwrap(val));
  mlir::FailureOr<xla::Array<mlir::Value>> failure_or_vals =
      mlir::tpu::disassemble(builder, *unwrap(layout), vector_val,
                             unwrap(target_shape));
  if (failed(failure_or_vals)) {
    return {{nullptr, 0}, nullptr};
  }
  return MlirTpuValueArrayFromXlaArray(std::move(failure_or_vals).value());
}

MlirLogicalResult mlirTpuApplyLayoutOp(MlirTpuApplyVectorLayoutContext ctx,
                                       MlirOperation op) {
  mlir::tpu::ApplyVectorLayoutContext unwrapped_ctx = unwrap(ctx);
  return wrap(mlir::tpu::applyLayoutOp(unwrapped_ctx, *unwrap(op)));
}

MlirValue mlirTpuRelayout(MlirTpuInsertionPoint insertion_point, MlirValue val,
                          MlirTpuVectorLayout src, MlirTpuVectorLayout dst,
                          MlirTpuApplyVectorLayoutContext ctx) {
  mlir::OpBuilder builder = mlirTpuInsertionPointToOpBuilder(insertion_point);
  // This cast will fail and assert if the caller passed a non-vector
  auto vector_val = mlir::cast<mlir::TypedValue<mlir::VectorType>>(unwrap(val));
  auto apply_layout_ctx = unwrap(ctx);
  mlir::FailureOr<mlir::TypedValue<mlir::VectorType>> failure_or_new_val =
      mlir::tpu::relayout(apply_layout_ctx, builder, vector_val, *unwrap(src),
                          *unwrap(dst));
  if (failed(failure_or_new_val)) {
    return {nullptr};
  }
  return wrap(std::move(failure_or_new_val).value());
}
}

MLIR_CAPI_EXPORTED void mlirTpuRegisterMosaicSerdePass() {
  mlir::tpu::registerMosaicSerdePass();
}

#include "mlir/CAPI/Pass.h"  // IWYU pragma: keep
#include "mlir/CAPI/Support.h"  // IWYU pragma: keep

extern "C" {
using namespace mlir::tpu;

#include "xla/mosaic/dialect/tpu/integrations/c/tpu_passes.capi.cc.inc"
}
