/* Copyright 2023 The OpenXLA Authors. All Rights Reserved.

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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/dialects/comparators.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace {

class MaskSideChannel : public InterpreterSideChannel {
 public:
  MaskSideChannel(TensorOrMemref<bool> mask,
                  std::optional<InterpreterValue> passthrough)
      : mask_(std::move(mask)), passthrough_(std::move(passthrough)) {}

  const TensorOrMemref<bool>& GetMask() const { return mask_; }
  const std::optional<InterpreterValue>& GetPassthrough() const {
    return passthrough_;
  }

 private:
  const TensorOrMemref<bool> mask_;
  const std::optional<InterpreterValue> passthrough_;
};

template <typename T>
using combiner_t = T (*)(T, T);

template <typename T>
combiner_t<T> GetCombiner(vector::CombiningKind kind) {
  if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
    switch (kind) {
      case vector::CombiningKind::ADD:
        return +[](T a, T b) -> T { return a + b; };
      case vector::CombiningKind::MUL:
        return +[](T a, T b) -> T { return a * b; };
      case vector::CombiningKind::MINSI:
      case vector::CombiningKind::MINIMUMF:
        return +[](T a, T b) -> T { return std::min(a, b); };
      case vector::CombiningKind::MAXSI:
      case vector::CombiningKind::MAXIMUMF:
        return +[](T a, T b) -> T { return std::max(a, b); };
      default: {
      }
    }
  }

  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    switch (kind) {
      case vector::CombiningKind::MINUI:
        return &Iumin::apply<T>;
      case vector::CombiningKind::MAXUI:
        return &Iumax::apply<T>;
      default: {
      }
    }
  }

  if constexpr (std::is_integral_v<T>) {
    switch (kind) {
      case vector::CombiningKind::AND:
        return +[](T a, T b) -> T { return a & b; };
      case vector::CombiningKind::OR:
        return +[](T a, T b) -> T { return a | b; };
      case vector::CombiningKind::XOR:
        return +[](T a, T b) -> T { return a ^ b; };
      default: {
      }
    }
  }

  llvm_unreachable("unknown combining kind");
}

InterpreterValue GetNeutralElement(vector::CombiningKind kind, mlir::Type ty) {
  return DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    using T = decltype(dummy);
    switch (kind) {
      case vector::CombiningKind::AND:
      case vector::CombiningKind::MINUI:
        if constexpr (std::is_same_v<T, bool>) {
          return {true};
        } else if constexpr (std::is_integral_v<T>) {
          return {static_cast<T>(static_cast<std::make_unsigned_t<T>>(-1))};
        }
        break;
      case vector::CombiningKind::ADD:
      case vector::CombiningKind::MAXUI:
      case vector::CombiningKind::OR:
      case vector::CombiningKind::XOR:
        return {T{0}};
      case vector::CombiningKind::MUL:
        return {T{1}};
      case vector::CombiningKind::MINSI:
        return {std::numeric_limits<T>::max()};
      case vector::CombiningKind::MINIMUMF:
        return {std::numeric_limits<T>::infinity()};
      case vector::CombiningKind::MAXSI:
        return {std::numeric_limits<T>::min()};  // NOLINT
      case vector::CombiningKind::MAXIMUMF:
        return {-std::numeric_limits<T>::infinity()};
      default: {
      }
    }
    llvm_unreachable("invalid combining kind");
  });
}

template <typename IntType>
SmallVector<IntType> ExtractVector(ArrayAttr array_attr) {
  return llvm::to_vector<4>(llvm::map_range(
      array_attr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

InterpreterValue Bitcast(InterpreterState&, vector::BitCastOp op,
                         const InterpreterValue& vector) {
  ShapedType ty = cast<ShapedType>(op->getResultTypes()[0]);
  auto flattened = vector.CoerceLayout({});
  auto buffer = flattened.GetBuffer();
  auto view = flattened.View();
  view.sizes = llvm::to_vector(ty.getShape());
  view.strides = BufferView::GetDefaultStrides(view.sizes);
  return DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    // TODO(jreiffers): i1 semantics are currently broken (both here and
    // upstream).
    using T = decltype(dummy);
    return {TensorOrMemref<T>{buffer, view}};
  });
}

InterpreterValue Broadcast(InterpreterState&, vector::BroadcastOp broadcast,
                           const InterpreterValue& value) {
  auto result =
      value.IsTensor() ? value : value.AsUnitTensor(/*is_vector=*/true);
  auto& result_view = result.View();

  // Insert additional leading stride 0 dims.
  SmallVector<int64_t> strides(broadcast.getResultVectorType().getRank());
  llvm::copy(llvm::reverse(result_view.strides), strides.rbegin());
  // Zero out broadcast dimension strides.
  for (int64_t i : broadcast.computeBroadcastedUnitDims()) {
    strides[i] = 0;
  }

  result_view.strides = std::move(strides);
  result_view.sizes =
      llvm::to_vector(broadcast.getResultVectorType().getShape());
  return result;
}

void CompressStore(InterpreterState& state, vector::CompressStoreOp,
                   InterpreterValue dst, ArrayRef<int64_t> indices,
                   TensorOrMemref<bool> mask, InterpreterValue value) {
  auto dst_buffer = dst.GetBuffer();
  const auto& dst_view = dst.View();
  if (dst_view.strides.back() != 1) {
    state.AddFailure("trailing dimension must be continguous");
    return;
  }
  auto offset = dst_view.GetPhysicalIndex(indices);
  if (!offset) {
    state.AddFailure("index out of bounds");
    return;
  }

  auto src_buffer = value.GetBuffer();
  const auto& src_view = value.View();

  // TODO(jreiffers): Bounds checks.
  int64_t n = src_view.sizes[0];
  int64_t element_size = value.GetByteSizeOfElement();
  for (int64_t i = 0; i < n; ++i) {
    if (mask.at(i)) {
      memcpy(dst_buffer->at(offset, element_size),
             src_buffer->at(i, element_size), element_size);
      ++*offset;
    }
  }
}

InterpreterValue MaskImpl(mlir::Operation* op, ArrayRef<int64_t> mask_sizes) {
  auto out_sizes = cast<ShapedType>(op->getResultTypes()[0]).getShape();
  auto result = TensorOrMemref<bool>::Empty(out_sizes);
  result.view.is_vector = true;
  BufferView iter;
  iter.sizes = llvm::to_vector(mask_sizes);
  for (const auto& indices : iter.Indices()) {
    result.at(indices) = true;
  }
  return {result};
}

InterpreterValue ConstantMask(InterpreterState&, vector::ConstantMaskOp mask) {
  return MaskImpl(mask, ExtractVector<int64_t>(mask.getMaskDimSizes()));
}

// TODO(jreiffers): Support masked contractions.
InterpreterValue Contract(InterpreterState&, vector::ContractionOp contraction,
                          const InterpreterValue& lhs,
                          const InterpreterValue& rhs,
                          const InterpreterValue& acc) {
  BufferView iter;
  contraction.getIterationBounds(iter.sizes);
  auto maps = contraction.getIndexingMapsArray();
  auto result_ty = contraction->getResultTypes()[0];
  auto shaped_ty = result_ty.dyn_cast<ShapedType>();
  auto result =
      DispatchScalarType(result_ty, [&](auto dummy) -> InterpreterValue {
        using T = decltype(dummy);
        using TT = TensorOrMemref<T>;
        const auto& lhs_t = std::get<TT>(lhs.storage);
        const auto& rhs_t = std::get<TT>(rhs.storage);
        auto result_value = shaped_ty ? acc.Clone() : acc.AsUnitTensor();
        auto& result = std::get<TT>(result_value.storage);
        auto combiner = *GetCombiner<T>(contraction.getKind());
        for (const auto& indices : iter.Indices()) {
          auto lhs_indices = EvalAffineMap(maps[0], indices);
          auto rhs_indices = EvalAffineMap(maps[1], indices);
          auto result_indices = EvalAffineMap(maps[2], indices);

          auto& result_item = result.at(result_indices);
          result_item = combiner(result_item,
                                 lhs_t.at(lhs_indices) * rhs_t.at(rhs_indices));
        }
        return result_value;
      });

  return shaped_ty ? result : result.ExtractElement({});
}

InterpreterValue CreateMask(InterpreterState&, vector::CreateMaskOp op,
                            ArrayRef<int64_t> sizes) {
  return MaskImpl(op, sizes);
}

InterpreterValue ExpandLoad(InterpreterState& state, vector::ExpandLoadOp,
                            InterpreterValue memref,
                            SmallVector<int64_t> offsets,
                            TensorOrMemref<bool> mask,
                            const InterpreterValue& passthrough) {
  if (memref.View().strides.back() != 1) {
    state.AddFailure("expected last dimension to be contiguous");
    return {};
  }

  auto out = passthrough.Clone();
  for (int64_t i = 0, e = out.View().sizes[0]; i < e; ++i) {
    if (mask.at({i})) {
      out.InsertElement({i}, memref.ExtractElement(offsets));
      ++offsets.back();
    }
  }
  return out;
}

InterpreterValue Extract(InterpreterState& state, vector::ExtractOp extract,
                         const InterpreterValue& vector) {
  auto result = vector;
  auto& result_view = result.View();
  for (int64_t offset : extract.getStaticPosition()) {
    state.CheckSuccess(result_view.Slice(0, offset), "index out of bounds");
  }
  return result_view.Rank() == 0 ? result.ExtractElement({}) : result;
}

InterpreterValue ExtractElement(InterpreterState& state,
                                vector::ExtractElementOp,
                                const InterpreterValue& vector,
                                std::optional<int64_t> index) {
  if (!index) {
    return vector.ExtractElement({});
  }
  if (!vector.View().InBounds(*index)) {
    state.AddFailure("array index out of bounds");
    return {};
  }
  return vector.ExtractElement(*index);
}

InterpreterValue ExtractSlice(InterpreterState& state,
                              vector::ExtractStridedSliceOp extract,
                              const InterpreterValue& vector) {
  auto out = vector;
  state.CheckSuccess(
      out.View().Subview(ExtractVector<int64_t>(extract.getOffsets()),
                         ExtractVector<int64_t>(extract.getSizes()),
                         ExtractVector<int64_t>(extract.getStrides())),
      "subview out of bounds");
  return out;
}

InterpreterValue FlatTranspose(InterpreterState&,
                               vector::FlatTransposeOp transpose,
                               const InterpreterValue& vector) {
  auto out = vector.Clone();
  // We currently only implement -matrix-default-layout=column-major.
  int64_t rows = transpose.getRows();
  int64_t cols = transpose.getColumns();
  for (int64_t i = 0; i < rows * cols; ++i) {
    int64_t src_index = (i % cols) * rows + (i / cols);
    out.InsertElement({i}, vector.ExtractElement({src_index}));
  }
  return out;
}

InterpreterValue FusedMultiplyAdd(InterpreterState&, vector::FMAOp op,
                                  const InterpreterValue& lhs,
                                  const InterpreterValue& rhs,
                                  const InterpreterValue& acc) {
  auto out = acc.Clone();
  DispatchScalarType(op->getResultTypes()[0], [&](auto dummy) {
    using TT = TensorOrMemref<decltype(dummy)>;
    const auto& lhs_t = std::get<TT>(lhs.storage);
    const auto& rhs_t = std::get<TT>(rhs.storage);
    auto& result = std::get<TT>(out.storage);

    for (const auto& indices : lhs_t.view.Indices()) {
      result.at(indices) += lhs_t.at(indices) * rhs_t.at(indices);
    }
  });
  return out;
}

InterpreterValue Gather(InterpreterState& state, vector::GatherOp op,
                        const InterpreterValue& src, ArrayRef<int64_t> offsets,
                        const InterpreterValue& indices,
                        const TensorOrMemref<bool>& mask,
                        const InterpreterValue& pass_through) {
  if (isa<MemRefType>(op->getOperandTypes()[0]) &&
      src.View().strides.back() != 1) {
    state.AddFailure("expected trailing dimension to be contiguous");
    return {};
  }
  auto out = pass_through.Clone();
  for (const auto& out_index : out.View().Indices()) {
    if (!mask.at(out_index)) {
      continue;
    }
    auto in_index = llvm::to_vector(offsets);
    in_index[0] += indices.ExtractElement(out_index).AsInt();
    out.InsertElement(out_index, src.ExtractElement(in_index));
  }
  return out;
}

InterpreterValue Insert(InterpreterState& state, vector::InsertOp insert,
                        const InterpreterValue& src,
                        const InterpreterValue& dst) {
  auto result = dst.Clone();
  auto result_slice = result;
  auto& result_slice_view = result_slice.View();
  for (int64_t offset : insert.getStaticPosition()) {
    state.CheckSuccess(result_slice_view.Slice(0, offset),
                       "index out of bounds");
  }
  result_slice.Fill([&](auto indices) { return src.ExtractElement(indices); });
  return result;
}

InterpreterValue InsertElement(InterpreterState& state, vector::InsertElementOp,
                               const InterpreterValue& value,
                               const InterpreterValue& vector,
                               std::optional<int64_t> index) {
  auto result = vector.Clone();
  if (!index) {
    result.InsertElement({}, value);
    return result;
  }
  if (!result.View().InBounds(*index)) {
    state.AddFailure("array index out of bounds");
    return {};
  }
  result.InsertElement(*index, value);
  return result;
}

InterpreterValue InsertSlice(InterpreterState& state,
                             vector::InsertStridedSliceOp insert,
                             InterpreterValue src,
                             const InterpreterValue& dst) {
  auto out = dst.Clone();
  auto out_slice = out;
  if (!out_slice.View()
           .Subview(ExtractVector<int64_t>(insert.getOffsets()),
                    insert.getSourceVectorType().getShape(),
                    ExtractVector<int64_t>(insert.getStrides()))
           .succeeded()) {
    state.AddFailure("subview out of bounds");
  }
  for (const auto& index : src.View().Indices()) {
    out_slice.InsertElement(index, src.ExtractElement(index));
  }
  return out;
}

InterpreterValue Load(InterpreterState& state, vector::LoadOp load,
                      const InterpreterValue& memref,
                      ArrayRef<int64_t> indices) {
  if (memref.View().num_vector_dims > 0) {
    return {memref.ExtractElement(indices)};
  }
  auto out = memref;
  if (!out.View()
           .Subview(indices, load.getVectorType().getShape(),
                    SmallVector<int64_t>(load.getVectorType().getRank(), 1))
           .succeeded()) {
    // "Not all targets may support out-of-bounds vector loads"
    state.AddFailure("out of bounds loads not supported");
    return {};
  }
  out = out.Clone();
  out.View().is_vector = true;
  return out;
}

llvm::SmallVector<InterpreterValue> Mask(
    InterpreterState& state, vector::MaskOp op, TensorOrMemref<bool> mask,
    std::optional<InterpreterValue> passthrough) {
  InterpreterScope scope(state);
  scope.SetSideChannel(std::make_shared<MaskSideChannel>(mask, passthrough));
  return Interpret(state, op.getMaskRegion(), {});
}

InterpreterValue MaskedLoad(InterpreterState& state, vector::MaskedLoadOp,
                            const InterpreterValue& memref,
                            SmallVector<int64_t> offsets,
                            TensorOrMemref<bool> mask,
                            const InterpreterValue& passthrough) {
  if (memref.View().strides.back() != 1) {
    state.AddFailure("expected last dimension to be contiguous");
    return {};
  }

  auto out = passthrough.Clone();
  for (int64_t i = 0, e = mask.view.sizes[0]; i < e; ++i) {
    if (mask.at(i)) {
      out.InsertElement(i, memref.ExtractElement(offsets));
    }
    ++offsets.back();
  }
  return out;
}

void MaskedStore(InterpreterState& state, vector::MaskedStoreOp,
                 InterpreterValue memref, SmallVector<int64_t> offsets,
                 TensorOrMemref<bool> mask, const InterpreterValue& vector) {
  if (memref.View().strides.back() != 1) {
    state.AddFailure("expected last dimension to be contiguous");
    return;
  }

  for (int64_t i = 0, e = mask.view.sizes[0]; i < e; ++i) {
    if (mask.at({i})) {
      memref.InsertElement(offsets, vector.ExtractElement({i}));
    }
    ++offsets.back();
  }
}

InterpreterValue ReductionImpl(InterpreterState& state,
                               const InterpreterValue& v,
                               const InterpreterValue* acc,
                               vector::CombiningKind kind,
                               SmallVector<int64_t> dims, Type element_type) {
  llvm::sort(dims);
  SmallVector<int64_t> kept_dims;
  SmallVector<int64_t> result_shape;
  for (auto [dim, size] : llvm::enumerate(v.View().sizes)) {
    if (!llvm::is_contained(dims, dim)) {
      kept_dims.push_back(dim);
      result_shape.push_back(size);
    }
  }
  return DispatchScalarType(element_type, [&](auto dummy) -> InterpreterValue {
    using T = decltype(dummy);
    using TT = TensorOrMemref<T>;

    auto combiner = *GetCombiner<T>(kind);

    TT result = acc ? std::get<TT>((kept_dims.empty() ? acc->AsUnitTensor()
                                                      : acc->Clone())
                                       .storage)
                    : TensorOrMemref<T>::Empty(result_shape);

    for (const auto& result_index : result.view.Indices()) {
      auto src = std::get<TT>(v.storage);
      for (auto [dim, index] :
           llvm::reverse(llvm::zip(kept_dims, result_index))) {
        state.CheckSuccess(src.view.Slice(dim, index), "index out of bounds");
      }

      T& item = result.at(result_index);
      bool first = acc == nullptr;
      for (const auto& src_index : src.view.Indices()) {
        if (first) {
          item = src.at(src_index);
          first = false;
        } else {
          item = combiner(item, src.at(src_index));
        }
      }
    }

    if (kept_dims.empty()) {
      return {result.at({})};
    }
    return {result};
  });
}

InterpreterValue MultiReduction(InterpreterState& state,
                                vector::MultiDimReductionOp reduction,
                                const InterpreterValue& source,
                                const InterpreterValue& acc) {
  auto element_ty = getElementTypeOrSelf(reduction->getResultTypes()[0]);
  return {ReductionImpl(state, source, &acc, reduction.getKind(),
                        ExtractVector<int64_t>(reduction.getReductionDims()),
                        element_ty)};
}

InterpreterValue OuterProduct(InterpreterState&,
                              vector::OuterProductOp outer_product,
                              const InterpreterValue& lhs,
                              const InterpreterValue& rhs,
                              std::optional<InterpreterValue> acc) {
  ShapedType ty = cast<ShapedType>(outer_product->getResultTypes()[0]);
  return DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    using T = decltype(dummy);
    using TT = TensorOrMemref<T>;
    const TT& lhs_t = std::get<TT>(lhs.storage);

    auto combiner = GetCombiner<T>(outer_product.getKind());
    auto result =
        acc ? std::get<TT>(acc->storage).Clone() : TT::Empty(ty.getShape());
    if (std::holds_alternative<T>(rhs.storage)) {
      T rhs_scalar = std::get<T>(rhs.storage);
      for (int64_t i : llvm::seq(int64_t{0}, lhs_t.view.sizes[0])) {
        result.at(i) = combiner(result.at(i), lhs_t.at(i) * rhs_scalar);
      }
    } else {
      const TT& rhs_t = std::get<TT>(rhs.storage);
      for (int64_t i : llvm::seq(int64_t{0}, lhs_t.view.sizes[0])) {
        for (int64_t j : llvm::seq(int64_t{0}, rhs_t.view.sizes[0])) {
          result.at({i, j}) =
              combiner(result.at({i, j}), lhs_t.at(i) * rhs_t.at(j));
        }
      }
    }
    return {result};
  });
}

InterpreterValue Reduction(InterpreterState& state,
                           vector::ReductionOp reduction, InterpreterValue arg,
                           std::optional<InterpreterValue> acc) {
  auto* mask =
      state.GetTopScope()->GetSideChannel<MaskSideChannel>(/*optional=*/true);
  auto ty = reduction->getResultTypes()[0];
  if (mask) {
    if (mask->GetPassthrough()) {
      state.AddFailure("passthrough should not be set with masked reduction");
      return {};
    }
    arg = arg.Clone();
    if (mask->GetMask().view.sizes != arg.View().sizes) {
      state.AddFailure("mask shape should match argument shape");
      return {};
    }
    auto neutral = GetNeutralElement(reduction.getKind(), ty);
    for (const auto& idx : arg.View().Indices()) {
      if (!mask->GetMask().at(idx)) {
        arg.InsertElement(idx, neutral);
      }
    }
  }

  return ReductionImpl(state, arg, acc ? &acc.value() : nullptr,
                       reduction.getKind(), {0}, ty);
}

InterpreterValue ShapeCast(InterpreterState&, vector::ShapeCastOp op,
                           const InterpreterValue& in) {
  auto out = in.CoerceLayout({});
  auto& out_view = out.View();
  out_view.sizes =
      llvm::to_vector(op->getResultTypes()[0].cast<ShapedType>().getShape());
  out_view.strides = BufferView::GetDefaultStrides(out_view.sizes);
  return out;
}

InterpreterValue Shuffle(InterpreterState& state, vector::ShuffleOp shuffle,
                         const InterpreterValue& v0,
                         const InterpreterValue& v1) {
  auto result = v0.TypedAlike(shuffle.getResultVectorType().getShape());
  auto& result_view = result.View();
  result_view.is_vector = true;

  auto mask = ExtractVector<int64_t>(shuffle.getMask());
  bool is_zero_dim = v0.View().Rank() == 0;
  int64_t size0 = is_zero_dim ? 1 : v0.View().sizes[0];
  for (auto [dst_index, src_index] : llvm::enumerate(mask)) {
    auto src = src_index < size0 ? v0 : v1;
    if (!is_zero_dim) {
      state.CheckSuccess(
          src.View().Slice(0,
                           src_index < size0 ? src_index : src_index - size0),
          "index out of bounds");
    }
    auto dst = result;
    state.CheckSuccess(dst.View().Slice(0, dst_index), "index out of bounds");
    dst.Fill([&](auto indices) { return src.ExtractElement(indices); });
  }
  return result;
}

InterpreterValue Splat(InterpreterState&, vector::SplatOp op,
                       const InterpreterValue& in) {
  auto out = in.AsUnitTensor(/*is_vector=*/true);
  auto& view = out.View();
  view.sizes =
      llvm::to_vector(op->getResultTypes()[0].cast<ShapedType>().getShape());
  view.strides = SmallVector<int64_t>(view.sizes.size(), 0);
  return out;
}

void Store(InterpreterState& state, vector::StoreOp,
           const InterpreterValue& src, InterpreterValue dst,
           ArrayRef<int64_t> offsets) {
  if (!dst.View().InBounds(offsets)) {
    state.AddFailure("array index out of bounds");
    return;
  }
  const auto& out_view = dst.View();
  if (out_view.num_vector_dims > 0) {
    dst.InsertElement(offsets, src);
  } else {
    for (const auto& src_index : src.View().Indices()) {
      auto dst_index = src_index;
      for (int64_t i = 0; i < dst_index.size(); ++i) {
        dst_index[i] += offsets[i];
      }
      if (out_view.InBounds(dst_index)) {
        dst.InsertElement(dst_index, src.ExtractElement(src_index));
      }
    }
  }
}

std::optional<InterpreterValue> ExtractMemorySlice(
    InterpreterState& state, const AffineMap& map,
    const InterpreterValue& memory, const InterpreterValue& vector,
    ArrayRef<int64_t> offsets, std::optional<ArrayAttr> in_bounds_attr) {
  llvm::SmallVector<bool> in_bounds(offsets.size());
  if (in_bounds_attr) {
    llvm::copy(in_bounds_attr->getAsValueRange<BoolAttr>(),
               in_bounds.end() - in_bounds_attr->size());
  }

  auto mem_slice = memory;
  auto& mem_slice_view = mem_slice.View();
  auto& vector_view = vector.View();
  for (int64_t i = 0; i < mem_slice_view.Rank(); ++i) {
    bool found = false;
    for (int64_t j = 0; !found && j < vector_view.Rank(); ++j) {
      if (map.getResult(j).isFunctionOfDim(i)) {
        int64_t size = mem_slice_view.sizes[i] - offsets[i];
        bool is_in_bounds = size >= vector_view.sizes[j];
        if (!is_in_bounds && in_bounds[i]) {
          state.AddFailure("index out of bounds");
          return std::nullopt;
        }
        (void)mem_slice_view.Slice(
            i, offsets[i],
            std::max(int64_t{0}, std::min(vector_view.sizes[j], size)));
        found = true;
      }
    }
    if (!found) {
      bool is_in_bounds = mem_slice_view.sizes[i] > offsets[i];
      if (!is_in_bounds) {
        state.AddFailure("index out of bounds");
        return std::nullopt;
      }

      (void)mem_slice_view.Slice(i, offsets[i], is_in_bounds ? 1 : 0);
    }
  }
  return mem_slice;
}

InterpreterValue TransferRead(InterpreterState& state,
                              vector::TransferReadOp transfer,
                              const InterpreterValue& src,
                              ArrayRef<int64_t> offsets,
                              const InterpreterValue& padding,
                              std::optional<TensorOrMemref<bool>> mask) {
  auto* mask_channel = state.GetTopScope()->GetSideChannel<MaskSideChannel>(
      /*optional=*/true);
  if (mask_channel) {
    if (mask) {
      state.AddFailure(
          "vector.mask and transfer_read with mask should not be used "
          "simultaneously");
      return {};
    }
    mask = mask_channel->GetMask();
  }

  InterpreterValue dst = src.TypedAlike(transfer.getVectorType().getShape());
  if (mask_channel && mask_channel->GetPassthrough()) {
    dst.Fill([&](auto indices) {
      if (mask->at(indices)) {
        return padding;
      }
      return mask_channel->GetPassthrough()->ExtractElement(indices);
    });
  } else {
    dst.Fill([&](auto) { return padding; });
  }
  dst.View().is_vector = true;

  auto src_slice = ExtractMemorySlice(state, transfer.getPermutationMap(), src,
                                      dst, offsets, transfer.getInBounds());

  if (!src_slice) {
    return {};
  }
  for (const auto& src_indices : src_slice->View().Indices()) {
    SmallVector<int64_t> dst_indices =
        EvalAffineMap(transfer.getPermutationMap(), src_indices);

    // Note: the handling of padding and passthrough values is somewhat
    // arbitrary here. At the time of writing this, there seems to be little
    // evidence of actual usage of this feature.
    if (!mask || mask->at(dst_indices)) {
      dst.InsertElement(dst_indices, src_slice->ExtractElement(src_indices));
    }
  }

  return dst;
}

llvm::SmallVector<InterpreterValue> TransferWrite(
    InterpreterState& state, vector::TransferWriteOp transfer,
    InterpreterValue src, InterpreterValue dst, ArrayRef<int64_t> offsets,
    std::optional<TensorOrMemref<bool>> mask) {
  if (auto* mask_channel = state.GetTopScope()->GetSideChannel<MaskSideChannel>(
          /*optional=*/true)) {
    if (mask) {
      state.AddFailure(
          "vector.mask and transfer_write with mask should not be used "
          "simultaneously");
      return {};
    }
    if (mask_channel->GetPassthrough()) {
      state.AddFailure(
          "vector.mask with passthrough should not be used with "
          "transfer_write");
      return {};
    }
    mask = mask_channel->GetMask();
  }

  const auto& src_view = src.View();
  assert(transfer.getPermutationMap().getNumResults() == src_view.Rank() &&
         "expected matching number of results");

  dst = transfer.getSource().getType().isa<TensorType>() ? dst.Clone() : dst;
  auto dst_slice = ExtractMemorySlice(state, transfer.getPermutationMap(), dst,
                                      src, offsets, transfer.getInBounds());
  if (!dst_slice) {
    return {};
  }

  for (const auto& dst_indices : dst_slice->View().Indices()) {
    SmallVector<int64_t> src_indices =
        EvalAffineMap(transfer.getPermutationMap(), dst_indices);
    if (src_view.InBounds(src_indices) && (!mask || mask->at(src_indices))) {
      dst_slice->InsertElement(dst_indices, src.ExtractElement(src_indices));
    }
  }

  if (transfer->getNumResults() == 0) {
    return {};
  }
  return {dst};
}

InterpreterValue Transpose(InterpreterState&, vector::TransposeOp transpose,
                           const InterpreterValue& vector) {
  return TransposeImpl(vector, transpose.getPermutation());
}

InterpreterValue TypeCast(InterpreterState&, vector::TypeCastOp,
                          InterpreterValue vector) {
  vector.View().num_vector_dims = vector.View().Rank();
  return vector;
}

uint64_t VScale(InterpreterState&, vector::VectorScaleOp) { return 1; }

REGISTER_MLIR_INTERPRETER_OP("vector.yield", NoOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(Bitcast);
REGISTER_MLIR_INTERPRETER_OP(Broadcast);
REGISTER_MLIR_INTERPRETER_OP(CompressStore);
REGISTER_MLIR_INTERPRETER_OP(ConstantMask);
REGISTER_MLIR_INTERPRETER_OP(Contract);
REGISTER_MLIR_INTERPRETER_OP(CreateMask);
REGISTER_MLIR_INTERPRETER_OP(ExpandLoad);
REGISTER_MLIR_INTERPRETER_OP(Extract);
REGISTER_MLIR_INTERPRETER_OP(ExtractElement);
REGISTER_MLIR_INTERPRETER_OP(ExtractSlice);
REGISTER_MLIR_INTERPRETER_OP(FusedMultiplyAdd);
REGISTER_MLIR_INTERPRETER_OP(FlatTranspose);
REGISTER_MLIR_INTERPRETER_OP(Gather);
REGISTER_MLIR_INTERPRETER_OP(Insert);
REGISTER_MLIR_INTERPRETER_OP(InsertElement);
REGISTER_MLIR_INTERPRETER_OP(InsertSlice);
REGISTER_MLIR_INTERPRETER_OP(Load);
REGISTER_MLIR_INTERPRETER_OP(Mask);
REGISTER_MLIR_INTERPRETER_OP(MaskedLoad);
REGISTER_MLIR_INTERPRETER_OP(MaskedStore);
REGISTER_MLIR_INTERPRETER_OP(MultiReduction);
REGISTER_MLIR_INTERPRETER_OP(OuterProduct);
REGISTER_MLIR_INTERPRETER_OP(Reduction);
REGISTER_MLIR_INTERPRETER_OP(ShapeCast);
REGISTER_MLIR_INTERPRETER_OP(Shuffle);
REGISTER_MLIR_INTERPRETER_OP(Splat);
REGISTER_MLIR_INTERPRETER_OP(Store);
REGISTER_MLIR_INTERPRETER_OP(TransferRead);
REGISTER_MLIR_INTERPRETER_OP(TransferWrite);
REGISTER_MLIR_INTERPRETER_OP(Transpose);
REGISTER_MLIR_INTERPRETER_OP(TypeCast);
REGISTER_MLIR_INTERPRETER_OP(VScale);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
