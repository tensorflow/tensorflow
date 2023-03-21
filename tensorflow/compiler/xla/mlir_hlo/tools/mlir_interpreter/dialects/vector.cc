/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "tools/mlir_interpreter/dialects/comparators.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

class MaskSideChannel : public InterpreterSideChannel {
 public:
  MaskSideChannel(TensorOrMemref<bool> mask,
                  std::optional<InterpreterValue> passthrough)
      : mask(std::move(mask)), passthrough(std::move(passthrough)) {}

  const TensorOrMemref<bool>& getMask() const { return mask; }
  const std::optional<InterpreterValue>& getPassthrough() const {
    return passthrough;
  }

 private:
  const TensorOrMemref<bool> mask;
  const std::optional<InterpreterValue> passthrough;
};

template <typename T>
using combiner_t = T (*)(T, T);

template <typename T>
combiner_t<T> getCombiner(vector::CombiningKind kind) {
  if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
    switch (kind) {
      case vector::CombiningKind::ADD:
        return +[](T a, T b) -> T { return a + b; };
      case vector::CombiningKind::MUL:
        return +[](T a, T b) -> T { return a * b; };
      case vector::CombiningKind::MINSI:
      case vector::CombiningKind::MINF:
        return +[](T a, T b) -> T { return std::min(a, b); };
      case vector::CombiningKind::MAXSI:
      case vector::CombiningKind::MAXF:
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

InterpreterValue getNeutralElement(vector::CombiningKind kind, mlir::Type ty) {
  return dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
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
      case vector::CombiningKind::MINF:
        return {std::numeric_limits<T>::infinity()};
      case vector::CombiningKind::MAXSI:
        return {std::numeric_limits<T>::min()};
      case vector::CombiningKind::MAXF:
        return {-std::numeric_limits<T>::infinity()};
      default: {
      }
    }
    llvm_unreachable("invalid combining kind");
  });
}

template <typename IntType>
SmallVector<IntType> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

InterpreterValue bitcast(InterpreterState&, vector::BitCastOp op,
                         const InterpreterValue& vector) {
  ShapedType ty = op->getResultTypes()[0];
  auto flattened = vector.coerceLayout({});
  auto buffer = flattened.buffer();
  auto view = flattened.view();
  view.sizes = llvm::to_vector(ty.getShape());
  view.strides = BufferView::getDefaultStrides(view.sizes);
  return dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    // TODO(jreiffers): i1 semantics are currently broken (both here and
    // upstream).
    using T = decltype(dummy);
    return {TensorOrMemref<T>{buffer, view}};
  });
}

InterpreterValue broadcast(InterpreterState&, vector::BroadcastOp broadcast,
                           const InterpreterValue& value) {
  auto result =
      value.isTensor() ? value : value.asUnitTensor(/*isVector=*/true);
  auto& resultView = result.view();

  // Insert additional leading stride 0 dims.
  SmallVector<int64_t> strides(broadcast.getResultVectorType().getRank());
  llvm::copy(llvm::reverse(resultView.strides), strides.rbegin());
  // Zero out broadcast dimension strides.
  for (int64_t i : broadcast.computeBroadcastedUnitDims()) strides[i] = 0;

  resultView.strides = std::move(strides);
  resultView.sizes =
      llvm::to_vector(broadcast.getResultVectorType().getShape());
  return result;
}

void compressStore(InterpreterState& state, vector::CompressStoreOp,
                   InterpreterValue dst, ArrayRef<int64_t> indices,
                   TensorOrMemref<bool> mask, InterpreterValue value) {
  auto dstBuffer = dst.buffer();
  const auto& dstView = dst.view();
  if (dstView.strides.back() != 1) {
    state.addFailure("trailing dimension must be continguous");
    return;
  }
  auto offset = dstView.getPhysicalIndex(indices);
  if (!offset) {
    state.addFailure("index out of bounds");
    return;
  }

  auto srcBuffer = value.buffer();
  const auto& srcView = value.view();

  // TODO(jreiffers): Bounds checks.
  int64_t n = srcView.sizes[0];
  int64_t elementSize = value.getByteSizeOfElement();
  for (int64_t i = 0; i < n; ++i) {
    if (mask.at(i)) {
      memcpy(dstBuffer->at(offset, elementSize), srcBuffer->at(i, elementSize),
             elementSize);
      ++*offset;
    }
  }
}

InterpreterValue maskImpl(mlir::Operation* op, ArrayRef<int64_t> maskSizes) {
  auto outSizes = op->getResultTypes()[0].cast<ShapedType>().getShape();
  auto result = TensorOrMemref<bool>::empty(outSizes);
  result.view.isVector = true;
  BufferView iter;
  iter.sizes = llvm::to_vector(maskSizes);
  for (const auto& indices : iter.indices()) result.at(indices) = true;
  return {result};
}

InterpreterValue constantMask(InterpreterState&, vector::ConstantMaskOp mask) {
  return maskImpl(mask, extractVector<int64_t>(mask.getMaskDimSizes()));
}

// TODO(jreiffers): Support maksed contractions.
InterpreterValue contract(InterpreterState&, vector::ContractionOp contraction,
                          const InterpreterValue& lhs,
                          const InterpreterValue& rhs,
                          const InterpreterValue& acc) {
  BufferView iter;
  contraction.getIterationBounds(iter.sizes);
  auto maps = contraction.getIndexingMapsArray();
  auto resultTy = contraction->getResultTypes()[0];
  auto shapedTy = resultTy.dyn_cast<ShapedType>();
  auto result =
      dispatchScalarType(resultTy, [&](auto dummy) -> InterpreterValue {
        using T = decltype(dummy);
        using TT = TensorOrMemref<T>;
        const auto& lhsT = std::get<TT>(lhs.storage);
        const auto& rhsT = std::get<TT>(rhs.storage);
        auto resultValue = shapedTy ? acc.clone() : acc.asUnitTensor();
        auto& result = std::get<TT>(resultValue.storage);
        auto combiner = *getCombiner<T>(contraction.getKind());
        for (const auto& indices : iter.indices()) {
          auto lhsIndices = evalAffineMap(maps[0], indices);
          auto rhsIndices = evalAffineMap(maps[1], indices);
          auto resultIndices = evalAffineMap(maps[2], indices);

          auto& resultItem = result.at(resultIndices);
          resultItem =
              combiner(resultItem, lhsT.at(lhsIndices) * rhsT.at(rhsIndices));
        }
        return resultValue;
      });

  return shapedTy ? result : result.extractElement({});
}

InterpreterValue createMask(InterpreterState&, vector::CreateMaskOp op,
                            ArrayRef<int64_t> sizes) {
  return maskImpl(op, sizes);
}

InterpreterValue expandLoad(InterpreterState& state, vector::ExpandLoadOp,
                            InterpreterValue memref,
                            SmallVector<int64_t> offsets,
                            TensorOrMemref<bool> mask,
                            const InterpreterValue& passThrough) {
  if (memref.view().strides.back() != 1) {
    state.addFailure("expected last dimension to be contiguous");
    return {};
  }

  auto out = passThrough.clone();
  for (int64_t i = 0, e = out.view().sizes[0]; i < e; ++i) {
    if (mask.at({i})) {
      out.insertElement({i}, memref.extractElement(offsets));
      ++offsets.back();
    }
  }
  return out;
}

InterpreterValue extract(InterpreterState& state, vector::ExtractOp extract,
                         const InterpreterValue& vector) {
  auto result = vector;
  auto& resultView = result.view();
  for (int64_t offset : extractVector<int64_t>(extract.getPosition())) {
    state.checkSuccess(resultView.slice(0, offset), "index out of bounds");
  }
  return resultView.rank() == 0 ? result.extractElement({}) : result;
}

InterpreterValue extractElement(InterpreterState& state,
                                vector::ExtractElementOp,
                                const InterpreterValue& vector,
                                std::optional<int64_t> index) {
  if (!index) {
    return vector.extractElement({});
  }
  if (!vector.view().inBounds(*index)) {
    state.addFailure("array index out of bounds");
    return {};
  }
  return vector.extractElement(*index);
}

InterpreterValue extractSlice(InterpreterState& state,
                              vector::ExtractStridedSliceOp extract,
                              const InterpreterValue& vector) {
  auto out = vector;
  state.checkSuccess(
      out.view().subview(extractVector<int64_t>(extract.getOffsets()),
                         extractVector<int64_t>(extract.getSizes()),
                         extractVector<int64_t>(extract.getStrides())),
      "subview out of bounds");
  return out;
}

InterpreterValue flatTranspose(InterpreterState&,
                               vector::FlatTransposeOp transpose,
                               const InterpreterValue& vector) {
  auto out = vector.clone();
  // We currently only implement -matrix-default-layout=column-major.
  int64_t rows = transpose.getRows();
  int64_t cols = transpose.getColumns();
  for (int64_t i = 0; i < rows * cols; ++i) {
    int64_t srcIndex = (i % cols) * rows + (i / cols);
    out.insertElement({i}, vector.extractElement({srcIndex}));
  }
  return out;
}

InterpreterValue fusedMultiplyAdd(InterpreterState&, vector::FMAOp op,
                                  const InterpreterValue& lhs,
                                  const InterpreterValue& rhs,
                                  const InterpreterValue& acc) {
  auto out = acc.clone();
  dispatchScalarType(op->getResultTypes()[0], [&](auto dummy) {
    using TT = TensorOrMemref<decltype(dummy)>;
    const auto& lhsT = std::get<TT>(lhs.storage);
    const auto& rhsT = std::get<TT>(rhs.storage);
    auto& result = std::get<TT>(out.storage);

    for (const auto& indices : lhsT.view.indices()) {
      result.at(indices) += lhsT.at(indices) * rhsT.at(indices);
    }
  });
  return out;
}

InterpreterValue gather(InterpreterState& state, vector::GatherOp op,
                        const InterpreterValue& src, ArrayRef<int64_t> offsets,
                        const InterpreterValue& indices,
                        const TensorOrMemref<bool>& mask,
                        const InterpreterValue& passThrough) {
  if (op->getOperandTypes()[0].isa<MemRefType>() &&
      src.view().strides.back() != 1) {
    state.addFailure("expected trailing dimension to be contiguous");
    return {};
  }
  auto out = passThrough.clone();
  for (const auto& outIndex : out.view().indices()) {
    if (!mask.at(outIndex)) continue;
    auto inIndex = llvm::to_vector(offsets);
    inIndex[0] += indices.extractElement(outIndex).asInt();
    out.insertElement(outIndex, src.extractElement(inIndex));
  }
  return out;
}

InterpreterValue insert(InterpreterState& state, vector::InsertOp insert,
                        const InterpreterValue& src,
                        const InterpreterValue& dst) {
  auto result = dst.clone();
  auto resultSlice = result;
  auto& resultSliceView = resultSlice.view();
  for (int64_t offset : extractVector<int64_t>(insert.getPosition())) {
    state.checkSuccess(resultSliceView.slice(0, offset), "index out of bounds");
  }
  resultSlice.fill([&](auto indices) { return src.extractElement(indices); });
  return result;
}

InterpreterValue insertElement(InterpreterState& state, vector::InsertElementOp,
                               const InterpreterValue& value,
                               const InterpreterValue& vector,
                               std::optional<int64_t> index) {
  auto result = vector.clone();
  if (!index) {
    result.insertElement({}, value);
    return result;
  }
  if (!result.view().inBounds(*index)) {
    state.addFailure("array index out of bounds");
    return {};
  }
  result.insertElement(*index, value);
  return result;
}

InterpreterValue insertSlice(InterpreterState& state,
                             vector::InsertStridedSliceOp insert,
                             InterpreterValue src,
                             const InterpreterValue& dst) {
  auto out = dst.clone();
  auto outSlice = out;
  if (!outSlice.view()
           .subview(extractVector<int64_t>(insert.getOffsets()),
                    insert.getSourceVectorType().getShape(),
                    extractVector<int64_t>(insert.getStrides()))
           .succeeded()) {
    state.addFailure("subview out of bounds");
  }
  for (const auto& index : src.view().indices()) {
    outSlice.insertElement(index, src.extractElement(index));
  }
  return out;
}

InterpreterValue load(InterpreterState& state, vector::LoadOp load,
                      const InterpreterValue& memref,
                      ArrayRef<int64_t> indices) {
  if (memref.view().numVectorDims > 0) {
    return {memref.extractElement(indices)};
  }
  auto out = memref;
  if (!out.view()
           .subview(indices, load.getVectorType().getShape(),
                    SmallVector<int64_t>(load.getVectorType().getRank(), 1))
           .succeeded()) {
    // "Not all targets may support out-of-bounds vector loads"
    state.addFailure("out of bounds loads not supported");
    return {};
  }
  out = out.clone();
  out.view().isVector = true;
  return out;
}

llvm::SmallVector<InterpreterValue> mask(
    InterpreterState& state, vector::MaskOp op, TensorOrMemref<bool> mask,
    std::optional<InterpreterValue> passthrough) {
  InterpreterScope scope(state);
  scope.setSideChannel(std::make_shared<MaskSideChannel>(mask, passthrough));
  return interpret(state, op.getMaskRegion(), {});
}

InterpreterValue maskedLoad(InterpreterState& state, vector::MaskedLoadOp,
                            const InterpreterValue& memref,
                            SmallVector<int64_t> offsets,
                            TensorOrMemref<bool> mask,
                            const InterpreterValue& passThrough) {
  if (memref.view().strides.back() != 1) {
    state.addFailure("expected last dimension to be contiguous");
    return {};
  }

  auto out = passThrough.clone();
  for (int64_t i = 0, e = mask.view.sizes[0]; i < e; ++i) {
    if (mask.at(i)) {
      out.insertElement(i, memref.extractElement(offsets));
    }
    ++offsets.back();
  }
  return out;
}

void maskedStore(InterpreterState& state, vector::MaskedStoreOp,
                 InterpreterValue memref, SmallVector<int64_t> offsets,
                 TensorOrMemref<bool> mask, const InterpreterValue& vector) {
  if (memref.view().strides.back() != 1) {
    state.addFailure("expected last dimension to be contiguous");
    return;
  }

  for (int64_t i = 0, e = mask.view.sizes[0]; i < e; ++i) {
    if (mask.at({i})) {
      memref.insertElement(offsets, vector.extractElement({i}));
    }
    ++offsets.back();
  }
}

InterpreterValue reductionImpl(InterpreterState& state,
                               const InterpreterValue& v,
                               const InterpreterValue* acc,
                               vector::CombiningKind kind,
                               SmallVector<int64_t> dims, Type elementType) {
  llvm::sort(dims);
  SmallVector<int64_t> keptDims;
  SmallVector<int64_t> resultShape;
  for (auto [dim, size] : llvm::enumerate(v.view().sizes)) {
    if (!llvm::is_contained(dims, dim)) {
      keptDims.push_back(dim);
      resultShape.push_back(size);
    }
  }
  return dispatchScalarType(elementType, [&](auto dummy) -> InterpreterValue {
    using T = decltype(dummy);
    using TT = TensorOrMemref<T>;

    auto combiner = *getCombiner<T>(kind);

    TT result = acc ? std::get<TT>((keptDims.empty() ? acc->asUnitTensor()
                                                     : acc->clone())
                                       .storage)
                    : TensorOrMemref<T>::empty(resultShape);

    for (const auto& resultIndex : result.view.indices()) {
      auto src = std::get<TT>(v.storage);
      for (auto [dim, index] :
           llvm::reverse(llvm::zip(keptDims, resultIndex))) {
        state.checkSuccess(src.view.slice(dim, index), "index out of bounds");
      }

      T& item = result.at(resultIndex);
      bool first = acc == nullptr;
      for (const auto& srcIndex : src.view.indices()) {
        if (first) {
          item = src.at(srcIndex);
          first = false;
        } else {
          item = combiner(item, src.at(srcIndex));
        }
      }
    }

    if (keptDims.empty()) {
      return {result.at({})};
    }
    return {result};
  });
}

InterpreterValue multiReduction(InterpreterState& state,
                                vector::MultiDimReductionOp reduction,
                                const InterpreterValue& source,
                                const InterpreterValue& acc) {
  auto elementTy = getElementTypeOrSelf(reduction->getResultTypes()[0]);
  return {reductionImpl(state, source, &acc, reduction.getKind(),
                        extractVector<int64_t>(reduction.getReductionDims()),
                        elementTy)};
}

InterpreterValue outerProduct(InterpreterState&,
                              vector::OuterProductOp outerproduct,
                              const InterpreterValue& lhs,
                              const InterpreterValue& rhs,
                              std::optional<InterpreterValue> acc) {
  ShapedType ty = outerproduct->getResultTypes()[0];
  return dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    using T = decltype(dummy);
    using TT = TensorOrMemref<T>;
    const TT& lhsT = std::get<TT>(lhs.storage);

    auto combiner = *getCombiner<T>(outerproduct.getKind());
    auto result =
        acc ? std::get<TT>(acc->storage).clone() : TT::empty(ty.getShape());
    if (std::holds_alternative<T>(rhs.storage)) {
      T rhsS = std::get<T>(rhs.storage);
      for (int64_t i : llvm::seq(int64_t{0}, lhsT.view.sizes[0])) {
        result.at(i) = combiner(result.at(i), lhsT.at(i) * rhsS);
      }
    } else {
      const TT& rhsT = std::get<TT>(rhs.storage);
      for (int64_t i : llvm::seq(int64_t{0}, lhsT.view.sizes[0])) {
        for (int64_t j : llvm::seq(int64_t{0}, rhsT.view.sizes[0])) {
          result.at({i, j}) =
              combiner(result.at({i, j}), lhsT.at(i) * rhsT.at(j));
        }
      }
    }
    return {result};
  });
}

InterpreterValue reduction(InterpreterState& state,
                           vector::ReductionOp reduction, InterpreterValue arg,
                           std::optional<InterpreterValue> acc) {
  auto* mask =
      state.getTopScope()->getSideChannel<MaskSideChannel>(/*optional=*/true);
  auto ty = reduction->getResultTypes()[0];
  if (mask) {
    if (mask->getPassthrough()) {
      state.addFailure("passthrough should not be set with masked reduction");
      return {};
    }
    arg = arg.clone();
    if (mask->getMask().view.sizes != arg.view().sizes) {
      state.addFailure("mask shape should match argument shape");
      return {};
    }
    auto neutral = getNeutralElement(reduction.getKind(), ty);
    for (const auto& idx : arg.view().indices()) {
      if (!mask->getMask().at(idx)) {
        arg.insertElement(idx, neutral);
      }
    }
  }

  return reductionImpl(state, arg, acc ? &acc.value() : nullptr,
                       reduction.getKind(), {0}, ty);
}

InterpreterValue shapeCast(InterpreterState&, vector::ShapeCastOp op,
                           const InterpreterValue& in) {
  auto out = in.coerceLayout({});
  auto& outView = out.view();
  outView.sizes =
      llvm::to_vector(op->getResultTypes()[0].cast<ShapedType>().getShape());
  outView.strides = BufferView::getDefaultStrides(outView.sizes);
  return out;
}

InterpreterValue shuffle(InterpreterState& state, vector::ShuffleOp shuffle,
                         const InterpreterValue& v0,
                         const InterpreterValue& v1) {
  auto result = v0.typedAlike(shuffle.getResultVectorType().getShape());
  auto& resultView = result.view();
  resultView.isVector = true;

  auto mask = extractVector<int64_t>(shuffle.getMask());
  bool isZeroDim = v0.view().rank() == 0;
  int64_t size0 = isZeroDim ? 1 : v0.view().sizes[0];
  for (auto [dstIndex, srcIndex] : llvm::enumerate(mask)) {
    auto src = srcIndex < size0 ? v0 : v1;
    if (!isZeroDim) {
      state.checkSuccess(
          src.view().slice(0, srcIndex < size0 ? srcIndex : srcIndex - size0),
          "index out of bounds");
    }
    auto dst = result;
    state.checkSuccess(dst.view().slice(0, dstIndex), "index out of bounds");
    dst.fill([&](auto indices) { return src.extractElement(indices); });
  }
  return result;
}

InterpreterValue splat(InterpreterState&, vector::SplatOp op,
                       const InterpreterValue& in) {
  auto out = in.asUnitTensor(/*isVector=*/true);
  auto& view = out.view();
  view.sizes =
      llvm::to_vector(op->getResultTypes()[0].cast<ShapedType>().getShape());
  view.strides = SmallVector<int64_t>(view.sizes.size(), 0);
  return out;
}

void store(InterpreterState&, vector::StoreOp, const InterpreterValue& src,
           InterpreterValue dst, ArrayRef<int64_t> offsets) {
  const auto& outView = dst.view();
  if (outView.numVectorDims > 0) {
    dst.insertElement(offsets, src);
  } else {
    for (const auto& srcIndex : src.view().indices()) {
      auto dstIndex = srcIndex;
      for (int64_t i = 0; i < dstIndex.size(); ++i) {
        dstIndex[i] += offsets[i];
      }
      if (outView.inBounds(dstIndex)) {
        dst.insertElement(dstIndex, src.extractElement(srcIndex));
      }
    }
  }
}

std::optional<InterpreterValue> extractMemorySlice(
    InterpreterState& state, const AffineMap& map,
    const InterpreterValue& memory, const InterpreterValue& vector,
    ArrayRef<int64_t> offsets, std::optional<ArrayAttr> inBoundsAttr) {
  llvm::SmallVector<bool> inBounds(offsets.size());
  if (inBoundsAttr) {
    llvm::copy(inBoundsAttr->getAsValueRange<BoolAttr>(),
               inBounds.end() - inBoundsAttr->size());
  }

  auto memSlice = memory;
  auto& memSliceView = memSlice.view();
  auto& vectorView = vector.view();
  for (int64_t i = 0; i < memSliceView.rank(); ++i) {
    bool found = false;
    for (int64_t j = 0; !found && j < vectorView.rank(); ++j) {
      if (map.getResult(j).isFunctionOfDim(i)) {
        int64_t size = memSliceView.sizes[i] - offsets[i];
        bool isInBounds = size >= vectorView.sizes[j];
        if (!isInBounds && inBounds[i]) {
          state.addFailure("index out of bounds");
          return std::nullopt;
        }
        (void)memSliceView.slice(
            i, offsets[i],
            std::max(int64_t{0}, std::min(vectorView.sizes[j], size)));
        found = true;
      }
    }
    if (!found) {
      bool isInBounds = memSliceView.sizes[i] > offsets[i];
      if (!isInBounds) {
        state.addFailure("index out of bounds");
        return std::nullopt;
      }

      (void)memSliceView.slice(i, offsets[i], isInBounds ? 1 : 0);
    }
  }
  return memSlice;
}

InterpreterValue transferRead(InterpreterState& state,
                              vector::TransferReadOp transfer,
                              const InterpreterValue& src,
                              ArrayRef<int64_t> offsets,
                              const InterpreterValue& padding,
                              std::optional<TensorOrMemref<bool>> mask) {
  auto* maskChannel = state.getTopScope()->getSideChannel<MaskSideChannel>(
      /*optional=*/true);
  if (maskChannel) {
    if (mask) {
      state.addFailure(
          "vector.mask and transfer_read with mask should not be used "
          "simultaneously");
      return {};
    }
    mask = maskChannel->getMask();
  }

  InterpreterValue dst = src.typedAlike(transfer.getVectorType().getShape());
  if (maskChannel && maskChannel->getPassthrough()) {
    dst.fill([&](auto indices) {
      if (mask->at(indices)) {
        return padding;
      }
      return maskChannel->getPassthrough()->extractElement(indices);
    });
  } else {
    dst.fill([&](auto) { return padding; });
  }
  dst.view().isVector = true;

  auto srcSlice = extractMemorySlice(state, transfer.getPermutationMap(), src,
                                     dst, offsets, transfer.getInBounds());

  if (!srcSlice) {
    return {};
  }
  for (const auto& srcIndices : srcSlice->view().indices()) {
    SmallVector<int64_t> dstIndices =
        evalAffineMap(transfer.getPermutationMap(), srcIndices);

    // Note: the handling of padding and passthrough values is somewhat
    // arbitrary here. At the time of writing this, there seems to be little
    // evidence of actual usage of this feature.
    if (!mask || mask->at(dstIndices)) {
      dst.insertElement(dstIndices, srcSlice->extractElement(srcIndices));
    }
  }

  return dst;
}

llvm::SmallVector<InterpreterValue> transferWrite(
    InterpreterState& state, vector::TransferWriteOp transfer,
    InterpreterValue src, InterpreterValue dst, ArrayRef<int64_t> offsets,
    std::optional<TensorOrMemref<bool>> mask) {
  if (auto* maskChannel = state.getTopScope()->getSideChannel<MaskSideChannel>(
          /*optional=*/true)) {
    if (mask) {
      state.addFailure(
          "vector.mask and transfer_write with mask should not be used "
          "simultaneously");
      return {};
    }
    if (maskChannel->getPassthrough()) {
      state.addFailure(
          "vector.mask with passthrough should not be used with "
          "transfer_write");
      return {};
    }
    mask = maskChannel->getMask();
  }

  const auto& srcView = src.view();
  int64_t srcRank = srcView.rank();
  (void)srcRank;
  assert(transfer.getPermutationMap().getNumResults() == srcRank &&
         "expected matching number of results");

  dst = transfer.getSource().getType().isa<TensorType>() ? dst.clone() : dst;
  auto dstSlice = extractMemorySlice(state, transfer.getPermutationMap(), dst,
                                     src, offsets, transfer.getInBounds());
  if (!dstSlice) {
    return {};
  }

  for (const auto& dstIndices : dstSlice->view().indices()) {
    SmallVector<int64_t> srcIndices =
        evalAffineMap(transfer.getPermutationMap(), dstIndices);
    if (srcView.inBounds(srcIndices) && (!mask || mask->at(srcIndices))) {
      dstSlice->insertElement(dstIndices, src.extractElement(srcIndices));
    }
  }

  if (transfer->getNumResults() == 0) return {};
  return {dst};
}

InterpreterValue transpose(InterpreterState&, vector::TransposeOp transpose,
                           const InterpreterValue& vector) {
  auto permutation = extractVector<int64_t>(transpose.getTransp());
  return transposeImpl(vector, permutation);
}

InterpreterValue typeCast(InterpreterState&, vector::TypeCastOp,
                          InterpreterValue vector) {
  vector.view().numVectorDims = vector.view().rank();
  return vector;
}

uint64_t vScale(InterpreterState&, vector::VectorScaleOp) { return 1; }

REGISTER_MLIR_INTERPRETER_OP("vector.yield", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(bitcast);
REGISTER_MLIR_INTERPRETER_OP(broadcast);
REGISTER_MLIR_INTERPRETER_OP(compressStore);
REGISTER_MLIR_INTERPRETER_OP(constantMask);
REGISTER_MLIR_INTERPRETER_OP(contract);
REGISTER_MLIR_INTERPRETER_OP(createMask);
REGISTER_MLIR_INTERPRETER_OP(expandLoad);
REGISTER_MLIR_INTERPRETER_OP(extract);
REGISTER_MLIR_INTERPRETER_OP(extractElement);
REGISTER_MLIR_INTERPRETER_OP(extractSlice);
REGISTER_MLIR_INTERPRETER_OP(fusedMultiplyAdd);
REGISTER_MLIR_INTERPRETER_OP(flatTranspose);
REGISTER_MLIR_INTERPRETER_OP(gather);
REGISTER_MLIR_INTERPRETER_OP(insert);
REGISTER_MLIR_INTERPRETER_OP(insertElement);
REGISTER_MLIR_INTERPRETER_OP(insertSlice);
REGISTER_MLIR_INTERPRETER_OP(load);
REGISTER_MLIR_INTERPRETER_OP(mask);
REGISTER_MLIR_INTERPRETER_OP(maskedLoad);
REGISTER_MLIR_INTERPRETER_OP(maskedStore);
REGISTER_MLIR_INTERPRETER_OP(multiReduction);
REGISTER_MLIR_INTERPRETER_OP(outerProduct);
REGISTER_MLIR_INTERPRETER_OP(reduction);
REGISTER_MLIR_INTERPRETER_OP(shapeCast);
REGISTER_MLIR_INTERPRETER_OP(shuffle);
REGISTER_MLIR_INTERPRETER_OP(splat);
REGISTER_MLIR_INTERPRETER_OP(store);
REGISTER_MLIR_INTERPRETER_OP(transferRead);
REGISTER_MLIR_INTERPRETER_OP(transferWrite);
REGISTER_MLIR_INTERPRETER_OP(transpose);
REGISTER_MLIR_INTERPRETER_OP(typeCast);
REGISTER_MLIR_INTERPRETER_OP(vScale);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
