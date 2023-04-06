/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// NOTE: These ops aim to match the StableHLO spec and the behavior of the XLA
// compiler. For op semantics and a reference implementation, check the
// StableHLO repository (the spec is here:
// https://github.com/openxla/stablehlo/blob/main/docs/spec.md).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "tools/mlir_interpreter/dialects/comparators.h"
#include "tools/mlir_interpreter/dialects/cwise_math.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"
#include "tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue makeTuple(MutableArrayRef<InterpreterValue> values) {
  Tuple result;
  for (auto& value : values) {
    result.values.push_back(
        std::make_shared<InterpreterValue>(std::move(value)));
  }
  return {result};
}

InterpreterValue getTupleElement(InterpreterState&, mhlo::GetTupleElementOp get,
                                 const InterpreterValue& tuple) {
  return *std::get<Tuple>(tuple.storage).values[get.getIndex()];
}

InterpreterValue bitcastConvert(InterpreterState&, mhlo::BitcastConvertOp op,
                                const InterpreterValue& in) {
  ShapedType ty = op->getResultTypes()[0];
  auto result = dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    TensorOrMemref<decltype(dummy)> result;
    result.view = {};
    result.buffer = in.clone().buffer();
    return {result};
  });
  auto& outView = result.view();
  outView.strides = BufferView::getDefaultStrides(ty.getShape());
  outView.sizes = llvm::to_vector(ty.getShape());
  return result;
}

InterpreterValue broadcastInDim(InterpreterState&,
                                mhlo::BroadcastInDimOp broadcast,
                                InterpreterValue in) {
  auto broadcastDims = broadcast.getBroadcastDimensions().getValues<int64_t>();
  const auto& inSizes = in.view().sizes;
  auto out = in.typedAlike(
      broadcast->getResultTypes()[0].cast<ShapedType>().getShape());
  // TODO(jreiffers): Skip the copy.
  out.fill([&](llvm::ArrayRef<int64_t> outIndices) {
    llvm::SmallVector<int64_t> inIndices;
    for (auto [inDim, outDim] : llvm::enumerate(broadcastDims)) {
      inIndices.push_back(inSizes[inDim] == 1 ? 0 : outIndices[outDim]);
    }
    return in.extractElement(inIndices);
  });
  return out;
}

InterpreterValue clamp(InterpreterState&, mhlo::ClampOp,
                       const InterpreterValue& lb, const InterpreterValue& arg,
                       const InterpreterValue& ub) {
  auto result = arg.clone();
  for (const auto& index : arg.view().indices()) {
    auto lbScalar = lb.isTensor() ? lb.extractElement(index) : lb;
    auto ubScalar = ub.isTensor() ? ub.extractElement(index) : ub;
    assert(arg.isTensor() && "clamp only bcasts scalar bounds");
    auto argScalar = arg.extractElement(index);
    auto resultScalar = applyCwiseBinaryMap<Min>(
        applyCwiseBinaryMap<Max>(argScalar, lbScalar), ubScalar);
    result.insertElement(index, resultScalar);
  }
  return result;
}

InterpreterValue concatenate(InterpreterState&, mhlo::ConcatenateOp concat,
                             ArrayRef<InterpreterValue> vals) {
  uint64_t dim = concat.getDimension();
  auto sizes = vals[0].view().sizes;
  sizes[dim] = 0;
  for (const auto& val : vals) {
    sizes[dim] += val.view().sizes[dim];
  }

  auto result = vals[0].typedAlike(sizes);
  int64_t offset = 0;
  for (const auto& val : vals) {
    for (auto index : val.view().indices()) {
      auto item = val.extractElement(index);
      index[dim] += offset;
      result.insertElement(index, item);
    }
    offset += val.view().sizes[dim];
  }
  return result;
}

InterpreterValue reshape(InterpreterState&, mhlo::ReshapeOp reshape,
                         const InterpreterValue& in) {
  auto ty = reshape->getResultTypes()[0].cast<mlir::ShapedType>();
  return reshapeTensor(in, ty.getShape());
}

llvm::SmallVector<int64_t> clampStarts(ArrayRef<int64_t> starts,
                                       ArrayRef<int64_t> sliceSizes,
                                       ArrayRef<int64_t> tensorSizes) {
  llvm::SmallVector<int64_t> result;
  for (auto [start, sliceSize, tensorSize] :
       llvm::zip(starts, sliceSizes, tensorSizes)) {
    result.push_back(
        std::max(int64_t{0}, std::min(tensorSize - sliceSize, start)));
  }
  return result;
}

InterpreterValue dynamicSlice(InterpreterState&, mhlo::DynamicSliceOp slice,
                              const InterpreterValue& in,
                              ArrayRef<int64_t> starts) {
  auto result = in.typedAlike(
      llvm::to_vector(slice.getSliceSizes().getValues<int64_t>()));
  auto clampedStarts =
      clampStarts(starts, result.view().sizes, in.view().sizes);
  // TODO(jreiffers): Skip the copy.
  result.fill([&](llvm::ArrayRef<int64_t> outIndices) {
    llvm::SmallVector<int64_t> inIndices;
    for (auto [start, index] : llvm::zip(clampedStarts, outIndices)) {
      inIndices.push_back(start + index);
    }
    return in.extractElement(inIndices);
  });
  return result;
}

InterpreterValue dynamicUpdateSlice(InterpreterState&,
                                    mhlo::DynamicUpdateSliceOp,
                                    const InterpreterValue& in,
                                    const InterpreterValue& updates,
                                    ArrayRef<int64_t> starts) {
  auto result = in.clone();
  auto clampedStarts =
      clampStarts(starts, updates.view().sizes, result.view().sizes);
  for (auto inIndices : updates.view().indices()) {
    llvm::SmallVector<int64_t> outIndices;
    for (auto [start, index] : llvm::zip(clampedStarts, inIndices)) {
      outIndices.push_back(start + index);
    }
    result.insertElement(outIndices, updates.extractElement(inIndices));
  }
  return result;
}

InterpreterValue slice(InterpreterState&, mhlo::SliceOp slice,
                       InterpreterValue in) {
  auto starts = slice.getStartIndices().getValues<int64_t>();
  auto limits = slice.getLimitIndices().getValues<int64_t>();
  auto strides = slice.getStrides().getValues<int64_t>();

  llvm::SmallVector<int64_t> sizes;
  for (auto [start, limit, stride] : llvm::zip(starts, limits, strides)) {
    sizes.push_back(((limit - start) + (stride - 1)) / stride);
  }
  // TODO(jreiffers): Skip the copy.
  auto result = in.typedAlike(sizes);
  result.fill([&](llvm::ArrayRef<int64_t> outIndices) {
    llvm::SmallVector<int64_t> inIndices;
    for (auto [start, stride, index] : llvm::zip(starts, strides, outIndices)) {
      inIndices.push_back(start + stride * index);
    }
    return in.extractElement(inIndices);
  });
  return result;
}

llvm::SmallVector<InterpreterValue> constant(InterpreterState&,
                                             mhlo::ConstantOp constant) {
  auto ty = constant->getResultTypes()[0].cast<ShapedType>();
  return {dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    if (ty.getElementType().isUnsignedInteger()) {
      if constexpr (!std::is_same_v<decltype(dummy), bool> &&
                    std::is_integral_v<decltype(dummy)>) {
        auto values =
            constant.getValue()
                .getValues<
                    typename std::make_unsigned<decltype(dummy)>::type>();
        auto result = TensorOrMemref<decltype(dummy)>::empty(ty.getShape());
        auto valueIt = values.begin();
        for (const auto& index : result.view.indices()) {
          result.at(index) = *valueIt;
          ++valueIt;
        }
        return {result};
      } else {
        llvm_unreachable("invalid input");
      }
    } else {
      auto values = constant.getValue().getValues<decltype(dummy)>();
      auto result = TensorOrMemref<decltype(dummy)>::empty(ty.getShape());
      auto valueIt = values.begin();
      for (const auto& index : result.view.indices()) {
        result.at(index) = *valueIt;
        ++valueIt;
      }
      return {result};
    }
  })};
}

InterpreterValue pad(InterpreterState&, mhlo::PadOp pad, InterpreterValue arg,
                     InterpreterValue paddingValue) {
  paddingValue = paddingValue.extractElement({});
  auto his = pad.getEdgePaddingHigh().getValues<int64_t>();
  auto los = pad.getEdgePaddingLow().getValues<int64_t>();
  auto ins = pad.getInteriorPadding().getValues<int64_t>();

  llvm::SmallVector<int64_t> sizes;
  for (auto [size, lo, in, hi] : llvm::zip(arg.view().sizes, los, ins, his)) {
    sizes.push_back(size + lo + hi + (size - 1) * in);
  }

  auto result = arg.typedAlike(sizes);
  result.fill([&](llvm::ArrayRef<int64_t>) { return paddingValue; });

  for (const auto& inIndices : arg.view().indices()) {
    llvm::SmallVector<int64_t> outIndices;
    for (auto [inIndex, in, lo] : llvm::zip(inIndices, ins, los)) {
      outIndices.push_back(inIndex * (in + 1) + lo);
    }
    if (result.view().inBounds(outIndices)) {
      result.insertElement(outIndices, arg.extractElement(inIndices));
    }
  }

  return result;
}

InterpreterValue compare(InterpreterState&, mhlo::CompareOp compare,
                         const InterpreterValue& lhs,
                         const InterpreterValue& rhs) {
  switch (compare.getComparisonDirection()) {
    case mlir::mhlo::ComparisonDirection::EQ:
      return applyCwiseBinaryMap<Foeq>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::NE:
      return applyCwiseBinaryMap<Fune>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::GE:
      return applyCwiseBinaryMap<Foge>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::GT:
      return applyCwiseBinaryMap<Fogt>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::LE:
      return applyCwiseBinaryMap<Fole>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::LT:
      return applyCwiseBinaryMap<Folt>(lhs, rhs);
  }
}

llvm::SmallVector<InterpreterValue> gather(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto gather = llvm::dyn_cast<mhlo::GatherOp>(op);
  auto dynamicGather = llvm::dyn_cast<mhlo::DynamicGatherOp>(op);

  if (!gather && !dynamicGather) {
    state.addFailure("invalid gather");
    return {};
  }

  const auto& dims = gather ? gather.getDimensionNumbers()
                            : dynamicGather.getDimensionNumbers();

  auto indexVectorDim = dims.getIndexVectorDim();
  auto startIndexMap = dims.getStartIndexMap();
  auto offsetDims = dims.getOffsetDims();
  auto collapsedSliceDims = dims.getCollapsedSliceDims();
  auto sliceSizes =
      gather ? llvm::to_vector(gather.getSliceSizes().getValues<int64_t>())
             : llvm::to_vector(llvm::map_range(
                   args[2].view().indices(), [&](const auto& indices) {
                     return args[2].extractElement(indices).asInt();
                   }));

  auto& operand = args[0];
  auto& startIndices = args[1];
  const auto& operandView = operand.view();
  int64_t operandRank = operandView.rank();

  // Make a fake BufferView for the start indices.
  BufferView startIndicesView = startIndices.view();
  auto outputRank =
      static_cast<int64_t>(startIndicesView.rank() + offsetDims.size());
  if (indexVectorDim < startIndicesView.rank()) {
    --outputRank;
    startIndicesView.sizes[indexVectorDim] = 1;
  }

  SmallVector<int64_t> batchDims;
  for (int64_t i = 0; i < outputRank; ++i) {
    if (!llvm::is_contained(offsetDims, i)) {
      batchDims.push_back(i);
    }
  }

  // Make a fake BufferView for the slice indices.
  BufferView sliceIndicesView{0, SmallVector<int64_t>{sliceSizes}, {}};

  SmallVector<int64_t> nonCollapsedSliceDims;
  for (int64_t i = 0; i < operandRank; ++i) {
    if (!llvm::is_contained(collapsedSliceDims, i)) {
      nonCollapsedSliceDims.push_back(i);
    }
  }

  SmallVector<int64_t> outputSizes(outputRank);
  for (auto [outputDim, sliceDim] :
       llvm::zip(offsetDims, nonCollapsedSliceDims)) {
    outputSizes[outputDim] = sliceSizes[sliceDim];
  }
  for (auto [batchIndex, outputDim] : llvm::enumerate(batchDims)) {
    if (batchIndex >= indexVectorDim) {
      ++batchIndex;
    }
    outputSizes[outputDim] = startIndicesView.sizes[batchIndex];
  }

  auto output = operand.typedAlike(outputSizes);
  for (auto startIndicesIndex : startIndicesView.indices()) {
    SmallVector<int64_t> operandBaseIndices(operandRank);
    for (auto [i, dim] : llvm::enumerate(startIndexMap)) {
      if (indexVectorDim < startIndicesView.rank()) {
        startIndicesIndex[indexVectorDim] = static_cast<int64_t>(i);
      }
      operandBaseIndices[dim] = std::max<int64_t>(
          0, std::min(startIndices.extractElement(startIndicesIndex).asInt(),
                      operandView.sizes[dim] - sliceSizes[dim]));
    }

    for (const auto& sliceIndices : sliceIndicesView.indices()) {
      SmallVector<int64_t> operandIndices;
      for (int64_t i = 0; i < operandRank; ++i) {
        operandIndices.push_back(operandBaseIndices[i] + sliceIndices[i]);
      }

      SmallVector<int64_t> outputIndices(outputRank);
      for (auto [outputDim, sliceDim] :
           llvm::zip(offsetDims, nonCollapsedSliceDims)) {
        outputIndices[outputDim] = sliceIndices[sliceDim];
      }
      for (auto [batchIndex, outputDim] : llvm::enumerate(batchDims)) {
        outputIndices[outputDim] =
            startIndicesIndex[batchIndex >= indexVectorDim ? batchIndex + 1
                                                           : batchIndex];
      }

      auto value = operand.extractElement(operandIndices);
      output.insertElement(outputIndices, value);
    }
  }

  return {output};
}

llvm::SmallVector<InterpreterValue> scatter(
    InterpreterState& state, mhlo::ScatterOp scatter,
    ArrayRef<InterpreterValue> nInputs, InterpreterValue scatterIndices,
    ArrayRef<InterpreterValue> nUpdates) {
  const auto& dims = scatter.getScatterDimensionNumbers();
  auto indexVectorDim = dims.getIndexVectorDim();
  auto scatterDimsToOperandDims = dims.getScatterDimsToOperandDims();
  auto insertedWindowDims = dims.getInsertedWindowDims();
  auto updateWindowDims = dims.getUpdateWindowDims();

  auto inputView = nInputs.front().view();
  int64_t operandRank = inputView.rank();
  int64_t updatesRank = nUpdates.front().view().rank();
  int64_t indicesRank = scatterIndices.view().rank();

  llvm::SmallVector<int64_t> batchDims;
  for (int64_t dim = 0; dim < operandRank; ++dim) {
    if (!llvm::is_contained(insertedWindowDims, dim)) {
      batchDims.push_back(dim);
    }
  }

  llvm::SmallVector<int64_t> updateScatterDims;
  for (int64_t dim = 0; dim < updatesRank; ++dim) {
    if (!llvm::is_contained(updateWindowDims, dim)) {
      updateScatterDims.push_back(dim);
    }
  }

  llvm::SmallVector<InterpreterValue> nResults;
  for (auto& inputs : nInputs) {
    nResults.push_back(inputs.clone());
  }

  for (auto [inputs, updates, results] :
       llvm::zip(nResults, nUpdates, nResults)) {
    const auto& updatesView = updates.view();
    for (const auto& updateIndices : updatesView.indices()) {
      llvm::SmallVector<int64_t> inputIndices(operandRank);
      llvm::SmallVector<int64_t> maxIndices(operandRank);
      llvm::SmallVector<int64_t> minIndices(operandRank);
      llvm::SmallVector<int64_t> scatterIndicesIndex(indicesRank);

      for (auto [index, dim] : llvm::enumerate(updateScatterDims)) {
        scatterIndicesIndex[index >= indexVectorDim ? index + 1 : index] +=
            updateIndices[dim];
      }

      for (auto [updateDim, operandDim] :
           llvm::zip(updateWindowDims, batchDims)) {
        inputIndices[operandDim] = updateIndices[updateDim];
        maxIndices[operandDim] = updatesView.sizes[updateDim] - 1;
      }

      for (auto [index, dim] : llvm::enumerate(scatterDimsToOperandDims)) {
        if (indexVectorDim < indicesRank) {
          scatterIndicesIndex[indexVectorDim] = static_cast<int64_t>(index);
        }

        int64_t scatterIndex =
            scatterIndices.extractElement(scatterIndicesIndex).asInt();
        inputIndices[dim] += scatterIndex;
        minIndices[dim] += scatterIndex;
        maxIndices[dim] += scatterIndex;
      }

      if (!inputView.inBounds(minIndices)) continue;
      if (!inputView.inBounds(maxIndices)) continue;

      auto currentValue = inputs.extractElement(inputIndices).asUnitTensor();
      auto update = updates.extractElement(updateIndices).asUnitTensor();

      auto result = interpret(state, scatter.getUpdateComputation(),
                              {currentValue, update});
      if (state.hasFailure()) {
        return nResults;
      }
      inputs.insertElement(inputIndices, result.front().extractElement({}));
    }
  }

  return nResults;
}

InterpreterValue select(InterpreterState&, mhlo::SelectOp,
                        TensorOrMemref<bool> condition,
                        const InterpreterValue& trueVals,
                        const InterpreterValue& falseVals) {
  // TODO(jreiffers): Support implicit broadcasting.
  auto result = trueVals.clone();
  for (const auto& indices : condition.view.indices()) {
    if (!condition.at(indices)) {
      result.insertElement(indices, falseVals.extractElement(indices));
    }
  }
  return result;
}

InterpreterValue transpose(InterpreterState&, mhlo::TransposeOp transpose,
                           const InterpreterValue& tensor) {
  auto permutation = transpose.getPermutation().getValues<int64_t>();
  return transposeImpl(tensor, llvm::to_vector(permutation));
}

InterpreterValue iota(InterpreterState&, mhlo::IotaOp iota) {
  auto dim = iota.getIotaDimension();
  auto ty = iota->getResultTypes()[0].cast<ShapedType>();
  return dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    auto result = TensorOrMemref<decltype(dummy)>::empty({ty.getShape()[dim]});
    for (const auto& index : result.view.indices()) {
      result.at(index) = index[0];
    }
    result.view.sizes = SmallVector<int64_t>(ty.getShape());
    result.view.strides = SmallVector<int64_t>(result.view.sizes.size());
    result.view.strides[dim] = 1;
    return {result};
  });
}

template <typename R>
struct Caster {
  template <typename A>
  constexpr static bool supportedType() {
    return std::is_convertible_v<A, R>;
  }

  template <typename A>
  static R apply(A v) {
    return v;
  }
};

InterpreterValue convert(InterpreterState&, mhlo::ConvertOp op,
                         InterpreterValue in) {
  return dispatchScalarType(op->getResultTypes()[0],
                            [&](auto dummy) -> InterpreterValue {
                              return applyCwiseMap<Caster<decltype(dummy)>>(in);
                            });
}

llvm::SmallVector<InterpreterValue> mhloWhile(
    InterpreterState& state, mhlo::WhileOp whileop,
    SmallVector<InterpreterValue> loopVars) {
  auto cond = interpret(state, whileop.getRegion(0), loopVars);
  while (!state.hasFailure() &&
         std::get<TensorOrMemref<bool>>(cond.front().storage).at({})) {
    loopVars = interpret(state, whileop.getRegion(1), loopVars);
    if (state.hasFailure()) break;
    cond = interpret(state, whileop.getRegion(0), loopVars);
  }
  return loopVars;
}

InterpreterValue copy(InterpreterState&, mhlo::CopyOp op,
                      const InterpreterValue& tensor) {
  auto layout = op->getAttr("result_layout");
  if (!layout) return tensor.clone();
  return tensor.clone(llvm::to_vector(
      layout.cast<DenseIntElementsAttr>().getValues<int64_t>()));
}

llvm::SmallVector<InterpreterValue> fusion(InterpreterState& state,
                                           mhlo::FusionOp fusion,
                                           ArrayRef<InterpreterValue> args) {
  return interpret(state, fusion.getRegion(), args);
}

llvm::SmallVector<InterpreterValue> reduce(InterpreterState& state,
                                           mhlo::ReduceOp reduce,
                                           ArrayRef<InterpreterValue> operands,
                                           ArrayRef<InterpreterValue> inits) {
  auto dims = reduce.getDimensions().getValues<int64_t>();

  auto outSizes = operands.front().view().sizes;
  for (int64_t dim : llvm::reverse(dims)) {
    outSizes.erase(outSizes.begin() + dim);
  }

  SmallVector<InterpreterValue> results;
  for (auto [operand, init] : llvm::zip(operands, inits)) {
    results.push_back(operand.typedAlike(outSizes));
    auto initScalar = init.extractElement({});
    results.back().fill([&](llvm::ArrayRef<int64_t>) { return initScalar; });
  }

  for (const auto& index : operands[0].view().indices()) {
    auto dstIndex = index;
    for (int64_t dim : llvm::reverse(dims)) {
      dstIndex.erase(dstIndex.begin() + dim);
    }

    SmallVector<InterpreterValue> args;
    for (auto& result : results) {
      args.push_back(result.extractElement(dstIndex).asUnitTensor());
    }
    for (auto& operand : operands) {
      args.push_back(operand.extractElement(index).asUnitTensor());
    }
    auto newValues = interpret(state, reduce.getRegion(), args);
    if (state.hasFailure()) return results;

    for (auto [result, value] : llvm::zip(results, newValues)) {
      result.insertElement(dstIndex, value.extractElement({}));
    }
  }
  return results;
}

SmallVector<InterpreterValue> sort(InterpreterState& state, mhlo::SortOp op,
                                   ArrayRef<InterpreterValue> inputs) {
  const auto& shape = inputs.front().view().sizes;
  uint64_t dim =
      op.getDimension() < shape.size() ? op.getDimension() : shape.size() - 1;
  SmallVector<int64_t> indices(shape[dim]);

  auto iterView = inputs.front().view();
  iterView.sizes[dim] = 1;
  SmallVector<InterpreterValue> results;
  for (const auto& input : inputs) {
    results.push_back(input.typedAlike(input.view().sizes));
  }
  for (const auto& offset : iterView.indices()) {
    std::iota(indices.begin(), indices.end(), 0);
    auto comparator = [&](int64_t a, int64_t b) {
      if (state.hasFailure()) {
        return false;
      }
      SmallVector<InterpreterValue> args;
      auto indexA = offset;
      indexA[dim] = a;
      auto indexB = offset;
      indexB[dim] = b;
      for (const auto& input : inputs) {
        args.push_back(input.extractElement(indexA).asUnitTensor());
        args.push_back(input.extractElement(indexB).asUnitTensor());
      }
      auto result = interpret(state, op.getComparator(), args);
      return !state.hasFailure() && result[0].extractElement({}).asInt() != 0;
    };

    if (op.getIsStable()) {
      std::stable_sort(indices.begin(), indices.end(), comparator);
    } else {
      std::sort(indices.begin(), indices.end(), comparator);
    }

    auto copy = offset;
    for (auto [dst, src] : llvm::enumerate(indices)) {
      for (auto [input, result] : llvm::zip(inputs, results)) {
        copy[dim] = src;
        auto elem = input.extractElement(copy);
        copy[dim] = static_cast<int64_t>(dst);
        result.insertElement(copy, elem);
      }
    }
  }
  return results;
}

SmallVector<InterpreterValue> mhloCase(InterpreterState& state, mhlo::CaseOp op,
                                       int64_t index) {
  if (index < 0 || index >= op->getNumResults()) {
    index = op->getNumRegions() - 1;
  }
  return interpret(state, op->getRegion(index), {});
}

InterpreterValue dotGeneralImpl(InterpreterValue& lhs, InterpreterValue& rhs,
                                ArrayRef<int64_t> lhsContracting,
                                ArrayRef<int64_t> rhsContracting,
                                ArrayRef<int64_t> lhsBatch,
                                ArrayRef<int64_t> rhsBatch,
                                mlir::Type elementTy) {
  auto& lhsv = lhs.view();
  auto& rhsv = rhs.view();
  SmallVector<int64_t> dimensions;
  SmallVector<int64_t> lhsNonBatch;
  SmallVector<int64_t> rhsNonBatch;
  auto nbatch = static_cast<int64_t>(lhsBatch.size());
  for (const int64_t lhsDim : lhsBatch) {
    dimensions.push_back(lhsv.sizes[lhsDim]);
  }
  for (int64_t i = 0; i < lhsv.rank(); i++) {
    if (!llvm::is_contained(lhsContracting, i) &&
        !llvm::is_contained(lhsBatch, i)) {
      dimensions.push_back(lhsv.sizes[i]);
      lhsNonBatch.push_back(i);
    }
  }
  for (int64_t i = 0; i < rhs.view().rank(); i++) {
    if (!llvm::is_contained(rhsContracting, i) &&
        !llvm::is_contained(rhsBatch, i)) {
      dimensions.push_back(rhsv.sizes[i]);
      rhsNonBatch.push_back(i);
    }
  }

  SmallVector<int64_t> lhsIndex(lhsv.rank());
  SmallVector<int64_t> rhsIndex(rhsv.rank());
  SmallVector<int64_t> outputIndex(dimensions.size());
  auto output = lhs.typedAlike(dimensions);

  std::function<void(int64_t)> applyBatch, applyLhs, applyRhs, applyContract;

  applyBatch = [&](int64_t dim) {
    if (dim >= nbatch) {
      applyLhs(0);
      return;
    }
    for (int64_t i = 0; i < dimensions[dim]; ++i) {
      lhsIndex[lhsBatch[dim]] = i;
      rhsIndex[rhsBatch[dim]] = i;
      outputIndex[dim] = i;
      applyBatch(dim + 1);
    }
  };

  applyLhs = [&](int64_t dim) {
    if (dim >= lhsNonBatch.size()) {
      applyRhs(0);
      return;
    }
    int64_t outDim = nbatch + dim;
    for (int64_t i = 0; i < dimensions[outDim]; ++i) {
      lhsIndex[lhsNonBatch[dim]] = i;
      outputIndex[outDim] = i;
      applyLhs(dim + 1);
    }
  };

  applyRhs = [&](int64_t dim) {
    if (dim >= rhsNonBatch.size()) {
      applyContract(0);
      return;
    }
    auto outDim = static_cast<int64_t>(nbatch + lhsNonBatch.size() + dim);
    for (int64_t i = 0; i < dimensions[outDim]; ++i) {
      rhsIndex[rhsNonBatch[dim]] = i;
      outputIndex[outDim] = i;
      applyRhs(dim + 1);
    }
  };

  applyContract = [&](int64_t dim) {
    if (dim >= lhsContracting.size()) {
      dispatchScalarType(elementTy, [&](auto dummy) {
        using T = TensorOrMemref<decltype(dummy)>;
        std::get<T>(output.storage).at(outputIndex) +=
            std::get<T>(lhs.storage).at(lhsIndex) *
            std::get<T>(rhs.storage).at(rhsIndex);
      });
      return;
    }
    for (int64_t i = 0; i < lhsv.sizes[lhsContracting[dim]]; ++i) {
      lhsIndex[lhsContracting[dim]] = i;
      rhsIndex[rhsContracting[dim]] = i;
      applyContract(dim + 1);
    }
  };

  applyBatch(0);
  return output;
}

// TODO(jreiffers): Unify this with DotGeneral.
InterpreterValue dot(InterpreterState& state, mhlo::DotOp op,
                     const InterpreterValue& lhs, const InterpreterValue& rhs) {
  ShapedType ty = op->getResultTypes()[0];
  auto result = lhs.typedAlike(ty.getShape());

  if (lhs.view().rank() == 1 && rhs.view().rank() == 1) {
    dispatchScalarType(ty, [&](auto dummy) {
      using T = decltype(dummy);
      using TT = TensorOrMemref<T>;
      auto lhsTensor = std::get<TT>(lhs.storage);
      auto rhsTensor = std::get<TT>(rhs.storage);

      T product = 0;
      for (int64_t i = 0; i < lhsTensor.view.sizes[0]; ++i) {
        product += lhsTensor.at(i) * rhsTensor.at(i);
      }
      std::get<TT>(result.storage).at({}) += product;
    });
  } else if (lhs.view().rank() == 2 && rhs.view().rank() == 1) {
    dispatchScalarType(ty, [&](auto dummy) {
      using TT = TensorOrMemref<decltype(dummy)>;
      auto lhsTensor = std::get<TT>(lhs.storage);
      auto rhsTensor = std::get<TT>(rhs.storage);
      auto resultTensor = std::get<TT>(result.storage);
      for (int64_t i = 0; i < resultTensor.view.sizes[0]; ++i) {
        for (int64_t j = 0; j < rhsTensor.view.sizes[0]; ++j) {
          resultTensor.at(i) += lhsTensor.at({i, j}) * rhsTensor.at(j);
        }
      }
    });
  } else if (lhs.view().rank() == 2 && rhs.view().rank() == 2) {
    dispatchScalarType(ty, [&](auto dummy) {
      using TT = TensorOrMemref<decltype(dummy)>;
      auto lhsTensor = std::get<TT>(lhs.storage);
      auto rhsTensor = std::get<TT>(rhs.storage);
      auto resultTensor = std::get<TT>(result.storage);
      for (int64_t i = 0; i < resultTensor.view.sizes[0]; ++i) {
        for (int64_t j = 0; j < resultTensor.view.sizes[1]; ++j) {
          for (int64_t k = 0; k < lhsTensor.view.sizes[1]; ++k) {
            resultTensor.at({i, j}) +=
                lhsTensor.at({i, k}) * rhsTensor.at({k, j});
          }
        }
      }
    });
  } else {
    state.addFailure("unsupported dot");
  }

  return result;
}

InterpreterValue dotGeneral(InterpreterState&, mhlo::DotGeneralOp op,
                            InterpreterValue lhs, InterpreterValue rhs) {
  const auto& dims = op.getDotDimensionNumbers();
  return dotGeneralImpl(
      lhs, rhs, dims.getLhsContractingDimensions(),
      dims.getRhsContractingDimensions(), dims.getLhsBatchingDimensions(),
      dims.getRhsBatchingDimensions(),
      op->getResultTypes()[0].cast<ShapedType>().getElementType());
}

InterpreterValue computeReshapeShape(InterpreterState&,
                                     mhlo::ComputeReshapeShapeOp,
                                     const InterpreterValue& numElements,
                                     const InterpreterValue& dynamicShape) {
  auto out = dynamicShape.clone();
  SmallVector<int64_t> dynamicIndex;
  int64_t staticProduct = 1;
  for (const auto& index : dynamicShape.view().indices()) {
    auto value = dynamicShape.extractElement(index).asInt();
    if (value == -1) {
      dynamicIndex = index;
    } else {
      staticProduct *= value;
    }
  }

  if (!dynamicIndex.empty()) {
    std::visit(
        [&](auto& it) {
          if constexpr (is_tensor_or_memref_v<decltype(it)>) {
            it.at(dynamicIndex) = numElements.asInt() / staticProduct;
          } else {
            llvm_unreachable("invalid input");
          }
        },
        out.storage);
  }
  return out;
}

// TODO(jreiffers): Migrate remaining ops to the safer signature.
REGISTER_MLIR_INTERPRETER_OP("mhlo.dynamic_gather", gather);
REGISTER_MLIR_INTERPRETER_OP("mhlo.gather", gather);
REGISTER_MLIR_INTERPRETER_OP("mhlo.return", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP("mhlo.tuple", makeTuple);
REGISTER_MLIR_INTERPRETER_OP(bitcastConvert);
REGISTER_MLIR_INTERPRETER_OP(broadcastInDim);
REGISTER_MLIR_INTERPRETER_OP(clamp);
REGISTER_MLIR_INTERPRETER_OP(compare);
REGISTER_MLIR_INTERPRETER_OP(computeReshapeShape);
REGISTER_MLIR_INTERPRETER_OP(concatenate);
REGISTER_MLIR_INTERPRETER_OP(constant);
REGISTER_MLIR_INTERPRETER_OP(convert);
REGISTER_MLIR_INTERPRETER_OP(copy);
REGISTER_MLIR_INTERPRETER_OP(dot);
REGISTER_MLIR_INTERPRETER_OP(dotGeneral);
REGISTER_MLIR_INTERPRETER_OP(dynamicSlice);
REGISTER_MLIR_INTERPRETER_OP(dynamicUpdateSlice);
REGISTER_MLIR_INTERPRETER_OP(fusion);
REGISTER_MLIR_INTERPRETER_OP(getTupleElement);
REGISTER_MLIR_INTERPRETER_OP(iota);
REGISTER_MLIR_INTERPRETER_OP(mhloCase);
REGISTER_MLIR_INTERPRETER_OP(mhloWhile);
REGISTER_MLIR_INTERPRETER_OP(pad);
REGISTER_MLIR_INTERPRETER_OP(reduce);
REGISTER_MLIR_INTERPRETER_OP(reshape);
REGISTER_MLIR_INTERPRETER_OP(scatter);
REGISTER_MLIR_INTERPRETER_OP(select);
REGISTER_MLIR_INTERPRETER_OP(slice);
REGISTER_MLIR_INTERPRETER_OP(sort);
REGISTER_MLIR_INTERPRETER_OP(transpose);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
