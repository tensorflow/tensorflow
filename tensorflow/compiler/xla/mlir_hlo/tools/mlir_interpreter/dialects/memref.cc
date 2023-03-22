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

#include "mlir/Dialect/MemRef/IR/MemRef.h"

// clang-format erroneously puts the MemRef header above.
#include <algorithm>  // NOLINT
#include <iterator>   // NOLINT
#include <limits>     // NOLINT
#include <variant>    // NOLINT

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue load(InterpreterState& state, memref::LoadOp,
                      const InterpreterValue& memref,
                      ArrayRef<int64_t> indices) {
  if (!memref.buffer()) {
    state.addFailure("null pointer dereference.");
    return {};
  }
  if (!memref.view().inBounds(indices)) {
    state.addFailure("array index out of bounds");
    return {};
  }
  return memref.extractElement(indices);
}

void store(InterpreterState& state, memref::StoreOp,
           const InterpreterValue& value, InterpreterValue memref,
           ArrayRef<int64_t> indices) {
  if (memref.view().inBounds(indices)) {
    memref.insertElement(indices, value);
  } else {
    state.addFailure("array index out of bounds");
  }
}

// TODO(jreiffers): Support symbol operands.
InterpreterValue alloc(InterpreterState& state, memref::AllocOp alloc,
                       ArrayRef<int64_t> dynamicSizes) {
  auto ty = alloc->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = replaceDynamicVals(ty.getShape(), dynamicSizes);
  auto result = InterpreterValue::makeTensor(ty.getElementType(), shape);
  if (auto* stats = state.getOptions().stats) {
    stats->heapSize += result.buffer()->getByteSize();
    stats->peakHeapSize = std::max(stats->peakHeapSize, stats->heapSize);
    ++stats->numAllocations;
  }
  return result;
}

InterpreterValue allocA(InterpreterState&, memref::AllocaOp alloc,
                        ArrayRef<int64_t> dynamicSizes) {
  auto ty = alloc->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = replaceDynamicVals(ty.getShape(), dynamicSizes);
  auto result = InterpreterValue::makeTensor(ty.getElementType(), shape);
  result.buffer()->setIsAlloca();
  return result;
}

void dealloc(InterpreterState& state, memref::DeallocOp,
             InterpreterValue memref) {
  if (!memref.buffer()) {
    state.addFailure("attempting to deallocate null pointer.");
    return;
  }
  auto buffer = memref.buffer();
  const auto& view = memref.view();
  if (auto* stats = state.getOptions().stats) {
    stats->heapSize -= buffer->getByteSize();
    ++stats->numDeallocations;
  }
  if (view.getNumElements() * memref.getByteSizeOfElement() !=
      buffer->getByteSize()) {
    state.addFailure("Attempting to deallocate a subview");
  } else if (!state.getOptions().disableDeallocations) {
    buffer->deallocate();
  }
}

void copy(InterpreterState&, memref::CopyOp, InterpreterValue source,
          InterpreterValue dest) {
  dest.fill([&](llvm::ArrayRef<int64_t> indices) {
    return source.extractElement(indices);
  });
}

InterpreterValue subview(InterpreterState& state, memref::SubViewOp subview,
                         const InterpreterValue& memref,
                         ArrayRef<int64_t> dynamicOffsets,
                         ArrayRef<int64_t> dynamicSizes,
                         ArrayRef<int64_t> dynamicStrides) {
  auto v = extractOffsetsSizesStrides(dynamicOffsets, dynamicSizes,
                                      dynamicStrides, subview);
  auto out = memref;
  auto& outView = out.view();
  if (!outView.subview(v.offsets, v.sizes, v.strides).succeeded()) {
    state.addFailure("subview out of bounds");
    return {};
  }

  if (subview.getResult().getType().getRank() == outView.rank()) {
    return out;
  }

  auto shape = subview.getResult().getType().getShape();
  // TODO(jreiffers): Check why subview.getDroppedDims() yields the wrong shape
  // here for 1x2x2x3 (-> 1x2x1x3) -> 1x2x3 (claiming 0 is dropped).
  int64_t dim = 0;
  while (dim < outView.rank() && dim < shape.size()) {
    if (shape[dim] != 1 && outView.sizes[dim] == 1) {
      outView.sizes.erase(outView.sizes.begin() + dim);
      outView.strides.erase(outView.strides.begin() + dim);
    } else {
      assert((shape[dim] < 0 || outView.sizes[dim] == shape[dim]) &&
             "expected static size to match");
      ++dim;
    }
  }
  while (dim < outView.rank()) {
    assert(outView.sizes.back() == 1 && "expected remaining dims to be 1");
    outView.sizes.pop_back();
    outView.strides.pop_back();
  }
  return out;
}

llvm::SmallVector<InterpreterValue> collapseShape(
    InterpreterState& state, memref::CollapseShapeOp collapse,
    const InterpreterValue& memref) {
  const BufferView& inputView = memref.view();
  InterpreterValue out = memref;
  auto& outView = out.view();
  outView.sizes.clear();
  outView.strides.clear();

  for (const auto& group : collapse.getReassociationIndices()) {
    if (auto stride = inputView.getCollapsedStride(group)) {
      outView.strides.push_back(*stride);
      int64_t& size = outView.sizes.emplace_back(1);
      for (int64_t dim : group) size *= inputView.sizes[dim];
    } else {
      state.addFailure("cannot collapse dimensions without a common stride");
      return {};
    }
  }

  return {out};
}

InterpreterValue cast(InterpreterState&, memref::CastOp,
                      InterpreterValue memref) {
  return memref;
}

// TODO(jreiffers): Implement full expand_shape support.
InterpreterValue expandShape(InterpreterState& state, memref::ExpandShapeOp op,
                             InterpreterValue memref) {
  BufferView inputView = memref.view();
  auto outTy = op->getResultTypes()[0].template cast<MemRefType>();
  if (outTy.getNumDynamicDims() > 0) {
    state.addFailure("dynamic dimensions unsupported.");
    return {};
  }

  InterpreterValue out = memref;
  auto& outView = out.view();
  outView.strides.clear();
  outView.sizes = llvm::to_vector(outTy.getShape());
  int64_t dummy;
  if (!getStridesAndOffset(outTy, outView.strides, dummy).succeeded()) {
    if (inputView.strides != BufferView::getDefaultStrides(inputView.sizes)) {
      state.addFailure("unsupported strides");
      return {};
    }
    outView.strides = BufferView::getDefaultStrides(outView.sizes);
  }

  return out;
}

InterpreterValue getGlobal(InterpreterState& state,
                           memref::GetGlobalOp getGlobal) {
  auto global = llvm::cast<memref::GlobalOp>(
      state.getSymbols().lookup(getGlobal.getName()));

  auto value = global.getConstantInitValue();
  assert(value && "mutable globals are not implemented");

  auto ty = getGlobal->getResultTypes()[0].cast<ShapedType>();
  return dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    auto values = value.getValues<decltype(dummy)>();
    auto result = TensorOrMemref<decltype(dummy)>::empty(ty.getShape());
    auto valueIt = values.begin();
    for (const auto& index : result.view.indices()) {
      result.at(index) = *valueIt;
      ++valueIt;
    }
    return {result};
  });
}

int64_t dim(InterpreterState& state, memref::DimOp,
            const InterpreterValue& memref, int64_t dim) {
  return dimImpl(memref, dim, state);
}

REGISTER_MLIR_INTERPRETER_OP(alloc);
REGISTER_MLIR_INTERPRETER_OP(allocA);
REGISTER_MLIR_INTERPRETER_OP(cast);
REGISTER_MLIR_INTERPRETER_OP(collapseShape);
REGISTER_MLIR_INTERPRETER_OP(copy);
REGISTER_MLIR_INTERPRETER_OP(dealloc);
REGISTER_MLIR_INTERPRETER_OP(dim);
REGISTER_MLIR_INTERPRETER_OP(expandShape);
REGISTER_MLIR_INTERPRETER_OP(getGlobal);
REGISTER_MLIR_INTERPRETER_OP(load);
REGISTER_MLIR_INTERPRETER_OP(store);
REGISTER_MLIR_INTERPRETER_OP(subview);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
