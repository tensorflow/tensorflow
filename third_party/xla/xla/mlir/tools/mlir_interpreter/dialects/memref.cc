/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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
#include <cassert>    // NOLINT
#include <cstdint>    // NOLINT
#include <utility>    // NOLINT

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue Load(InterpreterState& state, memref::LoadOp,
                      const InterpreterValue& memref,
                      ArrayRef<int64_t> indices) {
  if (!memref.GetBuffer()) {
    state.AddFailure("null pointer dereference.");
    return {};
  }
  if (!memref.View().InBounds(indices)) {
    state.AddFailure("array index out of bounds");
    return {};
  }
  return memref.ExtractElement(indices);
}

void Store(InterpreterState& state, memref::StoreOp,
           const InterpreterValue& value, InterpreterValue memref,
           ArrayRef<int64_t> indices) {
  if (memref.View().InBounds(indices)) {
    memref.InsertElement(indices, value);
  } else {
    state.AddFailure("array index out of bounds");
  }
}

// TODO(jreiffers): Support symbol operands.
InterpreterValue Alloc(InterpreterState& state, memref::AllocOp alloc,
                       ArrayRef<int64_t> dynamic_sizes) {
  auto ty = cast<ShapedType>(alloc->getResultTypes().front());
  auto shape = ReplaceDynamicVals(ty.getShape(), dynamic_sizes);
  auto result =
      InterpreterValue::MakeTensor(ty.getElementType(), std::move(shape));
  if (auto* stats = state.GetOptions().stats) {
    stats->heap_size += result.GetBuffer()->GetByteSize();
    stats->peak_heap_size = std::max(stats->peak_heap_size, stats->heap_size);
    ++stats->num_allocations;
  }
  result.GetBuffer()->SetAllocatedBy(alloc);
  return result;
}

InterpreterValue AllocA(InterpreterState&, memref::AllocaOp alloc,
                        ArrayRef<int64_t> dynamic_sizes) {
  auto ty = cast<ShapedType>(alloc->getResultTypes().front());
  auto shape = ReplaceDynamicVals(ty.getShape(), dynamic_sizes);
  auto result =
      InterpreterValue::MakeTensor(ty.getElementType(), std::move(shape));
  result.GetBuffer()->SetIsAlloca();
  result.GetBuffer()->SetAllocatedBy(alloc);
  return result;
}

void Dealloc(InterpreterState& state, memref::DeallocOp op,
             InterpreterValue memref) {
  if (!memref.GetBuffer()) {
    state.AddFailure("attempting to deallocate null pointer.");
    return;
  }
  auto buffer = memref.GetBuffer();
  const auto& view = memref.View();
  if (auto* stats = state.GetOptions().stats) {
    stats->heap_size -= buffer->GetByteSize();
    ++stats->num_deallocations;
  }
  if (view.GetNumElements() * memref.GetByteSizeOfElement() !=
      buffer->GetByteSize()) {
    state.AddFailure("Attempting to deallocate a subview");
  } else if (!state.GetOptions().disable_deallocations) {
    buffer->Deallocate(op);
  }
}

void Copy(InterpreterState&, memref::CopyOp, InterpreterValue source,
          InterpreterValue dest) {
  dest.Fill([&](llvm::ArrayRef<int64_t> indices) {
    return source.ExtractElement(indices);
  });
}

InterpreterValue Subview(InterpreterState& state, memref::SubViewOp subview,
                         const InterpreterValue& memref,
                         ArrayRef<int64_t> dynamic_offsets,
                         ArrayRef<int64_t> dynamic_sizes,
                         ArrayRef<int64_t> dynamic_strides) {
  auto v = ExtractOffsetsSizesStrides(dynamic_offsets, dynamic_sizes,
                                      dynamic_strides, subview);
  auto out = memref;
  auto& out_view = out.View();
  if (!out_view.Subview(v.offsets, v.sizes, v.strides).succeeded()) {
    state.AddFailure("subview out of bounds");
    return {};
  }

  if (subview.getResult().getType().getRank() == out_view.num_dimensions()) {
    return out;
  }

  auto shape = subview.getResult().getType().getShape();
  // TODO(jreiffers): Check why subview.getDroppedDims() yields the wrong shape
  // here for 1x2x2x3 (-> 1x2x1x3) -> 1x2x3 (claiming 0 is dropped).
  int64_t dim = 0;
  while (dim < out_view.num_dimensions() && dim < shape.size()) {
    if (shape[dim] != 1 && out_view.sizes[dim] == 1) {
      out_view.sizes.erase(out_view.sizes.begin() + dim);
      out_view.strides.erase(out_view.strides.begin() + dim);
    } else {
      assert((shape[dim] < 0 || out_view.sizes[dim] == shape[dim]) &&
             "expected static size to match");
      ++dim;
    }
  }
  while (dim < out_view.num_dimensions()) {
    assert(out_view.sizes.back() == 1 && "expected remaining dims to be 1");
    out_view.sizes.pop_back();
    out_view.strides.pop_back();
  }
  return out;
}

llvm::SmallVector<InterpreterValue> CollapseShape(
    InterpreterState& state, memref::CollapseShapeOp collapse,
    const InterpreterValue& memref) {
  const BufferView& input_view = memref.View();
  InterpreterValue out = memref;
  auto& out_view = out.View();
  out_view.sizes.clear();
  out_view.strides.clear();

  for (const auto& group : collapse.getReassociationIndices()) {
    if (auto stride = input_view.GetCollapsedStride(group)) {
      out_view.strides.push_back(*stride);
      int64_t& size = out_view.sizes.emplace_back(1);
      for (int64_t dim : group) size *= input_view.sizes[dim];
    } else {
      state.AddFailure("cannot collapse dimensions without a common stride");
      return {};
    }
  }

  return {out};
}

InterpreterValue Cast(InterpreterState&, memref::CastOp,
                      InterpreterValue memref) {
  return memref;
}

// TODO(jreiffers): Implement full expand_shape support.
InterpreterValue ExpandShape(InterpreterState& state, memref::ExpandShapeOp op,
                             InterpreterValue memref) {
  BufferView input_view = memref.View();
  auto out_ty = cast<MemRefType>(op->getResultTypes()[0]);
  if (out_ty.getNumDynamicDims() > 0) {
    state.AddFailure("dynamic dimensions unsupported.");
    return {};
  }

  InterpreterValue out = memref;
  auto& out_view = out.View();
  out_view.strides.clear();
  out_view.sizes = llvm::to_vector(out_ty.getShape());
  int64_t dummy;
  if (!out_ty.getStridesAndOffset(out_view.strides, dummy).succeeded()) {
    if (input_view.strides != BufferView::GetDefaultStrides(input_view.sizes)) {
      state.AddFailure("unsupported strides");
      return {};
    }
    out_view.strides = BufferView::GetDefaultStrides(out_view.sizes);
  }

  return out;
}

InterpreterValue GetGlobal(InterpreterState& state,
                           memref::GetGlobalOp get_global) {
  auto global = llvm::cast<memref::GlobalOp>(
      state.GetSymbols().lookup(get_global.getName()));

  auto value = global.getConstantInitValue();
  assert(value && "mutable globals are not implemented");

  auto ty = cast<ShapedType>(get_global->getResultTypes()[0]);
  return DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    auto values = value.getValues<decltype(dummy)>();
    auto result = TensorOrMemref<decltype(dummy)>::Empty(ty.getShape());
    auto value_it = values.begin();
    for (const auto& index : result.view.Indices()) {
      result.at(index) = *value_it;
      ++value_it;
    }
    return {result};
  });
}

int64_t Dim(InterpreterState& state, memref::DimOp,
            const InterpreterValue& memref, int64_t dim) {
  return DimImpl(memref, dim, state);
}

REGISTER_MLIR_INTERPRETER_OP(Alloc);
REGISTER_MLIR_INTERPRETER_OP(AllocA);
REGISTER_MLIR_INTERPRETER_OP(Cast);
REGISTER_MLIR_INTERPRETER_OP(CollapseShape);
REGISTER_MLIR_INTERPRETER_OP(Copy);
REGISTER_MLIR_INTERPRETER_OP(Dealloc);
REGISTER_MLIR_INTERPRETER_OP(Dim);
REGISTER_MLIR_INTERPRETER_OP(ExpandShape);
REGISTER_MLIR_INTERPRETER_OP(GetGlobal);
REGISTER_MLIR_INTERPRETER_OP(Load);
REGISTER_MLIR_INTERPRETER_OP(Store);
REGISTER_MLIR_INTERPRETER_OP(Subview);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
