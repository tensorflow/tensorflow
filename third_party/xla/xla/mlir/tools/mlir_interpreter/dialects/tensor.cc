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

#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project

#include <cassert>   // NOLINT
#include <cstdint>   // NOLINT
#include <optional>  // NOLINT

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

int64_t Dim(InterpreterState& state, tensor::DimOp,
            const InterpreterValue& tensor, int64_t dim) {
  return DimImpl(tensor, dim, state);
}

InterpreterValue Empty(InterpreterState&, tensor::EmptyOp op,
                       ArrayRef<int64_t> dynamic_sizes) {
  auto ty = op->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = ReplaceDynamicVals(ty.getShape(), dynamic_sizes);
  return InterpreterValue::MakeTensor(ty.getElementType(), shape);
}

InterpreterValue Extract(InterpreterState& state, tensor::ExtractOp,
                         InterpreterValue tensor, ArrayRef<int64_t> indices) {
  if (!tensor.View().InBounds(indices)) {
    state.AddFailure("array index out of bounds");
    return {};
  }
  return tensor.ExtractElement(indices);
}

InterpreterValue FromElements(InterpreterState&, tensor::FromElementsOp op,
                              MutableArrayRef<InterpreterValue> elements) {
  auto ty = op->getResultTypes().front().cast<mlir::ShapedType>();
  auto result = InterpreterValue::MakeTensor(ty.getElementType(),
                                             llvm::to_vector(ty.getShape()));
  for (auto [index, element] : llvm::zip(result.View().Indices(), elements)) {
    result.InsertElement(index, element);
  }
  return result;
}

llvm::SmallVector<InterpreterValue> CollapseShape(
    InterpreterState&, tensor::CollapseShapeOp op,
    const InterpreterValue& tensor) {
  SmallVector<int64_t> sizes;
  for (const auto& indices : op.getReassociationIndices()) {
    int64_t size = 1;
    for (auto dim : indices) {
      size *= tensor.View().sizes[dim];
    }
    sizes.push_back(size);
  }
  return {ReshapeTensor(tensor, sizes)};
}

llvm::SmallVector<InterpreterValue> ExpandShape(
    InterpreterState&, tensor::ExpandShapeOp op,
    const InterpreterValue& tensor) {
  auto ty = cast<ShapedType>(op->getResultTypes().front());
  auto sizes = llvm::to_vector(ty.getShape());
  for (const auto& [src_index, dst_indices] :
       llvm::enumerate(op.getReassociationIndices())) {
    int64_t size = tensor.View().sizes[src_index];
    std::optional<int64_t> dyn_index = std::nullopt;
    for (auto dim : dst_indices) {
      if (sizes[dim] < 0) {
        dyn_index = dim;
      } else {
        size /= sizes[dim];
      }
    }
    if (dyn_index) {
      sizes[*dyn_index] = size;
    }
  }
  return {ReshapeTensor(tensor, sizes)};
}

llvm::SmallVector<InterpreterValue> ExtractSlice(
    InterpreterState&, tensor::ExtractSliceOp extract, InterpreterValue tensor,
    ArrayRef<int64_t> dynamic_offsets, ArrayRef<int64_t> dynamic_sizes,
    ArrayRef<int64_t> dynamic_strides) {
  auto v = ExtractOffsetsSizesStrides(dynamic_offsets, dynamic_sizes,
                                      dynamic_strides, extract);
  int64_t rank = v.offsets.size();
  auto out = tensor.TypedAlike(v.sizes);
  out.Fill([&](llvm::ArrayRef<int64_t> indices) {
    llvm::SmallVector<int64_t> src_indices;
    for (int64_t i = 0; i < rank; ++i) {
      src_indices.push_back(indices[i] * v.strides[i] + v.offsets[i]);
    }
    return tensor.ExtractElement(src_indices);
  });

  int64_t num_dropped = 0;
  auto& out_view = out.View();

  // TODO(jreiffers): Figure out why getDroppedDims fails here when there's
  // no rank reduction and the output has a dynamic shape.
  int64_t dim = 0;
  const auto& result_sizes = extract.getResultType().getShape();
  const auto& static_sizes = extract.getStaticSizes();
  while (dim < out_view.Rank()) {
    if (static_sizes[num_dropped + dim] == 1 &&
        (dim >= result_sizes.size() || result_sizes[dim] != 1)) {
      out_view.sizes.erase(out_view.sizes.begin() + dim);
      out_view.strides.erase(out_view.strides.begin() + dim);
      ++num_dropped;
    } else {
      ++dim;
    }
  }

  return {out};
}

template <typename Op>
llvm::SmallVector<InterpreterValue> InsertSlice(
    InterpreterState&, Op insert, InterpreterValue src, InterpreterValue dest,
    ArrayRef<int64_t> dynamic_offsets, ArrayRef<int64_t> dynamic_sizes,
    ArrayRef<int64_t> dynamic_strides) {
  // parallel_insert_slice actually writes to its destination.
  if (insert->getNumResults() == 1) {
    dest = dest.Clone();
  }
  auto v = ExtractOffsetsSizesStrides(dynamic_offsets, dynamic_sizes,
                                      dynamic_strides, insert);

  auto static_sizes = insert.getStaticSizes();
  llvm::SmallVector<int64_t> inserted_dims;
  auto& src_view = src.View();
  auto* src_size_it = src_view.sizes.begin();
  for (auto [dim, size] : llvm::enumerate(static_sizes)) {
    if (src_size_it == src_view.sizes.end() ||
        (*src_size_it != size && size >= 0)) {
      assert(size == 1 && "Can only insert unit dims");
      inserted_dims.push_back(dim);
    } else {
      ++src_size_it;
    }
  }

  for (const auto& src_indices : src_view.Indices()) {
    llvm::SmallVector<int64_t> src_with_inserted_dims = src_indices;
    for (int64_t dim : inserted_dims) {
      src_with_inserted_dims.insert(src_with_inserted_dims.begin() + dim, 0);
    }
    llvm::SmallVector<int64_t> dst_indices;
    for (auto [src_index, stride, offset] :
         llvm::zip(src_with_inserted_dims, v.strides, v.offsets)) {
      dst_indices.push_back(src_index * stride + offset);
    }
    dest.InsertElement(dst_indices, src.ExtractElement(src_indices));
  }
  if (insert->getNumResults() == 1) {
    return {dest};
  }
  return {};
}

InterpreterValue Generate(InterpreterState& state, tensor::GenerateOp generate,
                          ArrayRef<int64_t> dynamic_sizes) {
  auto ty = generate->getResultTypes().front().cast<ShapedType>();
  auto sizes = ReplaceDynamicVals(ty.getShape(), dynamic_sizes);

  auto result = InterpreterValue::MakeTensor(ty.getElementType(), sizes);
  result.Fill([&](ArrayRef<int64_t> indices) -> InterpreterValue {
    auto values = Interpret(state, generate.getRegion(),
                            PackInterpreterValues<int64_t>(indices));
    if (state.HasFailure()) {
      return {result.ExtractElement(indices)};
    }
    return values.front();
  });
  return {result};
}

InterpreterValue Insert(InterpreterState& state, tensor::InsertOp,
                        const InterpreterValue& value,
                        const InterpreterValue& tensor,
                        ArrayRef<int64_t> indices) {
  auto result = tensor.Clone();
  if (result.View().InBounds(indices)) {
    result.InsertElement(indices, value);
  } else {
    state.AddFailure("array index out of bounds");
  }
  return result;
}

InterpreterValue Pad(InterpreterState& state, tensor::PadOp pad,
                     InterpreterValue tensor, ArrayRef<int64_t> dynamic_lows,
                     ArrayRef<int64_t> dynamic_highs) {
  auto lows = ReplaceDynamicVals(pad.getStaticLow(), dynamic_lows);
  auto highs = ReplaceDynamicVals(pad.getStaticHigh(), dynamic_highs);

  auto& view = tensor.View();
  llvm::SmallVector<int64_t> result_sizes;
  for (auto [size, low, high] : llvm::zip(view.sizes, lows, highs)) {
    result_sizes.push_back(size + low + high);
  }

  auto result = tensor.TypedAlike(result_sizes);
  result.Fill([&](llvm::ArrayRef<int64_t> out_index) -> InterpreterValue {
    llvm::SmallVector<int64_t> in_index;
    for (auto [index, low] : llvm::zip(out_index, lows)) {
      in_index.push_back(index - low);
    }
    if (view.InBounds(in_index)) {
      return tensor.ExtractElement(in_index);
    }
    return Interpret(state, pad.getRegion(), PackInterpreterValues(out_index))
        .front();
  });
  return result;
}

REGISTER_MLIR_INTERPRETER_OP("tensor.cast",
                             "builtin.unrealized_conversion_cast");
REGISTER_MLIR_INTERPRETER_OP("tensor.yield", NoOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(CollapseShape);
REGISTER_MLIR_INTERPRETER_OP(Dim);
REGISTER_MLIR_INTERPRETER_OP(Empty);
REGISTER_MLIR_INTERPRETER_OP(ExpandShape);
REGISTER_MLIR_INTERPRETER_OP(Extract);
REGISTER_MLIR_INTERPRETER_OP(ExtractSlice);
REGISTER_MLIR_INTERPRETER_OP(FromElements);
REGISTER_MLIR_INTERPRETER_OP(Generate);
REGISTER_MLIR_INTERPRETER_OP(Insert);
REGISTER_MLIR_INTERPRETER_OP(InsertSlice<tensor::InsertSliceOp>);
REGISTER_MLIR_INTERPRETER_OP(InsertSlice<tensor::ParallelInsertSliceOp>);
REGISTER_MLIR_INTERPRETER_OP(Pad);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
