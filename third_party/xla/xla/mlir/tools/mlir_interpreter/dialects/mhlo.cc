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

// NOTE: These ops aim to match the StableHLO spec and the behavior of the XLA
// compiler. For op semantics and a reference implementation, check the
// StableHLO repository (the spec is here:
// https://github.com/openxla/stablehlo/blob/main/docs/spec.md).

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/dialects/comparators.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/cwise_math.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue MakeTuple(MutableArrayRef<InterpreterValue> values) {
  Tuple result;
  for (auto& value : values) {
    result.values.push_back(
        std::make_shared<InterpreterValue>(std::move(value)));
  }
  return {result};
}

InterpreterValue GetTupleElement(InterpreterState&, mhlo::GetTupleElementOp get,
                                 const InterpreterValue& tuple) {
  return *std::get<Tuple>(tuple.storage).values[get.getIndex()];
}

InterpreterValue BitcastConvert(InterpreterState&, mhlo::BitcastConvertOp op,
                                const InterpreterValue& in) {
  ShapedType ty = cast<ShapedType>(op->getResultTypes()[0]);
  auto result = DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    TensorOrMemref<decltype(dummy)> result;
    result.view = {};
    result.buffer = in.Clone().GetBuffer();
    return {result};
  });
  auto& out_view = result.View();
  out_view.strides = BufferView::GetDefaultStrides(ty.getShape());
  out_view.sizes = llvm::to_vector(ty.getShape());
  return result;
}

InterpreterValue BroadcastInDim(InterpreterState&,
                                mhlo::BroadcastInDimOp broadcast,
                                InterpreterValue in) {
  auto boradcast_dims = broadcast.getBroadcastDimensions().getValues<int64_t>();
  const auto& in_sizes = in.View().sizes;
  auto out = in.TypedAlike(
      cast<ShapedType>(broadcast->getResultTypes()[0]).getShape());
  // TODO(jreiffers): Skip the copy.
  out.Fill([&](llvm::ArrayRef<int64_t> out_indices) {
    llvm::SmallVector<int64_t> in_indices;
    for (auto [inDim, out_dim] : llvm::enumerate(boradcast_dims)) {
      in_indices.push_back(in_sizes[inDim] == 1 ? 0 : out_indices[out_dim]);
    }
    return in.ExtractElement(in_indices);
  });
  return out;
}

InterpreterValue Clamp(InterpreterState&, mhlo::ClampOp,
                       const InterpreterValue& lb, const InterpreterValue& arg,
                       const InterpreterValue& ub) {
  auto result = arg.Clone();
  for (const auto& index : arg.View().Indices()) {
    auto lb_scalar = lb.IsTensor() ? lb.ExtractElement(index) : lb;
    auto ub_scalar = ub.IsTensor() ? ub.ExtractElement(index) : ub;
    assert(arg.IsTensor() && "clamp only bcasts scalar bounds");
    auto arg_scalar = arg.ExtractElement(index);
    auto result_scalar = ApplyCwiseBinaryMap<Min>(
        ApplyCwiseBinaryMap<Max>(arg_scalar, lb_scalar), ub_scalar);
    result.InsertElement(index, result_scalar);
  }
  return result;
}

InterpreterValue Concatenate(InterpreterState&, mhlo::ConcatenateOp concat,
                             ArrayRef<InterpreterValue> vals) {
  uint64_t dim = concat.getDimension();
  auto sizes = vals[0].View().sizes;
  sizes[dim] = 0;
  for (const auto& val : vals) {
    sizes[dim] += val.View().sizes[dim];
  }

  auto result = vals[0].TypedAlike(sizes);
  int64_t offset = 0;
  for (const auto& val : vals) {
    for (auto index : val.View().Indices()) {
      auto item = val.ExtractElement(index);
      index[dim] += offset;
      result.InsertElement(index, item);
    }
    offset += val.View().sizes[dim];
  }
  return result;
}

InterpreterValue Reshape(InterpreterState&, mhlo::ReshapeOp reshape,
                         const InterpreterValue& in) {
  auto ty = reshape->getResultTypes()[0].cast<mlir::ShapedType>();
  return ReshapeTensor(in, ty.getShape());
}

llvm::SmallVector<int64_t> ClampStarts(ArrayRef<int64_t> starts,
                                       ArrayRef<int64_t> slice_sizes,
                                       ArrayRef<int64_t> tensor_sizes) {
  llvm::SmallVector<int64_t> result;
  for (auto [start, slice_size, tensor_size] :
       llvm::zip(starts, slice_sizes, tensor_sizes)) {
    result.push_back(
        std::max(int64_t{0}, std::min(tensor_size - slice_size, start)));
  }
  return result;
}

InterpreterValue DynamicSlice(InterpreterState&, mhlo::DynamicSliceOp slice,
                              const InterpreterValue& in,
                              ArrayRef<int64_t> starts) {
  auto result = in.TypedAlike(
      llvm::to_vector(slice.getSliceSizes().getValues<int64_t>()));
  auto clamed_starts =
      ClampStarts(starts, result.View().sizes, in.View().sizes);
  // TODO(jreiffers): Skip the copy.
  result.Fill([&](llvm::ArrayRef<int64_t> out_indices) {
    llvm::SmallVector<int64_t> in_indices;
    for (auto [start, index] : llvm::zip(clamed_starts, out_indices)) {
      in_indices.push_back(start + index);
    }
    return in.ExtractElement(in_indices);
  });
  return result;
}

InterpreterValue DynamicUpdateSlice(InterpreterState&,
                                    mhlo::DynamicUpdateSliceOp,
                                    const InterpreterValue& in,
                                    const InterpreterValue& updates,
                                    ArrayRef<int64_t> starts) {
  auto result = in.Clone();
  auto clamed_starts =
      ClampStarts(starts, updates.View().sizes, result.View().sizes);
  for (auto in_indices : updates.View().Indices()) {
    llvm::SmallVector<int64_t> out_indices;
    for (auto [start, index] : llvm::zip(clamed_starts, in_indices)) {
      out_indices.push_back(start + index);
    }
    result.InsertElement(out_indices, updates.ExtractElement(in_indices));
  }
  return result;
}

InterpreterValue Slice(InterpreterState&, mhlo::SliceOp slice,
                       InterpreterValue in) {
  auto starts = slice.getStartIndices().getValues<int64_t>();
  auto limits = slice.getLimitIndices().getValues<int64_t>();
  auto strides = slice.getStrides().getValues<int64_t>();

  llvm::SmallVector<int64_t> sizes;
  for (auto [start, limit, stride] : llvm::zip(starts, limits, strides)) {
    sizes.push_back(((limit - start) + (stride - 1)) / stride);
  }
  // TODO(jreiffers): Skip the copy.
  auto result = in.TypedAlike(sizes);
  result.Fill([&](llvm::ArrayRef<int64_t> out_indices) {
    llvm::SmallVector<int64_t> in_indices;
    for (auto [start, stride, index] :
         llvm::zip(starts, strides, out_indices)) {
      in_indices.push_back(start + stride * index);
    }
    return in.ExtractElement(in_indices);
  });
  return result;
}

llvm::SmallVector<InterpreterValue> Constant(InterpreterState&,
                                             mhlo::ConstantOp constant) {
  auto ty = constant->getResultTypes()[0].cast<ShapedType>();
  return {DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    if (ty.getElementType().isUnsignedInteger()) {
      if constexpr (!std::is_same_v<decltype(dummy), bool> &&
                    std::is_integral_v<decltype(dummy)>) {
        auto values =
            constant.getValue()
                .getValues<
                    typename std::make_unsigned<decltype(dummy)>::type>();
        auto result = TensorOrMemref<decltype(dummy)>::Empty(ty.getShape());
        auto value_it = values.begin();
        for (const auto& index : result.view.Indices()) {
          result.at(index) = *value_it;
          ++value_it;
        }
        return {result};
      } else {
        llvm_unreachable("invalid input");
      }
    } else {
      auto values = constant.getValue().getValues<decltype(dummy)>();
      auto result = TensorOrMemref<decltype(dummy)>::Empty(ty.getShape());
      auto value_it = values.begin();
      for (const auto& index : result.view.Indices()) {
        result.at(index) = *value_it;
        ++value_it;
      }
      return {result};
    }
  })};
}

InterpreterValue Pad(InterpreterState&, mhlo::PadOp pad, InterpreterValue arg,
                     InterpreterValue padding_value) {
  padding_value = padding_value.ExtractElement({});
  auto his = pad.getEdgePaddingHigh().getValues<int64_t>();
  auto los = pad.getEdgePaddingLow().getValues<int64_t>();
  auto ins = pad.getInteriorPadding().getValues<int64_t>();

  llvm::SmallVector<int64_t> sizes;
  for (auto [size, lo, in, hi] : llvm::zip(arg.View().sizes, los, ins, his)) {
    sizes.push_back(size + lo + hi + (size - 1) * in);
  }

  auto result = arg.TypedAlike(sizes);
  result.Fill([&](llvm::ArrayRef<int64_t>) { return padding_value; });

  for (const auto& in_indices : arg.View().Indices()) {
    llvm::SmallVector<int64_t> out_indices;
    for (auto [in_index, in, lo] : llvm::zip(in_indices, ins, los)) {
      out_indices.push_back(in_index * (in + 1) + lo);
    }
    if (result.View().InBounds(out_indices)) {
      result.InsertElement(out_indices, arg.ExtractElement(in_indices));
    }
  }

  return result;
}

InterpreterValue Compare(InterpreterState&, mhlo::CompareOp compare,
                         const InterpreterValue& lhs,
                         const InterpreterValue& rhs) {
  switch (compare.getComparisonDirection()) {
    case mlir::mhlo::ComparisonDirection::EQ:
      return ApplyCwiseBinaryMap<Foeq>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::NE:
      return ApplyCwiseBinaryMap<Fune>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::GE:
      return ApplyCwiseBinaryMap<Foge>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::GT:
      return ApplyCwiseBinaryMap<Fogt>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::LE:
      return ApplyCwiseBinaryMap<Fole>(lhs, rhs);
    case mlir::mhlo::ComparisonDirection::LT:
      return ApplyCwiseBinaryMap<Folt>(lhs, rhs);
  }
}

llvm::SmallVector<InterpreterValue> Gather(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto gather = llvm::dyn_cast<mhlo::GatherOp>(op);
  auto dynamic_gather = llvm::dyn_cast<mhlo::DynamicGatherOp>(op);

  if (!gather && !dynamic_gather) {
    state.AddFailure("invalid gather");
    return {};
  }

  const auto& dims = gather ? gather.getDimensionNumbers()
                            : dynamic_gather.getDimensionNumbers();

  auto index_vector_dim = dims.getIndexVectorDim();
  auto start_index_map = dims.getStartIndexMap();
  auto offset_dims = dims.getOffsetDims();
  auto collapsed_slice_dims = dims.getCollapsedSliceDims();
  auto slice_sizes =
      gather ? llvm::to_vector(gather.getSliceSizes().getValues<int64_t>())
             : llvm::to_vector(llvm::map_range(
                   args[2].View().Indices(), [&](const auto& indices) {
                     return args[2].ExtractElement(indices).AsInt();
                   }));

  auto& operand = args[0];
  auto& start_indices = args[1];
  const auto& operand_view = operand.View();
  int64_t operand_rank = operand_view.Rank();

  // Make a fake BufferView for the start indices.
  BufferView start_indices_view = start_indices.View();
  auto output_rank =
      static_cast<int64_t>(start_indices_view.Rank() + offset_dims.size());
  if (index_vector_dim < start_indices_view.Rank()) {
    --output_rank;
    start_indices_view.sizes[index_vector_dim] = 1;
  }

  SmallVector<int64_t> batch_dims;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (!llvm::is_contained(offset_dims, i)) {
      batch_dims.push_back(i);
    }
  }

  // Make a fake BufferView for the slice indices.
  BufferView slice_indices_view{0, SmallVector<int64_t>{slice_sizes}, {}};

  SmallVector<int64_t> non_collapsed_slice_dims;
  for (int64_t i = 0; i < operand_rank; ++i) {
    if (!llvm::is_contained(collapsed_slice_dims, i)) {
      non_collapsed_slice_dims.push_back(i);
    }
  }

  SmallVector<int64_t> output_sizes(output_rank);
  for (auto [output_dim, slice_dim] :
       llvm::zip(offset_dims, non_collapsed_slice_dims)) {
    output_sizes[output_dim] = slice_sizes[slice_dim];
  }
  for (auto [batch_index, output_dim] : llvm::enumerate(batch_dims)) {
    if (batch_index >= index_vector_dim) {
      ++batch_index;
    }
    output_sizes[output_dim] = start_indices_view.sizes[batch_index];
  }

  auto output = operand.TypedAlike(output_sizes);
  for (auto start_indices_index : start_indices_view.Indices()) {
    SmallVector<int64_t> operand_base_indices(operand_rank);
    for (auto [i, dim] : llvm::enumerate(start_index_map)) {
      if (index_vector_dim < start_indices_view.Rank()) {
        start_indices_index[index_vector_dim] = static_cast<int64_t>(i);
      }
      operand_base_indices[dim] = std::max<int64_t>(
          0, std::min(start_indices.ExtractElement(start_indices_index).AsInt(),
                      operand_view.sizes[dim] - slice_sizes[dim]));
    }

    for (const auto& slice_indices : slice_indices_view.Indices()) {
      SmallVector<int64_t> operand_indices;
      for (int64_t i = 0; i < operand_rank; ++i) {
        operand_indices.push_back(operand_base_indices[i] + slice_indices[i]);
      }

      SmallVector<int64_t> output_indices(output_rank);
      for (auto [output_dim, slice_dim] :
           llvm::zip(offset_dims, non_collapsed_slice_dims)) {
        output_indices[output_dim] = slice_indices[slice_dim];
      }
      for (auto [batch_index, output_dim] : llvm::enumerate(batch_dims)) {
        output_indices[output_dim] =
            start_indices_index[batch_index >= index_vector_dim
                                    ? batch_index + 1
                                    : batch_index];
      }

      auto value = operand.ExtractElement(operand_indices);
      output.InsertElement(output_indices, value);
    }
  }

  return {output};
}

llvm::SmallVector<InterpreterValue> Scatter(
    InterpreterState& state, mhlo::ScatterOp scatter,
    ArrayRef<InterpreterValue> n_inputs, InterpreterValue scatter_indices,
    ArrayRef<InterpreterValue> n_updates) {
  const auto& dims = scatter.getScatterDimensionNumbers();
  auto index_vector_dim = dims.getIndexVectorDim();
  auto scatter_dims_to_operand_dims = dims.getScatterDimsToOperandDims();
  auto inserted_window_dims = dims.getInsertedWindowDims();
  auto update_window_dims = dims.getUpdateWindowDims();

  auto input_view = n_inputs.front().View();
  int64_t operand_rank = input_view.Rank();
  int64_t updates_rank = n_updates.front().View().Rank();
  int64_t indices_rank = scatter_indices.View().Rank();

  llvm::SmallVector<int64_t> batch_dims;
  for (int64_t dim = 0; dim < operand_rank; ++dim) {
    if (!llvm::is_contained(inserted_window_dims, dim)) {
      batch_dims.push_back(dim);
    }
  }

  llvm::SmallVector<int64_t> update_scatter_dims;
  for (int64_t dim = 0; dim < updates_rank; ++dim) {
    if (!llvm::is_contained(update_window_dims, dim)) {
      update_scatter_dims.push_back(dim);
    }
  }

  llvm::SmallVector<InterpreterValue> n_results;
  for (auto& inputs : n_inputs) {
    n_results.push_back(inputs.Clone());
  }

  for (auto [inputs, updates, results] :
       llvm::zip(n_results, n_updates, n_results)) {
    const auto& updates_view = updates.View();
    for (const auto& update_indices : updates_view.Indices()) {
      llvm::SmallVector<int64_t> input_indices(operand_rank);
      llvm::SmallVector<int64_t> max_indices(operand_rank);
      llvm::SmallVector<int64_t> min_indices(operand_rank);
      llvm::SmallVector<int64_t> scatter_indices_index(indices_rank);

      for (auto [index, dim] : llvm::enumerate(update_scatter_dims)) {
        scatter_indices_index[index >= index_vector_dim ? index + 1 : index] +=
            update_indices[dim];
      }

      for (auto [update_dim, operand_dim] :
           llvm::zip(update_window_dims, batch_dims)) {
        input_indices[operand_dim] = update_indices[update_dim];
        max_indices[operand_dim] = updates_view.sizes[update_dim] - 1;
      }

      for (auto [index, dim] : llvm::enumerate(scatter_dims_to_operand_dims)) {
        if (index_vector_dim < indices_rank) {
          scatter_indices_index[index_vector_dim] = static_cast<int64_t>(index);
        }

        int64_t scatter_index =
            scatter_indices.ExtractElement(scatter_indices_index).AsInt();
        input_indices[dim] += scatter_index;
        min_indices[dim] += scatter_index;
        max_indices[dim] += scatter_index;
      }

      if (!input_view.InBounds(min_indices) ||
          !input_view.InBounds(max_indices)) {
        continue;
      }

      auto current_value = inputs.ExtractElement(input_indices).AsUnitTensor();
      auto update = updates.ExtractElement(update_indices).AsUnitTensor();

      auto result = Interpret(state, scatter.getUpdateComputation(),
                              {current_value, update});
      if (state.HasFailure()) {
        return n_results;
      }
      inputs.InsertElement(input_indices, result.front().ExtractElement({}));
    }
  }

  return n_results;
}

InterpreterValue Select(InterpreterState&, mhlo::SelectOp,
                        TensorOrMemref<bool> condition,
                        const InterpreterValue& true_vals,
                        const InterpreterValue& false_vals) {
  // TODO(jreiffers): Support implicit broadcasting.
  auto result = true_vals.Clone();
  for (const auto& indices : condition.view.Indices()) {
    if (!condition.at(indices)) {
      result.InsertElement(indices, false_vals.ExtractElement(indices));
    }
  }
  return result;
}

InterpreterValue Transpose(InterpreterState&, mhlo::TransposeOp transpose,
                           const InterpreterValue& tensor) {
  auto permutation = transpose.getPermutation().getValues<int64_t>();
  return TransposeImpl(tensor, llvm::to_vector(permutation));
}

InterpreterValue Iota(InterpreterState&, mhlo::IotaOp iota) {
  auto dim = iota.getIotaDimension();
  auto ty = iota->getResultTypes()[0].cast<ShapedType>();
  return DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    auto result = TensorOrMemref<decltype(dummy)>::Empty({ty.getShape()[dim]});
    for (const auto& index : result.view.Indices()) {
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
  constexpr static bool SupportedType() {
    return std::is_convertible_v<A, R>;
  }

  template <typename A>
  static R Apply(A v) {
    return v;
  }
};

InterpreterValue Convert(InterpreterState&, mhlo::ConvertOp op,
                         InterpreterValue in) {
  return DispatchScalarType(op->getResultTypes()[0],
                            [&](auto dummy) -> InterpreterValue {
                              return ApplyCwiseMap<Caster<decltype(dummy)>>(in);
                            });
}

llvm::SmallVector<InterpreterValue> While(
    InterpreterState& state, mhlo::WhileOp whileop,
    SmallVector<InterpreterValue> loop_vars) {
  auto cond = Interpret(state, whileop.getRegion(0), loop_vars);
  while (!state.HasFailure() &&
         std::get<TensorOrMemref<bool>>(cond.front().storage).at({})) {
    loop_vars = Interpret(state, whileop.getRegion(1), loop_vars);
    if (state.HasFailure()) {
      break;
    }
    cond = Interpret(state, whileop.getRegion(0), loop_vars);
  }
  return loop_vars;
}

InterpreterValue Copy(InterpreterState&, mhlo::CopyOp op,
                      const InterpreterValue& tensor) {
  auto layout = op->getAttr("result_layout");
  if (!layout) {
    return tensor.Clone();
  }
  return tensor.Clone(
      llvm::to_vector(cast<DenseIntElementsAttr>(layout).getValues<int64_t>()));
}

llvm::SmallVector<InterpreterValue> Fusion(InterpreterState& state,
                                           mhlo::FusionOp fusion,
                                           ArrayRef<InterpreterValue> args) {
  return Interpret(state, fusion.getRegion(), args);
}

llvm::SmallVector<InterpreterValue> Reduce(InterpreterState& state,
                                           mhlo::ReduceOp reduce,
                                           ArrayRef<InterpreterValue> operands,
                                           ArrayRef<InterpreterValue> inits) {
  auto dims = reduce.getDimensions().getValues<int64_t>();

  auto out_sizes = operands.front().View().sizes;
  for (int64_t dim : llvm::reverse(dims)) {
    out_sizes.erase(out_sizes.begin() + dim);
  }

  SmallVector<InterpreterValue> results;
  for (auto [operand, init] : llvm::zip(operands, inits)) {
    results.push_back(operand.TypedAlike(out_sizes));
    auto init_scalar = init.ExtractElement({});
    results.back().Fill([&](llvm::ArrayRef<int64_t>) { return init_scalar; });
  }

  for (const auto& index : operands[0].View().Indices()) {
    auto dst_index = index;
    for (int64_t dim : llvm::reverse(dims)) {
      dst_index.erase(dst_index.begin() + dim);
    }

    SmallVector<InterpreterValue> args;
    for (auto& result : results) {
      args.push_back(result.ExtractElement(dst_index).AsUnitTensor());
    }
    for (auto& operand : operands) {
      args.push_back(operand.ExtractElement(index).AsUnitTensor());
    }
    auto new_values = Interpret(state, reduce.getRegion(), args);
    if (state.HasFailure()) {
      return results;
    }

    for (auto [result, value] : llvm::zip(results, new_values)) {
      result.InsertElement(dst_index, value.ExtractElement({}));
    }
  }
  return results;
}

SmallVector<InterpreterValue> Sort(InterpreterState& state, mhlo::SortOp op,
                                   ArrayRef<InterpreterValue> inputs) {
  const auto& shape = inputs.front().View().sizes;
  uint64_t dim =
      op.getDimension() < shape.size() ? op.getDimension() : shape.size() - 1;
  SmallVector<int64_t> indices(shape[dim]);

  auto iter_view = inputs.front().View();
  iter_view.sizes[dim] = 1;
  SmallVector<InterpreterValue> results;
  for (const auto& input : inputs) {
    results.push_back(input.TypedAlike(input.View().sizes));
  }
  for (const auto& offset : iter_view.Indices()) {
    std::iota(indices.begin(), indices.end(), 0);
    auto comparator = [&](int64_t a, int64_t b) {
      if (state.HasFailure()) {
        return false;
      }
      SmallVector<InterpreterValue> args;
      auto index_a = offset;
      index_a[dim] = a;
      auto index_b = offset;
      index_b[dim] = b;
      for (const auto& input : inputs) {
        args.push_back(input.ExtractElement(index_a).AsUnitTensor());
        args.push_back(input.ExtractElement(index_b).AsUnitTensor());
      }
      auto result = Interpret(state, op.getComparator(), args);
      return !state.HasFailure() && result[0].ExtractElement({}).AsInt() != 0;
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
        auto elem = input.ExtractElement(copy);
        copy[dim] = static_cast<int64_t>(dst);
        result.InsertElement(copy, elem);
      }
    }
  }
  return results;
}

SmallVector<InterpreterValue> Case(InterpreterState& state, mhlo::CaseOp op,
                                   int64_t index) {
  if (index < 0 || index >= op->getNumResults()) {
    index = op->getNumRegions() - 1;
  }
  return Interpret(state, op->getRegion(index), {});
}

InterpreterValue DotGeneralImpl(InterpreterValue& lhs, InterpreterValue& rhs,
                                ArrayRef<int64_t> lhs_contracting,
                                ArrayRef<int64_t> rhs_contracting,
                                ArrayRef<int64_t> lhs_batch,
                                ArrayRef<int64_t> rhs_batch,
                                mlir::Type element_ty) {
  auto& lhsv = lhs.View();
  auto& rhsv = rhs.View();
  SmallVector<int64_t> dimensions;
  SmallVector<int64_t> lhs_non_batch;
  SmallVector<int64_t> rhs_non_batch;
  auto nbatch = static_cast<int64_t>(lhs_batch.size());
  for (int64_t lhs_dim : lhs_batch) {
    dimensions.push_back(lhsv.sizes[lhs_dim]);
  }
  for (int64_t i = 0; i < lhsv.Rank(); i++) {
    if (!llvm::is_contained(lhs_contracting, i) &&
        !llvm::is_contained(lhs_batch, i)) {
      dimensions.push_back(lhsv.sizes[i]);
      lhs_non_batch.push_back(i);
    }
  }
  for (int64_t i = 0; i < rhs.View().Rank(); i++) {
    if (!llvm::is_contained(rhs_contracting, i) &&
        !llvm::is_contained(rhs_batch, i)) {
      dimensions.push_back(rhsv.sizes[i]);
      rhs_non_batch.push_back(i);
    }
  }

  SmallVector<int64_t> lhs_index(lhsv.Rank());
  SmallVector<int64_t> rhs_index(rhsv.Rank());
  SmallVector<int64_t> output_index(dimensions.size());
  auto output = lhs.TypedAlike(dimensions);

  std::function<void(int64_t)> apply_batch, apply_lhs, apply_rhs,
      apply_contract;

  apply_batch = [&](int64_t dim) {
    if (dim >= nbatch) {
      apply_lhs(0);
      return;
    }
    for (int64_t i = 0; i < dimensions[dim]; ++i) {
      lhs_index[lhs_batch[dim]] = i;
      rhs_index[rhs_batch[dim]] = i;
      output_index[dim] = i;
      apply_batch(dim + 1);
    }
  };

  apply_lhs = [&](int64_t dim) {
    if (dim >= lhs_non_batch.size()) {
      apply_rhs(0);
      return;
    }
    int64_t out_dim = nbatch + dim;
    for (int64_t i = 0; i < dimensions[out_dim]; ++i) {
      lhs_index[lhs_non_batch[dim]] = i;
      output_index[out_dim] = i;
      apply_lhs(dim + 1);
    }
  };

  apply_rhs = [&](int64_t dim) {
    if (dim >= rhs_non_batch.size()) {
      apply_contract(0);
      return;
    }
    auto out_dim = static_cast<int64_t>(nbatch + lhs_non_batch.size() + dim);
    for (int64_t i = 0; i < dimensions[out_dim]; ++i) {
      rhs_index[rhs_non_batch[dim]] = i;
      output_index[out_dim] = i;
      apply_rhs(dim + 1);
    }
  };

  apply_contract = [&](int64_t dim) {
    if (dim >= lhs_contracting.size()) {
      DispatchScalarType(element_ty, [&](auto dummy) {
        using T = TensorOrMemref<decltype(dummy)>;
        std::get<T>(output.storage).at(output_index) +=
            std::get<T>(lhs.storage).at(lhs_index) *
            std::get<T>(rhs.storage).at(rhs_index);
      });
      return;
    }
    for (int64_t i = 0; i < lhsv.sizes[lhs_contracting[dim]]; ++i) {
      lhs_index[lhs_contracting[dim]] = i;
      rhs_index[rhs_contracting[dim]] = i;
      apply_contract(dim + 1);
    }
  };

  apply_batch(0);
  return output;
}

// TODO(jreiffers): Unify this with DotGeneral.
InterpreterValue Dot(InterpreterState& state, mhlo::DotOp op,
                     const InterpreterValue& lhs, const InterpreterValue& rhs) {
  auto ty = cast<ShapedType>(op->getResultTypes()[0]);
  auto result = lhs.TypedAlike(ty.getShape());

  if (lhs.View().Rank() == 1 && rhs.View().Rank() == 1) {
    DispatchScalarType(ty, [&](auto dummy) {
      using T = decltype(dummy);
      using TT = TensorOrMemref<T>;
      auto lhs_tensor = std::get<TT>(lhs.storage);
      auto rhs_tensor = std::get<TT>(rhs.storage);

      T product = 0;
      for (int64_t i = 0; i < lhs_tensor.view.sizes[0]; ++i) {
        product += lhs_tensor.at(i) * rhs_tensor.at(i);
      }
      std::get<TT>(result.storage).at({}) += product;
    });
  } else if (lhs.View().Rank() == 2 && rhs.View().Rank() == 1) {
    DispatchScalarType(ty, [&](auto dummy) {
      using TT = TensorOrMemref<decltype(dummy)>;
      auto lhs_tensor = std::get<TT>(lhs.storage);
      auto rhs_tensor = std::get<TT>(rhs.storage);
      auto result_tensor = std::get<TT>(result.storage);
      for (int64_t i = 0; i < result_tensor.view.sizes[0]; ++i) {
        for (int64_t j = 0; j < rhs_tensor.view.sizes[0]; ++j) {
          result_tensor.at(i) += lhs_tensor.at({i, j}) * rhs_tensor.at(j);
        }
      }
    });
  } else if (lhs.View().Rank() == 2 && rhs.View().Rank() == 2) {
    DispatchScalarType(ty, [&](auto dummy) {
      using TT = TensorOrMemref<decltype(dummy)>;
      auto lhs_tensor = std::get<TT>(lhs.storage);
      auto rhs_tensor = std::get<TT>(rhs.storage);
      auto result_tensor = std::get<TT>(result.storage);
      for (int64_t i = 0; i < result_tensor.view.sizes[0]; ++i) {
        for (int64_t j = 0; j < result_tensor.view.sizes[1]; ++j) {
          for (int64_t k = 0; k < lhs_tensor.view.sizes[1]; ++k) {
            result_tensor.at({i, j}) +=
                lhs_tensor.at({i, k}) * rhs_tensor.at({k, j});
          }
        }
      }
    });
  } else {
    state.AddFailure("unsupported dot");
  }

  return result;
}

InterpreterValue DotGeneral(InterpreterState&, mhlo::DotGeneralOp op,
                            InterpreterValue lhs, InterpreterValue rhs) {
  const auto& dims = op.getDotDimensionNumbers();
  return DotGeneralImpl(
      lhs, rhs, dims.getLhsContractingDimensions(),
      dims.getRhsContractingDimensions(), dims.getLhsBatchingDimensions(),
      dims.getRhsBatchingDimensions(),
      op->getResultTypes()[0].cast<ShapedType>().getElementType());
}

// TODO(jreiffers): Migrate remaining ops to the safer signature.
REGISTER_MLIR_INTERPRETER_OP("mhlo.dynamic_gather", Gather);
REGISTER_MLIR_INTERPRETER_OP("mhlo.gather", Gather);
REGISTER_MLIR_INTERPRETER_OP("mhlo.return", NoOpTerminator);
REGISTER_MLIR_INTERPRETER_OP("mhlo.tuple", MakeTuple);
REGISTER_MLIR_INTERPRETER_OP(BitcastConvert);
REGISTER_MLIR_INTERPRETER_OP(BroadcastInDim);
REGISTER_MLIR_INTERPRETER_OP(Clamp);
REGISTER_MLIR_INTERPRETER_OP(Compare);
REGISTER_MLIR_INTERPRETER_OP(Concatenate);
REGISTER_MLIR_INTERPRETER_OP(Constant);
REGISTER_MLIR_INTERPRETER_OP(Convert);
REGISTER_MLIR_INTERPRETER_OP(Copy);
REGISTER_MLIR_INTERPRETER_OP(Dot);
REGISTER_MLIR_INTERPRETER_OP(DotGeneral);
REGISTER_MLIR_INTERPRETER_OP(DynamicSlice);
REGISTER_MLIR_INTERPRETER_OP(DynamicUpdateSlice);
REGISTER_MLIR_INTERPRETER_OP(Fusion);
REGISTER_MLIR_INTERPRETER_OP(GetTupleElement);
REGISTER_MLIR_INTERPRETER_OP(Iota);
REGISTER_MLIR_INTERPRETER_OP(Case);
REGISTER_MLIR_INTERPRETER_OP(While);
REGISTER_MLIR_INTERPRETER_OP(Pad);
REGISTER_MLIR_INTERPRETER_OP(Reduce);
REGISTER_MLIR_INTERPRETER_OP(Reshape);
REGISTER_MLIR_INTERPRETER_OP(Scatter);
REGISTER_MLIR_INTERPRETER_OP(Select);
REGISTER_MLIR_INTERPRETER_OP(Slice);
REGISTER_MLIR_INTERPRETER_OP(Sort);
REGISTER_MLIR_INTERPRETER_OP(Transpose);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
