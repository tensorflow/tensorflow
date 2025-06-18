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

#include "mlir/Dialect/Linalg/IR/Linalg.h"

// clang-format erroneously puts the Linalg header above.
#include <algorithm>   // NOLINT
#include <cstdint>     // NOLINT
#include <functional>  // NOLINT
#include <memory>      // NOLINT
#include <utility>     // NOLINT

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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

class IterationIndexSideChannel : public InterpreterSideChannel {
 public:
  explicit IterationIndexSideChannel(ArrayRef<int64_t> indices)
      : indices_(indices) {}
  ArrayRef<int64_t> GetIndices() const { return indices_; }

 private:
  ArrayRef<int64_t> indices_;
};

llvm::SmallVector<InterpreterValue> Broadcast(InterpreterState&,
                                              linalg::BroadcastOp broadcast,
                                              const InterpreterValue& input,
                                              InterpreterValue init) {
  auto broadcast_dims = llvm::to_vector(broadcast.getDimensions());
  llvm::sort(broadcast_dims);

  auto out_shape = init.View().sizes;
  auto out = input;
  auto& out_view = out.View();
  for (int64_t dim : broadcast_dims) {
    out_view.sizes.insert(out_view.sizes.begin() + dim, out_shape[dim]);
    out_view.strides.insert(out_view.strides.begin() + dim, 0);
  }

  if (broadcast.getNumResults() == 1) {
    return {std::move(out)};
  }

  init.Fill([&](llvm::ArrayRef<int64_t> indices) {
    return out.ExtractElement(indices);
  });
  return {};
}

llvm::SmallVector<InterpreterValue> Generic(
    InterpreterState& state, linalg::GenericOp generic,
    MutableArrayRef<InterpreterValue> inputs,
    MutableArrayRef<InterpreterValue> outputs_ref) {
  SmallVector<int64_t> shapes;
  for (auto& value : llvm::concat<InterpreterValue>(inputs, outputs_ref)) {
    if (value.IsTensor() /* (or memref) */) {
      llvm::append_range(
          shapes, ArrayRef<int64_t>(value.View().sizes)
                      .drop_back(value.View().num_vector_dims.value_or(0)));
    }
  }
  auto ranges = generic.getShapesToLoopsMap().compose(shapes);

  llvm::SmallVector<InterpreterValue> outputs;
  for (int64_t output = 0; output < outputs_ref.size(); ++output) {
    outputs.push_back(
        GetInitOperand(generic.getOutputs(), output, outputs_ref));
  }

  llvm::SmallVector<int64_t> ivs(ranges.size());
  InterpreterScope scope(state);
  scope.SetSideChannel(std::make_shared<IterationIndexSideChannel>(ivs));

  auto indexing_maps = generic.getIndexingMapsArray();
  auto output_maps =
      ArrayRef<AffineMap>(indexing_maps).drop_front(inputs.size());
  std::function<void(int64_t)> run;
  run = [&](int64_t loop_index) {
    // Abort recursion if we encountered some error previously.
    if (state.HasFailure()) {
      return;
    }

    if (loop_index < ranges.size()) {
      for (int64_t index = 0; index < ranges[loop_index]; ++index) {
        ivs[loop_index] = index;
        run(loop_index + 1);
      }
    } else {
      llvm::SmallVector<InterpreterValue> bbargs;
      // Build bbargs: 1. inputs, 2. outputs.
      for (auto [input, map] : llvm::zip(inputs, indexing_maps)) {
        auto indices = EvalAffineMap(map, ivs);
        bbargs.push_back(input.ExtractElement(indices));
      }
      llvm::SmallVector<llvm::SmallVector<int64_t>> output_indices;
      for (auto [output, map] : llvm::zip(outputs, output_maps)) {
        auto& indices = output_indices.emplace_back(EvalAffineMap(map, ivs));
        bbargs.push_back(output.ExtractElement(indices));
      }
      // Evaluate region.
      auto yielded = Interpret(state, generic.getRegion(), bbargs);
      if (state.HasFailure()) {
        return;
      }
      // Insert yielded values in the outputs.
      for (auto [output, indices, yield] :
           llvm::zip(outputs, output_indices, yielded)) {
        output.InsertElement(indices, yield);
      }
    }
  };
  run(0);

  if (generic.getNumResults() == 0) return {};
  return outputs;
}

llvm::SmallVector<InterpreterValue> Map(InterpreterState& state,
                                        linalg::MapOp op,
                                        ArrayRef<InterpreterValue> inputs,
                                        const InterpreterValue& init) {
  InterpreterValue output =
      isa<TensorType>(op.getInit().getType()) ? init.Clone() : init;

  InterpreterScope scope(state);
  SmallVector<int64_t> ivs(output.View().num_dimensions());
  scope.SetSideChannel(std::make_shared<IterationIndexSideChannel>(ivs));
  for (const auto& indices : output.View().Indices()) {
    std::copy(indices.begin(), indices.end(), ivs.begin());
    llvm::SmallVector<InterpreterValue> args;
    for (auto& input : inputs) {
      args.push_back(input.ExtractElement(indices));
    }
    auto yielded = Interpret(state, op.getRegion(), args);
    if (state.HasFailure()) {
      break;
    }
    output.InsertElement(indices, yielded[0]);
  }

  if (op.getNumResults() == 0) {
    return {};
  }
  return {std::move(output)};
}

llvm::SmallVector<InterpreterValue> Reduce(InterpreterState& state,
                                           linalg::ReduceOp reduce,
                                           ArrayRef<InterpreterValue> ins,
                                           ArrayRef<InterpreterValue> inits) {
  auto dims = reduce.getDimensions();
  SmallVector<InterpreterValue> output;
  for (auto [ty, init] : llvm::zip(reduce.getInits().getTypes(), inits)) {
    output.push_back(isa<TensorType>(ty) ? init.Clone() : init);
  }
  for (const auto& index : ins[0].View().Indices()) {
    auto dst_index = index;
    for (int64_t dim : llvm::reverse(dims)) {
      dst_index.erase(dst_index.begin() + dim);
    }

    SmallVector<InterpreterValue> args;
    for (auto& in : ins) {
      args.push_back(in.ExtractElement(index));
    }
    for (auto& out : output) {
      args.push_back(out.ExtractElement(dst_index));
    }
    auto new_values = Interpret(state, reduce.getRegion(), args);
    if (state.HasFailure()) {
      return {};
    }

    for (auto [out, value] : llvm::zip(output, new_values)) {
      out.InsertElement(dst_index, value);
    }
  }
  if (reduce->getNumResults() == 0) {
    return {};
  }
  return output;
}

llvm::SmallVector<InterpreterValue> Fill(InterpreterState&, linalg::FillOp op,
                                         const InterpreterValue& value,
                                         const InterpreterValue& init) {
  // TODO(jreiffers): Support variadic Fill.
  InterpreterValue output = GetInitOperand(op.getOutputs(), 0, {init});
  output.Fill([&](llvm::ArrayRef<int64_t>) { return value; });
  if (op.getNumResults() == 0) {
    return {};
  }
  return {std::move(output)};
}

int64_t Index(InterpreterState& state, linalg::IndexOp index) {
  return state.GetTopScope()
      ->GetSideChannel<IterationIndexSideChannel>()
      ->GetIndices()[index.getDim()];
}

SmallVector<InterpreterValue> Matmul(InterpreterState& state,
                                     linalg::MatmulOp matmul,
                                     ArrayRef<InterpreterValue> inputs,
                                     const InterpreterValue& init) {
  if (inputs.size() != 2) {
    state.AddFailure("Invalid matmul");
    return {};
  }
  const auto& lhs = inputs[0];
  const auto& rhs = inputs[1];
  auto ty = matmul.getOutputs()[0].getType();
  auto result = isa<TensorType>(ty) ? init.Clone() : init;
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

  if (matmul.getNumResults() == 0) {
    return {};
  }
  return {result};
}

SmallVector<InterpreterValue> Transpose(InterpreterState&,
                                        linalg::TransposeOp transpose,
                                        const InterpreterValue& input,
                                        InterpreterValue init) {
  auto transposed = TransposeImpl(input, transpose.getPermutation());
  if (transpose.getNumResults() == 1) {
    return {transposed};
  }

  init.Fill([&](auto index) { return transposed.ExtractElement(index); });
  return {};
}

SmallVector<InterpreterValue> Dot(InterpreterState&, linalg::DotOp op,
                                  ArrayRef<InterpreterValue> inputs,
                                  InterpreterValue acc) {
  const auto& lhs = inputs[0];
  const auto& rhs = inputs[1];
  if (mlir::isa<TensorType>(op.getOutputs()[0].getType())) {
    acc = acc.Clone();
  }
  DispatchScalarType(op.getOutputs()[0].getType(), [&](auto dummy) {
    using TT = TensorOrMemref<decltype(dummy)>;
    auto lhs_tensor = std::get<TT>(lhs.storage);
    auto rhs_tensor = std::get<TT>(rhs.storage);
    auto result_tensor = std::get<TT>(acc.storage);
    for (int64_t k = 0; k < lhs_tensor.view.sizes[0]; ++k) {
      result_tensor.at({}) += lhs_tensor.at(k) * rhs_tensor.at(k);
    }
  });

  if (op.getNumResults() == 0) {
    return {};
  }
  return {std::move(acc)};
}

SmallVector<InterpreterValue> Vecmat(InterpreterState&, linalg::VecmatOp op,
                                     ArrayRef<InterpreterValue> inputs,
                                     InterpreterValue acc) {
  const auto& lhs = inputs[0];
  const auto& rhs = inputs[1];
  if (mlir::isa<TensorType>(op.getOutputs()[0].getType())) {
    acc = acc.Clone();
  }
  DispatchScalarType(op.getOutputs()[0].getType(), [&](auto dummy) {
    using TT = TensorOrMemref<decltype(dummy)>;
    auto lhs_tensor = std::get<TT>(lhs.storage);
    auto rhs_tensor = std::get<TT>(rhs.storage);
    auto result_tensor = std::get<TT>(acc.storage);
    for (int64_t j = 0; j < result_tensor.view.sizes[0]; ++j) {
      for (int64_t k = 0; k < lhs_tensor.view.sizes[0]; ++k) {
        result_tensor.at(j) += lhs_tensor.at(k) * rhs_tensor.at({k, j});
      }
    }
  });

  if (op.getNumResults() == 0) {
    return {};
  }
  return {std::move(acc)};
}

REGISTER_MLIR_INTERPRETER_OP("linalg.yield", NoOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(Broadcast);
REGISTER_MLIR_INTERPRETER_OP(Dot);
REGISTER_MLIR_INTERPRETER_OP(Fill);
REGISTER_MLIR_INTERPRETER_OP(Generic);
REGISTER_MLIR_INTERPRETER_OP(Index);
REGISTER_MLIR_INTERPRETER_OP(Map);
REGISTER_MLIR_INTERPRETER_OP(Matmul);
REGISTER_MLIR_INTERPRETER_OP(Reduce);
REGISTER_MLIR_INTERPRETER_OP(Transpose);
REGISTER_MLIR_INTERPRETER_OP(Vecmat);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
