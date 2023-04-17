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

#include "mlir/Dialect/Linalg/IR/Linalg.h"

// clang-format erroneously puts the Linalg header above.
#include <algorithm>   // NOLINT
#include <cstdint>     // NOLINT
#include <functional>  // NOLINT
#include <memory>      // NOLINT

#include "llvm/ADT/STLExtras.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

class IterationIndexSideChannel : public InterpreterSideChannel {
 public:
  explicit IterationIndexSideChannel(ArrayRef<int64_t> indices)
      : indices(indices) {}
  ArrayRef<int64_t> getIndices() const { return indices; }

 private:
  ArrayRef<int64_t> indices;
};

llvm::SmallVector<InterpreterValue> broadcast(InterpreterState&,
                                              linalg::BroadcastOp broadcast,
                                              const InterpreterValue& input,
                                              InterpreterValue init) {
  auto broadcastDims = llvm::to_vector(broadcast.getDimensions());
  llvm::sort(broadcastDims);

  auto outShape = init.view().sizes;
  auto out = input;
  auto& outView = out.view();
  for (int64_t dim : broadcastDims) {
    outView.sizes.insert(outView.sizes.begin() + dim, outShape[dim]);
    outView.strides.insert(outView.strides.begin() + dim, 0);
  }

  if (broadcast.getNumResults() == 1) {
    return {out};
  }

  init.fill([&](llvm::ArrayRef<int64_t> indices) {
    return out.extractElement(indices);
  });
  return {};
}

llvm::SmallVector<InterpreterValue> generic(
    InterpreterState& state, linalg::GenericOp generic,
    MutableArrayRef<InterpreterValue> inputs,
    MutableArrayRef<InterpreterValue> outputsRef) {
  SmallVector<int64_t> shapes;
  for (auto& value : llvm::concat<InterpreterValue>(inputs, outputsRef)) {
    if (value.isTensor() /* (or memref) */) {
      llvm::append_range(
          shapes, ArrayRef<int64_t>(value.view().sizes)
                      .drop_back(value.view().numVectorDims.value_or(0)));
    }
  }
  auto ranges = generic.getShapesToLoopsMap().compose(shapes);

  llvm::SmallVector<InterpreterValue> outputs;
  for (int64_t output = 0; output < outputsRef.size(); ++output) {
    outputs.push_back(getInitOperand(generic.getOutputs(), output, outputsRef));
  }

  llvm::SmallVector<int64_t> ivs(ranges.size());
  InterpreterScope scope(state);
  scope.setSideChannel(std::make_shared<IterationIndexSideChannel>(ivs));

  auto indexingMaps = generic.getIndexingMapsArray();
  auto outputMaps = ArrayRef<AffineMap>(indexingMaps).drop_front(inputs.size());
  std::function<void(int64_t)> run;
  run = [&](int64_t loopIndex) {
    // Abort recursion if we encountered some error previously.s
    if (state.hasFailure()) return;

    if (loopIndex < ranges.size()) {
      for (int64_t index = 0; index < ranges[loopIndex]; ++index) {
        ivs[loopIndex] = index;
        run(loopIndex + 1);
      }
    } else {
      llvm::SmallVector<InterpreterValue> bbargs;
      // Build bbargs: 1. inputs, 2. outputs.
      for (auto [input, map] : llvm::zip(inputs, indexingMaps)) {
        auto indices = evalAffineMap(map, ivs);
        bbargs.push_back(input.extractElement(indices));
      }
      llvm::SmallVector<llvm::SmallVector<int64_t>> outputIndices;
      for (auto [output, map] : llvm::zip(outputs, outputMaps)) {
        auto& indices = outputIndices.emplace_back(evalAffineMap(map, ivs));
        bbargs.push_back(output.extractElement(indices));
      }
      // Evaluate region.
      auto yielded = interpret(state, generic.getRegion(), bbargs);
      if (state.hasFailure()) return;
      // Insert yielded values in the outputs.
      for (auto [output, indices, yield] :
           llvm::zip(outputs, outputIndices, yielded)) {
        output.insertElement(indices, yield);
      }
    }
  };
  run(0);

  if (generic.getNumResults() == 0) return {};
  return outputs;
}

llvm::SmallVector<InterpreterValue> map(InterpreterState& state,
                                        linalg::MapOp op,
                                        ArrayRef<InterpreterValue> inputs,
                                        const InterpreterValue& init) {
  InterpreterValue output =
      op.getInit().getType().isa<TensorType>() ? init.clone() : init;

  InterpreterScope scope(state);
  SmallVector<int64_t> ivs(output.view().rank());
  scope.setSideChannel(std::make_shared<IterationIndexSideChannel>(ivs));
  for (const auto& indices : output.view().indices()) {
    std::copy(indices.begin(), indices.end(), ivs.begin());
    llvm::SmallVector<InterpreterValue> args;
    for (auto& input : inputs) {
      args.push_back(input.extractElement(indices));
    }
    auto yielded = interpret(state, op.getRegion(), args);
    if (state.hasFailure()) break;
    output.insertElement(indices, yielded[0]);
  }

  if (op.getNumResults() == 0) return {};
  return {output};
}

llvm::SmallVector<InterpreterValue> reduce(InterpreterState& state,
                                           linalg::ReduceOp reduce,
                                           ArrayRef<InterpreterValue> ins,
                                           ArrayRef<InterpreterValue> inits) {
  auto dims = reduce.getDimensions();
  SmallVector<InterpreterValue> output;
  for (auto [ty, init] : llvm::zip(reduce.getInits().getTypes(), inits)) {
    output.push_back(ty.isa<TensorType>() ? init.clone() : init);
  }
  for (const auto& index : ins[0].view().indices()) {
    auto dstIndex = index;
    for (int64_t dim : llvm::reverse(dims)) {
      dstIndex.erase(dstIndex.begin() + dim);
    }

    SmallVector<InterpreterValue> args;
    for (auto& in : ins) {
      args.push_back(in.extractElement(index));
    }
    for (auto& out : output) {
      args.push_back(out.extractElement(dstIndex));
    }
    auto newValues = interpret(state, reduce.getRegion(), args);
    if (state.hasFailure()) return {};

    for (auto [out, value] : llvm::zip(output, newValues)) {
      out.insertElement(dstIndex, value);
    }
  }
  if (reduce->getNumResults() == 0) return {};
  return output;
}

llvm::SmallVector<InterpreterValue> fill(InterpreterState&, linalg::FillOp op,
                                         const InterpreterValue& value,
                                         const InterpreterValue& init) {
  // TODO(jreiffers): Support variadic fill.
  InterpreterValue output = getInitOperand(op.getOutputs(), 0, {init});
  output.fill([&](llvm::ArrayRef<int64_t>) { return value; });
  if (op.getNumResults() == 0) return {};
  return {output};
}

int64_t index(InterpreterState& state, linalg::IndexOp index) {
  return state.getTopScope()
      ->getSideChannel<IterationIndexSideChannel>()
      ->getIndices()[index.getDim()];
}

SmallVector<InterpreterValue> matmul(InterpreterState& state,
                                     linalg::MatmulOp matmul,
                                     ArrayRef<InterpreterValue> inputs,
                                     const InterpreterValue& init) {
  if (inputs.size() != 2) {
    state.addFailure("Invalid matmul");
    return {};
  }
  const auto& lhs = inputs[0];
  const auto& rhs = inputs[1];
  auto ty = matmul.getOutputs()[0].getType();
  auto result = ty.isa<TensorType>() ? init.clone() : init;
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

  if (matmul.getNumResults() == 0) return {};
  return {result};
}

SmallVector<InterpreterValue> transpose(InterpreterState&,
                                        linalg::TransposeOp transpose,
                                        const InterpreterValue& input,
                                        InterpreterValue init) {
  auto transposed = transposeImpl(input, transpose.getPermutation());
  if (transpose.getNumResults() == 1) {
    return {transposed};
  }

  init.fill([&](auto index) { return transposed.extractElement(index); });
  return {};
}

SmallVector<InterpreterValue> dot(InterpreterState&, linalg::DotOp op,
                                  ArrayRef<InterpreterValue> inputs,
                                  InterpreterValue acc) {
  const auto& lhs = inputs[0];
  const auto& rhs = inputs[1];
  if (op.getOutputs()[0].getType().isa<TensorType>()) {
    acc = acc.clone();
  }
  dispatchScalarType(op.getOutputs()[0].getType(), [&](auto dummy) {
    using TT = TensorOrMemref<decltype(dummy)>;
    auto lhsTensor = std::get<TT>(lhs.storage);
    auto rhsTensor = std::get<TT>(rhs.storage);
    auto resultTensor = std::get<TT>(acc.storage);
    for (int64_t k = 0; k < lhsTensor.view.sizes[0]; ++k) {
      resultTensor.at({}) += lhsTensor.at(k) * rhsTensor.at(k);
    }
  });

  if (op.getNumResults() == 0) return {};
  return {acc};
}

SmallVector<InterpreterValue> vecmat(InterpreterState&, linalg::VecmatOp op,
                                     ArrayRef<InterpreterValue> inputs,
                                     InterpreterValue acc) {
  const auto& lhs = inputs[0];
  const auto& rhs = inputs[1];
  if (op.getOutputs()[0].getType().isa<TensorType>()) {
    acc = acc.clone();
  }
  dispatchScalarType(op.getOutputs()[0].getType(), [&](auto dummy) {
    using TT = TensorOrMemref<decltype(dummy)>;
    auto lhsTensor = std::get<TT>(lhs.storage);
    auto rhsTensor = std::get<TT>(rhs.storage);
    auto resultTensor = std::get<TT>(acc.storage);
    for (int64_t j = 0; j < resultTensor.view.sizes[0]; ++j) {
      for (int64_t k = 0; k < lhsTensor.view.sizes[0]; ++k) {
        resultTensor.at(j) += lhsTensor.at(k) * rhsTensor.at({k, j});
      }
    }
  });

  if (op.getNumResults() == 0) return {};
  return {acc};
}

REGISTER_MLIR_INTERPRETER_OP("linalg.yield", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(broadcast);
REGISTER_MLIR_INTERPRETER_OP(dot);
REGISTER_MLIR_INTERPRETER_OP(fill);
REGISTER_MLIR_INTERPRETER_OP(generic);
REGISTER_MLIR_INTERPRETER_OP(index);
REGISTER_MLIR_INTERPRETER_OP(map);
REGISTER_MLIR_INTERPRETER_OP(matmul);
REGISTER_MLIR_INTERPRETER_OP(reduce);
REGISTER_MLIR_INTERPRETER_OP(transpose);
REGISTER_MLIR_INTERPRETER_OP(vecmat);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
