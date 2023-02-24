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

#include <iterator>
#include <memory>

#include "gml_st/IR/gml_st_ops.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Operation.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

OffsetsSizesStrides unpackTileArgs(const TensorOrMemref<int64_t>& tile) {
  OffsetsSizesStrides result;
  int64_t rank = tile.view.sizes[0] / 3;
  for (int64_t i = 0; i < rank; ++i) {
    result.offsets.push_back(tile.at(i));
    result.sizes.push_back(tile.at(i + rank));
    result.strides.push_back(tile.at(i + 2 * rank));
  }
  return result;
}

llvm::SmallVector<InterpreterValue> gmlStLoop(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  bool isBufferized = op->getNumResults() == 0;
  auto parallelOp = llvm::cast<gml_st::ParallelOp>(op);

  auto terminator = parallelOp.getTerminator();

  int64_t numOutputs = terminator.getDsts().size();
  assert((args.size() - numOutputs) % 3 == 0 &&
         "expected uniform sizes for lbs, ubs and steps");

  size_t numLoops = (args.size() - numOutputs) / 3;
  auto boundArgs = args.take_front(numLoops * 3);
  auto lbs = unpackInterpreterValues<int64_t>(boundArgs.take_front(numLoops));
  auto ubs =
      unpackInterpreterValues<int64_t>(boundArgs.slice(numLoops, numLoops));
  auto steps = unpackInterpreterValues<int64_t>(boundArgs.take_back(numLoops));

  SmallVector<InterpreterValue> outputs;
  for (size_t i = args.size() - numOutputs; i < args.size(); ++i) {
    outputs.push_back(getInitOperand(op, static_cast<int64_t>(i), args));
  }

  SmallVector<int64_t> iterSizes;
  for (auto [lb, ub, step] : llvm::zip(lbs, ubs, steps)) {
    if (step == 0) {
      state.addFailure("invalid step");
      return {};
    }
    iterSizes.push_back((ub - lb + (step - 1)) / step);
  }

  // Make a fake buffer view to abuse its index iterator.
  BufferView view{0, iterSizes, {}};
  for (const auto& indices : view.indices()) {
    SmallVector<InterpreterValue> args;
    for (auto [i, lb, step] : llvm::zip(indices, lbs, steps)) {
      args.push_back(InterpreterValue{i * step + lb});
    }
    llvm::copy(outputs, std::back_inserter(args));

    auto yielded = interpret(state, op->getRegion(0), args);
    if (state.hasFailure()) break;

    assert(yielded.size() == 3 * numOutputs &&
           "expected equal number of srcs, dsts and sets");

    MutableArrayRef<InterpreterValue> yieldedRef = yielded;

    // The dsts of set yield are always the outputs, so we can ignore them.
    auto srcs = yieldedRef.take_front(numOutputs);
    auto tiles = yieldedRef.take_back(numOutputs);

    for (auto [src, tile, output] : llvm::zip(srcs, tiles, outputs)) {
      auto tileArgs =
          unpackTileArgs(std::get<TensorOrMemref<int64_t>>(tile.storage));
      if (!src.isTensor()) {
        output.insertElement(tileArgs.offsets, src);
      } else {
        for (const auto& srcIndices : src.view().indices()) {
          assert(srcIndices.size() == tileArgs.sizes.size() &&
                 "mismatched tile/src rank");
          // The sizes of the tile must match the sizes of the src, so we can
          // ignore them.
          SmallVector<int64_t> dstIndices;
          for (auto [src_index, offset, stride] :
               llvm::zip(srcIndices, tileArgs.offsets, tileArgs.strides)) {
            dstIndices.push_back(src_index * stride + offset);
          }
          output.insertElement(dstIndices, src.extractElement(srcIndices));
        }
      }
    }
  }

  if (isBufferized) return {};
  return outputs;
}

InterpreterValue tile(InterpreterState&, gml_st::TileOp op,
                      ArrayRef<int64_t> dynamicOffsets,
                      ArrayRef<int64_t> dynamicSizes,
                      ArrayRef<int64_t> dynamicStrides) {
  auto values = extractOffsetsSizesStrides(dynamicOffsets, dynamicSizes,
                                           dynamicStrides, op);
  int64_t rank = static_cast<int64_t>(values.offsets.size());

  auto result = TensorOrMemref<int64_t>::empty({rank * 3});
  for (int64_t i = 0; i < rank; ++i) {
    result.at(i) = values.offsets[i];
    result.at(i + rank) = values.sizes[i];
    result.at(i + 2 * rank) = values.strides[i];
  }

  return {result};
}

REGISTER_MLIR_INTERPRETER_OP("gml_st.parallel", gmlStLoop);
REGISTER_MLIR_INTERPRETER_OP("gml_st.set_yield", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP("gml_st.yield", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(tile);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
