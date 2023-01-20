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

#include "tools/mlir_interpreter/dialects/util.h"

#include <variant>

#include "tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {

SmallVector<int64_t> replaceDynamicVals(llvm::ArrayRef<int64_t> staticVals,
                                        ArrayRef<InterpreterValue>& args) {
  llvm::SmallVector<int64_t> out;
  for (int64_t val : staticVals) {
    if (ShapedType::isDynamic(val)) {
      out.push_back(std::get<int64_t>(args.front().storage));
      args = args.drop_front(1);
    } else {
      out.push_back(val);
    }
  }
  return out;
}

SmallVector<int64_t> replaceDynamicVals(ArrayRef<int64_t> staticVals,
                                        ArrayRef<int64_t> dynamicVals) {
  llvm::SmallVector<int64_t> out;
  for (int64_t val : staticVals) {
    if (ShapedType::isDynamic(val)) {
      out.push_back(dynamicVals.front());
      dynamicVals = dynamicVals.drop_front(1);
    } else {
      out.push_back(val);
    }
  }
  assert(dynamicVals.empty() && "expected no leftover dynamic values");
  return out;
}

OffsetsSizesStrides extractOffsetsSizesStrides(
    ArrayRef<InterpreterValue> args, OffsetSizeAndStrideOpInterface op) {
  auto offsets = replaceDynamicVals(op.static_offsets(), args);
  auto sizes = replaceDynamicVals(op.static_sizes(), args);
  auto strides = replaceDynamicVals(op.static_strides(), args);
  return {offsets, sizes, strides};
}

OffsetsSizesStrides extractOffsetsSizesStrides(
    ArrayRef<int64_t> dynamicOffsets, ArrayRef<int64_t> dynamicSizes,
    ArrayRef<int64_t> dynamicStrides, OffsetSizeAndStrideOpInterface op) {
  auto offsets = replaceDynamicVals(op.static_offsets(), dynamicOffsets);
  auto sizes = replaceDynamicVals(op.static_sizes(), dynamicSizes);
  auto strides = replaceDynamicVals(op.static_strides(), dynamicStrides);
  return {offsets, sizes, strides};
}

InterpreterValue reshapeTensor(const InterpreterValue& in,
                               ArrayRef<int64_t> shape) {
  // This doesn't need a copy in many cases, but it's easier that way.
  auto out = in.typedAlike(shape);
  for (const auto& [inIndex, outIndex] :
       llvm::zip(in.view().indices(), out.view().indices())) {
    out.insertElement(outIndex, in.extractElement(inIndex));
  }
  return out;
}

InterpreterValue getInitOperand(mlir::Operation* op, int64_t index,
                                MutableArrayRef<InterpreterValue> args) {
  return getInitOperand(op->getOperands(), index, args);
}

InterpreterValue getInitOperand(mlir::ValueRange values, int64_t index,
                                ArrayRef<InterpreterValue> args) {
  assert(args.size() == values.size() && "expected matching sizes");
  return values[index].getType().isa<TensorType>() ? args[index].clone()
                                                   : args[index];
}

InterpreterValue transposeImpl(const InterpreterValue& in,
                               ArrayRef<int64_t> permutation) {
  auto out = in;
  auto& view = out.view();

  view.sizes.clear();
  view.strides.clear();
  for (int64_t p : permutation) {
    view.sizes.push_back(in.view().sizes[p]);
    view.strides.push_back(in.view().strides[p]);
  }

  return out;
}

int64_t dimImpl(const InterpreterValue& in, int64_t index,
                InterpreterState& state) {
  if (index < 0 || index >= in.view().rank()) {
    state.addFailure("dimension index out of bounds");
    return 0;
  }
  return in.view().sizes[index];
}

llvm::SmallVector<InterpreterValue> noOpTerminator(
    MutableArrayRef<InterpreterValue> args, mlir::Operation*,
    InterpreterState&) {
  return llvm::to_vector(args);
}

}  // namespace interpreter
}  // namespace mlir
