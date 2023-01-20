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

#ifndef MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_
#define MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"

namespace mlir {
namespace interpreter {

struct OffsetsSizesStrides {
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> strides;
};

// Replaces dynamic placeholders in staticVals using elements from the front
// of args, which are removed.
SmallVector<int64_t> replaceDynamicVals(ArrayRef<int64_t> staticVals,
                                        ArrayRef<InterpreterValue>& args);
// Same as above, but the values are already unpacked. `dynamicVals.size` must
// match the number of dynamic values in `staticVals`.
SmallVector<int64_t> replaceDynamicVals(ArrayRef<int64_t> staticVals,
                                        ArrayRef<int64_t> dynamicVals);

OffsetsSizesStrides extractOffsetsSizesStrides(
    ArrayRef<InterpreterValue> args, OffsetSizeAndStrideOpInterface op);
OffsetsSizesStrides extractOffsetsSizesStrides(
    ArrayRef<int64_t> dynamicOffsets, ArrayRef<int64_t> dynamicSizes,
    ArrayRef<int64_t> dynamicStrides, OffsetSizeAndStrideOpInterface op);

InterpreterValue reshapeTensor(const InterpreterValue& in,
                               ArrayRef<int64_t> shape);

// Gets the given operand, cloning its storage if it is a tensor.
InterpreterValue getInitOperand(mlir::Operation* op, int64_t index,
                                MutableArrayRef<InterpreterValue> args);
InterpreterValue getInitOperand(mlir::ValueRange values, int64_t index,
                                ArrayRef<InterpreterValue> args);

// Common implementations for ops from different dialects but sharing the same
// semantics.
InterpreterValue transposeImpl(const InterpreterValue& in,
                               ArrayRef<int64_t> permutation);
int64_t dimImpl(const InterpreterValue& in, int64_t index,
                InterpreterState& state);

// Terminator that just returns its args.
llvm::SmallVector<InterpreterValue> noOpTerminator(
    MutableArrayRef<InterpreterValue> args, mlir::Operation*,
    InterpreterState&);

int64_t evalAffineExpr(AffineExpr expr, ArrayRef<int64_t> dims);
llvm::SmallVector<int64_t> evalAffineMap(AffineMap map, ArrayRef<int64_t> dims);

}  // namespace interpreter
}  // namespace mlir

#endif  // MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_
