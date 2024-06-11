/* Copyright 2023 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_
#define XLA_MLIR_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_

#include <cstdint>

#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/ViewLikeInterface.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"

namespace mlir {
namespace interpreter {

struct OffsetsSizesStrides {
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> strides;
};

// Replaces dynamic placeholders in static_vals using elements from the front
// of args, which are removed.
SmallVector<int64_t> ReplaceDynamicVals(ArrayRef<int64_t> static_vals,
                                        ArrayRef<InterpreterValue>& args);
// Same as above, but the values are already unpacked. `dynamicVals.size` must
// match the number of dynamic values in `staticVals`.
SmallVector<int64_t> ReplaceDynamicVals(ArrayRef<int64_t> static_vals,
                                        ArrayRef<int64_t> dynamic_vals);

OffsetsSizesStrides ExtractOffsetsSizesStrides(
    ArrayRef<InterpreterValue> args, OffsetSizeAndStrideOpInterface op);
OffsetsSizesStrides ExtractOffsetsSizesStrides(
    ArrayRef<int64_t> dynamic_offsets, ArrayRef<int64_t> dynamic_sizes,
    ArrayRef<int64_t> dynamic_strides, OffsetSizeAndStrideOpInterface op);

InterpreterValue ReshapeTensor(const InterpreterValue& in,
                               ArrayRef<int64_t> shape);

// gets the given operand, cloning its storage if it is a tensor.
InterpreterValue GetInitOperand(mlir::Operation* op, int64_t index,
                                MutableArrayRef<InterpreterValue> args);
InterpreterValue GetInitOperand(mlir::ValueRange values, int64_t index,
                                ArrayRef<InterpreterValue> args);

// Common implementations for ops from different dialects but sharing the same
// semantics.
InterpreterValue TransposeImpl(const InterpreterValue& in,
                               ArrayRef<int64_t> permutation);
int64_t DimImpl(const InterpreterValue& in, int64_t index,
                InterpreterState& state);

// Terminator that just returns its args.
llvm::SmallVector<InterpreterValue> NoOpTerminator(
    MutableArrayRef<InterpreterValue> args, mlir::Operation*,
    InterpreterState&);

int64_t EvalAffineExpr(AffineExpr expr, ArrayRef<int64_t> dims,
                       ArrayRef<int64_t> symbols);
llvm::SmallVector<int64_t> EvalAffineMap(AffineMap map, ArrayRef<int64_t> dims,
                                         ArrayRef<int64_t> symbols);
llvm::SmallVector<int64_t> EvalAffineMap(AffineMap map,
                                         ArrayRef<int64_t> operands);

}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_INTERPRETER_DIALECTS_UTIL_H_
