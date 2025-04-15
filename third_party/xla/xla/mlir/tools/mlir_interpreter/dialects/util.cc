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

#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"

#include <cassert>
#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"

namespace mlir {
namespace interpreter {

SmallVector<int64_t> ReplaceDynamicVals(llvm::ArrayRef<int64_t> static_vals,
                                        ArrayRef<InterpreterValue>& args) {
  llvm::SmallVector<int64_t> out;
  for (int64_t val : static_vals) {
    if (ShapedType::isDynamic(val)) {
      out.push_back(std::get<int64_t>(args.front().storage));
      args = args.drop_front(1);
    } else {
      out.push_back(val);
    }
  }
  return out;
}

SmallVector<int64_t> ReplaceDynamicVals(ArrayRef<int64_t> static_vals,
                                        ArrayRef<int64_t> dynamic_vals) {
  llvm::SmallVector<int64_t> out;
  for (int64_t val : static_vals) {
    if (ShapedType::isDynamic(val)) {
      out.push_back(dynamic_vals.front());
      dynamic_vals = dynamic_vals.drop_front(1);
    } else {
      out.push_back(val);
    }
  }
  assert(dynamic_vals.empty() && "expected no leftover dynamic values");
  return out;
}

OffsetsSizesStrides ExtractOffsetsSizesStrides(
    ArrayRef<InterpreterValue> args, OffsetSizeAndStrideOpInterface op) {
  auto offsets = ReplaceDynamicVals(op.getStaticOffsets(), args);
  auto sizes = ReplaceDynamicVals(op.getStaticSizes(), args);
  auto strides = ReplaceDynamicVals(op.getStaticStrides(), args);
  return {offsets, sizes, strides};
}

OffsetsSizesStrides ExtractOffsetsSizesStrides(
    ArrayRef<int64_t> dynamic_offsets, ArrayRef<int64_t> dynamic_sizes,
    ArrayRef<int64_t> dynamic_strides, OffsetSizeAndStrideOpInterface op) {
  auto offsets = ReplaceDynamicVals(op.getStaticOffsets(), dynamic_offsets);
  auto sizes = ReplaceDynamicVals(op.getStaticSizes(), dynamic_sizes);
  auto strides = ReplaceDynamicVals(op.getStaticStrides(), dynamic_strides);
  return {offsets, sizes, strides};
}

InterpreterValue ReshapeTensor(const InterpreterValue& in,
                               ArrayRef<int64_t> shape) {
  // This doesn't need a copy in many cases, but it's easier that way.
  auto out = in.TypedAlike(shape);
  for (const auto& [in_index, out_index] :
       llvm::zip(in.View().Indices(), out.View().Indices())) {
    out.InsertElement(out_index, in.ExtractElement(in_index));
  }
  return out;
}

InterpreterValue GetInitOperand(mlir::Operation* op, int64_t index,
                                MutableArrayRef<InterpreterValue> args) {
  return GetInitOperand(op->getOperands(), index, args);
}

InterpreterValue GetInitOperand(mlir::ValueRange values, int64_t index,
                                ArrayRef<InterpreterValue> args) {
  assert(args.size() == values.size() && "expected matching sizes");
  return mlir::isa<TensorType>(values[index].getType()) ? args[index].Clone()
                                                        : args[index];
}

InterpreterValue TransposeImpl(const InterpreterValue& in,
                               ArrayRef<int64_t> permutation) {
  auto out = in;
  auto& view = out.View();

  view.sizes.clear();
  view.strides.clear();
  for (int64_t p : permutation) {
    view.sizes.push_back(in.View().sizes[p]);
    view.strides.push_back(in.View().strides[p]);
  }

  return out;
}

int64_t DimImpl(const InterpreterValue& in, int64_t index,
                InterpreterState& state) {
  if (index < 0 || index >= in.View().num_dimensions()) {
    state.AddFailure("dimension index out of bounds");
    return 0;
  }
  return in.View().sizes[index];
}

llvm::SmallVector<InterpreterValue> NoOpTerminator(
    MutableArrayRef<InterpreterValue> args, mlir::Operation*,
    InterpreterState&) {
  return llvm::to_vector(args);
}

int64_t EvalAffineExpr(AffineExpr expr, ArrayRef<int64_t> dims,
                       ArrayRef<int64_t> symbols) {
  int64_t lhs = 0, rhs = 0;
  if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
    lhs = EvalAffineExpr(bin.getLHS(), dims, symbols);
    rhs = EvalAffineExpr(bin.getRHS(), dims, symbols);
  }
  switch (expr.getKind()) {
    case AffineExprKind::Add:
      return lhs + rhs;
    case AffineExprKind::Mul:
      return lhs * rhs;
    case AffineExprKind::Mod:
      return llvm::mod(lhs, rhs);
    case AffineExprKind::FloorDiv:
      return llvm::divideFloorSigned(lhs, rhs);
    case AffineExprKind::CeilDiv:
      return llvm::divideCeilSigned(lhs, rhs);
    case AffineExprKind::Constant:
      return expr.cast<AffineConstantExpr>().getValue();
    case AffineExprKind::DimId:
      return dims[expr.cast<AffineDimExpr>().getPosition()];
    case AffineExprKind::SymbolId:
      return symbols[expr.cast<AffineSymbolExpr>().getPosition()];
  }
}

SmallVector<int64_t> EvalAffineMap(AffineMap map, ArrayRef<int64_t> dims,
                                   ArrayRef<int64_t> symbols) {
  SmallVector<int64_t> result;
  for (auto expr : map.getResults()) {
    result.push_back(EvalAffineExpr(expr, dims, symbols));
  }
  return result;
}

llvm::SmallVector<int64_t> EvalAffineMap(AffineMap map,
                                         ArrayRef<int64_t> operands) {
  return EvalAffineMap(map, operands.take_front(map.getNumDims()),
                       operands.drop_front(map.getNumDims()));
}

}  // namespace interpreter
}  // namespace mlir
