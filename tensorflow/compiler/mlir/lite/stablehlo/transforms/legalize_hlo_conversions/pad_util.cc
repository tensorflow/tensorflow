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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad_util.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"

namespace mlir::odml {

DenseI64ArrayAttr SliceStartFromNegPadLows(stablehlo::PadOp op) {
  auto vals = op.getEdgePaddingLow();
  auto starts = llvm::map_range(
      vals, [](auto v) -> int64_t { return (v >= 0) ? 0 : -1 * v; });
  return DenseI64ArrayAttr::get(op.getContext(), llvm::to_vector(starts));
}

DenseI64ArrayAttr SliceEndFromNegPadHighs(stablehlo::PadOp op) {
  auto vals = op.getEdgePaddingHigh();
  auto zip = llvm::zip(vals, op.getOperand().getType().getShape());
  auto ends = llvm::map_range(zip, [](auto it) -> int64_t {
    return (std::get<0>(it) >= 0) ? std::get<1>(it)
                                  : std::get<1>(it) + std::get<0>(it);
  });
  return DenseI64ArrayAttr::get(op.getContext(), llvm::to_vector(ends));
}

DenseI64ArrayAttr ReplaceNegsWithZero(llvm::ArrayRef<int64_t> data,
                                      MLIRContext* ctx) {
  auto res =
      llvm::map_range(data, [](auto v) -> int64_t { return (v < 0) ? 0 : v; });
  return DenseI64ArrayAttr::get(ctx, llvm::to_vector(res));
}

bool AnyNegativePads(stablehlo::PadOp op) {
  auto is_neg = [](int64_t v) { return v < 0; };
  auto lows_data = op.getEdgePaddingLow();
  auto highs_data = op.getEdgePaddingHigh();
  return llvm::any_of(lows_data, is_neg) || llvm::any_of(highs_data, is_neg);
}

bool TrivialInterior(stablehlo::PadOp op) {
  auto interior = op.getInteriorPadding();
  return llvm::all_of(interior, [](auto v) { return v == 0; });
}

}  // namespace mlir::odml
