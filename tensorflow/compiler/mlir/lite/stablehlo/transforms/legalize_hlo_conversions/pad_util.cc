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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

ShapedType GetPaddingAttrType(mhlo::PadOp op) {
  return op.getEdgePaddingLow().getType();
}

DenseIntElementsAttr SliceStartFromNegPadLows(mhlo::PadOp op) {
  auto vals = UnrollI64Splat(op.getEdgePaddingLow());
  auto starts = llvm::map_range(
      vals, [](auto v) -> int64_t { return (v >= 0) ? 0 : -1 * v; });
  return DenseIntElementsAttr::get(GetPaddingAttrType(op),
                                   llvm::to_vector(starts));
}

DenseIntElementsAttr SliceEndFromNegPadHighs(mhlo::PadOp op) {
  auto vals = UnrollI64Splat(op.getEdgePaddingHigh());
  auto zip = llvm::zip(vals, op.getOperand().getType().getShape());
  auto ends = llvm::map_range(zip, [](auto it) -> int64_t {
    return (std::get<0>(it) >= 0) ? std::get<1>(it)
                                  : std::get<1>(it) + std::get<0>(it);
  });
  return DenseIntElementsAttr::get(GetPaddingAttrType(op),
                                   llvm::to_vector(ends));
}

DenseIntElementsAttr ReplaceNegsWithZero(DenseElementsAttr data) {
  auto vals = UnrollI64Splat(data);
  auto res =
      llvm::map_range(vals, [](auto v) -> int64_t { return (v < 0) ? 0 : v; });
  return DenseIntElementsAttr::get(data.getType(), llvm::to_vector(res));
}

bool AnyNegativePads(mhlo::PadOp op) {
  auto is_neg = [](int64_t v) { return v < 0; };
  auto lows_data = UnrollI64Splat(op.getEdgePaddingLow());
  auto highs_data = UnrollI64Splat(op.getEdgePaddingHigh());
  return llvm::any_of(lows_data, is_neg) || llvm::any_of(highs_data, is_neg);
}

bool TrivialInterior(mhlo::PadOp op) {
  auto interior = op.getInteriorPadding();
  const bool trivial_splat =
      interior.isSplat() && interior.getSplatValue<int64_t>() == 0;
  const bool all_trivial = llvm::all_of(interior.getValues<int64_t>(),
                                        [](auto v) { return v == 0; });
  return trivial_splat || all_trivial;
}

}  // namespace mlir::odml
