/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/conv_util.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

bool Layout::IsTFLNativeLayout() const {
  const bool s1_first = SpecialDim1() == 0;
  const bool s2_last = SpecialDim2() == (Rank() - 1);

  const bool are_spatials_iota = IsIotaAttr(spatials_, NumSpatials(), 1);

  return s1_first && s2_last && are_spatials_iota;
}

llvm::SmallVector<int64_t, 2> ResolveStridesOrDilations(
    const int64_t num_spatials,
    std::optional<mlir::DenseIntElementsAttr> opt_attr) {
  if (!opt_attr.has_value()) {
    return llvm::SmallVector<int64_t, 2>(num_spatials, 1);
  }
  auto attr = opt_attr.value();
  if (attr.isSplat()) {
    return llvm::SmallVector<int64_t, 2>(num_spatials,
                                         attr.getSplatValue<int64_t>());
  }
  return llvm::SmallVector<int64_t, 2>(attr.getValues<int64_t>());
}

llvm::SmallVector<DimPadding, 2> ResolvePadding(
    const int64_t num_spatials,
    std::optional<mlir::DenseIntElementsAttr> opt_padding) {
  llvm::SmallVector<DimPadding, 2> res;
  if (!opt_padding.has_value()) {
    res.push_back(DimPadding(0, 0));
    res.push_back(DimPadding(0, 0));
    return res;
  }
  auto padding = opt_padding.value();
  if (padding.isSplat()) {
    const int64_t val = padding.getSplatValue<int64_t>();
    res.push_back(DimPadding(val, val));
    res.push_back(DimPadding(val, val));
    return res;
  }
  int64_t prev;
  for (const auto [ind, val] : llvm::enumerate(padding.getValues<int64_t>())) {
    const int64_t side = ind % 2;
    if (side == 1) {
      res.push_back(DimPadding(prev, val));
    }
    prev = val;
  }
  return res;
}

llvm::SmallVector<bool, 2> ResolveWindowReversal(
    const int64_t num_spatials,
    std::optional<mlir::DenseElementsAttr> opt_reversals) {
  if (!opt_reversals.has_value()) {
    return llvm::SmallVector<bool, 2>(num_spatials, false);
  }
  auto reversals = opt_reversals.value();
  if (reversals.isSplat()) {
    return llvm::SmallVector<bool, 2>(num_spatials,
                                      reversals.getSplatValue<bool>());
  }
  return llvm::SmallVector<bool, 2>(reversals.getValues<bool>());
}

ConvData::ConvData(mhlo::ConvolutionOp op)
    : layouts_(op.getDimensionNumbersAttr()),
      batch_group_count_(op.getBatchGroupCount()),
      feature_group_count_(op.getFeatureGroupCount()) {
  const int64_t num_spatials = layouts_.Input().NumSpatials();

  strides_ = ResolveStridesOrDilations(num_spatials, op.getWindowStrides());

  input_dilations_ =
      ResolveStridesOrDilations(num_spatials, op.getLhsDilation());
  kernel_dilations_ =
      ResolveStridesOrDilations(num_spatials, op.getRhsDilation());

  padding_ = ResolvePadding(num_spatials, op.getPadding());

  window_reversal_ =
      ResolveWindowReversal(num_spatials, op.getWindowReversal());

  input_shape_ = op.getLhs().getType().getShape();
  kernel_shape_ = op.getRhs().getType().getShape();
  output_shape_ = op.getResult().getType().getShape();

  element_type_ = op.getLhs().getType().getElementType();
}

}  // namespace mlir::odml
