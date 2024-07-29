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

#include <cstddef>
#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

llvm::SmallVector<int64_t, 4> Layout::GetPermForReLayout(
    const Layout& to_layout) const {
  llvm::SmallVector<int64_t, 4> perm(to_layout.Rank());
  perm[to_layout.SpecialDim1()] = SpecialDim1();
  perm[to_layout.SpecialDim2()] = SpecialDim2();
  for (const auto [to_spatial, from_spatial] :
       llvm::zip(to_layout.Spatials(), Spatials())) {
    perm[to_spatial] = from_spatial;
  }
  return perm;
}

llvm::SmallVector<int64_t, 4> Layout::PermuteShape(
    const Layout& to_layout, llvm::ArrayRef<int64_t> shape) const {
  llvm::SmallVector<int64_t, 4> new_shape(to_layout.Rank());
  const auto perm = GetPermForReLayout(to_layout);
  for (const auto [ind, val] : llvm::enumerate(perm)) {
    new_shape[ind] = shape[val];
  }
  return new_shape;
}

bool Layout::HasSpecialDims(int64_t special_dim1, int64_t special_dim2) const {
  return SpecialDim1() == special_dim1 && SpecialDim2() == special_dim2;
}

bool Layout::AreSpatialsIota() const {
  llvm::ArrayRef<int64_t> spatials = Spatials();
  return llvm::all_of(llvm::enumerate(spatials), [&](const auto& it) {
    return it.index() == 0 || (it.value() == spatials[it.index() - 1] + 1);
  });
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
    for (int i = 0; i < num_spatials; ++i) {
      res.push_back(DimPadding(0, 0));
    }
    return res;
  }
  auto padding = opt_padding.value();
  if (padding.isSplat()) {
    const int64_t val = padding.getSplatValue<int64_t>();
    for (int i = 0; i < num_spatials; ++i) {
      res.push_back(DimPadding(val, val));
    }
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
    : input_layout_(
          Layout{op.getDimensionNumbers().getInputBatchDimension(),
                 op.getDimensionNumbers().getInputFeatureDimension(),
                 op.getDimensionNumbers().getInputSpatialDimensions()}),
      kernel_layout_(
          Layout{op.getDimensionNumbers().getKernelInputFeatureDimension(),
                 op.getDimensionNumbers().getKernelOutputFeatureDimension(),
                 op.getDimensionNumbers().getKernelSpatialDimensions()}),
      output_layout_(
          Layout{op.getDimensionNumbers().getOutputBatchDimension(),
                 op.getDimensionNumbers().getOutputFeatureDimension(),
                 op.getDimensionNumbers().getOutputSpatialDimensions()}),
      input_shape_(
          llvm::SmallVector<int64_t, 4>(op.getLhs().getType().getShape())),
      kernel_shape_(
          llvm::SmallVector<int64_t, 4>(op.getRhs().getType().getShape())),
      output_shape_(
          llvm::SmallVector<int64_t, 4>(op.getResult().getType().getShape())),
      batch_group_count_(op.getBatchGroupCount()),
      feature_group_count_(op.getFeatureGroupCount()),
      element_type_(op.getLhs().getType().getElementType()) {
  const int64_t num_spatials = InputLayout().NumSpatials();

  strides_ = ResolveStridesOrDilations(num_spatials, op.getWindowStrides());

  input_dilations_ =
      ResolveStridesOrDilations(num_spatials, op.getLhsDilation());
  kernel_dilations_ =
      ResolveStridesOrDilations(num_spatials, op.getRhsDilation());

  padding_ = ResolvePadding(num_spatials, op.getPadding());

  window_reversal_ =
      ResolveWindowReversal(num_spatials, op.getWindowReversal());
}

Value CreatePadOpFromConvPadding(OpBuilder& b, mhlo::ConvolutionOp op) {
  const ConvData data(op);
  const auto rank = data.InputLayout().Rank();
  auto input_spatials = data.InputLayout().Spatials();

  llvm::SmallVector<int64_t, 4> hi_padding(rank, 0);
  llvm::SmallVector<int64_t, 4> lo_padding(rank, 0);

  for (const auto& [ind, dim_padding] : llvm::enumerate(data.Padding())) {
    const size_t cur_input_spatial = input_spatials[ind];
    hi_padding[cur_input_spatial] = dim_padding.Hi();
    lo_padding[cur_input_spatial] = dim_padding.Lo();
  }

  const llvm::SmallVector<int64_t, 4> interior_padding(rank, 0);

  auto padding_attr_type = RankedTensorType::get({rank}, b.getI64Type());
  auto hi_padding_attr =
      DenseIntElementsAttr::get(padding_attr_type, hi_padding);
  auto lo_padding_attr =
      DenseIntElementsAttr::get(padding_attr_type, lo_padding);
  auto interior_padding_attr =
      DenseIntElementsAttr::get(padding_attr_type, interior_padding);

  auto padding_value_type = RankedTensorType::get({}, data.ElementType());
  auto padding_value_attr = b.getZeroAttr(padding_value_type);
  auto padding_value_op =
      b.create<arith::ConstantOp>(op->getLoc(), padding_value_attr);

  auto pad_op = b.create<mhlo::PadOp>(padding_value_op->getLoc(), op.getLhs(),
                                      padding_value_op, hi_padding_attr,
                                      lo_padding_attr, interior_padding_attr);

  return pad_op;
}

}  // namespace mlir::odml
