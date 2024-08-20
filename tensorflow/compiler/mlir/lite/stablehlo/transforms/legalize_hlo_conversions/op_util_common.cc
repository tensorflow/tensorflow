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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

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

llvm::SmallVector<int64_t, 4> ResolveStridesOrDilations(
    int64_t rank, std::optional<mlir::DenseIntElementsAttr> opt_attr) {
  if (!opt_attr.has_value()) {
    return llvm::SmallVector<int64_t, 4>(rank, 1);
  }
  auto attr = opt_attr.value();
  if (attr.isSplat()) {
    return llvm::SmallVector<int64_t, 4>(rank, attr.getSplatValue<int64_t>());
  }
  return llvm::SmallVector<int64_t, 4>(attr.getValues<int64_t>());
}

llvm::SmallVector<DimPadding, 2> ResolvePadding(
    int64_t rank, std::optional<mlir::DenseIntElementsAttr> opt_padding) {
  llvm::SmallVector<DimPadding, 4> res;
  if (!opt_padding.has_value()) {
    for (int i = 0; i < rank; ++i) {
      res.push_back(DimPadding(0, 0));
    }
    return res;
  }
  auto padding = opt_padding.value();
  if (padding.isSplat()) {
    const int64_t val = padding.getSplatValue<int64_t>();
    for (int i = 0; i < rank; ++i) {
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

bool IsSamePaddingOnDim(int64_t in, int64_t dilate, int64_t stride, int64_t k,
                        const DimPadding& pad) {
  const int64_t pad_diff = pad.Hi() - pad.Lo();
  if (pad_diff > 1 || pad_diff < 0) {
    return false;
  }
  const int64_t pad_total = pad.Lo() + pad.Hi();
  const int64_t out = (in + stride - 1) / stride;
  const int effective_filter = (k - 1) * dilate + 1;
  return ((out - 1) * stride + effective_filter) == in + pad_total;
}

}  // namespace mlir::odml
