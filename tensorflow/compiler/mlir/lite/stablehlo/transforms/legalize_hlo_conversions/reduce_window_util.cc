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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce_window_util.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {

ReduceWindowView::ReduceWindowView(mhlo::ReduceWindowOp op) {
  rank_ = op.getWindowDimensions().size();
  window_dims_ =
      SmallVector<int64_t, 4>(op.getWindowDimensions().getValues<int64_t>());
  window_strides_ = ResolveStridesOrDilations(rank_, op.getWindowStrides());
  window_dilations_ = ResolveStridesOrDilations(rank_, op.getWindowDilations());
  base_dilations_ = ResolveStridesOrDilations(rank_, op.getBaseDilations());
  paddings_ = ResolvePadding(rank_, op.getPadding());
  window_size_ = 1;
  for (auto d : window_dims_) {
    window_size_ *= d;
  }
}

std::optional<Layout> ReduceWindowView::GuessLayout() const {
  auto zip_dims_strides = llvm::zip(WindowDims(), WindowStrides());
  auto simple_window_dims =
      llvm::to_vector(llvm::map_range(zip_dims_strides, [](auto it) {
        return std::get<0>(it) == 1 && std::get<1>(it) == 1;
      }));

  if (llvm::count(simple_window_dims, 1) < 2) {
    return std::nullopt;
  }

  const bool is_channel_last =
      simple_window_dims[0] && simple_window_dims[Rank() - 1];
  if (is_channel_last) {
    return Layout(0, Rank() - 1,
                  llvm::to_vector(llvm::seq<int64_t>(1, Rank() - 1)));
  }

  const bool is_channel_first = simple_window_dims[0] && simple_window_dims[1];
  if (is_channel_first) {
    return Layout(0, 1, llvm::to_vector(llvm::seq<int64_t>(2, Rank())));
  }

  // In theory, we can support any layout with at least 2 1's in
  // `simple_window_dims` by permuting layouts such that the 1's are
  // the first and last position. Unclear if such a case ever comes up.
  return std::nullopt;
}

}  // namespace mlir::odml
