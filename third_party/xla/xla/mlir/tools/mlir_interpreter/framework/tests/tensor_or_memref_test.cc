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

#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace interpreter {
namespace {

using ::testing::ElementsAre;

TEST(TensorOrMemrefTest, DefaultStrides) {
  EXPECT_THAT(BufferView::GetDefaultStrides({1, 2, 3}), ElementsAre(6, 3, 1));
}

TEST(TensorOrMemrefTest, StridesForLayout) {
  EXPECT_THAT(BufferView::GetStridesForLayout({1, 2, 3}, {2, 1, 0}),
              ElementsAre(6, 3, 1));
  EXPECT_THAT(BufferView::GetStridesForLayout({1, 2, 3}, {0, 1, 2}),
              ElementsAre(1, 1, 2));
  EXPECT_THAT(BufferView::GetStridesForLayout({3, 3, 3, 3}, {3, 0, 1, 2}),
              ElementsAre(27, 1, 3, 9));
}

std::optional<int64_t> GetCollapsedStrideNaive(llvm::ArrayRef<int64_t> dims,
                                               const BufferView& view) {
  BufferView f;
  for (int64_t dim : dims) {
    f.sizes.push_back(view.sizes[dim]);
  }

  // Find all physical indices for the dimensions.
  llvm::SmallBitVector v(view.GetNumElements());
  for (const auto& indices : f.Indices()) {
    SmallVector<int64_t> view_indices(view.Rank());
    for (auto [dim, index] : llvm::zip(dims, indices)) {
      view_indices[dim] = index;
    }
    v[*view.GetPhysicalIndex(view_indices)] = true;
  }

  if (v.count() != f.GetNumElements()) return std::nullopt;
  if (f.GetNumElements() <= 1) return 0;

  // Check that they have a common stride.
  int64_t min = v.find_first();
  int64_t expected_stride = (v.find_last() - min) / (f.GetNumElements() - 1);
  for (int64_t i = 0; i < f.GetNumElements(); ++i) {
    if (!v[i * expected_stride + min]) {
      return std::nullopt;
    }
  }

  return expected_stride;
}

TEST(TensorOrMemrefTest, CollapsedStride) {
  BufferView view{.sizes = {1, 2, 3, 1, 5},
                  .strides = BufferView::GetDefaultStrides({1, 2, 3, 1, 5})};

  auto check_all = [&]() {
    for (int64_t i = 0; i < (1 << view.Rank()); ++i) {
      SmallVector<int64_t> dims;
      for (int64_t dim = 0; dim < view.Rank(); ++dim) {
        if (i & (1 << dim)) dims.push_back(dim);
      }

      do {
        auto v = view.GetCollapsedStride(dims);
        auto n = GetCollapsedStrideNaive(dims, view);
        EXPECT_EQ(n, v) << "checking " << absl::StrJoin(dims, ", ");
      } while (std::next_permutation(dims.begin(), dims.end()));
    }
  };

  check_all();
  ASSERT_TRUE(view.Slice(3, 0).succeeded());
  check_all();
}

}  // namespace
}  // namespace interpreter
}  // namespace mlir
