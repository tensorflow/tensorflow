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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/permutation.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::quant {
namespace {

using testing::ElementsAre;
using testing::IsEmpty;

TEST(PermutationTest, PermuteEmptyArray) {
  const SmallVector<int> permutation_result =
      Permute<int>(SmallVector<int>{}, SmallVector<int64_t>{});
  EXPECT_THAT(permutation_result, IsEmpty());
}

TEST(PermutationTest, PermuteOneElement) {
  const SmallVector<int> single_element_array = {8};
  const SmallVector<int64_t> permutation = {0};

  const SmallVector<int> permutation_result =
      Permute<int>(single_element_array, permutation);
  EXPECT_THAT(permutation_result, ElementsAre(8));
}

TEST(PermutationTest, PermuteFourElements) {
  const SmallVector<int> arr = {0, 3, 1, 2};
  // Permutation inverse of {0, 3, 1, 2}.
  const SmallVector<int64_t> permutation = {0, 2, 3, 1};

  const SmallVector<int> permutation_result = Permute<int>(arr, permutation);
  EXPECT_THAT(permutation_result, ElementsAre(0, 1, 2, 3));
}

TEST(PermutationTest, PermuteFourStringElements) {
  const SmallVector<std::string> arr = {"a", "b", "c", "d"};
  const SmallVector<int64_t> permutation = {0, 2, 3, 1};

  const SmallVector<std::string> permutation_result =
      Permute<std::string>(arr, permutation);
  EXPECT_THAT(permutation_result, ElementsAre("a", "c", "d", "b"));
}

}  // namespace
}  // namespace mlir::quant
