/* Copyright 2025 The JAX Authors.

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

#include "xla/mosaic/dialect/tpu/array_util.h"

#include <cstdint>
#include <initializer_list>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Support/LLVM.h"
#include "xla/array.h"

namespace mlir::tpu {

namespace {

using ::testing::Address;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::StrEq;

TEST(ArrayUtilTest, XlaArrayToFlatArrayRef) {
  xla::Array<int32_t> arr({2, 3}, 0);
  arr.FillIota(0);

  ArrayRef<int32_t> ref = XlaArrayToFlatArrayRef(arr);

  ASSERT_EQ(ref.size(), arr.num_elements());
  EXPECT_THAT(ref, ElementsAre(0, 1, 2, 3, 4, 5));

  // Make sure it's not a copy but a view.
  int* ptr = arr.begin();
  for (int i = 0; i < ref.size() && ptr != arr.end(); ++i, ++ptr) {
    EXPECT_THAT(ref[i], Address(Eq(ptr)));
  }
}

TEST(ArrayUtilTest, XlaArrayFromShapeAndValues) {
  xla::Array<int32_t> arr = XlaArrayFromShapeAndValues<int32_t>(
      {2, 3}, std::initializer_list<int32_t>{0, 1, 2, 3, 4, 5});

  EXPECT_THAT(arr.ToString(), StrEq(R"([[0, 1, 2],
 [3, 4, 5]])"));
}

TEST(ArrayUtilTest, UpdateSlice) {
  xla::Array<int32_t> arr({4, 5}, 0);
  updateSlice(arr, 1, {1, 1}, {3, 4});

  EXPECT_THAT(arr.ToString(), StrEq(R"([[0, 0, 0, 0, 0],
 [0, 1, 1, 1, 0],
 [0, 1, 1, 1, 0],
 [0, 0, 0, 0, 0]])"));
}

TEST(ArrayUtilTest, UpdateSliceFromRange) {
  xla::Array<int32_t> arr({4, 5}, 0);
  updateSliceFromRange(arr, std::initializer_list<int32_t>{1, 2, 3, 4, 5, 6},
                       {1, 1}, {3, 4});

  EXPECT_THAT(arr.ToString(), StrEq(R"([[0, 0, 0, 0, 0],
 [0, 1, 2, 3, 0],
 [0, 4, 5, 6, 0],
 [0, 0, 0, 0, 0]])"));
}

}  // namespace

}  // namespace mlir::tpu
