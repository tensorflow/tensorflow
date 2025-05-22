/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/array.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

TEST(ArrayTest, UninitializedDimsCtor) {
  Array<int> uninit({2, 3});
  EXPECT_EQ(uninit.num_dimensions(), 2);
  EXPECT_EQ(uninit.dim(0), 2);
  EXPECT_EQ(uninit.dim(1), 3);
  EXPECT_EQ(uninit.num_elements(), 6);
}

TEST(ArrayTest, FillCtor) {
  Array<int> fullof7({1, 2, 3}, 7);

  EXPECT_EQ(fullof7.dim(0), 1);
  EXPECT_EQ(fullof7.dim(1), 2);
  EXPECT_EQ(fullof7.dim(2), 3);

  for (int64_t n0 = 0; n0 < fullof7.dim(0); ++n0) {
    for (int64_t n1 = 0; n1 < fullof7.dim(1); ++n1) {
      for (int64_t n2 = 0; n2 < fullof7.dim(2); ++n2) {
        EXPECT_EQ(fullof7(n0, n1, n2), 7);
      }
    }
  }
}

TEST(ArrayTest, InitializerListCtor) {
  Array<int> arr({{1, 2, 3}, {4, 5, 6}});

  EXPECT_EQ(arr.dim(0), 2);
  EXPECT_EQ(arr.dim(1), 3);

  EXPECT_EQ(arr(0, 0), 1);
  EXPECT_EQ(arr(0, 1), 2);
  EXPECT_EQ(arr(0, 2), 3);
  EXPECT_EQ(arr(1, 0), 4);
  EXPECT_EQ(arr(1, 1), 5);
  EXPECT_EQ(arr(1, 2), 6);
}

TEST(ArrayTest, Transpose1DNoOp) {
  Array<int> arr({3});
  arr.FillWithMultiples(10);  // {0, 10, 20}

  ASSERT_EQ(arr.num_dimensions(), 1);
  ASSERT_EQ(arr.dim(0), 3);

  arr.TransposeDimensions({0});
  ASSERT_EQ(arr.num_dimensions(), 1);
  ASSERT_EQ(arr.dim(0), 3);
  EXPECT_EQ(arr(0), 0);
  EXPECT_EQ(arr(1), 10);
  EXPECT_EQ(arr(2), 20);
}

TEST(ArrayTest, Transpose2DNoOp) {
  Array<int> arr({{0, 10, 20}, {30, 40, 50}});
  ASSERT_EQ(arr.num_dimensions(), 2);
  ASSERT_EQ(arr.dim(0), 2);
  ASSERT_EQ(arr.dim(1), 3);

  arr.TransposeDimensions({0, 1});
  ASSERT_EQ(arr.num_dimensions(), 2);
  ASSERT_EQ(arr.dim(0), 2);
  ASSERT_EQ(arr.dim(1), 3);
  EXPECT_EQ(arr(0, 0), 0);
  EXPECT_EQ(arr(0, 1), 10);
  EXPECT_EQ(arr(0, 2), 20);
  EXPECT_EQ(arr(1, 0), 30);
  EXPECT_EQ(arr(1, 1), 40);
  EXPECT_EQ(arr(1, 2), 50);
}

TEST(ArrayTest, Transpose2DSwap) {
  Array<int> arr({{0, 10, 20}, {30, 40, 50}});
  ASSERT_EQ(arr.num_dimensions(), 2);
  ASSERT_EQ(arr.dim(0), 2);
  ASSERT_EQ(arr.dim(1), 3);

  arr.TransposeDimensions({1, 0});
  ASSERT_EQ(arr.num_dimensions(), 2);
  ASSERT_EQ(arr.dim(0), 3);
  ASSERT_EQ(arr.dim(1), 2);
  EXPECT_EQ(arr(0, 0), 0);
  EXPECT_EQ(arr(0, 1), 30);
  EXPECT_EQ(arr(1, 0), 10);
  EXPECT_EQ(arr(1, 1), 40);
  EXPECT_EQ(arr(2, 0), 20);
  EXPECT_EQ(arr(2, 1), 50);
}

TEST(ArrayTest, Transpose3DNoOp) {
  Array<int> arr({{{0, 10}, {20, 30}}, {{40, 50}, {60, 70}}});
  ASSERT_EQ(arr.num_dimensions(), 3);
  ASSERT_EQ(arr.dim(0), 2);
  ASSERT_EQ(arr.dim(1), 2);
  ASSERT_EQ(arr.dim(2), 2);

  arr.TransposeDimensions({0, 1, 2});
  ASSERT_EQ(arr.num_dimensions(), 3);
  ASSERT_EQ(arr.dim(0), 2);
  ASSERT_EQ(arr.dim(1), 2);
  ASSERT_EQ(arr.dim(2), 2);
  EXPECT_EQ(arr(0, 0, 0), 0);
  EXPECT_EQ(arr(0, 0, 1), 10);
  EXPECT_EQ(arr(0, 1, 0), 20);
  EXPECT_EQ(arr(0, 1, 1), 30);
  EXPECT_EQ(arr(1, 0, 0), 40);
  EXPECT_EQ(arr(1, 0, 1), 50);
  EXPECT_EQ(arr(1, 1, 0), 60);
  EXPECT_EQ(arr(1, 1, 1), 70);
}

TEST(ArrayTest, Transpose3DCyclic) {
  Array<int> arr({{{0, 10}, {20, 30}, {40, 50}},
                  {{60, 70}, {80, 90}, {100, 110}},
                  {{120, 130}, {140, 150}, {160, 170}},
                  {{180, 190}, {200, 210}, {220, 230}}});
  ASSERT_EQ(arr.num_dimensions(), 3);
  ASSERT_EQ(arr.dim(0), 4);
  ASSERT_EQ(arr.dim(1), 3);
  ASSERT_EQ(arr.dim(2), 2);
  Array<int> arr_before_transpose = arr;

  arr.TransposeDimensions({1, 2, 0});
  ASSERT_EQ(arr.num_dimensions(), 3);
  ASSERT_EQ(arr.dim(0), 3);
  ASSERT_EQ(arr.dim(1), 2);
  ASSERT_EQ(arr.dim(2), 4);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(arr(i, j, k), arr_before_transpose(k, i, j));
      }
    }
  }
}

Array<int> Make4DArray() {
  Array<int> arr({2, 4, 8, 16});
  CHECK_EQ(arr.num_elements(), 1024);
  arr.FillWithMultiples(10);
  return arr;
}

TEST(ArrayTest, Transpose4DNoOp) {
  Array<int> arr1 = Make4DArray();
  Array<int> arr2 = arr1;
  arr2.TransposeDimensions({0, 1, 2, 3});
  ASSERT_EQ(arr2.num_dimensions(), 4);
  ASSERT_EQ(arr2.dim(0), 2);
  ASSERT_EQ(arr2.dim(1), 4);
  ASSERT_EQ(arr2.dim(2), 8);
  ASSERT_EQ(arr2.dim(3), 16);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 8; ++k) {
        for (int l = 0; l < 16; ++l) {
          EXPECT_EQ(arr1(i, j, k, l), arr2(i, j, k, l));
        }
      }
    }
  }
}

TEST(ArrayTest, Transpose4DCyclic) {
  Array<int> arr1 = Make4DArray();
  Array<int> arr2 = arr1;
  arr2.TransposeDimensions({1, 2, 3, 0});
  ASSERT_EQ(arr2.num_dimensions(), 4);
  ASSERT_EQ(arr2.dim(0), 4);
  ASSERT_EQ(arr2.dim(1), 8);
  ASSERT_EQ(arr2.dim(2), 16);
  ASSERT_EQ(arr2.dim(3), 2);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j) {
      for (int k = 0; k < 16; ++k) {
        for (int l = 0; l < 2; ++l) {
          EXPECT_EQ(arr2(i, j, k, l), arr1(l, i, j, k));
        }
      }
    }
  }
}

TEST(ArrayTest, Transpose4DSomePermutation) {
  Array<int> arr1 = Make4DArray();
  Array<int> arr2 = arr1;
  arr2.TransposeDimensions({3, 1, 0, 2});
  ASSERT_EQ(arr2.num_dimensions(), 4);
  ASSERT_EQ(arr2.dim(0), 16);
  ASSERT_EQ(arr2.dim(1), 4);
  ASSERT_EQ(arr2.dim(2), 2);
  ASSERT_EQ(arr2.dim(3), 8);
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 8; ++l) {
          EXPECT_EQ(arr2(i, j, k, l), arr1(k, j, l, i));
        }
      }
    }
  }
}

TEST(ArrayTest, Transpose1DEmpty) {
  Array<int> arr({0});

  ASSERT_EQ(arr.num_dimensions(), 1);
  ASSERT_EQ(arr.dim(0), 0);

  arr.TransposeDimensions({0});
  EXPECT_EQ(arr.num_dimensions(), 1);
  EXPECT_EQ(arr.dim(0), 0);
}

TEST(ArrayTest, Transpose2DEmpty) {
  Array<int> arr({0, 2});

  ASSERT_EQ(arr.num_dimensions(), 2);
  ASSERT_EQ(arr.dim(0), 0);
  ASSERT_EQ(arr.dim(1), 2);

  arr.TransposeDimensions({1, 0});
  EXPECT_EQ(arr.num_dimensions(), 2);
  EXPECT_EQ(arr.dim(0), 2);
  EXPECT_EQ(arr.dim(1), 0);
}

TEST(ArrayTest, Transpose3DEmpty) {
  Array<int> arr({0, 5, 7});

  ASSERT_EQ(arr.num_dimensions(), 3);
  ASSERT_EQ(arr.dim(0), 0);
  ASSERT_EQ(arr.dim(1), 5);
  ASSERT_EQ(arr.dim(2), 7);

  arr.TransposeDimensions({1, 2, 0});
  EXPECT_EQ(arr.num_dimensions(), 3);
  EXPECT_EQ(arr.dim(0), 5);
  EXPECT_EQ(arr.dim(1), 7);
  EXPECT_EQ(arr.dim(2), 0);
}

TEST(ArrayTest, InitializerListCtorHalf) {
  Array<Eigen::half> d2({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  EXPECT_EQ(d2.dim(0), 2);
  EXPECT_EQ(d2.dim(1), 3);

  Array<Eigen::half> d3({{{1.0f}, {4.0f}}, {{1.0f}, {4.0f}}, {{1.0f}, {4.0f}}});
  EXPECT_EQ(d3.dim(0), 3);
  EXPECT_EQ(d3.dim(1), 2);
  EXPECT_EQ(d3.dim(2), 1);

  Array<Eigen::half> d4(
      {{{{1.0f}, {4.0f}}, {{1.0f}, {4.0f}}, {{1.0f}, {4.0f}}},
       {{{1.0f}, {4.0f}}, {{1.0f}, {4.0f}}, {{1.0f}, {4.0f}}}});
  EXPECT_EQ(d4.dim(0), 2);
  EXPECT_EQ(d4.dim(1), 3);
  EXPECT_EQ(d4.dim(2), 2);
  EXPECT_EQ(d4.dim(3), 1);
}

TEST(ArrayTest, IndexingReadWrite) {
  Array<int> arr({2, 3});

  EXPECT_EQ(arr(1, 1), 0);
  EXPECT_EQ(arr(1, 2), 0);
  arr(1, 1) = 51;
  arr(1, 2) = 61;
  EXPECT_EQ(arr(1, 1), 51);
  EXPECT_EQ(arr(1, 2), 61);
}

TEST(ArrayTest, DynamicIndexingReadWrite) {
  Array<int> arr({2, 3});

  std::vector<int64_t> index1 = {1, 1};
  std::vector<int64_t> index2 = {1, 2};
  EXPECT_EQ(arr(index1), 0);
  EXPECT_EQ(arr(index2), 0);
  arr(index1) = 51;
  arr(index2) = 61;
  EXPECT_EQ(arr(1, 1), 51);
  EXPECT_EQ(arr(1, 2), 61);
}

TEST(ArrayTest, IndexingReadWriteBool) {
  Array<bool> arr{{false, true, false}, {false, true, false}};

  EXPECT_EQ(arr(0, 1), true);
  EXPECT_EQ(arr(0, 2), false);
  arr(0, 1) = false;
  arr(0, 2) = true;
  EXPECT_EQ(arr(0, 1), false);
  EXPECT_EQ(arr(0, 2), true);
}

TEST(ArrayTest, Fill) {
  Array<int> fullof7({2, 3}, 7);
  for (int64_t n1 = 0; n1 < fullof7.dim(0); ++n1) {
    for (int64_t n2 = 0; n2 < fullof7.dim(1); ++n2) {
      EXPECT_EQ(fullof7(n1, n2), 7);
    }
  }

  fullof7.Fill(11);
  for (int64_t n1 = 0; n1 < fullof7.dim(0); ++n1) {
    for (int64_t n2 = 0; n2 < fullof7.dim(1); ++n2) {
      EXPECT_EQ(fullof7(n1, n2), 11);
    }
  }
}

TEST(ArrayTest, DataPointer) {
  Array<int> arr{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(arr.data()[0], 1);
}

TEST(ArrayTest, StringificationEmpty) {
  Array<int64_t> arr({}, 0);
  constexpr absl::string_view expected = "";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, Stringification1D) {
  Array<int64_t> arr({2}, 1);
  const std::string expected = R"([1, 1])";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, StringificationEmpty1D) {
  Array<int64_t> arr({0}, 0);
  constexpr absl::string_view expected = "[]";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, Stringification2D) {
  Array<int64_t> arr({2, 3}, 7);
  const std::string expected = "[[7, 7, 7],\n [7, 7, 7]]";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, StringificationEmpty2D) {
  Array<int64_t> arr({0, 0}, 0);
  constexpr absl::string_view expected = "[[]]";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, Stringification3D) {
  Array<int64_t> arr({2, 3, 4}, 5);
  const std::string expected = R"([[[5, 5, 5, 5],
  [5, 5, 5, 5],
  [5, 5, 5, 5]],
 [[5, 5, 5, 5],
  [5, 5, 5, 5],
  [5, 5, 5, 5]]])";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, StringificationEmpty3D) {
  Array<int64_t> arr({0, 0, 0}, 0);
  constexpr absl::string_view expected = "[[[]]]";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, Stringification3DOneZeroDim) {
  Array<int64_t> arr({1, 0, 2}, 0);
  constexpr absl::string_view expected = "[[[, ]]]";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, Each) {
  Array<int64_t> arr({2, 3, 4});
  arr.FillWithMultiples(1);

  int64_t each_count = 0, each_sum = 0;
  arr.Each([&](absl::Span<const int64_t> idx, int cell) {
    int64_t lin_idx = idx[0] * 12 + idx[1] * 4 + idx[2];
    EXPECT_EQ(lin_idx, cell);
    each_count++;
    each_sum += cell;
  });
  EXPECT_EQ(arr.num_elements(), each_count);
  EXPECT_EQ(arr.num_elements() * (arr.num_elements() - 1) / 2, each_sum);
}

TEST(ArrayTest, Slice) {
  Array<int64_t> arr({2, 4});
  arr.FillWithMultiples(1);

  Array<int64_t> identity_slice = arr.Slice({0, 0}, {2, 4});
  EXPECT_EQ(identity_slice.dimensions(), arr.dimensions());
  for (auto it1 = arr.begin(), it2 = identity_slice.begin(), e = arr.end();
       it1 != e; ++it1, ++it2) {
    EXPECT_EQ(*it1, *it2);
  }

  Array<int64_t> sub_slice = arr.Slice({1, 0}, {2, 2});
  EXPECT_EQ(sub_slice.dimensions(), (std::vector<int64_t>{1, 2}));
  const std::string expected = R"([[4, 5]])";
  EXPECT_EQ(expected, sub_slice.ToString());
}

TEST(ArrayTest, UpdateSlice) {
  Array<int64_t> arr({3, 4});
  arr.FillWithMultiples(1);

  Array<int64_t> sub_arr({2, 2});
  sub_arr.FillWithMultiples(3);

  arr.UpdateSlice(sub_arr, {1, 1});

  const std::string expected = R"([[0, 1, 2, 3],
 [4, 0, 3, 7],
 [8, 6, 9, 11]])";
  EXPECT_EQ(expected, arr.ToString());
}

}  // namespace
}  // namespace xla
