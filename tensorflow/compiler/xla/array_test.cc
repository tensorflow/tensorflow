/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/array.h"

#include <initializer_list>

#include "tensorflow/compiler/xla/test.h"

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

  for (int64 n0 = 0; n0 < fullof7.dim(0); ++n0) {
    for (int64 n1 = 0; n1 < fullof7.dim(1); ++n1) {
      for (int64 n2 = 0; n2 < fullof7.dim(2); ++n2) {
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

  std::vector<int64> index1 = {1, 1};
  std::vector<int64> index2 = {1, 2};
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
  for (int64 n1 = 0; n1 < fullof7.dim(0); ++n1) {
    for (int64 n2 = 0; n2 < fullof7.dim(1); ++n2) {
      EXPECT_EQ(fullof7(n1, n2), 7);
    }
  }

  fullof7.Fill(11);
  for (int64 n1 = 0; n1 < fullof7.dim(0); ++n1) {
    for (int64 n2 = 0; n2 < fullof7.dim(1); ++n2) {
      EXPECT_EQ(fullof7(n1, n2), 11);
    }
  }
}

TEST(ArrayTest, DataPointer) {
  Array<int> arr{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(arr.data()[0], 1);
}

TEST(ArrayTest, Stringification1D) {
  Array<int64> arr({2}, 1);
  const string expected = R"([1, 1])";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, Stringification2D) {
  Array<int64> arr({2, 3}, 7);
  const string expected = "[[7, 7, 7],\n [7, 7, 7]]";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, Stringification3D) {
  Array<int64> arr({2, 3, 4}, 5);
  const string expected = R"([[[5, 5, 5, 5],
  [5, 5, 5, 5],
  [5, 5, 5, 5]],
 [[5, 5, 5, 5],
  [5, 5, 5, 5],
  [5, 5, 5, 5]]])";
  EXPECT_EQ(expected, arr.ToString());
}

TEST(ArrayTest, Each) {
  Array<int64> arr({2, 3, 4});
  arr.FillWithMultiples(1);

  int64 each_count = 0, each_sum = 0;
  arr.Each([&](absl::Span<const int64> idx, int cell) {
    int64 lin_idx = idx[0] * 12 + idx[1] * 4 + idx[2];
    EXPECT_EQ(lin_idx, cell);
    each_count++;
    each_sum += cell;
  });
  EXPECT_EQ(arr.num_elements(), each_count);
  EXPECT_EQ(arr.num_elements() * (arr.num_elements() - 1) / 2, each_sum);
}

TEST(ArrayTest, Slice) {
  Array<int64> arr({2, 4});
  arr.FillWithMultiples(1);

  Array<int64> identity_slice = arr.Slice({0, 0}, {2, 4});
  EXPECT_EQ(identity_slice.dimensions(), arr.dimensions());
  for (auto it1 = arr.begin(), it2 = identity_slice.begin(), e = arr.end();
       it1 != e; ++it1, ++it2) {
    EXPECT_EQ(*it1, *it2);
  }

  Array<int64> sub_slice = arr.Slice({1, 0}, {2, 2});
  EXPECT_EQ(sub_slice.dimensions(), (std::vector<int64>{1, 2}));
  const string expected = R"([[4, 5]])";
  EXPECT_EQ(expected, sub_slice.ToString());
}

TEST(ArrayTest, UpdateSlice) {
  Array<int64> arr({3, 4});
  arr.FillWithMultiples(1);

  Array<int64> sub_arr({2, 2});
  sub_arr.FillWithMultiples(3);

  arr.UpdateSlice(sub_arr, {1, 1});

  const string expected = R"([[0, 1, 2, 3],
 [4, 0, 3, 7],
 [8, 6, 9, 11]])";
  EXPECT_EQ(expected, arr.ToString());
}

}  // namespace
}  // namespace xla
