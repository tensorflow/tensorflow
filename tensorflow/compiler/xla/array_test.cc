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

TEST(ArrayTest, IndexingReadWrite) {
  Array<int> arr({2, 3});

  EXPECT_EQ(arr(1, 1), 0);
  EXPECT_EQ(arr(1, 2), 0);
  arr(1, 1) = 51;
  arr(1, 2) = 61;
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
  arr.Each([&](tensorflow::gtl::ArraySlice<int64> idx, int cell) {
    int64 lin_idx = idx[0] * 12 + idx[1] * 4 + idx[2];
    EXPECT_EQ(lin_idx, cell);
    each_count++;
    each_sum += cell;
  });
  EXPECT_EQ(arr.num_elements(), each_count);
  EXPECT_EQ(arr.num_elements() * (arr.num_elements() - 1) / 2, each_sum);
}

}  // namespace
}  // namespace xla
