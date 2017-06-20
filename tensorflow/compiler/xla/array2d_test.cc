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

#include "tensorflow/compiler/xla/array2d.h"

#include <initializer_list>

#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace {

TEST(Array2dTest, DefaultCtor) {
  Array2D<int> empty;
  EXPECT_EQ(empty.n1(), 0);
  EXPECT_EQ(empty.n2(), 0);
  EXPECT_EQ(empty.num_elements(), 0);
}

TEST(Array2dTest, UninitializedDimsCtor) {
  Array2D<int> uninit(2, 3);
  EXPECT_EQ(uninit.n1(), 2);
  EXPECT_EQ(uninit.n2(), 3);
  EXPECT_EQ(uninit.num_elements(), 6);
}

TEST(Array2dTest, FillCtor) {
  Array2D<int> fullof7(2, 3, 7);

  EXPECT_EQ(fullof7.n1(), 2);
  EXPECT_EQ(fullof7.n2(), 3);

  for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) {
    for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) {
      EXPECT_EQ(fullof7(n1, n2), 7);
    }
  }
}

TEST(Array2dTest, InitializerListCtor) {
  Array2D<int> arr = {{1, 2, 3}, {4, 5, 6}};

  EXPECT_EQ(arr.n1(), 2);
  EXPECT_EQ(arr.n2(), 3);

  EXPECT_EQ(arr(0, 0), 1);
  EXPECT_EQ(arr(0, 1), 2);
  EXPECT_EQ(arr(0, 2), 3);
  EXPECT_EQ(arr(1, 0), 4);
  EXPECT_EQ(arr(1, 1), 5);
  EXPECT_EQ(arr(1, 2), 6);
}

TEST(Array2dTest, Accessors) {
  Array2D<int> arr = {{1, 2, 3}, {4, 5, 6}};

  EXPECT_EQ(arr.n1(), 2);
  EXPECT_EQ(arr.n2(), 3);
  EXPECT_EQ(arr.height(), 2);
  EXPECT_EQ(arr.width(), 3);
  EXPECT_EQ(arr.num_elements(), 6);
}

TEST(Array2dTest, IndexingReadWrite) {
  Array2D<int> arr = {{1, 2, 3}, {4, 5, 6}};

  EXPECT_EQ(arr(1, 1), 5);
  EXPECT_EQ(arr(1, 2), 6);
  arr(1, 1) = 51;
  arr(1, 2) = 61;
  EXPECT_EQ(arr(1, 1), 51);
  EXPECT_EQ(arr(1, 2), 61);
}

TEST(Array2dTest, IndexingReadWriteBool) {
  Array2D<bool> arr = {{false, true, false}, {true, true, false}};

  EXPECT_EQ(arr(1, 1), true);
  EXPECT_EQ(arr(1, 2), false);
  arr(1, 1) = false;
  arr(1, 2) = true;
  EXPECT_EQ(arr(1, 1), false);
  EXPECT_EQ(arr(1, 2), true);
}

TEST(Array2dTest, Fill) {
  Array2D<int> fullof7(2, 3, 7);
  for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) {
    for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) {
      EXPECT_EQ(fullof7(n1, n2), 7);
    }
  }

  fullof7.Fill(11);
  for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) {
    for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) {
      EXPECT_EQ(fullof7(n1, n2), 11);
    }
  }
}

TEST(Array2dTest, DataPointer) {
  Array2D<int> arr = {{1, 2, 3}, {4, 5, 6}};

  EXPECT_EQ(arr.data()[0], 1);
}

TEST(Array2dTest, Linspace) {
  auto arr = MakeLinspaceArray2D(1.0, 3.5, 3, 2);

  EXPECT_EQ(arr->n1(), 3);
  EXPECT_EQ(arr->n2(), 2);

  EXPECT_FLOAT_EQ((*arr)(0, 0), 1.0);
  EXPECT_FLOAT_EQ((*arr)(0, 1), 1.5);
  EXPECT_FLOAT_EQ((*arr)(1, 0), 2.0);
  EXPECT_FLOAT_EQ((*arr)(1, 1), 2.5);
  EXPECT_FLOAT_EQ((*arr)(2, 0), 3.0);
  EXPECT_FLOAT_EQ((*arr)(2, 1), 3.5);
}

TEST(Array2dTest, Stringification) {
  auto arr = MakeLinspaceArray2D(1.0, 3.5, 3, 2);
  const string expected = R"([[1, 1.5],
 [2, 2.5],
 [3, 3.5]])";
  EXPECT_EQ(expected, arr->ToString());
}

}  // namespace
}  // namespace xla
