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

#include "tensorflow/compiler/xla/array3d.h"

#include <initializer_list>

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

TEST(Array3dTest, UninitializedDimsCtor) {
  Array3D<int> uninit(2, 3, 4);
  EXPECT_EQ(uninit.n1(), 2);
  EXPECT_EQ(uninit.n2(), 3);
  EXPECT_EQ(uninit.n3(), 4);
  EXPECT_EQ(uninit.num_elements(), 24);
}

TEST(Array3dTest, FillCtor) {
  Array3D<int> fullof7(2, 3, 4, 7);

  EXPECT_EQ(fullof7.n1(), 2);
  EXPECT_EQ(fullof7.n2(), 3);
  EXPECT_EQ(fullof7.n3(), 4);

  for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) {
    for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) {
      for (int64 n3 = 0; n3 < fullof7.n3(); ++n3) {
        EXPECT_EQ(fullof7(n1, n2, n3), 7);
      }
    }
  }
}

TEST(Array3dTest, InitializerListCtor) {
  Array3D<int> arr = {{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
                      {{9, 10}, {11, 12}, {13, 14}, {15, 16}},
                      {{17, 18}, {19, 20}, {21, 22}, {23, 24}}};

  EXPECT_EQ(arr.n1(), 3);
  EXPECT_EQ(arr.n2(), 4);
  EXPECT_EQ(arr.n3(), 2);
  EXPECT_EQ(arr.num_elements(), 24);

  EXPECT_EQ(arr(0, 0, 0), 1);
  EXPECT_EQ(arr(0, 0, 1), 2);
  EXPECT_EQ(arr(0, 1, 0), 3);
  EXPECT_EQ(arr(0, 3, 1), 8);
  EXPECT_EQ(arr(1, 0, 0), 9);
  EXPECT_EQ(arr(1, 1, 1), 12);
  EXPECT_EQ(arr(2, 0, 0), 17);
  EXPECT_EQ(arr(2, 1, 1), 20);
  EXPECT_EQ(arr(2, 2, 0), 21);
  EXPECT_EQ(arr(2, 3, 1), 24);
}

TEST(Array3dTest, Fill) {
  Array3D<int> fullof7(2, 3, 4, 7);
  for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) {
    for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) {
      for (int64 n3 = 0; n3 < fullof7.n3(); ++n3) {
        EXPECT_EQ(fullof7(n1, n2, n3), 7);
      }
    }
  }

  fullof7.Fill(11);
  for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) {
    for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) {
      for (int64 n3 = 0; n3 < fullof7.n3(); ++n3) {
        EXPECT_EQ(fullof7(n1, n2, n3), 11);
      }
    }
  }
}

}  // namespace
}  // namespace xla
