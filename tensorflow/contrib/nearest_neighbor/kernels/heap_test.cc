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

#include "tensorflow/contrib/nearest_neighbor/kernels/heap.h"

#include "tensorflow/core/kernels/ops_testutil.h"

namespace tensorflow {
namespace nearest_neighbor {
namespace {

TEST(HeapTest, SimpleHeapTest1) {
  SimpleHeap<float, int> h;
  h.Resize(10);
  h.InsertUnsorted(2.0, 2);
  h.InsertUnsorted(1.0, 1);
  h.InsertUnsorted(5.0, 5);
  h.InsertUnsorted(3.0, 3);
  h.Heapify();

  float k;
  int d;
  h.ExtractMin(&k, &d);
  ASSERT_EQ(1.0, k);
  ASSERT_EQ(1, d);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);

  h.Insert(4.0, 4);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(3.0, k);
  ASSERT_EQ(3, d);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(4.0, k);
  ASSERT_EQ(4, d);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(5.0, k);
  ASSERT_EQ(5, d);

  h.Reset();
  h.InsertUnsorted(2.0, 2);
  h.InsertUnsorted(10.0, 10);
  h.InsertUnsorted(8.0, 8);
  h.Heapify();
  h.ExtractMin(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(8.0, k);
  ASSERT_EQ(8, d);

  h.Insert(9.5, 9);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(9.5, k);
  ASSERT_EQ(9, d);

  h.ExtractMin(&k, &d);
  ASSERT_EQ(10.0, k);
  ASSERT_EQ(10, d);
}

TEST(HeapTest, SimpleHeapTest2) {
  // Same as above, but without initial resize
  SimpleHeap<float, int> h;
  h.InsertUnsorted(2.0, 2);
  h.InsertUnsorted(1.0, 1);
  h.InsertUnsorted(5.0, 5);
  h.InsertUnsorted(3.0, 3);
  h.Heapify();

  float k;
  int d;
  h.ExtractMin(&k, &d);
  ASSERT_EQ(1.0, k);
  ASSERT_EQ(1, d);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);

  h.Insert(4.0, 4);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(3.0, k);
  ASSERT_EQ(3, d);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(4.0, k);
  ASSERT_EQ(4, d);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(5.0, k);
  ASSERT_EQ(5, d);

  h.Reset();
  h.InsertUnsorted(2.0, 2);
  h.InsertUnsorted(10.0, 10);
  h.InsertUnsorted(8.0, 8);
  h.Heapify();
  h.ExtractMin(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(8.0, k);
  ASSERT_EQ(8, d);

  h.Insert(9.5, 9);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(9.5, k);
  ASSERT_EQ(9, d);

  h.ExtractMin(&k, &d);
  ASSERT_EQ(10.0, k);
  ASSERT_EQ(10, d);
}

TEST(HeapTest, SimpleHeapTest3) {
  SimpleHeap<float, int> h;
  h.InsertUnsorted(2.0, 2);
  h.InsertUnsorted(1.0, 1);
  h.InsertUnsorted(5.0, 5);
  h.InsertUnsorted(3.0, 3);
  h.Heapify();

  EXPECT_EQ(1.0, h.MinKey());

  h.ReplaceTop(0.5, 0);
  float k;
  int d;
  h.ExtractMin(&k, &d);
  EXPECT_EQ(0.5, k);
  EXPECT_EQ(0, d);

  h.ExtractMin(&k, &d);
  EXPECT_EQ(2.0, k);
  EXPECT_EQ(2, d);
}

TEST(HeapTest, AugmentedHeapTest1) {
  AugmentedHeap<float, int> h;
  h.InsertUnsorted(2.0, 2);
  h.InsertUnsorted(1.0, 1);
  h.InsertUnsorted(5.0, 5);
  h.InsertUnsorted(3.0, 3);
  h.Heapify();

  float k;
  int d;
  h.ExtractMin(&k, &d);
  ASSERT_EQ(1.0, k);
  ASSERT_EQ(1, d);

  h.InsertGuaranteedTop(1.0, 10);
  h.ExtractMin(&k, &d);
  ASSERT_EQ(1.0, k);
  ASSERT_EQ(10, d);

  h.ExtractMin(&k, &d);
  ASSERT_EQ(2.0, k);
  ASSERT_EQ(2, d);

  h.Insert(4.0, 4);

  h.ExtractMin(&k, &d);
  ASSERT_EQ(3.0, k);
  ASSERT_EQ(3, d);

  h.ExtractMin(&k, &d);
  ASSERT_EQ(4.0, k);
  ASSERT_EQ(4, d);

  h.ExtractMin(&k, &d);
  ASSERT_EQ(5.0, k);
  ASSERT_EQ(5, d);
}

}  // namespace
}  // namespace nearest_neighbor
}  // namespace tensorflow
