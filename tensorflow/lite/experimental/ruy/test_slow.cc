/* Copyright 2019 Google LLC. All Rights Reserved.

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

// This test contains more expensive test cases.

#include "tensorflow/lite/experimental/ruy/test.h"

namespace ruy {

using LhsScalar = RUY_TEST_LHSSCALAR;
using RhsScalar = RUY_TEST_RHSSCALAR;
using AccumScalar = RUY_TEST_ACCUMSCALAR;
using DstScalar = RUY_TEST_DSTSCALAR;

using TestSetType =
    TestSet<LhsScalar, RhsScalar, BasicSpec<AccumScalar, DstScalar>>;

TEST(RuyTest, TestBigNarrowMuls) {
  for (int width : {1, 2, 3, 4, 5, 8}) {
    TestRCC<TestSetType>(width, 401, 601);
    TestRCC<TestSetType>(587, 443, width);
  }
  TestRCC<TestSetType>(7, 45984,
                       5);  // Large enough to trigger row-sum overflows.
  TestRCC<TestSetType>(512, 256, 16);
}

TEST(RuyTest, TestBigShallowMuls) {
  TestLinearAllOrders<TestSetType>(501, 1, 321);
  TestLinearAllOrders<TestSetType>(301, 5, 403);
  TestLinearAllOrders<TestSetType>(256, 32, 512);
}

TEST(RuyTest, TestBigMuls) {
  TestRCC<TestSetType>(225, 303, 199);
  TestLinearAllOrders<TestSetType>(256, 192, 128);
}

TEST(RuyTest, TestBigPowerOfTwoDepthWithAvoidAliasing) {
  // Important to test some power-of-two depths: that's when the
  // RUY_AVOID_ALIASING optimization kicks in and makes packed matrices
  // strided, exposing bugs in kernels mixing up size and stride.
  // Moreover, it's important that the test matrices be sufficiently wide
  // that they will result in multiple blocks, exposing bugs in the
  // computation of the base address of each block.
  TestLinearAllOrders<TestSetType>(70, 1024, 80);
  TestLinearAllOrders<TestSetType>(60, 2048, 70);
  TestLinearAllOrders<TestSetType>(40, 4096, 50);
}

TEST(RuyTest, TestGEMV) {
  for (int size = 1025; size <= 1409; size += 384) {
    for (int depth = 350; depth < 500; depth += 47) {
      TestLinearAllOrders<TestSetType>(size, depth, 1);
    }
  }
}

}  // namespace ruy
