/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor_matcher.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace test {
namespace {

using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::FloatEq;

TEST(TensorMatcherTest, BasicPod) {
  std::vector<Tensor> expected;
  int16_t in1 = 100;
  expected.push_back(Tensor(in1));
  int16_t in2 = 16;
  expected.push_back(Tensor(in2));

  EXPECT_THAT(expected,
              ElementsAre(TensorEq(Tensor(in1)), TensorEq(Tensor(in2))));
}

TEST(TensorMatcherTest, Basic4Bit) {
  Tensor in1(DT_INT4, TensorShape({1}));
  in1.flat<int4>()(0) = static_cast<int4>(7);

  Tensor in2(DT_UINT4, TensorShape({1}));
  in2.flat<uint4>()(0) = static_cast<uint4>(15);

  std::vector<Tensor> expected;
  expected.push_back(Tensor(in1));
  expected.push_back(Tensor(in2));

  EXPECT_THAT(expected,
              ElementsAre(TensorEq(Tensor(in1)), TensorEq(Tensor(in2))));
}

TEST(TensorMatcherTest, BasicString) {
  std::vector<Tensor> expected;
  std::string s1 = "random 1";
  expected.push_back(Tensor(s1));
  std::string s2 = "random 2";
  expected.push_back(Tensor(s2));

  EXPECT_THAT(expected,
              ElementsAre(TensorEq(Tensor(s1)), TensorEq(Tensor(s2))));
}

TEST(TensorMatcherTest, FloatComparisonUsesTolerance) {
  // Two floats that are *nearly* equal.
  float f1(1);
  float f2 = std::nextafter(f1, f1 + 1);

  // Direct equality checks should fail, but use of the specialized `FloatEq`
  // should succeed since this matcher applies ULP-based comparison.
  // go/matchers#FpMatchers
  ASSERT_NE(f1, f2);
  ASSERT_THAT(f1, FloatEq(f2));

  EXPECT_THAT(Tensor(f1), TensorEq(Tensor(f2)));
}

TEST(TensorMatcherTest, DoubleComparisonUsesTolerance) {
  // Two doubles that are *nearly* equal.
  double d1(1);
  double d2 = std::nextafter(d1, d1 + 1);

  // Direct equality checks should fail, but use of the specialized `DoubleEq`
  // should succeed since this matcher applies ULP-based comparison.
  // go/matchers#FpMatchers
  ASSERT_NE(d1, d2);
  ASSERT_THAT(d1, DoubleEq(d2));

  EXPECT_THAT(Tensor(d1), TensorEq(Tensor(d2)));
}

}  // namespace
}  // namespace test
}  // namespace tensorflow
