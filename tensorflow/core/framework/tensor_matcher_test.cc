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

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace test {
namespace {

using ::testing::ElementsAre;

TEST(TensorMatcherTest, BasicPod) {
  std::vector<Tensor> expected;
  int16_t in1 = 100;
  expected.push_back(Tensor(in1));
  int16_t in2 = 16;
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

}  // namespace
}  // namespace test
}  // namespace tensorflow
