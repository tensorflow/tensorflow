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

#include "tensorflow/core/grappler/costs/robust_stats.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class RobustStatsTest : public ::testing::Test {
 public:
  void SetUp() override {
    for (double d = 1.0; d <= 5.0; d += 1.0) {
      values1_.push_back(5.0 - d);
      values1_.push_back(5.0 + d);
      values2_.push_back(25.0 - 2 * d);
      values2_.push_back(25.0 + 2 * d);
      values3_.push_back(-3.0 - d);
      values3_.push_back(-3.0 + d);
    }
    values1_.push_back(5.0);  // Odd # elements, mean is 5.0
    values3_.push_back(197.0);
    values3_.push_back(-203.0);  // Even # elements, mean is -3.0
  }

  std::vector<double> values1_;
  std::vector<double> values2_;
  std::vector<double> values3_;
};

TEST_F(RobustStatsTest, Simple) {
  RobustStats s1(values1_);
  EXPECT_EQ(5.0, s1.mean());
  EXPECT_EQ(0.0, s1.lo());
  EXPECT_EQ(10.0, s1.hi());

  RobustStats s2(values2_);
  EXPECT_EQ(25.0, s2.mean());
  EXPECT_EQ(15.0, s2.lo());
  EXPECT_EQ(35.0, s2.hi());

  RobustStats s3(values3_);
  EXPECT_EQ(-3.0, s3.mean());
  EXPECT_EQ(-203.0, s3.lo());
  EXPECT_EQ(197.0, s3.hi());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
