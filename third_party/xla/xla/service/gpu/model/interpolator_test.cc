/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/model/interpolator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace xla::gpu {
namespace {

using ::testing::TestWithParam;
using ::testing::Values;

class InterpolatorFake : public InterpolatorBase<int, 2> {
 public:
  // Fake eval function which just returns the size of the consumed set.
  int Eval(std::array<int64_t, 2>& x) override { return plane_.size(); }
};

TEST(Interpolator, PersistsEuclideanPoints) {
  InterpolatorFake interpolator;
  std::array<int64_t, 2> dummy = {0, 1};

  ASSERT_EQ(interpolator.Eval(dummy), 0);
  interpolator.Add(dummy, 0);
  interpolator.Add(dummy, 0);
  interpolator.Add(dummy, 0);

  EXPECT_EQ(interpolator.Eval(dummy), 3);
}

template <typename R, size_t N>
struct EuclideanNNInterpolatorTestCase {
  std::string test_name;
  std::array<int64_t, N> eval_point;
  R expected_value;
};

class EuclideanNN2DInterpolatorTest
    : public TestWithParam<EuclideanNNInterpolatorTestCase<int, 2>> {
  void SetUp() override {
    std::array<int64_t, 2> p1 = {3, 4};
    std::array<int64_t, 2> p2 = {5, 7};
    interpolator_.Add(p1, /*val=*/1);
    interpolator_.Add(p2, /*val=*/2);
  }

 protected:
  EuclideanNNInterpolator<int64_t, 2> interpolator_;
  std::vector<std::pair<std::array<int64_t, 2>, int>> plane_;
};

TEST_P(EuclideanNN2DInterpolatorTest, ReturnsNearestNeighbour) {
  auto param = GetParam();
  for (auto& [plane_point, val] : plane_) {
    interpolator_.Add(plane_point, val);
  }
  EXPECT_EQ(interpolator_.Eval(param.eval_point), param.expected_value);
}

// We have 2 points on a 2D plane.
// X = {(3,4), (5,7)}
INSTANTIATE_TEST_SUITE_P(EuclideanNNInterpolator2DIntegerTest,
                         EuclideanNN2DInterpolatorTest,
                         Values(EuclideanNNInterpolatorTestCase<int, 2>{
                                    /*test_name=*/"near_first_point",
                                    /*eval_point=*/{4, 3},
                                    /*expected_value=*/1,
                                },
                                EuclideanNNInterpolatorTestCase<int, 2>{
                                    /*test_name=*/"near_second_point",
                                    /*eval_point=*/{7, 5},
                                    /*expected_value=*/2,
                                },
                                EuclideanNNInterpolatorTestCase<int, 2>{
                                    /*test_name=*/"nearer_only_by_one",
                                    /*eval_point=*/{4, 6},
                                    /*expected_value=*/2,
                                },
                                EuclideanNNInterpolatorTestCase<int, 2>{
                                    /*test_name=*/"extrapolate_first_point",
                                    /*eval_point=*/{2, 3},
                                    /*expected_value=*/1,
                                },
                                EuclideanNNInterpolatorTestCase<int, 2>{
                                    /*test_name=*/"extrapolate_second_point",
                                    /*eval_point=*/{6, 8},
                                    /*expected_value=*/2,
                                }),
                         [](const auto& info) { return info.param.test_name; });

}  // namespace
}  // namespace xla::gpu
