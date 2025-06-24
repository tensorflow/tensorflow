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
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/functional/overload.h"
#include "absl/log/log.h"

namespace xla::gpu {
namespace {

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

constexpr float kEpsilon = 0.01;

enum class InterpolatorType {
  NN = 0,
  Complement = 1,
};

template <typename T, size_t N>
using Interpolator =
    std::variant<std::unique_ptr<EuclideanNNInterpolator<T, N>>,
                 std::unique_ptr<EuclideanComplementInterpolator<T, N>>>;

class InterpolatorFake : public InterpolatorBase<int, 2> {
 public:
  ~InterpolatorFake() override = default;

  // Fake eval function which just returns the size of the consumed set.
  int Eval(std::array<int64_t, 2>& x) const override { return plane_.size(); }

  void Add(std::array<int64_t, 2>& x, int val) override { plane_.push_back(x); }

 private:
  std::vector<std::array<int64_t, 2>> plane_;
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
    : public TestWithParam<std::tuple<
          InterpolatorType, EuclideanNNInterpolatorTestCase<int, 2>>> {
  void SetUp() override {
    std::array<int64_t, 2> p1 = {8, 16};
    std::array<int64_t, 2> p2 = {8, 8};
    std::array<int64_t, 2> p3 = {16, 8};
    std::array<int64_t, 2> p4 = {16, 16};
    plane_.push_back({p1, 1});
    plane_.push_back({p2, 2});
    plane_.push_back({p3, 3});
    plane_.push_back({p4, 4});
  }

 protected:
  std::vector<std::pair<std::array<int64_t, 2>, int>> plane_;

  Interpolator<int64_t, 2> DispatchInterpolator(InterpolatorType type) {
    if (type == InterpolatorType::NN) {
      auto interpolator =
          std::make_unique<EuclideanNNInterpolator<int64_t, 2>>();
      return std::move(interpolator);
    }
    if (type == InterpolatorType::Complement) {
      auto interpolator =
          std::make_unique<EuclideanComplementInterpolator<int64_t, 2>>(
              /*next_context=*/std::array<int64_t, 2>{8, 8},
              /*next_power_context=*/std::array<int64_t, 2>{-1, -1},
              /*max_context=*/std::array<int64_t, 2>{16, 16},
              /*min_context=*/std::array<int64_t, 2>{8, 8});
      return std::move(interpolator);
    }
    LOG(FATAL) << "Unreachable.";
  }
};

TEST_P(EuclideanNN2DInterpolatorTest, ReturnsNearestNeighbour) {
  InterpolatorType interpolator_type = std::get<0>(GetParam());
  auto param = std::get<1>(GetParam());

  Interpolator<int64_t, 2> interpolator =
      DispatchInterpolator(interpolator_type);
  for (const auto& point : plane_) {
    std::array<int64_t, 2> plane_point = point.first;
    int val = point.second;
    std::visit(
        absl::Overload(
            [&](const std::unique_ptr<EuclideanNNInterpolator<int64_t, 2>>&
                    nn) { return nn->Add(plane_point, val); },
            [&](const std::unique_ptr<
                EuclideanComplementInterpolator<int64_t, 2>>& comp) {
              return comp->Add(plane_point, val);
            }),
        interpolator);
  }
  std::visit(
      absl::Overload(
          [&](const std::unique_ptr<EuclideanNNInterpolator<int64_t, 2>>& nn) {
            EXPECT_EQ(nn->Eval(param.eval_point), param.expected_value);
          },
          [&](const std::unique_ptr<
              EuclideanComplementInterpolator<int64_t, 2>>& comp) {
            EXPECT_EQ(comp->Eval(param.eval_point), param.expected_value);
          }),
      interpolator);
}

// We have 4 points on a 2D plane.
// X = {(8,8), (8,16), (16,8), (16,16)}
INSTANTIATE_TEST_SUITE_P(
    EuclideanNNInterpolator2DIntegerTest, EuclideanNN2DInterpolatorTest,
    Combine(ValuesIn({InterpolatorType::NN, InterpolatorType::Complement}),
            ValuesIn({
                EuclideanNNInterpolatorTestCase<int, 2>{
                    /*test_name=*/"near_first_point",
                    /*eval_point=*/{7, 9},
                    /*expected_value=*/2,
                },
                EuclideanNNInterpolatorTestCase<int, 2>{
                    /*test_name=*/"near_second_point",
                    /*eval_point=*/{15, 17},
                    /*expected_value=*/4,
                },
                EuclideanNNInterpolatorTestCase<int, 2>{
                    /*test_name=*/"nearer_only_by_one",
                    /*eval_point=*/{13, 8},
                    /*expected_value=*/3,
                },
                EuclideanNNInterpolatorTestCase<int, 2>{
                    /*test_name=*/"extrapolate_first_point",
                    /*eval_point=*/{7, 7},
                    /*expected_value=*/2,
                },
                EuclideanNNInterpolatorTestCase<int, 2>{
                    /*test_name=*/"extrapolate_second_point",
                    /*eval_point=*/{17, 9},
                    /*expected_value=*/3,
                },
            })),
    [](const auto& info) {
      return absl::StrCat(std::get<1>(info.param).test_name, "x",
                          std::get<0>(info.param));
    });

TEST(EuclideanWeightedAverage2DInterpolatorTest, ReturnsWeightedAverage) {
  auto interpolator = std::make_unique<EuclideanWeightedAverageInterpolator<2>>(
      /*next_context=*/std::array<int64_t, 2>{-1, -1},
      /*next_power_context=*/std::array<int64_t, 2>{1, 1},
      /*max_context=*/std::array<int64_t, 2>{16, 16},
      /*min_context=*/std::array<int64_t, 2>{8, 8});
  std::array<int64_t, 2> p1 = {8, 16};
  std::array<int64_t, 2> p2 = {8, 8};
  std::array<int64_t, 2> p3 = {16, 8};
  std::array<int64_t, 2> p4 = {16, 16};

  std::vector<std::pair<std::array<int64_t, 2>, int>> plane;
  plane.push_back({p1, 1});
  plane.push_back({p2, 2});
  plane.push_back({p3, 3});
  plane.push_back({p4, 4});

  for (const auto& point : plane) {
    std::array<int64_t, 2> plane_point = point.first;
    int val = point.second;
    interpolator->Add(plane_point, val);
  }
  // Near the first point.
  std::array<int64_t, 2> p = {7, 9};
  EXPECT_NEAR(interpolator->Eval(p), 1.94, kEpsilon);
  // Near the second point.
  p = {15, 17};
  EXPECT_NEAR(interpolator->Eval(p), 3.83, kEpsilon);
  // Nearer only for first dim.
  p = {13, 8};
  EXPECT_NEAR(interpolator->Eval(p), 2.72, kEpsilon);
  // Extrapolate first point.
  p = {7, 7};
  EXPECT_NEAR(interpolator->Eval(p), 2.0, kEpsilon);
  // Extrapolate second point.
  p = {17, 9};
  EXPECT_NEAR(interpolator->Eval(p), 3.05, kEpsilon);
  // Exact point.
  p = {8, 16};
  EXPECT_NEAR(interpolator->Eval(p), 1.00, kEpsilon);
}

}  // namespace
}  // namespace xla::gpu
