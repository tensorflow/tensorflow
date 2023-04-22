// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/core/kernels/boosted_trees/tree_helper.h"

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

using std::vector;

namespace tensorflow {
namespace {

const double kDelta = 1e-5;

TEST(TreeHelper, MulticlassFullHessianTest) {
  const int kNumClasses = 4;
  Eigen::VectorXf g(kNumClasses);
  g << 0.5, 0.33, -9, 1;
  Eigen::VectorXf h(kNumClasses * kNumClasses);
  h << 3, 5, 7, 8, 5, 4, 1, 5, 7, 1, 8, 4, 8, 5, 4, 9;
  float l1 = 0;
  float l2 = 0.3;
  Eigen::VectorXf weight(kNumClasses);
  float gain;
  CalculateWeightsAndGains(g, h, l1, l2, &weight, &gain);
  std::vector<float> expected_weight = {0.9607576, 0.4162569, 0.9863192,
                                        -1.5820024};
  for (int i = 0; i < kNumClasses; ++i) {
    EXPECT_NEAR(expected_weight[i], weight[i], kDelta);
  }
  EXPECT_NEAR(9.841132, gain, kDelta);
}

TEST(TreeHelper, MulticlassDiagonalHessianTest) {
  const int kNumClasses = 4;
  Eigen::VectorXf g(kNumClasses);
  g << 0.5, 0.33, -9, 1;
  float l1 = 0.1;
  // Normal case.
  {
    float l2 = 0.3;
    // Full Hessian.
    Eigen::VectorXf h_full(kNumClasses * kNumClasses);
    h_full << 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9;
    Eigen::VectorXf weight_full(kNumClasses);
    float gain_full;
    CalculateWeightsAndGains(g, h_full, l1, l2, &weight_full, &gain_full);
    // Diagonal Hessian.
    Eigen::VectorXf h_diagonal(kNumClasses);
    h_diagonal << 3, 4, 8, 9;
    Eigen::VectorXf weight_diagonal(kNumClasses);
    float gain_diagonal;
    CalculateWeightsAndGains(g, h_diagonal, l1, l2, &weight_diagonal,
                             &gain_diagonal);

    for (int i = 0; i < kNumClasses; ++i) {
      EXPECT_NEAR(weight_full[i], weight_diagonal[i], kDelta);
    }
    EXPECT_EQ(gain_full, gain_diagonal);
  }
  // Zero entries in diagonal, no regularization; use matrix solver, just like
  // full Hessian.
  {
    float l2 = 0.0;
    // Full Hessian.
    Eigen::VectorXf h_full(kNumClasses * kNumClasses);
    h_full << 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 9;
    Eigen::VectorXf weight_full(kNumClasses);
    float gain_full;
    CalculateWeightsAndGains(g, h_full, l1, l2, &weight_full, &gain_full);
    // Diagonal Hessian.
    Eigen::VectorXf h_diagonal(kNumClasses);
    h_diagonal << 3, 0, 8, 9;
    Eigen::VectorXf weight_diagonal(kNumClasses);
    float gain_diagonal;
    CalculateWeightsAndGains(g, h_diagonal, l1, l2, &weight_diagonal,
                             &gain_diagonal);

    for (int i = 0; i < kNumClasses; ++i) {
      EXPECT_NEAR(weight_full[i], weight_diagonal[i], kDelta);
    }
    EXPECT_EQ(gain_full, gain_diagonal);
  }
}

TEST(TreeHelper, DiagonalHessianVsSingleClass) {
  float l1 = 0;
  float l2 = 0.3;
  // Solve as multi-class using 2 logits. For cross entropy loss, gradient and
  // hessian are only non-zero when label probability > 0. For this example the
  // one-hot label would be [0, 1].
  Eigen::VectorXf g_diagonal(2);
  g_diagonal << 0, -0.25;
  // Diagonal Hessian.
  Eigen::VectorXf h_diagonal(2);
  h_diagonal << 0, 0.11;
  Eigen::VectorXf weight_diagonal(2);
  float gain_diagonal;
  CalculateWeightsAndGains(g_diagonal, h_diagonal, l1, l2, &weight_diagonal,
                           &gain_diagonal);
  // Single logit.
  Eigen::VectorXf g_single(1);
  g_single << -0.25;
  Eigen::VectorXf h_single(1);
  h_single << 0.11;
  Eigen::VectorXf weight_single(1);
  float gain_single;
  CalculateWeightsAndGains(g_single, h_single, l1, l2, &weight_single,
                           &gain_single);

  EXPECT_NEAR(weight_diagonal[1], weight_single[0], kDelta);
  EXPECT_EQ(gain_diagonal, gain_single);
}

}  // namespace
}  // namespace tensorflow
