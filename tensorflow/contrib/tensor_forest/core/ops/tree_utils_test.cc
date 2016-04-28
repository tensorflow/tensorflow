// Copyright 2016 Google Inc. All Rights Reserved.
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
#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"
#include "testing/base/public/gunit.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace tensorforest {

TEST(TestSum, Basic) {
  Tensor two_dim = test::AsTensor<float>({1, 3, 5, 7, 2, 4, 6, 8}, {2, 4});
  EXPECT_EQ(Sum<float>(two_dim), 36.0);

  Tensor multi_dim_single_value = test::AsTensor<int>({42}, {1, 1, 1, 1});
  EXPECT_EQ(Sum<int>(multi_dim_single_value), 42);

  Tensor nothing = test::AsTensor<float>({}, {0});
  EXPECT_EQ(Sum<float>(nothing), 0.0);
}

TEST(TestWeightedGiniImpurity, Basic) {
  Tensor vals = test::AsTensor<float>({2, 2, 2}, {3});
  EXPECT_EQ(WeightedGiniImpurity(vals.unaligned_flat<float>()), 6);

  Tensor zero = test::AsTensor<float>({0}, {1});
  EXPECT_EQ(WeightedGiniImpurity(zero.unaligned_flat<float>()), 0);
}

TEST(TestWeightedVariance, Basic) {
  // Lets say values were: (2, 2), (3, 2), (4, 2)
  Tensor sums = test::AsTensor<float>({9, 6}, {2});
  Tensor squares = test::AsTensor<float>({29, 12}, {2});

  EXPECT_FLOAT_EQ(WeightedVariance(sums.unaligned_flat<float>(),
                                   squares.unaligned_flat<float>(), 3), 2.0);

  Tensor zero = test::AsTensor<float>({0}, {1});
  EXPECT_FLOAT_EQ(WeightedVariance(zero.unaligned_flat<float>(),
                                   zero.unaligned_flat<float>(), 1), 0);
}

TEST(TestInitialize, Basic) {
  Tensor t = test::AsTensor<float>({0, 0, 0, 0, 0, 0, 0, 0}, {4, 2});

  Initialize<float>(t, 42.0);

  const auto vals = t.tensor<float, 2>();
  EXPECT_FLOAT_EQ(vals(0, 0), 42);
  EXPECT_FLOAT_EQ(vals(1, 1), 42);
  EXPECT_FLOAT_EQ(vals(3, 0), 42);

  Tensor nothing = test::AsTensor<float>({}, {0});

  // Just make sure this runs.
  Initialize<float>(nothing, -1.0);
}

TEST(DecideNode, Basic) {
  // Even though we only want a 1-D point, make sure to test a 2-D tensor
  // because that's usually what comes off a Slice().
  Tensor point = test::AsTensor<float>({10, 10, 10, 10}, {1, 4});

  EXPECT_TRUE(DecideNode(point, 0, 9));
  EXPECT_TRUE(DecideNode(point, 1, 0));
  EXPECT_TRUE(DecideNode(point, 2, -3));
  EXPECT_FALSE(DecideNode(point, 3, 11));
}

TEST(IsAllInitialized, Basic) {
  // Even though we only want a 1-D feature set, make sure to test a 2-D tensor
  // because that's usually what comes off a Slice().
  Tensor features = test::AsTensor<int32>({10, 2, -1}, {1, 3});

  EXPECT_FALSE(IsAllInitialized(features));

  features = test::AsTensor<int32>({10, 2, 0}, {1, 3});

  EXPECT_TRUE(IsAllInitialized(features));
}

TEST(BestFeatureClassification, Basic) {
  const int32 num_accumulators = 4;
  const int32 num_splits = 3;
  const int32 num_classes = 4;
  Tensor totals = test::AsTensor<float>({1, 5, 6, 7,
                                         0, 0, 0, 0,
                                         30, 10, 10, 10,      // this one
                                         -1, -1, -1, -1},
                                        {num_accumulators, num_classes});
  Tensor splits = test::AsTensor<float>(
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       30, 10, 10, 10, 10, 0, 0, 10, 19, 5, 6, 8,  // this one
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {num_accumulators, num_splits, num_classes});

  EXPECT_EQ(BestFeatureClassification(totals, splits, 2), 1);
}

TEST(BestFeatureClassification, NoWinner) {
  const int32 num_accumulators = 4;
  const int32 num_splits = 3;
  const int32 num_classes = 4;
  // When counts are all the same, the most reasonable thing to do is pick 0.
  Tensor totals = test::AsTensor<float>({1, 5, 6, 7,
                                         0, 0, 0, 0,
                                         18, 6, 6, 6,      // this one
                                         -1, -1, -1, -1},
                                        {num_accumulators, num_classes});
  Tensor splits = test::AsTensor<float>(
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       9, 3, 3, 3, 9, 3, 3, 3, 9, 3, 3, 3,     // this one
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {num_accumulators, num_splits, num_classes});

  EXPECT_EQ(BestFeatureClassification(totals, splits, 2), 0);
}

TEST(BestFeatureRegression, Basic) {
  const int32 num_accumulators = 4;
  const int32 num_splits = 3;
  const int32 num_classes = 4;
  Tensor total_sums = test::AsTensor<float>(
      {1, 5, 6, 7,
       0, 0, 0, 0,
       10, 8, 6, 9,      // this one
       -1, -1, -1, -1},
      {num_accumulators, num_classes});
  Tensor total_squares = test::AsTensor<float>(
      {1, 5, 6, 7,
       0, 0, 0, 0,
       100, 50, 40, 45,      // this one
       -1, -1, -1, -1},
      {num_accumulators, num_classes});

  Tensor split_sums = test::AsTensor<float>(
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       10, 8, 6, 9, 9, 8, 5, 9, 0, 0, 0, 0,      // this one
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {num_accumulators, num_splits, num_classes});

  // lower the variance by lowering one of the squares just a little.
  Tensor split_squares = test::AsTensor<float>(
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       100, 50, 40, 45, 100, 50, 40, 43, 0, 0, 0, 0,    // this one
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {num_accumulators, num_splits, num_classes});

  EXPECT_EQ(BestFeatureRegression(total_sums, total_squares, split_sums,
                                  split_squares, 2), 1);
}

TEST(BestFeatureRegression, NoWinner) {
  const int32 num_accumulators = 4;
  const int32 num_splits = 3;
  const int32 num_classes = 4;
  // when counts are all the same, the most reasonable thing to do is pick 0.
  Tensor total_sums = test::AsTensor<float>(
      {1, 5, 6, 7,
       0, 0, 0, 0,
       10, 8, 6, 9,      // this one
       -1, -1, -1, -1},
      {num_accumulators, num_classes});
  Tensor total_squares = test::AsTensor<float>(
      {1, 5, 6, 7,
       0, 0, 0, 0,
       100, 50, 40, 45,      // this one
       -1, -1, -1, -1},
      {num_accumulators, num_classes});

  Tensor split_sums = test::AsTensor<float>(
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       10, 8, 6, 9, 10, 8, 6, 9, 10, 8, 6, 9,      // this one
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {num_accumulators, num_splits, num_classes});

  Tensor split_squares = test::AsTensor<float>(
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       100, 50, 40, 45, 100, 50, 40, 45, 100, 50, 40, 45,    // this one
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {num_accumulators, num_splits, num_classes});

  EXPECT_EQ(BestFeatureRegression(total_sums, total_squares, split_sums,
                                  split_squares, 2), 0);
}

}  // namespace tensorforest
}  // namespace tensorflow

