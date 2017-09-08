// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/contrib/boosted_trees/lib/utils/batch_features.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {
namespace {

using test::AsTensor;
using errors::InvalidArgument;

class BatchFeaturesTest : public ::testing::Test {};

TEST_F(BatchFeaturesTest, InvalidNumFeatures) {
  BatchFeatures batch_features(8);
  EXPECT_DEATH(({ batch_features.Initialize({}, {}, {}, {}, {}, {}, {}); })
                   .IgnoreError(),
               "Must have at least one feature column.");
}

TEST_F(BatchFeaturesTest, DenseFloatFeatures_WrongShape) {
  BatchFeatures batch_features(8);
  auto dense_vec = AsTensor<float>({3.0f, 7.0f});
  auto expected_error =
      InvalidArgument("Dense float feature must be a matrix.");
  EXPECT_EQ(expected_error,
            batch_features.Initialize({dense_vec}, {}, {}, {}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, DenseFloatFeatures_WrongBatchDimension) {
  BatchFeatures batch_features(8);
  auto dense_vec = AsTensor<float>({3.0f, 7.0f}, {2, 1});
  auto expected_error =
      InvalidArgument("Dense float vector must have batch_size rows: 8 vs. 2");
  EXPECT_EQ(expected_error,
            batch_features.Initialize({dense_vec}, {}, {}, {}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, DenseFloatFeatures_Multivalent) {
  BatchFeatures batch_features(1);
  auto dense_vec = AsTensor<float>({3.0f, 7.0f}, {1, 2});
  auto expected_error = InvalidArgument(
      "Dense float features may not be multi-valent: dim_size(1) = 2");
  EXPECT_EQ(expected_error,
            batch_features.Initialize({dense_vec}, {}, {}, {}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, SparseFloatFeatures_WrongShapeIndices) {
  BatchFeatures batch_features(2);
  auto sparse_float_feature_indices = AsTensor<int64>({0, 0, 1, 0});
  auto sparse_float_feature_values = AsTensor<float>({3.0f, 7.0f});
  auto sparse_float_feature_shape = AsTensor<int64>({2, 1});
  auto expected_error =
      InvalidArgument("Sparse float feature indices must be a matrix.");
  EXPECT_EQ(expected_error, batch_features.Initialize(
                                {}, {sparse_float_feature_indices},
                                {sparse_float_feature_values},
                                {sparse_float_feature_shape}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, SparseFloatFeatures_WrongShapeValues) {
  BatchFeatures batch_features(2);
  auto sparse_float_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_float_feature_values = AsTensor<float>({3.0f, 7.0f}, {1, 2});
  auto sparse_float_feature_shape = AsTensor<int64>({2, 1});
  auto expected_error =
      InvalidArgument("Sparse float feature values must be a vector.");
  EXPECT_EQ(expected_error, batch_features.Initialize(
                                {}, {sparse_float_feature_indices},
                                {sparse_float_feature_values},
                                {sparse_float_feature_shape}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, SparseFloatFeatures_WrongShapeShape) {
  BatchFeatures batch_features(2);
  auto sparse_float_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_float_feature_values = AsTensor<float>({3.0f, 7.0f});
  auto sparse_float_feature_shape = AsTensor<int64>({2, 1}, {1, 2});
  auto expected_error =
      InvalidArgument("Sparse float feature shape must be a vector.");
  EXPECT_EQ(expected_error, batch_features.Initialize(
                                {}, {sparse_float_feature_indices},
                                {sparse_float_feature_values},
                                {sparse_float_feature_shape}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, SparseFloatFeatures_WrongSizeShape) {
  BatchFeatures batch_features(2);
  auto sparse_float_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_float_feature_values = AsTensor<float>({3.0f, 7.0f});
  auto sparse_float_feature_shape = AsTensor<int64>({2, 1, 9});
  auto expected_error =
      InvalidArgument("Sparse float feature column must be two-dimensional.");
  EXPECT_EQ(expected_error, batch_features.Initialize(
                                {}, {sparse_float_feature_indices},
                                {sparse_float_feature_values},
                                {sparse_float_feature_shape}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, SparseFloatFeatures_IncompatibleShape) {
  BatchFeatures batch_features(2);
  auto sparse_float_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_float_feature_values = AsTensor<float>({3.0f, 7.0f});
  auto sparse_float_feature_shape = AsTensor<int64>({8, 1});
  auto expected_error = InvalidArgument(
      "Sparse float feature shape incompatible with batch size.");
  EXPECT_EQ(expected_error, batch_features.Initialize(
                                {}, {sparse_float_feature_indices},
                                {sparse_float_feature_values},
                                {sparse_float_feature_shape}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, SparseFloatFeatures_Multivalent) {
  BatchFeatures batch_features(2);
  auto sparse_float_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_float_feature_values = AsTensor<float>({3.0f, 7.0f});
  auto sparse_float_feature_shape = AsTensor<int64>({2, 2});
  auto expected_error =
      InvalidArgument("Sparse float features may not be multi-valent.");
  EXPECT_EQ(expected_error, batch_features.Initialize(
                                {}, {sparse_float_feature_indices},
                                {sparse_float_feature_values},
                                {sparse_float_feature_shape}, {}, {}, {}));
}

TEST_F(BatchFeaturesTest, SparseIntFeatures_WrongShapeIndices) {
  BatchFeatures batch_features(2);
  auto sparse_int_feature_indices = AsTensor<int64>({0, 0, 1, 0});
  auto sparse_int_feature_values = AsTensor<int64>({3, 7});
  auto sparse_int_feature_shape = AsTensor<int64>({2, 1});
  auto expected_error =
      InvalidArgument("Sparse int feature indices must be a matrix.");
  EXPECT_EQ(expected_error,
            batch_features.Initialize(
                {}, {}, {}, {}, {sparse_int_feature_indices},
                {sparse_int_feature_values}, {sparse_int_feature_shape}));
}

TEST_F(BatchFeaturesTest, SparseIntFeatures_WrongShapeValues) {
  BatchFeatures batch_features(2);
  auto sparse_int_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_int_feature_values = AsTensor<int64>({3, 7}, {1, 2});
  auto sparse_int_feature_shape = AsTensor<int64>({2, 1});
  auto expected_error =
      InvalidArgument("Sparse int feature values must be a vector.");
  EXPECT_EQ(expected_error,
            batch_features.Initialize(
                {}, {}, {}, {}, {sparse_int_feature_indices},
                {sparse_int_feature_values}, {sparse_int_feature_shape}));
}

TEST_F(BatchFeaturesTest, SparseIntFeatures_WrongShapeShape) {
  BatchFeatures batch_features(2);
  auto sparse_int_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_int_feature_values = AsTensor<int64>({3, 7});
  auto sparse_int_feature_shape = AsTensor<int64>({2, 1}, {1, 2});
  auto expected_error =
      InvalidArgument("Sparse int feature shape must be a vector.");
  EXPECT_EQ(expected_error,
            batch_features.Initialize(
                {}, {}, {}, {}, {sparse_int_feature_indices},
                {sparse_int_feature_values}, {sparse_int_feature_shape}));
}

TEST_F(BatchFeaturesTest, SparseIntFeatures_WrongSizeShape) {
  BatchFeatures batch_features(2);
  auto sparse_int_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_int_feature_values = AsTensor<int64>({3, 7});
  auto sparse_int_feature_shape = AsTensor<int64>({2, 1, 9});
  auto expected_error =
      InvalidArgument("Sparse int feature column must be two-dimensional.");
  EXPECT_EQ(expected_error,
            batch_features.Initialize(
                {}, {}, {}, {}, {sparse_int_feature_indices},
                {sparse_int_feature_values}, {sparse_int_feature_shape}));
}

TEST_F(BatchFeaturesTest, SparseIntFeatures_IncompatibleShape) {
  BatchFeatures batch_features(2);
  auto sparse_int_feature_indices = AsTensor<int64>({0, 0, 1, 0}, {2, 2});
  auto sparse_int_feature_values = AsTensor<int64>({3, 7});
  auto sparse_int_feature_shape = AsTensor<int64>({8, 1});
  auto expected_error =
      InvalidArgument("Sparse int feature shape incompatible with batch size.");
  EXPECT_EQ(expected_error,
            batch_features.Initialize(
                {}, {}, {}, {}, {sparse_int_feature_indices},
                {sparse_int_feature_values}, {sparse_int_feature_shape}));
}

}  // namespace
}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
