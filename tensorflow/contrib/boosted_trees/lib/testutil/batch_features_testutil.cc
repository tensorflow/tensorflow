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
#include "tensorflow/contrib/boosted_trees/lib/testutil/batch_features_testutil.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace boosted_trees {
namespace testutil {

using tensorflow::Tensor;

void RandomlyInitializeBatchFeatures(
    tensorflow::random::SimplePhilox* rng, uint32 num_dense_float_features,
    uint32 num_sparse_float_features, double sparsity_lo, double sparsity_hi,
    boosted_trees::utils::BatchFeatures* batch_features) {
  const int64 batch_size = static_cast<int64>(batch_features->batch_size());

  // Populate dense features.
  std::vector<tensorflow::Tensor> dense_float_features_list;
  for (int i = 0; i < num_dense_float_features; ++i) {
    std::vector<float> values;
    for (int64 j = 0; j < batch_size; ++j) {
      values.push_back(rng->RandFloat());
    }
    auto dense_tensor = Tensor(tensorflow::DT_FLOAT, {batch_size, 1});
    tensorflow::test::FillValues<float>(&dense_tensor, values);
    dense_float_features_list.push_back(dense_tensor);
  }

  // Populate sparse features.
  std::vector<tensorflow::Tensor> sparse_float_feature_indices_list;
  std::vector<tensorflow::Tensor> sparse_float_feature_values_list;
  std::vector<tensorflow::Tensor> sparse_float_feature_shapes_list;
  for (int i = 0; i < num_sparse_float_features; ++i) {
    std::set<uint64> indices;
    const double sparsity =
        sparsity_lo + rng->RandDouble() * (sparsity_hi - sparsity_lo);
    const double density = 1 - sparsity;
    for (int64 k = 0; k < static_cast<int64>(density * batch_size) + 1; ++k) {
      indices.insert(rng->Uniform64(batch_size));
    }
    const int64 sparse_values_size = indices.size();
    std::vector<int64> indices_vector;
    for (auto idx : indices) {
      indices_vector.push_back(idx);
      indices_vector.push_back(0);
    }
    auto indices_tensor = Tensor(tensorflow::DT_INT64, {sparse_values_size, 2});
    tensorflow::test::FillValues<int64>(&indices_tensor, indices_vector);
    sparse_float_feature_indices_list.push_back(indices_tensor);

    std::vector<float> values;
    for (int64 j = 0; j < sparse_values_size; ++j) {
      values.push_back(rng->RandFloat());
    }
    auto values_tensor = Tensor(tensorflow::DT_FLOAT, {sparse_values_size});
    tensorflow::test::FillValues<float>(&values_tensor, values);
    sparse_float_feature_values_list.push_back(values_tensor);

    auto shape_tensor = Tensor(tensorflow::DT_INT64, {2});
    tensorflow::test::FillValues<int64>(&shape_tensor, {batch_size, 1});
    sparse_float_feature_shapes_list.push_back(shape_tensor);
  }

  // TODO(salehay): Add categorical feature generation support.
  TF_EXPECT_OK(batch_features->Initialize(
      dense_float_features_list, sparse_float_feature_indices_list,
      sparse_float_feature_values_list, sparse_float_feature_shapes_list, {},
      {}, {}));
}

}  // namespace testutil
}  // namespace boosted_trees
}  // namespace tensorflow
