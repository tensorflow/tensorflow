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
#include "tensorflow/contrib/boosted_trees/lib/utils/macros.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/tensor_utils.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

Status BatchFeatures::Initialize(
    std::vector<Tensor> dense_float_features_list,
    std::vector<Tensor> sparse_float_feature_indices_list,
    std::vector<Tensor> sparse_float_feature_values_list,
    std::vector<Tensor> sparse_float_feature_shapes_list,
    std::vector<Tensor> sparse_int_feature_indices_list,
    std::vector<Tensor> sparse_int_feature_values_list,
    std::vector<Tensor> sparse_int_feature_shapes_list) {
  // Validate number of feature columns.
  auto num_dense_float_features = dense_float_features_list.size();
  auto num_sparse_float_features = sparse_float_feature_indices_list.size();
  auto num_sparse_int_features = sparse_int_feature_indices_list.size();
  QCHECK(num_dense_float_features + num_sparse_float_features +
             num_sparse_int_features >
         0)
      << "Must have at least one feature column.";

  // Read dense float features.
  dense_float_feature_columns_.reserve(num_dense_float_features);
  for (uint32 dense_feat_idx = 0; dense_feat_idx < num_dense_float_features;
       ++dense_feat_idx) {
    auto dense_float_feature = dense_float_features_list[dense_feat_idx];
    TF_CHECK_AND_RETURN_IF_ERROR(
        TensorShapeUtils::IsMatrix(dense_float_feature.shape()),
        errors::InvalidArgument("Dense float feature must be a matrix."));
    TF_CHECK_AND_RETURN_IF_ERROR(
        dense_float_feature.dim_size(0) == batch_size_,
        errors::InvalidArgument(
            "Dense float vector must have batch_size rows: ", batch_size_,
            " vs. ", dense_float_feature.dim_size(0)));
    TF_CHECK_AND_RETURN_IF_ERROR(
        dense_float_feature.dim_size(1) == 1,
        errors::InvalidArgument(
            "Dense float features may not be multivalent: dim_size(1) = ",
            dense_float_feature.dim_size(1)));
    dense_float_feature_columns_.emplace_back(dense_float_feature);
  }

  // Read sparse float features.
  sparse_float_feature_columns_.reserve(num_sparse_float_features);
  TF_CHECK_AND_RETURN_IF_ERROR(
      sparse_float_feature_values_list.size() == num_sparse_float_features &&
          sparse_float_feature_shapes_list.size() == num_sparse_float_features,
      errors::InvalidArgument("Inconsistent number of sparse float features."));
  for (uint32 sparse_feat_idx = 0; sparse_feat_idx < num_sparse_float_features;
       ++sparse_feat_idx) {
    auto sparse_float_feature_indices =
        sparse_float_feature_indices_list[sparse_feat_idx];
    auto sparse_float_feature_values =
        sparse_float_feature_values_list[sparse_feat_idx];
    auto sparse_float_feature_shape =
        sparse_float_feature_shapes_list[sparse_feat_idx];
    TF_CHECK_AND_RETURN_IF_ERROR(
        TensorShapeUtils::IsMatrix(sparse_float_feature_indices.shape()),
        errors::InvalidArgument(
            "Sparse float feature indices must be a matrix."));
    TF_CHECK_AND_RETURN_IF_ERROR(
        TensorShapeUtils::IsVector(sparse_float_feature_values.shape()),
        errors::InvalidArgument(
            "Sparse float feature values must be a vector."));
    TF_CHECK_AND_RETURN_IF_ERROR(
        TensorShapeUtils::IsVector(sparse_float_feature_shape.shape()),
        errors::InvalidArgument(
            "Sparse float feature shape must be a vector."));
    auto shape_flat = sparse_float_feature_shape.flat<int64>();
    TF_CHECK_AND_RETURN_IF_ERROR(
        shape_flat.size() == 2,
        errors::InvalidArgument(
            "Sparse float feature column must be two-dimensional."));
    TF_CHECK_AND_RETURN_IF_ERROR(
        shape_flat(0) == batch_size_,
        errors::InvalidArgument(
            "Sparse float feature shape incompatible with batch size."));
    auto tensor_shape = TensorShape({shape_flat(0), shape_flat(1)});
    auto order_dims = sparse::SparseTensor::VarDimArray({0, 1});
    sparse_float_feature_columns_.emplace_back(sparse_float_feature_indices,
                                               sparse_float_feature_values,
                                               tensor_shape, order_dims);
  }

  // Read sparse int features.
  sparse_int_feature_columns_.reserve(num_sparse_int_features);
  TF_CHECK_AND_RETURN_IF_ERROR(
      sparse_int_feature_values_list.size() == num_sparse_int_features &&
          sparse_int_feature_shapes_list.size() == num_sparse_int_features,
      errors::InvalidArgument("Inconsistent number of sparse int features."));
  for (uint32 sparse_feat_idx = 0; sparse_feat_idx < num_sparse_int_features;
       ++sparse_feat_idx) {
    auto sparse_int_feature_indices =
        sparse_int_feature_indices_list[sparse_feat_idx];
    auto sparse_int_feature_values =
        sparse_int_feature_values_list[sparse_feat_idx];
    auto sparse_int_feature_shape =
        sparse_int_feature_shapes_list[sparse_feat_idx];
    TF_CHECK_AND_RETURN_IF_ERROR(
        TensorShapeUtils::IsMatrix(sparse_int_feature_indices.shape()),
        errors::InvalidArgument(
            "Sparse int feature indices must be a matrix."));
    TF_CHECK_AND_RETURN_IF_ERROR(
        TensorShapeUtils::IsVector(sparse_int_feature_values.shape()),
        errors::InvalidArgument("Sparse int feature values must be a vector."));
    TF_CHECK_AND_RETURN_IF_ERROR(
        TensorShapeUtils::IsVector(sparse_int_feature_shape.shape()),
        errors::InvalidArgument("Sparse int feature shape must be a vector."));
    auto shape_flat = sparse_int_feature_shape.flat<int64>();
    TF_CHECK_AND_RETURN_IF_ERROR(
        shape_flat.size() == 2,
        errors::InvalidArgument(
            "Sparse int feature column must be two-dimensional."));
    TF_CHECK_AND_RETURN_IF_ERROR(
        shape_flat(0) == batch_size_,
        errors::InvalidArgument(
            "Sparse int feature shape incompatible with batch size."));
    auto tensor_shape = TensorShape({shape_flat(0), shape_flat(1)});
    auto order_dims = sparse::SparseTensor::VarDimArray({0, 1});
    sparse_int_feature_columns_.emplace_back(sparse_int_feature_indices,
                                             sparse_int_feature_values,
                                             tensor_shape, order_dims);
  }
  return Status::OK();
}

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
