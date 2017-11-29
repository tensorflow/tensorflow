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

#include "tensorflow/contrib/boosted_trees/lib/utils/tensor_utils.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/macros.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

std::vector<Tensor> TensorUtils::OpInputListToTensorVec(
    const OpInputList& input_list) {
  std::vector<Tensor> tensor_vec;
  tensor_vec.reserve(input_list.size());
  for (const Tensor& tensor : input_list) {
    tensor_vec.emplace_back(tensor);
  }
  return tensor_vec;
}

Status TensorUtils::ReadDenseFloatFeatures(OpKernelContext* const context,
                                           OpInputList* features_list) {
  // Constants.
  constexpr auto kDenseFloatFeaturesName = "dense_float_features";

  // Read dense float features list;
  TF_RETURN_IF_ERROR(
      context->input_list(kDenseFloatFeaturesName, features_list));
  return Status::OK();
}

Status TensorUtils::ReadSparseFloatFeatures(OpKernelContext* const context,
                                            OpInputList* features_indices_list,
                                            OpInputList* feature_values_list,
                                            OpInputList* feature_shapes_list) {
  // Constants.
  constexpr auto kSparseFloatFeatureIndicesName =
      "sparse_float_feature_indices";
  constexpr auto kSparseFloatFeatureValuesName = "sparse_float_feature_values";
  constexpr auto kSparseFloatFeatureShapesName = "sparse_float_feature_shapes";

  // Read sparse float features list;
  TF_RETURN_IF_ERROR(context->input_list(kSparseFloatFeatureIndicesName,
                                         features_indices_list));
  TF_RETURN_IF_ERROR(
      context->input_list(kSparseFloatFeatureValuesName, feature_values_list));
  TF_RETURN_IF_ERROR(
      context->input_list(kSparseFloatFeatureShapesName, feature_shapes_list));
  return Status::OK();
}

Status TensorUtils::ReadSparseIntFeatures(OpKernelContext* const context,
                                          OpInputList* features_indices_list,
                                          OpInputList* feature_values_list,
                                          OpInputList* feature_shapes_list) {
  // Constants.
  constexpr auto kSparseIntFeatureIndicesName = "sparse_int_feature_indices";
  constexpr auto kSparseIntFeatureValuesName = "sparse_int_feature_values";
  constexpr auto kSparseIntFeatureShapesName = "sparse_int_feature_shapes";

  // Read sparse int features list;
  TF_RETURN_IF_ERROR(
      context->input_list(kSparseIntFeatureIndicesName, features_indices_list));
  TF_RETURN_IF_ERROR(
      context->input_list(kSparseIntFeatureValuesName, feature_values_list));
  TF_RETURN_IF_ERROR(
      context->input_list(kSparseIntFeatureShapesName, feature_shapes_list));
  return Status::OK();
}

int64 TensorUtils::InferBatchSize(
    const OpInputList& dense_float_features_list,
    const OpInputList& sparse_float_feature_shapes_list,
    const OpInputList& sparse_int_feature_shapes_list) {
  if (dense_float_features_list.size() > 0) {
    return dense_float_features_list[0].dim_size(0);
  }
  if (sparse_float_feature_shapes_list.size() > 0) {
    return sparse_float_feature_shapes_list[0].flat<int64>()(0);
  }
  if (sparse_int_feature_shapes_list.size() > 0) {
    return sparse_int_feature_shapes_list[0].flat<int64>()(0);
  }
  QCHECK(false) << "Could not infer batch size due to empty feature set.";
}

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
