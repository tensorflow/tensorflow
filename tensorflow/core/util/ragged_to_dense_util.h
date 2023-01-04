/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_H_
#define TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/util/ragged_to_dense_util_common.h"

namespace tensorflow {

string RowPartitionTypeToString(RowPartitionType row_partition_type);

Status GetRowPartitionTypesHelper(
    const std::vector<string>& row_partition_type_strings,
    std::vector<RowPartitionType>* row_partition_types);

// ContextType must be InferenceContext or OpKernelConstruction.
template <typename ContextType>
Status GetRowPartitionTypes(
    ContextType* context, std::vector<RowPartitionType>* row_partition_types) {
  std::vector<string> row_partition_type_strings;
  TF_RETURN_IF_ERROR(
      context->GetAttr("row_partition_types", &row_partition_type_strings));
  return GetRowPartitionTypesHelper(row_partition_type_strings,
                                    row_partition_types);
}

Status GetRowPartitionTypesHelper(
    const std::vector<string>& row_partition_type_strings,
    std::vector<RowPartitionType>* row_partition_types);

Status CombineRaggedTensorToTensorShapes(int ragged_rank,
                                         const TensorShapeProto& shape,
                                         const TensorShapeProto& value_shape,
                                         TensorShapeProto* output_shape);

int GetRaggedRank(const std::vector<RowPartitionType>& row_partition_types);

Status ValidateDefaultValueShape(const TensorShapeProto& default_value_shape,
                                 const TensorShapeProto& value_shape);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_H_
